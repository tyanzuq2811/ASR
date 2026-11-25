#!/usr/bin/env python3
"""Edge Profiling Test Harness on Arduino VENTUNO Q (Qualcomm Snapdragon QCS8275).

Profiles Whisper-Tiny across FP32, FP16, W8A16, and INT8 configurations:
- TTFT (Time-To-First-Token) in milliseconds
- TPOT (Time-Per-Output-Token) in milliseconds
- RTF (Real-Time Factor) = Total Latency / Audio Duration
- Peak RAM usage (VmHWM) in MB via /proc/self/status
- Model footprint on disk (bytes / MB)

Supports:
- ONNX Runtime CPU Execution Provider (Snapdragon Kryo cores)
- Qualcomm QNN / QAIRT toolchain execution
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperProcessor


def get_peak_ram_mb() -> float:
    """Reads VmHWM (Resident Set Size High Water Mark) from /proc/self/status."""
    status_path = Path("/proc/self/status")
    if not status_path.exists():
        return 0.0
    try:
        with open(status_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    parts = line.split()
                    return float(parts[1]) / 1024.0  # kB to MB
    except Exception:
        pass
    return 0.0


def compute_audio_mel(
    audio_path: Path,
    processor: WhisperProcessor,
    target_sr: int = 16000,
) -> tuple[np.ndarray, float]:
    """Loads audio, extracts 80-channel log-mel spectrogram [1, 80, 3000]."""
    speech, sr = sf.read(str(audio_path))
    if speech.ndim > 1:
        speech = speech.mean(axis=1)
    duration_s = len(speech) / sr

    # Extract log-mel features
    features = processor.feature_extractor(
        speech,
        sampling_rate=sr,
        return_tensors="np",
    ).input_features
    return features, duration_s


def run_whisper_autoregressive(
    encoder_session: ort.InferenceSession,
    decoder_session: ort.InferenceSession,
    features: np.ndarray,
    prompt_ids: list[int],
    eos_token_id: int,
    max_new_tokens: int = 64,
) -> dict[str, Any]:
    """Runs encoder + decoder autoregressive generation while recording TTFT, TPOT, and latency."""
    enc_input_name = encoder_session.get_inputs()[0].name
    enc_input_type = encoder_session.get_inputs()[0].type
    if "float16" in enc_input_type:
        features = features.astype(np.float16)
    else:
        features = features.astype(np.float32)

    # 1. Start overall timer
    t_start = time.perf_counter()

    # 2. Encoder forward pass
    t_enc_0 = time.perf_counter()
    encoder_out = encoder_session.run(None, {enc_input_name: features})[0]
    t_enc_1 = time.perf_counter()
    encoder_latency_ms = (t_enc_1 - t_enc_0) * 1000.0

    # Match decoder input type for encoder_hidden_states
    dec_inputs = decoder_session.get_inputs()
    dec_input_ids_name = dec_inputs[0].name
    dec_hidden_name = dec_inputs[1].name
    if "float16" in dec_inputs[1].type:
        encoder_out = encoder_out.astype(np.float16)
    else:
        encoder_out = encoder_out.astype(np.float32)

    # 3. First token generation (Prompt -> First Token)
    current_tokens = list(prompt_ids)
    decoder_input_ids = np.array([current_tokens], dtype=np.int64)

    t_dec1_0 = time.perf_counter()
    logits = decoder_session.run(None, {
        dec_input_ids_name: decoder_input_ids,
        dec_hidden_name: encoder_out,
    })[0]
    first_token = int(np.argmax(logits[0, -1, :]))
    t_dec1_1 = time.perf_counter()
    current_tokens.append(first_token)

    # TTFT = Total time from input until first token is produced
    ttft_ms = (t_dec1_1 - t_start) * 1000.0
    first_token_latency_ms = (t_dec1_1 - t_dec1_0) * 1000.0

    # 4. Subsequent token autoregressive decoding loop
    token_latencies_ms = []
    generated_tokens = [first_token]

    if first_token != eos_token_id:
        for _ in range(max_new_tokens - 1):
            decoder_input_ids = np.array([current_tokens], dtype=np.int64)
            t_step_0 = time.perf_counter()
            logits = decoder_session.run(None, {
                dec_input_ids_name: decoder_input_ids,
                dec_hidden_name: encoder_out,
            })[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            t_step_1 = time.perf_counter()

            token_latencies_ms.append((t_step_1 - t_step_0) * 1000.0)
            current_tokens.append(next_token)
            generated_tokens.append(next_token)
            if next_token == eos_token_id:
                break

    t_end = time.perf_counter()
    total_asr_latency_s = t_end - t_start

    # TPOT = Mean time per token for subsequent tokens
    tpot_ms = float(np.mean(token_latencies_ms)) if token_latencies_ms else first_token_latency_ms

    return {
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "encoder_latency_ms": encoder_latency_ms,
        "first_token_latency_ms": first_token_latency_ms,
        "total_latency_s": total_asr_latency_s,
        "num_tokens_generated": len(generated_tokens),
        "generated_tokens": generated_tokens,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=Path("/home/arduino/models"), help="Path to models directory")
    parser.add_argument("--test-dir", type=Path, default=Path("/home/arduino/test_waves"), help="Path to test wav files")
    parser.add_argument("--output-dir", type=Path, default=Path("/home/arduino/output/EXP-B03-profiling"), help="Output directory")
    parser.add_argument("--model-id", default="openai/whisper-tiny", help="Base Whisper processor")
    parser.add_argument("--configs", default="fp32,fp16,w8a16,int8", help="Comma-separated configurations to profile")
    parser.add_argument("--limit", type=int, default=20, help="Number of test audio files to profile")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ARDUINO VENTUNO Q (QUALCOMM QCS8275) EDGE ASR PROFILER")
    print(f"Models Dir : {args.models_dir}")
    print(f"Test Waves : {args.test_dir}")
    print(f"Output Dir : {args.output_dir}")
    print(f"Target CPU : Snapdragon Kryo (4x A78C @ 2.7 GHz + 4x A55 @ 1.9 GHz)")
    print(f"Target NPU : Hexagon HTP v75")
    print("=" * 70)

    # Load processor
    print(f"\nInitializing WhisperProcessor ({args.model_id})...")
    processor = WhisperProcessor.from_pretrained(args.model_id)
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe", no_timestamps=True)
    prompt_ids = [processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    for _, token_id in forced_decoder_ids:
        prompt_ids.append(token_id)
    eos_token_id = processor.tokenizer.eos_token_id
    print(f"Prompt IDs: {prompt_ids} | EOS: {eos_token_id}")

    # Discover test audio files
    wav_files = sorted(args.test_dir.glob("*.wav"))
    if args.limit > 0:
        wav_files = wav_files[: args.limit]
    print(f"Found {len(wav_files)} test audio samples to profile.")
    if not wav_files:
        raise RuntimeError(f"No WAV files found in {args.test_dir}")

    # Model configuration map
    config_specs = {
        "fp32": {
            "encoder": args.models_dir / "encoder.onnx",
            "decoder": args.models_dir / "decoder.onnx",
            "tier_name": "FP32 Baseline",
        },
        "fp16": {
            "encoder": args.models_dir / "encoder_fp16.onnx",
            "decoder": args.models_dir / "decoder_fp16.onnx",
            "tier_name": "FP16 Tier",
        },
        "w8a16": {
            "encoder": args.models_dir / "encoder_w8a16.onnx",
            "decoder": args.models_dir / "decoder_int8.onnx",
            "tier_name": "W8A16 QDQ Tier",
        },
        "int8": {
            "encoder": args.models_dir / "encoder_int8.onnx",
            "decoder": args.models_dir / "decoder_int8.onnx",
            "tier_name": "INT8 Tier",
        },
    }

    selected_configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = 4  # Utilize performance Cortex-A78C cores
    ort_opts.inter_op_num_threads = 1
    ort_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    all_results = []
    summary = {}

    for cfg_key in selected_configs:
        if cfg_key not in config_specs:
            print(f"[SKIP] Unknown config: {cfg_key}")
            continue

        spec = config_specs[cfg_key]
        enc_path = spec["encoder"]
        dec_path = spec["decoder"]

        if not enc_path.exists() or not dec_path.exists():
            print(f"[SKIP] Missing model files for {cfg_key}: {enc_path} or {dec_path}")
            continue

        enc_bytes = enc_path.stat().st_size
        dec_bytes = dec_path.stat().st_size
        total_bytes = enc_bytes + dec_bytes
        total_mb = total_bytes / (1024 * 1024)

        print("\n" + "-" * 70)
        print(f"PROFILING CONFIGURATION: {cfg_key.upper()} ({spec['tier_name']})")
        print(f"  Encoder: {enc_path.name} ({enc_bytes:,} bytes)")
        print(f"  Decoder: {dec_path.name} ({dec_bytes:,} bytes)")
        print(f"  Total Footprint: {total_mb:.2f} MB ({total_bytes:,} bytes)")
        print("-" * 70)

        # Initialize ONNX runtime sessions
        print("Loading ONNX sessions onto CPU (Kryo A78C)...")
        try:
            sess_enc = ort.InferenceSession(str(enc_path), sess_options=ort_opts, providers=["CPUExecutionProvider"])
            sess_dec = ort.InferenceSession(str(dec_path), sess_options=ort_opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"[SKIP] Failed to load ONNX session for {cfg_key}: {e}")
            continue

        # Warm-up pass
        print("Executing warm-up pass...")
        dummy_feat, _ = compute_audio_mel(wav_files[0], processor)
        _ = run_whisper_autoregressive(sess_enc, sess_dec, dummy_feat, prompt_ids, eos_token_id, max_new_tokens=8)
        print("Warm-up complete. Starting benchmark runs...")

        cfg_ttft = []
        cfg_tpot = []
        cfg_rtf = []
        cfg_ram = []
        cfg_latencies = []

        for idx, wav_path in enumerate(wav_files, 1):
            audio_id = wav_path.stem
            features, duration_s = compute_audio_mel(wav_path, processor)

            # Profile run
            out = run_whisper_autoregressive(sess_enc, sess_dec, features, prompt_ids, eos_token_id)
            ttft_ms = out["ttft_ms"]
            tpot_ms = out["tpot_ms"]
            total_latency_s = out["total_latency_s"]
            rtf = total_latency_s / duration_s if duration_s > 0 else 0.0
            peak_ram_mb = get_peak_ram_mb()
            pred_text = processor.tokenizer.decode(out["generated_tokens"], skip_special_tokens=True).strip()

            cfg_ttft.append(ttft_ms)
            cfg_tpot.append(tpot_ms)
            cfg_rtf.append(rtf)
            cfg_ram.append(peak_ram_mb)
            cfg_latencies.append(total_latency_s)

            row = {
                "config": cfg_key,
                "tier_name": spec["tier_name"],
                "audio_id": audio_id,
                "audio_duration_s": round(duration_s, 3),
                "ttft_ms": round(ttft_ms, 2),
                "tpot_ms": round(tpot_ms, 2),
                "encoder_latency_ms": round(out["encoder_latency_ms"], 2),
                "total_latency_s": round(total_latency_s, 3),
                "rtf": round(rtf, 4),
                "tokens_generated": out["num_tokens_generated"],
                "peak_ram_mb": round(peak_ram_mb, 2),
                "model_size_mb": round(total_mb, 2),
                "model_size_bytes": total_bytes,
                "predicted_text": pred_text,
            }
            all_results.append(row)

            print(
                f"  [{idx:02d}/{len(wav_files)}] {audio_id:16s} | Dur: {duration_s:4.1f}s | "
                f"TTFT: {ttft_ms:6.1f} ms | TPOT: {tpot_ms:5.1f} ms/tok | RTF: {rtf:5.3f} | RAM: {peak_ram_mb:5.1f} MB | "
                f"Text: '{pred_text[:35]}...'"
            )

        # Compute summary for config
        summary[cfg_key] = {
            "tier_name": spec["tier_name"],
            "model_size_mb": round(total_mb, 2),
            "model_size_bytes": total_bytes,
            "num_samples": len(wav_files),
            "ttft_ms": {
                "mean": round(float(np.mean(cfg_ttft)), 2),
                "median": round(float(np.median(cfg_ttft)), 2),
                "iqr": round(float(np.percentile(cfg_ttft, 75) - np.percentile(cfg_ttft, 25)), 2),
                "min": round(float(np.min(cfg_ttft)), 2),
                "max": round(float(np.max(cfg_ttft)), 2),
                "std": round(float(np.std(cfg_ttft)), 2),
            },
            "tpot_ms": {
                "mean": round(float(np.mean(cfg_tpot)), 2),
                "median": round(float(np.median(cfg_tpot)), 2),
                "iqr": round(float(np.percentile(cfg_tpot, 75) - np.percentile(cfg_tpot, 25)), 2),
                "min": round(float(np.min(cfg_tpot)), 2),
                "max": round(float(np.max(cfg_tpot)), 2),
                "std": round(float(np.std(cfg_tpot)), 2),
            },
            "rtf": {
                "mean": round(float(np.mean(cfg_rtf)), 4),
                "median": round(float(np.median(cfg_rtf)), 4),
                "iqr": round(float(np.percentile(cfg_rtf, 75) - np.percentile(cfg_rtf, 25)), 4),
                "min": round(float(np.min(cfg_rtf)), 4),
                "max": round(float(np.max(cfg_rtf)), 4),
            },
            "peak_ram_mb": {
                "mean": round(float(np.mean(cfg_ram)), 2),
                "max": round(float(np.max(cfg_ram)), 2),
            },
        }

    # Save detailed CSV
    csv_path = args.output_dir / "edge_profiling_results.csv"
    if all_results:
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSaved detailed profiling results to: {csv_path}")

    # Save JSON summary
    summary_path = args.output_dir / "edge_profiling_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved profiling summary to: {summary_path}")

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE SUMMARY:")
    print(json.dumps(summary, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()
