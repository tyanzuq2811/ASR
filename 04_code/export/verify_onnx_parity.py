#!/usr/bin/env python3
"""Verify numerical and decoding parity between PyTorch and ONNX Whisper models.

Tests:
1. Encoder numerical parity (max absolute difference, cosine similarity).
2. Decoder numerical parity on forced Vietnamese prompt prefix.
3. End-to-end greedy token and text parity across test utterances.

Produces reproducible JSON summary and CSV results.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten().astype(np.float64)
    b_flat = b.flatten().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def onnx_greedy_decode(
    ort_encoder: ort.InferenceSession,
    ort_decoder: ort.InferenceSession,
    features: np.ndarray,
    prompt_ids: list[int],
    eos_token_id: int,
    max_new_tokens: int = 100,
) -> list[int]:
    ort_hidden = ort_encoder.run(["encoder_hidden_states"], {"input_features": features})[0]

    current_tokens = list(prompt_ids)
    for _ in range(max_new_tokens):
        decoder_input = np.array([current_tokens], dtype=np.int64)
        logits = ort_decoder.run(["logits"], {
            "decoder_input_ids": decoder_input,
            "encoder_hidden_states": ort_hidden,
        })[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        current_tokens.append(next_token)
        if next_token == eos_token_id:
            break

    return current_tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Path to PyTorch checkpoint or HuggingFace ID")
    parser.add_argument("--onnx-dir", type=Path, required=True, help="Directory containing encoder.onnx and decoder.onnx")
    parser.add_argument("--manifest", type=Path, required=True, help="CSV manifest with audio_path and split")
    parser.add_argument("--data-root", type=Path, required=True, help="Root directory for audio files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save parity results")
    parser.add_argument("--split", default="test_in_domain", help="Split name to evaluate")
    parser.add_argument("--limit", type=int, default=20, help="Number of utterances to test (0 for all)")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"Running Parity Verification on device: {device}")

    print(f"Loading PyTorch checkpoint: {args.model_id}...")
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id).to(device).eval()
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    encoder_path = args.onnx_dir / "encoder.onnx"
    decoder_path = args.onnx_dir / "decoder.onnx"
    print(f"Loading ONNX Encoder: {encoder_path}...")
    ort_sess_opts = ort.SessionOptions()
    ort_sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_encoder = ort.InferenceSession(str(encoder_path), ort_sess_opts, providers=["CPUExecutionProvider"])
    print(f"Loading ONNX Decoder: {decoder_path}...")
    ort_decoder = ort.InferenceSession(str(decoder_path), ort_sess_opts, providers=["CPUExecutionProvider"])

    forced_prompt = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    prompt_ids = [model.config.decoder_start_token_id] + [token_id for _, token_id in forced_prompt]
    print(f"Forced Vietnamese Prompt IDs: {prompt_ids}")

    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == args.split]

    if args.limit > 0:
        rows = rows[: args.limit]
    print(f"Evaluating {len(rows)} utterances from split '{args.split}'...")

    results = []
    encoder_max_diffs = []
    encoder_cosines = []
    decoder_max_diffs = []
    decoder_cosines = []
    text_match_count = 0
    token_match_count = 0

    for idx, row in enumerate(rows, start=1):
        audio_path = args.data_root / row["audio_path"]
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).input_features

        features_np = input_features.numpy()
        features_pt = input_features.to(device)

        # 1. Compare Encoder
        with torch.no_grad():
            pt_hidden = model.model.encoder(features_pt).last_hidden_state.cpu().numpy()
        ort_hidden = ort_encoder.run(["encoder_hidden_states"], {"input_features": features_np})[0]

        enc_diff = float(np.max(np.abs(pt_hidden - ort_hidden)))
        enc_cos = cosine_similarity(pt_hidden, ort_hidden)
        encoder_max_diffs.append(enc_diff)
        encoder_cosines.append(enc_cos)

        # 2. Compare Decoder on initial prompt
        dec_prompt_np = np.array([prompt_ids], dtype=np.int64)
        dec_prompt_pt = torch.tensor(dec_prompt_np, dtype=torch.long, device=device)

        with torch.no_grad():
            pt_decoder_out = model.model.decoder(
                input_ids=dec_prompt_pt,
                encoder_hidden_states=torch.tensor(ort_hidden, device=device),
                use_cache=False,
            )
            pt_logits = model.proj_out(pt_decoder_out.last_hidden_state).cpu().numpy()

        ort_logits = ort_decoder.run(["logits"], {
            "decoder_input_ids": dec_prompt_np,
            "encoder_hidden_states": ort_hidden,
        })[0]

        dec_diff = float(np.max(np.abs(pt_logits - ort_logits)))
        dec_cos = cosine_similarity(pt_logits, ort_logits)
        decoder_max_diffs.append(dec_diff)
        decoder_cosines.append(dec_cos)

        # 3. End-to-end ONNX greedy decode
        t0 = time.perf_counter()
        onnx_tokens = onnx_greedy_decode(
            ort_encoder, ort_decoder, features_np, prompt_ids, model.config.eos_token_id, max_new_tokens=100
        )
        onnx_time = time.perf_counter() - t0

        # PyTorch greedy decode
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_gen = model.generate(
                features_pt,
                forced_decoder_ids=forced_prompt,
                max_new_tokens=100,
                do_sample=False,
            )[0].cpu().tolist()
        pt_time = time.perf_counter() - t0

        # Strip prompt tokens and trailing eos for content comparison
        eos_id = model.config.eos_token_id
        onnx_content = [t for t in onnx_tokens if t not in prompt_ids and t != eos_id]
        pt_content = [t for t in pt_gen if t not in prompt_ids and t != eos_id]
        tokens_match = (onnx_content == pt_content)
        if tokens_match:
            token_match_count += 1

        onnx_text = processor.tokenizer.decode(onnx_tokens, skip_special_tokens=True).strip()
        pt_text = processor.tokenizer.decode(pt_gen, skip_special_tokens=True).strip()
        text_match = (onnx_text == pt_text)
        if text_match:
            text_match_count += 1

        print(f"[{idx}/{len(rows)}] {row.get('audio_id', audio_path.name)} | EncCos: {enc_cos:.6f} | DecCos: {dec_cos:.6f} | TextMatch: {text_match} | TokensMatch: {tokens_match}")

        results.append({
            "audio_id": row.get("audio_id", audio_path.stem),
            "encoder_max_abs_diff": enc_diff,
            "encoder_cosine_sim": enc_cos,
            "decoder_max_abs_diff": dec_diff,
            "decoder_cosine_sim": dec_cos,
            "tokens_match": tokens_match,
            "text_match": text_match,
            "onnx_latency_s": onnx_time,
            "pytorch_latency_s": pt_time,
            "onnx_text": onnx_text,
            "pytorch_text": pt_text,
            "reference_text": row.get("text_normalized", ""),
        })

    summary = {
        "model_id": args.model_id,
        "onnx_dir": str(args.onnx_dir),
        "total_samples": len(rows),
        "text_matched_samples": text_match_count,
        "text_match_rate": text_match_count / len(rows) if rows else 0.0,
        "tokens_matched_samples": token_match_count,
        "token_match_rate": token_match_count / len(rows) if rows else 0.0,
        "encoder_mean_cosine_sim": float(np.mean(encoder_cosines)),
        "encoder_min_cosine_sim": float(np.min(encoder_cosines)),
        "encoder_max_abs_diff": float(np.max(encoder_max_diffs)),
        "decoder_mean_cosine_sim": float(np.mean(decoder_cosines)),
        "decoder_min_cosine_sim": float(np.min(decoder_cosines)),
        "decoder_max_abs_diff": float(np.max(decoder_max_diffs)),
        "parity_pass": bool(text_match_count == len(rows) and np.min(encoder_cosines) > 0.999),
    }

    csv_path = args.output_dir / "parity_results.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    json_path = args.output_dir / "parity_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("PARITY VERIFICATION SUMMARY:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("=" * 50)
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
