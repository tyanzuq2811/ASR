#!/usr/bin/env python3
"""Task 4 & 5: Calibration Scaling and Sensitive Layer Preservation Experiments.

Task 4:
- Scale calibration from 20 samples to 100 samples.
- Multi-condition calibration: 25 Clean, 25 @ 20dB, 25 @ 10dB, 25 @ 5dB (MUSAN noise).
- Builds encoder_int8_calib100_noisy.onnx.

Task 5:
- Identify and preserve sensitive layers:
  - Exclude Self-Attention (q_proj, k_proj, v_proj, out_proj, Softmax) from 8-bit activation clipping.
  - Only quantize Feed-Forward Network (fc1, fc2) MatMuls (FFN-only INT8).
- Builds encoder_int8_sensitive_preserved.onnx.

Evaluates:
- Cosine similarity vs FP32 across all 20 test utterances.
- Decoder generation length and EOS probability.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
import soundfile as sf
import torch
from transformers import WhisperProcessor


def add_musan_noise(clean_speech: np.ndarray, noise_speech: np.ndarray, snr_db: float) -> np.ndarray:
    """Adds noise to speech at specified SNR in dB."""
    if len(noise_speech) < len(clean_speech):
        repeats = int(np.ceil(len(clean_speech) / len(noise_speech)))
        noise_speech = np.tile(noise_speech, repeats)
    start_idx = random.randint(0, len(noise_speech) - len(clean_speech))
    noise_segment = noise_speech[start_idx : start_idx + len(clean_speech)]

    s_power = np.mean(clean_speech ** 2)
    n_power = np.mean(noise_segment ** 2)
    if s_power == 0 or n_power == 0:
        return clean_speech

    snr_factor = 10.0 ** (-snr_db / 20.0)
    scaled_noise = noise_segment * np.sqrt(s_power / n_power) * snr_factor
    mixed = clean_speech + scaled_noise
    # Prevent clipping
    max_val = np.max(np.abs(mixed))
    if max_val > 0.99:
        mixed = mixed / max_val * 0.99
    return mixed.astype(np.float32)


class DirectoryCalibrationDataReader(CalibrationDataReader):
    def __init__(self, raw_dir: Path, input_name: str = "input_features"):
        self.raw_files = sorted(raw_dir.glob("*.raw"))
        self.input_name = input_name
        self.enum_data = iter(self.raw_files)
        print(f"DirectoryCalibrationDataReader initialized with {len(self.raw_files)} raw files.")

    def get_next(self) -> dict | None:
        try:
            path = next(self.enum_data)
        except StopIteration:
            return None
        data = np.fromfile(str(path), dtype=np.float32).reshape(1, 80, 3000)
        return {self.input_name: data}

    def rewind(self):
        self.enum_data = iter(self.raw_files)


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.astype(np.float64).flatten()
    b_f = b.astype(np.float64).flatten()
    return float(np.dot(a_f, b_f) / (np.linalg.norm(a_f) * np.linalg.norm(b_f)))


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data-dir", type=Path, default=Path("/hdd3/users/baolt/ICCIT_2026_VENTUNO_Q_Paper_DATA"))
    parser.add_argument("--work-dir", type=Path, default=Path("/home/dunglt/exp_tasks_4_5"))
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    calib_raw_dir = args.work_dir / "calib_raw"
    calib_raw_dir.mkdir(parents=True, exist_ok=True)

    vivos_root = args.base_data_dir / "03_data" / "raw" / "vivos" / "vivos"
    musan_root = args.base_data_dir / "03_data" / "raw" / "musan" / "extracted" / "musan" / "noise"
    manifest_csv = args.base_data_dir / "03_data" / "processed" / "vivos_speaker_split.csv"
    encoder_fp32_path = args.base_data_dir / "05_models" / "onnx" / "EXP-A08-noise-aware" / "run-001" / "encoder.onnx"
    decoder_int8_path = args.base_data_dir / "05_models" / "quantized" / "EXP-A08-noise-aware" / "run-001" / "decoder_int8.onnx"

    print("=" * 80)
    print("TASK 4 & 5: CALIBRATION SCALING & SENSITIVE LAYER PRESERVATION")
    print("=" * 80)

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

    # -------------------------------------------------------------
    # 1. GENERATE TASK 4 CALIBRATION DATA (100 samples: Clean + Multi-SNR Noise)
    # -------------------------------------------------------------
    print("\n--- [TASK 4] Generating 100 Multi-SNR Calibration Tensors ---")
    with open(manifest_csv, "r", encoding="utf-8") as f:
        train_rows = [r for r in csv.DictReader(f) if r.get("split") == "train"]

    noise_wavs = sorted(musan_root.rglob("*.wav"))
    print(f"Found {len(train_rows)} VIVOS training utterances and {len(noise_wavs)} MUSAN noise files.")

    num_samples = 100
    selected_train = train_rows[:num_samples]
    conditions = ["clean"] * 25 + [20.0] * 25 + [10.0] * 25 + [5.0] * 25

    calib_files = []
    for idx, (row, cond) in enumerate(zip(selected_train, conditions), start=1):
        raw_file = calib_raw_dir / f"calib_{idx:03d}_{cond}.raw"
        if not raw_file.exists():
            wav_path = vivos_root / row["audio_path"]
            speech, sr = sf.read(str(wav_path), dtype="float32")
            if speech.ndim > 1:
                speech = speech.mean(axis=1)

            if cond != "clean":
                noise_f = random.choice(noise_wavs)
                noise_speech, n_sr = sf.read(str(noise_f), dtype="float32")
                if noise_speech.ndim > 1:
                    noise_speech = noise_speech.mean(axis=1)
                speech = add_musan_noise(speech, noise_speech, float(cond))

            features = processor.feature_extractor(speech, sampling_rate=16000, return_tensors="np").input_features.astype(np.float32)
            features.tofile(str(raw_file))

        calib_files.append(raw_file)
        if idx % 25 == 0:
            print(f"  Generated {idx}/100 calibration files (Condition: {cond})")

    # -------------------------------------------------------------
    # 2. RUN TASK 4: Static INT8 with 100 Multi-Condition Samples
    # -------------------------------------------------------------
    enc_int8_calib100 = args.work_dir / "encoder_int8_calib100_noisy.onnx"
    if not enc_int8_calib100.exists():
        print("\n--- [TASK 4] Quantizing INT8 with 100 Multi-Condition Calibration Samples ---")
        reader100 = DirectoryCalibrationDataReader(calib_raw_dir)
        quantize_static(
            model_input=str(encoder_fp32_path),
            model_output=str(enc_int8_calib100),
            calibration_data_reader=reader100,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
        )
        print(f"Generated {enc_int8_calib100.name}: {enc_int8_calib100.stat().st_size:,} bytes")
    else:
        print(f"Using existing {enc_int8_calib100.name}")

    # -------------------------------------------------------------
    # 3. RUN TASK 5: Sensitive Layer Preservation (Mixed-Precision / Outlier-Safe INT8)
    # -------------------------------------------------------------
    print("\n--- [TASK 5] Analyzing Encoder Graph for Sensitive Layers ---")
    model_onnx = onnx.load(str(encoder_fp32_path))
    nodes_to_exclude = []

    # Exclude all Self-Attention nodes (Q/K/V projections, out_proj, and Softmax) + LayerNorm
    for node in model_onnx.graph.node:
        name_lower = node.name.lower()
        # Check if node belongs to self_attn or layer_norm
        if any(kw in name_lower for kw in ["self_attn", "layer_norm", "layernorm", "attn", "softmax"]):
            nodes_to_exclude.append(node.name)
        elif node.op_type in ["Softmax", "LayerNormalization", "ReduceMean"]:
            nodes_to_exclude.append(node.name)

    print(f"Identified {len(nodes_to_exclude)} sensitive nodes to EXCLUDE from 8-bit activation quantization:")
    print(f"  Sample excluded nodes: {nodes_to_exclude[:5]} ... {nodes_to_exclude[-3:]}")

    enc_int8_sensitive = args.work_dir / "encoder_int8_sensitive_preserved.onnx"
    if not enc_int8_sensitive.exists():
        print("\n--- [TASK 5] Quantizing Mixed-Precision INT8 (Sensitive Attention/Norm Excluded) ---")
        reader100 = DirectoryCalibrationDataReader(calib_raw_dir)
        reader100.rewind()
        quantize_static(
            model_input=str(encoder_fp32_path),
            model_output=str(enc_int8_sensitive),
            calibration_data_reader=reader100,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            weight_type=QuantType.QInt8,
            activation_type=QuantType.QInt8,
            nodes_to_exclude=nodes_to_exclude,
        )
        print(f"Generated {enc_int8_sensitive.name}: {enc_int8_sensitive.stat().st_size:,} bytes")
    else:
        print(f"Using existing {enc_int8_sensitive.name}")

    # -------------------------------------------------------------
    # 4. BENCHMARK ALL VARIANTS ON THE 20 TEST AUDIO SAMPLES
    # -------------------------------------------------------------
    print("\n--- Evaluating Cosine Similarity & Decoding Behavior on 20 Test Samples ---")
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = 4
    ort_opts.inter_op_num_threads = 1

    models = {
        "FP32_Baseline": ort.InferenceSession(str(encoder_fp32_path), sess_options=ort_opts, providers=["CPUExecutionProvider"]),
        "W8A16_Paper_Tier": ort.InferenceSession(str(args.base_data_dir / "05_models" / "quantized" / "EXP-A08-noise-aware" / "run-001" / "encoder_w8a16.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"]),
        "INT8_Orig_20calib": ort.InferenceSession(str(args.base_data_dir / "05_models" / "quantized" / "EXP-A08-noise-aware" / "run-001" / "encoder_int8.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"]),
        "INT8_Task4_100noisy": ort.InferenceSession(str(enc_int8_calib100), sess_options=ort_opts, providers=["CPUExecutionProvider"]),
        "INT8_Task5_SensPreserved": ort.InferenceSession(str(enc_int8_sensitive), sess_options=ort_opts, providers=["CPUExecutionProvider"]),
    }

    decoder_sess = ort.InferenceSession(str(decoder_int8_path), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    dec_input_name = decoder_sess.get_inputs()[0].name
    dec_hidden_name = decoder_sess.get_inputs()[1].name

    forced_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe", no_timestamps=True)
    prompt_ids = [processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    for _, tid in forced_ids:
        prompt_ids.append(tid)
    eos_id = processor.tokenizer.eos_token_id

    test_waves_dir = Path("/home/dunglt/test_waves")
    test_files = sorted(test_waves_dir.glob("*.wav"))[:20]

    results_by_model = {m: {"cosine_sims": [], "tokens_gen": [], "eos_probs": [], "first_tokens": []} for m in models}

    for idx, wav_path in enumerate(test_files, start=1):
        speech, sr = sf.read(str(wav_path))
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        features = processor.feature_extractor(speech, sampling_rate=sr, return_tensors="np").input_features.astype(np.float32)

        # 1. FP32 reference
        out_fp32 = models["FP32_Baseline"].run(None, {"input_features": features})[0]

        for m_name, sess in models.items():
            if m_name == "FP32_Baseline":
                out_m = out_fp32
                sim = 1.0
            else:
                out_m = sess.run(None, {"input_features": features})[0]
                sim = compute_cosine_similarity(out_fp32, out_m)

            results_by_model[m_name]["cosine_sims"].append(sim)

            # Step 1 decoding with decoder_int8
            curr_tokens = list(prompt_ids)
            input_ids = np.array([curr_tokens], dtype=np.int64)
            logits = decoder_sess.run(None, {dec_input_name: input_ids, dec_hidden_name: out_m.astype(np.float32)})[0]
            step1_probs = softmax(logits[0, -1, :])

            first_t = int(np.argmax(logits[0, -1, :]))
            first_t_text = processor.tokenizer.decode([first_t])
            eos_p = float(step1_probs[eos_id])

            results_by_model[m_name]["eos_probs"].append(eos_p)
            results_by_model[m_name]["first_tokens"].append(first_t_text)

            # Autoregressive generation
            generated = [first_t]
            curr_tokens.append(first_t)
            if first_t != eos_id:
                for _ in range(63):
                    input_ids = np.array([curr_tokens], dtype=np.int64)
                    l = decoder_sess.run(None, {dec_input_name: input_ids, dec_hidden_name: out_m.astype(np.float32)})[0]
                    nxt = int(np.argmax(l[0, -1, :]))
                    curr_tokens.append(nxt)
                    generated.append(nxt)
                    if nxt == eos_id:
                        break

            results_by_model[m_name]["tokens_gen"].append(len(generated))

        print(f"[{idx}/20] Evaluated all variants on {wav_path.stem}")

    # Summary table
    table_rows = []
    for m_name, d in results_by_model.items():
        table_rows.append({
            "Configuration": m_name,
            "Encoder_Cos_Sim_Mean": round(float(np.mean(d["cosine_sims"])), 6),
            "Encoder_Cos_Sim_Min": round(float(np.min(d["cosine_sims"])), 6),
            "Mean_Tokens_Generated": round(float(np.mean(d["tokens_gen"])), 1),
            "Median_Tokens_Generated": round(float(np.median(d["tokens_gen"])), 1),
            "Mean_EOS_Prob_Step1": round(float(np.mean(d["eos_probs"])), 6),
            "First_Tokens_Distinct": len(set(d["first_tokens"])),
            "Status": "HEALTHY (Normal)" if np.mean(d["tokens_gen"]) > 10 else "COLLAPSED (EOS Early Stop)",
        })

    import pandas as pd
    res_df = pd.DataFrame(table_rows)
    print("\n" + "=" * 95)
    print("TASK 4 & 5 FINAL COMPARISON TABLE")
    print("=" * 95)
    print(res_df.to_string(index=False))
    print("=" * 95)

    res_df.to_csv(args.work_dir / "tasks_4_5_comparison_results.csv", index=False)
    with open(args.work_dir / "tasks_4_5_summary.json", "w", encoding="utf-8") as f:
        json.dump(table_rows, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {args.work_dir}")


if __name__ == "__main__":
    main()
