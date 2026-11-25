#!/usr/bin/env python3
"""Diagnostic Script for Whisper Quantization Collapse (Tasks 1, 2, 3).

1. Compares Encoder outputs of FP32 vs W8A16 vs INT8 via Cosine Similarity, MAE, L2 error.
2. Evaluates Decoder step-1 logits:
   - First predicted token
   - Probability of EOS (50257)
   - Top-5 candidate tokens and probabilities
   - Total number of tokens generated
3. Tests cross-combinations:
   - (Encoder FP32, Decoder FP32)
   - (Encoder FP32, Decoder INT8)  <-- isolate whether decoder_int8 is broken
   - (Encoder W8A16, Decoder INT8) <-- W8A16 tier
   - (Encoder INT8, Decoder INT8)  <-- pure INT8 tier
   - (Encoder INT8, Decoder FP32)  <-- isolate whether encoder_int8 causes collapse
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List

import numpy as np
import onnxruntime as ort
import soundfile as sf
from transformers import WhisperProcessor


def compute_cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Computes cosine similarity between two flattened arrays."""
    a_flat = a.astype(np.float64).flatten()
    b_flat = b.astype(np.float64).flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def run_diagnostics(
    models_dir: Path,
    test_dir: Path,
    output_dir: Path,
    limit: int = 20,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ort_opts = ort.SessionOptions()
    ort_opts.intra_op_num_threads = 4
    ort_opts.inter_op_num_threads = 1
    ort_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print("Loading Whisper Processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    forced_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe", no_timestamps=True)
    prompt_ids = [processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    for _, tid in forced_ids:
        prompt_ids.append(tid)
    eos_id = processor.tokenizer.eos_token_id

    # 1. Load Sessions
    print("\nLoading Encoders...")
    enc_fp32 = ort.InferenceSession(str(models_dir / "encoder.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    enc_w8a16 = ort.InferenceSession(str(models_dir / "encoder_w8a16.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    enc_int8 = ort.InferenceSession(str(models_dir / "encoder_int8.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"])

    print("Loading Decoders...")
    dec_fp32 = ort.InferenceSession(str(models_dir / "decoder.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"])
    dec_int8 = ort.InferenceSession(str(models_dir / "decoder_int8.onnx"), sess_options=ort_opts, providers=["CPUExecutionProvider"])

    wav_files = sorted(test_dir.glob("*.wav"))[:limit]
    print(f"\nProcessing {len(wav_files)} test audio samples for diagnostics...")

    encoder_comparison_records = []
    decoder_behavior_records = []

    for idx, wav_path in enumerate(wav_files, start=1):
        aid = wav_path.stem
        speech, sr = sf.read(str(wav_path))
        if speech.ndim > 1:
            speech = speech.mean(axis=1)
        duration_s = len(speech) / sr

        features = processor.feature_extractor(speech, sampling_rate=sr, return_tensors="np").input_features.astype(np.float32)

        # ---------------------------------------------------------
        # TASK 1: Compare Encoder outputs
        # ---------------------------------------------------------
        out_fp32 = enc_fp32.run(None, {"input_features": features})[0]
        out_w8a16 = enc_w8a16.run(None, {"input_features": features})[0]
        out_int8 = enc_int8.run(None, {"input_features": features})[0]

        # Cosine similarity and errors
        cos_fp32_w8a16 = compute_cosine_similarity(out_fp32, out_w8a16)
        cos_fp32_int8 = compute_cosine_similarity(out_fp32, out_int8)
        mae_w8a16 = float(np.mean(np.abs(out_fp32 - out_w8a16)))
        mae_int8 = float(np.mean(np.abs(out_fp32 - out_int8)))
        max_err_w8a16 = float(np.max(np.abs(out_fp32 - out_w8a16)))
        max_err_int8 = float(np.max(np.abs(out_fp32 - out_int8)))

        encoder_comparison_records.append({
            "audio_id": aid,
            "duration_s": duration_s,
            "cos_sim_w8a16_vs_fp32": cos_fp32_w8a16,
            "cos_sim_int8_vs_fp32": cos_fp32_int8,
            "mae_w8a16": mae_w8a16,
            "mae_int8": mae_int8,
            "max_abs_err_w8a16": max_err_w8a16,
            "max_abs_err_int8": max_err_int8,
        })

        # ---------------------------------------------------------
        # TASK 2: Step-1 Logits, EOS Probability & Decoding Behavior
        # Combinations:
        # A: FP32 Enc + FP32 Dec
        # B: FP32 Enc + INT8 Dec
        # C: W8A16 Enc + INT8 Dec
        # D: INT8 Enc + INT8 Dec
        # E: INT8 Enc + FP32 Dec
        # ---------------------------------------------------------
        combos = [
            ("FP32_Enc_FP32_Dec", out_fp32, dec_fp32),
            ("FP32_Enc_INT8_Dec", out_fp32, dec_int8),
            ("W8A16_Enc_INT8_Dec", out_w8a16, dec_int8),
            ("INT8_Enc_INT8_Dec", out_int8, dec_int8),
            ("INT8_Enc_FP32_Dec", out_int8, dec_fp32),
        ]

        for combo_name, enc_out, dec_sess in combos:
            dec_inputs = dec_sess.get_inputs()
            input_ids_name = dec_inputs[0].name
            hidden_name = dec_inputs[1].name

            # Format hidden states
            h_in = enc_out.astype(np.float32)

            # Step 1: Prompt decoding
            curr_tokens = list(prompt_ids)
            input_ids = np.array([curr_tokens], dtype=np.int64)

            logits = dec_sess.run(None, {input_ids_name: input_ids, hidden_name: h_in})[0]
            step1_logits = logits[0, -1, :]
            step1_probs = softmax(step1_logits)

            first_token_id = int(np.argmax(step1_logits))
            first_token_prob = float(step1_probs[first_token_id])
            first_token_text = processor.tokenizer.decode([first_token_id])

            eos_prob = float(step1_probs[eos_id])
            eos_rank = int(np.sum(step1_logits > step1_logits[eos_id])) + 1

            # Top 3 candidates
            top3_ids = np.argsort(step1_logits)[::-1][:3]
            top3_info = [(int(t), processor.tokenizer.decode([t]), float(step1_probs[t])) for t in top3_ids]

            # Autoregressive generation
            generated = [first_token_id]
            curr_tokens.append(first_token_id)
            if first_token_id != eos_id:
                for _ in range(63):
                    input_ids = np.array([curr_tokens], dtype=np.int64)
                    logits = dec_sess.run(None, {input_ids_name: input_ids, hidden_name: h_in})[0]
                    next_token = int(np.argmax(logits[0, -1, :]))
                    curr_tokens.append(next_token)
                    generated.append(next_token)
                    if next_token == eos_id:
                        break

            pred_text = processor.tokenizer.decode([t for t in generated if t != eos_id]).strip()

            decoder_behavior_records.append({
                "audio_id": aid,
                "combo_name": combo_name,
                "first_token_id": first_token_id,
                "first_token_text": first_token_text,
                "first_token_prob": first_token_prob,
                "eos_prob_step1": eos_prob,
                "eos_rank_step1": eos_rank,
                "top3_candidates": str(top3_info),
                "num_tokens_generated": len(generated),
                "terminated_by_eos": (generated[-1] == eos_id),
                "predicted_text": pred_text,
            })

        print(f"[{idx}/{len(wav_files)}] {aid} analyzed across all 5 configurations.")

    # Save CSVs using built-in csv module (no pandas dependency)
    import csv
    if encoder_comparison_records:
        enc_csv = output_dir / "diagnostic_encoder_comparison.csv"
        with open(enc_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(encoder_comparison_records[0].keys()))
            writer.writeheader()
            writer.writerows(encoder_comparison_records)
        print(f"Saved {enc_csv}")

    if decoder_behavior_records:
        dec_csv = output_dir / "diagnostic_decoder_behavior.csv"
        with open(dec_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(decoder_behavior_records[0].keys()))
            writer.writeheader()
            writer.writerows(decoder_behavior_records)
        print(f"Saved {dec_csv}")

    # Statistical summary using numpy
    w8_cos = [r["cos_sim_w8a16_vs_fp32"] for r in encoder_comparison_records]
    int8_cos = [r["cos_sim_int8_vs_fp32"] for r in encoder_comparison_records]
    w8_mae = [r["mae_w8a16"] for r in encoder_comparison_records]
    int8_mae = [r["mae_int8"] for r in encoder_comparison_records]

    summary = {
        "encoder_cosine_similarity": {
            "w8a16_vs_fp32": {
                "mean": float(np.mean(w8_cos)),
                "min": float(np.min(w8_cos)),
                "max": float(np.max(w8_cos)),
                "mean_mae": float(np.mean(w8_mae)),
            },
            "int8_vs_fp32": {
                "mean": float(np.mean(int8_cos)),
                "min": float(np.min(int8_cos)),
                "max": float(np.max(int8_cos)),
                "mean_mae": float(np.mean(int8_mae)),
            },
        },
        "decoder_behavior_by_combo": {},
    }

    # Group decoder behavior by combo_name
    from collections import Counter
    unique_combos = sorted(list(set(r["combo_name"] for r in decoder_behavior_records)))
    for combo in unique_combos:
        sub = [r for r in decoder_behavior_records if r["combo_name"] == combo]
        tokens_gen = [r["num_tokens_generated"] for r in sub]
        eos_probs = [r["eos_prob_step1"] for r in sub]
        eos_ranks = [r["eos_rank_step1"] for r in sub]
        term_eos = [1.0 if r["terminated_by_eos"] else 0.0 for r in sub]
        first_token_texts = [r["first_token_text"] for r in sub]

        summary["decoder_behavior_by_combo"][combo] = {
            "mean_tokens_generated": float(np.mean(tokens_gen)),
            "median_tokens_generated": float(np.median(tokens_gen)),
            "mean_eos_prob_step1": float(np.mean(eos_probs)),
            "median_eos_rank_step1": float(np.median(eos_ranks)),
            "eos_termination_rate": float(np.mean(term_eos)),
            "first_tokens_sample": dict(Counter(first_token_texts).most_common(5)),
        }

    sum_json = output_dir / "diagnostic_summary.json"
    with open(sum_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {sum_json}")

    print("\n" + "=" * 80)
    print("DIAGNOSTIC SUMMARY COMPLETE")
    print("=" * 80)
    print(f"Encoder Cosine Sim W8A16 vs FP32 : {summary['encoder_cosine_similarity']['w8a16_vs_fp32']['mean']:.6f} (Min: {summary['encoder_cosine_similarity']['w8a16_vs_fp32']['min']:.6f})")
    print(f"Encoder Cosine Sim INT8 vs FP32  : {summary['encoder_cosine_similarity']['int8_vs_fp32']['mean']:.6f} (Min: {summary['encoder_cosine_similarity']['int8_vs_fp32']['min']:.6f})")
    print("\nDecoder Behavior:")
    for c, v in summary["decoder_behavior_by_combo"].items():
        print(f"  [{c}]: Mean Tokens = {v['mean_tokens_generated']:.1f} | Mean EOS Prob Step 1 = {v['mean_eos_prob_step1']:.4f} | Median EOS Rank = {v['median_eos_rank_step1']}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", type=Path, default=Path("/home/arduino/models"))
    parser.add_argument("--test-dir", type=Path, default=Path("/home/arduino/test_waves"))
    parser.add_argument("--output-dir", type=Path, default=Path("/home/arduino/output/EXP-DIAGNOSTIC"))
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    run_diagnostics(args.models_dir, args.test_dir, args.output_dir, args.limit)


if __name__ == "__main__":
    main()
