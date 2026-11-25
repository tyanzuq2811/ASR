#!/usr/bin/env python3
"""Compute exact empirical WER and CER on Arduino VENTUNO Q edge profiling results.

Compares predicted text from FP32, W8A16, and INT8 against official VIVOS reference text.
Outputs per-utterance accuracy CSV and statistical summary JSON.
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np


def levenshtein_distance(ref_tokens: list[str], hyp_tokens: list[str]) -> int:
    """Computes Levenshtein edit distance between token sequences."""
    r_len, h_len = len(ref_tokens), len(hyp_tokens)
    dp = np.zeros((r_len + 1, h_len + 1), dtype=int)
    for i in range(r_len + 1):
        dp[i, 0] = i
    for j in range(h_len + 1):
        dp[0, j] = j

    for i in range(1, r_len + 1):
        for j in range(1, h_len + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i, j] = dp[i - 1, j - 1]
            else:
                dp[i, j] = 1 + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return int(dp[r_len, h_len])


def compute_wer_cer(ref: str, hyp: str) -> tuple[float, float]:
    ref_clean = ref.strip().lower()
    hyp_clean = hyp.strip().lower()

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()

    # WER
    if not ref_words:
        wer = 1.0 if hyp_words else 0.0
    else:
        wer = levenshtein_distance(ref_words, hyp_words) / len(ref_words)

    # CER
    ref_chars = list(ref_clean.replace(" ", ""))
    hyp_chars = list(hyp_clean.replace(" ", ""))
    if not ref_chars:
        cer = 1.0 if hyp_chars else 0.0
    else:
        cer = levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)

    return float(wer), float(cer)


def main():
    base_dir = Path(__file__).resolve().parents[2]
    parity_csv = base_dir / "06_experiments" / "EXP-B01-parity" / "parity_results.csv"
    edge_csv = base_dir / "06_experiments" / "EXP-B03-profiling" / "edge_profiling_results.csv"

    ref_df = pd.read_csv(parity_csv)[["audio_id", "reference_text"]]
    ref_map = dict(zip(ref_df["audio_id"], ref_df["reference_text"]))

    edge_df = pd.read_csv(edge_csv)

    results = []
    for _, row in edge_df.iterrows():
        aid = row["audio_id"]
        cfg = row["config"]
        tier = row["tier_name"]
        pred = str(row["predicted_text"]) if pd.notna(row["predicted_text"]) else ""
        ref = ref_map.get(aid, "")

        wer, cer = compute_wer_cer(ref, pred)
        results.append({
            "config": cfg,
            "tier_name": tier,
            "audio_id": aid,
            "audio_duration_s": row["audio_duration_s"],
            "reference_text": ref,
            "predicted_text": pred,
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "ttft_ms": row["ttft_ms"],
            "tpot_ms": row["tpot_ms"],
            "rtf": row["rtf"],
            "peak_ram_mb": row["peak_ram_mb"],
        })

    out_df = pd.DataFrame(results)
    out_csv = base_dir / "06_experiments" / "EXP-B03-profiling" / "edge_accuracy_results.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    # Summary by config
    summary = {}
    for cfg in ["fp32", "w8a16", "int8"]:
        sub = out_df[out_df["config"] == cfg]
        wers = sub["wer"].tolist()
        cers = sub["cer"].tolist()
        summary[cfg] = {
            "tier_name": sub["tier_name"].iloc[0],
            "num_samples": len(sub),
            "wer": {
                "mean": round(float(np.mean(wers)), 4),
                "median": round(float(np.median(wers)), 4),
                "iqr": round(float(np.percentile(wers, 75) - np.percentile(wers, 25)), 4),
                "min": round(float(np.min(wers)), 4),
                "max": round(float(np.max(wers)), 4),
            },
            "cer": {
                "mean": round(float(np.mean(cers)), 4),
                "median": round(float(np.median(cers)), 4),
                "iqr": round(float(np.percentile(cers, 75) - np.percentile(cers, 25)), 4),
                "min": round(float(np.min(cers)), 4),
                "max": round(float(np.max(cers)), 4),
            },
        }

    out_json = base_dir / "06_experiments" / "EXP-B03-profiling" / "edge_accuracy_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_json}")
    print("\n--- EXACT EMPIRICAL EDGE ACCURACY ON ARDUINO VENTUNO Q ---")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
