#!/usr/bin/env python3
"""Task B-08: Empirical WER/CER Evaluation and Paired Bootstrap CI across Quantization Tiers.

Aligns predictions across:
- A30 GPU (CUDA / FP16 Native)
- Server CPU Reference (FP32)
- Edge FP32 Baseline (Kryo CPU)
- Edge W8A16 QDQ Tier (HTP v75 / Kryo CPU)
- Edge INT8 Tier (Kryo CPU SIMD)

Computes per-utterance WER/CER and 10,000-iteration paired bootstrap 95% confidence intervals.
Outputs raw predictions CSV, bootstrap CI CSV, and publication-ready summary JSON.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


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


def compute_wer_cer(ref: str, hyp: str) -> tuple[float, float, int, int, int, int]:
    """Computes WER and CER along with edit counts."""
    ref_clean = ref.strip().lower()
    hyp_clean = hyp.strip().lower()

    ref_words = ref_clean.split()
    hyp_words = hyp_clean.split()
    w_edits = levenshtein_distance(ref_words, hyp_words)
    w_len = len(ref_words)
    wer = (w_edits / w_len) if w_len > 0 else (1.0 if hyp_words else 0.0)

    ref_chars = list(ref_clean.replace(" ", ""))
    hyp_chars = list(hyp_clean.replace(" ", ""))
    c_edits = levenshtein_distance(ref_chars, hyp_chars)
    c_len = len(ref_chars)
    cer = (c_edits / c_len) if c_len > 0 else (1.0 if hyp_chars else 0.0)

    return float(wer), float(cer), w_edits, w_len, c_edits, c_len


def paired_bootstrap_ci(
    base_values: np.ndarray,
    target_values: np.ndarray,
    n_bootstrap: int = 10000,
    seed: int = 20261022,
) -> dict[str, float]:
    """Computes paired bootstrap 95% confidence intervals and p-value."""
    rng = np.random.default_rng(seed)
    n = len(base_values)
    diffs = target_values - base_values

    # Bootstrap resampling
    boot_indices = rng.integers(0, n, size=(n_bootstrap, n))
    boot_base_means = np.mean(base_values[boot_indices], axis=1)
    boot_target_means = np.mean(target_values[boot_indices], axis=1)
    boot_diff_means = np.mean(diffs[boot_indices], axis=1)

    # 95% Confidence Intervals
    base_ci = (float(np.percentile(boot_base_means, 2.5)), float(np.percentile(boot_base_means, 97.5)))
    target_ci = (float(np.percentile(boot_target_means, 2.5)), float(np.percentile(boot_target_means, 97.5)))
    diff_ci = (float(np.percentile(boot_diff_means, 2.5)), float(np.percentile(boot_diff_means, 97.5)))

    # Two-tailed p-value for H0: diff == 0
    obs_diff = np.mean(diffs)
    if obs_diff > 0:
        p_val = 2.0 * np.mean(boot_diff_means <= 0)
    elif obs_diff < 0:
        p_val = 2.0 * np.mean(boot_diff_means >= 0)
    else:
        p_val = 1.0
    p_val = min(1.0, float(p_val))

    return {
        "base_mean": float(np.mean(base_values)),
        "base_ci_95_low": base_ci[0],
        "base_ci_95_high": base_ci[1],
        "target_mean": float(np.mean(target_values)),
        "target_ci_95_low": target_ci[0],
        "target_ci_95_high": target_ci[1],
        "delta_mean": float(obs_diff),
        "delta_ci_95_low": diff_ci[0],
        "delta_ci_95_high": diff_ci[1],
        "p_value": p_val,
    }


def main():
    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "06_experiments" / "EXP-B08-quantization-eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Parity References
    parity_csv = base_dir / "06_experiments" / "EXP-B01-parity" / "parity_results.csv"
    parity_df = pd.read_csv(parity_csv)
    ref_map = dict(zip(parity_df["audio_id"], parity_df["reference_text"]))

    # 2. Load Edge Profiling Predictions (FP32, W8A16, INT8)
    edge_csv = base_dir / "06_experiments" / "EXP-B03-profiling" / "edge_profiling_results.csv"
    edge_df = pd.read_csv(edge_csv)

    # 3. Load Server-side A30 GPU & CPU clean results for same 20 utterances
    a08_gpu_csv = base_dir / "06_experiments" / "EXP-A08-noise-aware" / "run-001" / "vivos_noise_test_results.csv"
    a08_cpu_csv = base_dir / "06_experiments" / "EXP-A08-noise-aware" / "run-001" / "vivos_noise_test_results_cpu.csv"
    gpu_df = pd.read_csv(a08_gpu_csv)
    cpu_df = pd.read_csv(a08_cpu_csv)

    clean_gpu = gpu_df[gpu_df["snr_db"].isna() & gpu_df["utterance_id"].isin(ref_map.keys())]
    clean_cpu = cpu_df[cpu_df["snr_db"].isna() & cpu_df["utterance_id"].isin(ref_map.keys())]
    gpu_map = dict(zip(clean_gpu["utterance_id"], clean_gpu["hypothesis"]))
    cpu_map = dict(zip(clean_cpu["utterance_id"], clean_cpu["hypothesis"]))

    # 4. Construct unified table across all 20 utterances
    rows = []
    audio_ids = sorted(ref_map.keys())

    for aid in audio_ids:
        ref = ref_map[aid]

        # A30 GPU (FP16/CUDA)
        gpu_hyp = gpu_map.get(aid, "")
        gpu_wer, gpu_cer, _, _, _, _ = compute_wer_cer(ref, gpu_hyp)
        rows.append({
            "audio_id": aid,
            "platform": "Server NVIDIA A30",
            "tier_id": "a30_gpu_fp16",
            "tier_name": "A30 GPU Reference (CUDA/FP16)",
            "reference_text": ref,
            "hypothesis_text": gpu_hyp,
            "wer": gpu_wer,
            "cer": gpu_cer,
        })

        # Server CPU (FP32)
        cpu_hyp = cpu_map.get(aid, "")
        cpu_wer, cpu_cer, _, _, _, _ = compute_wer_cer(ref, cpu_hyp)
        rows.append({
            "audio_id": aid,
            "platform": "Server Intel/AMD CPU",
            "tier_id": "server_cpu_fp32",
            "tier_name": "Server CPU Reference (FP32)",
            "reference_text": ref,
            "hypothesis_text": cpu_hyp,
            "wer": cpu_wer,
            "cer": cpu_cer,
        })

        # Edge Tiers: FP32, W8A16, INT8
        for cfg_id, t_name in [("fp32", "Edge FP32 Baseline"), ("w8a16", "Edge W8A16 QDQ Tier"), ("int8", "Edge INT8 Tier")]:
            sub = edge_df[(edge_df["audio_id"] == aid) & (edge_df["config"] == cfg_id)]
            hyp = str(sub["predicted_text"].iloc[0]) if len(sub) > 0 and pd.notna(sub["predicted_text"].iloc[0]) else ""
            wer, cer, _, _, _, _ = compute_wer_cer(ref, hyp)
            rows.append({
                "audio_id": aid,
                "platform": "Arduino VENTUNO Q (QCS8275)",
                "tier_id": f"edge_{cfg_id}",
                "tier_name": t_name,
                "reference_text": ref,
                "hypothesis_text": hyp,
                "wer": wer,
                "cer": cer,
            })

    pred_df = pd.DataFrame(rows)
    pred_csv = out_dir / "quantization_tier_predictions.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"Saved {pred_csv}")

    # 5. Compute Paired Bootstrap CIs relative to Edge FP32 Baseline
    edge_fp32_sub = pred_df[pred_df["tier_id"] == "edge_fp32"].sort_values("audio_id")
    fp32_wer_arr = edge_fp32_sub["wer"].to_numpy()
    fp32_cer_arr = edge_fp32_sub["cer"].to_numpy()

    comparison_tiers = [
        ("a30_gpu_fp16", "A30 GPU Reference (CUDA/FP16)"),
        ("server_cpu_fp32", "Server CPU Reference (FP32)"),
        ("edge_w8a16", "Edge W8A16 QDQ Tier"),
        ("edge_int8", "Edge INT8 Tier"),
    ]

    ci_records = []
    summary_dict = {
        "manifest_info": {
            "num_samples": len(audio_ids),
            "manifest_type": "VIVOS Test Representative Fixed Set (VIVOSDEV02)",
            "bootstrap_replicates": 10000,
            "confidence_level": 0.95,
        },
        "baseline_edge_fp32": {
            "mean_wer": float(np.mean(fp32_wer_arr)),
            "median_wer": float(np.median(fp32_wer_arr)),
            "wer_ci_95": [float(np.percentile(fp32_wer_arr, 2.5)), float(np.percentile(fp32_wer_arr, 97.5))],
            "mean_cer": float(np.mean(fp32_cer_arr)),
            "median_cer": float(np.median(fp32_cer_arr)),
        },
        "comparisons_vs_edge_fp32": {},
    }

    for tid, tname in comparison_tiers:
        t_sub = pred_df[pred_df["tier_id"] == tid].sort_values("audio_id")
        t_wer = t_sub["wer"].to_numpy()
        t_cer = t_sub["cer"].to_numpy()

        wer_res = paired_bootstrap_ci(fp32_wer_arr, t_wer)
        cer_res = paired_bootstrap_ci(fp32_cer_arr, t_cer)

        ci_records.append({
            "comparison_tier": tname,
            "tier_id": tid,
            "baseline_mean_wer_pct": round(wer_res["base_mean"] * 100, 2),
            "target_mean_wer_pct": round(wer_res["target_mean"] * 100, 2),
            "delta_wer_abs_pct": round(wer_res["delta_mean"] * 100, 2),
            "delta_wer_ci_95_low": round(wer_res["delta_ci_95_low"] * 100, 2),
            "delta_wer_ci_95_high": round(wer_res["delta_ci_95_high"] * 100, 2),
            "wer_p_value": round(wer_res["p_value"], 4),
            "baseline_mean_cer_pct": round(cer_res["base_mean"] * 100, 2),
            "target_mean_cer_pct": round(cer_res["target_mean"] * 100, 2),
            "delta_cer_abs_pct": round(cer_res["delta_mean"] * 100, 2),
            "delta_cer_ci_95_low": round(cer_res["delta_ci_95_low"] * 100, 2),
            "delta_cer_ci_95_high": round(cer_res["delta_ci_95_high"] * 100, 2),
            "cer_p_value": round(cer_res["p_value"], 4),
        })

        summary_dict["comparisons_vs_edge_fp32"][tid] = {
            "tier_name": tname,
            "wer_analysis": wer_res,
            "cer_analysis": cer_res,
        }

    ci_df = pd.DataFrame(ci_records)
    ci_csv = out_dir / "quantization_tier_bootstrap_ci.csv"
    ci_df.to_csv(ci_csv, index=False)
    print(f"Saved {ci_csv}")

    summary_json = out_dir / "quantization_tier_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print(f"Saved {summary_json}")

    print("\n" + "=" * 75)
    print("TASK B-08: PAIRED BOOTSTRAP CI RESULTS RELATIVE TO EDGE FP32 BASELINE")
    print("=" * 75)
    print(ci_df.to_string(index=False))
    print("=" * 75)


if __name__ == "__main__":
    main()
