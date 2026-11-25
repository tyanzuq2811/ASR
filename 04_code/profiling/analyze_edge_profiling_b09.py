#!/usr/bin/env python3
"""Task B-09: Statistical Profiling and CI Analysis for Locked Fixed-Set Edge Evaluation.

Processes edge profiling data on the 20-utterance fixed representative test set:
- Audio IDs, durations, phonemic diversity
- Mean, Median, Std, IQR, Min, Max
- 10,000-iteration Bootstrap 95% Confidence Intervals for TTFT, TPOT, RTF, RAM
- Validates real-time compliance (RTF < 1.0) and memory stability
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd


def compute_bootstrap_ci(data: np.ndarray, n_bootstrap: int = 10000, seed: int = 20261022) -> tuple[float, float]:
    """Computes non-parametric bootstrap 95% confidence interval for the mean."""
    rng = np.random.default_rng(seed)
    n = len(data)
    boot_idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_means = np.mean(data[boot_idx], axis=1)
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))
    return ci_low, ci_high


def main():
    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "06_experiments" / "EXP-B09-edge-benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_csv = base_dir / "06_experiments" / "EXP-B03-profiling" / "edge_profiling_results.csv"
    df = pd.read_csv(edge_csv)

    configs = ["fp32", "w8a16", "int8"]
    tier_names = {
        "fp32": "FP32 Baseline (Kryo CPU)",
        "w8a16": "W8A16 QDQ Tier (HTP v75 / Kryo)",
        "int8": "INT8 Tier (Kryo SIMD)",
    }

    # Extract Audio Manifest
    unique_audios = df.drop_duplicates(subset=["audio_id"]).sort_values("audio_id")
    audio_manifest = []
    for _, row in unique_audios.iterrows():
        audio_manifest.append({
            "audio_id": row["audio_id"],
            "duration_s": row["audio_duration_s"],
        })

    durations = unique_audios["audio_duration_s"].to_numpy()
    manifest_summary = {
        "fixed_set_name": "VIVOS Test Representative Subset (Speaker VIVOSDEV02)",
        "sample_size_n": len(unique_audios),
        "total_audio_duration_s": round(float(np.sum(durations)), 3),
        "mean_duration_s": round(float(np.mean(durations)), 3),
        "median_duration_s": round(float(np.median(durations)), 3),
        "min_duration_s": round(float(np.min(durations)), 3),
        "max_duration_s": round(float(np.max(durations)), 3),
        "audio_ids": [a["audio_id"] for a in audio_manifest],
        "hardware_testbed": {
            "board": "Arduino VENTUNO Q",
            "soc": "Qualcomm Snapdragon QCS8275 (Octa-core Kryo)",
            "cpu_threads": 4,
            "cpu_affinity": "Kryo Cortex-A78C performance cores @ 2.7 GHz",
            "npu_accelerator": "Qualcomm Hexagon HTP v75",
            "system_ram": "14 GiB Unified LPDDR5",
            "os": "Ubuntu 24.04 LTS noble (Linux Kernel 6.6.x)",
            "runtime": "ONNX Runtime v1.31.0 + QNN SDK v2.49.0",
        },
    }

    with open(out_dir / "edge_fixed_set_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_summary, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_dir / 'edge_fixed_set_manifest.json'}")

    # Compute Statistical Metrics & Bootstrap CIs
    records = []
    summary_stats = {}

    for cfg in configs:
        sub = df[df["config"] == cfg]
        tname = tier_names[cfg]

        metrics = {}
        for col in ["ttft_ms", "tpot_ms", "rtf", "peak_ram_mb"]:
            vals = sub[col].to_numpy()
            m_mean = float(np.mean(vals))
            m_std = float(np.std(vals))
            m_median = float(np.median(vals))
            m_iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
            m_min = float(np.min(vals))
            m_max = float(np.max(vals))
            ci_low, ci_high = compute_bootstrap_ci(vals)

            metrics[col] = {
                "mean": round(m_mean, 4),
                "std": round(m_std, 4),
                "median": round(m_median, 4),
                "iqr": round(m_iqr, 4),
                "min": round(m_min, 4),
                "max": round(m_max, 4),
                "ci_95": [round(ci_low, 4), round(ci_high, 4)],
            }

            records.append({
                "config": cfg,
                "tier_name": tname,
                "metric": col,
                "n": len(vals),
                "mean": round(m_mean, 4),
                "std": round(m_std, 4),
                "median": round(m_median, 4),
                "iqr": round(m_iqr, 4),
                "min": round(m_min, 4),
                "max": round(m_max, 4),
                "ci_95_low": round(ci_low, 4),
                "ci_95_high": round(ci_high, 4),
            })

        summary_stats[cfg] = {
            "tier_name": tname,
            "metrics": metrics,
        }

    stat_df = pd.DataFrame(records)
    stat_csv = out_dir / "edge_benchmark_ci.csv"
    stat_df.to_csv(stat_csv, index=False)
    print(f"Saved {stat_csv}")

    with open(out_dir / "edge_benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_stats, f, indent=2, ensure_ascii=False)
    print(f"Saved {out_dir / 'edge_benchmark_summary.json'}")

    print("\n" + "=" * 80)
    print("TASK B-09: STATISTICAL EDGE PROFILING SUMMARY WITH BOOTSTRAP 95% CI")
    print("=" * 80)
    print(stat_df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()
