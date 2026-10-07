import json
from pathlib import Path
from typing import Optional
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def set_style():
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8

def plot_benchmark_results(results_dir: str = "benchmark_results", output_dir: Optional[str] = None):
    results_path = Path(results_dir)
    save_path = Path(output_dir or results_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    json_path = results_path / "summary_metrics.json"
    csv_path = results_path / "detailed_results.csv"

    if not json_path.exists() or not csv_path.exists():
        print(f"Results not found in {results_path}. Please run benchmark first.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    df = pd.read_csv(csv_path)
    set_style()

    # 1. Chart: RTF Comparison (Baseline vs Medusa)
    if "baseline_avg_rtf" in summary["overall"]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        categories = ["Tổng thể (All)", "Tiếng Anh (EN)", "Tiếng Việt (VI)"]
        base_rtfs = [
            summary["overall"]["baseline_avg_rtf"],
            summary["by_language"].get("en", {}).get("baseline_avg_rtf", 0),
            summary["by_language"].get("vi", {}).get("baseline_avg_rtf", 0),
        ]
        medusa_rtfs = [
            summary["overall"]["medusa_avg_rtf"],
            summary["by_language"].get("en", {}).get("medusa_avg_rtf", 0),
            summary["by_language"].get("vi", {}).get("medusa_avg_rtf", 0),
        ]

        x = range(len(categories))
        width = 0.35

        rects1 = ax.bar([i - width/2 for i in x], base_rtfs, width, label="Vanilla Whisper", color="#4A90E2")
        rects2 = ax.bar([i + width/2 for i in x], medusa_rtfs, width, label="Whisper-Medusa", color="#E94E77")

        ax.set_ylabel("Real-Time Factor (RTF) - Càng thấp càng tốt", fontsize=11, fontweight="bold")
        ax.set_title("So sánh Tốc độ RTF: Vanilla Whisper vs Whisper-Medusa", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.axhline(1.0, color="grey", linestyle="--", alpha=0.7, label="Thời gian thực (RTF=1.0)")
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

        for rect in rects1 + rects2:
            height = rect.get_height()
            if height > 0:
                ax.annotate(f"{height:.3f}",
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        rtf_fig_path = save_path / "chart_rtf_comparison.png"
        plt.savefig(rtf_fig_path)
        plt.close()
        print(f"Chart saved: {rtf_fig_path}")

    # 2. Chart: Speedup Factor across Industrial Domains
    if "speedup_factor" in df.columns:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        domain_speedup = df.groupby("domain")["speedup_factor"].mean().sort_values(ascending=True)

        colors = ["#2ECC71" if s >= 1.3 else "#3498DB" for s in domain_speedup.values]
        bars = ax.barh(domain_speedup.index, domain_speedup.values, color=colors, height=0.6)

        ax.set_xlabel("Hệ số Tăng tốc (Speedup Factor: Baseline Latency / Medusa Latency)", fontsize=11, fontweight="bold")
        ax.set_title("Tốc độ Tăng tốc của Whisper-Medusa theo Miền Ứng dụng Công nghiệp", fontsize=13, fontweight="bold", pad=15)
        ax.axvline(1.0, color="red", linestyle="--", label="Mốc Baseline (1.0x)")

        for bar in bars:
            width = bar.get_width()
            ax.annotate(f"{width:.2f}x",
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(5, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=9, fontweight="bold")

        ax.legend(loc="lower right")
        plt.tight_layout()
        speedup_fig_path = save_path / "chart_domain_speedup.png"
        plt.savefig(speedup_fig_path)
        plt.close()
        print(f"Chart saved: {speedup_fig_path}")

    # 3. Chart: WER Comparison (Accuracy preservation)
    if "baseline_wer" in summary["overall"]:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        categories = ["Tổng thể (All)", "Tiếng Anh (EN)", "Tiếng Việt (VI)"]
        base_wer = [
            summary["overall"]["baseline_wer"],
            summary["by_language"].get("en", {}).get("baseline_wer", 0),
            summary["by_language"].get("vi", {}).get("baseline_wer", 0),
        ]
        medusa_wer = [
            summary["overall"]["medusa_wer"],
            summary["by_language"].get("en", {}).get("medusa_wer", 0),
            summary["by_language"].get("vi", {}).get("medusa_wer", 0),
        ]

        x = range(len(categories))
        width = 0.35

        rects1 = ax.bar([i - width/2 for i in x], base_wer, width, label="Vanilla Whisper", color="#34495E")
        rects2 = ax.bar([i + width/2 for i in x], medusa_wer, width, label="Whisper-Medusa", color="#9B59B6")

        ax.set_ylabel("Word Error Rate - WER (%) - Càng thấp càng tốt", fontsize=11, fontweight="bold")
        ax.set_title("Độ chính xác Nhận dạng (WER): Vanilla Whisper vs Whisper-Medusa", fontsize=13, fontweight="bold", pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")

        for rect in rects1 + rects2:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.tight_layout()
        wer_fig_path = save_path / "chart_wer_comparison.png"
        plt.savefig(wer_fig_path)
        plt.close()
        print(f"Chart saved: {wer_fig_path}")

if __name__ == "__main__":
    plot_benchmark_results()
