#!/usr/bin/env python3
"""Generate publication-ready 600 DPI figures for ICCIT 2026 paper.

Produces:
1. fig1_system_architecture.png: System Architecture & FastRPC Zero-Audio Security Boundary
2. fig2_tradeoff_latency_footprint_accuracy.png: Edge Trade-off (Footprint vs. TPOT/TTFT vs. RTF)
3. fig3_noise_robustness.png: Noise Robustness across SNR conditions (Clean, 20dB, 10dB, 5dB)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[2]
FIG_DIR = BASE_DIR / "07_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set global publication styling
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})


def generate_figure_1():
    """Figure 1: Edge ASR Architecture & Secure Zero-Audio RPC Boundary."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")

    # Title
    ax.text(5.0, 5.2, "Arduino VENTUNO Q Offline Edge ASR & RPC Security Boundary",
            ha="center", va="center", fontsize=13, fontweight="bold", color="#1A202C")

    # 1. Qualcomm Application Processor (AP) Box
    ap_box = patches.FancyBboxPatch((0.4, 0.6), 5.6, 4.2, boxstyle="round,pad=0.2",
                                    facecolor="#EBF8FF", edgecolor="#2B6CB0", linewidth=2)
    ax.add_patch(ap_box)
    ax.text(3.2, 4.5, "Qualcomm Snapdragon QCS8275 (Application Processor)",
            ha="center", va="center", fontsize=11, fontweight="bold", color="#2B6CB0")

    # Audio input module
    audio_box = patches.FancyBboxPatch((0.8, 3.4), 2.2, 0.7, boxstyle="round,pad=0.1",
                                       facecolor="#FFFFFF", edgecolor="#4A5568", linewidth=1.2)
    ax.add_patch(audio_box)
    ax.text(1.9, 3.75, "I2S Audio Codec\n(16 kHz PCM In)", ha="center", va="center", fontsize=9)

    # Local RAM Buffer (Ring Buffer)
    ram_box = patches.FancyBboxPatch((0.8, 2.3), 2.2, 0.7, boxstyle="round,pad=0.1",
                                     facecolor="#FEFCBF", edgecolor="#D69E2E", linewidth=1.2)
    ax.add_patch(ram_box)
    ax.text(1.9, 2.65, "Host RAM Ring Buffer\n(Raw Audio Locked)", ha="center", va="center",
            fontsize=9, fontweight="bold", color="#744210")

    # Log-Mel Extractor
    mel_box = patches.FancyBboxPatch((0.8, 1.2), 2.2, 0.7, boxstyle="round,pad=0.1",
                                     facecolor="#FFFFFF", edgecolor="#4A5568", linewidth=1.2)
    ax.add_patch(mel_box)
    ax.text(1.9, 1.55, "Log-Mel Spectrogram\n(80-channel Filterbank)", ha="center", va="center", fontsize=9)

    # Whisper Neural Engine Box (HTP NPU + Kryo CPU)
    npu_box = patches.FancyBboxPatch((3.4, 1.2), 2.3, 2.9, boxstyle="round,pad=0.1",
                                     facecolor="#EDF2F7", edgecolor="#4A5568", linewidth=1.5)
    ax.add_patch(npu_box)
    ax.text(4.55, 3.8, "Quantized Whisper Engine", ha="center", va="center", fontsize=9.5, fontweight="bold")

    enc_box = patches.FancyBboxPatch((3.55, 2.8), 2.0, 0.7, boxstyle="round,pad=0.08",
                                     facecolor="#C6F6D5", edgecolor="#276749", linewidth=1.2)
    ax.add_patch(enc_box)
    ax.text(4.55, 3.15, "Whisper Encoder (W8A16)\nHexagon HTP v75 NPU", ha="center", va="center", fontsize=8.5)

    dec_box = patches.FancyBboxPatch((3.55, 1.8), 2.0, 0.7, boxstyle="round,pad=0.08",
                                     facecolor="#E9D8FD", edgecolor="#6B46C1", linewidth=1.2)
    ax.add_patch(dec_box)
    ax.text(4.55, 2.15, "Whisper Decoder (INT8)\nSnapdragon Kryo CPU", ha="center", va="center", fontsize=8.5)

    # Arrows inside AP
    ax.annotate("", xy=(1.9, 3.0), xytext=(1.9, 3.4), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748"))
    ax.annotate("", xy=(1.9, 1.9), xytext=(1.9, 2.3), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748"))
    ax.annotate("", xy=(3.55, 3.15), xytext=(3.0, 1.55),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748", connectionstyle="arc3,rad=-0.2"))
    ax.annotate("", xy=(4.55, 2.5), xytext=(4.55, 2.8), arrowprops=dict(arrowstyle="->", lw=1.5, color="#2D3748"))

    # 2. FastRPC Boundary (Security Membrane)
    rpc_box = patches.FancyBboxPatch((6.4, 1.0), 1.2, 3.8, boxstyle="round,pad=0.15",
                                     facecolor="#FFF5F5", edgecolor="#E53E3E", linewidth=2, linestyle="--")
    ax.add_patch(rpc_box)
    ax.text(7.0, 4.3, "FastRPC Bridge", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#C53030")
    ax.text(7.0, 2.8, "AUDITED\nFASTRPC\nBOUNDARY\n\nNo raw-audio\nsignature\nobserved\n\nUTF-8 Text\nOnly (<100 B)",
            ha="center", va="center", fontsize=8, fontweight="bold", color="#9B2C2C")

    # Arrow from Decoder to FastRPC
    ax.annotate("", xy=(6.4, 2.15), xytext=(5.55, 2.15),
                arrowprops=dict(arrowstyle="->", lw=2, color="#E53E3E"))
    ax.text(5.95, 2.35, "Decoded\nText", ha="center", va="center", fontsize=8, color="#E53E3E", fontweight="bold")

    # 3. Arduino MCU Box
    mcu_box = patches.FancyBboxPatch((8.0, 1.0), 1.6, 3.8, boxstyle="round,pad=0.2",
                                     facecolor="#F0FFF4", edgecolor="#38A169", linewidth=2)
    ax.add_patch(mcu_box)
    ax.text(8.8, 4.4, "Arduino MCU", ha="center", va="center", fontsize=11, fontweight="bold", color="#276749")
    ax.text(8.8, 3.0, "Peripheral Controller\n\n• Actuator Triggers\n• Display Drivers\n• Sensor Interface\n• Offline Logic",
            ha="center", va="center", fontsize=8.5, color="#22543D")

    # Arrow from FastRPC to MCU
    ax.annotate("", xy=(8.0, 2.15), xytext=(7.6, 2.15),
                arrowprops=dict(arrowstyle="->", lw=2, color="#38A169"))

    # 4. Network Firewall Banner
    net_box = patches.FancyBboxPatch((0.4, 0.1), 9.2, 0.4, boxstyle="round,pad=0.08",
                                     facecolor="#2D3748", edgecolor="#1A202C", linewidth=1)
    ax.add_patch(net_box)
    ax.text(5.0, 0.3, "Audited session: no outbound IP packets observed | no raw-audio signature observed",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#FFFFFF")

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    fig.savefig(FIG_DIR / "fig1_system_architecture.png")
    plt.close(fig)
    print("Generated fig1_system_architecture.png (600 DPI)")


def generate_figure_2():
    """Figure 2: Multi-panel Latency-Footprint Trade-off & RTF Distribution."""
    b03_csv = BASE_DIR / "06_experiments" / "EXP-B03-profiling" / "edge_profiling_results.csv"
    df = pd.read_csv(b03_csv)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot (a): Footprint vs Generation Speed (TPOT) vs TTFT
    configs = ["fp32", "w8a16", "int8"]
    tier_labels = ["FP32 Base", "W8A16 QDQ", "INT8 Tier"]
    colors = ["#3182CE", "#DD6B20", "#38A169"]

    footprints = [220.19, 56.49, 55.93]
    tpots = [47.70, 20.83, 19.48]  # Median TPOT
    ttfts = [685.56, 824.99, 431.98]  # Median TTFT (bubble size)

    for i in range(len(configs)):
        size_scaled = ttfts[i] * 1.5
        ax1.scatter(footprints[i], tpots[i], s=size_scaled, color=colors[i], alpha=0.7,
                    edgecolor="black", linewidth=1.5, label=f"{tier_labels[i]} (TTFT: {ttfts[i]:.0f} ms)")
        ax1.annotate(f"{tier_labels[i]}\n({tpots[i]:.1f} ms/tok)",
                     (footprints[i], tpots[i]),
                     textcoords="offset points",
                     xytext=(15, -5 if i != 1 else 10),
                     fontsize=9, fontweight="bold", color=colors[i])

    ax1.set_xlabel("Model Footprint on Flash Disk (MB)")
    ax1.set_ylabel("Generation Latency TPOT (ms/token)")
    ax1.set_title("(a) Footprint vs. Generation Latency & TTFT", fontweight="bold")
    ax1.set_xlim(30, 260)
    ax1.set_ylim(10, 55)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=8.5)

    # Subplot (b): Real-Time Factor (RTF) Distributions
    sns.boxplot(x="tier_name", y="rtf", data=df, ax=ax2, palette=["#90CDF4", "#FBD38D", "#9AE6B4"],
                width=0.45, linewidth=1.2, fliersize=3)
    sns.stripplot(x="tier_name", y="rtf", data=df, ax=ax2, color="#2D3748", size=5, jitter=0.15, alpha=0.8)

    # Real-time threshold line
    ax2.axhline(1.0, color="#E53E3E", linestyle="--", linewidth=1.5, label="Real-Time Boundary (RTF = 1.0)")
    ax2.text(0.1, 0.92, "Real-Time Threshold (RTF = 1.0)", color="#E53E3E", fontweight="bold", fontsize=8.5, transform=ax2.transAxes)

    ax2.set_xlabel("Deployment Configuration Tier")
    ax2.set_ylabel("Real-Time Factor (RTF)")
    ax2.set_title("(b) Measured RTF Distribution (VIVOS Test)", fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig2_tradeoff_latency_footprint_accuracy.png")
    plt.close(fig)
    print("Generated fig2_tradeoff_latency_footprint_accuracy.png (600 DPI)")


def generate_figure_3():
    """Figure 3: Noise Robustness across SNR levels (Clean, 20dB, 10dB, 5dB)."""
    snr_labels = ["Clean", "20 dB", "10 dB", "5 dB"]
    x = np.arange(len(snr_labels))

    # Data from Table B-05
    clean_ft_wer = [26.99, 28.97, 33.58, 37.40]
    clean_ft_cer = [11.95, 12.96, 15.95, 18.61]

    noise_aware_wer = [27.14, 28.36, 31.34, 35.24]
    noise_aware_cer = [12.15, 12.81, 14.94, 17.33]

    pretrained_wer = [59.10, np.nan, np.nan, np.nan]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # Subplot (a): Word Error Rate (WER %)
    ax1.plot(x, clean_ft_wer, "o-", color="#E53E3E", linewidth=2, markersize=7, label="Clean Fine-Tuned (A30)")
    ax1.plot(x, noise_aware_wer, "s-", color="#2B6CB0", linewidth=2.2, markersize=7, label="Noise-Aware Fine-Tuned (A30/CPU)")
    ax1.scatter([0], [pretrained_wer[0]], color="#718096", s=70, marker="^", label="Pretrained Whisper-Tiny Base (59.1%)", zorder=5)

    # Shading improvement region
    ax1.fill_between(x[1:], clean_ft_wer[1:], noise_aware_wer[1:], color="#63B3ED", alpha=0.35, label="Noise-Aware Robustness Margin")

    # Annotate improvement at 10dB and 5dB
    ax1.annotate(r"$\mathbf{-2.24\%}$ abs (+6.7% rel)", (2, 31.34), textcoords="offset points", xytext=(-25, -20),
                 fontsize=8.5, color="#2B6CB0", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1))
    ax1.annotate(r"$\mathbf{-2.16\%}$ abs (+5.8% rel)", (3, 35.24), textcoords="offset points", xytext=(-35, -20),
                 fontsize=8.5, color="#2B6CB0", fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color="#2B6CB0", lw=1))

    ax1.set_xticks(x)
    ax1.set_xticklabels(snr_labels)
    ax1.set_xlabel("Acoustic Environment / SNR Condition")
    ax1.set_ylabel("Word Error Rate (WER %)")
    ax1.set_title("(a) WER vs. Noise Degradation (MUSAN)", fontweight="bold")
    ax1.set_ylim(22, 62)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", frameon=True, fontsize=8)

    # Subplot (b): Character Error Rate (CER %)
    ax2.plot(x, clean_ft_cer, "o-", color="#E53E3E", linewidth=2, markersize=7, label="Clean Fine-Tuned")
    ax2.plot(x, noise_aware_cer, "s-", color="#2B6CB0", linewidth=2.2, markersize=7, label="Noise-Aware Fine-Tuned")
    ax2.fill_between(x[1:], clean_ft_cer[1:], noise_aware_cer[1:], color="#63B3ED", alpha=0.35)

    ax2.set_xticks(x)
    ax2.set_xticklabels(snr_labels)
    ax2.set_xlabel("Acoustic Environment / SNR Condition")
    ax2.set_ylabel("Character Error Rate (CER %)")
    ax2.set_title("(b) CER vs. Noise Degradation (MUSAN)", fontweight="bold")
    ax2.set_ylim(10, 22)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper left", frameon=True, fontsize=8.5)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "fig3_noise_robustness.png")
    plt.close(fig)
    print("Generated fig3_noise_robustness.png (600 DPI)")


def main():
    print("=" * 60)
    print("TASK B-06: GENERATING 600 DPI PUBLICATION FIGURES")
    print("=" * 60)
    generate_figure_1()
    generate_figure_2()
    generate_figure_3()
    print("\n[ALL 3 FIGURES GENERATED SUCCESSFULLY AT 600 DPI]")


if __name__ == "__main__":
    main()
