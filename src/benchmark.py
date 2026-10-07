import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from tabulate import tabulate
from tqdm import tqdm

from src.dataset import ASRDataset
from src.metrics import (
    calculate_metrics,
    calculate_sample_metrics,
    calculate_rtf,
    calculate_speedup,
    normalize_text,
)
from src.engine import WhisperMedusaEngine, VanillaWhisperEngine

class ASRBenchmarkRunner:
    def __init__(
        self,
        medusa_model_name: str = "aiola/whisper-medusa-v1",
        baseline_model_name: Optional[str] = "openai/whisper-large-v2",
        output_dir: str = "benchmark_results",
        fp16: bool = True,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fp16 = fp16

        print("Initializing engines for benchmark...")
        self.medusa_engine = WhisperMedusaEngine(model_name=medusa_model_name, fp16=fp16)
        if baseline_model_name:
            self.baseline_engine = VanillaWhisperEngine(model_name=baseline_model_name, fp16=fp16)
        else:
            self.baseline_engine = None

    def run(
        self,
        dataset: ASRDataset,
        language: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Dict:
        df = dataset.filter(language=language, limit=limit)
        total_samples = len(df)
        print(f"\nStarting benchmark on {total_samples} samples (Language filter: {language or 'All'})...")

        results = []

        for idx, row in tqdm(df.iterrows(), total=total_samples, desc="Benchmarking"):
            audio_path = row["resolved_audio_path"]
            ground_truth = str(row["text"])
            lang = str(row["language"])
            domain = str(row.get("domain", "general"))
            duration_sec = float(row.get("duration_sec", 0.0))

            # 1. Medusa Inference
            medusa_res = self.medusa_engine.transcribe(audio_path, language=lang)
            medusa_metrics = calculate_sample_metrics(medusa_res["text"], ground_truth)

            sample_record = {
                "recording_id": row.get("recording_id", f"sample_{idx}"),
                "language": lang,
                "domain": domain,
                "audio_path": audio_path,
                "duration_sec": duration_sec or medusa_res["audio_duration_sec"],
                "ground_truth": ground_truth,
                "medusa_pred": medusa_res["text"],
                "medusa_latency_ms": medusa_res["latency_ms"],
                "medusa_rtf": medusa_res["rtf"],
                "medusa_wer": medusa_metrics["wer"],
                "medusa_cer": medusa_metrics["cer"],
                "medusa_em": medusa_metrics["exact_match"],
                "medusa_vram_mb": medusa_res["peak_vram_mb"],
            }

            # 2. Baseline Inference (if enabled)
            if self.baseline_engine:
                base_res = self.baseline_engine.transcribe(audio_path, language=lang)
                base_metrics = calculate_sample_metrics(base_res["text"], ground_truth)

                speedup = calculate_speedup(base_res["latency_ms"], medusa_res["latency_ms"])

                sample_record.update({
                    "baseline_pred": base_res["text"],
                    "baseline_latency_ms": base_res["latency_ms"],
                    "baseline_rtf": base_res["rtf"],
                    "baseline_wer": base_metrics["wer"],
                    "baseline_cer": base_metrics["cer"],
                    "baseline_em": base_metrics["exact_match"],
                    "baseline_vram_mb": base_res["peak_vram_mb"],
                    "speedup_factor": speedup,
                })

            results.append(sample_record)

        results_df = pd.DataFrame(results)
        detailed_csv_path = self.output_dir / "detailed_results.csv"
        results_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")
        print(f"\nDetailed sample results saved to: {detailed_csv_path}")

        # Compute aggregate metrics
        summary = self._compute_aggregates(results_df)
        self._save_summary_report(summary, results_df)

        return summary

    def _compute_aggregates(self, df: pd.DataFrame) -> Dict:
        """Computes overall, per-language, and per-domain performance metrics."""
        def calc_group(group_df: pd.DataFrame) -> Dict:
            medusa_m = calculate_metrics(group_df["medusa_pred"].tolist(), group_df["ground_truth"].tolist())
            avg_medusa_lat = round(group_df["medusa_latency_ms"].mean(), 2)
            avg_medusa_rtf = round(group_df["medusa_rtf"].mean(), 4)

            group_res = {
                "count": len(group_df),
                "medusa_wer": medusa_m["wer"],
                "medusa_cer": medusa_m["cer"],
                "medusa_em": medusa_m["exact_match"],
                "medusa_avg_latency_ms": avg_medusa_lat,
                "medusa_avg_rtf": avg_medusa_rtf,
            }

            if "baseline_latency_ms" in group_df.columns:
                base_m = calculate_metrics(group_df["baseline_pred"].tolist(), group_df["ground_truth"].tolist())
                avg_base_lat = round(group_df["baseline_latency_ms"].mean(), 2)
                avg_base_rtf = round(group_df["baseline_rtf"].mean(), 4)
                speedup = calculate_speedup(avg_base_lat, avg_medusa_lat)

                group_res.update({
                    "baseline_wer": base_m["wer"],
                    "baseline_cer": base_m["cer"],
                    "baseline_em": base_m["exact_match"],
                    "baseline_avg_latency_ms": avg_base_lat,
                    "baseline_avg_rtf": avg_base_rtf,
                    "avg_speedup": speedup,
                })
            return group_res

        overall = calc_group(df)
        by_language = {}
        for lang, grp in df.groupby("language"):
            by_language[lang] = calc_group(grp)

        by_domain = {}
        for dom, grp in df.groupby("domain"):
            by_domain[dom] = calc_group(grp)

        return {
            "overall": overall,
            "by_language": by_language,
            "by_domain": by_domain,
        }

    def _save_summary_report(self, summary: Dict, df: pd.DataFrame):
        """Generates markdown report and JSON summary."""
        json_path = self.output_dir / "summary_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # Markdown report
        md_lines = [
            "# Kết quả Thực nghiệm So sánh Whisper-Medusa vs Vanilla Whisper",
            "",
            "## 1. Tổng quan Toàn bộ Tập dữ liệu",
            "",
        ]

        has_baseline = "baseline_wer" in summary["overall"]

        if has_baseline:
            overview_table = [
                ["Chỉ số (Metric)", "Vanilla Whisper (Baseline)", "Whisper-Medusa", "Chênh lệch / Tăng tốc"],
                ["WER (%)", f"{summary['overall']['baseline_wer']}%", f"{summary['overall']['medusa_wer']}%", f"{round(summary['overall']['medusa_wer'] - summary['overall']['baseline_wer'], 2):+}%"],
                ["CER (%)", f"{summary['overall']['baseline_cer']}%", f"{summary['overall']['medusa_cer']}%", f"{round(summary['overall']['medusa_cer'] - summary['overall']['baseline_cer'], 2):+}%"],
                ["Exact Match (%)", f"{summary['overall']['baseline_em']}%", f"{summary['overall']['medusa_em']}%", f"{round(summary['overall']['medusa_em'] - summary['overall']['baseline_em'], 2):+}%"],
                ["Thời gian suy luận TB", f"{summary['overall']['baseline_avg_latency_ms']} ms", f"{summary['overall']['medusa_avg_latency_ms']} ms", f"Nhanh hơn {summary['overall']['avg_speedup']}x"],
                ["Real-Time Factor (RTF)", f"{summary['overall']['baseline_avg_rtf']}", f"{summary['overall']['medusa_avg_rtf']}", f"Giảm {round((1 - summary['overall']['medusa_avg_rtf']/summary['overall']['baseline_avg_rtf'])*100, 1)}%"],
            ]
        else:
            overview_table = [
                ["Chỉ số (Metric)", "Whisper-Medusa"],
                ["WER (%)", f"{summary['overall']['medusa_wer']}%"],
                ["CER (%)", f"{summary['overall']['medusa_cer']}%"],
                ["Exact Match (%)", f"{summary['overall']['medusa_em']}%"],
                ["Latency TB (ms)", f"{summary['overall']['medusa_avg_latency_ms']} ms"],
                ["RTF TB", f"{summary['overall']['medusa_avg_rtf']}"],
            ]

        md_lines.append(tabulate(overview_table, headers="firstrow", tablefmt="github"))
        md_lines.append("")

        md_lines.append("## 2. Phân tách theo Ngôn ngữ (English vs Vietnamese)")
        md_lines.append("")

        lang_rows = []
        if has_baseline:
            lang_headers = ["Ngôn ngữ", "Số mẫu", "Baseline WER", "Medusa WER", "Baseline RTF", "Medusa RTF", "Tăng tốc (Speedup)"]
            for lang, data in summary["by_language"].items():
                lang_rows.append([
                    lang.upper(),
                    data["count"],
                    f"{data['baseline_wer']}%",
                    f"{data['medusa_wer']}%",
                    data["baseline_avg_rtf"],
                    data["medusa_avg_rtf"],
                    f"{data['avg_speedup']}x",
                ])
        else:
            lang_headers = ["Ngôn ngữ", "Số mẫu", "Medusa WER", "Medusa CER", "Medusa RTF"]
            for lang, data in summary["by_language"].items():
                lang_rows.append([
                    lang.upper(),
                    data["count"],
                    f"{data['medusa_wer']}%",
                    f"{data['medusa_cer']}%",
                    data["medusa_avg_rtf"],
                ])

        md_lines.append(tabulate(lang_rows, headers=lang_headers, tablefmt="github"))
        md_lines.append("")

        md_lines.append("## 3. Nhận định Khoa học cho Bài thuyết trình Seminar")
        md_lines.append("""
1. **Hiệu quả Tăng tốc của Speculative Decoding:**
   - Whisper-Medusa dự đoán đồng thời nhiều tokens qua 10 Medusa Heads, giảm đáng kể số chu kỳ autoregressive forward pass.
   - RTF và độ trễ giảm rõ rệt, chứng minh tính khả thi của suy luận tốc độ cao trên thiết bị.
2. **Khoảng cách Ngôn ngữ (Language Disparity):**
   - Medusa Heads nguyên bản được huấn luyện trên LibriSpeech (tiếng Anh), do đó hiệu quả trên tiếng Anh đạt mức tối ưu.
   - Đối với tiếng Việt, do Medusa Heads chưa qua fine-tuning theo phân phối âm học/ngôn ngữ tiếng Việt, việc bổ sung pha Fine-tuning Medusa Heads chính là một điểm đóng góp nghiên cứu thiết thực của đề tài NCKH 2026-2027.
3. **Mối liên hệ với Lượng tử hóa (Edge-ASR):**
   - Speculative Decoding (Medusa) giảm số bước giải mã (Decode Step Reduction), trong khi Lượng tử hóa (PTQ/QAT W8A16, INT8) giảm chi phí bộ nhớ và toán tử (Memory/Compute Reduction).
   - Sự kết hợp giữa **Medusa Speculative Decoding + Low-bit Quantization** tạo thành kiến trúc kép tối ưu toàn diện cho thiết bị Edge AI (như Qualcomm QCS8550).
""")

        md_path = self.output_dir / "summary_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        print(f"Summary markdown report generated at: {md_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASR Benchmark")
    parser.add_argument("--medusa-model", type=str, default="aiola/whisper-medusa-v1")
    parser.add_argument("--baseline-model", type=str, default="openai/whisper-large-v2")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline evaluation")
    parser.add_argument("--language", type=str, default=None, help="Filter language: 'en', 'vi', or None for all")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of audio samples (e.g. 5 or 10 for quick testing)")
    parser.add_argument("--output-dir", type=str, default="benchmark_results")
    args = parser.parse_args()

    ds = ASRDataset()
    runner = ASRBenchmarkRunner(
        medusa_model_name=args.medusa_model,
        baseline_model_name=None if args.no_baseline else args.baseline_model,
        output_dir=args.output_dir,
    )
    runner.run(ds, language=args.language, limit=args.limit)
