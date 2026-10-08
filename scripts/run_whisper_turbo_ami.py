#!/usr/bin/env python3
"""
Benchmark Whisper-Turbo (openai/whisper-large-v3-turbo) trên bộ dữ liệu
AMI Meeting Corpus (16 cuộc họp test chuẩn BUT).

Tính năng:
- Tự động nạp mô hình openai/whisper-large-v3-turbo (FP16 trên GPU CUDA).
- Xử lý long-form audio (15 - 45 phút) bằng sliding window chunking (30s chunks).
- Đo đạc chi tiết: Thời gian suy luận, RTF, WER, CER so với Ground Truth XML.
- Xuất kết quả CSV và Báo cáo Markdown chuẩn NCKH.
"""

import os
import sys
import time
import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor
import jiwer

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MANIFEST_PATH = ROOT_DIR / "data" / "ami_test" / "manifest_ami_test.csv"
OUTPUT_DIR = ROOT_DIR / "benchmark_results_ami"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def normalize_english_text(text: str) -> str:
    """Chuẩn hóa văn bản tiếng Anh phục vụ tính WER chuẩn."""
    if not text:
        return ""
    import re
    text = text.lower()
    # Loại bỏ các ký tự đặc biệt, dấu câu, giữ lại chữ cái và số
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_metrics(hypothesis: str, reference: str):
    hyp_norm = normalize_english_text(hypothesis)
    ref_norm = normalize_english_text(reference)

    if not ref_norm:
        return {"wer": 0.0, "cer": 0.0, "hyp_norm": hyp_norm, "ref_norm": ref_norm}

    wer = round(jiwer.wer(ref_norm, hyp_norm) * 100, 2)
    cer = round(jiwer.cer(ref_norm, hyp_norm) * 100, 2)
    return {
        "wer": wer,
        "cer": cer,
        "hyp_norm": hyp_norm,
        "ref_norm": ref_norm
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper-Turbo on AMI Meeting Corpus")
    parser.add_argument("--model-id", type=str, default="openai/whisper-large-v3-turbo", help="HuggingFace model ID")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for chunked inference")
    parser.add_argument("--limit", type=int, default=None, help="Số lượng cuộc họp muốn test thử (None = all 16)")
    parser.add_argument("--device", type=str, default=None, help="Device cuda or cpu")
    args = parser.parse_args()

    print("======================================================================")
    print("   BENCHMARK WHISPER-TURBO TRÊN AMI MEETING CORPUS (TEST SPLIT)      ")
    print(f"   Model       : {args.model_id}")
    print(f"   Manifest    : {MANIFEST_PATH}")
    print(f"   Output Dir  : {OUTPUT_DIR}")
    print("======================================================================")

    if not MANIFEST_PATH.exists():
        print(f"\n[Lỗi] Chưa tìm thấy file manifest tại {MANIFEST_PATH}!")
        print("Vui lòng chạy lệnh sau để tải và chuẩn bị dữ liệu trước:")
        print("  python scripts/download_ami_data.py")
        sys.exit(1)

    df = pd.read_csv(MANIFEST_PATH)
    if args.limit:
        df = df.head(args.limit)
    total_meetings = len(df)
    print(f"Tìm thấy {total_meetings} cuộc họp cần đánh giá.\n")

    # Xác định device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Đang nạp mô hình {args.model_id} lên {device} ({dtype})...")
    pipe = pipeline(
        "automatic-speech-recognition",
        model=args.model_id,
        torch_dtype=dtype,
        device=device,
        chunk_length_s=30,      # Sliding window 30 giây chuẩn Whisper
        stride_length_s=4,       # Overlap 4 giây tránh mất từ ở mép chunk
        return_timestamps=False,
    )
    print("Mô hình đã sẵn sàng!\n")

    results = []
    total_audio_time = 0.0
    total_infer_time = 0.0

    for idx, row in df.iterrows():
        meeting_id = row["meeting_id"]
        audio_path = row["audio_path"]
        transcript_path = row["transcript_path"]
        duration_sec = float(row["duration_sec"])

        print(f"[{idx+1}/{total_meetings}] Đang giải mã {meeting_id} (Thời lượng: {duration_sec/60:.1f} phút)...")

        # Đọc Ground Truth
        with open(transcript_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        # Đo thời gian suy luận
        if device == "cuda":
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        prediction = pipe(
            audio_path,
            batch_size=args.batch_size,
            generate_kwargs={"language": "english", "task": "transcribe"}
        )

        if device == "cuda":
            torch.cuda.synchronize()
        latency_sec = time.perf_counter() - start_time

        pred_text = prediction["text"].strip()
        rtf = round(latency_sec / duration_sec, 4) if duration_sec > 0 else 0.0

        metrics = calculate_metrics(pred_text, ground_truth)

        total_audio_time += duration_sec
        total_infer_time += latency_sec

        print(f"    -> Xong trong {latency_sec:.2f}s | RTF: {rtf:.4f} | WER: {metrics['wer']}% | CER: {metrics['cer']}%")

        results.append({
            "meeting_id": meeting_id,
            "duration_sec": duration_sec,
            "duration_min": round(duration_sec / 60, 2),
            "latency_sec": round(latency_sec, 2),
            "rtf": rtf,
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "predicted_text_preview": pred_text[:120] + "..." if len(pred_text) > 120 else pred_text,
            "ground_truth_preview": ground_truth[:120] + "..." if len(ground_truth) > 120 else ground_truth,
        })

    # Xuất file CSV chi tiết
    res_df = pd.DataFrame(results)
    csv_out = OUTPUT_DIR / "ami_turbo_results.csv"
    res_df.to_csv(csv_out, index=False, encoding="utf-8")

    # Tính toán tổng hợp
    avg_wer = round(res_df["wer"].mean(), 2)
    avg_cer = round(res_df["cer"].mean(), 2)
    avg_rtf = round(res_df["rtf"].mean(), 4)
    overall_speedup = round(total_audio_time / total_infer_time, 2) if total_infer_time > 0 else 0.0

    summary_md = f"""# Báo cáo Đánh giá Whisper-Turbo trên AMI Meeting Corpus

- **Mô hình**: `{args.model_id}`
- **Tập kiểm thử**: AMI Meeting Corpus (16 meetings, BUT Test Split)
- **Thiết bị**: `{device}`
- **Tổng thời lượng âm thanh**: {total_audio_time/60:.2f} phút ({total_audio_time/3600:.2f} giờ)
- **Tổng thời gian xử lý**: {total_infer_time/60:.2f} phút ({total_infer_time:.2f} giây)

---

## 1. Kết quả Tổng thể

| Chỉ số | Giá trị |
| :--- | :---: |
| **WER Trung bình (%)** | **{avg_wer}%** |
| **CER Trung bình (%)** | **{avg_cer}%** |
| **RTF Trung bình (Real-Time Factor)** | **{avg_rtf}** |
| **Tốc độ xử lý (gấp thời gian thực)** | **{overall_speedup}x Faster** |

---

## 2. Chi tiết theo từng cuộc họp

| Meeting ID | Thời lượng (phút) | Thời gian suy luận (s) | RTF | WER (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for r in results:
        summary_md += f"| **{r['meeting_id']}** | {r['duration_min']} | {r['latency_sec']} | {r['rtf']} | {r['wer']}% | {r['cer']}% |\n"

    md_out = OUTPUT_DIR / "ami_turbo_summary.md"
    md_out.write_text(summary_md, encoding="utf-8")

    print("\n======================================================================")
    print("   KẾT QUẢ TỔNG HỢP BENCHMARK WHISPER-TURBO                          ")
    print(f"   WER Trung bình   : {avg_wer}%")
    print(f"   CER Trung bình   : {avg_cer}%")
    print(f"   RTF Trung bình   : {avg_rtf} ({overall_speedup}x Faster)")
    print(f"   File CSV chi tiết: {csv_out}")
    print(f"   Báo cáo Markdown : {md_out}")
    print("======================================================================")

if __name__ == "__main__":
    main()
