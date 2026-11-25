# Nghiên cứu Fine-tuning và Lượng tử hóa Mô hình Nhận dạng Tiếng nói Tiếng Việt cho Triển khai trên Thiết bị Edge AI

Kho mã nguồn và dữ liệu thực nghiệm của **Bài tập lớn học phần Xử lý ngôn ngữ tự nhiên (CSC4007)** — Khoa Công nghệ Thông tin, Trường Đại học Đại Nam.

## Thông tin chung

- **Học phần:** Xử lý ngôn ngữ tự nhiên (CSC4007)
- **Giảng viên hướng dẫn:** ThS. Lê Thị Thùy Trang
- **Nhóm sinh viên thực hiện:**

| STT | Mã sinh viên | Họ và tên | Ngày sinh | Lớp |
| :---: | :---: | :--- | :---: | :---: |
| 1 | 1771020189 | Lê Tuấn Dũng | 28/11/2005 | KHMT 17-01 |
| 2 | 1771040024 | Nguyễn Duy Thuận | 14/01/2005 | KHMT 17-01 |
| 3 | 1671040022 | Đinh Tấn Phát | 06/12/1999 | KHMT 16-01 |

---

## 1. Giới thiệu bài toán và mục tiêu

Hệ thống nhận dạng tiếng nói tự động (ASR) dựa trên điện toán đám mây gặp ba rào cản lớn khi ứng dụng thực tế: độ trễ đường truyền biến động, phụ thuộc kết nối Internet và nguy cơ rò rỉ dữ liệu âm thanh riêng tư. Đề tài xây dựng quy trình thực nghiệm end-to-end đưa mô hình ASR tiếng Việt xuống chạy hoàn toàn ngoại tuyến trên thiết bị biên **Arduino VENTUNO Q** (vi xử lý Qualcomm Snapdragon QCS8275):

1. **Chuẩn hóa dữ liệu & chống rò rỉ người nói:** Sử dụng tập **VIVOS** (12.420 câu thoại, 15,67 giờ, 65 người nói) phân hoạch speaker-disjoint tuyệt đối (41 train / 5 valid / 19 test; 0% trùng lặp) kết hợp phối trộn tạp âm **MUSAN** ở 4 mức SNR (`Clean`, `20 dB`, `10 dB`, `5 dB`; tổng 14.700 mẫu).
2. **Huấn luyện thích ứng nhiễu (Noise-aware fine-tuning):** Tinh chỉnh mô hình `Whisper-Tiny` (39M tham số) trên GPU NVIDIA A30, đối chứng giữa chiến lược Clean và Noise-aware.
3. **Xuất đồ thị ONNX & lượng tử hóa hỗn hợp W8A16/INT8:** Phân tích hiện tượng sụp đổ mô hình khi nén Pure INT8 (`WER = 99,08%` do bão hòa activation outlier ở Encoder) và đề xuất nén hỗn hợp `W8A16 QDQ` (Encoder) + `Dynamic INT8` (Decoder), giảm **74,34%** dung lượng (`220,19 MiB -> 56,49 MiB`).
4. **Đo đạc hiệu năng biên & kiểm định thống kê:** Benchmark trực tiếp trên bo mạch Arduino VENTUNO Q (RTF, TTFT, TPOT, RAM đỉnh VmHWM), đối chứng các cấu hình **E0 – E1 – E2**, kiểm định tương đương **TOST** + **10.000 Bootstrap** và phân tích 10 ca lỗi ngữ âm học tiếng Việt.

---

## 2. Kết quả thực nghiệm nổi bật

### 2.1. Độ bền vững trước nhiễu trên 3.040 lượt kiểm thử (760 câu × 4 mức SNR)

| Điều kiện kiểm thử | Clean Fine-tuning WER | Noise-aware WER | Chênh lệch ($\Delta$WER) |
| :--- | :---: | :---: | :---: |
| Clean | 26,99% | 27,14% | +0,15% |
| Nhiễu 20 dB | 28,97% | **28,36%** | **-0,61%** |
| Nhiễu 10 dB | 33,58% | **31,34%** | **-2,24%** |
| Nhiễu 5 dB | 37,40% | **35,24%** | **-2,16%** |
| **Trung bình vĩ mô** | **31,73%** | **30,52%** | **-1,21%** |

### 2.2. Đối chứng cấu hình E0 – E1 – E2 trên 760 câu test VIVOS (Clean)

| Cấu hình | Mô hình | Phần cứng | Định dạng | WER (%) | CER (%) | RTF trung vị |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| E0 | Whisper-Tiny gốc (Baseline) | Host CPU | FP32 | 59,10% | 31,47% | - |
| E0 | Whisper-Base gốc | Host CPU | FP32 | 45,60% | 22,86% | - |
| Ref | Whisper-Tiny fine-tuned | Server A30 | FP32 | 27,14% | 12,15% | 0,0919 |
| **E1** | **Whisper-Tiny đề xuất** | **Edge CPU** | **W8A16/INT8** | **28,10%** | **12,58%** | **0,3381** |
| - | Whisper-Tiny đề xuất | Edge QNN+CPU | W8A16/INT8 | 28,09% | 12,59% | 0,4555 |
| **E2** | **PhoWhisper-Tiny đề xuất** | **Edge CPU** | **W8A16/INT8** | **19,63%** | **7,08%** | **0,3488** |

### 2.3. Hiệu năng phần cứng trên Arduino VENTUNO Q & kiểm định thống kê

- **Dung lượng mô hình:** Giảm từ `220,19 MiB` (FP32) xuống `56,49 MiB` (W8A16/INT8) — tiết kiệm **74,34%**.
- **Tốc độ suy luận (Kryo CPU 4 threads):** RTF trung vị `0,3381` (nhanh gấp ~3 lần thời gian thực), TTFT `719,30 ms`, TPOT `20,28 ms/token` (nhanh gấp 2,3 lần bản FP32), Peak RAM `529,70 MiB`, `0/760` lỗi thực thi.
- **Kiểm định tương đương TOST & Bootstrap (biên $\delta = \pm 2,0\%$ WER):** Độ lệch cặp trung bình $+0,8144\%$, 95% Paired Bootstrap CI `[+0,1710, +1,4521]`, 95% Speaker-Cluster Bootstrap CI `[+0,3919, +1,2079]`, $p_L < 0,0001$, $p_U = 0,000173$ -> **Đạt chuẩn tương đương thống kê**.

---

## 3. Cấu trúc thư mục dự án

```text
.
|-- 03_data/
|   |-- DATASET_MANIFEST.csv            # Nguồn và bản quyền dữ liệu VIVOS & MUSAN
|   |-- DATASET_SOURCES.md              # Mô tả chi tiết nguồn ngữ liệu
|   |-- preprocess_config.yaml          # Cấu hình tiền xử lý và phối trộn nhiễu
|   `-- processed/
|       |-- speaker_overlap_report.json # Báo cáo kiểm toán 0% rò rỉ người nói
|       |-- vivos_speaker_split.csv     # Bảng phân hoạch train / valid / test
|       `-- vivos_noise_mixed_manifest.csv # Danh sách ma trận 14.700 mẫu nhiễu
|-- 04_code/
|   |-- preprocess/                     # Tiền xử lý, chuẩn hóa NFC, chia tập & trộn nhiễu
|   |-- baseline/                       # Chạy baseline E0 và tổng hợp kết quả
|   |-- train/                          # Fine-tuning Whisper-Tiny (Clean & Noise-aware)
|   |-- export/                         # Xuất đồ thị ONNX (Encoder/Decoder) & kiểm tra parity
|   |-- quantization/                   # Hiệu chuẩn tĩnh & nén W8A16 QDQ / Dynamic INT8
|   |-- compile/                        # Biên dịch và kiểm thử trên QNN HTP NPU
|   |-- profiling/                      # Đo RTF/TTFT/TPOT/RAM và chẩn đoán sụp đổ INT8
|   `-- figures/                        # Mã nguồn vẽ biểu đồ thực nghiệm
|-- 06_experiments/
|   |-- EXP-A08-clean/                  # Log huấn luyện và đánh giá Clean fine-tuning
|   |-- EXP-A08-noise-aware/            # Log huấn luyện Noise-aware & metadata ONNX
|   |-- EXP-B08-quantization-eval/      # So sánh FP32 vs Pure INT8 vs W8A16/INT8
|   |-- EXP-B09-edge-benchmarks/        # Kết quả benchmark trên bo mạch Arduino VENTUNO Q
|   |-- EXP-R01-non-inferiority/        # Kiểm định TOST, Bootstrap & dự đoán 760 câu E1
|   `-- EXP-R02-baselines/              # Kết quả đối chứng các mô hình baseline E0
|-- 07_figures/                         # Hình kiến trúc, trade-off và độ bền vững nhiễu
`-- 10_evidence/
    `-- public_results_20261029/        # Kết quả dự đoán 760 câu của PhoWhisper-Tiny (E2)
```

> **Lưu ý về dữ liệu và trọng số lớn:** Thư mục `03_data/raw/` (âm thanh WAV gốc VIVOS/MUSAN > 3 GB) và `05_models/` (tệp trọng số ONNX ~220 MB) được loại trừ qua `.gitignore` để kho mã nguồn gọn nhẹ. Xem `03_data/DATASET_SOURCES.md` để tải dữ liệu gốc nếu cần huấn luyện lại từ đầu.

---

## 4. Hướng dẫn cài đặt và tái lập nhanh

### 4.1. Cài đặt môi trường Python

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install torch transformers datasets evaluate jiwer librosa soundfile onnx onnxruntime pandas numpy scipy matplotlib
```

### 4.2. Kiểm tra lại nhanh WER / CER từ tệp dự đoán 760 câu (không cần GPU)

```bash
python -c "
import pandas as pd
from evaluate import load

df = pd.read_csv('06_experiments/EXP-R01-non-inferiority/w8a16_760_predictions.csv')
wer = load('wer').compute(predictions=df['hypothesis'].tolist(), references=df['reference'].tolist())
cer = load('cer').compute(predictions=df['hypothesis'].tolist(), references=df['reference'].tolist())
print(f'E1 Whisper-Tiny W8A16/INT8 -> WER: {wer*100:.2f}%, CER: {cer*100:.2f}%')
"
```

### 4.3. Quy trình tái lập đầy đủ End-to-End

1. **Phân hoạch dữ liệu & trộn nhiễu:**
   ```bash
   python 04_code/preprocess/make_speaker_split.py
   python 04_code/preprocess/make_noise_manifest.py
   ```
2. **Huấn luyện fine-tuning (GPU):**
   ```bash
   python 04_code/train/finetune_whisper.py --mode noise_aware
   ```
3. **Xuất ONNX & lượng tử hóa W8A16/INT8:**
   ```bash
   python 04_code/export/export_whisper_onnx.py
   python 04_code/quantization/prepare_calibration_data.py
   python 04_code/quantization/build_quantized_models.py
   ```
4. **Đánh giá hiệu năng & kiểm chứng:**
   ```bash
   python 04_code/profiling/profile_edge.py
   python 04_code/profiling/compute_edge_accuracy.py
   ```

---

## 5. Bản quyền dữ liệu và đạo đức nghiên cứu

- Tập ngữ liệu **VIVOS** sử dụng theo giấy phép phi thương mại **CC BY-NC-SA 4.0**.
- Tập tạp âm **MUSAN** sử dụng theo giấy phép **CC BY 4.0**.
- Toàn bộ kết quả báo cáo đều có nhật ký định lượng kiểm chứng và đảm bảo không rò rỉ dữ liệu giữa tập huấn luyện và tập kiểm thử.
