# 🎙️ Whisper-Medusa ASR Benchmark & Edge Deployment Toolkit

> **Đề tài Nghiên cứu Khoa học Sinh viên (2026 - 2027):**  
> *"Nghiên cứu fine-tuning và lượng tử hóa mô hình nhận dạng tiếng nói tiếng Việt cho triển khai trên thiết bị Edge AI"*  
> **Chủ nhiệm đề tài:** Lê Tuấn Dũng (MSSV: 1771020189) - Khoa CNTT, Đại học Đại Nam  
> **Giảng viên hướng dẫn:** ThS. Lê Thái Bảo  
> **Repository:** [https://github.com/tyanzuq2811/ASR.git](https://github.com/tyanzuq2811/ASR.git)

---

## 📌 1. Giới thiệu Tổng quan (Overview)

Repository này chứa bộ công cụ hoàn chỉnh để thực nghiệm đối chứng, đo lường hiệu năng và demo trực quan giữa:
1. **Mô hình Cơ sở (Vanilla Whisper):** Giải mã tự hồi quy tiêu chuẩn (Autoregressive decoding).
2. **Mô hình Whisper-Medusa:** Tăng tốc suy luận bằng kỹ thuật **Speculative Decoding** với 10 Medusa Heads dự đoán đồng thời nhiều tokens trên mỗi chu kỳ tính toán.

Hệ thống được thiết kế để kiểm thử toàn diện trên tập dữ liệu thực tế **Real 200 Audio** (100 câu tiếng Anh + 100 câu tiếng Việt) thuộc các nghiệp vụ công nghiệp (an toàn lao động, kho bãi, logistics, cảm biến, máy móc, kiểm định chất lượng).

---

## 📂 2. Cấu trúc Thư mục (Directory Structure)

```text
ASR/
├── real200/                       # Tập dữ liệu kiểm thử thực tế
│   ├── manifest.csv               # Metadata 200 audio (mã, văn bản, thời lượng, miền ứng dụng)
│   └── audio/
│       ├── en/                    # 100 audio tiếng Anh công nghiệp
│       └── vi/                    # 100 audio tiếng Việt công nghiệp
├── src/                           # Mã nguồn lõi (Core Python Modules)
│   ├── dataset.py                 # Quản lý nạp, lọc và tiền xử lý dữ liệu
│   ├── metrics.py                 # Chuẩn hóa văn bản, tính WER, CER, RTF, Speedup
│   ├── engine.py                  # Engine suy luận Whisper-Medusa & Vanilla Whisper
│   ├── benchmark.py               # Module chạy benchmark hàng loạt & tổng hợp báo cáo
│   ├── visualize.py               # Xuất biểu đồ phân tích phục vụ bài thuyết trình
│   └── app.py                     # Giao diện Web Streamlit Demo tương tác trực tiếp
├── scripts/
│   ├── run_benchmark.py           # CLI script chạy đánh giá linh hoạt
│   ├── run_demo.bat               # Khởi chạy Web Demo trên máy Local (Windows)
│   └── run_server.sh              # Kịch bản triển khai & chạy benchmark trên Server (Linux A30)
├── whisper-medusa/                # Mã nguồn thư viện whisper-medusa (aiOla)
├── benchmark_results/             # Thư mục lưu kết quả thực nghiệm (CSV, Markdown, Biểu đồ)
├── requirements.txt               # Danh mục thư viện Python chuẩn (không xung đột)
├── environment.yml                # File cấu hình môi trường Conda tự động
└── README.md
```

---

## ⚡ 3. Hướng dẫn Cài đặt Môi trường (Installation)

### Cách A: Cài đặt bằng Conda (Khuyên dùng trên cả Local & Server)
```bash
# 1. Tạo môi trường mới dành riêng cho ASR (Python 3.11)
conda create -n asr_env python=3.11 -y
conda activate asr_env

# 2. Cài đặt PyTorch với CUDA (CUDA 12.4 hoặc tương thích với driver của bạn)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Cài đặt các thư viện cần thiết theo phiên bản chuẩn
pip install -r requirements.txt
```

### Cách B: Tạo tự động từ file `environment.yml`
```bash
conda env create -f environment.yml
conda activate asr_env
```

---

## 🚀 4. Hướng dẫn Chạy Thực nghiệm & Demo

### 4.1. Khởi chạy Giao diện Demo Trực quan (FastAPI + Tailwind CSS)
Giao diện Web chuẩn hiện đại (Tailwind CSS, hỗ trợ Chế độ Sáng/Tối, trực quan, không chữ rườm rà):
- **Cách 1:** Nhấp đúp trực tiếp vào file: `scripts\run_demo.bat`
- **Cách 2:** Mở terminal tại thư mục `ASR` và chạy:
  ```powershell
  conda activate asr_env
  python src/server.py
  # Hoặc: uvicorn src.server:app --port 8000
  ```
  Sau đó truy cập trình duyệt: **`http://127.0.0.1:8000`**

- **Tính năng nổi bật:**
  * Toggle Chế độ **Sáng / Tối (Dark & Light Mode)** tức thì.
  * Lọc nhanh 200 mẫu theo Tiếng Việt / Tiếng Anh / Miền nghiệp vụ.
  * Nghe phát trực tiếp âm thanh, hiển thị văn bản gốc.
  * Tùy chọn lượng tử hóa: **INT8 (W8A16)**, **INT4 (W4A16)**, hoặc **FP16 Gốc**.
  * Bảng so sánh đối chứng song song (Side-by-side): Độ trễ (ms), RTF, WER (%), CER (%), VRAM (MB), Huy hiệu Tăng tốc **1.5x Speedup** và Mức giảm bộ nhớ.

### 4.2. Chạy Benchmark Đánh giá Tự động (CLI)
- **Chạy thử nhanh 5 mẫu (Sanity Check):**
  ```bash
  python scripts/run_benchmark.py --limit 5
  ```
- **Chạy toàn bộ 200 mẫu trên Server (NVIDIA A30 24GB):**
  ```bash
  python scripts/run_benchmark.py
  ```
- **Chạy riêng cho tiếng Việt hoặc tiếng Anh:**
  ```bash
  python scripts/run_benchmark.py --language vi
  python scripts/run_benchmark.py --language en
  ```
- Kết quả sẽ tự động lưu vào thư mục `benchmark_results/`:
  * `detailed_results.csv`: Kết quả chi tiết từng file audio.
  * `summary_report.md`: Bảng thống kê tóm tắt định dạng Markdown chuẩn cho báo cáo.
  * `summary_metrics.json`: Dữ liệu số học thô.
  * Các biểu đồ so sánh: `chart_rtf_comparison.png`, `chart_domain_speedup.png`, `chart_wer_comparison.png`.

---

## 🖥️ 5. Quy trình Đẩy lên Git & Chạy trên Server A30

### Bước 1: Commit và Push từ Local lên GitHub
```bash
cd d:\NCKH\NCKH2026_2027\ASR\ASR
git add .
git commit -m "feat: complete whisper-medusa benchmark toolkit, demo app, and server scripts"
git push origin main
```

### Bước 2: Kéo về Server và Chạy Benchmark
Trên terminal Server (`dunglt@aiotlab-ws1:/hdd3/users/dunglt/asr`):
```bash
# 1. Clone repository về server
git clone https://github.com/tyanzuq2811/ASR.git .

# 2. Cấp quyền thực thi và chạy script tự động
chmod +x scripts/run_server.sh
./scripts/run_server.sh
```

---

## 🔬 6. Liên hệ Khoa học với Bài báo Edge-ASR và Đề tài NCKH

| Khía cạnh | Bài báo Edge-ASR (arXiv 2025) | Mô hình Whisper-Medusa | Đóng góp của Đề tài NCKH 2026-2027 |
| :--- | :--- | :--- | :--- |
| **Cơ chế Tối ưu** | Lượng tử hóa sau huấn luyện (PTQ: W8A16, INT8, INT4) | Giải mã phỏng đoán đa đầu (Speculative Multi-Head Decoding) | **Kết hợp cả hai:** Lượng tử hóa trọng số + Giải mã phỏng đoán |
| **Nút thắt Giải quyết** | Giảm dung lượng VRAM & băng thông bộ nhớ (Memory Bandwidth) | Giảm số bước giải mã tự hồi quy (Decode Latency & RTF) | Đạt RTF cực thấp trên phần cứng giới hạn (Qualcomm QCS8550 / Edge AI) |
| **Hạn chế Hiện tại** | Chỉ thử nghiệm trên tiếng Anh (Slide 26 của Seminar) | Medusa Heads chỉ train trên LibriSpeech tiếng Anh | **Thực nghiệm chứng minh khoảng cách đa ngữ** & thực hiện Fine-tuning tiếng Việt |

Kết quả thực nghiệm trên tập 200 audio khẳng định: Trong khi tiếng Anh đạt mức tăng tốc ~1.5x lý tưởng, việc xử lý tiếng Việt đòi hỏi phải tiếp tục fine-tuning các nhánh Medusa hoặc kết hợp lượng tử hóa để tối ưu hóa hiệu quả thực thi trên chip biên.
