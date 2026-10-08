# Báo cáo Đánh giá Whisper-Turbo trên AMI Meeting Corpus (Đã Tinh Chỉnh)

- **Mô hình**: `openai/whisper-large-v3-turbo`
- **Tập kiểm thử**: AMI Meeting Corpus (16 meetings, BUT Test Split)
- **Thiết bị**: `cuda`
- **Tổng thời lượng âm thanh**: 543.73 phút (9.06 giờ)
- **Tổng thời gian xử lý**: 3.63 phút (217.67 giây)

---

## 1. Kết quả Tổng thể

| Chỉ số | Giá trị | Ghi chú |
| :--- | :---: | :--- |
| **WER Chuẩn Khoa học (%)** | **26.23%** | Đã lọc bỏ từ đệm ngập ngừng (*uh, um, hmm*) & chuẩn hóa viết tắt |
| **WER Thô (%)** | **30.17%** | Giữ nguyên từ đệm ngập ngừng trong file ghi chú |
| **CER Trung bình (%)** | **21.25%** | Tỷ lệ lỗi cấp ký tự (Character Error Rate) |
| **RTF Trung bình (Real-Time Factor)** | **0.0067** | Nhanh hơn thời gian thực **149.88 lần** |

---

## 2. Chi tiết theo từng cuộc họp

| Meeting ID | Thời lượng (phút) | Thời gian suy luận (s) | RTF | WER Chuẩn (%) | WER Thô (%) | CER (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **IS1009a** | 13.98 | 6.33 | 0.0076 | **24.28%** | 28.58% | 19.41% |
| **IS1009b** | 34.21 | 15.57 | 0.0076 | **20.99%** | 26.05% | 16.27% |
| **IS1009c** | 30.35 | 11.38 | 0.0063 | **18.38%** | 25.51% | 13.07% |
| **IS1009d** | 32.41 | 12.21 | 0.0063 | **19.03%** | 25.24% | 14.87% |
| **ES2004a** | 17.49 | 7.24 | 0.0069 | **25.42%** | 28.92% | 21.6% |
| **ES2004b** | 39.09 | 15.87 | 0.0068 | **23.69%** | 27.11% | 18.88% |
| **ES2004c** | 38.91 | 16.53 | 0.0071 | **21.77%** | 24.47% | 16.69% |
| **ES2004d** | 37.04 | 14.89 | 0.0067 | **31.59%** | 33.53% | 26.57% |
| **TS3003a** | 25.09 | 8.03 | 0.0053 | **20.33%** | 24.57% | 15.14% |
| **TS3003b** | 36.84 | 12.61 | 0.0057 | **19.55%** | 25.43% | 14.78% |
| **TS3003c** | 42.83 | 13.34 | 0.0052 | **19.04%** | 27.25% | 14.0% |
| **TS3003d** | 43.64 | 15.13 | 0.0058 | **24.49%** | 28.74% | 20.55% |
| **EN2002a** | 35.71 | 15.23 | 0.0071 | **41.04%** | 42.42% | 35.9% |
| **EN2002b** | 29.78 | 12.94 | 0.0072 | **37.36%** | 39.03% | 31.55% |
| **EN2002c** | 49.54 | 23.09 | 0.0078 | **34.54%** | 36.09% | 28.2% |
| **EN2002d** | 36.83 | 17.29 | 0.0078 | **38.22%** | 39.79% | 32.48% |

---

## 3. Nhận định Kỹ thuật

1. **Hiệu năng Xử lý Siêu tốc:**
   - Hệ số RTF đạt mức **0.0067**, tương đương tốc độ xử lý nhanh gấp **~149.88x** thời gian thực trên GPU.
   - Toàn bộ hơn 9 giờ âm thanh hội thảo được xử lý trọn vẹn trong khoảng **3 phút**.

2. **Đặc thù Âm học và Độ chính xác (WER):**
   - Bộ dữ liệu AMI là tập hội thoại đa người nói tự nhiên (spontaneous speech) trên kênh Mix-Headset không phân tách người nói (unsegmented speech), có mật độ nói chồng âm (overlapping speech) cao.
   - Nhóm cuộc họp kịch bản chuẩn (như `TS3003`, `IS1009`, `ES2004`) đạt độ chính xác cao.
   - Nhóm cuộc họp `EN2002` có WER cao hơn do các diễn giả nói tiếng Anh ngữ điệu không bản xứ (Non-native European Accents).
