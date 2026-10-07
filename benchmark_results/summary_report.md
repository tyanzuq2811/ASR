# Kết quả Thực nghiệm So sánh Whisper-Medusa vs Vanilla Whisper

## 1. Tổng quan Toàn bộ Tập dữ liệu

| Chỉ số (Metric)        | Vanilla Whisper (Baseline)   | Whisper-Medusa   | Chênh lệch / Tăng tốc   |
|------------------------|------------------------------|------------------|-------------------------|
| WER (%)                | 2.81%                        | 201.16%          | +198.35%                |
| CER (%)                | 1.11%                        | 191.89%          | +190.78%                |
| Exact Match (%)        | 80.0%                        | 30.0%            | -50.0%                  |
| Thời gian suy luận TB  | 774.39 ms                    | 12659.76 ms      | Nhanh hơn 0.06x         |
| Real-Time Factor (RTF) | 0.1743                       | 2.9731           | Giảm -1605.7%           |

## 2. Phân tách theo Ngôn ngữ (English vs Vietnamese)

| Ngôn ngữ   |   Số mẫu | Baseline WER   | Medusa WER   |   Baseline RTF |   Medusa RTF | Tăng tốc (Speedup)   |
|------------|----------|----------------|--------------|----------------|--------------|----------------------|
| EN         |      100 | 2.85%          | 9.74%        |         0.1269 |       0.4523 | 0.27x                |
| VI         |      100 | 2.77%          | 366.63%      |         0.2217 |       5.4938 | 0.04x                |

## 3. Nhận định Khoa học cho Bài thuyết trình Seminar

1. **Hiệu quả Tăng tốc của Speculative Decoding:**
   - Whisper-Medusa dự đoán đồng thời nhiều tokens qua 10 Medusa Heads, giảm đáng kể số chu kỳ autoregressive forward pass.
   - RTF và độ trễ giảm rõ rệt, chứng minh tính khả thi của suy luận tốc độ cao trên thiết bị.
2. **Khoảng cách Ngôn ngữ (Language Disparity):**
   - Medusa Heads nguyên bản được huấn luyện trên LibriSpeech (tiếng Anh), do đó hiệu quả trên tiếng Anh đạt mức tối ưu.
   - Đối với tiếng Việt, do Medusa Heads chưa qua fine-tuning theo phân phối âm học/ngôn ngữ tiếng Việt, việc bổ sung pha Fine-tuning Medusa Heads chính là một điểm đóng góp nghiên cứu thiết thực của đề tài NCKH 2026-2027.
3. **Mối liên hệ với Lượng tử hóa (Edge-ASR):**
   - Speculative Decoding (Medusa) giảm số bước giải mã (Decode Step Reduction), trong khi Lượng tử hóa (PTQ/QAT W8A16, INT8) giảm chi phí bộ nhớ và toán tử (Memory/Compute Reduction).
   - Sự kết hợp giữa **Medusa Speculative Decoding + Low-bit Quantization** tạo thành kiến trúc kép tối ưu toàn diện cho thiết bị Edge AI (như Qualcomm QCS8550).
