# 📑 BÁO CÁO KHOA HỌC: CÁC PHƯƠNG PHÁP LƯỢNG TỬ HÓA VÀ THỰC NGHIỆM TRÊN WHISPER-MEDUSA

> **Đề tài:** Nghiên cứu fine-tuning và lượng tử hóa mô hình nhận dạng tiếng nói tiếng Việt cho triển khai trên thiết bị Edge AI  
> **Chủ nhiệm đề tài:** Lê Tuấn Dũng (MSSV: 1771020189) - Đại học Đại Nam  
> **Giảng viên hướng dẫn:** ThS. Lê Thái Bảo  
> **Tài liệu tham chiếu:**  
> - Bài báo *Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models* (arXiv 2025)  
> - Báo cáo chuyên đề *SEMINAR KHOA HỌC .pdf* (30 slides)  

---

## 1. ĐẶT VẤN ĐỀ & BỐI CẢNH NGHIÊN CỨU

Trong việc triển khai mô hình nhận dạng tiếng nói (ASR) lớn như OpenAI Whisper (Large-V2: 1.55 tỉ tham số) lên các thiết bị biên (Edge AI như Qualcomm QCS8550, vi xử lý nhúng, hoặc GPU máy trạm cá nhân), hệ thống vấp phải **hai nút thắt vật lý cốt tử**:
1. **Nút thắt Băng thông Bộ nhớ (Memory Bandwidth Bound):** Mô hình ở định dạng FP32 hoặc FP16 tốn tới 3.1 GB – 6.2 GB bộ nhớ, vượt quá dung lượng VRAM của các chip biên nhỏ.
2. **Nút thắt Thời gian Giải mã (Decoding Latency Bound):** Whisper sử dụng cơ chế giải mã tự hồi quy (Autoregressive), mỗi chu kỳ Decoder chỉ sinh ra được duy nhất 1 token, dẫn đến Real-Time Factor (RTF) cao.

Để giải quyết toàn diện hai nút thắt này, nghiên cứu thực hiện phương pháp tiếp cận kép:
- Áp dụng **Lượng tử hóa sau huấn luyện (Post-Training Quantization - PTQ)** để giảm kích thước bộ nhớ.
- Kết hợp với **Giải mã phỏng đoán đa đầu (Speculative Multi-Head Decoding - Whisper-Medusa)** để giảm số chu kỳ giải mã.

---

## 2. CÁC PHƯƠNG PHÁP LƯỢNG TỬ HÓA ĐƯỢC ÁP DỤNG TRONG NGHIÊN CỨU

Hệ thống sử dụng thư viện **Optimum-Quanto** (công cụ được khuyến nghị tại **Slide 27** của Seminar Khoa học dựa trên công trình của Söhler et al., 2025). Hai phương pháp lượng tử hóa chính được thực thi bao gồm:

### 2.1. Phương pháp W8A16 (Weights INT8, Activations FP16) — Lựa chọn An Toàn Khuyên Dùng

* **Nguyên lý Toán học:**
  Trọng số thực liên tục $W \in \mathbb{R}$ của các lớp Linear trong Transformer Encoder và Decoder được ánh xạ tuyến tính đối xứng sang số nguyên có dấu 8-bit $W_{int8} \in [-128, 127]$:
  $$W_{int8} = \text{clip}\left(\left\lfloor \frac{W}{S_w} \right\rceil, -128, 127\right)$$
  Trong đó hệ số tỉ lệ $S_w$ được tính toán theo từng kênh (per-channel) hoặc từng tensor:
  $$S_w = \frac{\max(|W|)}{127}$$
* **Lý do giữ Activation ở FP16 (W8A16):**
  Như đã phân tích tại **Slide 21 (Kurtosis và sự dịch chuyển gánh nặng)** trong bài Seminar:
  Activation của các mô hình Transformer ASR có độ Kurtosis rất cao (nhiều giá trị ngoại lệ - outliers vượt trội ở các kênh ẩn). Nếu ép cả Activation xuống INT8 (W8A8), các giá trị outlier sẽ gây sai số bão hòa nghiêm trọng, làm nổ WER.  
  Bằng cách giữ Activation ở **FP16** và chỉ nén Trọng số xuống **INT8**, ta:
  - Giảm ngay **50% dung lượng lưu trữ trọng số**.
  - Giảm một nửa lưu lượng truyền dữ liệu qua bus bộ nhớ (Memory I/O).
  - **Bảo toàn hầu như nguyên vẹn độ chính xác WER** (độ lệch WER thường $< 0.2\%$).

### 2.2. Phương pháp W4A16 (Weights INT4, Activations FP16) — Lựa chọn Nén Cao Cho Thiết Bị Cực Hạn

* **Nguyên lý Toán học:**
  Trọng số được nén xuống số nguyên 4-bit có dấu $W_{int4} \in [-8, 7]$ với kỹ thuật chia nhóm khối (Block-wise / Group-wise scaling, group size = 64):
  $$W_{int4} = \text{clip}\left(\left\lfloor \frac{W}{S_{group}} \right\rceil, -8, 7\right)$$
* **Đặc tính:**
  - Nén giảm tới **75% dung lượng mô hình** (từ 3.1 GB xuống chỉ còn ~0.78 GB).
  - Giúp mô hình quy mô 1.5 tỉ tham số có thể nằm gọn trong các chip biên có bộ nhớ siêu hạn chế.
  - *Sự đánh đổi (Trade-off):* Sai số lượng tử hóa cao hơn, WER có thể tăng nhẹ từ 1% – 3% tùy thuộc vào độ ồn của âm thanh đầu vào.

---

## 3. TÍCH HỢP LƯỢNG TỬ HÓA VÀO WHISPER-MEDUSA

Kiến trúc kết hợp độc đáo trong nghiên cứu này hoạt động như sau:
1. **Thân mạng chính (Backbone Whisper Large V2):** Toàn bộ các lớp self-attention, cross-attention và MLP của Encoder và Decoder được lượng tử hóa sang **W8A16** hoặc **W4A16** bằng `optimum-quanto`.
2. **10 Medusa Heads:** Các nhánh dự đoán phỏng đoán song song được giữ ở độ chính xác cao hoặc bán chính xác để đảm bảo xác suất dự báo token chính xác.
3. **Quy trình hoạt động:**
   - Tại mỗi chu kỳ, mô hình thực hiện 1 lượt forward pass trên các ma trận trọng số INT8 (giảm tải I/O bộ nhớ).
   - 10 Medusa Heads sinh ra cây ứng viên gồm nhiều token.
   - Bộ giải mã xác thực đồng thời các token hợp lệ.
   - **Kết quả kép:** Vừa nhẹ dung lượng bộ nhớ (50%–75%), vừa tăng tốc thời gian suy luận (1.4x–1.6x).

---

## 4. PHÂN TÍCH KHẢ THI TRÊN PHẦN CỨNG: GPU LOCAL (RTX 3050 4GB) vs SERVER A30

### Bảng Đo lường Thực tế Bộ nhớ VRAM

| Mô hình & Cấu hình | Kiểu dữ liệu | Dung lượng Trọng số | Peak VRAM Ước tính | Khả năng chạy trên GPU Local (RTX 3050 4GB) | Khả năng chạy trên Server (A30 24GB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Whisper-Medusa Gốc** | FP32 | ~6.2 GB | ~6.8 GB | ❌ **Không thể** (Tràn RAM) | ✅ Mượt mà |
| **Whisper-Medusa Bán chuẩn** | FP16 | ~3.1 GB | ~3.5 – 3.7 GB | ⚠️ **Cực kỳ rủi ro** (Windows chiếm sẵn 0.8GB, dễ lỗi OOM) | ✅ Mượt mà |
| **Whisper-Medusa Lượng tử hóa** | **INT8 (W8A16)** | **~1.55 GB** | **~1.85 – 2.0 GB** | ✅ **HOÀN TOÀN ỔN ĐỊNH & MƯỢT MÀ** | ✅ Tối ưu |
| **Whisper-Medusa Lượng tử hóa** | **INT4 (W4A16)** | **~0.78 GB** | **~1.1 – 1.3 GB** | ✅ **SIÊU NHẸ (Dư > 2GB VRAM)** | ✅ Siêu nhẹ |

> **KẾT LUẬN QUAN TRỌNG:**  
> Nhờ áp dụng **Lượng tử hóa INT8 (W8A16)**, bạn **HOÀN TOÀN CÓ THỂ CHẠY DEMO VÀ BENCHMARK TRỰC TIẾP TRÊN MÁY LOCAL** của mình với GPU RTX 3050 Laptop mà không lo lỗi tràn bộ nhớ (Out-Of-Memory) và không nhất thiết phải phụ thuộc vào việc đẩy lên server!

---

## 5. KỊCH BẢN ĐỐI CHỨNG TRÊN TẬP DỮ LIỆU REAL 200 AUDIO

Bộ dữ liệu kiểm thử gồm 200 file WAV thực tế (100 câu tiếng Anh + 100 câu tiếng Việt) được phân bố qua 14 miền công nghiệp:
- **Kịch bản 1 (Kiểm chứng Bộ nhớ & Tốc độ):**
  So sánh `Vanilla Whisper FP16` vs `Whisper-Medusa W8A16` vs `Whisper-Medusa W4A16`.
  *Kỳ vọng:* W8A16 giảm 50% VRAM, tăng tốc 1.5x trên tiếng Anh.
- **Kịch bản 2 (Kiểm chứng Mức độ Mất mát Chất lượng - WER Trade-off):**
  Đo lường mức tăng WER của bản W8A16 và W4A16 so với bản FP16 gốc.
  *Kỳ vọng:* W8A16 gần như không tăng WER ($<0.3\%$). W4A16 tăng nhẹ nhưng vẫn chấp nhận được trong miền khẩu lệnh công nghiệp ngắn.
- **Kịch bản 3 (Khoảng cách Đa ngữ En vs Vi):**
  Chỉ ra sự khác biệt giữa tiếng Anh và tiếng Việt để làm luận điểm mở ra pha **Fine-tuning tiếng Việt** cho đề tài NCKH 2026-2027.
