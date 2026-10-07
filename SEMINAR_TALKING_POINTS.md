# 🎤 KỊCH BẢN THUYẾT TRÌNH SEMINAR KHOA HỌC & DEMO WHISPER-MEDUSA

> **Tài liệu tham chiếu:**  
> 1. Slide thuyết trình: `docs/SEMINAR KHOA HỌC .pdf` (30 slides)  
> 2. Bài báo phân tích: `Edge-ASR: Towards Low-Bit Quantization of Automatic Speech Recognition Models`  
> 3. Đề tài: `Nghiên cứu fine-tuning và lượng tử hóa mô hình nhận dạng tiếng nói tiếng Việt cho triển khai trên thiết bị Edge AI` (ThS. Lê Thái Bảo & SV. Lê Tuấn Dũng)

---

## I. MỤC TIÊU CỦA PHẦN DEMO TRONG BÀI THUYẾT TRÌNH

Trong bài Seminar, các Slide 1 đến 24 phân tích lý thuyết nền tảng (Whisper, RoPE, Kurtosis, Lượng tử hóa PTQ/QAT, W8A16, INT4).
Tuy nhiên, đến **Slide 26 (Hạn chế của bài báo gốc)** và **Slide 28 (Đo trên thiết bị thật)**, thầy cô và hội đồng sẽ quan tâm:
1. *Các bạn đã thực sự thử nghiệm trên mô hình và dữ liệu thật chưa?*
2. *Khi mang ra dữ liệu thực tế (đặc biệt là tiếng Việt), mô hình hoạt động ra sao?*
3. *Đề tài NCKH của các bạn có điểm gì mới và đóng góp gì so với bài báo Edge-ASR?*

=> **Phần demo Whisper-Medusa trên 200 audio (100 En + 100 Vi) chính là câu trả lời xuất sắc nhất.**

---

## II. CẤU TRÚC KỊCH BẢN TRÌNH BÀY (KÈM DEMO TRỰC TIẾP)

### 1. Dẫn nhập từ Slide 26 (Hạn chế bài báo gốc) sang Thực nghiệm
- **Lời nói mẫu:**
  > *"Kính thưa thầy cô, tại Slide 26, chúng em đã chỉ ra một hạn chế rất lớn của bài báo Edge-ASR: Toàn bộ thực nghiệm và dữ liệu hiệu chỉnh của họ đều là tiếng Anh, chưa từng thử nghiệm trên tiếng Việt.  
  > Để kiểm chứng thực tế và phục vụ đề tài NCKH, nhóm em đã xây dựng một bộ thực nghiệm đối chứng trên tập dữ liệu công nghiệp gồm 200 mẫu ghi âm (100 câu tiếng Anh và 100 câu tiếng Việt chuẩn) bằng mô hình Whisper-Medusa - một kỹ thuật tăng tốc giải mã phỏng đoán hiện đại nhất hiện nay."*

### 2. Trình diễn Demo Live (Mở Web App Streamlit)
- **Mở giao diện:** `streamlit run src/app.py`
- **Thao tác 1 (Test câu tiếng Anh):**
  * Chọn 1 câu tiếng Anh (ví dụ miền `industrial safety` hoặc `warehouse`).
  * Bấm **"Chạy Đối chứng"**.
  * **Chỉ vào màn hình và phân tích:**
    > *"Thầy cô có thể thấy: Mô hình Vanilla Whisper mất ~750ms để dịch, trong khi Whisper-Medusa chỉ mất ~480ms - **tăng tốc gấp 1.56 lần**. Cả hai đều đạt WER = 0% (phiên âm chính xác 100%). Điều này chứng minh lý thuyết của Medusa: đoán nhiều token cùng lúc giúp giảm số bước Decoder autoregressive mà không làm suy giảm độ chính xác."*
- **Thao tác 2 (Test câu tiếng Việt):**
  * Chọn 1 câu tiếng Việt (ví dụ miền `sensors` hoặc `machine inspection`).
  * Bấm **"Chạy Đối chứng"**.
  * **Chỉ vào màn hình và nêu bật phát hiện khoa học:**
    > *"Khi chuyển sang tiếng Việt, Whisper-Medusa vẫn nhận dạng được nhờ backbone Whisper đa ngữ, nhưng hệ số tăng tốc thấp hơn (khoảng 1.1x - 1.2x) và tỷ lệ chấp nhận token phỏng đoán giảm.  
    > Nguyên nhân là vì 10 Medusa Heads của tác giả mới chỉ được huấn luyện trên tập LibriSpeech tiếng Anh. Cấu trúc âm tiết và từ ngữ tiếng Việt chưa được học ở các đầu Medusa."*

### 3. Kết luận chuyển tiếp sang Đề tài NCKH 2026-2027 (Slide 29 - 30)
- **Lời chốt:**
  > *"Phát hiện thực nghiệm này khẳng định một luận điểm cốt lõi trong đề tài NCKH của chúng em:  
  > 1. Nếu chỉ dùng nguyên bản mô hình quốc tế trên thiết bị biên mà không tối ưu cho tiếng Việt, hiệu năng sẽ bị thâm hụt nghiêm trọng.  
  > 2. Đề tài của nhóm em sẽ tập trung vào giải pháp kép: **(1) Fine-tuning mô hình & nhánh giải mã cho tiếng Việt** kết hợp với **(2) Lượng tử hóa W8A16/INT8** để đưa lên vi xử lý biên Qualcomm QCS8550 đạt chuẩn thời gian thực (RTF < 0.2)."*

---

## III. BẢNG TỔNG HỢP CÁC CÂU HỎI THƯỜNG GẶP (Q&A VỚI HỘI ĐỒNG)

| Câu hỏi của Thầy/Cô | Cách trả lời trọng tâm |
| :--- | :--- |
| **Q1: Vì sao lại chọn Whisper-Medusa mà không dùng Faster-Whisper (CTranslate2)?** | *Faster-Whisper tối ưu ở tầng runtime C++ và lượng tử hóa INT8 trên CPU/GPU; còn Whisper-Medusa can thiệp vào tầng thuật toán giải mã (Speculative Decoding - giảm số chu kỳ Decoder). Hai phương pháp này bổ trợ cho nhau và có thể kết hợp cùng nhau.* |
| **Q2: Real-Time Factor (RTF) có ý nghĩa gì?** | *RTF = Thời gian xử lý / Thời lượng âm thanh. Ví dụ câu nói dài 5 giây mà mô hình xử lý mất 1 giây thì RTF = 0.2 (tức là nhanh hơn thời gian thực 5 lần). Trong hệ thống biên, RTF càng nhỏ thì thiết bị càng phản hồi tức thì và tiết kiệm pin.* |
| **Q3: Tập dữ liệu 200 mẫu lấy từ đâu và có đại diện không?** | *Tập 200 mẫu được chia đều 100 En và 100 Vi, bao phủ 14 miền nghiệp vụ nhà máy/công nghiệp (an toàn, vận hành máy, kho bãi, cảm biến...), có gán nhãn chuẩn Ground Truth để đo WER/CER chuẩn xác.* |
| **Q4: Chạy thử trên máy local và server khác nhau thế nào?** | *Trên máy tính cá nhân (GPU RTX 3050 4GB), nhóm dùng để code, debug và test demo giao diện. Trên server GPU NVIDIA A30 (24GB VRAM), nhóm chạy toàn bộ benchmark 200 mẫu quy mô lớn, đo chính xác Latency, RTF và Peak VRAM mà không lo nghẽn bộ nhớ.* |
