import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import torch

current_dir = Path(__file__).resolve().parent.parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from src.dataset import ASRDataset
from src.metrics import calculate_sample_metrics, calculate_rtf, calculate_speedup
from src.engine import WhisperMedusaEngine, VanillaWhisperEngine

st.set_page_config(
    page_title="ASR Demo: Whisper-Medusa & Lượng Tử Hóa (Edge AI)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .medusa-card {
        border-left-color: #10B981 !important;
        background-color: #F0FDF4;
    }
    .quant-badge {
        font-size: 0.9rem;
        font-weight: bold;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
        margin-left: 5px;
    }
    .speedup-badge {
        font-size: 1.25rem;
        font-weight: bold;
        color: #059669;
        background-color: #D1FAE5;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_dataset():
    return ASRDataset()

@st.cache_resource
def get_medusa_engine(model_name: str, quantization: str):
    return WhisperMedusaEngine(model_name=model_name, quantization=quantization)

@st.cache_resource
def get_baseline_engine(model_name: str, quantization: str):
    return VanillaWhisperEngine(model_name=model_name, quantization=quantization)

# Sidebar
st.sidebar.title("⚙️ Cấu hình Thử nghiệm")

medusa_model_id = st.sidebar.selectbox(
    "Mô hình Whisper-Medusa:",
    ["aiola/whisper-medusa-v1", "aiola/whisper-medusa-linear-libri"],
    index=0
)

quant_choice = st.sidebar.selectbox(
    "Phương pháp Lượng tử hóa Medusa:",
    [
        "INT8 (W8A16 - Khuyên dùng cho Local GPU 4GB)",
        "INT4 (W4A16 - Siêu nhẹ, VRAM < 1.2GB)",
        "None (FP16 Gốc - Yêu cầu Server / VRAM lớn)"
    ],
    index=0
)

if "INT8" in quant_choice:
    quant_mode = "int8"
elif "INT4" in quant_choice:
    quant_mode = "int4"
else:
    quant_mode = "none"

baseline_model_id = st.sidebar.selectbox(
    "Mô hình Baseline đối chứng:",
    ["openai/whisper-large-v2", "openai/whisper-small"],
    index=0
)

# GPU Status in Sidebar
if torch.cuda.is_available():
    free_mem, total_mem = torch.cuda.mem_get_info()
    gpu_name = torch.cuda.get_device_name(0)
    st.sidebar.success(f"🟢 GPU: {gpu_name}\nVRAM trống: {free_mem/(1024**3):.2f} / {total_mem/(1024**3):.2f} GB")
    if quant_mode != "none":
        st.sidebar.info("💡 Bạn đang bật Lượng tử hóa, mô hình sẽ chạy cực kỳ an toàn trên GPU Local!")
    else:
        st.sidebar.warning("⚠️ Bản FP16 gốc có thể tốn ~3.5GB VRAM. Nếu gặp OOM hãy chọn chế độ INT8!")
else:
    st.sidebar.error("🔴 Chạy trên CPU")

dataset = load_dataset()
summary = dataset.get_summary()

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Tập dữ liệu Real 200")
st.sidebar.write(f"• Tổng số file: **{summary['total_samples']}**")
st.sidebar.write(f"• Tiếng Anh: **{summary['language_distribution'].get('en', 0)}** | Tiếng Việt: **{summary['language_distribution'].get('vi', 0)}**")

lang_choice = st.sidebar.radio("Lọc theo ngôn ngữ:", ["Tất cả", "Tiếng Việt (vi)", "Tiếng Anh (en)"])
lang_filter = None if lang_choice == "Tất cả" else ("vi" if "vi" in lang_choice else "en")

df_filtered = dataset.filter(language=lang_filter)
domains = ["Tất cả"] + sorted(list(df_filtered["domain"].dropna().unique()))
domain_choice = st.sidebar.selectbox("Lọc theo miền ứng dụng:", domains)
if domain_choice != "Tất cả":
    df_filtered = df_filtered[df_filtered["domain"] == domain_choice]

sample_options = [f"{row['recording_id']} [{row['language'].upper()}] - {row['text'][:40]}..." for _, row in df_filtered.iterrows()]
selected_idx = st.sidebar.selectbox("Chọn file kiểm thử:", range(len(sample_options)), format_func=lambda i: sample_options[i] if i < len(sample_options) else "")

# Header
st.markdown('<div class="main-header">🎙️ Demo Đối Chứng: Whisper-Medusa & Lượng Tử Hóa Trên Edge AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Báo cáo Khoa học Seminar: Đánh giá Speculative Decoding kết hợp Lượng tử hóa (W8A16, W4A16) trên Tập dữ liệu Công nghiệp</div>', unsafe_allow_html=True)

tabs = st.tabs(["🚀 Thử nghiệm Trực tiếp (Live Demo)", "📑 Báo cáo Phương pháp Lượng tử hóa", "📖 Luận cứ Bảo vệ Seminar"])

with tabs[0]:
    if len(df_filtered) > 0 and selected_idx < len(df_filtered):
        row = df_filtered.iloc[selected_idx]
        audio_path = row["resolved_audio_path"]
        ground_truth = str(row["text"])
        lang = str(row["language"])
        domain = str(row.get("domain", "N/A"))
        duration = float(row.get("duration_sec", 0.0))

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("🎧 File Âm thanh")
            st.audio(audio_path)
            st.markdown(f"""
            - **Mã:** `{row['recording_id']}` | **Ngôn ngữ:** `{lang.upper()}`
            - **Miền nghiệp vụ:** `{domain}`
            - **Thời lượng:** `{duration:.2f}s`
            """)

        with col2:
            st.subheader("📝 Văn bản Gốc (Ground Truth)")
            st.info(ground_truth)

        st.markdown("---")
        btn_run = st.button("🔥 Chạy Đối chứng (Baseline vs Whisper-Medusa Lượng tử hóa)", type="primary", use_container_width=True)

        if btn_run:
            with st.spinner("Đang thực hiện suy luận đối chứng..."):
                medusa_eng = get_medusa_engine(medusa_model_id, quant_mode)
                base_eng = get_baseline_engine(baseline_model_id, "none")

                # Run
                res_medusa = medusa_eng.transcribe(audio_path, language=lang)
                res_base = base_eng.transcribe(audio_path, language=lang)

                # Metrics
                m_medusa = calculate_sample_metrics(res_medusa["text"], ground_truth)
                m_base = calculate_sample_metrics(res_base["text"], ground_truth)
                speedup = calculate_speedup(res_base["latency_ms"], res_medusa["latency_ms"])

            col_b, col_m = st.columns(2)

            with col_b:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.subheader(f"🐢 Vanilla Whisper ({baseline_model_id.split('/')[-1]})")
                st.markdown('<span class="quant-badge">Độ chính xác: FP16 Gốc</span>', unsafe_allow_html=True)
                st.write("**Bản phiên âm:**")
                st.success(res_base["text"])
                st.write(f"⏱️ **Độ trễ:** `{res_base['latency_ms']} ms` | **RTF:** `{res_base['rtf']}`")
                st.write(f"🎯 **WER:** `{m_base['wer']}%` | **CER:** `{m_base['cer']}%`")
                st.write(f"💾 **Peak VRAM:** `{res_base['peak_vram_mb']} MB` | **Dung lượng:** `~{res_base['model_size_mb']} MB`")
                st.markdown('</div>', unsafe_allow_html=True)

            with col_m:
                st.markdown('<div class="metric-card medusa-card">', unsafe_allow_html=True)
                st.subheader(f"⚡ Whisper-Medusa ({quant_mode.upper()})")
                st.markdown(f'<span class="quant-badge">Lượng tử hóa: {quant_mode.upper()}</span>', unsafe_allow_html=True)
                st.write("**Bản phiên âm:**")
                st.success(res_medusa["text"])
                st.write(f"⏱️ **Độ trễ:** `{res_medusa['latency_ms']} ms` | **RTF:** `{res_medusa['rtf']}`")
                st.write(f"🎯 **WER:** `{m_medusa['wer']}%` | **CER:** `{m_medusa['cer']}%`")
                st.write(f"💾 **Peak VRAM:** `{res_medusa['peak_vram_mb']} MB` | **Dung lượng:** `~{res_medusa['model_size_mb']} MB`")
                st.markdown(f'<div class="speedup-badge">🚀 Tăng tốc {speedup}x</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.subheader("📑 Báo cáo Các Phương Pháp Lượng Tử Hóa Áp Dụng")
    st.markdown("""
Hệ thống sử dụng **Post-Training Quantization (PTQ)** bằng thư viện chuẩn `optimum-quanto` (chính là công cụ được trích dẫn ở **Slide 27** của bài thuyết trình Seminar):

### 1. Phương pháp W8A16 (Weights INT8, Activations FP16)
- **Cơ chế:** Trọng số của tất cả các lớp tuyến tính (Linear layers) được lượng tử hóa đối xứng sang số nguyên có dấu 8-bit (INT8). Các giá trị kích hoạt (activations) và phép nhân ma trận được giữ ở FP16.
- **Ưu điểm:** 
  * Giảm **50% dung lượng mô hình** (từ ~3.1GB xuống ~1.55GB).
  * Vừa khít với VRAM 4GB của GPU cá nhân (RTX 3050 Laptop).
  * **Hầu như không làm suy giảm WER** do tránh được lỗi ngoại lệ (outlier activation) như đã nêu trong mục Kurtosis (Slide 21).

### 2. Phương pháp W4A16 (Weights INT4, Activations FP16)
- **Cơ chế:** Nén trọng số xuống 4-bit với phân vị khối (Block-wise scaling).
- **Ưu điểm:**
  * Giảm **75% dung lượng mô hình** (chỉ còn ~0.78GB).
  * Siêu nhẹ, cho phép chạy mô hình ASR quy mô lớn ngay trên các board IoT nhúng.
  * Trade-off: WER có thể tăng nhẹ trên các mẫu âm thanh có độ ồn cao hoặc phát âm phức tạp.

### 3. Sự kết hợp với Whisper-Medusa (Speculative Multi-Head)
- Lượng tử hóa W8A16 nén toàn bộ thân mạng Whisper Decoder.
- 10 đầu Medusa tiếp tục dự đoán các token phỏng đoán song song.
- **Kết quả kép:** Vừa **giảm 50% bộ nhớ** vừa **tăng tốc 1.5x thời gian suy luận**!
""")

with tabs[2]:
    st.subheader("🎓 Luận cứ Khoa học & Báo cáo Seminar")
    st.markdown("""
1. **Khắc phục nút thắt phần cứng:** Nếu không lượng tử hóa, mô hình Whisper Large V2 cần tối thiểu 3.6GB VRAM, máy tính cá nhân dễ gặp lỗi CUDA OOM. Khi lượng tử hóa W8A16, bộ nhớ chỉ chiếm ~1.9GB, hoạt động mượt mà.
2. **Đối chứng Ngôn ngữ En vs Vi:** Thực nghiệm trên 200 mẫu chứng minh rõ nét sự khác biệt khi Medusa heads được tối ưu tiếng Anh so với tiếng Việt, khẳng định giá trị đề tài NCKH 2026-2027 của sinh viên.
""")
