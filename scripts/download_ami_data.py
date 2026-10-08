#!/usr/bin/env python3
"""
Script tải và chuẩn bị dữ liệu AMI Meeting Corpus (BUT Test Split - 16 meetings):
1. Tải 16 file Audio Mix-Headset chính thức từ mirror của University of Edinburgh.
2. Tải và giải nén annotation gốc (ami_public_manual_1.6.2.zip).
3. Clone repository BUTSpeechFIT/AMI-diarization-setup để lấy test split, rttm, uem.
4. Chuyển đổi toàn bộ audio sang chuẩn Mono 16 kHz.
5. Ghép transcript Ground Truth chuẩn từ file XML của tất cả speaker theo starttime.
6. Xuất bảng manifest.csv tổng hợp cho 16 cuộc họp.
"""

import os
import sys
import zipfile
import subprocess
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
import pandas as pd

TEST_MEETINGS = [
    "IS1009a", "IS1009b", "IS1009c", "IS1009d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
    "TS3003a", "TS3003b", "TS3003c", "TS3003d",
    "EN2002a", "EN2002b", "EN2002c", "EN2002d"
]

ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DATA_DIR = ROOT_DIR / "data" / "ami_test"
RAW_AUDIO_DIR = BASE_DATA_DIR / "audio_raw"
MONO_AUDIO_DIR = BASE_DATA_DIR / "audio_16k_mono"
ANNOTATION_DIR = BASE_DATA_DIR / "annotations"
TRANSCRIPT_DIR = BASE_DATA_DIR / "transcripts"
BUT_REPO_DIR = BASE_DATA_DIR / "AMI-diarization-setup"

for d in [RAW_AUDIO_DIR, MONO_AUDIO_DIR, ANNOTATION_DIR, TRANSCRIPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def download_file_with_progress(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"  [Đã có sẵn] {dest_path.name}")
        return
    print(f"  [Đang tải] {url}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
        with open(temp_dest, "wb") as f, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    size = f.write(chunk)
                    pbar.update(size)
        temp_dest.rename(dest_path)
    except Exception as e:
        print(f"  [Lỗi tải {url}]: {e}")
        if temp_dest.exists():
            temp_dest.unlink()
        raise e

def clone_but_repo():
    print("\n--- 1. CLONE BUT DIARIZATION SETUP (TEST SPLIT & RTTM) ---")
    if BUT_REPO_DIR.exists():
        print("  [Đã có sẵn] BUT Diarization Setup repo.")
    else:
        print("  Đang clone BUTSpeechFIT/AMI-diarization-setup...")
        subprocess.run(
            ["git", "clone", "https://github.com/BUTSpeechFIT/AMI-diarization-setup.git", str(BUT_REPO_DIR)],
            check=True
        )

def download_annotations():
    print("\n--- 2. TẢI VÀ GIẢI NÉN ANNOTATION GỐC (MANUAL 1.6.2) ---")
    zip_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
    zip_path = ANNOTATION_DIR / "ami_public_manual_1.6.2.zip"
    download_file_with_progress(zip_url, zip_path)

    words_extracted_dir = ANNOTATION_DIR / "words"
    if not words_extracted_dir.exists():
        print("  Đang giải nén file XML words/...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Chỉ giải nén thư mục words để tiết kiệm không gian
            word_members = [m for m in zf.namelist() if m.startswith("words/")]
            zf.extractall(path=ANNOTATION_DIR, members=word_members)
        print("  Đã giải nén xong thư mục words/.")
    else:
        print("  [Đã có sẵn] Thư mục words/ XML.")

def download_audios():
    print("\n--- 3. TẢI 16 AUDIO MIX-HEADSET WAV (BUT TEST SPLIT) ---")
    base_url = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"
    for idx, meeting_id in enumerate(TEST_MEETINGS, 1):
        wav_url = f"{base_url}/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav"
        dest_file = RAW_AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
        print(f"[{idx}/16] Meeting: {meeting_id}")
        download_file_with_progress(wav_url, dest_file)

def convert_audio_to_mono_16k():
    print("\n--- 4. CHUYỂN ĐỔI AUDIO SANG MONO 16 KHZ ---")
    for idx, meeting_id in enumerate(TEST_MEETINGS, 1):
        raw_wav = RAW_AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
        out_wav = MONO_AUDIO_DIR / f"{meeting_id}.wav"
        if out_wav.exists() and out_wav.stat().st_size > 0:
            print(f"  [{idx}/16] [Đã có sẵn 16k mono] {out_wav.name}")
            continue

        print(f"  [{idx}/16] Đang xử lý: {meeting_id} -> mono 16kHz...")
        audio, sr = librosa.load(str(raw_wav), sr=16000, mono=True)
        sf.write(str(out_wav), audio, 16000, subtype="PCM_16")

def parse_and_build_transcripts():
    print("\n--- 5. TRÍCH XUẤT VÀ GHÉP TRANSCRIPT TỪ WORDS XML ---")
    words_dir = ANNOTATION_DIR / "words"
    records = []

    for idx, meeting_id in enumerate(TEST_MEETINGS, 1):
        xml_files = sorted(list(words_dir.glob(f"{meeting_id}.*.words.xml")))
        all_words = []

        for xf in xml_files:
            try:
                tree = ET.parse(xf)
                root = tree.getroot()
                for elem in root.findall(".//w"):
                    text = elem.text
                    start_time = elem.get("starttime")
                    # Lấy các từ hợp lệ có mốc thời gian
                    if text and start_time is not None:
                        text_clean = text.strip()
                        if text_clean:
                            all_words.append((float(start_time), text_clean))
            except Exception as e:
                print(f"  Cảnh báo lỗi đọc XML {xf.name}: {e}")

        # Ghép tất cả speaker theo thời điểm bắt đầu từ (starttime)
        all_words.sort(key=lambda x: x[0])
        full_transcript = " ".join([w[1] for w in all_words])

        # Ghi file text ground truth
        txt_path = TRANSCRIPT_DIR / f"{meeting_id}.txt"
        txt_path.write_text(full_transcript, encoding="utf-8")

        # Đọc độ dài file audio mono
        mono_path = MONO_AUDIO_DIR / f"{meeting_id}.wav"
        duration_sec = 0.0
        if mono_path.exists():
            info = sf.info(str(mono_path))
            duration_sec = round(info.duration, 2)

        records.append({
            "meeting_id": meeting_id,
            "audio_path": str(mono_path),
            "transcript_path": str(txt_path),
            "duration_sec": duration_sec,
            "word_count": len(all_words),
            "preview_text": full_transcript[:120] + "..." if len(full_transcript) > 120 else full_transcript
        })
        print(f"  [{idx}/16] {meeting_id}: {len(all_words)} từ, thời lượng {duration_sec}s")

    manifest_df = pd.DataFrame(records)
    manifest_path = BASE_DATA_DIR / "manifest_ami_test.csv"
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8")
    print(f"\n[Xong] Bảng manifest đã lưu tại: {manifest_path}")

def main():
    print("======================================================================")
    print("   CHUẨN BỊ BỘ DỮ LIỆU AMI MEETING CORPUS (16 TEST MEETINGS - BUT)   ")
    print("   Thư mục lưu: data/ami_test")
    print("======================================================================")
    clone_but_repo()
    download_annotations()
    download_audios()
    convert_audio_to_mono_16k()
    parse_and_build_transcripts()
    print("\n======================================================================")
    print("   HOÀN TẤT 100%! BỘ DỮ LIỆU ĐÃ SẴN SÀNG CHO WHISPER-TURBO BENCHMARK   ")
    print("======================================================================")

if __name__ == "__main__":
    main()
