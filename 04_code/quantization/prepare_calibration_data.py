#!/usr/bin/env python3
"""Prepare raw calibration data from VIVOS dataset for Qualcomm QNN quantization.

Generates raw float32 binary tensors for Whisper Encoder (input_features: [1, 80, 3000])
and creates the standard QNN input_list.txt for qnn-onnx-converter / qairt-converter.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="Path to fine-tuned Whisper checkpoint")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to vivos_speaker_split.csv")
    parser.add_argument("--data-root", type=Path, required=True, help="Root folder for VIVOS audio")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for raw files and input_list.txt")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of calibration samples")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading processor from {args.model_id}...")
    processor = WhisperProcessor.from_pretrained(args.model_id)

    print(f"Reading manifest from {args.manifest}...")
    with open(args.manifest, "r", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") == "train"]

    selected_rows = rows[: args.num_samples]
    print(f"Extracting {len(selected_rows)} calibration tensors...")

    input_list_lines = []
    for idx, row in enumerate(selected_rows, start=1):
        audio_id = row.get("utterance_id", f"sample_{idx:03d}")
        audio_path = args.data_root / row["audio_path"]

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        input_features = processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="np"
        ).input_features.astype(np.float32)

        raw_filename = f"encoder_input_{audio_id}.raw"
        raw_filepath = raw_dir / raw_filename
        input_features.tofile(str(raw_filepath))

        input_list_lines.append(f"input_features:={raw_filepath}")
        if idx % 5 == 0 or idx == len(selected_rows):
            print(f"[{idx}/{len(selected_rows)}] Saved: {raw_filename} ({input_features.nbytes:,} bytes, shape {input_features.shape})")

    input_list_path = args.output_dir / "encoder_calibration_input_list.txt"
    with open(input_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(input_list_lines) + "\n")

    print(f"\nCalibration generation complete!")
    print(f"Input list saved to: {input_list_path}")
    print(f"Total raw files: {len(input_list_lines)}")


if __name__ == "__main__":
    main()
