#!/usr/bin/env python3
"""Build and quantize Whisper models to FP16, W8A16, and INT8 configurations.

Produces:
1. FP16: Half-precision floating point model.
2. W8A16: 8-bit weights and 16-bit activations (QDQ for Hexagon HTP v75).
3. INT8: 8-bit weights and 8-bit activations (QDQ for maximum throughput).

Calibrated using representative VIVOS speech samples.
Outputs reproducible checksums, model sizes, and validation logs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


class RawCalibrationDataReader(CalibrationDataReader):
    """Feeds pre-extracted float32 log-mel spectrogram tensors into ORT quantizer."""

    def __init__(self, raw_dir: Path, input_name: str = "input_features"):
        self.raw_files = sorted(raw_dir.glob("*.raw"))
        self.input_name = input_name
        self.enum_data = iter(self.raw_files)
        print(f"CalibrationDataReader initialized with {len(self.raw_files)} samples.")

    def get_next(self) -> dict | None:
        try:
            raw_path = next(self.enum_data)
        except StopIteration:
            return None

        # Load float32 raw array of shape (1, 80, 3000)
        data = np.fromfile(str(raw_path), dtype=np.float32).reshape(1, 80, 3000)
        return {self.input_name: data}

    def rewind(self):
        self.enum_data = iter(self.raw_files)


def build_fp16(input_onnx: Path, output_onnx: Path) -> dict:
    print(f"\n[1/3] Building FP16 model: {output_onnx.name}...")
    model = onnx.load(str(input_onnx))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(model_fp16, str(output_onnx))
    onnx.checker.check_model(str(output_onnx))

    size_bytes = output_onnx.stat().st_size
    sha256 = compute_sha256(output_onnx)
    print(f"  -> Generated {output_onnx.name}: {size_bytes:,} bytes, SHA256: {sha256}")
    return {"precision": "FP16", "bytes": size_bytes, "sha256": sha256}


def build_int8(input_onnx: Path, output_onnx: Path, raw_dir: Path) -> dict:
    print(f"\n[2/3] Building INT8 (W8A8 QDQ) model: {output_onnx.name}...")
    reader = RawCalibrationDataReader(raw_dir, input_name="input_features")
    quantize_static(
        model_input=str(input_onnx),
        model_output=str(output_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )
    onnx.checker.check_model(str(output_onnx))

    size_bytes = output_onnx.stat().st_size
    sha256 = compute_sha256(output_onnx)
    print(f"  -> Generated {output_onnx.name}: {size_bytes:,} bytes, SHA256: {sha256}")
    return {"precision": "INT8 (W8A8 QDQ)", "bytes": size_bytes, "sha256": sha256}


def build_w8a16(input_onnx: Path, output_onnx: Path, raw_dir: Path) -> dict:
    print(f"\n[3/3] Building W8A16 (Weight INT8, Activation INT16 QDQ) model: {output_onnx.name}...")
    reader = RawCalibrationDataReader(raw_dir, input_name="input_features")
    quantize_static(
        model_input=str(input_onnx),
        model_output=str(output_onnx),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt16,
    )
    onnx.checker.check_model(str(output_onnx))

    size_bytes = output_onnx.stat().st_size
    sha256 = compute_sha256(output_onnx)
    print(f"  -> Generated {output_onnx.name}: {size_bytes:,} bytes, SHA256: {sha256}")
    return {"precision": "W8A16 QDQ", "bytes": size_bytes, "sha256": sha256}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-onnx", type=Path, required=True, help="Path to base FP32 ONNX model")
    parser.add_argument("--calibration-dir", type=Path, required=True, help="Directory containing .raw calibration files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save quantized models")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("WHISPER QUANTIZATION PIPELINE (FP16, W8A16, INT8)")
    print(f"Input: {args.input_onnx}")
    print(f"Calibration: {args.calibration_dir}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    base_name = args.input_onnx.stem  # e.g., 'encoder'
    manifest = {
        "base_model": str(args.input_onnx),
        "base_sha256": compute_sha256(args.input_onnx),
        "base_bytes": args.input_onnx.stat().st_size,
        "configurations": {},
    }

    # 1. FP16
    fp16_path = args.output_dir / f"{base_name}_fp16.onnx"
    manifest["configurations"]["fp16"] = build_fp16(args.input_onnx, fp16_path)

    # 2. INT8 (W8A8)
    int8_path = args.output_dir / f"{base_name}_int8.onnx"
    manifest["configurations"]["int8"] = build_int8(args.input_onnx, int8_path, args.calibration_dir)

    # 3. W8A16
    w8a16_path = args.output_dir / f"{base_name}_w8a16.onnx"
    manifest["configurations"]["w8a16"] = build_w8a16(args.input_onnx, w8a16_path, args.calibration_dir)

    # Save metadata summary
    summary_path = args.output_dir / f"{base_name}_quantization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE. Summary:")
    print(json.dumps(manifest, indent=2))
    print("=" * 60)


if __name__ == "__main__":
    main()
