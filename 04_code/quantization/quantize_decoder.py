#!/usr/bin/env python3
"""Quantize Whisper Decoder model to FP16 and INT8 weight-only configurations.

Decoder has dynamic token length input (decoder_input_ids: [1, T]), making dynamic
quantization (weight INT8) and FP16 the standard stable configurations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import onnx
from onnxconverter_common import float16
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_name = args.input_onnx.stem

    # 1. FP16
    fp16_out = args.output_dir / f"{base_name}_fp16.onnx"
    print(f"Building FP16 decoder: {fp16_out.name}...")
    model = onnx.load(str(args.input_onnx))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(model_fp16, str(fp16_out))
    onnx.checker.check_model(str(fp16_out))
    print(f"  -> Generated {fp16_out.name}: {fp16_out.stat().st_size:,} bytes")

    # 2. Dynamic INT8 (weights int8, activations dynamic)
    int8_out = args.output_dir / f"{base_name}_int8.onnx"
    print(f"Building dynamic INT8 decoder: {int8_out.name}...")
    quantize_dynamic(
        model_input=str(args.input_onnx),
        model_output=str(int8_out),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )
    onnx.checker.check_model(str(int8_out))
    print(f"  -> Generated {int8_out.name}: {int8_out.stat().st_size:,} bytes")

    manifest = {
        "base_model": str(args.input_onnx),
        "base_bytes": args.input_onnx.stat().st_size,
        "base_sha256": compute_sha256(args.input_onnx),
        "fp16": {
            "bytes": fp16_out.stat().st_size,
            "sha256": compute_sha256(fp16_out),
        },
        "int8": {
            "bytes": int8_out.stat().st_size,
            "sha256": compute_sha256(int8_out),
        },
    }

    summary_path = args.output_dir / f"{base_name}_quantization_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("\nSummary:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
