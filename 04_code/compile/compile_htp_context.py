#!/usr/bin/env python3
"""Compile ONNX model to Qualcomm Hexagon HTP v75 Context Cache Binary."""

import argparse
import hashlib
import os
from pathlib import Path
import time
import numpy as np
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True, help="Path to input ONNX model")
    parser.add_argument("--output-bin", type=Path, required=True, help="Path to save context cache binary")
    parser.add_argument("--backend", default="/usr/lib/libQnnHtp.so", help="QNN backend library")
    parser.add_argument("--soc-model", default="675", help="Qualcomm SoC model (675 for QCS8275)")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_bin.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("QUALCOMM QNN HTP v75 CONTEXT COMPILER")
    print(f"Input model: {args.model_path}")
    print(f"Target binary: {args.output_bin}")
    print(f"Backend: {args.backend} | SoC ID: {args.soc_model}")
    print("=" * 60)

    ep_lib = qnn_ep.get_library_path()
    ort.register_execution_provider_library("QNNExecutionProvider", ep_lib)

    so = ort.SessionOptions()
    ep_devices = [dev for dev in ort.get_ep_devices() if dev.ep_name == "QNNExecutionProvider"]
    
    # Point ADSP_LIBRARY_PATH to onnxruntime_qnn site-packages where matching V75 Skel lives
    qnn_dir = Path(qnn_ep.__file__).parent
    os.environ["ADSP_LIBRARY_PATH"] = str(qnn_dir)
    os.environ["LD_LIBRARY_PATH"] = f"{qnn_dir}:/usr/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

    backend_path = args.backend
    if backend_path == "cpu":
        backend_path = "/usr/lib/libQnnCpu.so"
    elif backend_path == "htp":
        backend_path = str(qnn_dir / "libQnnHtp.so")
    print(f"Using QNN backend: {backend_path}")
    print(f"ADSP_LIBRARY_PATH: {os.environ.get('ADSP_LIBRARY_PATH')}")

    # QNN HTP configuration
    ep_options = {
        "backend_path": backend_path,
        "qnn_context_priority": "high",
        "soc_model": args.soc_model,
        "htp_graph_finalization_optimization_mode": "3",
        "qnn_context_cache_enable": "1",
        "qnn_context_cache_path": str(args.output_bin),
    }

    if ep_devices:
        so.add_provider_for_devices(ep_devices, ep_options)
    else:
        so.set_providers(["QNNExecutionProvider"], [ep_options])

    print("\nStarting graph compilation & HTP finalization...")
    t0 = time.perf_counter()
    sess = ort.InferenceSession(str(args.model_path), sess_options=so)
    compile_time = time.perf_counter() - t0
    print(f"Graph initialized in {compile_time:.3f} s.")

    # Run dummy warm-up inference to trigger HTP context generation
    input_meta = sess.get_inputs()[0]
    shape = [dim if isinstance(dim, int) else 1 for dim in input_meta.shape]
    dtype = np.float16 if "float16" in input_meta.type else np.float32
    print(f"Running warm-up execution with shape {shape} ({dtype.__name__})...")
    dummy_input = np.zeros(shape, dtype=dtype)
    
    t0 = time.perf_counter()
    out = sess.run(None, {input_meta.name: dummy_input})
    infer_time = time.perf_counter() - t0
    print(f"Warm-up inference completed in {infer_time:.3f} s.")

    if args.output_bin.exists():
        bin_size = args.output_bin.stat().st_size
        bin_hash = compute_sha256(args.output_bin)
        print(f"\n[SUCCESS] Context binary generated: {args.output_bin}")
        print(f"  Size: {bin_size:,} bytes")
        print(f"  SHA-256: {bin_hash}")
    else:
        print(f"\n[NOTE] Execution finished. Output bin not directly generated at {args.output_bin} (checking context cache state)")

    ort.unregister_execution_provider_library("QNNExecutionProvider")


if __name__ == "__main__":
    main()
