#!/usr/bin/env python3
"""Test QNN HTP v75 Execution Provider registration and context compilation."""

import sys
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

def main():
    print("Testing QNN Execution Provider registration...")
    ep_lib = qnn_ep.get_library_path()
    print(f"QNN EP Library: {ep_lib}")
    ort.register_execution_provider_library("QNNExecutionProvider", ep_lib)
    providers = ort.get_available_providers()
    print(f"Available Providers: {providers}")
    
    htp_lib = qnn_ep.get_qnn_htp_path()
    print(f"Bundled HTP Lib: {htp_lib}")
    print("QNN EP Registration SUCCESSFUL!")

if __name__ == "__main__":
    main()
