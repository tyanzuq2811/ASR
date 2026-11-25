# Public supplementary measurements (2026-10-29)

This directory contains sanitized, per-utterance metrics for the clean-speech PhoWhisper-Tiny comparison and the full-set unprofiled QNN+CPU run. Transcript text, audio, machine-local paths, credentials, and security reproduction inputs are excluded.

- `phowhisper_tiny_clean_760_per_utterance.csv`: 760 paired clean VIVOS rows for FP32 and W8A16-encoder/INT8-decoder PhoWhisper-Tiny; includes edit counts, timing, and model label.
- `phowhisper_tiny_clean_summary.json`: checkpoint, test-manifest SHA-256, decoding limits, model hashes, and aggregate/paired statistics.
- `qnn_cpu_unprofiled_760_per_utterance.csv`: 760 rows of timing, per-row error rates, and process memory; no reference or hypothesis strings.
- `qnn_cpu_unprofiled_summary.json`: provider registration, artifact hashes, and aggregate metrics. Profiling was disabled, so this does not provide node-level HTP attribution.

VIVOS audio is not redistributed; the manifest hash identifies the evaluated test set. The PhoWhisper quantization comparison is exploratory and has no preregistered non-inferiority margin. The security audit's raw inputs and logs remain withheld during coordinated disclosure.
