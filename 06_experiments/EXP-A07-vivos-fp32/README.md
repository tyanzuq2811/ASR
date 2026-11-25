# EXP-A07 — pretrained Whisper-Tiny FP32 reference

- Date: 2026-10-20
- Git/code: `run_whisper_baseline.py` from the repository at execution time
- Model: `openai/whisper-tiny`, 37,760,640 parameters
- Dataset/split: VIVOS official in-domain test, 760 utterances, 19 speakers
- Device: CPU on `aiotlab-ws1`; this is the reference accuracy run, not edge profiling
- Decode: Vietnamese `transcribe`, greedy (`num_beams=1`), `max_new_tokens=200`, seed `20261020`
- Input: frozen VIVOS speaker-disjoint manifest; 16 kHz mono PCM16 WAV

## Results

| Metric | Value |
|---|---:|
| WER | `0.5910385910` |
| CER | `0.3147294864` |
| Total audio | `2685.544563 s` |
| Total inference | `374.662751 s` |
| Median inference | `0.4746940136 s` |
| Median RTF | `0.1356880069` |
| Completed rows | `760/760` |

Artifacts:

- `vivos_tiny_fp32.csv`: per-utterance reference/hypothesis/timing.
- `vivos_tiny_fp32.json`: aggregate summary.
- `run.txt`: raw execution log copied from the A30 host.

This result is a pretrained CPU reference only. It does not support claims
about fine-tuning, quantization, noise robustness or Arduino latency/RAM.
