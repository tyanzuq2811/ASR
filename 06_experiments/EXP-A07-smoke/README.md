# EXP-A07 smoke test — Whisper-Tiny pretrained

- Date: 2026-10-20
- Model: `openai/whisper-tiny` (37,760,640 parameters)
- Split: first 5 rows of VIVOS `test_in_domain`
- Device: CPU on `aiotlab-ws1`; GPU was not used because another process occupied about 19.5 GiB and 99% utilization.
- Decode: Vietnamese `transcribe`, greedy (`num_beams=1`), `max_new_tokens=200`, seed `20261020`.
- Input: 16 kHz mono PCM16 WAV from the frozen VIVOS manifest.

## Output

- WER: `0.6428571428571429`
- CER: `0.3027027027027027`
- Median inference time: `0.4783169925 s`
- Median RTF: `0.1308149993`
- Rows completed: `5/5`

Artifacts: `vivos_tiny_smoke.csv` and `vivos_tiny_smoke.json`.

This is a pipeline smoke test only. It is not the paper's baseline result and
must not be used to support an accuracy or edge-performance claim. The complete
760-utterance baseline is pending a valid A30 GPU slot.
