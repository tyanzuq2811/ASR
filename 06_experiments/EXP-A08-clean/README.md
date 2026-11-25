# EXP-A08 clean fine-tune

The clean Whisper-Tiny arm completed on the NVIDIA A30 using the
speaker-disjoint VIVOS `train`/`validation` split and the same 16 kHz/NFC
transcript contract as A-07. The first run (`run-001`) exposed a decoder
prefix mismatch and is retained as an invalid diagnostic run, not as a paper
result.

Reproduction command used on `aiotlab-ws1`:

```bash
CUDA_VISIBLE_DEVICES=0 HF_HOME=/path/to/cache \
python 04_code/train/finetune_whisper.py \
  --manifest /path/to/vivos_speaker_split.csv \
  --data-root /path/to/vivos/vivos \
  --output-dir /path/to/EXP-A08-clean/run-003-corrected \
  --model-id openai/whisper-tiny --epochs 3 \
  --batch-size 8 --gradient-accumulation-steps 4 \
  --learning-rate 1e-5 --fp16 --seed 20261020
```

The corrected run completed 972/972 optimizer steps (3 epochs). The final
checkpoint remains on A30 and is intentionally not committed to Git; run
metadata, trainer result, evaluation outputs and checkpoint-tree hash are kept
beside the remote checkpoint.

Diagnostic result from invalid `run-001` (do not report as paper result):

- Train rows: 10,388; validation rows: 1,272; seed: `20261020`.
- Hardware: NVIDIA A30; PyTorch `2.6.0+cu124`; CUDA `12.4`; FP16 enabled.
- Runtime: `535.9992 s`; `1.813` steps/s; train loss `0.7373367877`.
- Validation loss: `0.6891747117` (step 250), `0.5907765627` (step 500),
  `0.5577820539` (step 750).
- Checkpoint-tree SHA-256: `f9ef56ba5f49c5b4c1bad48d569637cceaa9eeb77eae1462912cabc8786dcfbe`.

The invalid `run-001` decoded with WER `8.801865` and CER `9.459186` because
its training labels used a processor without the Vietnamese/transcribe prefix.
`run-002-corrected` aligned that prefix but still masked the EOS token because
Whisper uses the same ID for padding and EOS; its WER `8.035354` and CER
`8.436514` are also invalid. The collator now masks by `attention_mask`,
preserving EOS, and the valid run is `run-003-corrected`.

Valid `run-003-corrected` result on the full VIVOS official test split (760
utterances): WER `0.269878` (26.99%), CER `0.119526` (11.95%), median
inference `0.087217 s`, median RTF `0.024590`, total inference `66.144671 s`.
The per-utterance CSV and JSON summary are mirrored under this directory; the
model weights remain on A30. Its checkpoint-tree SHA-256 is
`e5706c70b7fad38271107ceba4c072ef4719bea9aa8d41113b8f158b17f7324a`.

See `10_evidence/EXP-A08_CLEAN_FINETUNE_2026-10-20.md` for the evidence
boundary and the exact remote paths. `DRYRUN.md` documents only a 2-row CPU
implementation check; it is not a paper result.
