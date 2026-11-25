# Whisper-Tiny baseline

`run_whisper_baseline.py` evaluates the pretrained `openai/whisper-tiny`
checkpoint with greedy decoding, Vietnamese language/task prompts and the
locked normalized transcript policy.

Example smoke test on the A30 host (CPU while the GPU is occupied):

```bash
CUDA_VISIBLE_DEVICES= HF_HOME=/path/to/cache \
python run_whisper_baseline.py \
  --manifest /path/to/vivos_speaker_split.csv \
  --data-root /path/to/vivos/vivos \
  --split test_in_domain --limit 5 --device cpu \
  --output /path/to/smoke.csv --summary /path/to/smoke.json
```

The complete 760-utterance baseline must be run only after recording a valid
GPU/CPU allocation and should be preserved with its raw log, Git commit and
model revision. A smoke test is a pipeline check, not a paper result.
