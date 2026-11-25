# EXP-A08 noise-aware fine-tune

This experiment trains Whisper-Tiny on the speaker-disjoint VIVOS split with
deterministic on-the-fly MUSAN mixing. Each train/validation row receives one
condition from `clean`, `20`, `10`, or `5` dB; the official VIVOS test rows are
expanded to all four conditions for a 3,040-row robustness evaluation. No
audio is copied into GitHub.

## Locked inputs

- Model: `openai/whisper-tiny`
- Dataset: VIVOS speaker split, 10,388 train rows and 1,272 validation rows
- Noise: MUSAN SLR17, 930 validated 16 kHz WAV files
- Manifest: `03_data/processed/vivos_noise_mixed_manifest.csv`
- Manifest SHA-256: `fd5375c257f23c1068b6f1c0a93903c520f449b37b68515ee432cef39f04f9b6`
- Seed: `20261020`
- Host: `aiotlab-ws1`, NVIDIA A30; PyTorch `2.6.0+cu124`; CUDA `12.4`

The MUSAN archive is 11,086,114,085 bytes and has SHA-256
`86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1`.

## Training

The run used 3 epochs, batch size 8, gradient accumulation 4, learning rate
`1e-5`, warmup ratio `0.05`, FP16, and the explicit Vietnamese/transcribe
decoder contract. Reproduction command (with local paths substituted):

```bash
python 04_code/preprocess/make_noise_manifest.py \
  --manifest /path/to/vivos_speaker_split.csv \
  --noise-root /path/to/musan/extracted/musan \
  --output 03_data/processed/vivos_noise_mixed_manifest.csv \
  --summary 03_data/processed/vivos_noise_mixed_manifest.json \
  --seed 20261020

python 04_code/train/finetune_whisper.py \
  --manifest /path/to/vivos_noise_mixed_manifest.csv \
  --data-root /path/to/vivos/vivos \
  --noise-root /path/to/musan/extracted/musan \
  --output-dir /path/to/EXP-A08-noise-aware/run-001 \
  --model-id openai/whisper-tiny --epochs 3 --batch-size 8 \
  --gradient-accumulation-steps 4 --learning-rate 1e-5 --fp16 \
  --seed 20261020
```

Remote output root:
`/hdd3/users/baolt/ICCIT_2026_VENTUNO_Q_Paper_DATA/06_experiments/EXP-A08-noise-aware/run-001/`.
The run completed 972/972 steps in `822.7036 s`, with train loss
`0.5687504109`. The latest logged validation losses were `0.6741663814`
(step 250), `0.6092818379` (step 500), and `0.5828081369` (step 750).
Checkpoint-tree SHA-256:
`6b4f9bc7be88fcab0ecbfe49cc87d06b4fe73872b5b0fdc610106c4bed678227`.

## Mixed-test results

The final checkpoint was decoded greedily with the same Vietnamese prompt as
the clean arm. Each condition has 760 held-out VIVOS test rows.

| Condition | Noise-aware WER | Noise-aware CER | Clean-fine-tuned WER | Clean-fine-tuned CER |
|---|---:|---:|---:|---:|
| clean | 0.271432 | 0.121510 | 0.269878 | 0.119526 |
| 20 dB | 0.283605 | 0.128071 | 0.289692 | 0.129566 |
| 10 dB | 0.313390 | 0.149400 | 0.335794 | 0.159531 |
| 5 dB | 0.352370 | 0.173324 | 0.373996 | 0.186110 |
| all 3,040 | 0.305199 | 0.143076 | 0.317340 | 0.148683 |

Relative to the clean-fine-tuned checkpoint on the same mixed-test rows,
noise-aware training reduces overall WER by `0.012141` (3.83% relative), with
the largest absolute gains at 10 dB (`0.022404`) and 5 dB (`0.021627`). The
clean condition is slightly worse by `0.001554` WER, so the result supports a
robustness trade-off rather than a universal clean-accuracy win.

## Mirrored artifacts

- `run-001/run_config.json`
- `run-001/train_result.json`
- `run-001/noise_manifest_summary.json`
- `run-001/noise_manifest_sha256.txt`
- `run-001/checkpoint_tree_sha256.txt`
- `run-001/vivos_noise_test_results.csv`
- `run-001/vivos_noise_test_summary.json`
- `run-001/vivos_noise_test_results_cleanft.csv`
- `run-001/vivos_noise_test_summary_cleanft.json`

Model weights and raw audio remain on the A30 host and are not committed.
The experiment does not yet support cross-corpus FLEURS claims, CPU parity,
QAIRT/QNN compilation, Arduino latency/RAM, or security claims.

## CPU evaluation

The same final checkpoint and 3,040-row mixed manifest were evaluated on CPU
with the same greedy Vietnamese/transcribe decoder contract:

- WER `0.3050375550`; CER `0.1429770224`
- Median inference `0.4167099893 s`; median RTF `0.1182620004`
- Total audio `10742.178252 s`; total inference `1259.838829 s`

Per-condition CPU WER/CER were clean `0.271044/0.121357`, 20 dB
`0.283476/0.128010`, 10 dB `0.313390/0.149431`, and 5 dB
`0.352240/0.173110`. The CPU and A30 WER difference is `0.000162` absolute;
this is a reproducibility check, not an edge-device measurement.

## ONNX handoff

Encoder and decoder graphs were exported from the final checkpoint with opset
17 and validated by `onnx.checker`. The decoder uses `use_cache=False` and
accepts a complete prefix; it is a portable parity artifact, not yet a
KV-cache/HTP-optimized graph.

- Remote binary directory:
  `/hdd3/users/baolt/ICCIT_2026_VENTUNO_Q_Paper_DATA/05_models/onnx/EXP-A08-noise-aware/run-001/`
- Encoder: 32,890,409 bytes; SHA-256
  `f22ba29ad65ae4cd993da9f36b9469e6e7914869aea66e3d55370635c830c522`
- Decoder: 197,990,359 bytes; SHA-256
  `1ffccb58c26225f3d4e4485ccf2b77042608c2e189d9c99f54fbf66b30b6778e`
- Sample shapes: encoder input `[1,80,3000]`, hidden `[1,1500,384]`, decoder
  prefix `[1,4]`, logits `[1,4,51865]`.
- Prompt IDs: `[50258,50278,50359,50363]` (`vi`, `transcribe`, `notimestamps`).

The export metadata is mirrored in `run-001/onnx_export_metadata.json`; runtime
ONNX/QAIRT parity remains Dũng's B-01 task.
