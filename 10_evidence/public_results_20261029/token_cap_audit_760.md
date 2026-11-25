# Token-ceiling audit for the 760 clean VIVOS pairs (2026-10-29)

This is a read-only analysis of archived predictions. No model or board was run.
The audit matched all 760 clean A30 FP32 rows to the 760 physical-edge CPU
rows by utterance ID. It used the multilingual Whisper fast tokenizer
(`openai/whisper-base` local tokenizer, Transformers 4.57.3 and Tokenizers
0.22.2, `add_special_tokens=False`) to count
decoded hypothesis text. The Whisper vocabulary is shared across the model
sizes. Re-tokenizing decoded text is a screening measure, not a record of the
original generated token IDs.

| Archived arm | Text-token range | Rows with text count >= 64 | Recorded generation count |
|---|---:|---:|---|
| A30 FP32, clean | 3-38 | 0/760 | Not retained |
| VENTUNO Q edge CPU W8A16, clean | 3-40 | 0/760 | 4-41; 0/760 reached 64 |

The edge inference script `tools/run_e1_edge_eval.py` has a 64-token ceiling:
one initial token followed by at most 63 more decoder calls. Its CSV records
`tokens_gen` for each row, including any EOS token, so none of the archived
edge rows exhausted that ceiling. The baseline PyTorch script
`04_code/baseline/run_whisper_baseline.py` currently defaults to
`--max-new-tokens 200`, but the A30 prediction artifact does **not** retain
the actual invocation, checkpoint `generation_config`, or per-row generated
IDs/counts. Thus the actual A30 generation ceiling cannot be verified from
the available local evidence.

Interpretation of the three relevant cases:

1. **Neither arm cut at the ceiling:** directly supported for edge and
   consistent with the FP32 decoded-text counts; likely no 64-token truncation
   effect on the stored comparison, but not fully demonstrated for A30.
2. **Only edge cut:** contradicted by the edge's recorded maximum of 41/64
   generated tokens (all 760 rows completed successfully).
3. **Only A30 cut:** the decoded FP32 text has at most 38 Whisper tokens, which
   offers no positive indication of cutting; the missing original token IDs
   and invocation prevent formally excluding it.

The existing paired WER and TOST calculations remain valid **for the stored
predictions**. A claim that the FP32-versus-edge comparison controlled the
generation ceiling still requires the A30 run command/configuration or a
common-cap FP32 rerun. The tokenizer screening should not be described as that
rerun.

Input SHA-256:

- A30 CSV: `f5a7a4d423ba3439ac1974d2648911a9503ea51e7baafedb02c5378ee778d1b8`
- Edge CSV: `6bcd57c15bb1c2b7f184f687544afe498e9c1c1be7d1651380382d727ef84f87`
- Local Whisper tokenizer JSON: `27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566`
