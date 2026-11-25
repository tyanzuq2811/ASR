# A-08 code dry-run (not a paper experiment)

- Date: 2026-10-20
- Host: `aiotlab-ws1`, CPU only because the A30 GPU was occupied by another user.
- Data: first 2 VIVOS train rows and first 2 validation rows.
- Command mode: one epoch, batch size 1, gradient accumulation 1, learning rate `1e-5`, seed `20261020`.
- Result: forward/backward, evaluation, checkpoint save and checkpoint hash all completed.
- Training loss: `5.6267843246`.
- Evaluation loss: `8.6085281372` at epoch 1.0.

Artifacts in `dryrun/` record the run config, metrics and checkpoint-tree hash.
This dry-run validates the implementation only; it must not be reported as a
fine-tuning result or used to support any manuscript claim.
