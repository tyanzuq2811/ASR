# Preprocessing contract

This folder will contain deterministic preprocessing utilities for the locked dataset protocol.

## Required order

1. Inventory each source archive/config and record byte size plus checksum in `03_data/DATASET_MANIFEST.csv`.
2. Extract metadata only; do not copy raw audio into Git.
3. Normalize transcript text according to `03_data/preprocess_config.yaml` while preserving Vietnamese diacritics.
4. Build a speaker-disjoint split and run an automated overlap check before any training.
5. Resample to 16 kHz mono and emit a manifest with source ID, speaker hash, duration, split and normalized transcript.
6. Generate noisy manifests only after the clean split is frozen. Store SNR and random seed per row.

## Non-negotiable leakage rules

- VIVOS official test speakers never enter training or validation.
- Common Voice test is excluded from training/tuning.
- FLEURS Vietnamese test is cross-corpus evaluation only.
- Noise is added after the clean split is frozen; the same utterance may have deterministic noisy views, but its split cannot change.

The scripts should fail loudly on missing speaker IDs, duplicate utterance IDs, non-NFC text, or speaker overlap.
