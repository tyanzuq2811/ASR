# Dataset sources and acquisition decision — 2026-10-20

## Locked sources

1. **VIVOS v1.0**: Zenodo record `7068130`, 1.5 GB, MD5 published by the record as `72972a8b14050f3f11ea7c3debabd7af`, license CC BY-NC-SA 4.0. The official test speakers remain held out.
2. **Common Voice Vietnamese 23.0**: Mozilla Data Collective, 413.72 MB, CC0. The official train/dev/test metadata exists, but this protocol uses only train/dev; Common Voice terms prohibit re-hosting/re-sharing and speaker identification.
3. **FLEURS Vietnamese**: `google/fleurs`, license CC BY 4.0. Load only the Vietnamese configuration and prefer streaming or selective materialization; the dataset card reports roughly 350 GB for all languages.
4. **MUSAN SLR17**: OpenSLR, 11 GB, CC BY 4.0. The archive is staged on `/hdd3`, verified with SHA-256 `86d1061c7e15b5c9e906777685c519701df51bfde3001e1070dcc9ffac955ee1`, extracted, and contains 930 validated 16 kHz noise WAV files.

## Acquisition gate

- Current A30 host storage: the large datasets and checkpoints are staged under
  `/hdd3/users/baolt`; `/` and `/hdd2` remain unsuitable for large artifacts.
- Account home is `/hdd3/users/baolt` with ample free space; VIVOS and MUSAN
  archives remain outside the Git repository.
- Do not start a multi-dataset download. The minimum safe first acquisition is VIVOS only, after confirming a dedicated destination and at least 3 GB working headroom.
- Common Voice must be obtained through the Mozilla Data Collective flow and must not be mirrored into GitHub.
- FLEURS should be streamed or selectively cached for Vietnamese evaluation; do not call `load_dataset("google/fleurs", "all")`.
- MUSAN allocation and checksum are now complete; the fixed-seed mixing rule and
  manifest hash are recorded with EXP-A08 noise-aware evidence.

## Expected verification artifacts

- `DATASET_MANIFEST.csv` updated with actual local path, byte size and SHA-256/MD5 after each download.
- `speaker_split.csv` and an overlap report after preprocessing.
- No audio or raw archive is committed to GitHub.
