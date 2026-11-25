#!/usr/bin/env python3
"""Create a deterministic speaker-disjoint split from a metadata CSV.

The input must contain ``utterance_id``, ``speaker_id``, ``dataset`` and
``source_split``.  Official test/cross-corpus rows are preserved and never
sampled into training or validation.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


REQUIRED = {"utterance_id", "speaker_id", "dataset", "source_split"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = REQUIRED - fields
        if missing:
            raise SystemExit(f"Missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    ids = [row["utterance_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate utterance_id detected")
    if any(not row["speaker_id"].strip() for row in rows):
        raise SystemExit("Empty speaker_id detected")
    return rows


def assign_split(row: dict[str, str], validation_speakers: set[str]) -> str:
    dataset = row["dataset"].lower()
    source_split = row["source_split"].lower()
    speaker = row["speaker_id"]

    if dataset == "vivos" and source_split in {"test", "official_test"}:
        return "test_in_domain"
    if dataset == "vivos" and source_split in {"train", "training"}:
        return "validation" if speaker in validation_speakers else "train"
    if dataset.startswith("commonvoice") and source_split in {"train", "training"}:
        return "train"
    if dataset.startswith("commonvoice") and source_split in {"dev", "validation"}:
        return "validation"
    if dataset.startswith("commonvoice") and source_split == "test":
        return "excluded"
    if dataset.startswith("fleurs") and source_split == "test":
        return "test_cross_corpus"
    return "excluded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        raise SystemExit("validation-fraction must be between 0 and 1")

    rows = read_rows(args.input_csv)
    vivos_train_speakers = sorted(
        {
            row["speaker_id"]
            for row in rows
            if row["dataset"].lower() == "vivos"
            and row["source_split"].lower() in {"train", "training"}
        }
    )
    if len(vivos_train_speakers) < 2:
        raise SystemExit("At least two VIVOS train speakers are required")
    validation_count = max(1, round(len(vivos_train_speakers) * args.validation_fraction))
    validation_count = min(validation_count, len(vivos_train_speakers) - 1)
    rng = random.Random(args.seed)
    rng.shuffle(vivos_train_speakers)
    validation_speakers = set(vivos_train_speakers[:validation_count])

    output_fields = list(rows[0].keys()) + ["split"]
    output_fields = list(dict.fromkeys(output_fields))
    for row in rows:
        row["split"] = assign_split(row, validation_speakers)

    grouped = defaultdict(set)
    for row in rows:
        if row["split"] != "excluded":
            grouped[row["speaker_id"]].add(row["split"])
    overlap = {speaker: sorted(splits) for speaker, splits in grouped.items() if len(splits) > 1}
    if overlap:
        raise SystemExit(f"Speaker overlap detected: {overlap}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} validation_speakers={sorted(validation_speakers)}")


if __name__ == "__main__":
    main()
