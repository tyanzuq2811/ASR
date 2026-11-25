#!/usr/bin/env python3
"""Add deterministic NFC/lowercase transcript text to a CSV manifest."""

from __future__ import annotations

import argparse
import csv
import re
import unicodedata
from pathlib import Path


WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    value = WHITESPACE.sub(" ", value.strip())
    return value.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    parser.add_argument("--text-column", default="text")
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or args.text_column not in reader.fieldnames:
            raise SystemExit(f"Missing text column: {args.text_column}")
        fieldnames = list(reader.fieldnames)
        normalized_column = "text_normalized"
        if normalized_column not in fieldnames:
            fieldnames.append(normalized_column)
        rows = []
        for row in reader:
            row[normalized_column] = normalize_text(row.get(args.text_column, ""))
            rows.append(row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
