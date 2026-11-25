#!/usr/bin/env python3
"""Create a deterministic mixed clean/MUSAN training manifest.

The clean VIVOS speaker split is never changed. Each train/validation row is
assigned exactly one condition from clean/20/10/5 dB; official test rows are
expanded to all four conditions for later evaluation. Audio is mixed on the
fly by the training/evaluation reader, so this script writes only metadata and
does not duplicate the VIVOS audio.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import soundfile as sf


CONDITIONS: tuple[str, ...] = ("clean", "20", "10", "5")


def stable_digest(seed: int, key: str) -> bytes:
    return hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()


def choose_condition(seed: int, key: str) -> str:
    digest = stable_digest(seed, key)
    return CONDITIONS[digest[0] % len(CONDITIONS)]


def choose_noise(noise_files: list[Path], seed: int, key: str) -> tuple[Path, int]:
    digest = stable_digest(seed, key)
    index = int.from_bytes(digest[:8], "big") % len(noise_files)
    noise_seed = int.from_bytes(digest[8:16], "big")
    return noise_files[index], noise_seed


def inspect_noise_files(noise_root: Path) -> list[Path]:
    noise_dir = noise_root / "noise"
    noise_files = sorted(path for path in noise_dir.rglob("*.wav") if path.is_file())
    if not noise_files:
        raise SystemExit(f"No MUSAN noise WAV files found below {noise_dir}")
    sample_rates: dict[int, int] = {}
    for path in noise_files:
        info = sf.info(path)
        sample_rates[info.samplerate] = sample_rates.get(info.samplerate, 0) + 1
    if set(sample_rates) != {16000}:
        raise SystemExit(f"MUSAN noise must be 16 kHz; observed {sample_rates}")
    return noise_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_files = inspect_noise_files(args.noise_root)
    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    required = {"utterance_id", "audio_path", "text_normalized", "speaker_id", "split"}
    missing = required - set(source_rows[0]) if source_rows else required
    if missing:
        raise SystemExit(f"Manifest missing columns: {', '.join(sorted(missing))}")

    output_rows: list[dict[str, str]] = []
    condition_counts: dict[str, dict[str, int]] = {}
    for source in source_rows:
        split = source["split"]
        if split in {"test_in_domain", "test_cross_corpus"}:
            conditions = CONDITIONS
        else:
            conditions = (choose_condition(args.seed, source["utterance_id"]),)
        for condition in conditions:
            row = dict(source)
            row["source_utterance_id"] = source["utterance_id"]
            row["augmentation"] = condition
            row["snr_db"] = "" if condition == "clean" else condition
            row["noise_path"] = ""
            row["noise_seed"] = ""
            if condition != "clean":
                noise_path, noise_seed = choose_noise(
                    noise_files, args.seed, f"{source['utterance_id']}:{condition}"
                )
                row["noise_path"] = noise_path.relative_to(args.noise_root).as_posix()
                row["noise_seed"] = str(noise_seed)
            output_rows.append(row)
            split_counts = condition_counts.setdefault(split, {})
            split_counts[condition] = split_counts.get(condition, 0) + 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(source_rows[0]) + [
        "source_utterance_id",
        "augmentation",
        "snr_db",
        "noise_path",
        "noise_seed",
    ]
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "source_manifest": str(args.manifest),
        "noise_root": str(args.noise_root),
        "output_manifest": str(args.output),
        "seed": args.seed,
        "conditions": list(CONDITIONS),
        "noise_files": len(noise_files),
        "noise_sample_rate_hz": 16000,
        "source_rows": len(source_rows),
        "output_rows": len(output_rows),
        "rows_by_split_condition": condition_counts,
    }
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
