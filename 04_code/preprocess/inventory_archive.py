#!/usr/bin/env python3
"""Inventory a speech archive without extracting audio.

The output is metadata only and is safe to commit after reviewing it.  It is
intentionally conservative: speaker names are reported as path candidates and
must be confirmed against the corpus documentation before creating splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(archive: Path) -> dict:
    top_level: set[str] = set()
    audio_members: list[str] = []
    candidate_speakers: set[str] = set()
    member_count = 0

    with tarfile.open(archive, mode="r:gz") as handle:
        for member in handle:
            member_count += 1
            parts = Path(member.name).parts
            if parts:
                top_level.add(parts[0])
            suffix = Path(member.name).suffix.lower()
            if not member.isfile() or suffix not in AUDIO_EXTENSIONS:
                continue
            audio_members.append(member.name)
            if "waves" in parts:
                waves_index = parts.index("waves")
                if waves_index + 1 < len(parts) - 1:
                    candidate_speakers.add(parts[waves_index + 1])

    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": sha256_file(archive),
        "member_count": member_count,
        "audio_member_count": len(audio_members),
        "top_level_entries": sorted(top_level),
        "candidate_speaker_count": len(candidate_speakers),
        "candidate_speakers": sorted(candidate_speakers),
        "sample_audio_members": audio_members[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = inventory(args.archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
