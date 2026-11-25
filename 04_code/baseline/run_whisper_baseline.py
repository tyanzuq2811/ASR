#!/usr/bin/env python3
"""Run a reproducible Whisper-Tiny reference evaluation from a CSV manifest.

This script writes text/timing metadata only. It never copies audio or model
weights into the repository. Use ``--limit`` for a CPU smoke test; a reported
baseline must use the complete held-out split and a recorded device.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


WHITESPACE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFC", value or "")
    return WHITESPACE.sub(" ", value.strip()).lower()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--split", default="test_in_domain")
    parser.add_argument(
        "--noise-root",
        type=Path,
        default=None,
        help="MUSAN root for manifests with augmentation/noise_path columns",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--limit", type=int, default=0, help="0 means the complete split")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("split") == args.split]
    if not rows:
        raise SystemExit(f"No rows found for split={args.split!r}")
    if args.limit > 0:
        rows = rows[: args.limit]
    has_noise_rows = any(row.get("augmentation", "clean") != "clean" for row in rows)
    if has_noise_rows and args.noise_root is None:
        raise SystemExit("Manifest contains noise rows; pass --noise-root")

    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id).to(device)
    model.eval()
    # Use the explicit language/task prompt below; do not silently combine it
    # with a checkpoint-level forced-decoder configuration.
    model.config.forced_decoder_ids = None
    model.generation_config.language = None
    model.generation_config.task = None
    model.generation_config.forced_decoder_ids = None
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")

    result_rows = []
    references = []
    hypotheses = []

    def load_audio(row: dict[str, str]) -> tuple[np.ndarray, int]:
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = args.data_root / audio_path
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim != 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise SystemExit(f"Unexpected sample rate {sample_rate} for {audio_path}")
        if row.get("augmentation", "clean") != "clean":
            if args.noise_root is None:
                raise SystemExit("Noise manifest row requires --noise-root")
            noise_path = Path(row["noise_path"])
            if not noise_path.is_absolute():
                noise_path = args.noise_root / noise_path
            noise, noise_rate = sf.read(noise_path, dtype="float32")
            if noise.ndim != 1:
                noise = noise.mean(axis=1)
            if noise_rate != 16000:
                raise SystemExit(f"Unexpected noise sample rate {noise_rate} for {noise_path}")
            if len(noise) == 0:
                raise SystemExit(f"Empty noise file: {noise_path}")
            rng = np.random.default_rng(int(row["noise_seed"]))
            if len(noise) < len(audio):
                start = int(rng.integers(0, len(noise)))
                tiled = np.tile(noise, (len(audio) // len(noise)) + 2)
                noise = np.roll(tiled, -start)[: len(audio)]
            else:
                start = int(rng.integers(0, len(noise) - len(audio) + 1))
                noise = noise[start : start + len(audio)]
            snr_db = float(row["snr_db"])
            signal_rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-12))
            noise_rms = float(np.sqrt(np.mean(np.square(noise)) + 1e-12))
            if noise_rms <= 1e-6:
                raise SystemExit(f"Near-silent noise file: {noise_path}")
            scale = signal_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
            audio = audio + (noise * scale).astype("float32")
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 0.999:
                audio = audio * (0.999 / peak)
        return audio, sample_rate

    for index, row in enumerate(rows, start=1):
        audio, sample_rate = load_audio(row)
        inputs = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            return_attention_mask=True,
        )
        input_features = inputs.input_features.to(device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        started = time.perf_counter()
        with torch.inference_mode():
            generate_kwargs = {
                "input_features": input_features,
                "forced_decoder_ids": forced_decoder_ids,
                "do_sample": False,
                "num_beams": 1,
                "max_new_tokens": args.max_new_tokens,
            }
            if attention_mask is not None:
                generate_kwargs["attention_mask"] = attention_mask
            generated = model.generate(**generate_kwargs)
        elapsed = time.perf_counter() - started
        hypothesis = processor.batch_decode(generated, skip_special_tokens=True)[0]
        reference = row.get("text_normalized") or normalize_text(row.get("text_raw", ""))
        hypothesis = normalize_text(hypothesis)
        references.append(reference)
        hypotheses.append(hypothesis)
        duration = len(audio) / sample_rate
        result_rows.append(
            {
                "utterance_id": row["utterance_id"],
                "speaker_id": row["speaker_id"],
                "split": args.split,
                "augmentation": row.get("augmentation", "clean"),
                "snr_db": row.get("snr_db", ""),
                "noise_path": row.get("noise_path", ""),
                "noise_seed": row.get("noise_seed", ""),
                "reference": reference,
                "hypothesis": hypothesis,
                "audio_duration_s": f"{duration:.6f}",
                "inference_s": f"{elapsed:.6f}",
                "rtf": f"{elapsed / duration:.6f}",
                "device": device,
                "model_id": args.model_id,
            }
        )
        if index == 1 or index % 10 == 0 or index == len(rows):
            print(f"processed={index}/{len(rows)} device={device}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        fields = list(result_rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(result_rows)

    summary = {
        "model_id": args.model_id,
        "split": args.split,
        "device": device,
        "seed": args.seed,
        "rows": len(result_rows),
        "wer": jiwer.wer(references, hypotheses),
        "cer": jiwer.cer(references, hypotheses),
        "median_inference_s": float(torch.tensor([float(r["inference_s"]) for r in result_rows]).median()),
        "median_rtf": float(torch.tensor([float(r["rtf"]) for r in result_rows]).median()),
        "total_audio_s": sum(float(r["audio_duration_s"]) for r in result_rows),
        "total_inference_s": sum(float(r["inference_s"]) for r in result_rows),
    }
    condition_summaries = {}
    for condition in sorted({r["augmentation"] for r in result_rows}):
        condition_rows = [r for r in result_rows if r["augmentation"] == condition]
        condition_references = [r["reference"] for r in condition_rows]
        condition_hypotheses = [r["hypothesis"] for r in condition_rows]
        condition_summaries[condition] = {
            "rows": len(condition_rows),
            "wer": jiwer.wer(condition_references, condition_hypotheses),
            "cer": jiwer.cer(condition_references, condition_hypotheses),
            "median_inference_s": float(
                torch.tensor([float(r["inference_s"]) for r in condition_rows]).median()
            ),
            "median_rtf": float(
                torch.tensor([float(r["rtf"]) for r in condition_rows]).median()
            ),
        }
    summary["conditions"] = condition_summaries
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
