#!/usr/bin/env python3
"""Fine-tune Whisper on a speaker-disjoint CSV manifest.

The script deliberately keeps audio decoding on the fly.  It stores model
checkpoints outside the Git repository and records the exact run metadata in
the output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class VivosDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        data_root: Path,
        processor: WhisperProcessor,
        noise_root: Path | None = None,
    ):
        self.rows = rows
        self.data_root = data_root
        self.processor = processor
        self.noise_root = noise_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = self.data_root / audio_path
        audio, sample_rate = sf.read(audio_path, dtype="float32")
        if audio.ndim != 1:
            audio = audio.mean(axis=1)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz, got {sample_rate}: {audio_path}")
        if row.get("augmentation", "clean") != "clean":
            if self.noise_root is None:
                raise ValueError("Noise manifest row requires --noise-root")
            noise_path = Path(row["noise_path"])
            if not noise_path.is_absolute():
                noise_path = self.noise_root / noise_path
            noise, noise_rate = sf.read(noise_path, dtype="float32")
            if noise.ndim != 1:
                noise = noise.mean(axis=1)
            if noise_rate != 16000:
                raise ValueError(f"Expected 16 kHz noise, got {noise_rate}: {noise_path}")
            if len(noise) == 0:
                raise ValueError(f"Empty noise file: {noise_path}")
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
                raise ValueError(f"Near-silent noise file: {noise_path}")
            scale = signal_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
            audio = audio + (noise * scale).astype("float32")
            peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
            if peak > 0.999:
                audio = audio * (0.999 / peak)
        features = self.processor.feature_extractor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        ).input_features[0]
        labels = self.processor.tokenizer(row["text_normalized"]).input_ids
        return {"input_features": features, "labels": labels}


class WhisperDataCollator:
    def __init__(self, processor: WhisperProcessor, decoder_start_token_id: int):
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        input_features = torch.stack([f["input_features"] for f in features])
        label_features = [{"input_ids": f["labels"]} for f in features]
        padded_labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = padded_labels.input_ids
        # Whisper uses the same ID for padding and EOS (50257). Mask padding
        # by the tokenizer attention mask so the real EOS target is retained;
        # masking by token ID makes generation run to max_new_tokens and causes
        # repeated trailing words after fine-tuning.
        labels = labels.masked_fill(padded_labels.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all():
            labels = labels[:, 1:]
        return {"input_features": input_features, "labels": labels}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rows(manifest: Path) -> list[dict[str, str]]:
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"audio_path", "text_normalized", "speaker_id", "split"}
    missing = required - set(rows[0]) if rows else required
    if missing:
        raise SystemExit(f"Manifest missing columns: {', '.join(sorted(missing))}")
    return rows


def assert_disjoint(train_rows: list[dict[str, str]], eval_rows: list[dict[str, str]]) -> None:
    train_speakers = {row["speaker_id"] for row in train_rows}
    eval_speakers = {row["speaker_id"] for row in eval_rows}
    overlap = sorted(train_speakers & eval_speakers)
    if overlap:
        raise SystemExit(f"Speaker overlap between train/eval: {overlap}")


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(str(file_path.relative_to(path)).encode("utf-8"))
            with file_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--noise-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default="openai/whisper-tiny")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--eval-steps", type=int, default=250)
    parser.add_argument("--save-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-eval-rows", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.fp16 and args.bf16:
        raise SystemExit("Choose at most one of --fp16 and --bf16")
    if (args.fp16 or args.bf16) and not torch.cuda.is_available():
        raise SystemExit("Mixed precision requested but CUDA is unavailable")
    set_seed(args.seed)
    rows = read_rows(args.manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    eval_rows = [row for row in rows if row["split"] == "validation"]
    if args.max_train_rows > 0:
        train_rows = train_rows[: args.max_train_rows]
    if args.max_eval_rows > 0:
        eval_rows = eval_rows[: args.max_eval_rows]
    if not train_rows or not eval_rows:
        raise SystemExit("Manifest must contain non-empty train and validation splits")
    assert_disjoint(train_rows, eval_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Keep the decoder prefix used by labels identical to the prefix used at
    # inference. Without language/task here, tokenizer labels start with only
    # <|notimestamps|>, while generation starts with <|vi|><|transcribe|>,
    # which produces a silent train/decode contract mismatch.
    processor = WhisperProcessor.from_pretrained(
        args.model_id, language="vi", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "vi"
    model.generation_config.task = "transcribe"
    model.config.use_cache = False

    has_noise_rows = any(row.get("augmentation", "clean") != "clean" for row in rows)
    if has_noise_rows and args.noise_root is None:
        raise SystemExit("Manifest contains noise rows; pass --noise-root")
    train_dataset = VivosDataset(train_rows, args.data_root, processor, args.noise_root)
    eval_dataset = VivosDataset(eval_rows, args.data_root, processor, args.noise_root)
    collator = WhisperDataCollator(processor, model.config.decoder_start_token_id)
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=[],
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=processor,
    )
    run_config = vars(args).copy()
    run_config.update(
        {
            "train_rows": len(train_rows),
            "eval_rows": len(eval_rows),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "noise_root": str(args.noise_root) if args.noise_root else None,
            "augmentation_rows": sum(
                row.get("augmentation", "clean") != "clean" for row in rows
            ),
            "augmentation_conditions": sorted(
                {row.get("augmentation", "clean") for row in rows}
            ),
            "train_speakers": sorted({row["speaker_id"] for row in train_rows}),
            "eval_speakers": sorted({row["speaker_id"] for row in eval_rows}),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "language": "vi",
            "task": "transcribe",
            "label_prefix_tokens": processor.tokenizer.prefix_tokens,
            "started_unix": time.time(),
        }
    )
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    result = trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(str(args.output_dir / "final"))
    (args.output_dir / "train_result.json").write_text(
        json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    (args.output_dir / "checkpoint_tree_sha256.txt").write_text(
        sha256_tree(args.output_dir / "final") + "\n", encoding="utf-8"
    )
    print(json.dumps(result.metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
