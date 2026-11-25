#!/usr/bin/env python3
"""Export a fine-tuned Whisper checkpoint as encoder/decoder ONNX graphs.

The decoder graph is exported without a KV cache so it is a portable parity
artifact for the next QAIRT/QNN step. It consumes the complete decoder prefix
and returns logits for that prefix. Model weights and ONNX binaries belong in
the external handoff directory, not in Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


class EncoderWrapper(torch.nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.encoder = model.model.encoder

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_features=input_features, return_dict=True)
        return outputs.last_hidden_state


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model: WhisperForConditionalGeneration):
        super().__init__()
        self.decoder = model.model.decoder
        self.proj_out = model.proj_out

    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            return_dict=True,
        )
        return self.proj_out(outputs.last_hidden_state)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--decoder-length", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_graph(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    output_path: Path,
    input_names: list[str],
    output_names: list[str],
    dynamic_axes: dict[str, dict[int, str]],
    opset: int,
) -> None:
    torch.onnx.export(
        module,
        args,
        str(output_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )


def main() -> None:
    args = parse_args()
    if args.decoder_length < 1:
        raise SystemExit("--decoder-length must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    processor = WhisperProcessor.from_pretrained(args.model_id)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id)
    model.config.use_cache = False
    model.generation_config.use_cache = False
    model = model.to(device).eval()

    # Whisper-Tiny expects 80 mel bins and 3,000 audio frames. The four-token
    # prefix is the explicit Vietnamese/transcribe prompt used by evaluation.
    input_features = torch.zeros((1, 80, 3000), dtype=torch.float32, device=device)
    forced_prompt = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
    prompt_ids = [model.config.decoder_start_token_id] + [token_id for _, token_id in forced_prompt]
    prompt_ids = prompt_ids[: args.decoder_length]
    if len(prompt_ids) < args.decoder_length:
        prompt_ids.extend([model.config.eos_token_id] * (args.decoder_length - len(prompt_ids)))
    decoder_input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    # Use regular no_grad tensors for tracing. Inference-mode tensors cannot be
    # captured by the legacy ONNX tracer when they flow into the decoder graph.
    with torch.no_grad():
        encoder_hidden_states = EncoderWrapper(model)(input_features).detach().clone()

    encoder_path = args.output_dir / "encoder.onnx"
    decoder_path = args.output_dir / "decoder.onnx"
    encoder = EncoderWrapper(model).eval()
    decoder = DecoderWrapper(model).eval()
    export_graph(
        encoder,
        (input_features,),
        encoder_path,
        ["input_features"],
        ["encoder_hidden_states"],
        {"input_features": {0: "batch"}, "encoder_hidden_states": {0: "batch"}},
        args.opset,
    )
    export_graph(
        decoder,
        (decoder_input_ids, encoder_hidden_states),
        decoder_path,
        ["decoder_input_ids", "encoder_hidden_states"],
        ["logits"],
        {
            "decoder_input_ids": {0: "batch", 1: "decoder_sequence"},
            "encoder_hidden_states": {0: "batch"},
            "logits": {0: "batch", 1: "decoder_sequence"},
        },
        args.opset,
    )

    try:
        import onnx

        for path in (encoder_path, decoder_path):
            onnx.checker.check_model(onnx.load(path, load_external_data=False))
    except ImportError as exc:
        raise SystemExit("The onnx package is required for model validation") from exc

    metadata = {
        "model_id": args.model_id,
        "device": device,
        "opset": args.opset,
        "encoder_input_shape": [1, 80, 3000],
        "encoder_output_shape_sample": list(encoder_hidden_states.shape),
        "decoder_input_shape_sample": list(decoder_input_ids.shape),
        "decoder_output_shape_sample": [1, args.decoder_length, model.config.vocab_size],
        "decoder_use_cache": False,
        "language": "vi",
        "task": "transcribe",
        "prompt_token_ids": prompt_ids,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "files": {
            "encoder.onnx": {
                "bytes": encoder_path.stat().st_size,
                "sha256": sha256_file(encoder_path),
            },
            "decoder.onnx": {
                "bytes": decoder_path.stat().st_size,
                "sha256": sha256_file(decoder_path),
            },
        },
    }
    (args.output_dir / "export_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
