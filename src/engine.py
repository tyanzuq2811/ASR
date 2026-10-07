import time
import os
import torch
import torchaudio
import soundfile as sf
from pathlib import Path
from typing import Dict, Optional, Union
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from whisper_medusa import WhisperMedusaModel
from optimum.quanto import quantize, freeze, qint8, qint4

class BaseASREngine:
    def __init__(self, model_name: str, device: Optional[str] = None, fp16: bool = True):
        self.model_name = model_name
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.fp16 = fp16 and self.device.type == "cuda"
        self.dtype = torch.float16 if self.fp16 else torch.float32
        self.sampling_rate = 16000

    def preprocess_audio(self, audio_input: Union[str, Path, torch.Tensor], sr: Optional[int] = None) -> torch.Tensor:
        if isinstance(audio_input, (str, Path)):
            data, sample_rate = sf.read(str(audio_input))
            speech = torch.from_numpy(data).float()
            if speech.ndim == 1:
                speech = speech.unsqueeze(0)
            elif speech.ndim == 2:
                speech = speech.t()
        else:
            speech = audio_input
            sample_rate = sr if sr is not None else self.sampling_rate

        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)

        if sample_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.sampling_rate)
            speech = resampler(speech)

        return speech.squeeze()

    def get_model_size_mb(self, model: torch.nn.Module) -> float:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return round((param_size + buffer_size) / (1024 * 1024), 2)

class WhisperMedusaEngine(BaseASREngine):
    """
    Inference Engine using Whisper-Medusa multi-head speculative decoding
    with optional Post-Training Quantization (W8A16 or W4A16).
    """
    def __init__(
        self,
        model_name: str = "aiola/whisper-medusa-v1",
        quantization: str = "int8",  # 'none' (fp16), 'int8' (w8a16), 'int4' (w4a16)
        device: Optional[str] = None,
        regulation_start: float = 140,
        regulation_factor: float = 1.01,
        fp16: bool = True,
        **kwargs
    ):
        super().__init__(model_name, device, fp16=(quantization != "none" or fp16))
        self.quantization = quantization.lower() if quantization else "none"
        print(f"Loading Whisper-Medusa: {model_name} on {self.device} [Quantization: {self.quantization.upper()}]...")

        self.processor = WhisperProcessor.from_pretrained(model_name)
        
        # Load base model in CPU/FP16 first to avoid VRAM spikes
        self.model = WhisperMedusaModel.from_pretrained(model_name)

        if self.quantization == "int8":
            print("Applying Post-Training Quantization: W8A16 (Weights INT8, Activations FP16 via Quanto)...")
            quantize(self.model, weights=qint8, activations=None)
            freeze(self.model)
        elif self.quantization == "int4":
            print("Applying Post-Training Quantization: W4A16 (Weights INT4, Activations FP16 via Quanto)...")
            quantize(self.model, weights=qint4, activations=None)
            freeze(self.model)
        elif self.fp16:
            self.model.to(self.dtype)

        self.model.to(self.device)
        self.model.eval()

        self.model_size_mb = self.get_model_size_mb(self.model)
        print(f"Model successfully loaded. Estimated Memory Size: {self.model_size_mb} MB")

        self.regulation_start = regulation_start
        self.regulation_factor = regulation_factor

    @torch.no_grad()
    def transcribe(
        self,
        audio_input: Union[str, Path, torch.Tensor],
        language: Optional[str] = None,
        sr: Optional[int] = None
    ) -> Dict:
        speech = self.preprocess_audio(audio_input, sr)
        audio_duration_sec = len(speech) / self.sampling_rate

        inputs = self.processor(
            speech,
            return_tensors="pt",
            sampling_rate=self.sampling_rate
        )
        input_features = inputs.input_features.to(self.device)
        if self.fp16 and self.quantization == "none":
            input_features = input_features.to(self.dtype)

        decay_penalty = (
            (self.regulation_start, self.regulation_factor)
            if self.regulation_factor != 1.0
            else None
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start_time = time.perf_counter()
        else:
            start_time = time.perf_counter()

        generate_kwargs = {}
        if language is not None and language.strip():
            generate_kwargs["language"] = language

        model_output = self.model.generate(
            input_features,
            exponential_decay_length_penalty=decay_penalty,
            **generate_kwargs
        )

        if self.device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            peak_vram_mb = 0.0

        predict_ids = model_output[0]
        prediction_text = self.processor.decode(predict_ids, skip_special_tokens=True).strip()

        latency_sec = latency_ms / 1000.0
        rtf = round(latency_sec / audio_duration_sec, 4) if audio_duration_sec > 0 else 0.0

        return {
            "text": prediction_text,
            "latency_ms": round(latency_ms, 2),
            "latency_sec": round(latency_sec, 4),
            "audio_duration_sec": round(audio_duration_sec, 3),
            "rtf": rtf,
            "peak_vram_mb": round(peak_vram_mb, 2),
            "model_size_mb": self.model_size_mb,
            "quantization": self.quantization,
            "tokens_count": len(predict_ids),
        }

class VanillaWhisperEngine(BaseASREngine):
    """
    Standard autoregressive Whisper baseline with quantization support.
    """
    def __init__(
        self,
        model_name: str = "openai/whisper-large-v2",
        quantization: str = "none",  # 'none', 'int8', 'int4'
        device: Optional[str] = None,
        fp16: bool = True,
        **kwargs
    ):
        super().__init__(model_name, device, fp16=(quantization != "none" or fp16))
        self.quantization = quantization.lower() if quantization else "none"
        print(f"Loading Vanilla Whisper: {model_name} on {self.device} [Quantization: {self.quantization.upper()}]...")

        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name)

        if self.quantization == "int8":
            quantize(self.model, weights=qint8, activations=None)
            freeze(self.model)
        elif self.quantization == "int4":
            quantize(self.model, weights=qint4, activations=None)
            freeze(self.model)
        elif self.fp16:
            self.model.to(self.dtype)

        self.model.to(self.device)
        self.model.eval()
        self.model_size_mb = self.get_model_size_mb(self.model)

    @torch.no_grad()
    def transcribe(
        self,
        audio_input: Union[str, Path, torch.Tensor],
        language: Optional[str] = None,
        sr: Optional[int] = None
    ) -> Dict:
        speech = self.preprocess_audio(audio_input, sr)
        audio_duration_sec = len(speech) / self.sampling_rate

        inputs = self.processor(
            speech,
            return_tensors="pt",
            sampling_rate=self.sampling_rate
        )
        input_features = inputs.input_features.to(self.device)
        if self.fp16 and self.quantization == "none":
            input_features = input_features.to(self.dtype)

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            start_time = time.perf_counter()
        else:
            start_time = time.perf_counter()

        generate_kwargs = {}
        if language is not None and language.strip():
            generate_kwargs["language"] = language

        predicted_ids = self.model.generate(input_features, **generate_kwargs)

        if self.device.type == "cuda":
            end_event.record()
            torch.cuda.synchronize()
            latency_ms = start_event.elapsed_time(end_event)
            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            peak_vram_mb = 0.0

        prediction_text = self.processor.decode(predicted_ids[0], skip_special_tokens=True).strip()

        latency_sec = latency_ms / 1000.0
        rtf = round(latency_sec / audio_duration_sec, 4) if audio_duration_sec > 0 else 0.0

        return {
            "text": prediction_text,
            "latency_ms": round(latency_ms, 2),
            "latency_sec": round(latency_sec, 4),
            "audio_duration_sec": round(audio_duration_sec, 3),
            "rtf": rtf,
            "peak_vram_mb": round(peak_vram_mb, 2),
            "model_size_mb": self.model_size_mb,
            "quantization": self.quantization,
            "tokens_count": len(predicted_ids[0]),
        }
