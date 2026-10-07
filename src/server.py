import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure src in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.dataset import ASRDataset
from src.metrics import calculate_sample_metrics, calculate_speedup
from src.engine import WhisperMedusaEngine, VanillaWhisperEngine

app = FastAPI(title="ASR Edge AI Benchmark API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize dataset
dataset = ASRDataset(base_dir=str(root_dir / "real200"))

# Engine cache
engines: Dict[str, object] = {}

def get_medusa_engine_cached(model_name: str, quant: str):
    key = f"medusa_{model_name}_{quant}"
    if key not in engines:
        print(f"Loading {key} into memory...")
        engines[key] = WhisperMedusaEngine(model_name=model_name, quantization=quant)
    return engines[key]

def get_baseline_engine_cached(model_name: str, quant: str = "none"):
    key = f"baseline_{model_name}_{quant}"
    if key not in engines:
        print(f"Loading {key} into memory...")
        engines[key] = VanillaWhisperEngine(model_name=model_name, quantization=quant)
    return engines[key]

class TranscribeRequest(BaseModel):
    recording_id: str
    quantization: str = "int8"  # int8, int4, none
    medusa_model: str = "aiola/whisper-medusa-v1"
    baseline_model: str = "openai/whisper-large-v2"

@app.get("/api/status")
def get_system_status():
    cuda_avail = torch.cuda.is_available()
    if cuda_avail:
        free_mem, total_mem = torch.cuda.mem_get_info()
        gpu_name = torch.cuda.get_device_name(0)
        return {
            "device": "cuda",
            "gpu_name": gpu_name,
            "total_vram_gb": round(total_mem / (1024**3), 2),
            "free_vram_gb": round(free_mem / (1024**3), 2),
            "used_vram_mb": round((total_mem - free_mem) / (1024**2), 0),
        }
    return {
        "device": "cpu",
        "gpu_name": "CPU",
        "total_vram_gb": 0,
        "free_vram_gb": 0,
        "used_vram_mb": 0,
    }

@app.get("/api/samples")
def get_samples(
    language: Optional[str] = Query(None),
    domain: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
):
    df = dataset.filter(language=language, domain=domain)
    if search:
        search_lower = search.lower()
        df = df[df["text"].str.lower().str.contains(search_lower, na=False) |
                df["recording_id"].str.lower().str.contains(search_lower, na=False)]

    records = []
    for _, row in df.iterrows():
        records.append({
            "recording_id": row["recording_id"],
            "language": row["language"],
            "domain": row.get("domain", "N/A"),
            "text": row["text"],
            "duration_sec": round(float(row.get("duration_sec", 0.0)), 2),
        })
    return {"total": len(records), "samples": records}

@app.get("/api/audio/{recording_id}")
def get_audio_file(recording_id: str):
    matches = dataset.df[dataset.df["recording_id"] == recording_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Recording not found")
    row = matches.iloc[0]
    path = Path(row["resolved_audio_path"])
    if not path.exists():
        raise HTTPException(status_code=404, detail="Audio file missing on disk")
    return FileResponse(path, media_type="audio/wav")

@app.post("/api/transcribe")
def run_transcription(req: TranscribeRequest):
    matches = dataset.df[dataset.df["recording_id"] == req.recording_id]
    if matches.empty:
        raise HTTPException(status_code=404, detail="Sample not found")

    row = matches.iloc[0]
    audio_path = row["resolved_audio_path"]
    ground_truth = str(row["text"])
    lang = str(row["language"])
    duration = float(row.get("duration_sec", 0.0))

    try:
        medusa_eng = get_medusa_engine_cached(req.medusa_model, req.quantization)
        base_eng = get_baseline_engine_cached(req.baseline_model, "none")

        # 1. Medusa Inference
        res_medusa = medusa_eng.transcribe(audio_path, language=lang)
        m_medusa = calculate_sample_metrics(res_medusa["text"], ground_truth)

        # 2. Baseline Inference
        res_base = base_eng.transcribe(audio_path, language=lang)
        m_base = calculate_sample_metrics(res_base["text"], ground_truth)

        speedup = calculate_speedup(res_base["latency_ms"], res_medusa["latency_ms"])
        
        mem_reduction = 0.0
        if res_base["model_size_mb"] > 0:
            mem_reduction = round((1 - (res_medusa["model_size_mb"] / res_base["model_size_mb"])) * 100, 1)

        return {
            "recording_id": req.recording_id,
            "language": lang,
            "ground_truth": ground_truth,
            "audio_duration_sec": duration or res_medusa["audio_duration_sec"],
            "speedup_factor": speedup,
            "memory_reduction_pct": mem_reduction,
            "medusa": {
                "text": res_medusa["text"],
                "latency_ms": res_medusa["latency_ms"],
                "rtf": res_medusa["rtf"],
                "wer": m_medusa["wer"],
                "cer": m_medusa["cer"],
                "exact_match": m_medusa["exact_match"],
                "peak_vram_mb": res_medusa["peak_vram_mb"],
                "model_size_mb": res_medusa["model_size_mb"],
                "quantization": req.quantization.upper(),
            },
            "baseline": {
                "text": res_base["text"],
                "latency_ms": res_base["latency_ms"],
                "rtf": res_base["rtf"],
                "wer": m_base["wer"],
                "cer": m_base["cer"],
                "exact_match": m_base["exact_match"],
                "peak_vram_mb": res_base["peak_vram_mb"],
                "model_size_mb": res_base["model_size_mb"],
                "quantization": "FP16 (Gốc)",
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=True)
