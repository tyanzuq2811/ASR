import re
import string
from typing import Dict, List, Tuple, Union
import jiwer

def normalize_text(text: Union[str, float], language: str = "en") -> str:
    """
    Standard text normalization for ASR evaluation (preserves Vietnamese accents).
    """
    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    # Remove standard punctuation and special characters, preserving letters and spaces
    # Preserve Vietnamese unicode characters: \w in Python 3 handles Vietnamese accents
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    
    # Replace multiple whitespaces with single space
    text = re.sub(r"\s+", " ", text).strip()
    return text

def calculate_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Computes global WER, CER, and Exact Match across a list of sentences.
    """
    norm_preds = [normalize_text(p) for p in predictions]
    norm_refs = [normalize_text(r) for r in references]

    # Handle edge case of empty strings
    valid_pairs = [(p, r) for p, r in zip(norm_preds, norm_refs) if len(r) > 0]
    if not valid_pairs:
        return {"wer": 0.0, "cer": 0.0, "exact_match": 100.0}

    filtered_preds, filtered_refs = zip(*valid_pairs)

    wer = jiwer.wer(reference=list(filtered_refs), hypothesis=list(filtered_preds))
    cer = jiwer.cer(reference=list(filtered_refs), hypothesis=list(filtered_preds))

    exact_matches = sum(1 for p, r in zip(filtered_preds, filtered_refs) if p == r)
    em_rate = (exact_matches / len(filtered_preds)) * 100.0

    return {
        "wer": round(float(wer) * 100.0, 2),
        "cer": round(float(cer) * 100.0, 2),
        "exact_match": round(float(em_rate), 2),
    }

def calculate_sample_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Computes sample-level WER, CER, and Exact Match.
    """
    norm_p = normalize_text(prediction)
    norm_r = normalize_text(reference)

    if not norm_r:
        return {"wer": 0.0 if not norm_p else 100.0, "cer": 0.0 if not norm_p else 100.0, "exact_match": 1 if norm_p == norm_r else 0}

    wer = jiwer.wer(reference=norm_r, hypothesis=norm_p)
    cer = jiwer.cer(reference=norm_r, hypothesis=norm_p)
    em = 1 if norm_p == norm_r else 0

    return {
        "wer": round(float(wer) * 100.0, 2),
        "cer": round(float(cer) * 100.0, 2),
        "exact_match": em,
    }

def calculate_rtf(latency_sec: float, audio_duration_sec: float) -> float:
    """
    Real-Time Factor (RTF) = Latency (s) / Audio Duration (s)
    RTF < 1.0 means faster than real-time.
    """
    if audio_duration_sec <= 0:
        return 0.0
    return round(float(latency_sec) / float(audio_duration_sec), 4)

def calculate_speedup(baseline_latency: float, test_latency: float) -> float:
    """
    Speedup ratio = Baseline Latency / Test Latency
    e.g. 1.5x means 50% faster.
    """
    if test_latency <= 0:
        return 1.0
    return round(float(baseline_latency) / float(test_latency), 2)
