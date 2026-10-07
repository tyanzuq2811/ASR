import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torchaudio
import soundfile as sf

class ASRDataset:
    """
    Manages loading, validating, and filtering the 200 real industrial audio dataset.
    """
    def __init__(self, base_dir: Optional[str] = None, manifest_path: Optional[str] = None):
        if base_dir is None:
            # Default to real200 directory relative to ASR root
            current_dir = Path(__file__).resolve().parent.parent
            self.base_dir = current_dir / "real200"
        else:
            self.base_dir = Path(base_dir)

        if manifest_path is None:
            self.manifest_path = self.base_dir / "manifest.csv"
        else:
            self.manifest_path = Path(manifest_path)

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest file not found at: {self.manifest_path}")

        self.df = pd.read_csv(self.manifest_path)
        self._prepare_data()

    def _prepare_data(self):
        """Resolves audio paths and calculates durations."""
        resolved_paths = []
        exists_flags = []

        for _, row in self.df.iterrows():
            # Check staged_path first
            staged = self.base_dir / str(row.get("staged_path", ""))
            if staged.exists():
                resolved_paths.append(str(staged))
                exists_flags.append(True)
            else:
                # Check direct audio_path or audio/language/id
                direct = self.base_dir / str(row.get("audio_path", ""))
                if direct.exists():
                    resolved_paths.append(str(direct))
                    exists_flags.append(True)
                else:
                    resolved_paths.append(str(staged))
                    exists_flags.append(False)

        self.df["resolved_audio_path"] = resolved_paths
        self.df["audio_exists"] = exists_flags
        
        # Ensure duration_sec
        if "duration_ms" in self.df.columns:
            self.df["duration_sec"] = self.df["duration_ms"] / 1000.0
        else:
            self.df["duration_sec"] = 0.0

    def get_summary(self) -> Dict:
        """Returns statistical overview of the dataset."""
        total_samples = len(self.df)
        valid_samples = self.df["audio_exists"].sum()
        total_duration = self.df["duration_sec"].sum()
        lang_counts = self.df["language"].value_counts().to_dict()
        domain_counts = self.df["domain"].value_counts().to_dict() if "domain" in self.df.columns else {}

        return {
            "total_samples": total_samples,
            "valid_samples": int(valid_samples),
            "total_duration_sec": round(float(total_duration), 2),
            "total_duration_min": round(float(total_duration) / 60.0, 2),
            "language_distribution": lang_counts,
            "domain_distribution": domain_counts,
        }

    def filter(self, language: Optional[str] = None, domain: Optional[str] = None, limit: Optional[int] = None) -> pd.DataFrame:
        """Filter dataset by language and/or domain."""
        sub_df = self.df[self.df["audio_exists"]].copy()

        if language and language.lower() != "all":
            sub_df = sub_df[sub_df["language"].str.lower() == language.lower()]

        if domain and domain.lower() != "all":
            sub_df = sub_df[sub_df["domain"].str.lower() == domain.lower()]

        if limit is not None and limit > 0:
            sub_df = sub_df.head(limit)

        return sub_df

    def load_audio(self, index: int, target_sr: int = 16000) -> Tuple[torch.Tensor, str, str, str]:
        """Loads single audio sample with metadata."""
        row = self.df.iloc[index]
        audio_path = str(row["resolved_audio_path"])
        data, sr = sf.read(audio_path)
        speech = torch.from_numpy(data).float()
        if speech.ndim == 1:
            speech = speech.unsqueeze(0)
        elif speech.ndim == 2:
            speech = speech.t()

        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)

        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            speech = resampler(speech)

        return speech, row["text"], row["language"], audio_path

if __name__ == "__main__":
    dataset = ASRDataset()
    print("Dataset Summary:")
    summary = dataset.get_summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")
