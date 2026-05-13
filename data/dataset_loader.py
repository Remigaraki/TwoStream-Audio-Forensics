from pathlib import Path
from typing import Dict, Optional

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.utils.audio_utils import process_waveform
from src.stream2.extract_bispectrum import Stream2ForensicFeaturePipeline


class ASVspoofDataset(Dataset):
    """
    PyTorch dataset for ASVspoof 5 and WaveFake protocol files.
    """

    def __init__(
        self,
        base_dir,
        protocol_file,
        allowed_extensions=None,
        stream2_pipeline: Optional[Stream2ForensicFeaturePipeline] = None,
        return_stream2: bool = False,
        cache_stream2_features: bool = True,
        training: bool = True,
    ):
        self.base_dir = Path(base_dir)
        self.protocol_file = Path(protocol_file)
        self.allowed_extensions = allowed_extensions or (".flac", ".wav", ".ogg", ".mp3")
        self.stream2_pipeline = stream2_pipeline
        self.return_stream2 = return_stream2
        self.cache_stream2_features = cache_stream2_features
        self.training = training
        self.file_list = []
        self.labels = []
        self._stream2_cache: Dict[str, torch.Tensor] = {}

        with self.protocol_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                file_token = parts[1]
                label_token = next((token for token in reversed(parts) if token.lower() in {"bonafide", "spoof", "fake", "real"}), None)
                if label_token is None and len(parts) >= 5:
                    label_token = parts[4]

                if label_token is None:
                    continue

                self.file_list.append(file_token)
                self.labels.append(0 if label_token.lower() in {"bonafide", "real"} else 1)

    def __len__(self):
        return len(self.file_list)

    def _resolve_audio_path(self, file_token):
        candidate = self.base_dir / file_token
        if candidate.suffix:
            if candidate.exists():
                return candidate
        else:
            for extension in self.allowed_extensions:
                extended_candidate = candidate.with_suffix(extension)
                if extended_candidate.exists():
                    return extended_candidate

        if candidate.exists():
            return candidate

        tried_paths = [str(candidate)]
        if not candidate.suffix:
            tried_paths.extend(str(candidate.with_suffix(extension)) for extension in self.allowed_extensions)
        raise FileNotFoundError(f"Audio file not found for '{file_token}'. Tried: {tried_paths}")

    def __getitem__(self, idx):
        file_path = self._resolve_audio_path(self.file_list[idx])

        waveform_np, sample_rate = sf.read(file_path)
        waveform = torch.from_numpy(waveform_np).float()
        waveform = process_waveform(waveform, sample_rate, training=self.training)

        label = self.labels[idx]
        label_tensor = torch.tensor(label, dtype=torch.long)

        if not self.return_stream2:
            return waveform, label_tensor

        if self.stream2_pipeline is None:
            raise ValueError("return_stream2=True requires a fitted stream2_pipeline")

        cache_key = str(file_path)
        if self.cache_stream2_features and cache_key in self._stream2_cache:
            stats = self._stream2_cache[cache_key]
        else:
            stats_np = self.stream2_pipeline.transform(waveform.squeeze(0).cpu().numpy(), sr=16000)
            stats = torch.from_numpy(stats_np).float()
            if self.cache_stream2_features:
                self._stream2_cache[cache_key] = stats

        return waveform, stats, label_tensor


def fit_stream2_pipeline_from_protocol(
    base_dir,
    protocol_file,
    allowed_extensions=None,
    max_items: Optional[int] = None,
    random_seed: int = 42,
) -> Stream2ForensicFeaturePipeline:
    """
    Fit the Stream 2 PCA basis from standardized waveforms listed in a protocol file.
    """
    dataset = ASVspoofDataset(base_dir, protocol_file, allowed_extensions=allowed_extensions)
    sample_count = len(dataset) if max_items is None else min(len(dataset), max_items)
    if sample_count < 2:
        raise ValueError("At least two samples are required to fit the Stream 2 pipeline")

    audio_arrays = []
    sample_rates = []
    for index in range(sample_count):
        file_path = dataset._resolve_audio_path(dataset.file_list[index])
        waveform_np, sample_rate = sf.read(file_path)
        waveform = torch.from_numpy(waveform_np).float()
        waveform = process_waveform(waveform, sample_rate, training=False)
        audio_arrays.append(waveform.squeeze(0).cpu().numpy())
        sample_rates.append(16000)

    pipeline = Stream2ForensicFeaturePipeline(random_state=random_seed)
    pipeline.fit(audio_arrays, sample_rates)
    return pipeline
