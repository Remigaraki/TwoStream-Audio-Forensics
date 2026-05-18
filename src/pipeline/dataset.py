"""HearingRealityDataset: manifest-driven dataset for deepfake audio detection.

Each row in the manifest JSON/CSV describes one file:
    {"path": "relative/path.wav", "label": 0, "split": "train"}
Labels: 0 = real (bonafide), 1 = fake (spoof)

__getitem__ returns (waveform [1, 64000], label int)
collate_fn  returns ([batch, 1, 64000], [batch])
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from src.pipeline.augment import transcode, CODEC_CONDITIONS


class HearingRealityDataset(Dataset):
    """
    Manifest-driven dataset enforcing strict [1, 64000] waveforms.

    Parameters
    ----------
    manifest_path  : path to a JSON file containing a list of records:
                     [{"path": str, "label": int, "split": str}, ...]
    split          : 'train', 'val', or 'test'
    augment        : if True, randomly apply codec transcoding
    augment_prob   : probability of applying augmentation per sample
    target_sr      : target sample rate (16000)
    clip_duration  : clip length in seconds (4.0 -> 64000 samples)
    """

    TARGET_LENGTH: int = 64000

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        augment: bool = False,
        augment_prob: float = 0.6,
        target_sr: int = 16000,
        clip_duration: float = 4.0,
    ):
        self.manifest_path = Path(manifest_path)
        self.split = split
        self.augment = augment
        self.augment_prob = augment_prob
        self.target_sr = target_sr
        self.target_length = int(clip_duration * target_sr)  # 64000

        self._records: List[dict] = []
        self._load_manifest()

    # ------------------------------------------------------------------
    # Manifest loading
    # ------------------------------------------------------------------

    def _load_manifest(self) -> None:
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            all_records = json.load(fh)

        self._records = [r for r in all_records if r.get("split") == self.split]
        if not self._records:
            raise ValueError(
                f"No records found for split='{self.split}' in {self.manifest_path}"
            )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[Tensor, int]:
        record = self._records[idx]
        audio_path = Path(record["path"])
        label = int(record["label"])

        # Load with soundfile (avoids torchcodec dependency)
        waveform_np, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)   # [1, T]
        elif waveform.dim() == 2 and waveform.shape[1] != 1:
            # soundfile returns [T, C] for multi-channel; transpose then mean
            waveform = waveform.T  # [C, T]
        if sr != self.target_sr:
            waveform = F.resample(waveform, sr, self.target_sr)

        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Optional codec augmentation (works on file path, re-loads)
        if self.augment and torch.rand(1).item() < self.augment_prob:
            codec, bitrate = CODEC_CONDITIONS[
                torch.randint(len(CODEC_CONDITIONS), (1,)).item()
            ]
            try:
                aug_np = transcode(str(audio_path), codec, bitrate)
                waveform = torch.from_numpy(aug_np).unsqueeze(0)  # [1, N]
            except Exception:
                pass  # Fall back to original waveform on transcode failure

        # Enforce exactly [1, target_length]
        waveform = self._pad_or_trim(waveform)

        assert waveform.shape == (1, self.target_length), (
            f"Expected [1, {self.target_length}], got {waveform.shape}"
        )

        return waveform, label

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pad_or_trim(self, waveform: Tensor) -> Tensor:
        length = waveform.shape[-1]
        if length > self.target_length:
            waveform = waveform[:, : self.target_length]
        elif length < self.target_length:
            pad = self.target_length - length
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        return waveform

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Tensor, int]]
    ) -> Tuple[Tensor, Tensor]:
        """Stack waveforms and labels into batched tensors.

        Returns
        -------
        waveforms : [batch, 1, 64000]
        labels    : [batch]
        """
        waveforms = torch.stack([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
        return waveforms, labels
