"""LFCCExtractor: torchaudio-based LFCC feature extractor (nn.Module)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio.transforms as T


class LFCCExtractor(nn.Module):
    """
    Extract LFCC features from raw waveforms using torchaudio.

    Input : [batch, 1, 64000]
    Output: [batch, 120]  (mean and std pooled along time, concatenated)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_filter: int = 128,
        n_lfcc: int = 60,
        dct_type: int = 2,
    ):
        super().__init__()
        self.lfcc_transform = T.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            dct_type=dct_type,
        )
        self.n_lfcc = n_lfcc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 1, T]
        Returns:
            [batch, 2*n_lfcc]  (mean + std along time axis, concat)
        """
        # x: [B, 1, T] -> squeeze channel -> [B, T]
        # Preserve [batch, channel, time] so decibel clipping is
        # independent for each recording.
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.dim() != 3 or x.shape[1] != 1:
            raise ValueError("Expected mono waveforms [batch, 1, time]")

        feats = self.lfcc_transform(x).squeeze(1)  # [B, n_lfcc, frames]

        mean = feats.mean(dim=2)   # [B, n_lfcc]
        std = feats.std(dim=2)     # [B, n_lfcc]

        return torch.cat([mean, std], dim=1)  # [B, 2*n_lfcc = 120]
