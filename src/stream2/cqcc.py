"""CQCCExtractor: Constant-Q Cepstral Coefficients, computed via librosa CQT + DCT.

CPU path (librosa has no GPU backend), mirroring the bispectrum CPU step in
Stream2: waveform batch -> per-sample CQT -> log-magnitude -> DCT -> mean/std
pooled cepstral coefficients.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.fftpack import dct
import librosa


class CQCCExtractor(nn.Module):
    """
    Input : [batch, 1, T] waveform tensor
    Output: [batch, 2*n_cqcc]  (mean and std pooled along time, concatenated)
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_cqcc: int = 60,
        hop_length: int = 256,
        fmin: float = 32.7,
        n_bins: int = 84,
        bins_per_octave: int = 12,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_cqcc = n_cqcc
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        x_np = x.squeeze(1).detach().cpu().numpy()  # [B, T]

        feats = []
        for i in range(x_np.shape[0]):
            cqt = librosa.cqt(
                x_np[i],
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=self.fmin,
                n_bins=self.n_bins,
                bins_per_octave=self.bins_per_octave,
            )
            log_mag = np.log(np.abs(cqt) + 1e-6)
            cqcc = dct(log_mag, type=2, axis=0, norm="ortho")[: self.n_cqcc, :]
            mean = cqcc.mean(axis=1)
            std = cqcc.std(axis=1)
            feats.append(np.concatenate([mean, std]).astype(np.float32))

        out_np = np.stack(feats, axis=0)  # [B, 2*n_cqcc]
        return torch.from_numpy(out_np).float().to(device)
