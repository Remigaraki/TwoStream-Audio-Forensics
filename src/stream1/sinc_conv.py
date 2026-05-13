from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SincConv1D(nn.Module):
    """
    Learnable sinc-function bandpass filter bank operating on raw waveforms.

    Each of the n_filters filters is parameterized by two scalars:
        low_hz_  — lower cutoff frequency (Hz)
        band_hz_ — bandwidth (Hz)
    Both are kept positive during the forward pass via torch.abs(), so
    gradients flow freely while the physical constraints are always satisfied.

    The ideal bandpass impulse response is:
        h(t) = 2*f2*sinc(2*f2*t) − 2*f1*sinc(2*f1*t)
    where f1 = low_hz, f2 = low_hz + band_hz (in normalised units, i.e. / sample_rate).
    A Hamming window is applied to make the filters finite and reduce spectral leakage.
    """

    def __init__(
        self,
        n_filters: int = 128,
        kernel_size: int = 1024,
        sample_rate: int = 16000,
        min_low_hz: float = 50.0,
        min_band_hz: float = 50.0,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1  # sinc filters must be symmetric (odd length)

        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # Initialise lower cutoffs uniformly across the audible range
        low_hz = torch.linspace(min_low_hz, sample_rate / 2.0 - min_band_hz - 1.0, n_filters)
        self.low_hz_ = nn.Parameter(low_hz)

        # Initialise all bandwidths at the minimum — they grow during training
        self.band_hz_ = nn.Parameter(torch.full((n_filters,), min_band_hz))

        # Time axis in seconds — registered as buffer so it moves to the right device
        n = (kernel_size - 1) // 2
        t = torch.linspace(-n, n, kernel_size, dtype=torch.float32) / sample_rate
        self.register_buffer("t_", t)

        # Hamming window — fixed, not learned
        self.register_buffer("window_", torch.hamming_window(kernel_size, periodic=False))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sinc(x: torch.Tensor) -> torch.Tensor:
        """Normalised sinc: sin(πx) / (πx), with sinc(0) = 1."""
        x = torch.where(x == 0.0, torch.full_like(x, 1e-20), x)
        return torch.sin(math.pi * x) / (math.pi * x)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw waveform tensor of shape [B, 1, T]
        Returns:
            Filter-bank output of shape [B, n_filters, T]  (same-length via padding)
        """
        low = self.min_low_hz + torch.abs(self.low_hz_)                          # [F]
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            max=self.sample_rate / 2.0 - 1.0,
        )                                                                          # [F]

        t = self.t_                                                                # [K]

        # Build bandpass filters: h(t) = 2f2·sinc(2f2·t) − 2f1·sinc(2f1·t)
        # Shapes: low/high unsqueeze(1) → [F,1]; t unsqueeze(0) → [1,K] → broadcast [F,K]
        f1 = 2.0 * low.unsqueeze(1) * self._sinc(2.0 * low.unsqueeze(1) * t.unsqueeze(0))
        f2 = 2.0 * high.unsqueeze(1) * self._sinc(2.0 * high.unsqueeze(1) * t.unsqueeze(0))

        filters = (f2 - f1) * self.window_.unsqueeze(0)  # [F, K]
        filters = filters.unsqueeze(1)                    # [F, 1, K]  (in_channels=1)

        return F.conv1d(x, filters, padding=self.kernel_size // 2)
