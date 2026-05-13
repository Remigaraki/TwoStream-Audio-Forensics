from __future__ import annotations

import torch
import torch.nn as nn

from src.stream1.sinc_conv import SincConv1D


class _Abs(nn.Module):
    """Element-wise absolute value — lets torch.abs sit inside nn.Sequential."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)


class _ResBlock(nn.Module):
    """
    Pre-activation residual block: BN → LeakyReLU → Conv → BN → LeakyReLU → Conv + skip.
    A 1×1 projection is added on the skip path when in_ch ≠ out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.BatchNorm1d(in_ch),
            nn.LeakyReLU(0.3),
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.LeakyReLU(0.3),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        )
        self.skip = (
            nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class RawNet2Encoder(nn.Module):
    """
    Stream 1 encoder.

    Architecture (adapted from Jung et al., "Improved RawNet with Feature Map Scaling
    for Text-Independent Speaker Verification using Raw Waveforms", Interspeech 2020,
    repurposed for countermeasure detection):

        Input  [B, 1, 64000]
        ↓  SincConv1D(128 filters, kernel=1025) → |·| → BN → LReLU → MaxPool(3)
        ↓  [B, 128, ~21333]
        ↓  ResBlock(128→128) → MaxPool(3)           [B, 128, ~7111]
        ↓  ResBlock(128→256) → MaxPool(3)           [B, 256, ~2370]
        ↓  ResBlock(256→256) → MaxPool(3)           [B, 256, ~790]
        ↓  ResBlock(256→256)                         [B, 256, ~790]
        ↓  GRU(input=256, hidden=1024) → last h_n   [B, 1024]
        ↓  FC(1024 → output_dim)                    [B, 256]  =  E_raw
    """

    def __init__(
        self,
        output_dim: int = 256,
        sinc_filters: int = 128,
        gru_hidden: int = 1024,
        sample_rate: int = 16000,
    ):
        super().__init__()

        self.sinc_frontend = nn.Sequential(
            SincConv1D(n_filters=sinc_filters, kernel_size=1024, sample_rate=sample_rate),
            _Abs(),
            nn.BatchNorm1d(sinc_filters),
            nn.LeakyReLU(0.3),
            nn.MaxPool1d(3),
        )

        self.res_blocks = nn.Sequential(
            _ResBlock(sinc_filters, sinc_filters),
            nn.MaxPool1d(3),
            _ResBlock(sinc_filters, sinc_filters * 2),
            nn.MaxPool1d(3),
            _ResBlock(sinc_filters * 2, sinc_filters * 2),
            nn.MaxPool1d(3),
            _ResBlock(sinc_filters * 2, sinc_filters * 2),
        )

        self.gru = nn.GRU(
            input_size=sinc_filters * 2,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )

        self.fc = nn.Linear(gru_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: waveform [B, T] or [B, 1, T]
        Returns:
            E_raw embedding [B, output_dim]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)          # [B, 1, T]

        x = self.sinc_frontend(x)       # [B, 128, T/3]
        x = self.res_blocks(x)          # [B, 256, T/81]
        x = x.transpose(1, 2)           # [B, T/81, 256]  — GRU expects (B, seq, feat)
        _, h_n = self.gru(x)            # h_n: [1, B, 1024]
        x = h_n.squeeze(0)              # [B, 1024]
        return self.fc(x)               # [B, output_dim=256]
