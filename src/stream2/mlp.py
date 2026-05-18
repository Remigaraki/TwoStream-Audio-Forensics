"""StatisticalMLP: three-layer MLP for Stream 2 statistical features."""
from __future__ import annotations

import torch
import torch.nn as nn


class StatisticalMLP(nn.Module):
    """
    Three-layer MLP that maps statistical features to a compact embedding.

    Architecture
    ------------
    L1: Linear(input_dim, 256) -> BN(256) -> ReLU
    L2: Linear(256, 128)       -> BN(128) -> ReLU
    L3: Linear(128, output_dim)-> BN(output_dim) -> ReLU

    Parameters
    ----------
    input_dim  : feature dimension (default 248 = 128 PCA + 120 LFCC)
    output_dim : embedding dimension (default 64)
    """

    def __init__(self, input_dim: int = 248, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            # L1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            # L2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # L3
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, input_dim]
        Returns:
            [batch, output_dim]
        """
        return self.net(x)
