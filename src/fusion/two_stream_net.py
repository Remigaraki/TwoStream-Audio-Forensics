"""TwoStreamFusionNet: combines RawNet2 (Stream1) and Stream2 via attention fusion.

Setups
------
A — Stream 1 only  (RawNet2Encoder → Linear(256,1) → Sigmoid)
B — Stream 2 only  (Stream2        → Linear(64,1)  → Sigmoid)
C — Full fusion    (both streams   → AttentionFusion → classifier → Sigmoid)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.stream1.rawnet2 import RawNet2Encoder
from src.fusion.attention_fusion import AttentionFusion
from src.stream2.stream2 import Stream2
from src.stream2.pca_compressor import BispectralPCA
from src.stream2.mlp import StatisticalMLP

# Backward-compatibility alias (was placeholder in original two_stream_net.py)
StatisticalStream = StatisticalMLP


class TwoStreamFusionNet(nn.Module):
    """
    The final combined PyTorch nn.Module.

    Parameters
    ----------
    pca_path : str | None
        Path to a pre-fitted BispectralPCA .pkl file.
        Required for setups B and C; ignored for setup A.
    setup : str
        'A' — Stream 1 only
        'B' — Stream 2 only
        'C' — Full fusion (default)
    """

    def __init__(self, pca_path: str | None = None, setup: str = "C"):
        super().__init__()
        self.setup = setup.upper()
        if self.setup not in ("A", "B", "C"):
            raise ValueError(f"setup must be 'A', 'B', or 'C', got '{setup}'")

        # ------------------------------------------------------------------
        # Stream 1 (always built for setup A and C)
        # ------------------------------------------------------------------
        if self.setup in ("A", "C"):
            self.stream1 = RawNet2Encoder(output_dim=256)

        # ------------------------------------------------------------------
        # Stream 2 (always built for setup B and C)
        # ------------------------------------------------------------------
        if self.setup in ("B", "C"):
            pca: BispectralPCA | None = None
            if pca_path is not None:
                pca = BispectralPCA.load(pca_path)
            self.stream2 = Stream2(pca=pca)

        # ------------------------------------------------------------------
        # Fusion + classifier (setup-dependent)
        # ------------------------------------------------------------------
        if self.setup == "A":
            # Stream 1 only: 256 → 1
            self.classifier = nn.Sequential(
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )

        elif self.setup == "B":
            # Stream 2 only: 64 → 1
            self.classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid(),
            )

        else:  # C — full fusion
            # AttentionFusion outputs [B, 256] (embed_dim=256)
            # We then concatenate stat_token [B, 64] → [B, 320] total
            # AttentionFusion.output maps embed_dim*2 → embed_dim internally;
            # its output is [B, 256].  We concatenate the stat projection
            # outside for the [B, 320] requirement.
            self.fusion = AttentionFusion(
                stream1_dim=256, stream2_dim=64, embed_dim=256
            )
            # After fusion output is [B, 256]; we concat s2 embedding [B,64] → [B,320]
            self.classifier = nn.Sequential(
                nn.Linear(320, 128),
                nn.BatchNorm1d(128),
                nn.SiLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )
            # Projection to produce the [B,64] token used for concat
            self._stat_proj = nn.Linear(64, 64)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw waveform [batch, 1, T]
        Returns:
            scores [batch, 1]  (sigmoid probabilities, fake=1)
        """
        if self.setup == "A":
            e_raw = self.stream1(x)          # [B, 256]
            return self.classifier(e_raw)    # [B, 1]

        if self.setup == "B":
            e_stat = self.stream2(x)         # [B, 64]
            return self.classifier(e_stat)   # [B, 1]

        # Setup C — full fusion
        e_raw = self.stream1(x)              # [B, 256]
        e_stat = self.stream2(x)             # [B, 64]
        fused = self.fusion(e_raw, e_stat)   # [B, 256]  (AttentionFusion output)
        combined = torch.cat([fused, e_stat], dim=1)  # [B, 320]
        return self.classifier(combined)     # [B, 1]
