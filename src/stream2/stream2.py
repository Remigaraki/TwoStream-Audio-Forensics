"""Stream2: full second-stream module combining LFCC + bispectrum PCA + MLP.

CPU<->GPU transfer notes
------------------------
- LFCC is computed on GPU (torchaudio transform).
- Bispectrum estimation is pure numpy (CPU). Waveforms are moved to CPU
  numpy before this step, then PCA output is moved back to the model device.
- The MLP runs on GPU (or whichever device the module lives on).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.stream2.lfcc import LFCCExtractor
from src.stream2.bispectrum import estimate_bispectrum
from src.stream2.pca_compressor import BispectralPCA
from src.stream2.mlp import StatisticalMLP


class Stream2(nn.Module):
    """
    Full Stream 2 encoder.

    Input : [batch, 1, 64000] waveform tensor
    Output: [batch, 64]       E_stat embedding

    Processing pipeline
    -------------------
    Step 1  LFCC extraction on GPU           -> [batch, 120]
    Step 2  CPU bispectrum per sample + PCA  -> [batch, 128]  (numpy, CPU)
    Step 3  Move PCA output to device        -> [batch, 128]  (tensor, device)
    Step 4  Concatenate                       -> [batch, 248]
    Step 5  StatisticalMLP                   -> [batch, 64]
    """

    def __init__(
        self,
        pca: BispectralPCA | None = None,
        sample_rate: int = 16000,
        n_lfcc: int = 60,          # LFCCExtractor outputs 2*n_lfcc = 120
        pca_dim: int = 128,
        output_dim: int = 64,
    ):
        super().__init__()
        self.lfcc = LFCCExtractor(sample_rate=sample_rate, n_lfcc=n_lfcc)
        self.pca = pca  # BispectralPCA or None (identity / zero-pad if None)
        self.pca_dim = pca_dim
        total_dim = n_lfcc * 2 + pca_dim  # 120 + 128 = 248
        self.mlp = StatisticalMLP(input_dim=total_dim, output_dim=output_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bispectrum_pca(self, x: torch.Tensor) -> np.ndarray:
        """
        CPU path: waveform batch -> bispectrum per sample -> PCA projection.

        Parameters
        ----------
        x : [batch, 1, T] on any device

        Returns
        -------
        np.ndarray [batch, pca_dim]
        """
        # Move to CPU numpy — GPU->CPU transfer happens here
        x_np = x.squeeze(1).detach().cpu().numpy()   # [B, T]
        batch_vecs = []
        for i in range(x_np.shape[0]):
            b_mat = estimate_bispectrum(x_np[i])      # (128, 128)
            if self.pca is not None and self.pca.is_fitted:
                vec = self.pca.transform(b_mat)        # (128,)
            else:
                # Fallback: flatten and take first pca_dim elements
                flat = b_mat.reshape(-1).astype(np.float32)
                vec = flat[: self.pca_dim] if flat.size >= self.pca_dim else np.pad(
                    flat, (0, self.pca_dim - flat.size)
                )
            batch_vecs.append(vec)
        return np.stack(batch_vecs, axis=0)  # [B, pca_dim]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: raw waveform [batch, 1, T]
        Returns:
            E_stat [batch, 64]
        """
        device = x.device

        # Step 1 — LFCC on GPU
        lfcc_feat = self.lfcc(x)                     # [B, 120]

        # Step 2 — Bispectrum on CPU (numpy)
        bisp_np = self._bispectrum_pca(x)             # [B, 128] numpy

        # Step 3 — Move PCA output back to device
        bisp_feat = torch.from_numpy(bisp_np).float().to(device)  # [B, 128]

        # Step 4 — Concatenate
        combined = torch.cat([lfcc_feat, bisp_feat], dim=1)  # [B, 248]

        # Step 5 — MLP
        return self.mlp(combined)                    # [B, 64]
