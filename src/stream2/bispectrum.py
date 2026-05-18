"""Bispectrum estimator for deepfake audio detection.

estimate_bispectrum: pure numpy, per-sample bi-periodogram averaging.
BispectralBatchExtractor: wraps the above for batch numpy inputs.
"""
from __future__ import annotations

from typing import List

import numpy as np


def estimate_bispectrum(
    waveform: np.ndarray,
    window_size: int = 256,
    overlap: float = 0.5,
) -> np.ndarray:
    """
    Estimate the bispectrum magnitude via direct (bi-periodogram) averaging.

    Parameters
    ----------
    waveform   : 1-D float array, e.g. 64000 samples @ 16kHz
    window_size: FFT window length (default 256)
    overlap    : fractional overlap between consecutive windows (default 0.5)

    Returns
    -------
    B_hat : np.ndarray, shape (window_size//2, window_size//2)
        Averaged |B(f1, f2)| over all K segments. All values >= 0.
    """
    waveform = np.asarray(waveform, dtype=np.float64).ravel()
    half = window_size // 2          # 128 one-sided bins
    hop = max(1, int(window_size * (1.0 - overlap)))

    # Collect all frame start indices
    starts = list(range(0, len(waveform) - window_size + 1, hop))
    K = len(starts)
    if K == 0:
        return np.zeros((half, half), dtype=np.float32)

    B_sum = np.zeros((half, half), dtype=np.complex128)

    for start in starts:
        frame = waveform[start : start + window_size]
        # Hamming window to reduce spectral leakage
        frame = frame * np.hamming(window_size)
        Xk = np.fft.fft(frame)[:half]   # one-sided, shape (half,)

        # Bi-periodogram: B(i,j) = X[i] * X[j] * conj(X[i+j])
        for i in range(half):
            max_j = half - i
            if max_j <= 0:
                break
            js = np.arange(max_j)
            ij = i + js  # i+j indices, all < half
            B_sum[i, :max_j] += Xk[i] * Xk[js] * np.conj(Xk[ij])

    B_hat = np.abs(B_sum / K).astype(np.float32)
    return B_hat


class BispectralBatchExtractor:
    """Convenience class to extract bispectra for a list of waveforms."""

    def __init__(self, window_size: int = 256, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap

    def __call__(self, waveforms: List[np.ndarray]) -> np.ndarray:
        """
        Parameters
        ----------
        waveforms : list of 1-D float arrays

        Returns
        -------
        np.ndarray shape (N, window_size//2, window_size//2)
        """
        results = [
            estimate_bispectrum(w, self.window_size, self.overlap)
            for w in waveforms
        ]
        return np.stack(results, axis=0)
