"""
Integration tests for Stream 2: feature pipeline + StatisticalStream MLP.

Tests
-----
1. test_lfcc_shape             — LFCC extractor produces [120]-dim vector
2. test_bispectrum_shape       — bispectrum estimator produces [128, 128] matrix
3. test_pipeline_output_shape  — full pipeline waveform → [248]-dim forensic vector, no NaN
4. test_statistical_stream     — StatisticalStream: [B, 248] → [B, 64], no NaN, grads flow
5. test_end_to_end_shape       — waveform [B, 1, T] → pipeline → MLP → [B, 64]
"""

import numpy as np
import pytest
import torch

from src.stream2.extract_bispectrum import (
    Stream2ForensicFeaturePipeline,
    extract_lfcc_features,
    extract_bispectrum_matrix,
)
from src.fusion.two_stream_net import StatisticalStream

SR = 16000
T = 64000       # 4 s @ 16 kHz
BATCH = 2
RNG = np.random.default_rng(42)


def _rand_audio(n_samples: int = T) -> np.ndarray:
    return RNG.standard_normal(n_samples).astype(np.float32)


@pytest.fixture(scope="module")
def fitted_pipeline() -> Stream2ForensicFeaturePipeline:
    """Pipeline fitted on a small set of synthetic waveforms."""
    audios = [_rand_audio() for _ in range(10)]
    pipeline = Stream2ForensicFeaturePipeline()
    pipeline.fit(audios)
    return pipeline


# ---------------------------------------------------------------------------
# 1. LFCC shape
# ---------------------------------------------------------------------------

def test_lfcc_shape():
    audio = _rand_audio()
    vec = extract_lfcc_features(audio, sr=SR, n_lfcc=120)
    assert vec.shape == (120,), f"Expected (120,), got {vec.shape}"
    assert not np.isnan(vec).any(), "NaN in LFCC output"


# ---------------------------------------------------------------------------
# 2. Bispectrum shape
# ---------------------------------------------------------------------------

def test_bispectrum_shape():
    audio = _rand_audio()
    mat = extract_bispectrum_matrix(audio, sr=SR, bispectrum_bins=128)
    assert mat.shape == (128, 128), f"Expected (128, 128), got {mat.shape}"
    assert not np.isnan(mat).any(), "NaN in bispectrum matrix"


# ---------------------------------------------------------------------------
# 3. Full pipeline → forensic vector
# ---------------------------------------------------------------------------

def test_pipeline_output_shape(fitted_pipeline):
    """transform() must return a [248]-dim float32 vector with no NaN."""
    audio = _rand_audio()
    vec = fitted_pipeline.transform(audio, sr=SR)

    expected_dim = fitted_pipeline.lfcc_dim + fitted_pipeline.bispectrum_components  # 120+128=248
    assert vec.shape == (expected_dim,), f"Expected ({expected_dim},), got {vec.shape}"
    assert vec.dtype == np.float32
    assert not np.isnan(vec).any(), "NaN in pipeline output"


# ---------------------------------------------------------------------------
# 4. StatisticalStream MLP
# ---------------------------------------------------------------------------

def test_statistical_stream_shape_and_gradients():
    """StatisticalStream: [B, 248] → [B, 64], finite, grads flow to all params."""
    mlp = StatisticalStream(input_dim=248, output_dim=64)
    mlp.train()

    x = torch.randn(BATCH, 248)
    out = mlp(x)

    assert out.shape == (BATCH, 64), f"Expected ({BATCH}, 64), got {out.shape}"
    assert not out.isnan().any(), "NaN in StatisticalStream output"
    assert not out.isinf().any(), "Inf in StatisticalStream output"

    out.sum().backward()
    no_grad = [name for name, p in mlp.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters missing gradients: {no_grad}"


# ---------------------------------------------------------------------------
# 5. End-to-end: waveform → pipeline → MLP → E_stat
# ---------------------------------------------------------------------------

def test_end_to_end_shape(fitted_pipeline):
    """
    Full Stream 2 path:
        waveform [B, 1, T]  →  numpy transform  →  [B, 248]  →  StatisticalStream  →  [B, 64]
    No NaN anywhere, gradients flow through the MLP.
    """
    mlp = StatisticalStream(input_dim=248, output_dim=64)
    mlp.train()

    audios = [_rand_audio() for _ in range(BATCH)]
    stats_list = [fitted_pipeline.transform(a, sr=SR) for a in audios]
    stats = torch.from_numpy(np.stack(stats_list)).float()  # [B, 248]

    assert not stats.isnan().any(), "NaN in pipeline output before MLP"

    out = mlp(stats)
    assert out.shape == (BATCH, 64), f"Expected ({BATCH}, 64), got {out.shape}"
    assert not out.isnan().any(), "NaN in end-to-end output"

    out.sum().backward()
    no_grad = [name for name, p in mlp.named_parameters() if p.grad is None]
    assert not no_grad, f"MLP parameters missing gradients: {no_grad}"
