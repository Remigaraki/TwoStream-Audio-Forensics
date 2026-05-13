"""
Unit tests for Stream 1: SincConv1D and RawNet2Encoder.

Tests
-----
1. test_output_shape          — [B, T] and [B, 1, T] both produce [B, 256]
2. test_gradient_flow         — loss.backward() succeeds; sinc filter params get grads
3. test_no_nan                — no NaN in output for random waveform input
4. test_batch_invariance      — output for sample i in a batch equals output when run alone (eval mode)
"""

import pytest
import torch

from src.stream1.sinc_conv import SincConv1D
from src.stream1.rawnet2 import RawNet2Encoder

BATCH = 4
T = 64000  # 4 s @ 16 kHz


@pytest.fixture(scope="module")
def encoder():
    model = RawNet2Encoder(output_dim=256)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. Shape
# ---------------------------------------------------------------------------

def test_output_shape_2d(encoder):
    """Flat [B, T] input → [B, 256]."""
    x = torch.randn(BATCH, T)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (BATCH, 256), f"Expected ({BATCH}, 256), got {out.shape}"


def test_output_shape_3d(encoder):
    """Channel-first [B, 1, T] input → [B, 256]."""
    x = torch.randn(BATCH, 1, T)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape == (BATCH, 256), f"Expected ({BATCH}, 256), got {out.shape}"


# ---------------------------------------------------------------------------
# 2. Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow():
    """Gradients reach every learnable parameter, including the sinc filter banks."""
    model = RawNet2Encoder(output_dim=256)
    model.train()

    x = torch.randn(2, T)
    out = model(x)
    out.sum().backward()

    # Every parameter should have a gradient
    no_grad = [name for name, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"Parameters missing gradients: {no_grad}"

    # Specifically verify the sinc filter learnable parameters
    low_grad = model.sinc_frontend[0].low_hz_.grad
    band_grad = model.sinc_frontend[0].band_hz_.grad
    assert low_grad is not None and low_grad.abs().sum() > 0, "low_hz_ has zero/no gradient"
    assert band_grad is not None and band_grad.abs().sum() > 0, "band_hz_ has zero/no gradient"


# ---------------------------------------------------------------------------
# 3. No NaN
# ---------------------------------------------------------------------------

def test_no_nan(encoder):
    """Output must be finite for random unit-normal waveforms."""
    x = torch.randn(BATCH, T)
    with torch.no_grad():
        out = encoder(x)
    assert not out.isnan().any(), "NaN detected in encoder output"
    assert not out.isinf().any(), "Inf detected in encoder output"


# ---------------------------------------------------------------------------
# 4. Batch invariance
# ---------------------------------------------------------------------------

def test_batch_invariance(encoder):
    """
    In eval mode, the embedding for sample i must be identical whether the sample
    is processed alone or as part of a larger batch.
    BatchNorm uses running statistics in eval mode, so this must hold exactly.
    """
    torch.manual_seed(0)
    x = torch.randn(BATCH, T)

    with torch.no_grad():
        batch_out = encoder(x)
        solo_out = torch.cat([encoder(x[i].unsqueeze(0)) for i in range(BATCH)], dim=0)

    assert torch.allclose(batch_out, solo_out, atol=1e-5), (
        f"Batch-invariance violated — max diff: {(batch_out - solo_out).abs().max().item():.2e}"
    )
