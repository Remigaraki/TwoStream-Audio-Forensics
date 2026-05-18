"""Tests for TwoStreamFusionNet (all three setups).

Test1: Setup C [4,1,64000] -> [4,1] output, no NaN
Test2: Setup A [4,1,64000] -> [4,1] output, no NaN
Test3: Setup B [4,1,64000] -> [4,1] output, no NaN
Test4: BCE loss + backward() on Setup C -- no error
Test5: all outputs between 0 and 1
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

PCA_PATH = "data/pca_model/pca.pkl"
BATCH = 4
T = 64000


def _make_input():
    return torch.randn(BATCH, 1, T)


@pytest.fixture(scope="module")
def model_c():
    from src.fusion.two_stream_net import TwoStreamFusionNet
    return TwoStreamFusionNet(pca_path=PCA_PATH, setup="C").eval()


@pytest.fixture(scope="module")
def model_a():
    from src.fusion.two_stream_net import TwoStreamFusionNet
    return TwoStreamFusionNet(pca_path=None, setup="A").eval()


@pytest.fixture(scope="module")
def model_b():
    from src.fusion.two_stream_net import TwoStreamFusionNet
    return TwoStreamFusionNet(pca_path=PCA_PATH, setup="B").eval()


def test_setup_c_shape_no_nan(model_c):
    """Test1: Setup C [4,1,64000] -> [4,1], no NaN."""
    x = _make_input()
    with torch.no_grad():
        out = model_c(x)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in Setup C output"


def test_setup_a_shape_no_nan(model_a):
    """Test2: Setup A [4,1,64000] -> [4,1], no NaN."""
    x = _make_input()
    with torch.no_grad():
        out = model_a(x)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in Setup A output"


def test_setup_b_shape_no_nan(model_b):
    """Test3: Setup B [4,1,64000] -> [4,1], no NaN."""
    x = _make_input()
    with torch.no_grad():
        out = model_b(x)
    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in Setup B output"


def test_backward_setup_c():
    """Test4: BCE loss + backward() on Setup C -- no error."""
    from src.fusion.two_stream_net import TwoStreamFusionNet

    model = TwoStreamFusionNet(pca_path=PCA_PATH, setup="C")
    model.train()
    x = _make_input()
    labels = torch.randint(0, 2, (BATCH, 1)).float()

    out = model(x)
    loss = nn.BCELoss()(out, labels)
    loss.backward()
    assert not torch.isnan(loss), "Loss is NaN after backward"


def test_outputs_in_range(model_a, model_b, model_c):
    """Test5: all outputs between 0 and 1."""
    x = _make_input()
    with torch.no_grad():
        for name, model in [("A", model_a), ("B", model_b), ("C", model_c)]:
            out = model(x)
            assert (out >= 0).all() and (out <= 1).all(), (
                f"Setup {name}: outputs outside [0,1]: min={out.min():.4f} max={out.max():.4f}"
            )
