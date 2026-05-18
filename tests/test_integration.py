"""End-to-end integration tests for TwoStreamFusionNet.

Test1: Setup C forward -> BCE loss -> backward -> loss not NaN, grads not None
Test2: Setup A smoke test
Test3: Setup B smoke test
Test4: checkpoint save+load -> two forward passes -> outputs identical
Test5: CUDA test (skip if not available)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PCA_PATH = "data/pca_model/pca.pkl"
BATCH = 4
T = 64000


def _make_input():
    return torch.randn(BATCH, 1, T)


def _make_labels():
    return torch.randint(0, 2, (BATCH, 1)).float()


def _smoke_test(setup: str, pca_path=None):
    from src.fusion.two_stream_net import TwoStreamFusionNet

    model = TwoStreamFusionNet(pca_path=pca_path, setup=setup)
    model.train()
    x = _make_input()
    labels = _make_labels()

    out = model(x)
    loss = nn.BCELoss()(out, labels)
    loss.backward()

    assert not torch.isnan(loss), f"Setup {setup}: loss is NaN"
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert not torch.isnan(param.grad).any(), (
                f"Setup {setup}: NaN grad in {name}"
            )
    return loss


def test_setup_c_forward_backward():
    """Test1: Setup C forward -> BCE loss -> backward."""
    loss = _smoke_test("C", pca_path=PCA_PATH)
    assert loss.item() > 0, "Loss should be > 0"


def test_setup_a_smoke():
    """Test2: Setup A smoke test."""
    loss = _smoke_test("A", pca_path=None)
    assert loss.item() >= 0


def test_setup_b_smoke():
    """Test3: Setup B smoke test."""
    loss = _smoke_test("B", pca_path=PCA_PATH)
    assert loss.item() >= 0


def test_checkpoint_save_load():
    """Test4: save checkpoint -> load -> two forward passes -> outputs identical."""
    from src.fusion.two_stream_net import TwoStreamFusionNet

    model = TwoStreamFusionNet(pca_path=PCA_PATH, setup="C")
    model.eval()
    x = _make_input()

    with torch.no_grad():
        out1 = model(x)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "test.pt"
        torch.save({"model_state": model.state_dict()}, ckpt_path)

        model2 = TwoStreamFusionNet(pca_path=PCA_PATH, setup="C")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model2.load_state_dict(ckpt["model_state"])
        model2.eval()

        with torch.no_grad():
            out2 = model2(x)

    assert torch.allclose(out1, out2, atol=1e-6), (
        f"Outputs differ after checkpoint reload: max diff = {(out1 - out2).abs().max():.6f}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_forward():
    """Test5: move model and input to GPU, full forward pass."""
    from src.fusion.two_stream_net import TwoStreamFusionNet

    device = torch.device("cuda")
    model = TwoStreamFusionNet(pca_path=PCA_PATH, setup="C").to(device)
    model.eval()
    x = _make_input().to(device)

    with torch.no_grad():
        out = model(x)

    assert out.shape == (BATCH, 1), f"Expected ({BATCH}, 1), got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in CUDA output"
    assert out.device.type == "cuda", "Output not on CUDA"
