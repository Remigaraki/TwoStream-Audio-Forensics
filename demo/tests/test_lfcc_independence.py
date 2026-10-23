"""Mixed amplitudes expose batch-wide decibel clipping in the old extractor."""
import pytest
import torch
from src.stream2.lfcc import LFCCExtractor


def audio():
    generator = torch.Generator().manual_seed(73)
    x = torch.randn(8, 1, 64000, generator=generator)
    return x * torch.tensor([1., .1, .01, .001, .5, .05, .005, .0005])[:, None, None]


@pytest.mark.parametrize('batch_size', [2, 4, 8])
@pytest.mark.parametrize('reverse', [False, True])
def test_lfcc_independent_of_batch_and_order(batch_size, reverse):
    torch.set_num_threads(2)
    extractor = LFCCExtractor().eval()
    x = audio()
    order = list(reversed(range(len(x)))) if reverse else list(range(len(x)))
    with torch.inference_mode():
        single = torch.cat([extractor(item[None]) for item in x])
        batched = torch.empty_like(single)
        for start in range(0, len(x), batch_size):
            positions = order[start:start + batch_size]
            batched[positions] = extractor(x[positions])
    # Feature tolerance accommodates float32 transform kernels; score tolerance
    # is tested separately with the actual checkpoint, at the unchanged 1e-4.
    torch.testing.assert_close(batched, single, atol=3e-4, rtol=1e-5)


def test_two_dimensional_input_matches_mono_channel_input():
    extractor = LFCCExtractor().eval()
    x = audio()[:2]
    torch.testing.assert_close(extractor(x), extractor(x[:, 0]))


@pytest.mark.parametrize('shape', [(2, 2, 64000), (64000,), (1, 1, 1, 64000)])
def test_invalid_channel_layout_rejected(shape):
    with pytest.raises(ValueError, match='mono'):
        LFCCExtractor()(torch.zeros(shape))


@pytest.mark.parametrize('name', ['C1', 'A1'])
def test_checkpoint_scores_independent_of_batch_and_order(name):
    from demo import inference as inf
    model = inf.load_model(name).eval()
    x = audio().repeat(4, 1, 1)
    with torch.inference_mode():
        single = torch.cat([model(item[None]).flatten() for item in x])
        for reverse in (False, True):
            order = list(reversed(range(len(x)))) if reverse else list(range(len(x)))
            for batch_size in (4, 16, 32):
                values = torch.empty_like(single)
                for start in range(0, len(x), batch_size):
                    positions = order[start:start + batch_size]
                    values[positions] = model(x[positions]).flatten()
                assert torch.isfinite(values).all()
                assert float((values - single).abs().max()) <= 1e-4
