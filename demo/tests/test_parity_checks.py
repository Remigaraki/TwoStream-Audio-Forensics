import torch
from demo.check_parity import compare_batches, spread_indices


def test_selection_is_spread_across_both_classes():
    rows = [{'label': str(i % 2), 'utterance_id': str(i)} for i in range(1000)]
    selected = spread_indices(rows, 256)
    assert len(selected) == len(set(selected)) == 256
    assert selected[0] == 0 and selected[-1] == 999
    assert sum(int(rows[i]['label']) for i in selected) == 128


def test_comparison_catches_batch_dependence_and_nan():
    x = torch.arange(32, dtype=torch.float32).reshape(32, 1, 1)
    reference = x.flatten()
    independent = lambda x: x.flatten()
    assert all(c['passed'] for c in compare_batches(independent, x, reference, 'cpu'))
    dependent = lambda x: x.flatten() + x.mean()
    assert not all(c['passed'] for c in compare_batches(dependent, x, reference, 'cpu'))
    invalid = lambda x: x.flatten() * float('nan')
    assert not any(c['passed'] for c in compare_batches(invalid, x, reference, 'cpu'))
