"""Demo CPU inference versus independent manifest evaluation, never threshold tuning.
Historical prediction CSVs are not read or modified. Reports fail closed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import soundfile as sf
import torch
from demo import inference as inf
from demo.artifact_identity import file_sha256, model_identity
from demo.calibration_resume import validate_scores
from src.train import _ManifestDataset
sys.path.insert(0, str(ROOT / 'scripts'))
from verify_manifest import compute_split_hash, REFERENCE_HASH
import pandas as pd

TOL = 1e-4


def spread_indices(records, limit):
    """Deterministic coverage across sorted IDs, stratified by class."""
    groups = [[i for i, r in enumerate(records) if int(r['label']) == lab] for lab in (0, 1)]
    selected = []
    for group in groups:
        count = min(len(group), limit // 2)
        selected.extend(group[i * (len(group) - 1) // (count - 1)]
                        for i in range(count)) if count > 1 else selected.extend(group[:count])
    return sorted(selected)


def compare_batches(model, audio, reference, device):
    checks = []
    count = len(audio)
    for reverse in (False, True):
        order = list(reversed(range(count))) if reverse else list(range(count))
        for batch_size in (4, 16, 32):
            values = torch.empty(count)
            with torch.inference_mode():
                for start in range(0, count, batch_size):
                    positions = order[start:start + batch_size]
                    values[positions] = model(audio[positions].to(device)).flatten().cpu()
            differences = (values - reference).abs()
            maximum = float(differences.max())
            checks.append({'order': 'reversed' if reverse else 'original', 'batch_size': batch_size,
                           'max_difference': maximum,
                           'passed': bool(torch.isfinite(values).all() and math.isfinite(maximum) and maximum <= TOL)})
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=sorted(inf.MODELS), required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data_root')
    ap.add_argument('--limit', type=int, default=256)
    ap.add_argument('--thresholds', required=True, help='Staged candidate; bound to report, not tuned.')
    ap.add_argument('--calibration_cache', required=True, help='Corrected C1 cache or preserved A1 cache.')
    ap.add_argument('--report', required=True)
    ap.add_argument('--evaluation_device', choices=['cpu', 'cuda', 'auto'], default='auto')
    args = ap.parse_args()
    if args.limit < 256:
        ap.error('At least 256 selected clips required; do not shrink coverage to avoid failures')
    current = model_identity(args.model, inf.MODELS[args.model])
    split_hash = compute_split_hash(pd.read_csv(args.manifest, usecols=['utterance_id', 'split']))
    if split_hash != REFERENCE_HASH:
        raise ValueError('Manifest split does not match evaluated data')
    thresholds = json.loads(Path(args.thresholds).read_text())
    if thresholds[args.model]['provenance'].get('identity') != current:
        raise ValueError('Candidate identity does not match current source and model')
    ds = _ManifestDataset(args.manifest, split='test', data_root=args.data_root)
    ds.records.sort(key=lambda r: r['utterance_id'])
    indices = spread_indices(ds.records, args.limit)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.evaluation_device == 'auto' else args.evaluation_device
    demo_model = inf.load_model(args.model).cpu().eval()
    evaluation_model = inf.load_model(args.model).to(device).eval()
    audio, scores, accepted_ids, rejected = [], [], [], []
    unexpected = 0
    for i in indices:
        row = ds.records[i]
        path = Path(row.get('file_path') or row['path'])
        if args.data_root:
            path = Path(args.data_root) / path.name
        info = sf.info(path)
        expected_short = info.duration < inf.MIN_DURATION_S
        try:
            with tempfile.TemporaryDirectory() as tmp:
                prepared = inf.prepare(path, Path(tmp))
                value, _ = inf.score(demo_model, prepared.x)
        except inf.InputError as exc:
            expected = expected_short and 'minimum is' in str(exc)
            unexpected += not expected
            rejected.append({'utterance_id': row['utterance_id'], 'reason': str(exc), 'expected_short': expected})
            continue
        if expected_short:
            unexpected += 1
        audio.append(ds[i][0])  # independent evaluation decoding/preprocessing
        scores.append(value)
        accepted_ids.append(row['utterance_id'])
    checks = compare_batches(evaluation_model, torch.stack(audio), torch.tensor(scores), device) if audio else []
    # Also establish that completed calibration scores match this code/version.
    val = _ManifestDataset(args.manifest, split='val', data_root=args.data_root)
    val.records.sort(key=lambda r: r['utterance_id'])
    cache = Path(args.calibration_cache) / f"val_{args.model}_{inf.MODELS[args.model]['checkpoint'][1][:12]}_clean.csv"
    expected_labels = {r['utterance_id']: int(r['label']) for r in val.records}
    validate_scores(cache, expected_labels)
    with cache.open() as f:
        saved = {r['utterance_id']: float(r['score']) for r in csv.DictReader(f)}
    clean_failures, clean_maximum = 0, 0.0
    val_indices = spread_indices(val.records, args.limit)
    with torch.inference_mode():
        for start in range(0, len(val_indices), 32):
            positions = val_indices[start:start + 32]
            values = evaluation_model(torch.stack([val[i][0] for i in positions]).to(device)).flatten().cpu().tolist()
            for i, value in zip(positions, values):
                difference = abs(value - saved[val.records[i]['utterance_id']])
                clean_failures += not math.isfinite(value) or difference > TOL
                clean_maximum = max(clean_maximum, difference)
    selected_label_counts = {str(lab): sum(int(ds.records[i]['label']) == lab for i in indices) for lab in (0, 1)}
    cache_label_counts = {str(lab): sum(int(val.records[i]['label']) == lab for i in val_indices) for lab in (0, 1)}
    passed = (len(indices) >= 256 and len(audio) >= 100 and not unexpected
              and all(count >= 128 for count in selected_label_counts.values())
              and all(count >= 128 for count in cache_label_counts.values())
              and len(val_indices) >= 256 and not clean_failures and len(checks) == 6
              and all(c['passed'] for c in checks))
    report = {'schema': 1, 'model': args.model, 'identity': current,
              'thresholds_sha256': file_sha256(args.thresholds), 'manifest_sha256': file_sha256(args.manifest),
              'manifest_split_hash': split_hash,
              'clean_cache_sha256': file_sha256(cache), 'split': 'test', 'tolerance': TOL,
              'selected': len(indices), 'compared': len(audio), 'selected_ids': [ds.records[i]['utterance_id'] for i in indices],
              'compared_ids': accepted_ids, 'rejections': rejected, 'unexpected_rejections': unexpected,
              'selected_label_counts': selected_label_counts, 'clean_cache_label_counts': cache_label_counts,
              'clean_cache_compared': len(val_indices), 'clean_cache_failures': clean_failures,
              'clean_cache_max_difference': clean_maximum, 'checks': checks, 'passed': passed,
              'torch': torch.__version__, 'evaluation_device': device,
              'note': 'No thresholds selected or changed; test scores used only for parity.'}
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{args.model}: selected={len(indices)}, compared={len(audio)}, rejected={len(rejected)}, unexpected={unexpected}")
    print(json.dumps(checks, indent=2))
    print(f"Calibration-cache matches: {len(val_indices)} checked, {clean_failures} failures")
    print('Parity:', 'PASS' if passed else 'FAIL', 'report:', output)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
