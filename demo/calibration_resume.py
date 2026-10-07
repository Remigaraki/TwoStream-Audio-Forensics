"""Dependency-free safeguards for resuming corrected Kaggle calibration."""
import csv
import hashlib
import json
import math
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_tree(source, destination):
    source, destination = Path(source), Path(destination)
    if not source.is_dir():
        raise ValueError(f'Missing backup calibration tree: {source}')
    files = sorted(p for p in source.rglob('*') if p.is_file())
    # Preflight the entire tree before writing anything.
    for src in files:
        dst = destination / src.relative_to(source)
        if src.is_symlink() or not src.resolve().is_relative_to(source.resolve()):
            raise ValueError(f'Unsafe backup path: {src}')
        if dst.exists() and (not dst.is_file() or sha256(src) != sha256(dst)):
            raise ValueError(f'Restore conflict; preserve both versions and inspect: {dst}')
    for src in files:
        dst = destination / src.relative_to(source)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def validate_scores(path, expected):
    with Path(path).open(encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    ids = [r['utterance_id'] for r in rows]
    if len(ids) != len(expected) or len(set(ids)) != len(ids) or set(ids) != set(expected):
        raise ValueError(f'Incomplete or duplicate validation IDs: {path}')
    for row in rows:
        score = float(row['score'])
        if int(row['true_label']) != expected[row['utterance_id']] or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f'Invalid score or label: {path}, {row["utterance_id"]}')


def backup(cache, thresholds, manifest, output_dir):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(output_dir) / f'calibration_backup_{stamp}.zip'
    partial = output.with_suffix('.partial')
    with zipfile.ZipFile(partial, 'w', zipfile.ZIP_DEFLATED) as z:
        z.write(thresholds, 'demo_thresholds.json')
        z.write(manifest, 'manifest.csv')
        for path in sorted(Path(cache).rglob('*')):
            if path.is_file():
                z.write(path, 'calibration/' + path.relative_to(cache).as_posix())
        z.writestr('backup_status.json', json.dumps({
            'created_utc': stamp, 'parity_status': 'unverified',
            'note': 'Corrected codec calibration progress; publication disabled.'
        }, indent=2))
    with zipfile.ZipFile(partial) as z:
        if z.testzip() is not None:
            raise ValueError('Backup ZIP integrity failure')
    partial.replace(output)
    print(f'Backup saved: {output} SHA256={sha256(output)}', flush=True)
    return output
