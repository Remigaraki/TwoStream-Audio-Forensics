"""Content identities and fail-closed verification, independent of torch."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
COMMON_SOURCES = ['demo/inference.py', 'demo/decode.py', 'demo/artifact_identity.py',
                  'demo/check_parity.py',
                  'src/train.py', 'src/pipeline/augment.py',
                  'src/fusion/two_stream_net.py', 'src/fusion/attention_fusion.py',
                  'src/stream1/rawnet2.py', 'src/stream1/sinc_conv.py']
C1_SOURCES = ['src/stream2/lfcc.py', 'src/stream2/stream2.py',
              'src/stream2/bispectrum.py', 'src/stream2/pca_compressor.py',
              'src/stream2/mlp.py']


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def source_sha256(path):
    # Git may check out CRLF on Windows and LF on Kaggle.
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def model_identity(name, spec, code_root=CODE_ROOT):
    paths = COMMON_SOURCES + (C1_SOURCES if name == 'C1' else [])
    return {'schema': 1, 'model': name, 'checkpoint_sha256': spec['checkpoint'][1],
            'pca_sha256': spec['pca'][1] if spec['pca'] else None,
            'runtime': {package: version(package).split('+')[0] for package in
                        ('torch', 'torchaudio', 'numpy', 'scipy', 'scikit-learn', 'soundfile')},
            'sources': {p: source_sha256(Path(code_root) / p) for p in paths}}


def metadata_path(path):
    return Path(str(path) + '.metadata.json')


def write_cache_metadata(path, identity, split_hash):
    data = {'identity': identity, 'manifest_split_hash': split_hash,
            'scores_sha256': file_sha256(path)}
    target = metadata_path(path)
    tmp = target.with_suffix('.partial')
    tmp.write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    tmp.replace(target)


def verify_cache_metadata(path, identity, split_hash):
    target = metadata_path(path)
    if not target.is_file():
        raise ValueError(f'Legacy cache has no preprocessing identity: {path}. Preserve it; use the explicit backup staging workflow.')
    data = json.loads(target.read_text(encoding='utf-8'))
    if data.get('identity') != identity or data.get('manifest_split_hash') != split_hash:
        raise ValueError(f'Cache model/preprocessing/PCA or manifest identity mismatch: {path}')
    if data.get('scores_sha256') != file_sha256(path):
        raise ValueError(f'Cache content changed: {path}')


def verify_reports(thresholds_path, reports, identities):
    """Require current C1/A1 reports bound to exactly these staged thresholds."""
    thresholds = json.loads(Path(thresholds_path).read_text())
    for name, current in identities.items():
        report = reports.get(name, {})
        checks = report.get('checks', [])
        expected_checks = {(order, batch) for order in ('original', 'reversed') for batch in (4, 16, 32)}
        valid_checks = (len(checks) == 6 and {(c.get('order'), c.get('batch_size')) for c in checks} == expected_checks
                        and all(c.get('passed') is True and 0 <= c.get('max_difference', float('inf')) <= 1e-4 for c in checks))
        if (report.get('passed') is not True or report.get('identity') != current
                or report.get('thresholds_sha256') != file_sha256(thresholds_path)
                or report.get('tolerance') != 1e-4 or report.get('selected', 0) < 256
                or report.get('compared', 0) < 100 or report.get('unexpected_rejections', 1) != 0
                or report.get('clean_cache_compared', 0) < 256
                or any(report.get('selected_label_counts', {}).get(str(lab), 0) < 128 for lab in (0, 1))
                or any(report.get('clean_cache_label_counts', {}).get(str(lab), 0) < 128 for lab in (0, 1))
                or report.get('clean_cache_failures', 1) != 0 or not valid_checks
                or not 0 <= report.get('clean_cache_max_difference', float('inf')) <= 1e-4
                or report.get('clean_cache_sha256') != thresholds[name]['provenance'].get('clean_cache_sha256')
                or report.get('manifest_split_hash') != thresholds[name]['provenance'].get('manifest_split_hash')):
            raise ValueError(f'{name} lacks current passing parity and calibrated-cache verification')
