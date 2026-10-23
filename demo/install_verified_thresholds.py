"""Install only identity-bound candidates with complete passing parity reports."""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import inference as inf
from demo.artifact_identity import model_identity, verify_reports


def install(candidate, report_dir, destination):
    candidate, report_dir, destination = Path(candidate), Path(report_dir), Path(destination)
    data = json.loads(candidate.read_text())
    identities = {name: model_identity(name, spec) for name, spec in inf.MODELS.items()}
    reports = {name: json.loads((report_dir / f'parity_{name}.json').read_text()) for name in inf.MODELS}
    for name, spec in inf.MODELS.items():
        inf.verify_asset(*spec['checkpoint'])
        if spec['pca']:
            inf.verify_asset(*spec['pca'])
        if data[name]['provenance'].get('identity') != identities[name]:
            raise ValueError('Candidate model/source identity mismatch')
        if data[name]['status'] != 'validation' or not 0 <= float(data[name]['threshold']) <= 1:
            raise ValueError('Invalid calibration candidate')
        if len(data[name]['provenance'].get('codec_check_val') or {}) != 6:
            raise ValueError('Codec checks incomplete')
    verify_reports(candidate, reports, identities)
    backup = destination.with_suffix('.before_verified_install.json')
    if destination.exists() and not backup.exists():
        shutil.copy2(destination, backup)
    # A partial install fails closed at load_thresholds until both files match.
    report_target = destination.with_suffix('.verification.json')
    report_target.write_text(json.dumps(reports, indent=2) + '\n')
    shutil.copy2(candidate, destination)
    print(f'Installed verified calibration: {destination}')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', required=True)
    ap.add_argument('--report_dir', required=True)
    ap.add_argument('--destination', default=str(inf.THRESHOLDS_PATH))
    args = ap.parse_args()
    install(args.candidate, args.report_dir, args.destination)
