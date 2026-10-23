"""Explicitly stage a completed legacy backup without rescoring or activation.

Identity describes the code to verify next, not a claim that the legacy backup
recorded every source hash. Clean-cache agreement is required by parity.
"""
import argparse
import csv
import io
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from demo import inference as inf
from demo.artifact_identity import file_sha256, model_identity, source_sha256, write_cache_metadata
from demo.calibration_resume import validate_scores

CONDITIONS = ['clean', 'opus_16', 'opus_32', 'opus_64', 'mp3_64', 'mp3_128', 'aac_128']


def stage(backup, output):
    output = Path(output)
    with zipfile.ZipFile(backup) as z:
        if z.testzip() is not None:
            raise ValueError('Corrupt backup')
        check = json.loads(z.read('calibration/batch_independence_check.json'))
        import hashlib
        patch = z.read('calibration/lfcc_corrected.py')
        if not check['passed'] or hashlib.sha256(patch).hexdigest() != check['lfcc_sha256']:
            raise ValueError('Saved LFCC patch/report mismatch')
        corrected_prefix = 'calibration/c1_corrected_' + check['lfcc_sha256'][:12]
        corrected_provenance = json.loads(z.read(corrected_prefix + '/provenance.json'))
        if (corrected_provenance['lfcc_sha256'] != check['lfcc_sha256']
                or z.read(corrected_prefix + '/lfcc.py') != patch):
            raise ValueError('Corrected-cache feature provenance mismatch')
        if hashlib.sha256(patch.replace(b'\r\n', b'\n')).hexdigest() != source_sha256(ROOT / 'src/stream2/lfcc.py'):
            raise ValueError('Current LFCC is not the saved corrected feature definition')
        manifest_bytes = z.read('manifest.csv')
        labels = {r['utterance_id']: int(r['label']) for r in csv.DictReader(io.StringIO(manifest_bytes.decode())) if r['split'] == 'val'}
        if len(labels) != 18769 or sum(labels.values()) != 11249:
            raise ValueError('Unexpected validation coverage')
        thresholds = json.loads(z.read('demo_thresholds.json'))
        files = {}
        for name in inf.MODELS:
            spec = inf.MODELS[name]
            inf.verify_asset(*spec['checkpoint'])
            if spec['pca']:
                inf.verify_asset(*spec['pca'])
            entry = thresholds[name]
            prov = entry['provenance']
            if prov['checkpoint_sha256'] != spec['checkpoint'][1] or prov['manifest_split_hash'] != 'ed4808bb0456de26':
                raise ValueError('Backup checkpoint/manifest mismatch')
            if set(prov['codec_check_val'] or {}) != set(CONDITIONS[1:]):
                raise ValueError(f'{name} codec calibration incomplete')
            prefix = ('calibration/c1_corrected_' + check['lfcc_sha256'][:12]) if name == 'C1' else 'calibration'
            prov['identity'] = model_identity(name, spec)
            prov['identity_binding'] = 'Staged against current source; legacy source provenance incomplete; parity and clean-cache agreement required.'
            prov['backup_content_sha256'] = hashlib.sha256(json.dumps(
                {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(z.namelist())
                 if not n.endswith('/') and (n.startswith('calibration/') or n in
                                             ('manifest.csv', 'demo_thresholds.json', 'backup_status.json'))},
                sort_keys=True).encode()).hexdigest()
            for condition in CONDITIONS:
                filename = f"val_{name}_{spec['checkpoint'][1][:12]}_{condition}.csv"
                files[Path('cache') / name / filename] = z.read(prefix + '/' + filename)
            clean_name = f"val_{name}_{spec['checkpoint'][1][:12]}_clean.csv"
            prov['clean_cache_sha256'] = hashlib.sha256(files[Path('cache') / name / clean_name]).hexdigest()
        files[Path('manifest.reference.csv')] = manifest_bytes
        files[Path('thresholds.candidate.json')] = (json.dumps(thresholds, indent=2) + '\n').encode()
        # Refuse conflicting reruns rather than overwrite saved candidates.
        for rel, data in files.items():
            target = output / rel
            if target.exists() and target.read_bytes() != data:
                raise ValueError(f'Staging conflict at {target}; choose a fresh output folder')
        output.mkdir(parents=True, exist_ok=True)
        for rel, data in files.items():
            target = output / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        lines = ['# Completed validation calibration', '',
                 'Staged only. Full parity remains pending. Thresholds were not retuned.', '',
                 '| Model | Condition | Genuine flagged as spoof (%) | Spoof accepted (%) | DCF at clean threshold |',
                 '|---|---|---:|---:|---:|']
        for name in inf.MODELS:
            for path in sorted((output / 'cache' / name).glob('*.csv')):
                validate_scores(path, labels)
                write_cache_metadata(path, thresholds[name]['provenance']['identity'], 'ed4808bb0456de26')
                with path.open() as f:
                    rows = list(csv.DictReader(f))
                tau = thresholds[name]['threshold']
                bona = [r for r in rows if int(r['true_label']) == 0]
                spoof = [r for r in rows if int(r['true_label']) == 1]
                miss = sum(float(r['score']) >= tau for r in bona) / len(bona)
                fa = sum(float(r['score']) < tau for r in spoof) / len(spoof)
                cost = 1.9 * miss + fa
                condition = path.stem.split(inf.MODELS[name]['checkpoint'][1][:12] + '_', 1)[1]
                saved = thresholds[name]['provenance']
                expected_miss = saved['val_miss_at_threshold'] if condition == 'clean' else saved['codec_check_val'][condition]['miss']
                expected_fa = saved['val_false_alarm_at_threshold'] if condition == 'clean' else saved['codec_check_val'][condition]['false_alarm']
                if abs(miss - expected_miss) > 1e-12 or abs(fa - expected_fa) > 1e-12:
                    raise ValueError(f'Saved errors disagree with scores: {name}/{condition}')
                lines.append(f'| {name} | {condition} | {100*miss:.3f} | {100*fa:.3f} | {cost:.5f} |')
        lines.extend(['', 'C1 genuine-audio rejection increases on MP3 at the clean threshold. '
                      'Codec-specific EER/minDCF values do not describe this fixed operating point. '
                      'These ASVspoof validation results do not establish performance on arbitrary uploads.', ''])
        (output / 'calibration_review.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Staged complete calibration at {output}. Active thresholds unchanged.')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backup', required=True)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    stage(args.backup, args.output)
