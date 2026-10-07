import csv
import tempfile
import unittest
from pathlib import Path

from demo import calibration_resume as resume


class ResumeTests(unittest.TestCase):
    def test_nested_restore_refuses_conflict_before_copying(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'src', Path(tmp) / 'dst'
            (src / 'nested').mkdir(parents=True)
            dst.mkdir()
            (src / 'nested' / 'scores.csv').write_text('saved')
            (src / 'conflict.json').write_text('original')
            (dst / 'conflict.json').write_text('different')
            with self.assertRaises(ValueError):
                resume.restore_tree(src, dst)
            self.assertFalse((dst / 'nested').exists())
            (dst / 'conflict.json').write_text('original')
            resume.restore_tree(src, dst)
            self.assertEqual((dst / 'nested' / 'scores.csv').read_text(), 'saved')

    def test_cache_rejects_wrong_labels_duplicates_and_nan(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.csv'
            for rows in [[('a', 1, .5)], [('a', 0, float('nan'))], [('a', 0, .5), ('a', 0, .5)]]:
                with path.open('w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['utterance_id', 'true_label', 'score'])
                    writer.writerows(rows)
                with self.assertRaises(ValueError):
                    resume.validate_scores(path, {'a': 0})

    def test_backup_preserves_nested_files_and_marks_unverified(self):
        import zipfile
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache = root / 'calibration'
            (cache / 'corrected').mkdir(parents=True)
            (cache / 'corrected' / 'scores.csv').write_text('saved')
            thresholds, manifest = root / 'thresholds.json', root / 'manifest.csv'
            thresholds.write_text('{}')
            manifest.write_text('manifest')
            output = resume.backup(cache, thresholds, manifest, root)
            with zipfile.ZipFile(output) as z:
                self.assertIsNone(z.testzip())
                self.assertEqual(z.read('calibration/corrected/scores.csv'), b'saved')
                self.assertEqual(json.loads(z.read('backup_status.json'))['parity_status'], 'unverified')


if __name__ == '__main__':
    unittest.main()
