import tempfile
import unittest
from pathlib import Path
from demo import artifact_identity as identity


class IdentityTests(unittest.TestCase):
    def test_reports_reject_stale_thresholds_and_incomplete_checks(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            candidate = Path(tmp) / 'thresholds.json'
            candidate.write_text(json.dumps({'C1': {'provenance': {'clean_cache_sha256': 'cache', 'manifest_split_hash': 'split'}}}))
            report = {'passed': True, 'identity': {}, 'thresholds_sha256': identity.file_sha256(candidate),
                      'selected': 256, 'compared': 250, 'unexpected_rejections': 0, 'tolerance': 1e-4,
                      'clean_cache_compared': 256, 'clean_cache_failures': 0, 'clean_cache_max_difference': 1e-6,
                      'selected_label_counts': {'0': 128, '1': 128}, 'clean_cache_label_counts': {'0': 128, '1': 128},
                      'clean_cache_sha256': 'cache', 'manifest_split_hash': 'split',
                      'checks': [{'order': order, 'batch_size': batch, 'passed': True, 'max_difference': 1e-6}
                                 for order in ('original', 'reversed') for batch in (4, 16, 32)]}
            identity.verify_reports(candidate, {'C1': report}, {'C1': {}})
            report['clean_cache_compared'] = 100
            with self.assertRaises(ValueError):
                identity.verify_reports(candidate, {'C1': report}, {'C1': {}})
            report['clean_cache_compared'] = 256
            report['checks'].pop()
            with self.assertRaises(ValueError):
                identity.verify_reports(candidate, {'C1': report}, {'C1': {}})
            report['thresholds_sha256'] = 'stale'
            with self.assertRaises(ValueError):
                identity.verify_reports(candidate, {'C1': report}, {'C1': {}})

    def test_cache_rejects_code_change_and_score_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scores.csv'
            path.write_text('scores')
            original = {'checkpoint_sha256': 'checkpoint', 'preprocessing': 'old', 'pca_sha256': None}
            identity.write_cache_metadata(path, original, 'split')
            identity.verify_cache_metadata(path, original, 'split')
            with self.assertRaises(ValueError):
                identity.verify_cache_metadata(path, dict(original, preprocessing='new'), 'split')
            path.write_text('modified')
            with self.assertRaises(ValueError):
                identity.verify_cache_metadata(path, original, 'split')

    def test_legacy_cache_cannot_be_reused_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'legacy.csv'
            path.write_text('scores')
            with self.assertRaises(ValueError):
                identity.verify_cache_metadata(path, {}, 'split')

    def test_source_identity_is_newline_portable_and_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'code.py'
            path.write_bytes(b'a\r\nb\r\n')
            first = identity.source_sha256(path)
            path.write_bytes(b'a\nb\n')
            self.assertEqual(first, identity.source_sha256(path))
            path.write_bytes(b'a\nc\n')
            self.assertNotEqual(first, identity.source_sha256(path))


if __name__ == '__main__':
    unittest.main()
