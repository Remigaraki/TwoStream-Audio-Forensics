"""Legacy threshold-only upload must remain blocked, even with the old marker."""
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch


class UploadTests(unittest.TestCase):
    def test_old_success_marker_cannot_authorize_publication(self):
        nb = json.loads(Path(__file__).with_name('Kaggle_Demo_Calibration.ipynb').read_text(encoding='utf-8'))
        source = next(''.join(c['source']) for c in nb['cells'] if 'def upload_thresholds(' in ''.join(c['source']))
        ns = {'UPLOAD_TO_GITHUB': False, 'PARITY_THRESHOLD_SHA256': hashlib.sha256(b'{}').hexdigest()}
        exec(source, ns)
        with patch('urllib.request.urlopen') as api:
            with self.assertRaisesRegex(RuntimeError, 'Threshold-only publication is disabled'):
                ns['upload_thresholds']()
            api.assert_not_called()

    def test_all_notebook_cells_compile(self):
        nb = json.loads(Path(__file__).with_name('Kaggle_Demo_Calibration.ipynb').read_text(encoding='utf-8'))
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(i), 'exec')


if __name__ == '__main__':
    unittest.main()
