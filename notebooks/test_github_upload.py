import base64
import hashlib
import json
from pathlib import Path
import tempfile
import types
import unittest
from urllib.error import HTTPError
from unittest.mock import patch


class UploadTests(unittest.TestCase):
    def setUp(self):
        nb = json.loads(Path(__file__).with_name('Kaggle_Demo_Calibration.ipynb').read_text(encoding='utf-8'))
        source = next((''.join(c['source']) for c in nb['cells'] if 'def upload_thresholds(' in ''.join(c['source'])), None)
        self.assertIsNotNone(source, 'Notebook needs a GitHub upload cell')
        self.ns = {'UPLOAD_TO_GITHUB': False}
        exec(source, self.ns)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'demo' / 'thresholds.json'
        self.path.parent.mkdir()
        self.path.write_text(json.dumps({'C1': {'status': 'validation', 'threshold': 0.5}}))
        self.data = self.path.read_bytes()
        self.ns.update(REPO=self.tmp.name, REPO_URL='https://github.com/owner/repo.git', BRANCH='main',
                       MODELS=['C1'], PARITY_THRESHOLD_SHA256=hashlib.sha256(self.data).hexdigest())

    def invoke(self, remote, error=None):
        class Response:
            def __init__(self, obj): self.obj = obj
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps(self.obj).encode()
        secret = types.SimpleNamespace(UserSecretsClient=lambda: types.SimpleNamespace(get_secret=lambda name: 'test-token'))
        effects = [Response(remote), error or Response({'commit': {'html_url': 'https://github.com/owner/repo/commit/abc'}})]
        with patch.dict('sys.modules', {'kaggle_secrets': secret}), patch('urllib.request.urlopen', side_effect=effects) as api, patch('subprocess.check_output', return_value='base-sha\n'):
            self.ns['upload_thresholds']()
        return api

    def test_upload(self):
        api = self.invoke({'sha': 'base-sha', 'content': base64.b64encode(b'old').decode()})
        payload = json.loads(api.call_args_list[1].args[0].data)
        self.assertEqual(base64.b64decode(payload['content']), self.data)
        self.assertEqual(payload['branch'], 'main')
        self.assertEqual(payload['sha'], 'base-sha')

    def test_unchanged_skips_write(self):
        api = self.invoke({'sha': 'same', 'content': base64.b64encode(self.data).decode()})
        self.assertEqual(api.call_count, 1)

    def test_missing_parity_blocks(self):
        self.ns.pop('PARITY_THRESHOLD_SHA256')
        with self.assertRaisesRegex(RuntimeError, 'Part C'):
            self.ns['upload_thresholds']()

    def test_changed_thresholds_blocks(self):
        self.path.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'Part C'):
            self.ns['upload_thresholds']()

    def test_remote_change_blocks(self):
        with self.assertRaisesRegex(RuntimeError, 'changed on GitHub'):
            self.invoke({'sha': 'different', 'content': base64.b64encode(b'other').decode()})

    def test_api_error_is_sanitized(self):
        error = HTTPError('https://api.github.com', 403, 'test-token', {}, None)
        with self.assertRaises(RuntimeError) as caught:
            self.invoke({'sha': 'base-sha', 'content': base64.b64encode(b'old').decode()}, error)
        self.assertIn('403', str(caught.exception))
        self.assertNotIn('test-token', str(caught.exception))

    def test_all_notebook_cells_compile(self):
        nb = json.loads(Path(__file__).with_name('Kaggle_Demo_Calibration.ipynb').read_text(encoding='utf-8'))
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), cell['id'], 'exec')


if __name__ == '__main__':
    unittest.main()
