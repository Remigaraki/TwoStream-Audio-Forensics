"""Setup must install absent LFS before configuring or fetching model assets."""
import json
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch


class SetupTests(unittest.TestCase):
    def test_environment_bypasses_ensurepip_and_targets_separate_python(self):
        import sys
        nb = json.loads(Path(__file__).with_name('Kaggle_Demo_Parity_Verification.ipynb').read_text())
        calls = []
        ns = {'sys': sys, 'Path': Path, 'MANIFEST': '/existing/manifest.csv',
              'DATA_ROOT': '/audio', 'run': lambda cmd, **kwargs: calls.append(cmd)}
        with patch('pathlib.Path.exists', return_value=True):
            exec(''.join(nb['cells'][4]['source']), ns)
        self.assertIn('--without-pip', calls[0])
        for cmd in calls[1:3]:
            self.assertEqual(cmd[:4], [sys.executable, '-m', 'pip', '--python'])
            self.assertEqual(cmd[4], str(Path('/kaggle/tmp/demo_verify_env') / 'bin/python'))

    def test_missing_lfs_is_installed_before_pull(self):
        nb = json.loads(Path(__file__).with_name('Kaggle_Demo_Parity_Verification.ipynb').read_text())
        source = ''.join(nb['cells'][2]['source'])
        calls = []
        class Process:
            stdout = []
            def wait(self): return 0
        def popen(cmd, **kwargs):
            calls.append(cmd)
            return Process()
        ns = {'REPO': '/kaggle/tmp/existing', 'REPO_URL': 'https://example.test/repo', 'BRANCH': 'main'}
        with patch('pathlib.Path.mkdir'), patch('pathlib.Path.is_dir', return_value=True), \
             patch('subprocess.run', return_value=subprocess.CompletedProcess([], 1)), \
             patch('subprocess.Popen', side_effect=popen):
            exec(source, ns)
        self.assertEqual(calls[:2], [['apt-get', '-qq', 'update'], ['apt-get', '-qq', 'install', '-y', 'git-lfs']])
        self.assertEqual(calls[2], ['git', 'lfs', 'install', '--local'])
        self.assertEqual(calls[3][:3], ['git', 'lfs', 'pull'])


if __name__ == '__main__':
    unittest.main()
