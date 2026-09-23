"""CPU checks for Colab persistence and accidental-generation guards; no model imports."""
import ast
import contextlib
import datetime
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_wan_lr4 import hardware_record, check_previous_arm

NOTEBOOK = json.loads((ROOT / 'wan_lr4_colab.ipynb').read_text(encoding='utf-8'))


class ColabChecks(unittest.TestCase):
    def test_hardware_and_cross_arm_identity(self):
        for name in ('NVIDIA A100-SXM4-80GB', 'NVIDIA H100 80GB HBM3'):
            record = hardware_record(name, 80 * 2**30, (8, 0), 'A100-SXM4-40GB')
            self.assertTrue(record['differs_from_baseline'])
            with tempfile.TemporaryDirectory() as tmp:
                forward = Path(tmp) / 'forward'
                forward.mkdir()
                (forward / 'experiment.json').write_text(json.dumps(dict(
                    hardware=record, profile_sha256='p', launcher_sha256='l')))
                check_previous_arm(Path(tmp) / 'reverse', record, 'p', 'l')
                for changed, profile, launcher in (({**record, 'name': 'other'}, 'p', 'l'),
                                                   (record, 'different', 'l'),
                                                   (record, 'p', 'different')):
                    with self.assertRaises(ValueError):
                        check_previous_arm(Path(tmp) / 'reverse', changed, profile, launcher)
        for name, memory in (('A100-SXM4-40GB', 40), ('NVIDIA L40', 80)):
            with self.assertRaises(ValueError):
                hardware_record(name, memory * 2**30, (8, 0), 'baseline')

    def test_notebook_syntax_and_default_budget(self):
        for cell in NOTEBOOK['cells']:
            if cell['cell_type'] == 'code':
                ast.parse(''.join(cell['source']))
                self.assertIsNone(cell['execution_count'])
                self.assertEqual(cell['outputs'], [])
        scope = {'run_arm': Mock()}
        exec(''.join(NOTEBOOK['cells'][8]['source']), scope)
        self.assertFalse(scope['RUN_GENERATIONS'])

    def test_disabled_generation_and_persistent_failure_log(self):
        tree = ast.parse(''.join(NOTEBOOK['cells'][6]['source']))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            scope = dict(RUN_GENERATIONS=False, DRIVE=base, OUTPUT=base / 'results',
                         PROJECT=ROOT, BASELINE=base / 'baseline', SOURCE_COMMIT='commit',
                         GIT_BRANCH='branch', git=lambda *args: 'commit' if args[0] == 'rev-parse' else '',
                         datetime=datetime, subprocess=subprocess, sys=sys, os=os, json=json)
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<run_arm>', 'exec'), scope)
            with patch.object(subprocess, 'Popen') as popen, contextlib.redirect_stdout(io.StringIO()):
                scope['run_arm']('forward', execute=True)
                popen.assert_not_called()
                self.assertFalse(scope['OUTPUT'].exists())
                proc = Mock(stdout=iter(['failure details\n']))
                proc.wait.return_value = 1
                popen.return_value = proc
                with self.assertRaisesRegex(RuntimeError, 'No automatic retry'):
                    scope['run_arm']('forward')
                command = popen.call_args.args[0]
                self.assertNotIn('--execute', command)
                self.assertEqual(command[command.index('--output-root') + 1], str(scope['OUTPUT']))
                logs = list((scope['OUTPUT'] / 'logs').glob('*.log'))
                self.assertEqual(len(logs), 1)
                self.assertIn('failure details', logs[0].read_text())

    def test_partial_results_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            output, baseline = base / 'candidate', base / 'baseline'
            (output / 'forward').mkdir(parents=True)
            (output / 'forward/failure.json').write_text('{"error":"partial"}')
            (baseline / 'off').mkdir(parents=True)
            (baseline / 'off/final.mp4').write_bytes(b'saved clip fixture')
            scope = dict(OUTPUT=output, BASELINE=baseline, PROJECT=ROOT, DRIVE_BASE=base,
                         datetime=datetime, json=json)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(''.join(NOTEBOOK['cells'][12]['source']), scope)
            with zipfile.ZipFile(scope['archive']) as archive:
                self.assertIn('wan_lr4_camel_s1/forward/failure.json', archive.namelist())
                self.assertIn('baseline/off/final.mp4', archive.namelist())
                self.assertIn('source/configs/wan_lr4_camel_s1.json', archive.namelist())


if __name__ == '__main__':
    unittest.main()
