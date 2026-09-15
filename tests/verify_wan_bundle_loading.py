"""Check the actual notebook loader and import gate without model dependencies."""
import hashlib
from contextlib import chdir
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root/'wan_port_acceptance.ipynb').read_text(encoding='utf-8'))
    cells = [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']
    loader = next(c for c in cells if 'Already imported project modules:' in c)
    gate = next(c for c in cells if 'imported_constructor_matches_disk' in c).split('run = AcceptanceRun(')[0]
    old_cwd, old_path = Path.cwd(), sys.path[:]
    source = 'class AcceptanceRun:\n    def __init__(self):\n        self.loss_type = "flow"\n'
    module_name = 'benchmark.wan_port_acceptance'
    try:
        with tempfile.TemporaryDirectory() as directory, chdir(old_cwd):
            base = Path(directory)
            project, bundle = base/'project', base/'bundle.zip'
            payloads = {'benchmark/__init__.py': b'', 'benchmark/wan_port_acceptance.py': source.encode()}
            manifest = {'revision':'wan-port-drive-v2', 'sha256':{
                name:hashlib.sha256(data).hexdigest() for name,data in payloads.items()}}
            with zipfile.ZipFile(bundle, 'w') as archive:
                for name,data in payloads.items(): archive.writestr(name, data)
                archive.writestr('bundle_manifest.json', json.dumps(manifest))
            # Replace only environment-specific assignments in the notebook cell.
            lines = loader.splitlines()
            lines = [f'PROJECT = Path({str(project)!r})' if l.startswith('PROJECT = ') else
                     f'BUNDLE_PATH = Path({str(bundle)!r})' if l.startswith('BUNDLE_PATH = ') else l for l in lines]
            loader = '\n'.join(lines)
            state = {}
            sys.modules[module_name] = types.ModuleType(module_name)
            try: exec(loader, state)
            except RuntimeError as error: assert 'Restart' in str(error)
            else: raise AssertionError('Loaded-module gate did not stop extraction')
            assert not project.exists()
            del sys.modules[module_name]
            exec(loader, state)
            runner = project/'benchmark/wan_port_acceptance.py'
            assert runner.read_text() == source
            # A same-revision manifest must not mask old or modified source.
            runner.write_text('# stale source\n')
            cache = runner.parent/'__pycache__'; cache.mkdir()
            (cache/'wan_port_acceptance.fake.pyc').write_bytes(b'stale')
            exec(loader, state)
            assert runner.read_text() == source and not list(cache.glob('*.pyc'))
            state['manifest'] = manifest
            exec(gate, state)
            assert state['matches'] is True
            # New bytes on disk with the old class still in memory must fail.
            runner.write_text(source.replace('"flow"', '"smm"'))
            manifest['sha256']['benchmark/wan_port_acceptance.py'] = hashlib.sha256(runner.read_bytes()).hexdigest()
            try: exec(gate, state)
            except AssertionError as error: assert 'Old constructor' in str(error)
            else: raise AssertionError('Stale constructor was accepted')
            print('PASS: cached imports blocked; same-revision source refreshed; stale bytecode removed; live constructor checked.')
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        sys.modules.pop(module_name, None)
        sys.modules.pop('benchmark', None)


if __name__ == '__main__': main()
