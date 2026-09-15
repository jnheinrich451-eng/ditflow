"""Exercise actual notebook clone/pull guards using isolated local Git repos."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    nb=json.loads((ROOT/'wan_port_acceptance.ipynb').read_text(encoding='utf-8'))
    cells=[''.join(c['source']) for c in nb['cells'] if c['cell_type']=='code']
    for i,source in enumerate(cells):
        if not source.lstrip().startswith('%'): ast.parse(source,filename=f'cell {i}')
    source=next(s for s in cells if "git('pull', '--ff-only'" in s)
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory); remote=root/'remote'; remote.mkdir(); project=root/'checkout'
        def git(repo,*args):
            return subprocess.check_output(['git','-C',str(repo),*args],text=True,stderr=subprocess.PIPE).strip()
        def commit(repo,message):
            git(repo,'add','.')
            git(repo,'-c','user.name=Test','-c','user.email=test@example.invalid','commit','-m',message)
        git(remote,'init','-b','main')
        (remote/'benchmark').mkdir()
        (remote/'benchmark/__init__.py').write_text('')
        (remote/'benchmark/wan_acceptance_runtime.py').write_bytes((ROOT/'benchmark/wan_acceptance_runtime.py').read_bytes())
        runner=remote/'benchmark/wan_port_acceptance.py'
        runner.write_text('class AcceptanceRun:\n    def __init__(self):\n        self.loss_type="flow"\n')
        commit(remote,'initial')
        source='\n'.join(f'PROJECT = Path({str(project)!r})' if s.startswith('PROJECT = ') else
                         f'REPO_URL = {str(remote)!r}' if s.startswith('REPO_URL = ') else s for s in source.splitlines())
        def execute(prefix='',suffix=''):
            return subprocess.run([sys.executable,'-c',prefix+source+'\n'+suffix],text=True,capture_output=True)
        def passed(result):
            assert result.returncode==0, result.stdout+result.stderr
        passed(execute())
        assert git(project,'rev-parse','HEAD')==git(remote,'rev-parse','HEAD')
        runner.write_text(runner.read_text()+'# update\n'); commit(remote,'update')
        passed(execute())
        assert git(project,'rev-parse','HEAD')==git(remote,'rev-parse','HEAD')
        stale=execute("import sys, types\nsys.modules['benchmark.wan_port_acceptance']=types.ModuleType('old')\n")
        assert stale.returncode and 'Restart' in stale.stderr
        local=project/'benchmark/wan_port_acceptance.py'
        previous=local.read_bytes(); local.write_bytes(previous+b'# local change\n')
        dirty=execute(); assert dirty.returncode and 'Local tracked edits' in dirty.stderr
        assert local.read_bytes()==previous+b'# local change\n'
        local.write_bytes(previous)
        passed(execute(suffix='''from benchmark.wan_acceptance_runtime import check_constructor
from benchmark.wan_port_acceptance import AcceptanceRun
check_constructor(AcceptanceRun,PROJECT,manifest)
source=PROJECT/'benchmark/wan_port_acceptance.py'
source.write_text(source.read_text().replace('"flow"','"smm"'))
manifest['sha256']['benchmark/wan_port_acceptance.py']=hashlib.sha256(source.read_bytes()).hexdigest()
try: check_constructor(AcceptanceRun,PROJECT,manifest)
except RuntimeError as e: assert 'Old constructor' in str(e)
else: raise AssertionError('Stale constructor accepted')
'''))
        # Restore only the fixture file; now create divergent committed history.
        local.write_bytes(previous+b'# local commit\n'); commit(project,'local')
        runner.write_text(runner.read_text()+'# remote update\n'); commit(remote,'remote')
        head=git(project,'rev-parse','HEAD')
        divergent=execute(); assert divergent.returncode!=0
        assert git(project,'rev-parse','HEAD')==head
        print('PASS: notebook syntax, clone, fast-forward pull, imported-class guard, dirty edits, stale constructor, divergent history.')


if __name__=='__main__': main()
