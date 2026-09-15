"""Weight-free source/data checks for the Git-based acceptance notebook."""
import hashlib
import json
from pathlib import Path
import subprocess
import types
import zipfile


def git_identity(project):
    project = Path(project).resolve()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(project), *args], text=True).strip()
    return dict(commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
                remote=git('remote', 'get-url', 'origin'),
                tracked_changes=git('status', '--porcelain', '--untracked-files=no'))


def source_manifest(project):
    project = Path(project).resolve()
    identity = git_identity(project)
    if identity['tracked_changes']:
        raise RuntimeError(f'Tracked changes must be resolved before a run: {identity}')
    names = subprocess.check_output(['git', '-C', str(project), 'ls-files', '-z'], text=True).split('\0')
    files = {name: hashlib.sha256((project/name).read_bytes()).hexdigest()
             for name in names if Path(name).suffix in ('.py', '.yaml', '.json')}
    return dict(git=identity, sha256=files)


def check_constructor(cls, project, manifest):
    """Compare executable code, not inspect.getsource() of a stale class."""
    import sys
    expected_path = (Path(project)/'benchmark/wan_port_acceptance.py').resolve()
    imported_path = Path(sys.modules[cls.__module__].__file__).resolve()
    if imported_path != expected_path:
        raise RuntimeError(f'Wrong import: expected={expected_path}, actual={imported_path}')
    actual = hashlib.sha256(expected_path.read_bytes()).hexdigest()
    expected = manifest['sha256']['benchmark/wan_port_acceptance.py']
    if actual != expected:
        raise RuntimeError(f'Runner changed: expected SHA256={expected}, actual={actual}')
    compiled = compile(expected_path.read_text(encoding='utf-8'), str(expected_path), 'exec', dont_inherit=True)
    code = next(c for c in compiled.co_consts if isinstance(c, types.CodeType) and c.co_name == cls.__name__)
    init = next(c for c in code.co_consts if isinstance(c, types.CodeType) and c.co_name == '__init__')
    matches = cls.__init__.__code__ == init
    result = dict(runner=str(imported_path), sha256=actual, imported_constructor_matches_disk=matches)
    print(result)
    if not matches:
        raise RuntimeError('Old constructor is still imported. Restart the kernel and rerun source setup.')
    return result


def restore_results(archive, destination):
    """Extract a results archive once; retain all existing results on mismatch."""
    archive, destination = Path(archive), Path(destination).resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    receipt = destination/'restore_receipt.json'
    if destination.exists():
        old = json.loads(receipt.read_text()) if receipt.is_file() else {}
        if old.get('archive_sha256') != digest:
            raise RuntimeError(f'Use a fresh data directory: {destination}; recorded={old}, incoming_sha256={digest}')
        return destination
    with zipfile.ZipFile(archive) as bundle:
        for item in bundle.infolist():
            if not (destination/item.filename).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive path: '+item.filename)
        required = {'manifest.json', 'parity.json', 'off/final.mp4', 'forward/reference/00000.png'}
        if not required.issubset(bundle.namelist()):
            raise ValueError(f'Not an acceptance results ZIP; missing={sorted(required-set(bundle.namelist()))}')
        if bundle.testzip() is not None:
            raise ValueError('Corrupt results ZIP')
        destination.mkdir(parents=True)
        bundle.extractall(destination)
    receipt.write_text(json.dumps(dict(archive=str(archive), archive_sha256=digest), indent=2))
    return destination


def restore_correspondence_inputs(archive, destination):
    """Read only controls/reports from the old ZIP, leaving its large Q/K zipped."""
    archive, destination=Path(archive),Path(destination).resolve()
    existing=destination.exists()
    with zipfile.ZipFile(archive) as bundle:
        plan_bytes=bundle.read('plan.json')
        plan=json.loads(plan_bytes)
        names=set(plan['input_sha256'])|{'plan.json','correspondence_report.json','source_revision.json',
                                         'rgb_regions.jpg','readouts/metadata.json'}
        for name in names:
            if not (destination/name).resolve().is_relative_to(destination):
                raise ValueError('Unsafe archive path: '+name)
            if name not in bundle.namelist(): raise ValueError('Missing correspondence input: '+name)
        destination.mkdir(parents=True,exist_ok=True)
        for name in sorted(names):
            data=bundle.read(name)  # ZIP CRC validates the selected member.
            if name in plan['input_sha256']:
                actual=hashlib.sha256(data).hexdigest()
                if actual!=plan['input_sha256'][name]:
                    raise ValueError(f"Input changed: {name}; expected={plan['input_sha256'][name]}, actual={actual}")
            path=destination/name
            if existing:
                actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else 'missing'
                expected=hashlib.sha256(data).hexdigest()
                if actual!=expected:
                    raise ValueError(f'Existing restored file differs: {path}; expected={expected}, actual={actual}')
            else:
                path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
    return destination


def save_diagnostic_archives(prepared, drive_directory):
    """Keep full evidence in Drive and provide a small first-review archive."""
    import shutil
    prepared,drive_directory=Path(prepared).resolve(),Path(drive_directory)
    drive_directory.mkdir(parents=True,exist_ok=True)
    full=Path(shutil.make_archive(str(prepared),'zip',root_dir=prepared))
    full_destination=drive_directory/full.name
    shutil.copy2(full,full_destination)
    small=drive_directory/(prepared.name+'_review.zip')
    names=['plan.json','correspondence_report.json','timing_comparison.json','source_revision.json',
           'rgb_regions.jpg','rgb_motion.npz','readouts/metadata.json']
    names += [str(p.relative_to(prepared)) for p in (prepared/'baseline').glob('*.json')]
    with zipfile.ZipFile(small,'w',zipfile.ZIP_DEFLATED) as archive:
        for name in names:
            if (prepared/name).is_file(): archive.write(prepared/name,name)
    return dict(full=str(full_destination),review=str(small),review_bytes=small.stat().st_size)
