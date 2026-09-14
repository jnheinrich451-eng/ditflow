"""Restore completed results ZIPs locally before auditing many small captures."""
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


def _inventory(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def _result_members(bundle):
    files = {}
    for info in bundle.infolist():
        name = PurePosixPath(info.filename.replace('\\', '/'))
        if name.is_absolute() or '..' in name.parts or any(':' in part for part in name.parts):
            raise ValueError('Unsafe archive member: '+info.filename)
        if stat.S_ISLNK(info.external_attr >> 16):
            raise ValueError('Archive symlinks are not supported: '+info.filename)
        if info.is_dir():
            continue
        if name in files:
            raise ValueError('Duplicate archive member: '+info.filename)
        files[name] = info
    required = ('plan.json', 'plan.sha256', 'environment.json', 'confirmation_done.json',
                'off_done.json', 'candidate_done.json')
    roots = [name.parent for name in files if name.name == 'plan.json'
             and all(name.parent / marker in files for marker in required)]
    if len(roots) != 1:
        raise ValueError('Use the completed noised-reference RESULTS ZIP, containing plan.json and '
                         'confirmation/off/candidate completion records. The reference-input ZIP '
                         'belongs in CONTROL_INPUT_ZIP. Expected one results folder, found '+str(len(roots)))
    root = roots[0]
    plan = json.loads(bundle.read(files[root/'plan.json']).decode('utf-8'))
    if plan.get('stage') != 'noised_reference_step9_visual':
        raise ValueError('This setup requires the completed noised-reference results archive')
    return {name.relative_to(root): info for name, info in files.items() if name.is_relative_to(root)}


def restore_previous_archive(archive, destination):
    """Copy ZIP off Drive, extract/check locally, and preserve existing folders.

    Returns the actual restored directory. Supports a containing run folder or
    a ZIP rooted directly at plan.json. Identical restores are reused; an
    incomplete/different destination is retained and a separate cache is used.
    """
    archive, destination = Path(archive).expanduser(), Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wan_restore_', dir=destination.parent) as temp:
        staging = Path(temp)
        local_zip = staging/'results.zip'
        try:
            shutil.copyfile(archive, local_zip)
        except OSError as error:
            raise OSError('Colab/Python cannot read the results ZIP at '+str(archive)+
                          '. Check the mounted Drive path; visibility in the Drive website '
                          'does not establish runtime readability. '+str(error)) from error
        restored = staging/'result'; restored.mkdir()
        try:
            with zipfile.ZipFile(local_zip) as bundle:
                members = _result_members(bundle)
                for relative, info in members.items():
                    path = restored.joinpath(*relative.parts)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, path.open('wb') as target:
                        shutil.copyfileobj(source, target)
        except zipfile.BadZipFile as error:
            raise ValueError('Results ZIP is incomplete or corrupt; restore the complete exported archive') from error
        expected = _inventory(restored)
        if destination.exists():
            if destination.is_dir() and _inventory(destination) == expected:
                print('Reusing verified local results:', destination)
                return destination
            identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
            destination = destination.with_name(destination.name+'_restored_'+identity)
            if destination.exists():
                if destination.is_dir() and _inventory(destination) == expected:
                    print('Reusing verified local results:', destination)
                    return destination
                raise ValueError('Restored cache differs from the ZIP; choose a fresh local destination: '+str(destination))
        # Only this newly created temporary tree is moved, to a checked unused sibling.
        if destination.parent.resolve() != staging.parent.resolve() or destination.exists():
            raise ValueError('Restore destination must be an unused path inside the selected local parent')
        restored.rename(destination)
    print(f'Restored {len(expected)} results files locally:', destination)
    return destination
