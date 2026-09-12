"""Matched prompt-direction x AMF diagnostic. Does not modify the generator."""
import copy
import csv
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from benchmark import wan_timing_pilot as timing

CASES = ('car-turn', 'camel')
CONDITIONS = ('original_off', 'original_amf', 'aligned_off', 'aligned_amf')
DIRECTION_SUFFIXES = {
    'car-turn': 'The truck approaches the camera around the bend, growing larger in view, '
                'with its front visible as it turns toward the left side of the image.',
    'camel': 'The horse faces right and walks forward toward the right side of the image, '
             'then gradually turns away from the camera.',
}


def build_jobs(rows, inputs, output, model='1.3b', cpu_offload=False):
    """Eight fresh runs: benchmark prompt versus direction suffix, AMF off/on."""
    if model not in timing.MODEL_KEYS:
        raise ValueError(f'Unsupported experiment model: {model}')
    if len(rows) != 2 or {r['clip_id'] for r in rows} != set(CASES):
        raise ValueError('Expected exactly car-turn and camel reference rows')
    jobs = []
    for base in timing.build_jobs(rows, inputs, output):
        if base['method'] != 'early_00_09':
            continue
        for condition in CONDITIONS:
            job = copy.deepcopy(base)
            aligned, enabled = condition.startswith('aligned'), condition.endswith('_amf')
            prompt = base['prompt']
            if aligned:
                prompt = prompt.rstrip().rstrip('.') + '. ' + DIRECTION_SUFFIXES[base['clip_id']]
            destination = Path(output)/base['clip_id']/condition/'seed1'
            command = job['command']
            command[command.index('--model') + 1] = model
            if cpu_offload:
                command.append('--low_vram')
            command[command.index('-p') + 1] = prompt
            command[command.index('--output_path') + 1] = str(destination)
            if not enabled:
                command.append('--no_guidance')
                job.update(sampling_indices=[], learning_rates=[], expected_updates=0)
            job.update(method=condition, prompt_condition='aligned' if aligned else 'original',
                       amf_enabled=enabled, prompt=prompt, original_prompt=base['prompt'],
                       output=str(destination), model=model, cpu_offload=cpu_offload)
            jobs.append(job)
    return jobs


def prepare_inputs(inputs, archive=None):
    """Reuse or safely extract the existing two-clip bundle and verify its inputs."""
    inputs = Path(inputs).expanduser().resolve()
    if not (inputs/'manifest.csv').is_file():
        candidates = ([Path(archive).expanduser()] if archive else [
            inputs.with_suffix('.zip'), Path('wan_reference_inputs.zip'),
            Path('/content/wan_reference_inputs.zip'),
            Path('/content/drive/MyDrive/wan_reference_inputs.zip'),
            Path('/content/drive/MyDrive/ditflow_probes/wan_reference_inputs.zip')])
        bundle_path = next((p for p in candidates if p.is_file()), None)
        if bundle_path is None:
            raise FileNotFoundError('Set DIR_ZIP to the kernel path of wan_reference_inputs.zip. '
                'In VS Code use Explorer -> Upload to Colab, or mount Google Drive. Searched: '
                + ', '.join(map(str, candidates)))
        # The prepared archive always contains this directory, irrespective of ZIP filename.
        inputs = inputs.parent/'wan_reference_inputs'
        with zipfile.ZipFile(bundle_path) as bundle:
            if 'wan_reference_inputs/manifest.csv' not in bundle.namelist():
                raise ValueError('Use wan_reference_inputs.zip, the existing two-clip reference bundle')
            for member in bundle.namelist():
                if not (inputs.parent/member).resolve().is_relative_to(inputs):
                    raise ValueError(f'Unexpected ZIP member: {member}')
            bundle.extractall(inputs.parent)
    checksums = json.loads((inputs/'checksums.json').read_text(encoding='utf-8'))
    for relative, expected in checksums.items():
        item = (inputs/relative).resolve()
        if not item.is_relative_to(inputs) or hashlib.sha256(item.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Changed diagnostic input: {relative}')
    with (inputs/'manifest.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    with Path('benchmark/davis50.csv').open(newline='', encoding='utf-8') as handle:
        official = {(r['clip_id'], r['prompt_id']): r for r in csv.DictReader(handle)}
    if len(rows) != 2 or {r['clip_id'] for r in rows} != set(CASES):
        raise ValueError('Expected exactly car-turn and camel')
    for row in rows:
        reference = official[(row['clip_id'], 'subject')]
        if row['prompt'] != reference['prompt'] or row['sha256'] != reference['sha256']:
            raise ValueError(f"Changed original benchmark row: {row['clip_id']}")
        frame_dir = (inputs/row['video_path']).resolve()
        if not frame_dir.is_relative_to(inputs):
            raise ValueError(f'Unexpected frame directory: {frame_dir}')
        frames = sorted(frame_dir.glob('*.jpg'))
        digest = hashlib.sha256()
        for frame in frames:
            digest.update(frame.read_bytes())
        if len(frames) != 24 or digest.hexdigest() != reference['sha256']:
            raise ValueError(f"Changed reference frames: {row['clip_id']}")
    return inputs, rows


def environment_snapshot():
    snapshot = timing.environment_snapshot()
    path = Path('benchmark/wan_direction_pilot.py')
    snapshot['source_sha256'][path.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return snapshot


def validate_run(job, run=None, expected_environment=None):
    from probe_report import load_trace
    condition = job['method']
    enabled = condition.endswith('_amf')
    expected_prompt = job['original_prompt']
    if condition.startswith('aligned'):
        expected_prompt = expected_prompt.rstrip().rstrip('.') + '. ' + DIRECTION_SUFFIXES[job['clip_id']]
    expected_rates = list(timing.learning_rates(range(10), [.002, .001], 10).values()) if enabled else []
    if (condition not in CONDITIONS or job['amf_enabled'] != enabled or job['prompt'] != expected_prompt
            or job['window'] != [50, 40] or job['sampling_indices'] != (list(range(10)) if enabled else [])
            or job['learning_rates'] != expected_rates or job['expected_updates'] != (50 if enabled else 0)):
        raise ValueError('Saved job does not match the fixed prompt-direction experiment')
    result = timing.validate_run(job, run, expected_environment)
    _, _, events = load_trace(Path(run or job['output']))
    sampling = [e for e in events if e['kind'] == 'sampling']
    if [e['step'] for e in sampling] != list(range(50)):
        raise ValueError('Expected exactly 50 sampling events in order')
    for event in sampling:
        delta = event['guidance_update']
        if delta['finite_fraction'] != 1:
            raise ValueError('Nonfinite latent guidance update')
        if event['step'] not in job['sampling_indices'] and delta['rms'] != 0:
            raise ValueError('Latents changed by guidance outside the active window')
    updates = [e for e in events if e['kind'] == 'optimization']
    if updates and not any(e['update']['rms'] > 0 for e in updates):
        raise ValueError('Guidance optimizer produced no latent changes')
    if any(e['update']['finite_fraction'] != 1 for e in updates):
        raise ValueError('Nonfinite optimizer update')
    result.update(prompt_condition=job['prompt_condition'], amf_enabled=job['amf_enabled'], prompt=job['prompt'],
                  model=job.get('model', '1.3b'), cpu_offload=job.get('cpu_offload', False))
    return result


def scheduler_config_differences(reference, compared):
    """Ignore only ordering of Diffusers' set-derived default-parameter names.

    Diffusers builds _use_default_values with list(set(...)), so separate Python
    processes can serialize the same names in different orders. All other
    fields, list orderings, and the membership of that list remain significant.
    Neither of the original metadata dictionaries is modified.
    """
    def normalized(config):
        result = dict(config)
        defaults = result.get('_use_default_values')
        if isinstance(defaults, list) and all(isinstance(name, str) for name in defaults):
            result['_use_default_values'] = sorted(defaults)
        return result

    left, right = normalized(reference), normalized(compared)
    return {key: {'reference': left.get(key, '<missing>'), 'compared': right.get(key, '<missing>')}
            for key in sorted(left.keys() | right.keys())
            if key not in left or key not in right or left[key] != right[key]}


def audit_experiment(root):
    """Relocatable audit of all four conditions per clip, including unguided controls."""
    import numpy as np
    from probe_report import load_trace
    root = Path(root)
    jobs = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    environment = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    with (root/'input_manifest.csv').open(newline='', encoding='utf-8') as handle:
        original_prompts = {r['clip_id']: r['prompt'] for r in csv.DictReader(handle)}
    if len(jobs) != 8 or {(j['clip_id'], j['method']) for j in jobs} != {
            (clip, condition) for clip in CASES for condition in CONDITIONS}:
        raise ValueError('Expected eight distinct prompt-by-AMF conditions')
    if len({(j.get('model', '1.3b'), j.get('cpu_offload', False)) for j in jobs}) != 1:
        raise ValueError('Do not mix model sizes or CPU-offload settings within an experiment')
    reports = []
    for clip in CASES:
        metas, references, rates = [], [], []
        for job in [j for j in jobs if j['clip_id'] == clip]:
            if job['original_prompt'] != original_prompts[clip]:
                raise ValueError(f'{clip}: original prompt differs from the saved input manifest')
            run = root/clip/job['method']/'seed1'
            reports.append(validate_run(job, run, environment))
            path, meta, events = load_trace(run)
            metas.append(meta)
            if job['amf_enabled']:
                captures = [e for e in events if e['kind'] == 'training_reference'
                            and e['block'] == 'block_10_attn1_processor']
                if len(captures) != 1:
                    raise ValueError(f'{run}: missing or ambiguous training reference AMF')
                with np.load(path/captures[0]['file']) as data:
                    references.append((data['flow'].copy(), data['mask'].copy()))
                rates.append(job['learning_rates'])
        if len({r['reference_decoded_sha256'] for r in reports if r['clip_id'] == clip}) != 1:
            raise ValueError(f'{clip}: reference videos differ across conditions')
        if len(references) != 2 or rates[0] != rates[1] or not all(
                np.array_equal(a, b) for a, b in zip(*references)):
            raise ValueError(f'{clip}: guided reference AMF or learning rates differ')
        for meta in metas[1:]:
            for key in ('packages', 'python', 'gpu', 'source_sha256', 'scheduler', 'grid'):
                if meta[key] != metas[0][key]:
                    raise ValueError(f'{clip}: different recorded {key}')
            scheduler_changes = scheduler_config_differences(metas[0]['scheduler_config'], meta['scheduler_config'])
            if scheduler_changes:
                raise ValueError(f'{clip}: different recorded scheduler_config in {meta["config"].get("output_path")}: '
                                 + json.dumps(scheduler_changes, sort_keys=True))
            allowed = {'target_prompt', 'guidance_blocks', 'output_path'}
            differences = {k for k in meta['config'].keys() | metas[0]['config'].keys()
                           if meta['config'].get(k) != metas[0]['config'].get(k)}
            if differences - allowed:
                raise ValueError(f'{clip}: extra configuration differences: {differences - allowed}')
    (root/'direction_audit.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports


def run_jobs(root, dry_run=False):
    """Resume the saved plan; completed runs require a matching completion marker and trace."""
    root = Path(root)
    jobs = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    expected = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    for job in jobs:
        print(job.get('model', '1.3b'), '|', job['clip_id'], '|', job['method'], '|', job['prompt'], flush=True)
        if dry_run:
            print(subprocess.list2cmdline(job['command']), flush=True)
            continue
        timing.require_same_environment(expected, environment_snapshot())
        output = Path(job['output'])
        if output.resolve() != (root/job['clip_id']/job['method']/'seed1').resolve():
            raise ValueError('Resume generation in the original output directory; copied archives are for review')
        marker = output/'direction_done.json'
        if marker.is_file():
            if json.loads(marker.read_text(encoding='utf-8')) != job:
                raise ValueError(f'Saved plan changed for completed run: {output}')
            validate_run(job, expected_environment=expected)
            print('Completed already:', output, flush=True)
            continue
        output.mkdir(parents=True, exist_ok=True)
        with (output/'generation.log').open('a', encoding='utf-8') as log:
            process = subprocess.Popen(job['command'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding='utf-8', errors='replace', bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        if code:
            raise RuntimeError(f'Generation failed; see {output / "generation.log"}')
        timing.require_same_environment(expected, environment_snapshot())
        result = validate_run(job, expected_environment=expected)
        (output/'direction_validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        marker.write_text(json.dumps(job, indent=2), encoding='utf-8')


def archive_experiment(root, drive_dir=None):
    """Bundle generated videos, traces and reports; exclude optimized embedding files."""
    import shutil
    from datetime import datetime, timezone
    root = Path(root).resolve()
    archive = root.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
        for item in sorted(root.rglob('*')):
            if item.is_file() and 'embeds' not in item.relative_to(root).parts and item.suffix != '.zip':
                bundle.write(item, item.relative_to(root.parent).as_posix())
    copied = None
    if drive_dir is not None and Path(drive_dir).parent.is_dir():
        drive_dir = Path(drive_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
        copied = drive_dir/archive.name
        if copied.exists():
            copied = drive_dir/(archive.stem + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.zip')
        shutil.copy2(archive, copied)
    return archive, copied
