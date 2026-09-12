"""Controlled AMF timing experiment helpers; no changes to the generator."""
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

from guidance_utils.wan_guidance_schedule import window_indices, learning_rates

WINDOWS = {'early_00_09': [50, 40], 'later_20_29': [30, 20]}
PROBE_STEPS = [0, 1, 4, 9, 10, 19, 20, 21, 24, 29, 30, 39, 49]


def build_jobs(rows, inputs, output):
    """Only output and guidance window differ within each clip pair."""
    jobs = []
    for row in rows:
        for method, window in WINDOWS.items():
            destination = Path(output)/row['clip_id']/method/'seed1'
            indices = window_indices(50, window)
            rates = list(learning_rates(indices, [.002, .001], 10).values())
            command = [sys.executable, '-u', 'motion_guidance_wan.py',
                '-v', str(Path(inputs)/row['video_path']), '-p', row['prompt'],
                '--model', '1.3b', '-n', '21', '--height', '480', '--width', '832',
                '--scheduler', 'flowmatch', '--flow_shift', '3', '--guidance_scale', '5',
                '--guidance_blocks', '10', '--flow_max_disp', '100', '--motion_temp', '2',
                '--loss_type', 'flow', '--flow_loss', 'mse', '--opt_mode', 'latent',
                '--lr', '0.002', '0.001', '--optimization_steps', '5', '--lr_decay_steps', '10',
                '--guidance_timestep_range', *map(str, window), '--injection_timestep_range', '50', '40',
                '--seed', '1', '--probe', '--probe_blocks', '0', '10', '15', '20',
                '--probe_steps', *map(str, PROBE_STEPS), '--no_injection', '--output_path', str(destination)]
            jobs.append(dict(clip_id=row['clip_id'], method=method, seed=1, output=str(destination),
                             prompt=row['prompt'], window=window, sampling_indices=indices,
                             learning_rates=rates, expected_updates=50, command=command))
    return jobs


def environment_snapshot():
    """Fresh interpreter checks imports and GPU; no pretrained model loading.

    This prevents a resumed comparison from silently changing Torch/CUDA as in
    the preceding upload. Only versions and source hashes are recorded.
    """
    code = '''import json, platform, torch, transformers, diffusers
from importlib.metadata import version, PackageNotFoundError
def installed(package):
    try: return version(package)
    except PackageNotFoundError: return None
print(json.dumps(dict(python=platform.python_version(), packages={p:installed(p) for p in
('torch','torchvision','diffusers','transformers','huggingface-hub','numpy','Pillow','accelerate','omegaconf','einops','imageio','imageio-ffmpeg')},
cuda=torch.version.cuda, gpu=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])))'''
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError('Environment import check failed before generation:\n'+result.stderr[-5000:])
    snapshot = json.loads(result.stdout.strip().splitlines()[-1])
    sources = [Path('motion_guidance.py'), Path('motion_guidance_wan.py'), Path('configs/guidance_config_wan.yaml'),
               Path('benchmark/wan_timing_pilot.py'), *sorted(Path('guidance_utils').glob('*.py'))]
    snapshot['source_sha256'] = {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    return snapshot


def require_same_environment(expected, current):
    if expected != current:
        changed = [key for key in expected.keys() | current.keys() if expected.get(key) != current.get(key)]
        raise RuntimeError('Experiment environment changed: '+', '.join(changed)+
                           '. Restore it or rerun setup to start a fresh paired experiment; do not mix these runs.')


def decoded_digest(video):
    import imageio.v3 as iio
    digest = hashlib.sha256()
    for frame in iio.imiter(video, plugin='FFMPEG'):
        digest.update(frame.tobytes())
    return digest.hexdigest()


def validate_run(job, run=None, expected_environment=None):
    """Verify the actual optimizer trace, not just the command-line intention."""
    from probe_report import load_trace
    from colab_utils import find_outputs
    path = Path(run or job['output'])
    reference, generated = find_outputs(path)
    if not reference or len(generated) != 1 or not all(p.stat().st_size for p in [reference, *generated]):
        raise ValueError(f'Incomplete videos: {path}')
    trace, meta, events = load_trace(path)
    config = meta['config']
    expected = dict(model_key='Wan-AI/Wan2.1-T2V-1.3B-Diffusers', opt_mode='latent', guidance_mode='latent', loss_type='flow', flow_loss='mse',
        flow_region_masks=None, guidance_blocks=[10] if job.get('amf_enabled', True) else [], injection_blocks=[], optimization_steps=5,
        guidance_timestep_range=job['window'], lr=[.002, .001], lr_decay_steps=10,
        num_inference_steps=50, seed=1, num_frames=21, height=480, width=832,
        scheduler='flowmatch', flow_shift=3., guidance_scale=5., motion_temp=2., flow_max_disp=100.,
        threshloss=True, argmax_motion_flow=True, flow_min_conf=None, softmax_fp32=True,
        source_prompt='', target_prompt=job['prompt'])
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'{path}: unexpected {key}: {config.get(key)!r} vs {value!r}')
    updates = [e for e in events if e['kind']=='optimization']
    observed = [(e['step'], e['iteration']) for e in updates]
    wanted = [(step, iteration) for step in job['sampling_indices'] for iteration in range(5)]
    if observed != wanted:
        raise ValueError(f'{path}: optimizer count/order does not match the expected {len(wanted)} updates')
    for e in updates:
        rate = job['learning_rates'][job['sampling_indices'].index(e['step'])]
        if not math.isclose(e['lr'], rate, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(f'{path}: learning-rate sequence changed')
        if e['gradient']['finite_fraction'] != 1 or not math.isfinite(e['loss_before_update']):
            raise ValueError(f'{path}: nonfinite guidance')
    if any(e.get('injected') for e in events if e['kind']=='attention'):
        raise ValueError(f'{path}: KV injection occurred')
    if expected_environment is not None:
        for package, installed in meta['packages'].items():
            if package in expected_environment['packages'] and expected_environment['packages'][package] != installed:
                raise ValueError(f'{path}: saved {package} differs from experiment environment')
        for name, digest in meta['source_sha256'].items():
            if expected_environment['source_sha256'].get(name.replace('\\', '/')) != digest:
                raise ValueError(f'{path}: saved source hash differs for {name}')
        if meta['gpu'] not in expected_environment['gpu']:
            raise ValueError(f'{path}: GPU differs from experiment environment')
    sampling = {e['step']:e for e in events if e['kind']=='sampling'}
    if sorted(sampling) != list(range(50)):
        raise ValueError(f'{path}: incomplete 50-step sampling trace')
    return dict(clip_id=job['clip_id'], method=job['method'], updates=len(updates),
        sampling_indices=job['sampling_indices'], learning_rates=job['learning_rates'],
        guided_sigmas=[sampling[step]['sigma'] for step in job['sampling_indices']],
        losses_before_updates=[e['loss_before_update'] for e in updates],
        reference_decoded_sha256=decoded_digest(reference), trace=str(trace),
        note='Losses at different noise levels are not directly comparable measures of motion quality.')


def audit_experiment(root):
    """Relocatable archive audit; notebook compares actual videos separately."""
    root = Path(root)
    jobs = json.loads((root/'plan.json').read_text(encoding='utf-8'))
    environment = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    reports = [validate_run(job, root/job['clip_id']/job['method']/'seed1', environment) for job in jobs]
    for clip in {j['clip_id'] for j in jobs}:
        pair = [r for r in reports if r['clip_id']==clip]
        if len(pair) != 2 or pair[0]['learning_rates'] != pair[1]['learning_rates']:
            raise ValueError(f'{clip}: invalid paired budget')
        if pair[0]['reference_decoded_sha256'] != pair[1]['reference_decoded_sha256']:
            raise ValueError(f'{clip}: reference previews differ; investigate before treating this as a clean pair')
    (root/'timing_audit.json').write_text(json.dumps(reports, indent=2), encoding='utf-8')
    return reports
