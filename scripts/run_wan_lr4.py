"""Fixed two-arm LR experiment adapter; preview by default, never changes guidance code."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PROFILE = ROOT / 'configs/wan_lr4_camel_s1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(arm, baseline_path=None, output_root=None):
    """Read-only, CPU-only preparation. Does not import the model or download weights."""
    from PIL import Image
    from benchmark.wan_source_fingerprints import verify_parity_sources

    plan = json.loads(PROFILE.read_text(encoding='utf-8'))
    baseline = Path(baseline_path or ROOT / plan['baseline']).expanduser().resolve()
    if sha(baseline / 'manifest.json') != plan['baseline_manifest_sha256']:
        raise ValueError('Different baseline manifest; do not substitute a new control.')
    meta = json.loads((baseline / 'manifest.json').read_text(encoding='utf-8'))
    if meta['config']['lr'] != plan['baseline_lr'] or plan['candidate_lr'] != [4*x for x in plan['baseline_lr']]:
        raise ValueError('This candidate must change only LR by fourfold.')
    sources = verify_parity_sources(ROOT, meta['source_sha256'])
    if not json.loads((baseline / 'parity.json').read_text())['passed']:
        raise ValueError('Saved parity record does not pass; no new parity run is scheduled.')
    for a in ('off', *plan['arms']):
        if not (baseline / a / 'final.mp4').is_file():
            raise ValueError(f'Missing saved baseline video: {a}')
    reference = baseline / arm / 'reference'
    expected = meta['reference_rgb_sha256'][::1 if arm == 'forward' else -1]
    files = sorted(reference.glob('*.png'))
    hashes = [hashlib.sha256(Image.open(p).convert('RGB').tobytes()).hexdigest() for p in files]
    if hashes != expected:
        raise ValueError(f'{arm}: reference RGB differs from the saved run.')
    destination = Path(output_root or ROOT / plan['output']).expanduser().resolve()
    if destination.is_relative_to(baseline) or baseline.is_relative_to(destination):
        raise ValueError('Keep candidate outputs and the saved baseline in separate directories.')
    output = destination / arm
    argv = ['motion_guidance_wan.py', *plan['cli_args'], '--lr', *map(str, plan['candidate_lr']),
            '--video_path', str(reference), '--prompt', meta['config']['target_prompt'],
            '--negative_prompt', meta['config']['negative_prompt'], '--seed', str(meta['config']['seed']),
            '--output_path', str(output)]
    return plan, meta, sources, output, argv


def hardware_record(name, total_memory, capability, baseline_gpu):
    """Allow the user's 80 GB hardware; retain the old GPU identity as a limitation."""
    if not any(family in name.upper() for family in ('A100', 'H100')) or total_memory < 70 * 2**30:
        raise ValueError('This Colab workflow requires an 80 GB-class A100 or H100.')
    return dict(name=name, total_memory_bytes=total_memory, capability=list(capability),
                baseline_gpu=baseline_gpu, differs_from_baseline=name != baseline_gpu,
                cpu_offload=True,
                limitation='Setup hashes do not establish cross-hardware trajectory equivalence. '
                           'Small differences from the saved 40 GB baseline cannot be attributed solely to LR.')


def check_previous_arm(output, hardware, profile_hash, launcher_hash):
    if output.name != 'reverse':
        return
    previous = json.loads((output.parent / 'forward/experiment.json').read_text(encoding='utf-8'))
    if (previous['hardware'] != hardware or previous['profile_sha256'] != profile_hash
            or previous['launcher_sha256'] != launcher_hash):
        raise ValueError('Hardware/runtime or experiment files changed between arms; preserve results and stop.')


def execute(plan, meta, sources, output, argv):
    """Invoke the existing CLI once, after baseline identity checks; no custom sampling."""
    if output.exists():
        raise ValueError('Arm already attempted. Preserve it and review; no automatic retry or overwrite.')
    if output.name == 'reverse' and not (output.parent / 'forward/completion.json').is_file():
        raise ValueError('Complete and review the forward arm before starting the reverse arm.')
    from benchmark.wan_runtime_preflight import verify_runtime
    pins = {**meta['packages'], **plan['additional_runtime_pins']}
    verify_runtime(pins)
    import torch
    if platform.python_version() != meta['python'] or torch.version.cuda != meta['cuda']:
        raise ValueError('Python/CUDA differs from the baseline; inspect compatibility before spending compute.')
    if not torch.cuda.is_available():
        raise ValueError('No CUDA GPU; select an 80 GB A100/H100 Colab server.')
    hardware = hardware_record(torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory,
                               torch.cuda.get_device_capability(), meta['gpu'])
    hardware.update(python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(),
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    deterministic_algorithms=torch.are_deterministic_algorithms_enabled())
    check_previous_arm(output, hardware, sha(PROFILE), sha(__file__))
    from huggingface_hub import snapshot_download
    import motion_guidance_wan as entry
    from omegaconf import OmegaConf
    from probe_wan_response import tensor_hash
    verify_runtime(pins)  # Also detect stale imported libraries.
    checkpoint = snapshot_download(meta['model'], revision=meta['checkpoint_revision'])
    original_model, original_factory = entry.MODEL_IDS['14b'], entry.WanGuidance
    entry.MODEL_IDS['14b'] = checkpoint  # Process-local pin; no source/default change.
    output.mkdir(parents=True, exist_ok=False)
    record = dict(profile=json.loads(PROFILE.read_text()), profile_sha256=sha(PROFILE),
                  launcher_sha256=sha(__file__), checkpoint=checkpoint,
                  baseline=meta, source_reuse=sources, argv=argv, runtime=verify_runtime(pins),
                  hardware=hardware)
    from benchmark.wan_acceptance_runtime import git_identity
    record['source_git'] = git_identity(ROOT)

    def write_record(name, data):
        (output / name).write_text(json.dumps(data, indent=2, default=str)+'\n', encoding='utf-8')

    def checked_factory(config):
        config.save_embeds = False  # Match acceptance; CLI otherwise saves extra embedding files.
        actual = OmegaConf.to_container(config, resolve=True)
        allowed = {'lr', 'model_key', 'video_path', 'output_path'}
        mismatch = [k for k, v in meta['config'].items() if k not in allowed and actual.get(k) != v]
        if mismatch or actual['lr'] != plan['candidate_lr']:
            raise ValueError(f'Unexpected configuration differences: {mismatch}')
        OmegaConf.save(config, output / 'config.yaml')
        record['effective_config'] = actual
        write_record('experiment.json', record)
        g = original_factory(config)
        checks = dict(initial_latent_sha256=tensor_hash(g.init_latents),
                      conditioning_sha256=tensor_hash(g.guidance_embeds),
                      source_conditioning_sha256=tensor_hash(g.source_embeds),
                      rope_sha256=tensor_hash(g.transformer.init_rope))
        mismatch = [k for k, v in checks.items() if v != meta[k]]
        if g.scheduler.sigmas.tolist() != meta['sigmas'] or g.timesteps.cpu().tolist() != meta['timesteps']:
            mismatch.append('schedule')
        record.update(setup_checks=checks, mismatches=mismatch)
        write_record('experiment.json', record)
        if mismatch:
            raise ValueError(f'Baseline setup mismatch: {mismatch}. Stop before sampling; do not retry automatically.')
        return g

    old_argv = sys.argv
    started = time.monotonic()
    entry.WanGuidance, sys.argv = checked_factory, argv
    try:
        entry.main()
        generated = [p for p in output.glob('*.mp4') if p.name != 'original.mp4']
        if len(generated) != 1:
            raise ValueError('Expected exactly one generated MP4.')
        write_record('completion.json', dict(video=generated[0].name, sha256=sha(generated[0]),
                     elapsed_seconds=time.monotonic()-started, motion_transfer_accepted=False,
                     status='generation complete; decoded motion and full-video quality review pending'))
    except BaseException as exc:
        write_record('failure.json', dict(error=repr(exc), elapsed_seconds=time.monotonic()-started))
        raise
    finally:
        sys.argv, entry.WanGuidance, entry.MODEL_IDS['14b'] = old_argv, original_factory, original_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('forward', 'reverse'), required=True)
    parser.add_argument('--baseline', type=Path, help='Extracted saved acceptance directory, including one on Drive.')
    parser.add_argument('--output-root', type=Path, help='Persistent candidate directory; notebook uses Google Drive.')
    parser.add_argument('--execute', action='store_true', help='Launch ONE generation; requires a separately authorized GPU budget.')
    args = parser.parse_args()
    os.chdir(ROOT)
    plan, meta, sources, output, argv = prepare(args.arm, args.baseline, args.output_root)
    print(json.dumps(dict(arm=args.arm, checkpoint_revision=meta['checkpoint_revision'],
                          argv=argv, source_checks=sources, execute=args.execute), indent=2))
    if args.execute:
        execute(plan, meta, sources, output, argv)
    else:
        print('Prepared only: no model import, download, GPU initialization, or generation.')


if __name__ == '__main__':
    main()
