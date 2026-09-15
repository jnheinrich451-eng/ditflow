"""Decisive Wan 14B test: does DiTFlow's AMF guidance steer decoded motion at all?

One plan is one clip and one seed: seven generations and one verdict fixed before the run.
The verdict uses decoded videos only; no internal AMF score enters it.
Protocol and decision rule: docs/WAN_DECISIVE_TEST.md.

Arms, all from the same initial noise, prompt and sampler:
  off                      vanilla Wan, no guidance
  forward / reverse        AMF guidance at full DiTFlow strength toward the reference / time-reversed reference
  random_a / random_b      the same per-step latent change as the AMF arms, in random directions
  sdedit_forward / _reverse positive control: sampling starts from the noised reference / reversed reference
"""
import base64
import csv
import hashlib
import html
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from benchmark import wan_direction_pilot as direction
from benchmark import wan_timing_pilot as timing
from guidance_utils import decoded_motion as dm

STAGE = 'decisive_decoded_selectivity_v1'
MODEL = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
ARMS = ('off', 'forward', 'reverse', 'random_a', 'random_b', 'sdedit_forward', 'sdedit_reverse')
KIND = dict(off='off', forward='amf', reverse='amf', random_a='random', random_b='random',
            sdedit_forward='sdedit', sdedit_reverse='sdedit')
REVERSED = frozenset({'reverse', 'sdedit_reverse'})
RANDOM_SEEDS = dict(random_a=101, random_b=202)
GUIDANCE_BLOCK = 20            # guidance_blocks_14b in configs/guidance_config_wan.yaml: the ported paper default
GUIDANCE_STEPS = tuple(range(10))
UPDATES_PER_STEP = 5
LEARNING_RATE = [.002, .001]
SDEDIT_START = 20              # sampling index 20, sigma 0.814 on the 50-step shift-3 schedule
FIRST_CLIP = ('camel', 1)      # the most favourable case for AMF: sustained planar travel and camera pan
CONFIRMATION = (('car-turn', 1), ('camel', 2))
SOURCES = ('probe_wan_decisive.py', 'benchmark/wan_decisive_pilot.py')


def stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_finite(value), indent=2, allow_nan=False), encoding='utf-8')


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def environment_snapshot():
    """Packages, GPU and source hashes. A resumed run must match the plan's snapshot exactly."""
    snapshot = timing.environment_snapshot()
    extra = {}
    for name in ('ftfy', 'tokenizers', 'sentencepiece', 'safetensors', 'opencv-python', 'opencv-python-headless'):
        try:
            extra[name] = version(name)
        except PackageNotFoundError:
            extra[name] = None
    snapshot['extra_packages'] = extra
    for name in (*SOURCES, 'benchmark/wan_direction_pilot.py'):
        snapshot['source_sha256'][name] = digest(name)
    return snapshot


def inventory(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): digest(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def write_reversed_frames(video, destination):
    """Reverse decoded RGB frames before the causal VAE; latents are never flipped."""
    from PIL import Image
    from guidance_utils.wan_reference_diagnostics import reference_images
    frames = reference_images(video, dm.FRAMES, dm.SIZE)
    destination = Path(destination)
    destination.mkdir()
    for index, frame in enumerate(frames[::-1]):
        Image.fromarray(frame).save(destination / f'{index:05d}.png')
    return inventory(destination)


# --------------------------------------------------------------------- plan
def make_plan(inputs, root, clip=FIRST_CLIP[0], seed=FIRST_CLIP[1], archive=None):
    root = Path(root).resolve()
    if clip not in direction.CASES or int(seed) != seed or seed < 0:
        raise ValueError(f'clip must be one of {direction.CASES} and seed a non-negative integer')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh experiment directory; restore an existing one to resume instead')
    inputs, rows = direction.prepare_inputs(inputs, archive)
    row = next(r for r in rows if r['clip_id'] == clip)
    snapshot = environment_snapshot()
    root.mkdir(parents=True, exist_ok=True)
    reverse = root / 'reference_reverse'
    plan = dict(schema_version=1, stage=STAGE, model=MODEL, clip_id=clip, seed=int(seed), prompt=row['prompt'],
                inputs=str(inputs), video=str(inputs / row['video_path']), reverse_video=str(reverse),
                arms=list(ARMS), guidance_block=GUIDANCE_BLOCK, guidance_indices=list(GUIDANCE_STEPS),
                updates_per_step=UPDATES_PER_STEP, learning_rate=LEARNING_RATE, sdedit_start=SDEDIT_START,
                random_seeds=RANDOM_SEEDS, thresholds=dm.THRESHOLDS,
                first_clip=list(FIRST_CLIP), confirmation=[list(c) for c in CONFIRMATION],
                input_manifest_sha256=digest(inputs / 'manifest.csv'),
                reverse_inventory=write_reversed_frames(inputs / row['video_path'], reverse))
    write_json(root / 'plan.json', plan)
    (root / 'plan.sha256').write_text(digest(root / 'plan.json'), encoding='utf-8')
    write_json(root / 'environment.json', snapshot)
    (root / 'input_manifest.csv').write_bytes((inputs / 'manifest.csv').read_bytes())
    return plan


def checked_plan(root):
    root = Path(root)
    if digest(root / 'plan.json') != (root / 'plan.sha256').read_text(encoding='utf-8').strip():
        raise ValueError('Plan changed after setup; create a fresh experiment')
    plan = read_json(root / 'plan.json')
    expected = dict(stage=STAGE, model=MODEL, arms=list(ARMS), guidance_block=GUIDANCE_BLOCK,
                    guidance_indices=list(GUIDANCE_STEPS), updates_per_step=UPDATES_PER_STEP,
                    learning_rate=LEARNING_RATE, sdedit_start=SDEDIT_START, random_seeds=RANDOM_SEEDS,
                    thresholds=dm.THRESHOLDS)
    changed = [k for k, v in expected.items() if plan.get(k) != v]
    if changed:
        raise ValueError('This plan used a different protocol (' + ', '.join(changed) + '); create a fresh plan')
    inputs, rows = direction.prepare_inputs(plan['inputs'])
    row = next(r for r in rows if r['clip_id'] == plan['clip_id'])
    if str(inputs / row['video_path']) != plan['video'] or row['prompt'] != plan['prompt']:
        raise ValueError('Reference input or prompt differs from the plan')
    if digest(inputs / 'manifest.csv') != plan['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    if inventory(plan['reverse_video']) != plan['reverse_inventory']:
        raise ValueError('Reversed reference frames changed')
    return plan


def fixed_config(plan, arm, output, dose=None):
    """Complete generation settings for one arm. Arms differ only where the protocol says they must."""
    if arm not in ARMS:
        raise ValueError(f'Unknown arm {arm!r}')
    kind = KIND[arm]
    if (kind == 'random') != (dose is not None):
        raise ValueError('Exactly the random arms take a matched dose')
    return dict(
        model_key=MODEL, enable_model_cpu_offload=True, output_path=str(output),
        video_path=plan['reverse_video'] if arm in REVERSED else plan['video'],
        target_prompt=plan['prompt'], source_prompt='', seed=plan['seed'],
        opt_mode='latent', guidance_mode='latent', loss_type='flow', save_format='mp4', save_embeds=False,
        inject_embeds=False, verbose=False, scheduler='flowmatch', flow_shift=3., num_frames=dm.FRAMES,
        height=dm.SIZE[1], width=dm.SIZE[0], num_inference_steps=50, guidance_scale=5.,
        guidance_blocks=[GUIDANCE_BLOCK] if kind in ('amf', 'random') else [], flow_head=None,
        injection_blocks=[], guidance_timestep_range=[50, 40], injection_timestep_range=[50, 40],
        lr=list(LEARNING_RATE), lr_decay_steps=None, optimization_steps=UPDATES_PER_STEP, motion_temp=2.,
        flow_max_disp=None, flow_min_conf=None, threshloss=True, argmax_motion_flow=True, flow_loss='mse',
        flow_region_masks=None, softmax_fp32=True, checkpoint_amf='auto', enable_gradient_checkpointing=True,
        probe=False, probe_rope=False, reference_only=False,
        decisive_stage=STAGE, decisive_arm=arm, decisive_kind=kind,
        random_seed=RANDOM_SEEDS.get(arm), random_dose=dict(dose) if dose is not None else None,
        sdedit_start=SDEDIT_START if kind == 'sdedit' else None)


# ------------------------------------------------------- arm-level helpers
def random_perturbation(latent, target_rms, seed, step):
    """latent + a Gaussian direction scaled to exactly target_rms. Own generator: global RNG untouched."""
    import torch
    generator = torch.Generator(device='cpu').manual_seed(int(seed) * 1000 + int(step))
    direction_ = torch.randn(tuple(latent.shape), generator=generator, dtype=torch.float64)
    direction_ = direction_ / direction_.square().mean().sqrt()
    return (latent.detach().double() + float(target_rms) * direction_.to(latent.device)).float()


def sdedit_start_latent(reference, noise, sigma):
    """Flow-matching forward noising, (1 - sigma) * reference + sigma * noise."""
    return (1. - float(sigma)) * reference.float() + float(sigma) * noise.float()


def initial_state(g):
    return dict(latent_sha256=tensor_sha256(g.init_latents), conditioning_sha256=tensor_sha256(g.guidance_embeds),
                timesteps=[float(t) for t in g.timesteps], sigmas=[float(s) for s in g.scheduler.sigmas],
                model=g.config.model_key)


def require_common_start(left, right):
    for key in ('latent_sha256', 'conditioning_sha256', 'timesteps', 'sigmas', 'model'):
        if left[key] != right[key]:
            raise ValueError('Arms do not share the same starting state: ' + key)


def completed(root, arm):
    marker = Path(root) / f'{arm}_done.json'
    if not marker.is_file():
        raise FileNotFoundError(f'Arm {arm!r} has not completed in {root}')
    return Path(root) / read_json(marker)['directory']


def matched_dose(root):
    """Per-step latent change of the two AMF arms, averaged: what the random arms must reproduce."""
    keys = [str(i) for i in GUIDANCE_STEPS]
    doses = []
    for arm in ('forward', 'reverse'):
        update = read_json(completed(root, arm) / 'dose.json')['update_rms']
        if sorted(update, key=int) != keys or not all(math.isfinite(v) and v > 0 for v in update.values()):
            raise ValueError(f'{arm} did not record a positive update at every guided step')
        doses.append(update)
    return {k: float(np.mean([d[k] for d in doses])) for k in keys}


def run_arm(root, arm, command, validator):
    """Fresh attempt per call; the done marker is written only after exit and validation."""
    root = Path(root).resolve()
    expected = read_json(root / 'environment.json')
    timing.require_same_environment(expected, environment_snapshot())
    marker = root / f'{arm}_done.json'
    if marker.is_file():
        directory = root / read_json(marker)['directory']
        validator(directory)
        print('Already complete and revalidated:', directory, flush=True)
        return directory
    attempt = f'{arm}_{stamp()}'
    directory = root / attempt
    with (root / f'{attempt}.log').open('w', encoding='utf-8') as log, subprocess.Popen(
            [*command, '--output_path', str(directory)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            if process.wait() != 0:
                raise RuntimeError(f'{arm} failed; log retained at {log.name}. Rerun to start a fresh attempt.')
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait()
            raise
    timing.require_same_environment(expected, environment_snapshot())
    audit = validator(directory)
    write_json(marker, dict(directory=directory.name, audit=audit))
    return directory


def validate_arm(directory, root, plan, arm):
    import imageio.v3 as iio
    from omegaconf import OmegaConf
    directory, root, kind = Path(directory), Path(root), KIND[arm]
    config = OmegaConf.to_container(OmegaConf.load(directory / 'suite_config.yaml'), resolve=True)
    dose = matched_dose(root) if kind == 'random' else None
    expected = fixed_config(plan, arm, config['output_path'], dose)
    changed = [k for k, v in expected.items() if k != 'random_dose' and config.get(k) != v]
    if changed:
        raise ValueError(f'{arm}: frozen configuration changed: ' + ', '.join(changed))
    if kind == 'random' and (set(config['random_dose']) != set(dose) or any(
            not math.isclose(config['random_dose'][k], dose[k], rel_tol=1e-9) for k in dose)):
        raise ValueError(f'{arm}: random dose differs from the completed AMF arms')
    if read_json(directory / 'complete.json') != dict(arm=arm, native_rope_unchanged=True, frozen_weights=True):
        raise ValueError(f'{arm}: incomplete generation, or pretrained weights/RoPE were modified')
    if arm != 'off':
        require_common_start(read_json(directory / 'initial_state.json'),
                             read_json(completed(root, 'off') / 'initial_state.json'))
    record = read_json(directory / 'dose.json')
    if record.get('arm') != arm or record.get('kind') != kind:
        raise ValueError(f'{arm}: dose record belongs to another arm')
    keys, audit = [str(i) for i in GUIDANCE_STEPS], dict(kind=kind)
    update = record.get('update_rms') or {}
    if kind in ('amf', 'random'):
        if sorted(update, key=int) != keys or not all(math.isfinite(v) and v > 0 for v in update.values()):
            raise ValueError(f'{arm}: guidance did not act at every step 0-9')
        audit['update_rms_mean'] = float(np.mean(list(update.values())))
    elif update:
        raise ValueError(f'{arm}: this arm must not change the latent during sampling')
    if kind == 'random' and any(not math.isclose(update[k], dose[k], rel_tol=1e-4) for k in keys):
        raise ValueError(f'{arm}: random perturbation size does not match the AMF arms')
    if kind == 'amf':
        losses = record.get('losses') or {}
        if sorted(losses, key=int) != keys or any(
                len(v) != UPDATES_PER_STEP or not all(math.isfinite(x) for x in v) for v in losses.values()):
            raise ValueError(f'{arm}: expected {UPDATES_PER_STEP} finite AMF losses at every guided step')
        audit.update(loss_first=losses['0'][0], loss_last=losses[keys[-1]][-1])
    if kind == 'sdedit' and record.get('sdedit_start') != SDEDIT_START:
        raise ValueError(f'{arm}: SDEdit start index changed')
    for name in ('original.mp4', 'final.mp4'):
        count = 0
        for frame in iio.imiter(directory / name, plugin='FFMPEG'):
            if frame.shape[:2] != (dm.SIZE[1], dm.SIZE[0]):
                raise ValueError(f'{arm}: unexpected resolution in {name}')
            count += 1
        if count != dm.FRAMES:
            raise ValueError(f'{arm}: {name} has {count} frames, expected {dm.FRAMES}')
    return audit


# ------------------------------------------------------------------ scoring
def score_videos(videos):
    """Pure decoded-video scoring. `videos` maps the seven arms plus reference_forward/_reverse to MP4 paths."""
    frames = {name: dm.read_frames(path) for name, path in videos.items()}
    flows = {name: dm.flow_field(value) for name, value in frames.items()}
    rf, rr = flows['reference_forward'], flows['reference_reverse']
    selectivity = dict(amf=dm.selectivity(flows['forward'], flows['reverse'], rf, rr),
                       random_null=dm.selectivity(flows['random_a'], flows['random_b'], rf, rr),
                       positive_control=dm.selectivity(flows['sdedit_forward'], flows['sdedit_reverse'], rf, rr))
    change = {arm: dict(dm.change_from(flows[arm], flows['off'], rr if arm in REVERSED else rf),
                        pixel_mae_255=float(np.abs(frames[arm].astype(float) - frames['off'].astype(float)).mean()))
              for arm in ARMS[1:]}
    return dict(selectivity=selectivity, change_from_off=change,
                reference_motion={k: float(np.linalg.norm(v, axis=-1).mean()) for k, v in
                                  (('forward', rf), ('reverse', rr))},
                verdict=dm.verdict(selectivity['amf'], selectivity['random_null'], selectivity['positive_control']))


def next_action(plan, decision):
    first = [plan['clip_id'], plan['seed']] == list(FIRST_CLIP)
    confirm = ', '.join(f'{c} seed {s}' for c, s in CONFIRMATION)
    if decision == 'INVALID':
        return ('The positive control did not separate the two references, so this run says nothing about AMF. '
                'Inspect the two SDEdit videos and the reference motion before any other run.')
    if decision == 'STOP':
        return ('Pre-registered decision: stop AMF guidance on Wan. At full DiTFlow strength it does not steer decoded '
                'motion on the most favourable clip. Do not run confirmation plans or new loss variants.' if first else
                'Pre-registered decision: a confirmation failed, so AMF guidance does not reliably steer motion on Wan. Stop.')
    return (f'Passed on the most favourable clip. Pre-registered next step: run the confirmation plans ({confirm}). '
            'The method counts as working only if both also pass.' if first else
            'This confirmation passed. The method counts as working once every confirmation plan passes.')


def summarize(root):
    root = Path(root).resolve()
    plan = checked_plan(root)
    audits = {arm: validate_arm(completed(root, arm), root, plan, arm) for arm in ARMS}
    directories = {arm: completed(root, arm) for arm in ARMS}
    videos = {arm: directories[arm] / 'final.mp4' for arm in ARMS}
    videos.update(reference_forward=directories['forward'] / 'original.mp4',
                  reference_reverse=directories['reverse'] / 'original.mp4')
    scores = score_videos(videos)
    decision = scores['verdict']['decision']
    report = dict(stage=STAGE, clip_id=plan['clip_id'], seed=plan['seed'], prompt=plan['prompt'], decision=decision,
                  next_action=next_action(plan, decision), **scores, arms=audits,
                  caution='Whole-frame decoded flow includes camera motion. A GO is necessary, not sufficient, '
                          'for subject motion transfer.')
    write_json(root / 'decisive_scores.json', report)
    with (root / 'decisive_scores.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['comparison', 'score', 'gain', 'separation_patches', 'target_magnitude_patches'])
        for name, value in scores['selectivity'].items():
            writer.writerow([name, *(value[k] for k in ('score', 'gain', 'separation', 'target_magnitude'))])
        writer.writerow([])
        writer.writerow(['arm_vs_off', 'alignment_with_own_reference', 'flow_change_patches', 'pixel_mae_255'])
        for arm, value in scores['change_from_off'].items():
            writer.writerow([arm, value['alignment'], value['magnitude'], value['pixel_mae_255']])
    write_verdict(root, report)
    contact_sheet(videos, root / 'contact_sheet.jpg')
    make_display(root, report, videos)
    return report


def _fmt(value):
    return 'n/a' if value is None or (isinstance(value, float) and not math.isfinite(value)) else f'{value:+.3f}'


def write_verdict(root, report):
    s, t = report['selectivity'], report['verdict']['thresholds']
    lines = [f"# Decisive Wan test: {report['decision']}", '',
             f"Clip `{report['clip_id']}`, seed {report['seed']}, prompt: {report['prompt']}", '',
             '| Comparison | Selectivity | Gain | Required |', '|---|---:|---:|---|',
             f"| AMF forward vs reverse | {_fmt(s['amf']['score'])} | {_fmt(s['amf']['gain'])} | "
             f"at least {t['amf_min']}, and {t['margin_over_null']} above the random pair |",
             f"| Random pair, equal dose | {_fmt(s['random_null']['score'])} | {_fmt(s['random_null']['gain'])} | null |",
             f"| SDEdit forward vs reverse | {_fmt(s['positive_control']['score'])} | "
             f"{_fmt(s['positive_control']['gain'])} | at least {t['positive_control_min']} for a valid run |", '',
             *[f'- {r}' for r in report['verdict']['reasons']], '', f"**Next:** {report['next_action']}", '',
             report['caution'], '']
    (Path(root) / 'verdict.md').write_text('\n'.join(lines), encoding='utf-8')


LABELS = dict(reference_forward='Reference', reference_reverse='Reversed reference', off='AMF off',
              forward='AMF toward reference', reverse='AMF toward reversed', random_a='Random, same dose (a)',
              random_b='Random, same dose (b)', sdedit_forward='SDEdit from reference',
              sdedit_reverse='SDEdit from reversed')
ORDER = ('reference_forward', 'reference_reverse', *ARMS)


def contact_sheet(videos, path, columns=(0, 5, 10, 15, 20), thumb=(208, 120)):
    from PIL import Image, ImageDraw
    label_width, gap = 190, 4
    sheet = Image.new('RGB', (label_width + len(columns) * (thumb[0] + gap), len(ORDER) * (thumb[1] + gap)), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, name in enumerate(ORDER):
        frames = dm.read_frames(videos[name])
        y = row * (thumb[1] + gap)
        draw.text((6, y + thumb[1] // 2 - 6), LABELS[name], fill='black')
        for col, index in enumerate(columns):
            sheet.paste(Image.fromarray(frames[index]).resize(thumb), (label_width + col * (thumb[0] + gap), y))
    sheet.save(path, quality=90)
    return path


def make_display(root, report, videos):
    def card(name):
        payload = base64.b64encode(Path(videos[name]).read_bytes()).decode('ascii')
        return (f'<figure><figcaption>{html.escape(LABELS[name])}</figcaption><video controls muted loop playsinline '
                f'src="data:video/mp4;base64,{payload}"></video></figure>')
    s = report['selectivity']
    rows = ''.join(f'<tr><td>{html.escape(label)}</td><td>{_fmt(s[key]["score"])}</td><td>{_fmt(s[key]["gain"])}</td></tr>'
                   for key, label in (('amf', 'AMF forward vs reverse'), ('random_null', 'Random pair, equal dose'),
                                      ('positive_control', 'SDEdit forward vs reverse')))
    colour = dict(GO='#1a7f37', STOP='#b42318', INVALID='#9a6700')[report['decision']]
    page = ['<!doctype html><meta charset="utf-8"><title>Decisive Wan test</title>',
            '<style>body{font:16px system-ui;margin:20px;color:#17202a}section{display:flex;flex-wrap:wrap;gap:12px}'
            'figure{margin:0;flex:1 1 360px;max-width:560px}video{width:100%}table{border-collapse:collapse}'
            'td,th{padding:6px 10px;border:1px solid #bbb;text-align:right}td:first-child{text-align:left}</style>',
            f'<h1 style="color:{colour}">{report["decision"]}</h1>',
            f'<p>{html.escape(report["clip_id"])}, seed {report["seed"]}. {html.escape(report["next_action"])}</p>',
            f'<table><tr><th>Comparison</th><th>Selectivity</th><th>Gain</th></tr>{rows}</table>',
            '<ul>' + ''.join(f'<li>{html.escape(r)}</li>' for r in report['verdict']['reasons']) + '</ul>',
            '<button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">Play all</button> '
            '<button onclick="document.querySelectorAll(\'video\').forEach(v=>v.pause())">Pause all</button>',
            '<section>' + ''.join(card(n) for n in ORDER) + '</section>',
            f'<p>{html.escape(report["caution"])}</p>']
    (Path(root) / 'decisive_comparison.html').write_text('\n'.join(page), encoding='utf-8')


def run(root):
    root = Path(root).resolve()
    plan = checked_plan(root)
    for arm in ARMS:
        command = [sys.executable, '-u', 'probe_wan_decisive.py', '--plan', str(root / 'plan.json'), '--arm', arm]
        run_arm(root, arm, command, lambda directory, arm=arm: validate_arm(directory, root, plan, arm))
    return summarize(root)
