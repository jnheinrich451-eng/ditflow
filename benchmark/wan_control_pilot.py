"""Matched directional AMF controls with actual sampler-response evidence."""
import base64
import csv
import html
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

from benchmark import wan_head_pilot as head
from benchmark import wan_noised_reference_pilot as noised
from benchmark import wan_pair_pilot as pairs
from probe_report import load_trace

STAGE = 'forward_adjacent_directional_control_v1'
ARMS = ('off', 'forward', 'reverse')
CHECKPOINTS = (9, 10, 19, 29, 39, 49)
read_json = noised.read_json
archive_experiment = head.archive_experiment


def inventory(directory):
    return {p.relative_to(directory).as_posix(): head.digest(p)
            for p in Path(directory).rglob('*') if p.is_file()}


def write_reversed_frames(video, destination):
    """Reverse decoded RGB frames before the causal VAE; never flip latents."""
    from PIL import Image
    from guidance_utils.wan_reference_diagnostics import reference_images
    frames = reference_images(video, 21, (832, 480))
    destination = Path(destination)
    destination.mkdir()
    for index, frame in enumerate(frames[::-1]):
        Image.fromarray(frame).save(destination / f'{index:05d}.png')
    return inventory(destination)


def make_plan(inputs, root, previous_run, archive=None):
    root, previous_run = Path(root).resolve(), Path(previous_run).resolve()
    if root == previous_run or previous_run in root.parents:
        raise ValueError('Create a fresh experiment outside the previous archive')
    previous, receipt = pairs.validate_previous(previous_run)
    pairs.require_reusable_environment(read_json(previous_run/'environment.json'), head.environment_snapshot())
    inputs, rows = head.direction.prepare_inputs(inputs, archive)
    if head.digest(inputs/'manifest.csv') != previous['input_manifest_sha256']:
        raise ValueError('Reuse the confirmed input manifest')
    video = inputs / next(r for r in rows if r['clip_id'] == 'car-turn')['video_path']
    with tempfile.TemporaryDirectory(prefix='wan_control_reference_') as temp:
        reversed_path = Path(temp)/'reverse'
        reverse_inventory = write_reversed_frames(video, reversed_path)
        plan = dict(schema_version=1, stage=STAGE, inputs=str(inputs), clip_id='car-turn',
            video=str(video), generation_video=str(video), reverse_video=str(root/'reference_reverse'),
            prompt=previous['prompt'], selection=dict(block=30, head=30), arms=list(ARMS),
            flow_pair_mode=pairs.PAIR_MODE, guidance_indices=[9], updates=5, learning_rate=.001,
            generation_seed=1, reference_noise_seed=29, checkpoints=list(CHECKPOINTS),
            input_manifest_sha256=head.digest(inputs/'manifest.csv'),
            reverse_inventory=reverse_inventory, previous_inventory=inventory(previous_run))
        head.save_plan(root, plan)
        shutil.copytree(reversed_path, root/'reference_reverse')
    shutil.copytree(previous_run, root/'previous')
    shutil.copyfile(inputs/'manifest.csv', root/'input_manifest.csv')
    head.write_json(root/'previous_audit.json', receipt)
    return plan


def checked_plan(root):
    root = Path(root); plan = head.load_plan(root)
    expected = dict(stage=STAGE, selection=dict(block=30, head=30), arms=list(ARMS),
        flow_pair_mode=pairs.PAIR_MODE, guidance_indices=[9], updates=5, learning_rate=.001,
        generation_seed=1, reference_noise_seed=29, checkpoints=list(CHECKPOINTS))
    if any(plan.get(k) != v for k, v in expected.items()):
        raise ValueError('Directional-control protocol changed; create a new experiment')
    head.checked_inputs(plan)
    if head.digest(Path(plan['inputs'])/'manifest.csv') != plan['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    if inventory(root/'reference_reverse') != plan['reverse_inventory']:
        raise ValueError('Reversed input frames changed')
    if inventory(root/'previous') != plan['previous_inventory']:
        raise ValueError('Previous confirmation/comparison evidence changed')
    return plan


def fixed_config(plan, arm, output):
    if arm not in ARMS:
        raise ValueError('Unknown control arm')
    adapted = dict(plan, generation_video=plan['reverse_video'] if arm == 'reverse' else plan['generation_video'])
    config = noised.fixed_config(adapted, 'off' if arm == 'off' else 'candidate', output)
    config.update(flow_pair_mode=pairs.PAIR_MODE, visual_protocol=STAGE, control_arm=arm)
    return config


def require_common_start(left, right):
    # Reference encoding differs deliberately in the reversed arm.
    for key in ('latent_sha256', 'conditioning_sha256', 'source_conditioning_sha256',
                'rope_sha256', 'timesteps', 'sigmas', 'model_revision'):
        if left[key] != right[key]:
            raise ValueError('Unmatched directional-control start: '+key)


def arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require_same_capture(left, right, keys=('latent', 'cond_velocity', 'uncond_velocity', 'cfg_velocity')):
    a, b = arrays(left), arrays(right)
    for key in keys:
        if not np.array_equal(a[key], b[key]):
            raise ValueError('Unmatched before-guidance '+key)


def require_directional_targets(reverse, forward):
    reverse, forward = Path(reverse), Path(forward)
    a, b = arrays(reverse/'reference_inputs.npz'), arrays(forward/'reference_inputs.npz')
    if not np.array_equal(a['noise'], b['noise']) or np.array_equal(a['clean'], b['clean']):
        raise ValueError('Reversal must change encoded reference while preserving matched noise')
    targets = []
    for directory in (reverse, forward):
        trace, _, events = load_trace(directory)
        ref = pairs.one_event(events, 'selected_pair_reference')
        targets.append(arrays(trace/ref['file']))
    common = targets[0]['mask'] & targets[1]['mask']
    if not common.any():
        raise ValueError('No shared reference support for directional comparison')
    if np.array_equal(targets[0]['flow'][common], targets[1]['flow'][common]):
        raise ValueError('Forward/reversed targets are identical on shared support; guidance cannot distinguish them')


def audit_capture(path, shape=(1, 16, 6, 60, 104), expected_block=30, expected_head=30):
    from diffusers import FlowMatchEulerDiscreteScheduler
    path = Path(path); values = arrays(path); meta = read_json(path.with_suffix('.json'))
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
    step = meta['step']
    if (meta['label'] not in ('before', 'after', 'denoise') or type(step) is not int or step not in CHECKPOINTS
            or path.stem != f"{meta['label']}_{step:02d}"
            or (meta['label'] != 'denoise' and step != 9)
            or meta['timestep'] != float(scheduler.timesteps[step])
            or meta['sigma'] != float(scheduler.sigmas[step])
            or meta['sigma_next'] != float(scheduler.sigmas[step+1]) or meta['guidance_scale'] != 5):
        raise ValueError('Changed checkpoint identity or sampler schedule')
    if set(meta['native']) != {'cond', 'uncond'}:
        raise ValueError('Missing native-head contribution')
    for result in meta['native'].values():
        for key in ('head_output_rms', 'projected_head_rms', 'projected_all_heads_rms', 'projected_rms_ratio'):
            value = result[key]
            if value is None and key == 'projected_rms_ratio' and result['projected_all_heads_rms'] == 0:
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError('Invalid native-head contribution')
    denoise = meta['label'] == 'denoise'
    keys = {'latent', 'cond_velocity', 'uncond_velocity', 'cfg_velocity', 'cond_frame_mass', 'uncond_frame_mass'}
    if denoise:
        keys.add('next_latent')
    if set(values) != keys or meta['block'] != expected_block or meta['head'] != expected_head:
        raise ValueError('Incomplete/mislabeled control capture')
    frames = shape[2]; tokens = frames*(shape[3]//2)*(shape[4]//2)
    for key, value in values.items():
        expected_shape = (tokens, frames) if key.endswith('frame_mass') else shape
        if value.shape != expected_shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError('Invalid control array: '+key)
        if key.endswith('frame_mass') and (np.any(value < 0) or not np.allclose(value.sum(-1), 1, atol=2e-6)):
            raise ValueError('Native temporal attention mass is not normalized')
    reconstructed = values['uncond_velocity'] + meta['guidance_scale']*(values['cond_velocity']-values['uncond_velocity'])
    if not np.array_equal(values['cfg_velocity'], reconstructed):
        raise ValueError('CFG prediction does not reconstruct')
    residual = None
    if denoise:
        expected = values['latent'] + np.float32(meta['sigma_next']-meta['sigma'])*reconstructed
        residual = float(np.abs(expected-values['next_latent']).max())
        if not np.allclose(expected, values['next_latent'], rtol=1e-6, atol=1e-6):
            raise ValueError('Sampler did not consume the captured latent/CFG velocity')
    return dict(step=meta['step'], label=meta['label'], euler_max_residual=residual, native=meta['native'])


def validate_result(directory, root, plan, arm):
    import imageio.v3 as iio
    from omegaconf import OmegaConf
    directory, root = Path(directory), Path(root)
    trace, metadata, events = load_trace(directory)
    config = metadata['config']
    if any(config.get(k) != v for k, v in fixed_config(plan, arm, config['output_path']).items()):
        raise ValueError('Control configuration changed')
    if config != OmegaConf.to_container(OmegaConf.load(directory/'suite_config.yaml')):
        raise ValueError('Configuration changed after initialization')
    if read_json(directory/'complete.json') != dict(arm=arm, native_rope_unchanged=True, frozen_weights=True):
        raise ValueError('Incomplete generation or modified pretrained weights/RoPE')
    report = noised.audit_trace(events, config, 'off' if arm == 'off' else 'candidate')
    if arm != 'off':
        report['selected_loss'] = pairs.audit_selected_loss(trace, events)
        provenance = read_json(directory/'reference_protocol.json')
        noised.audit_reference_inputs(directory/'reference_inputs.npz', provenance)
        if provenance['mode'] != 'matched_noised' or provenance['step'] != 9 or provenance['noise_seed'] != 29:
            raise ValueError('Reference noise protocol changed')
    expected = {'before_09', *(f'denoise_{step:02d}' for step in CHECKPOINTS)}
    if arm != 'off':
        expected.add('after_09')
    if {p.stem for p in (directory/'control').glob('*.npz')} != expected:
        raise ValueError('Missing/extra paired sampler checkpoints')
    report['captures'] = [audit_capture(directory/'control'/f'{name}.npz') for name in sorted(expected)]
    require_same_capture(directory/'control'/('before_09.npz' if arm == 'off' else 'after_09.npz'),
                         directory/'control/denoise_09.npz')
    if arm == 'forward':
        pairs.require_reference_match(directory, root/'previous')
        pairs.require_before_update_match(directory, root/'previous')
    if arm != 'off':
        off = pairs.completed(root, 'off')
        require_common_start(read_json(directory/'initial_state.json'), read_json(off/'initial_state.json'))
        require_same_capture(directory/'control/before_09.npz', off/'control/before_09.npz')
    if arm == 'reverse':
        require_directional_targets(directory, pairs.completed(root, 'forward'))
    files = ['original.mp4', 'final.mp4', 'estimated_clean_before.mp4']
    if arm != 'off':
        files.append('estimated_clean_after.mp4')
    for name in files:
        count = 0
        for frame in iio.imiter(directory/name, plugin='FFMPEG'):
            if frame.shape[:2] != (480, 832):
                raise ValueError('Unexpected video resolution')
            count += 1
        if count != 21:
            raise ValueError('Incomplete video: '+name)
    return report


def response_rows(root):
    """Paired differences at the SAME sampler index, not cross-noise loss ratios."""
    root = Path(root); off = pairs.completed(root, 'off')
    rows = []
    rms = lambda a: float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))
    for arm in ('forward', 'reverse'):
        directory = pairs.completed(root, arm)
        pre = arrays(directory/'control/before_09.npz')['latent'].astype(float)
        post = arrays(directory/'control/after_09.npz')['latent'].astype(float)
        update = post-pre; norm2 = float(np.sum(update**2))
        for step in CHECKPOINTS:
            a = arrays(directory/'control'/f'denoise_{step:02d}.npz')
            b = arrays(off/'control'/f'denoise_{step:02d}.npz')
            meta = read_json(directory/'control'/f'denoise_{step:02d}.json')
            delta = a['latent'].astype(float)-b['latent']
            velocity = a['cfg_velocity'].astype(float)-b['cfg_velocity']
            next_delta = a['next_latent'].astype(float)-b['next_latent']
            predicted_delta = delta+(meta['sigma_next']-meta['sigma'])*velocity
            rows.append(dict(arm=arm, step=step, sigma=meta['sigma'], guidance_update_rms=rms(update),
                latent_difference_before_sampler_rms=rms(delta), cfg_velocity_difference_rms=rms(velocity),
                cond_velocity_difference_rms=rms(a['cond_velocity'].astype(float)-b['cond_velocity']),
                uncond_velocity_difference_rms=rms(a['uncond_velocity'].astype(float)-b['uncond_velocity']),
                latent_difference_after_sampler_rms=rms(next_delta),
                difference_to_initial_update_rms_ratio=rms(next_delta)/rms(update) if norm2 else None,
                projection_on_initial_update=float(np.sum(next_delta*update)/norm2) if norm2 else None,
                paired_euler_residual_rms=rms(predicted_delta-next_delta)))
    return rows


def target_selectivity(root):
    """Score both references on their common support at fixed step 9."""
    targets, captures = {}, {}
    for arm in ('forward', 'reverse'):
        trace, _, events = load_trace(pairs.completed(root, arm))
        ref = pairs.one_event(events, 'selected_pair_reference')
        targets[arm] = arrays(trace/ref['file'])
        target_events = [e for e in events if e['kind'] == 'full_pair_target']
        captures[arm] = {label: arrays(trace/e['file'])['flow'] for label, e in
                         [('before', target_events[0]), ('after', target_events[-1])]}
    if np.array_equal(targets['forward']['flow'], targets['reverse']['flow']):
        raise ValueError('Forward and reversed reference targets are identical')
    mask = targets['forward']['mask'] & targets['reverse']['mask']
    if not mask.any():
        raise ValueError('No shared reference support for selectivity')
    rows = []
    for arm, stages in captures.items():
        for stage, flow in stages.items():
            errors = {name: float(np.square(flow.astype(float)-target['flow'])[mask].mean())
                      for name, target in targets.items()}
            rows.append(dict(arm=arm, stage=stage, common_positions=int(mask.sum()),
                forward_mse=errors['forward'], reverse_mse=errors['reverse'],
                forward_preference=errors['reverse']-errors['forward']))
    return rows


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def native_rows(root):
    rows = []
    for arm in ARMS:
        directory = pairs.completed(root, arm)
        for path in sorted((directory/'control').glob('*.npz')):
            value, meta = arrays(path), read_json(path.with_suffix('.json'))
            frames = value['latent'].shape[2]
            for branch in ('cond', 'uncond'):
                mass = value[branch+'_frame_mass'].reshape(frames, -1, frames)
                rows.append(dict(arm=arm, label=meta['label'], step=meta['step'], branch=branch,
                    same_frame_mass=float(np.mean([mass[i, :, i].mean() for i in range(frames)])),
                    forward_adjacent_frame_mass=float(np.mean([mass[i, :, i+1].mean() for i in range(frames-1)])),
                    **meta['native'][branch]))
    return rows


def summarize(root):
    root = Path(root); plan = checked_plan(root)
    audits = {arm: validate_result(pairs.completed(root, arm), root, plan, arm) for arm in ARMS}
    response, selectivity, native = response_rows(root), target_selectivity(root), native_rows(root)
    write_csv(root/'sampler_response.csv', response)
    write_csv(root/'target_selectivity.csv', selectivity)
    write_csv(root/'native_attention.csv', native)
    report = dict(arms=audits, response=response, selectivity=selectivity, native=native,
        caution='Latent differences and AMF preference do not establish physical motion quality.')
    head.write_json(root/'control_audit.json', report)
    review = root/'visual_review.csv'
    if not review.exists():
        write_csv(review, [dict(arm=a, heading='', lateral_motion='', size_change='', turn='',
            background_motion='', appearance='', notes='') for a in ARMS])
    make_display(root, report)
    return report


def make_display(root, report):
    root = Path(root)
    def card(label, path):
        payload = base64.b64encode(path.read_bytes()).decode('ascii')
        return '<figure><figcaption>'+html.escape(label)+'</figcaption><video controls muted loop playsinline src="data:video/mp4;base64,'+payload+'"></video></figure>'
    def table(rows, keys):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in keys)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+html.escape(f'{row[k]:.6g}' if isinstance(row[k], float) else str(row[k]))+'</td>'
                          for k in keys)+'</tr>' for row in rows)+'</table>'
    directories = {arm: pairs.completed(root, arm) for arm in ARMS}
    page = ['<!doctype html><meta charset="utf-8"><title>Wan AMF control</title>',
        '<style>body{font:16px system-ui;color:#17202a;background:#f7f8fa;margin:20px}section{display:flex;flex-wrap:wrap;gap:12px}'
        'figure{margin:0;flex:1 1 380px;max-width:832px}video{width:100%}figcaption{padding:8px 0}'
        'table{border-collapse:collapse}td,th{padding:8px;border:1px solid #aaa}</style>',
        '<h1>Does AMF control the generated movement?</h1><p>All generated videos use Wan2.1 T2V 14B. '
        'Matched prompt and seed; head 30; forward-adjacent loss; five updates at step 9.</p>',
        '<button onclick="document.querySelectorAll(\'#finals video\').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button> '
        '<button onclick="document.querySelectorAll(\'#finals video\').forEach(v=>v.pause())">Pause all</button><section id="finals">',
        card('Forward motion reference', directories['forward']/'original.mp4'),
        card('Reversed motion reference', directories['reverse']/'original.mp4'),
        *[card('Wan: '+label, directories[arm]/'final.mp4') for arm, label in
          [('off', 'AMF off'), ('forward', 'forward-reference guidance'), ('reverse', 'reversed-reference guidance')]],
        card('Previous Wan: head 30, all-pair loss', pairs.completed(root/'previous', 'candidate')/'final.mp4'),
        '</section><h2>Immediate model predictions</h2><p>High-noise estimates, not final videos.</p><section>',
        card('Shared pre-update estimate', directories['off']/'estimated_clean_before.mp4'),
        card('After forward guidance', directories['forward']/'estimated_clean_after.mp4'),
        card('After reverse guidance', directories['reverse']/'estimated_clean_after.mp4'),
        '</section><h2>Paired sampler response</h2><p>Differences from AMF off at the same step. '
        'These measure influence, not motion quality; ratios are not percentages of correct motion retained.</p>',
        table(report['response'], ['arm', 'step', 'cfg_velocity_difference_rms', 'latent_difference_after_sampler_rms',
                                  'projection_on_initial_update']),
        '<h2>Reference selectivity inside AMF</h2><p>Positive preference means closer to the forward target '
        'on shared support. Compare decoded heading, lateral movement, growth and turning separately.</p>',
        table(report['selectivity'], ['arm', 'stage', 'forward_mse', 'reverse_mse', 'forward_preference']),
        '<h2>Native head at the intervention</h2><p>Conditional branch. Attention mass is reconstructed in FP32 '
        'without AMF sharpening. Projected RMS ratio is not a share of causal influence: heads can cancel.</p>',
        table([row for row in report['native'] if row['step'] == 9 and row['branch'] == 'cond' and row['label'] != 'denoise'],
              ['arm', 'label', 'same_frame_mass', 'forward_adjacent_frame_mass', 'projected_rms_ratio'])]
    (root/'control_comparison.html').write_text('\n'.join(page), encoding='utf-8')


def run(root):
    root = Path(root).resolve(); plan = checked_plan(root)
    for arm in ARMS:
        command = [sys.executable, '-u', 'probe_wan_control.py', '--plan', str(root/'plan.json'), '--arm', arm]
        head.run_process(root, arm, command, lambda directory, arm=arm: validate_result(directory, root, plan, arm))
    return summarize(root)
