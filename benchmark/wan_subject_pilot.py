"""Frozen coarse-alignment experiment with matched prior controls and raw audits."""
import base64
import hashlib
import html
import json
import math
import shutil
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image

from benchmark import wan_control_pilot as control
from benchmark import wan_head_pilot as head
from benchmark import wan_pair_pilot as pairs
from guidance_utils.wan_subject_alignment import aligned_targets
from probe_report import load_trace

STAGE = 'coarse_subject_alignment_step9_v1'
ARMS = ('aligned_forward', 'aligned_reverse', 'balanced_forward', 'balanced_reverse')
ANNOTATION = Path('configs/wan_subject_alignment_car_turn.json')
read_json, arrays, inventory = control.read_json, control.arrays, control.inventory
archive_experiment = head.archive_experiment
PROTOCOL = dict(stage=STAGE, arms=list(ARMS), selection=dict(block=30, head=30),
                guidance_indices=[9], updates=5, learning_rate=.001, generation_seed=1,
                reference_noise_seed=29, subject_weight=.5, anchors=[0, 4, 8, 12, 16, 20])


def environment_snapshot():
    value = head.environment_snapshot()
    for name in ('benchmark/wan_subject_pilot.py', 'probe_wan_subject.py',
                 'guidance_utils/wan_subject_alignment.py', ANNOTATION.as_posix()):
        value['source_sha256'][name] = head.digest(name)
    return value


def restore_control_archive(archive, destination):
    """Restore the outer control experiment, never its nested older experiment."""
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.wan_subject_restore_', dir=destination.parent) as temp:
        staging = Path(temp); local_zip = staging/'results.zip'
        shutil.copyfile(Path(archive).expanduser(), local_zip)
        with zipfile.ZipFile(local_zip) as bundle:
            files = {}
            for info in bundle.infolist():
                name = PurePosixPath(info.filename.replace('\\', '/'))
                if (name.is_absolute() or '..' in name.parts or any(':' in part for part in name.parts)
                        or stat.S_ISLNK(info.external_attr >> 16) or name in files):
                    raise ValueError('Unsafe or duplicate archive member: '+info.filename)
                if not info.is_dir():
                    files[name] = info
            markers = ('plan.sha256', 'environment.json', 'off_done.json', 'forward_done.json', 'reverse_done.json')
            roots = [p.parent for p in files if p.name == 'plan.json'
                     and all(p.parent/m in files for m in markers)
                     and json.loads(bundle.read(files[p]))['stage'] == control.STAGE]
            if len(roots) != 1:
                raise ValueError('Use the completed CONTROL results ZIP with off/forward/reverse completion records')
            restored = staging/'result'; restored.mkdir()
            for name, info in files.items():
                if name.is_relative_to(roots[0]):
                    target = restored.joinpath(*name.relative_to(roots[0]).parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.open(info) as source, target.open('wb') as output:
                        shutil.copyfileobj(source, output)
        expected = inventory(restored)
        if destination.exists():
            if destination.is_dir() and inventory(destination) == expected:
                return destination
            identity = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()[:16]
            destination = destination.with_name(destination.name+'_restored_'+identity)
            if destination.exists():
                if destination.is_dir() and inventory(destination) == expected:
                    return destination
                raise ValueError('Restored cache differs; choose a fresh local destination')
        if destination.parent.resolve() != staging.parent.resolve() or destination.exists():
            raise ValueError('Restore requires an unused sibling of the local staging directory')
        restored.rename(destination)
    print('Restored control evidence locally:', destination)
    return destination


def validate_previous(previous):
    previous = Path(previous); plan = head.load_plan(previous)
    if plan['stage'] != control.STAGE:
        raise ValueError('Use the completed directional-control experiment')
    if inventory(previous/'previous') != plan['previous_inventory'] or inventory(previous/'reference_reverse') != plan['reverse_inventory']:
        raise ValueError('Changed previous evidence or reversed reference')
    _, confirmation = pairs.validate_previous(previous/'previous')
    reports = {arm: control.validate_result(pairs.completed(previous, arm), previous, plan, arm) for arm in control.ARMS}
    return plan, dict(arms=reports, confirmation=confirmation)


def prepare_targets(inputs, previous, destination, annotation=ANNOTATION):
    """Offline, deterministic map preview. No model and no environment mutation."""
    inputs, previous, destination = Path(inputs), Path(previous), Path(destination)
    note = read_json(annotation)
    off = pairs.completed(previous, 'off')
    if (note['schema_version'] != 1 or note['anchor_frames'] != PROTOCOL['anchors'] or note['image_size'] != [832, 480]
            or head.digest(off/'final.mp4') != note['baseline_video_sha256']
            or read_json(off/'initial_state.json')['latent_sha256'] != note['baseline_initial_latent_sha256']):
        raise ValueError('Frozen truck boxes do not match this baseline')
    files = sorted((inputs/'masks/car-turn').glob('*.png'))
    if [p.name for p in files] != [f'{i:05d}.png' for i in range(21)]:
        raise ValueError('Need all 21 aligned car-turn reference masks')
    full_masks = [np.asarray(Image.open(p)) > 0 for p in files]
    destination.mkdir(parents=True, exist_ok=True)
    head.write_json(destination/'annotation.json', note)
    receipt = dict(method='source-frame nearest-patch remapping with displacement scaling',
                   source_masks={p.name: head.digest(p) for p in files}, directions={})
    for direction in ('forward', 'reverse'):
        trace, _, events = load_trace(pairs.completed(previous, direction))
        source = arrays(trace/pairs.one_event(events, 'selected_pair_reference')['file'])
        masks = full_masks if direction == 'forward' else full_masks[::-1]
        boxes, regions = [], []
        for anchor in note['anchor_frames']:
            mask = masks[anchor]; ys, xs = np.where(mask)
            if not len(xs):
                raise ValueError('Empty reference subject annotation')
            scale = np.array([832/mask.shape[1], 480/mask.shape[0]])
            boxes.append(np.r_[np.array([xs.min(), ys.min()])*scale, np.array([xs.max()+1, ys.max()+1])*scale])
            regions.append(np.asarray(Image.fromarray(mask.astype(np.float32)).resize((52, 30), Image.Resampling.BOX)) >= .1)
        target = aligned_targets(source['flow'], source['mask'], np.stack(regions), boxes,
                                 note['generated_boxes_xyxy'], note['image_size'])
        counts = [{key: int(target[key][p].sum()) for key in ('subject', 'background')} for p in pairs.adjacent_pairs(6)]
        if any(c['subject'] < 16 or c['background'] < 64 for c in counts):
            raise ValueError('Insufficient retained aligned support in a forward-adjacent pair')
        np.savez_compressed(destination/f'{direction}.npz', **target, source_flow=source['flow'], source_mask=source['mask'])
        receipt['directions'][direction] = dict(pair_support=counts, scales=target['scales'].tolist())
    head.write_json(destination/'mapping.json', receipt)
    make_preview(previous, destination)
    return receipt


def make_preview(previous, destination):
    """Small portable preview; frozen boxes are explicitly labeled as coarse regions."""
    import io
    import imageio.v3 as iio
    from PIL import ImageDraw
    previous, destination = Path(previous), Path(destination)
    note = read_json(destination/'annotation.json')
    frames = list(iio.imiter(pairs.completed(previous, 'off')/'final.mp4', plugin='FFMPEG'))
    page = ['<!doctype html><meta charset="utf-8"><title>Frozen subject alignment</title>',
            '<style>body{font:16px system-ui;margin:24px;color:#182434}img{max-width:100%}figure{margin:12px 0}</style>',
            '<h1>Frozen coarse truck alignment</h1><p>Completed AMF-off video; boxes never follow guided outputs. '
            'Yellow: frozen truck box. Cyan: retained warped reference-subject samples. '
            'These are coarse regions, not pixel segmentation or body-part correspondence. Frame 0 has a baseline artifact.</p>']
    for direction in ('forward', 'reverse'):
        data = arrays(destination/f'{direction}.npz')
        sheet = Image.new('RGB', (832*3, 520*2), 'white'); draw = ImageDraw.Draw(sheet)
        for i, anchor in enumerate(note['anchor_frames']):
            frame = Image.fromarray(frames[anchor]).convert('RGBA')
            layer = Image.new('RGBA', frame.size, (0, 0, 0, 0))
            if i < 5:
                mask = Image.fromarray(data['subject'][i*7+1].reshape(30, 52)).resize((832, 480), Image.Resampling.NEAREST)
                color = Image.new('RGBA', frame.size, (0, 220, 235, 100)); layer.paste(color, (0, 0), mask)
            frame = Image.alpha_composite(frame, layer).convert('RGB')
            ImageDraw.Draw(frame).rectangle(note['generated_boxes_xyxy'][i], outline='#ffbf00', width=3)
            x, y = (i % 3)*832, (i // 3)*520
            sheet.paste(frame, (x, y+35)); draw.text((x+12, y+10), f'{direction}: frame {anchor}', fill='black')
        buffer = io.BytesIO(); sheet.save(buffer, format='JPEG', quality=85)
        (destination/f'{direction}_preview.jpg').write_bytes(buffer.getvalue())
        page.append('<h2>'+direction+'</h2><img src="data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()+'">')
    (destination/'alignment_preview.html').write_text('\n'.join(page), encoding='utf-8')


def make_plan(inputs, root, previous, input_zip=None):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    if root == previous or previous in root.parents or root in previous.parents:
        raise ValueError('Create a fresh experiment separate from previous evidence')
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh setup directory; restore the existing path only to resume')
    old_plan, audit = validate_previous(previous)
    env = environment_snapshot()
    pairs.require_reusable_environment(read_json(previous/'environment.json'), env)
    inputs, rows = head.direction.prepare_inputs(inputs, input_zip)
    if head.digest(inputs/'manifest.csv') != old_plan['input_manifest_sha256']:
        raise ValueError('Reference input manifest changed')
    car = next(row for row in rows if row['clip_id'] == 'car-turn')
    with tempfile.TemporaryDirectory(prefix='wan_subject_map_') as tmp:
        prepare_targets(inputs, previous, tmp)
        plan = dict(PROTOCOL, inputs=str(inputs), clip_id='car-turn',
            video=str(inputs/car['video_path']), generation_video=str(inputs/car['video_path']),
            reverse_video=str(root/'previous/reference_reverse'), prompt=old_plan['prompt'],
            input_manifest_sha256=old_plan['input_manifest_sha256'],
            previous_inventory=inventory(previous), target_inventory=inventory(tmp),
            target_root=str(root/'targets'), annotation_sha256=head.digest(ANNOTATION))
        root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(previous, root/'previous')
        shutil.copytree(tmp, root/'targets')
    head.write_json(root/'plan.json', plan)
    (root/'plan.sha256').write_text(head.digest(root/'plan.json'), encoding='utf-8')
    head.write_json(root/'environment.json', env)
    head.write_json(root/'previous_audit.json', audit)
    shutil.copyfile(inputs/'manifest.csv', root/'input_manifest.csv')
    return plan


def checked_plan(root):
    root = Path(root); plan = head.load_plan(root)
    if any(plan.get(k) != v for k, v in PROTOCOL.items()):
        raise ValueError('Subject-alignment protocol changed; start a fresh setup')
    if inventory(root/'previous') != plan['previous_inventory'] or inventory(root/'targets') != plan['target_inventory']:
        raise ValueError('Previous evidence or frozen alignment targets changed')
    inputs, _ = head.checked_inputs(plan)
    if head.digest(inputs/'manifest.csv') != plan['input_manifest_sha256']:
        raise ValueError('Reference input manifest changed')
    if Path(plan['target_root']).resolve() != (root/'targets').resolve() or Path(plan['reverse_video']).resolve() != (root/'previous/reference_reverse').resolve():
        raise ValueError('Restore the experiment at its original runtime path')
    return plan


def direction_of(arm):
    if arm not in ARMS:
        raise ValueError('Unknown subject arm')
    return arm.rsplit('_', 1)[1]


def fixed_config(plan, arm, output):
    direction = direction_of(arm)
    config = control.fixed_config(plan, direction, output)
    config.update(visual_protocol=STAGE, alignment_mode='balanced' if arm.startswith('balanced_') else 'uniform',
        alignment_target=str(Path(plan['target_root'])/f'{direction}.npz'), record_region_gradients=True)
    return config


def require_previous_reference(directory, root, arm):
    directory, root = Path(directory), Path(root)
    old = pairs.completed(root/'previous', direction_of(arm))
    control.require_common_start(read_json(directory/'initial_state.json'), read_json(old/'initial_state.json'))
    if (read_json(directory/'reference_protocol.json') != read_json(old/'reference_protocol.json')
            or not pairs.npz_equal(directory/'reference_inputs.npz', old/'reference_inputs.npz')):
        raise ValueError('Reference encoding/noise changed')
    for name in ('training_reference', 'selected_pair_reference'):
        a, _, ae = load_trace(directory); b, _, be = load_trace(old)
        if not pairs.npz_equal(a/pairs.one_event(ae, name)['file'], b/pairs.one_event(be, name)['file']):
            raise ValueError('Source reference AMF/mask changed')


def require_previous_start(directory, root, arm):
    directory, root = Path(directory), Path(root)
    old = pairs.completed(root/'previous', direction_of(arm))
    if read_json(directory/'step9_state.json') != read_json(old/'step9_state.json'):
        raise ValueError('Pre-update latent or scheduler changed')
    control.require_same_capture(directory/'control/before_09.npz', old/'control/before_09.npz')
    a, _, ae = load_trace(directory); b, _, be = load_trace(old)
    af = next(e for e in ae if e['kind'] == 'full_pair_target')
    bf = next(e for e in be if e['kind'] == 'full_pair_target')
    if not pairs.npz_equal(a/af['file'], b/bf['file']):
        raise ValueError('Pre-update target AMF changed')
    if head.timing.decoded_digest(directory/'estimated_clean_before.mp4') != head.timing.decoded_digest(old/'estimated_clean_before.mp4'):
        raise ValueError('Pre-update decoded estimate changed')


def region_scores(flow, target, mode):
    error = np.square(flow.astype(float)-target['flow']).mean(-1)
    fg, bg = target['subject'], target['background']
    if fg.shape != error.shape or bg.shape != error.shape or np.any(fg & bg) or not fg.any() or not bg.any():
        raise ValueError('Invalid aligned region support')
    sf, sb = float(error[fg].mean()), float(error[bg].mean())
    weight = .5 if mode == 'balanced' else float(fg.sum())/float((fg | bg).sum())
    return dict(subject_mse=sf, background_mse=sb, total=weight*sf+(1-weight)*sb, subject_weight=weight)


def audit_losses(directory, target_path, mode):
    trace, _, events = load_trace(directory)
    target = arrays(target_path)
    ref = pairs.one_event(events, 'aligned_reference')
    if ref['mode'] != mode or not pairs.npz_equal(trace/ref['file'], target_path):
        raise ValueError('Aligned reference differs from frozen target')
    captures = [e for e in events if e['kind'] == 'full_pair_target']
    records = [e for e in events if e['kind'] == 'aligned_loss']
    totals = [e for e in events if e['kind'] == 'full_pair_loss']
    if len(captures) != 7 or len(records) != 7 or len(totals) != 7:
        raise ValueError('Missing aligned loss evidence')
    results = []
    for capture, recorded, total in zip(captures, records, totals):
        if (capture['file'] != recorded['file'] or capture['file'] != total['file'] or recorded['mode'] != mode
                or capture['flow_head'] != 30 or capture['q_dtype'] != 'torch.bfloat16' or capture['k_dtype'] != 'torch.bfloat16'):
            raise ValueError('Changed aligned loss identity or precision')
        flow = arrays(trace/capture['file'])['flow']
        if flow.shape != target['flow'].shape or not np.isfinite(flow).all():
            raise ValueError('Invalid target AMF capture')
        score = region_scores(flow, target, mode)
        for key, value in score.items():
            if not math.isclose(value, recorded[key], rel_tol=2e-5, abs_tol=2e-5):
                raise ValueError('Aligned loss failed recomputation: '+key)
        if total['loss'] != recorded['total']:
            raise ValueError('Optimized total differs from aligned objective')
        results.append(score)
    return results


def rms(a):
    return float(np.sqrt(np.mean(np.square(np.asarray(a, dtype=float)))))


def cosine(a, b):
    a, b = np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()
    norm = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a, b)/norm) if norm else None


def gradient_rows(directory, mode):
    directory = Path(directory)
    trace, _, events = load_trace(directory)
    updates = [e for e in events if e['kind'] == 'optimization']
    gradient_events = [e for e in events if e['kind'] == 'region_gradient']
    paths = sorted((directory/'region_gradients').glob('*.npz'))
    if len(paths) != 5 or len(updates) != 5 or len(gradient_events) != 5:
        raise ValueError('Missing region gradients')
    values = [arrays(p) for p in paths]
    target = arrays(trace/pairs.one_event(events, 'aligned_reference')['file'])
    expected_weight = .5 if mode == 'balanced' else float(target['subject'].sum())/float(target['mask'].sum())
    initial = arrays(directory/'control/before_09.npz')['latent']
    final = arrays(directory/'control/after_09.npz')['latent']
    if not np.array_equal(values[0]['latent'], initial):
        raise ValueError('Gradient capture does not start at the matched latent')
    rows = []
    for i, value in enumerate(values):
        if set(value) != {'latent', 'subject_gradient', 'background_gradient'}:
            raise ValueError('Malformed region-gradient arrays')
        if any(a.dtype != np.float32 or a.shape != initial.shape or not np.isfinite(a).all() for a in value.values()):
            raise ValueError('Invalid region-gradient array')
        event = gradient_events[i]
        if (event['iteration'] != i or event['step'] != 9 or directory/event['file'] != paths[i]
                or paths[i].name != f'iteration_{i:02d}.npz'
                or event['subject_weight'] != expected_weight or event['background_weight'] != 1-expected_weight):
            raise ValueError('Mislabeled region-gradient capture')
        next_latent = values[i+1]['latent'] if i < 4 else final
        update = next_latent.astype(float)-value['latent']
        if not math.isclose(rms(update), updates[i]['update']['rms'], rel_tol=2e-5, abs_tol=1e-8):
            raise ValueError('Region-gradient trajectory disagrees with actual Adam updates')
        fg, bg = value['subject_gradient'], value['background_gradient']
        rows.append(dict(iteration=i, subject_gradient_rms=rms(fg), background_gradient_rms=rms(bg),
            subject_weight=event['subject_weight'], background_weight=event['background_weight'],
            weighted_subject_gradient_rms=event['subject_weight']*rms(fg),
            weighted_background_gradient_rms=event['background_weight']*rms(bg),
            gradient_cosine=cosine(fg, bg), update_rms=rms(update),
            update_cosine_to_negative_subject_gradient=cosine(update, -fg),
            update_cosine_to_negative_background_gradient=cosine(update, -bg)))
    return rows


def validate_result(directory, root, plan, arm):
    import imageio.v3 as iio
    from omegaconf import OmegaConf
    directory, root = Path(directory), Path(root)
    trace, metadata, events = load_trace(directory); config = metadata['config']
    if (any(config.get(k) != v for k, v in fixed_config(plan, arm, config['output_path']).items())
            or config != OmegaConf.to_container(OmegaConf.load(directory/'suite_config.yaml'))
            or read_json(directory/'complete.json') != dict(arm=arm, native_rope_unchanged=True, frozen_weights=True)):
        raise ValueError('Changed configuration or incomplete generation')
    report = control.noised.audit_trace(events, config, 'candidate')
    target = root/'targets'/f'{direction_of(arm)}.npz'
    report['losses'] = audit_losses(directory, target, config['alignment_mode'])
    expected = {'before_09', 'after_09', *(f'denoise_{i:02d}' for i in control.CHECKPOINTS)}
    if {p.stem for p in (directory/'control').glob('*.npz')} != expected:
        raise ValueError('Missing or extra sampler checkpoints')
    report['captures'] = [control.audit_capture(directory/'control'/f'{name}.npz') for name in sorted(expected)]
    control.require_same_capture(directory/'control/after_09.npz', directory/'control/denoise_09.npz')
    require_previous_reference(directory, root, arm); require_previous_start(directory, root, arm)
    report['gradients'] = gradient_rows(directory, config['alignment_mode'])
    for name in ('original.mp4', 'final.mp4', 'estimated_clean_before.mp4', 'estimated_clean_after.mp4'):
        frames = list(iio.imiter(directory/name, plugin='FFMPEG'))
        if len(frames) != 21 or any(f.shape != (480, 832, 3) for f in frames):
            raise ValueError('Incomplete or malformed video: '+name)
    return report


def response_rows(root):
    root = Path(root); off = pairs.completed(root/'previous', 'off'); rows = []
    for arm in ARMS:
        directory = pairs.completed(root, arm)
        update = arrays(directory/'control/after_09.npz')['latent'].astype(float)-arrays(directory/'control/before_09.npz')['latent']
        for step in control.CHECKPOINTS:
            a = arrays(directory/'control'/f'denoise_{step:02d}.npz'); b = arrays(off/'control'/f'denoise_{step:02d}.npz')
            meta = read_json(directory/'control'/f'denoise_{step:02d}.json')
            delta = a['latent'].astype(float)-b['latent']; velocity = a['cfg_velocity'].astype(float)-b['cfg_velocity']
            next_delta = a['next_latent'].astype(float)-b['next_latent']
            rows.append(dict(arm=arm, step=step, guidance_update_rms=rms(update), cfg_velocity_difference_rms=rms(velocity),
                latent_difference_after_sampler_rms=rms(next_delta), difference_relative_to_off_rms=rms(next_delta)/rms(b['next_latent']),
                projection_on_initial_update=float(np.sum(next_delta*update)/np.sum(update**2)),
                paired_euler_residual_rms=rms(delta+(meta['sigma_next']-meta['sigma'])*velocity-next_delta)))
    return rows


def comparison_rows(root):
    root = Path(root)
    targets = {d: arrays(root/'targets'/f'{d}.npz') for d in ('forward', 'reverse')}
    folders = {**{'previous_'+d: pairs.completed(root/'previous', d) for d in ('forward', 'reverse')},
               **{arm: pairs.completed(root, arm) for arm in ARMS}}
    rows = []
    for arm, directory in folders.items():
        trace, _, events = load_trace(directory)
        fields = [e for e in events if e['kind'] == 'full_pair_target']
        for label, event in [('before', fields[0]), ('after', fields[-1])]:
            flow = arrays(trace/event['file'])['flow']
            for region in ('subject', 'background'):
                common = targets['forward'][region] & targets['reverse'][region]
                if not common.any():
                    raise ValueError('No common support for directional region comparison')
                error = {d: float(np.square(flow.astype(float)-t['flow'])[common].mean()) for d, t in targets.items()}
                rows.append(dict(arm=arm, stage=label, region=region, common_positions=int(common.sum()),
                    forward_mse=error['forward'], reverse_mse=error['reverse'], forward_preference=error['reverse']-error['forward']))
    return rows


def make_display(root, report):
    root = Path(root)
    def card(label, path):
        return '<figure><figcaption>'+html.escape(label)+'</figcaption><video controls muted loop playsinline src="data:video/mp4;base64,'+base64.b64encode(Path(path).read_bytes()).decode()+'"></video></figure>'
    page = ['<!doctype html><meta charset="utf-8"><title>Wan subject alignment</title>',
        '<style>body{font:16px system-ui;background:#f7f8fa;color:#17202a;margin:24px}section{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;flex:1 1 380px;max-width:832px}video{width:100%}figcaption{padding:10px 0}td,th{border:1px solid #aaa;padding:8px}table{border-collapse:collapse}</style>',
        '<h1>Does subject alignment improve motion control?</h1><p>All generated videos are Wan 14B. '
        'Same head 30, seed 1, LR 0.001 and five updates at index 9. Alignment and weighting vary; '
        'boxes are frozen from the previous AMF-off video.</p>',
        '<button onclick="document.querySelectorAll(\'video\').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button> '
        '<button onclick="document.querySelectorAll(\'video\').forEach(v=>v.pause())">Pause all</button><section>']
    for d in ('forward', 'reverse'):
        page.append(card(d+' reference', pairs.completed(root/'previous', d)/'original.mp4'))
    page.append(card('Reused Wan: AMF off', pairs.completed(root/'previous', 'off')/'final.mp4'))
    for d in ('forward', 'reverse'):
        page.append(card('Previous: original coordinates / '+d, pairs.completed(root/'previous', d)/'final.mp4'))
    for arm in ARMS:
        page.append(card('New: '+arm.replace('_', ' '), pairs.completed(root, arm)/'final.mp4'))
    page.append('</section><h2>Aligned loss components</h2><p>Scores use each arm\'s frozen mapped support. '
                'Compare actual heading, lateral movement and size change separately.</p><table><tr><th>Arm</th><th>Subject before</th><th>Subject after</th><th>Background before</th><th>Background after</th></tr>')
    for arm, audit in report['arms'].items():
        first, last = audit['losses'][0], audit['losses'][-1]
        page.append('<tr><td>'+arm+'</td>'+''.join(f'<td>{value:.5g}</td>' for value in
            (first['subject_mse'], last['subject_mse'], first['background_mse'], last['background_mse']))+'</tr>')
    page.append('</table><p>See region_gradients.csv, sampler_response.csv and subject_selectivity.csv for separate derivative, retention and directional measurements. None is an automatic motion-quality pass.</p>')
    (root/'subject_comparison.html').write_text('\n'.join(page), encoding='utf-8')


def summarize(root):
    root = Path(root); plan = checked_plan(root)
    audits = {arm: validate_result(pairs.completed(root, arm), root, plan, arm) for arm in ARMS}
    response, selectivity = response_rows(root), comparison_rows(root)
    gradients = [dict(arm=arm, **row) for arm, audit in audits.items() for row in audit['gradients']]
    losses = [dict(arm=arm, capture=i, **row) for arm, audit in audits.items() for i, row in enumerate(audit['losses'])]
    for name, rows in [('sampler_response', response), ('subject_selectivity', selectivity),
                       ('region_gradients', gradients), ('region_losses', losses)]:
        control.write_csv(root/(name+'.csv'), rows)
    report = dict(arms=audits, response=response, selectivity=selectivity, gradients=gradients)
    head.write_json(root/'subject_audit.json', report)
    if not (root/'visual_review.csv').exists():
        control.write_csv(root/'visual_review.csv', [dict(arm=a, heading='', lateral_motion='', size_change='',
            turning='', background_motion='', artifacts='', notes='') for a in ARMS])
    make_display(root, report)
    return report


def run(root):
    root = Path(root).resolve(); plan = checked_plan(root)
    head.timing.require_same_environment(read_json(root/'environment.json'), environment_snapshot())
    for arm in ARMS:
        marker = root/f'{arm}_done.json'
        if marker.is_file():
            validate_result(pairs.completed(root, arm), root, plan, arm)
            print('Already complete and revalidated:', arm)
            continue
        directory = root/(arm+'_'+head.stamp()); log = directory.with_suffix('.log')
        command = [sys.executable, '-u', 'probe_wan_subject.py', '--plan', str(root/'plan.json'),
                   '--arm', arm, '--output_path', str(directory)]
        print('Generating:', arm, '| log:', log, flush=True)
        with log.open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'Failed attempt retained at {log}; rerun for a fresh attempt')
        audit = validate_result(directory, root, plan, arm)
        head.write_json(marker, dict(directory=directory.name, audit=audit))
    return summarize(root)
