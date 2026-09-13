"""One-variable pair-mask comparison against the completed noised-reference run."""
import base64
import csv
import hashlib
import html
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from benchmark import wan_head_pilot as head
from benchmark import wan_noised_reference_pilot as noised
from benchmark.wan_head_visual_pilot import require_matched_starts
from guidance_utils.wan_affine_diagnostics import field_metrics
from probe_report import load_trace

STAGE = 'forward_adjacent_step9_comparison'
PAIR_MODE = 'forward_adjacent'
read_json = noised.read_json
archive_experiment = head.archive_experiment


def adjacent_pairs(frames):
    """Source-major (source, target) indexing, matching production AMF."""
    if type(frames) is not int or frames<2: raise ValueError('At least two latent frames are required')
    return np.arange(frames-1)*(frames+1)+1


def selected_mask(mask, frames):
    if mask.ndim!=2 or mask.shape[0]!=frames*frames or mask.dtype!=np.bool_:
        raise ValueError('Invalid all-pair validity mask')
    keep = np.zeros((frames*frames,1),dtype=bool); keep[adjacent_pairs(frames)]=True
    result = mask & keep
    if not result.any(): raise ValueError('No valid forward-adjacent entries')
    return result


def require_reusable_environment(previous, current):
    for key in ('python','packages','cuda','gpu'):
        if previous[key]!=current[key]:
            raise ValueError(f'Cannot reuse previous videos: {key} changed. Restore the recorded environment or run fresh matched controls.')
    for name,expected in previous['source_sha256'].items():
        # This helper only expands the fingerprint list; model and loss sources stay unchanged.
        if name=='benchmark/wan_head_pilot.py': continue
        path = Path(name.replace('\\','/'))
        normalized = hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if expected not in (current['source_sha256'].get(name),normalized):
            raise ValueError('Cannot reuse previous videos: source changed: '+name)


def completed(root, arm):
    return Path(root)/read_json(Path(root)/f'{arm}_done.json')['directory']


def validate_previous(root):
    root = Path(root); plan = head.load_plan(root)
    if plan['stage']!=noised.STAGE: raise ValueError('Use the completed noised-reference experiment')
    confirmation = noised.validate_confirmation(completed(root,'confirmation'),root,plan)
    if not confirmation['gate']['passed']: raise ValueError('Independent head confirmation did not pass')
    reports = [noised.validate_visual(completed(root,arm),plan,arm) for arm in ('off','candidate')]
    require_matched_starts([r['initial'] for r in reports])
    for key in ('reference_decoded_sha256','step9_latent_sha256'):
        if reports[0][key]!=reports[1][key]: raise ValueError('Previous arms differ in '+key)
    return plan,dict(confirmation=confirmation,visual=reports)


def make_plan(inputs, root, previous_run, archive=None):
    root, previous_run = Path(root).resolve(), Path(previous_run).resolve()
    if root==previous_run or previous_run in root.parents:
        raise ValueError('Create the comparison outside the previous result folder')
    previous,receipt = validate_previous(previous_run)
    require_reusable_environment(read_json(previous_run/'environment.json'),head.environment_snapshot())
    inputs,rows = head.direction.prepare_inputs(inputs,archive)
    if head.digest(inputs/'manifest.csv')!=previous['input_manifest_sha256']:
        raise ValueError('Reuse the same input manifest')
    car = next(r for r in rows if r['clip_id']=='car-turn')
    inventory = {p.relative_to(previous_run).as_posix():head.digest(p) for p in previous_run.rglob('*') if p.is_file()}
    plan = dict(schema_version=1,stage=STAGE,inputs=str(inputs),clip_id='car-turn',
        video=str(inputs/car['video_path']),generation_video=str(inputs/car['video_path']),
        prompt=previous['prompt'],selection=dict(block=30,head=30),flow_pair_mode=PAIR_MODE,
        guidance_indices=[9],updates=5,learning_rate=.001,generation_seed=1,reference_noise_seed=29,
        input_manifest_sha256=head.digest(inputs/'manifest.csv'),previous_name=previous_run.name,
        previous_plan_sha256=head.digest(previous_run/'plan.json'),previous_inventory=inventory)
    head.save_plan(root,plan)
    shutil.copytree(previous_run,root/'previous')
    head.write_json(root/'previous_audit.json',receipt)
    shutil.copyfile(inputs/'manifest.csv',root/'input_manifest.csv')
    return plan


def verify_previous_copy(root, plan):
    previous = Path(root)/'previous'
    actual = {p.relative_to(previous).as_posix():head.digest(p) for p in previous.rglob('*') if p.is_file()}
    if actual!=plan['previous_inventory']: raise ValueError('Copied comparison evidence changed')
    return previous


def checked_plan(root):
    root = Path(root); plan = head.load_plan(root)
    if (plan['stage']!=STAGE or plan['flow_pair_mode']!=PAIR_MODE or plan['selection']!=dict(block=30,head=30)
            or plan['guidance_indices']!=[9] or plan['updates']!=5 or plan['learning_rate']!=.001
            or plan['generation_seed']!=1 or plan['reference_noise_seed']!=29):
        raise ValueError('Pair comparison differs from the frozen protocol')
    previous = verify_previous_copy(root,plan)
    if head.digest(previous/'plan.json')!=plan['previous_plan_sha256']:
        raise ValueError('Previous plan changed')
    head.checked_inputs(plan)
    if head.digest(Path(plan['inputs'])/'manifest.csv')!=plan['input_manifest_sha256']:
        raise ValueError('Input manifest changed')
    return plan


def fixed_config(plan, output):
    config = noised.fixed_config(plan,'candidate',output)
    config.update(visual_protocol=STAGE,flow_pair_mode=PAIR_MODE)
    return config


def npz_equal(a,b):
    with np.load(a,allow_pickle=False) as left,np.load(b,allow_pickle=False) as right:
        return set(left.files)==set(right.files) and all(np.array_equal(left[k],right[k]) for k in left.files)


def one_event(events,kind):
    result = [e for e in events if e['kind']==kind]
    if len(result)!=1: raise ValueError('Missing/duplicate '+kind)
    return result[0]


def require_reference_match(directory, previous):
    directory,old = Path(directory),completed(previous,'candidate')
    require_matched_starts([read_json(directory/'initial_state.json'),read_json(old/'initial_state.json')])
    if read_json(directory/'reference_protocol.json')!=read_json(old/'reference_protocol.json'):
        raise ValueError('Reference noise protocol changed')
    if not npz_equal(directory/'reference_inputs.npz',old/'reference_inputs.npz'):
        raise ValueError('Actual reference input/noise changed')
    trace,_,events = load_trace(directory); old_trace,_,old_events = load_trace(old)
    current_ref,old_ref = one_event(events,'training_reference'),one_event(old_events,'training_reference')
    if not npz_equal(trace/current_ref['file'],old_trace/old_ref['file']):
        raise ValueError('Unrestricted reference AMF/mask changed')


def require_before_update_match(directory, previous):
    directory,old = Path(directory),completed(previous,'candidate')
    if read_json(directory/'step9_state.json')!=read_json(old/'step9_state.json'):
        raise ValueError('Step-9 latent or scheduler differs from previous candidate')
    trace,_,events = load_trace(directory); old_trace,_,old_events = load_trace(old)
    current = next(e for e in events if e['kind']=='full_pair_target')
    earlier = next(e for e in old_events if e['kind']=='full_pair_target')
    if not npz_equal(trace/current['file'],old_trace/earlier['file']):
        raise ValueError('Pre-update all-pair target fields changed')
    if head.timing.decoded_digest(directory/'estimated_clean_before.mp4')!=head.timing.decoded_digest(old/'estimated_clean_before.mp4'):
        raise ValueError('Pre-update decoded estimate changed')


def audit_selected_loss(trace, events, frames=6, height=30, width=52):
    trace = Path(trace)
    original_event,selected_event = one_event(events,'training_reference'),one_event(events,'selected_pair_reference')
    with np.load(trace/original_event['file'],allow_pickle=False) as data:
        original,base_mask = data['flow'],data['mask']
    with np.load(trace/selected_event['file'],allow_pickle=False) as data:
        reference,mask = data['flow'],data['mask']
    shape = (frames*frames,height*width,2)
    if (original.shape!=shape or not np.isfinite(original).all() or not np.array_equal(original,reference)
            or not np.array_equal(base_mask,(np.linalg.norm(original,axis=-1)>0)&(np.linalg.norm(original,axis=-1)<=100))
            or not np.array_equal(mask,selected_mask(base_mask,frames))):
        raise ValueError('Reference or selected pair mask differs from protocol')
    if selected_event['pair_indices']!=adjacent_pairs(frames).tolist():
        raise ValueError('Selected pair indices are mislabeled')
    targets = [e for e in events if e['kind']=='full_pair_target']
    losses = [e for e in events if e['kind']=='full_pair_loss']
    if len(targets)!=7 or len(losses)!=7: raise ValueError('Missing complete selected-pair loss evidence')
    scores = []
    for event,loss in zip(targets,losses):
        if (event['block']!=original_event['block'] or event['block']!=selected_event['block']
                or event['block']!=loss['block'] or event['file']!=loss['file']
                or event['q_dtype']!='torch.bfloat16' or event['k_dtype']!='torch.bfloat16' or event['flow_head']!=30):
            raise ValueError('Changed production head/precision or capture identity')
        for key in ('stage','step','iteration'):
            if event.get(key)!=loss.get(key): raise ValueError('Capture and loss order disagree')
        with np.load(trace/event['file'],allow_pickle=False) as data: flow = data['flow']
        if flow.shape!=shape or not np.isfinite(flow).all(): raise ValueError('Invalid target field')
        mse = float(np.square(flow.astype(float)-reference)[mask].mean())
        if not math.isclose(mse,loss['loss'],rel_tol=2e-5,abs_tol=2e-5):
            raise ValueError('Selected-pair MSE disagrees with actual optimized loss')
        scores.append(mse)
    return dict(pair_indices=adjacent_pairs(frames).tolist(),valid_positions=int(mask.sum()),losses=scores)


def score_forward_adjacent(directory, frames=6):
    """Use the same support to score the old and new objectives."""
    trace,_,events = load_trace(directory)
    ref = one_event(events,'training_reference')
    with np.load(trace/ref['file'],allow_pickle=False) as data: reference,mask = data['flow'],data['mask']
    mask = selected_mask(mask,frames)
    targets = [e for e in events if e['kind']=='full_pair_target']
    result = {}
    for label,event in [('before',targets[0]),('after',targets[-1])]:
        with np.load(trace/event['file'],allow_pickle=False) as data: flow = data['flow']
        result[label] = dict(field_metrics(flow,reference,mask),
            mse=float(np.square(flow.astype(float)-reference)[mask].mean()))
    return result


def validate_result(directory, root, plan):
    from omegaconf import OmegaConf
    import imageio.v3 as iio
    directory, root = Path(directory), Path(root)
    trace,meta,events = load_trace(directory)
    config = meta['config']
    if any(config.get(k)!=v for k,v in fixed_config(plan,config['output_path']).items()):
        raise ValueError('Pair-comparison configuration changed')
    if OmegaConf.to_container(OmegaConf.load(directory/'suite_config.yaml'))!=config:
        raise ValueError('Configuration changed after initialization')
    if read_json(directory/'complete.json')!=dict(arm='forward_adjacent',native_rope_unchanged=True,frozen_weights=True):
        raise ValueError('Incomplete generation or changed frozen model')
    report = noised.audit_trace(events,config,'candidate')
    selected = audit_selected_loss(trace,events)
    require_reference_match(directory,root/'previous')
    require_before_update_match(directory,root/'previous')
    for name in ('original.mp4','final.mp4','estimated_clean_before.mp4','estimated_clean_after.mp4'):
        count = 0
        for frame in iio.imiter(directory/name,plugin='FFMPEG'):
            if frame.shape[:2]!=(480,832): raise ValueError('Unexpected video resolution '+name)
            count += 1
        if count!=21: raise ValueError('Incomplete video '+name)
    old = completed(root/'previous','candidate')
    if head.timing.decoded_digest(directory/'original.mp4')!=head.timing.decoded_digest(old/'original.mp4'):
        raise ValueError('Reference video changed')
    return dict(report,selected=selected,metrics=score_forward_adjacent(directory),
                matched_reference_and_before_update=True,directory=str(directory))


def run(root):
    root = Path(root).resolve(); plan = checked_plan(root)
    require_reusable_environment(read_json(root/'previous/environment.json'),head.environment_snapshot())
    command = [sys.executable,'-u','probe_wan_pairs.py','--plan',str(root/'plan.json')]
    head.run_process(root,'forward_adjacent',command,lambda directory:validate_result(directory,root,plan))
    return summarize(root)


def summarize(root):
    root = Path(root).resolve(); plan = head.load_plan(root)
    previous = verify_previous_copy(root,plan)
    directory = completed(root,'forward_adjacent')
    report = validate_result(directory,root,plan)
    measurements = dict(all_pairs=score_forward_adjacent(completed(previous,'candidate')),forward_adjacent=report['metrics'])
    rows = [dict(arm=arm,stage=stage,**scores) for arm,result in measurements.items() for stage,scores in result.items()]
    with (root/'comparison_metrics.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report.update(comparison=measurements)
    head.write_json(root/'comparison_audit.json',report)
    review = root/'visual_review.csv'
    if not review.exists():
        with review.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=['arm','subject_direction','subject_displacement','size_change',
                'background_motion','appearance','notes'])
            writer.writeheader(); writer.writerows(dict(arm=a) for a in ('off','all_pairs','forward_adjacent'))
    make_display(root,directory,rows)
    return report


def make_display(root, directory, rows):
    root,directory = Path(root),Path(directory)
    previous = root/'previous'; old = completed(previous,'candidate'); off = completed(previous,'off')
    def card(label,path):
        encoded = base64.b64encode(path.read_bytes()).decode('ascii')
        return '<figure><figcaption>'+html.escape(label)+'</figcaption><video controls muted loop playsinline src="data:video/mp4;base64,'+encoded+'"></video></figure>'
    page = ['<!doctype html><meta charset="utf-8"><title>Wan forward-adjacent comparison</title>',
        '<style>body{font:16px system-ui;color:#17202a;background:#f7f8fa;margin:20px}section{display:flex;flex-wrap:wrap;gap:12px}'
        'figure{margin:0;flex:1 1 380px;max-width:832px}video{width:100%}figcaption{padding:8px 0}'
        'td,th{padding:8px;border:1px solid #aaa}table{border-collapse:collapse}</style>',
        '<h1>Wan 14B: pair selection comparison</h1><p>All generated panels use Wan2.1 T2V 14B. '
        'Both guided arms use head 30, matched noised reference and five updates at step 9. Only the loss pair mask changes.</p>',
        '<button onclick="document.querySelectorAll(\'#finals video\').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button> '
        '<button onclick="document.querySelectorAll(\'#finals video\').forEach(v=>v.pause())">Pause all</button><section id="finals">',
        card('Motion reference',old/'original.mp4'),card('Wan 14B: AMF off (previous run)',off/'final.mp4'),
        card('Wan 14B: head 30, all-pair loss (previous run)',old/'final.mp4'),
        card('Wan 14B: head 30, forward-adjacent loss (new)',directory/'final.mp4'),'</section>',
        '<h2>Same forward-adjacent support for both guided arms</h2><p>AMF agreement with the reference, not decoded optical flow or video-quality scores.</p>',
        '<table><tr><th>Arm</th><th>Stage</th><th>MSE</th><th>Cosine</th><th>Amplitude</th><th>EPE</th></tr>']
    for row in rows:
        values=[row[k] for k in ('arm','stage','mse','direction_cosine','amplitude_ratio','epe')]
        page.append('<tr>'+''.join('<td>'+html.escape(f'{v:.4f}' if isinstance(v,float) else str(v))+'</td>' for v in values)+'</tr>')
    page.extend(['</table><h2>Intermediate step-9 estimates</h2><p>These are model predictions, not final videos.</p><section>',
        card('Shared pre-update estimate',old/'estimated_clean_before.mp4'),
        card('After all-pair guidance',old/'estimated_clean_after.mp4'),
        card('After forward-adjacent guidance',directory/'estimated_clean_after.mp4'),'</section>'])
    path = root/'pair_comparison.html'; path.write_text('\n'.join(page),encoding='utf-8')
    return path
