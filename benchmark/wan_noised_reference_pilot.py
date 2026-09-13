"""Frozen step-9 reference experiment; confirmation and actual decoded videos."""
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
from benchmark import wan_head_crossover_pilot as crossover
from benchmark import wan_head_visual_pilot as visual
from guidance_utils.wan_amf_calibration import CONTROLS, THRESHOLDS
from guidance_utils.wan_head_diagnostics import audit_heads

STAGE = 'noised_reference_step9_visual'
ARMS = visual.ARMS
BLOCK = 'block_30_attn1_processor'
read_json = crossover.read_json


def screen_step9(rows):
    """Only the frozen step-9 nominal soft gate; no clean-screen relabelling."""
    failures, measurements = [], []
    for control in CONTROLS:
        matches = [r for r in rows if r['block']==BLOCK and r['variant']=='head_30'
                   and r['noise_label']=='step_09' and r['field']=='soft'
                   and r['support']=='textured' and r['anchor_offset']==0 and r['control']==control]
        if len(matches)!=1:
            failures.append(control+': missing or duplicate row'); continue
        r = matches[0]; measurements.append(r)
        if r['patches']<=0 or r['finite_fraction']!=1:
            failures.append(control+': invalid support/prediction'); continue
        if control=='static':
            if r['epe'] is None or not math.isfinite(r['epe']) or r['epe']>THRESHOLDS['static_epe_max']:
                failures.append(control+': static EPE > 0.5')
        else:
            threshold = THRESHOLDS['translation_cosine_min'] if control.startswith('pan_') else THRESHOLDS['scale_cosine_min']
            if r['direction_cosine'] is None or not math.isfinite(r['direction_cosine']) or r['direction_cosine']<threshold:
                failures.append(control+f': cosine < {threshold}')
            if r['amplitude_ratio'] is None or not THRESHOLDS['amplitude_min']<=r['amplitude_ratio']<=THRESHOLDS['amplitude_max']:
                failures.append(control+': amplitude outside [0.5, 1.5]')
    return dict(passed=not failures,failures=failures,measurements=measurements,
                scope='Step 9, nominal textured soft AMF; not the original full head screen.')


def make_plan(inputs, root, crossover_run, archive=None):
    crossover_run = Path(crossover_run).resolve()
    previous = head.load_plan(crossover_run)
    if previous['stage']!='head_noise_timestep_crossover':
        raise ValueError('Use the completed noise/timestep crossover experiment')
    directory = crossover_run/read_json(crossover_run/'crossover_done.json')['directory']
    receipt = crossover.validate_result(directory,crossover_run,previous)
    if not receipt['interpretation_ready']:
        raise ValueError('Resolve crossover replication mismatch before this experiment')
    old = read_json(directory/'metadata.json')
    for name in crossover.CORE_SOURCES:
        current = hashlib.sha256(Path(name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        if old['source_sha256'][name]!=current:
            raise ValueError(f'Core source changed since crossover: {name}')
    inputs, rows = head.direction.prepare_inputs(inputs,archive)
    camel = next(r for r in rows if r['clip_id']=='camel')
    car = next(r for r in rows if r['clip_id']=='car-turn')
    video = str(inputs/camel['video_path'])
    files = ['metadata.json','metrics.json']
    plan = dict(schema_version=1,stage=STAGE,inputs=str(inputs),clip_id='camel',video=video,
        generation_video=str(inputs/car['video_path']),generation_clip_id='car-turn',
        prompt=car['prompt'].rstrip().rstrip('.')+'. '+head.direction.DIRECTION_SUFFIXES['car-turn'],
        selection=dict(block=30,head=30),noise_seed=29,reference_noise_seed=29,temperature=2.,
        input_manifest_sha256=head.digest(inputs/'manifest.csv'),
        thresholds=THRESHOLDS,gate=dict(state='step_09',field='soft',support='textured',anchor_offset=0),
        command=head.command_for(video,[30],[30],29),arms=list(ARMS),
        guidance_indices=[9],updates=5,learning_rate=.001,generation_seed=1,
        crossover_name=crossover_run.name,crossover_plan_sha256=head.digest(crossover_run/'plan.json'),
        evidence_inventory={name:head.digest(directory/name) for name in files})
    head.save_plan(root,plan)
    evidence = Path(root)/'crossover_evidence'; evidence.mkdir()
    for name in files: shutil.copyfile(directory/name,evidence/name)
    head.write_json(evidence/'audit_receipt.json',receipt)
    shutil.copyfile(inputs/'manifest.csv',Path(root)/'input_manifest.csv')
    return plan


def checked_plan(root):
    root = Path(root).resolve(); plan = head.load_plan(root)
    if (plan['stage']!=STAGE or plan['selection']!=dict(block=30,head=30)
            or plan['thresholds']!=THRESHOLDS or plan['arms']!=list(ARMS)
            or plan['command']!=head.command_for(plan['video'],[30],[30],29)
            or plan['gate']!=dict(state='step_09',field='soft',support='textured',anchor_offset=0)
            or plan['guidance_indices']!=[9] or plan['updates']!=5 or plan['learning_rate']!=.001
            or plan['noise_seed']!=29 or plan['reference_noise_seed']!=29 or plan['generation_seed']!=1):
        raise ValueError('Noised-reference protocol changed; create a fresh experiment')
    for name, expected in plan['evidence_inventory'].items():
        if head.digest(root/'crossover_evidence'/name)!=expected:
            raise ValueError('Crossover evidence changed: '+name)
    _, rows = head.checked_inputs(plan)
    if head.digest(Path(plan['inputs'])/'manifest.csv')!=plan['input_manifest_sha256']:
        raise ValueError('Input manifest changed after setup')
    car = next(r for r in rows if r['clip_id']=='car-turn')
    if str(Path(plan['inputs'])/car['video_path'])!=plan['generation_video']:
        raise ValueError('Generation reference changed')
    return plan


def validate_confirmation(directory, root, plan):
    from omegaconf import OmegaConf
    directory, root = Path(directory), Path(root)
    if str(OmegaConf.load(directory/'suite_config.yaml').video_path)!=plan['video']:
        raise ValueError('Confirmation input differs from plan')
    audit = audit_heads(directory,[30],[30],29)
    meta, old = read_json(directory/'metadata.json'), read_json(root/'crossover_evidence/metadata.json')
    for key in ('input_base_sha256','noise_sha256'):
        if meta[key]==old[key]: raise ValueError('Confirmation must change '+key)
    for key in ('model','model_revision','packages','model_dtype','grid'):
        if meta[key]!=old[key]: raise ValueError('Confirmation changed '+key)
    for key in crossover.CORE_SOURCES:
        if meta['source_sha256'][key]!=old['source_sha256'][key]:
            raise ValueError('Confirmation changed core source '+key)
    return dict(audit=audit,gate=screen_step9(read_json(directory/'metrics.json')),
                metadata_sha256=head.digest(directory/'metadata.json'),metrics_sha256=head.digest(directory/'metrics.json'))


def run_confirmation(root):
    root = Path(root).resolve(); plan = checked_plan(root)
    directory = head.run_process(root,'confirmation',plan['command'],
        lambda d: validate_confirmation(d,root,plan))
    report = validate_confirmation(directory,root,plan)
    head.write_json(root/'confirmation_report.json',report)
    print('Candidate step-9 confirmation:', 'PASS' if report['gate']['passed'] else 'FAIL')
    for failure in report['gate']['failures']: print(' ',failure)
    return directory,report


def fixed_config(plan, arm, output):
    adapted = dict(plan,video=plan['generation_video'])
    config = visual.fixed_config(adapted,arm,output)
    config.update(guidance_timestep_range=[41,40],lr=[.001,.001],lr_decay_steps=1,
        probe_steps=[9,10,29,49],visual_protocol=STAGE,
        reference_noise_step=9 if arm=='candidate' else None,reference_noise_seed=29)
    return config


def audit_trace(events, config, arm):
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
    updates = [e for e in events if e['kind']=='optimization']
    if [(e['step'],e['iteration']) for e in updates] != ([] if arm=='off' else [(9,j) for j in range(5)]):
        raise ValueError('Actual optimizer count/order differs from step-9 protocol')
    for e in updates:
        if (e['lr']!=.001 or e['timestep']!=float(scheduler.timesteps[9])
                or e['gradient']['finite_fraction']!=1 or e['update']['finite_fraction']!=1
                or not math.isfinite(e['loss_before_update'])):
            raise ValueError('Nonfinite guidance or changed update schedule')
    sampling = [e for e in events if e['kind']=='sampling']
    if [e['step'] for e in sampling]!=list(range(50)):
        raise ValueError('Incomplete, duplicate or unordered sampling trace')
    for e in sampling:
        i = e['step']
        if e['sigma']!=float(scheduler.sigmas[i]) or e['timestep']!=float(scheduler.timesteps[i]):
            raise ValueError('Changed sampling schedule')
        if e['latent']['finite_fraction']!=1: raise ValueError('Nonfinite sampled latent')
        if (arm=='off' or i!=9) and e['guidance_update']['abs_max']!=0:
            raise ValueError('Unexpected guidance outside the step-9 intervention')
    if any(e.get('injected') for e in events if e['kind']=='attention'):
        raise ValueError('Unexpected KV injection')
    estimates = [e for e in events if e['kind']=='visual_estimate']
    if [e['label'] for e in estimates] != (['before'] if arm=='off' else ['before','after']):
        raise ValueError('Missing/duplicate visual estimates')
    if any(e['step']!=9 or not e['state_preserved'] for e in estimates):
        raise ValueError('Estimate instrumentation changed state')
    actual = [e for e in events if e['kind']=='training_flow']
    block = None if arm=='off' else f"block_{config['guidance_blocks'][0]}_attn1_processor"
    # Production's ordinary training_flow captures only first/last iterations and endpoints.
    if len(actual)!=(0 if arm=='off' else 4) or any(e['block']!=block for e in actual):
        raise ValueError('Missing/mislabeled production loss evidence')
    all_losses = [e for e in events if e['kind']=='full_pair_loss']
    if len(all_losses)!=(0 if arm=='off' else 7): raise ValueError('Missing complete production losses')
    if arm!='off':
        if any(a['loss']!=b['loss_before_update'] for a,b in zip(all_losses[1:6],updates)):
            raise ValueError('Saved full-pair losses differ from optimizer losses')
        if any(a['loss']!=all_losses[i]['loss'] for a,i in zip(actual,[0,1,5,6])):
            raise ValueError('Adjacent and full-pair loss traces disagree')
    return dict(updates=len(updates),guidance_active=bool(updates) and all(
        e['gradient']['abs_max']>0 and e['update']['abs_max']>0 for e in updates),
        losses_before_updates=[e['loss_before_update'] for e in updates],
        step9_latent_sha256=estimates[0]['latent_sha256'])


def audit_full_pairs(trace, events, arm, frames=6, height=30, width=52):
    """Recompute MSE from all actual source/target pairs, including reverse/diagonal."""
    references = [e for e in events if e['kind']=='training_reference']
    captures = [e for e in events if e['kind']=='full_pair_target']
    losses = [e for e in events if e['kind']=='full_pair_loss']
    if (len(references)!=(0 if arm=='off' else 1) or len(captures)!=(0 if arm=='off' else 7)
            or len(losses)!=len(captures)):
        raise ValueError('Missing full-pair reference/target evidence')
    if arm=='off': return []
    shape = (frames*frames,height*width,2)
    with np.load(Path(trace)/references[0]['file'],allow_pickle=False) as data:
        reference,mask = data['flow'],data['mask']
    if (reference.shape!=shape or mask.shape!=shape[:-1] or mask.dtype!=np.bool_
            or not np.isfinite(reference).all() or not mask.any()):
        raise ValueError('Invalid full-pair reference/mask')
    # With hard integer displacements and cap 100, the declared mask is exactly nonzero.
    expected_mask = (np.linalg.norm(reference,axis=-1)>0)&(np.linalg.norm(reference,axis=-1)<=100.)
    if not np.array_equal(mask,expected_mask): raise ValueError('Reference validity mask changed')
    reports = []
    for event,loss in zip(captures,losses):
        if event['block']!=references[0]['block'] or event['block']!=loss['block']:
            raise ValueError('Full-pair block mismatch')
        if (event['file']!=loss['file'] or event['q_dtype']!='torch.bfloat16' or event['k_dtype']!='torch.bfloat16'
                or event['flow_head']!=(30 if arm=='candidate' else None)):
            raise ValueError('Changed production QK precision/head or capture order')
        for key in ('stage','step','iteration'):
            if event.get(key)!=loss.get(key): raise ValueError('Full-pair loss event order mismatch')
        with np.load(Path(trace)/event['file'],allow_pickle=False) as data: flow = data['flow']
        if flow.shape!=shape or not np.isfinite(flow).all(): raise ValueError('Invalid full-pair target')
        error = np.square(flow.astype(np.float64)-reference)
        mse = float(error[mask].mean())
        if not math.isclose(mse,loss['loss'],rel_tol=2e-5,abs_tol=2e-5):
            raise ValueError('Full-pair MSE disagrees with actual loss')
        reports.append(dict(file=event['file'],mse=mse,stage=event.get('stage'),
            pairs=[dict(source=i//frames,target=i%frames,kept=int(mask[i].sum()),
                        mse=float(error[i][mask[i]].mean()) if mask[i].any() else None)
                   for i in range(frames*frames)]))
    return reports


def audit_reference_inputs(path, provenance):
    import torch
    with np.load(path,allow_pickle=False) as data:
        clean,noise,value = data['clean'],data['noise'],data['model_input']
    if (clean.shape!=noise.shape or clean.shape!=value.shape or provenance['dtype']!='torch.bfloat16'
            or not all(np.isfinite(a).all() for a in (clean,noise,value))):
        raise ValueError('Invalid saved reference input tensors')
    for key,array in [('clean_sha256',clean),('noise_sha256',noise),('input_sha256',value)]:
        if hashlib.sha256(array.astype(np.float32).tobytes()).hexdigest()!=provenance[key]:
            raise ValueError('Saved reference fingerprint mismatch: '+key)
    sigma = provenance['sigma']
    expected = ((1-sigma)*torch.from_numpy(clean)+sigma*torch.from_numpy(noise)).bfloat16().float().numpy()
    if not np.array_equal(expected,value): raise ValueError('Reference did not use matched noise interpolation')
    return dict(shape=list(clean.shape),noise_interpolation_verified=True)


def validate_visual(directory, plan, arm):
    from omegaconf import OmegaConf
    from probe_report import load_trace
    directory = Path(directory); trace,meta,events = load_trace(directory)
    config = meta['config']; expected = fixed_config(plan,arm,config['output_path'])
    if any(config.get(k)!=v for k,v in expected.items()): raise ValueError('Visual configuration differs from plan')
    if OmegaConf.to_container(OmegaConf.load(directory/'suite_config.yaml'))!=config:
        raise ValueError('Configuration changed after initialization')
    complete = read_json(directory/'complete.json')
    if complete!=dict(arm=arm,native_rope_unchanged=True,frozen_weights=True):
        raise ValueError('Incomplete arm or altered model/RoPE')
    initial = read_json(directory/'initial_state.json')
    if initial['guidance_blocks']!=config['guidance_blocks'] or initial['flow_head']!=config['flow_head']:
        raise ValueError('Initialized readout differs from configuration')
    report = audit_trace(events,config,arm)
    pairs = audit_full_pairs(trace,events,arm)
    provenance = read_json(directory/'reference_protocol.json')
    if provenance['mode']!=('matched_noised' if arm=='candidate' else 'clean' if arm=='baseline' else 'unused'):
        raise ValueError('Reference protocol mismatch')
    if arm=='candidate':
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        if (provenance['noise_seed']!=29 or provenance['step']!=9
                or provenance['sigma']!=float(scheduler.sigmas[9]) or provenance['timestep']!=float(scheduler.timesteps[9])
                or provenance['clean_sha256']!=initial['reference_latent_sha256']
                or provenance['noise_sha256']==initial['latent_sha256']
                or provenance['input_sha256']==initial['reference_latent_sha256']):
            raise ValueError('Noised-reference provenance mismatch')
        inputs = audit_reference_inputs(directory/'reference_inputs.npz',provenance)
        if inputs['shape']!=[1,16,6,60,104]: raise ValueError('Unexpected 14B reference latent geometry')
    names = ['original.mp4','final.mp4','estimated_clean_before.mp4']
    if arm!='off': names.append('estimated_clean_after.mp4')
    import imageio.v3 as iio
    for name in names:
        path = directory/name
        if not path.is_file() or path.stat().st_size==0: raise ValueError('Missing video '+name)
        count = 0
        for frame in iio.imiter(path,plugin='FFMPEG'):
            if frame.shape[:2]!=(480,832): raise ValueError('Unexpected video resolution '+name)
            count += 1
        if count!=21: raise ValueError('Incomplete decoded video '+name)
    return dict(report,arm=arm,initial=initial,full_pairs=pairs,reference_protocol=provenance,
                reference_decoded_sha256=head.timing.decoded_digest(directory/'original.mp4'),directory=str(directory))


def run_visual(root):
    root = Path(root).resolve(); plan = checked_plan(root)
    directory = root/read_json(root/'confirmation_done.json')['directory']
    confirmation = validate_confirmation(directory,root,plan)
    head.write_json(root/'confirmation_report.json',confirmation)
    for arm in ARMS:
        if arm=='candidate' and not confirmation['gate']['passed']:
            head.write_json(root/'candidate_skipped.json',dict(reason='Independent step-9 screen failed',
                failures=confirmation['gate']['failures']))
            print('Candidate skipped:',confirmation['gate']['failures']); continue
        command = [sys.executable,'-u','probe_wan_noised_reference.py','--plan',str(root/'plan.json'),'--arm',arm]
        head.run_process(root,arm,command,lambda d,a=arm: validate_visual(d,plan,a))
    return summarize(root)


def summarize(root):
    root = Path(root).resolve(); plan = head.load_plan(root)
    reports = []
    for arm in ARMS:
        marker = root/f'{arm}_done.json'
        if marker.is_file(): reports.append(validate_visual(root/read_json(marker)['directory'],plan,arm))
    if not reports: raise ValueError('No completed generations to summarize')
    visual.require_matched_starts([r['initial'] for r in reports])
    for key in ('reference_decoded_sha256','step9_latent_sha256'):
        if len({r[key] for r in reports})!=1: raise ValueError('Unmatched '+key)
    head.write_json(root/'visual_audit.json',reports)
    review = root/'visual_review.csv'
    if not review.exists():
        with review.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=['arm','subject_direction','subject_displacement',
                'size_change','background_motion','appearance','immediate_effect','retained_in_final','notes'])
            writer.writeheader(); writer.writerows(dict(arm=r['arm']) for r in reports)
    make_display(root,reports)
    return reports


def make_display(root, reports):
    """Portable MP4 data URLs work in Colab and in the downloaded HTML archive."""
    root = Path(root)
    def card(label,path):
        payload = base64.b64encode(Path(path).read_bytes()).decode('ascii')
        return ('<figure><figcaption>'+html.escape(label)+'</figcaption>'
                '<video controls muted loop playsinline preload="metadata" src="data:video/mp4;base64,'+payload+'"></video></figure>')
    page = ['<!doctype html><meta charset="utf-8"><title>Wan matched generation</title>',
        '<style>body{font:16px system-ui;margin:20px;background:#f7f8fa}section{display:flex;flex-wrap:wrap;gap:12px}'
        'figure{margin:0;flex:1 1 380px;max-width:832px}video{width:100%}figcaption{padding:8px 0}</style>',
        '<h1>Completed Wan generations</h1><p>Same prompt, seed and step-9 starting latent. '
        'Guided arms: five updates at step 9, then complete 50-step sampling.</p>',
        '<button onclick="document.querySelectorAll(\'#finals video\').forEach(v=>{v.currentTime=0;v.play()})">Play all from start</button> '
        '<button onclick="document.querySelectorAll(\'#finals video\').forEach(v=>v.pause())">Pause all</button><section id="finals">',
        card('Motion reference',Path(reports[0]['directory'])/'original.mp4')]
    labels = dict(off='Vanilla Wan (AMF off)',baseline='Baseline readout: block 10 / mean, clean reference',
                  candidate='Candidate: block 30 / head 30, matched noised reference')
    for report in reports:
        page.append(card(labels[report['arm']],Path(report['directory'])/'final.mp4'))
    page.append('</section>')
    if (root/'candidate_skipped.json').is_file():
        page.append('<p>Candidate skipped: '+html.escape('; '.join(read_json(root/'candidate_skipped.json')['failures']))+'</p>')
    page.append('<h2>Intermediate step-9 model predictions</h2><p>These are noisy estimates, not final generated videos.</p><section>')
    page.append(card('Shared estimate before any guidance',Path(reports[0]['directory'])/'estimated_clean_before.mp4'))
    for report in reports:
        if report['arm']!='off': page.append(card(report['arm']+' estimate after guidance',Path(report['directory'])/'estimated_clean_after.mp4'))
    page.append('</section><p>Review subject direction, displacement, size, background and appearance in visual_review.csv. '
                'A finite gradient or lower AMF loss does not establish improved physical motion.</p>')
    path = root/'visual_comparison.html'; path.write_text('\n'.join(page),encoding='utf-8')
    return path


archive_experiment = head.archive_experiment
