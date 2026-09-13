"""Conditional three-arm visual test; trace audits do not grade physical motion."""
import csv
import json
import math
import sys
from pathlib import Path

from benchmark import wan_head_pilot as head
from benchmark import wan_direction_pilot as direction
from benchmark import wan_timing_pilot as timing
from guidance_utils.wan_amf_calibration import MODEL
from guidance_utils.wan_guidance_schedule import learning_rates

ARMS = ('off','baseline','candidate')


def require_matched_starts(states):
    for key in ('latent_sha256','reference_latent_sha256','conditioning_sha256','source_conditioning_sha256',
                'rope_sha256','timesteps','sigmas','model_revision'):
        if any(state[key]!=states[0][key] for state in states[1:]):
            raise ValueError(f'Unmatched starting state: {key}')


def fixed_config(plan, arm, output):
    if arm not in ARMS: raise ValueError('Unknown visual arm')
    candidate = plan['selection']
    return dict(model_key=MODEL,enable_model_cpu_offload=True,video_path=plan['video'],output_path=str(output),
        target_prompt=plan['prompt'],source_prompt='',seed=1,opt_mode='latent',guidance_mode='latent',
        loss_type='flow',save_format='mp4',save_embeds=False,inject_embeds=False,verbose=False,
        scheduler='flowmatch',flow_shift=3.,num_frames=21,height=480,width=832,num_inference_steps=50,
        guidance_scale=5.,guidance_blocks=[] if arm=='off' else [10 if arm=='baseline' else candidate['block']],
        flow_head=candidate['head'] if arm=='candidate' else None,injection_blocks=[],
        guidance_timestep_range=[50,40],injection_timestep_range=[50,40],lr=[.002,.001],lr_decay_steps=10,
        optimization_steps=5,motion_temp=2.,flow_max_disp=100.,flow_min_conf=None,threshloss=True,
        argmax_motion_flow=True,flow_loss='mse',flow_region_masks=None,softmax_fp32=True,
        checkpoint_amf='auto',enable_gradient_checkpointing=True,probe=True,probe_rope=False,
        probe_blocks=sorted({10,candidate['block']}),probe_steps=[0,9,10,29,49],reference_only=False)


def make_plan(inputs, root, development, confirmation, archive=None):
    selection = head.confirmed_candidate(development,confirmation)
    timing.require_same_environment(json.loads((Path(confirmation)/'environment.json').read_text(encoding='utf-8')),
                                    head.environment_snapshot())
    inputs,rows = direction.prepare_inputs(inputs,archive)
    row = next(r for r in rows if r['clip_id']=='car-turn')
    prompt = row['prompt'].rstrip().rstrip('.')+'. '+direction.DIRECTION_SUFFIXES['car-turn']
    plan = dict(schema_version=1,stage='visual',selection=selection,inputs=str(inputs),
        clip_id='car-turn',video=str(inputs/row['video_path']),prompt=prompt,arms=list(ARMS),
        sampling_steps=50,expected_updates=dict(off=0,baseline=50,candidate=50))
    head.save_plan(root,plan)
    (Path(root)/'input_manifest.csv').write_bytes((inputs/'manifest.csv').read_bytes())
    return plan


def audit_trace(events, config, arm):
    """Validate actual ordered events and schedules, independently of plan budgets."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
    updates = [e for e in events if e['kind']=='optimization']
    expected = [] if arm=='off' else [(i,j) for i in range(10) for j in range(5)]
    if [(e['step'],e['iteration']) for e in updates] != expected:
        raise ValueError('Actual optimizer count/order differs from protocol')
    rates = learning_rates(list(range(10)),[.002,.001],10)
    for event in updates:
        if (not math.isclose(event['lr'],rates[event['step']],rel_tol=1e-10)
                or event['timestep'] != float(scheduler.timesteps[event['step']])
                or event['gradient'].get('finite_fraction') != 1
                or event['update'].get('finite_fraction') != 1
                or not math.isfinite(event['loss_before_update'])):
            raise ValueError('Invalid gradient, update, loss or optimizer schedule')
    sampling = [e for e in events if e['kind']=='sampling']
    if [e['step'] for e in sampling] != list(range(50)):
        raise ValueError('Incomplete, duplicate or unordered sampling trace')
    for event in sampling:
        i = event['step']
        if event['sigma']!=float(scheduler.sigmas[i]) or event['timestep']!=float(scheduler.timesteps[i]):
            raise ValueError('Actual sampling schedule changed')
        if event['latent']['finite_fraction']!=1: raise ValueError('Nonfinite sampled latent')
        if (arm=='off' or i>=10) and event['guidance_update']['abs_max'] != 0:
            raise ValueError('Unexpected latent guidance outside the planned window')
    if any(e.get('injected') for e in events if e['kind']=='attention'):
        raise ValueError('Unexpected KV injection')
    actual = [e for e in events if e['kind']=='training_flow']
    expected_block = f"block_{config['guidance_blocks'][0]}_attn1_processor" if arm!='off' else None
    if arm!='off' and (not actual or any(e['block']!=expected_block for e in actual)):
        raise ValueError('Actual loss-path captures missing or mislabeled')
    return dict(updates=len(updates),guidance_active=bool(updates) and all(
        e['gradient']['abs_max']>0 and e['update']['abs_max']>0 for e in updates),
        losses_before_updates=[e['loss_before_update'] for e in updates])


def validate_result(directory, plan, arm, environment=None):
    from probe_report import load_trace
    from omegaconf import OmegaConf
    directory = Path(directory)
    _,meta,events = load_trace(directory)
    config = meta['config']
    expected = fixed_config(plan,arm,config['output_path'])
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('Visual arm configuration differs from protocol')
    saved = OmegaConf.to_container(OmegaConf.load(directory/'suite_config.yaml'))
    if saved != config: raise ValueError('Configuration changed between initialization and trace')
    complete = json.loads((directory/'complete.json').read_text(encoding='utf-8'))
    if complete.get('arm') != arm or complete.get('native_rope_unchanged') is not True:
        raise ValueError('Incomplete arm or changed native RoPE')
    initial = json.loads((directory/'initial_state.json').read_text(encoding='utf-8'))
    if initial['flow_head']!=config['flow_head'] or initial['guidance_blocks']!=config['guidance_blocks']:
        raise ValueError('Initialized readout differs from configuration')
    estimates = ['estimated_clean_before.mp4'] + ([] if arm=='off' else ['estimated_clean_after.mp4'])
    for name in ['original.mp4','final.mp4',*estimates]:
        if not (directory/name).is_file() or (directory/name).stat().st_size==0:
            raise ValueError(f'Missing visual evidence: {name}')
    evidence = [e for e in events if e['kind']=='visual_estimate']
    if [e['label'] for e in evidence] != (['before'] if arm=='off' else ['before','after']):
        raise ValueError('Missing or duplicated step-9 estimates')
    if any(e['step']!=9 or not e['state_preserved'] for e in evidence):
        raise ValueError('Estimate instrumentation changed sampling state')
    if environment:
        for key,value in meta['packages'].items():
            if key in environment['packages'] and value!=environment['packages'][key]:
                raise ValueError(f'Package changed: {key}')
        for key,value in meta['source_sha256'].items():
            if environment['source_sha256'].get(key.replace('\\','/'))!=value:
                raise ValueError(f'Source changed: {key}')
        if meta['gpu'] not in environment['gpu']: raise ValueError('GPU changed')
    report = audit_trace(events,config,arm)
    return dict(report,arm=arm,initial=initial,reference_decoded_sha256=timing.decoded_digest(directory/'original.mp4'),
                directory=str(directory),note='Audit only. Compare physical motion; losses across heads are not quality scores.')


def run(root):
    root = Path(root).resolve(); plan = head.load_plan(root)
    if plan['stage']!='visual' or plan['arms']!=list(ARMS): raise ValueError('Wrong visual plan')
    head.checked_inputs(plan)
    environment = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    for arm in ARMS:
        command = [sys.executable,'-u','probe_wan_head_visual.py','--plan',str(root/'plan.json'),'--arm',arm]
        head.run_process(root,arm,command,lambda directory,a=arm: validate_result(directory,plan,a,environment))
    return summarize(root)


def summarize(root):
    root = Path(root).resolve(); plan = head.load_plan(root)
    environment = json.loads((root/'environment.json').read_text(encoding='utf-8'))
    reports = []
    for arm in ARMS:
        directory = root/json.loads((root/f'{arm}_done.json').read_text(encoding='utf-8'))['directory']
        reports.append(validate_result(directory,plan,arm,environment))
    require_matched_starts([r['initial'] for r in reports])
    if len({r['reference_decoded_sha256'] for r in reports})!=1:
        raise ValueError('Different reference previews')
    head.write_json(root/'visual_audit.json',reports)
    review = root/'visual_review.csv'
    if not review.exists():
        with review.open('w',newline='',encoding='utf-8') as stream:
            writer = csv.DictWriter(stream,fieldnames=['arm','heading_approach','subject_displacement',
                'size_change','background_motion','appearance','immediate_effect','retained_in_final','notes'])
            writer.writeheader(); writer.writerows(dict(arm=arm) for arm in ARMS)
    return reports
