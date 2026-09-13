"""Exploratory 2x2 noise/time experiment, with archived diagonal replication."""
import html
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from benchmark import wan_head_pilot as head
from guidance_utils.wan_amf_calibration import CONTROLS, MODEL
from guidance_utils.wan_head_diagnostics import audit_heads

DIAGONAL = {'noise_clean_time_clean':'clean','noise_09_time_09':'step_09'}
CORE_SOURCES = ('motion_guidance_wan.py','guidance_utils/wan_modules.py',
    'guidance_utils/wan_transformer.py','guidance_utils/motion_probe.py','guidance_utils/wan_affine_diagnostics.py')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def command_for(video):
    return [sys.executable,'-u','probe_wan_affine.py','-v',str(video),'--model','14b','--low_vram',
        '--blocks','30','--readout_heads','30','--controls',*CONTROLS,'--noise_steps','9',
        '--noise_seed','17','--cross_noise_timestep']


def make_plan(inputs, root, baseline_run, archive=None):
    baseline_run = Path(baseline_run).resolve()
    previous = head.load_plan(baseline_run)
    if (previous['stage']!='development' or previous['model']!=MODEL or previous['clip_id']!='car-turn'
            or previous['noise_seed']!=17 or previous['blocks']!=[20,30] or previous['heads'] is not None):
        raise ValueError('Use the completed original 14B car/head development experiment as baseline')
    baseline = baseline_run/read_json(baseline_run/'readout_done.json')['directory']
    head.validate_result(baseline,previous)
    files = ['metadata.json'] + [f'{c}/{s}/block_30_attn1_processor/{v}.npz'
        for c in CONTROLS for s in DIAGONAL.values() for v in ('mean_logits','head_30')]
    inventory = {name:head.digest(baseline/name) for name in files}
    inputs,rows = head.direction.prepare_inputs(inputs,archive)
    row = next(r for r in rows if r['clip_id']=='car-turn')
    video = str(inputs/row['video_path'])
    plan = dict(schema_version=1,stage='head_noise_timestep_crossover',model=MODEL,inputs=str(inputs),
        video=video,clip_id='car-turn',blocks=[30],heads=[30],noise_seed=17,temperature=2.,
        baseline_name=baseline_run.name,baseline_plan_sha256=head.digest(baseline_run/'plan.json'),
        baseline_inventory=inventory,forward_passes=20,expected_rows=480,command=command_for(video),
        note='Exploratory diagnosis; no candidate promotion or target generation.')
    head.save_plan(root,plan)
    for name in files:
        destination=Path(root)/'baseline'/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(baseline/name,destination)
    (Path(root)/'input_manifest.csv').write_bytes((inputs/'manifest.csv').read_bytes())
    return plan


def verify_baseline(root,plan):
    root=Path(root)
    for name,expected in plan['baseline_inventory'].items():
        if head.digest(root/'baseline'/name)!=expected:
            raise ValueError(f'Copied baseline evidence changed: {name}')


def compare_diagonal(directory,baseline):
    """Data equality is checked independently of runtime/source compatibility."""
    directory,baseline=Path(directory),Path(baseline)
    comparisons=[]
    for new_state,old_state in DIAGONAL.items():
        for control in CONTROLS:
            for variant in ('mean_logits','head_30'):
                suffix=Path('block_30_attn1_processor')/f'{variant}.npz'
                with np.load(directory/control/new_state/suffix,allow_pickle=False) as a, \
                     np.load(baseline/control/old_state/suffix,allow_pickle=False) as b:
                    equal=set(a.files)==set(b.files) and all(np.array_equal(a[k],b[k]) for k in a.files)
                    delta={k:float(np.max(np.abs(a[k]-b[k]))) if a[k].shape==b[k].shape else None
                           for k in set(a.files)&set(b.files)}
                comparisons.append(dict(control=control,condition=new_state,variant=variant,equal=equal,max_abs_difference=delta))
    meta,old=read_json(directory/'metadata.json'),read_json(baseline/'metadata.json')
    metadata_equal={k:meta.get(k)==old.get(k) for k in ('model','grid','packages','python','gpu','model_dtype',
        'input_base_sha256','noise_sha256','conditioning','temperature','model_revision')}
    source_equal={k:meta['source_sha256'].get(k)==old['source_sha256'].get(k) for k in CORE_SOURCES}
    return dict(captures=len(comparisons),identical=sum(c['equal'] for c in comparisons),
        metadata_equal=metadata_equal,core_source_equal=source_equal,
        replication_pass=all(c['equal'] for c in comparisons) and all(metadata_equal.values()) and all(source_equal.values()),
        comparisons=comparisons,
        note='Probe control-flow source changes for the new matrix; core model/readout sources must agree.')


def validate_result(directory,root,plan):
    from omegaconf import OmegaConf
    verify_baseline(root,plan)
    config=OmegaConf.load(Path(directory)/'suite_config.yaml')
    if config.video_path!=plan['video']:
        raise ValueError('Crossover used an unplanned reference path')
    audit=audit_heads(directory,blocks=[30],heads=[30],noise_seed=17,cross_noise_timestep=True)
    replication=compare_diagonal(directory,Path(root)/'baseline')
    return dict(audit=audit,replication=replication,
        interpretation_ready=replication['replication_pass'],
        note='A replication mismatch needs review before interpreting the crossed conditions; no automatic promotion.')


def run(root):
    root=Path(root).resolve(); plan=head.load_plan(root)
    if plan['stage']!='head_noise_timestep_crossover' or plan['command']!=command_for(plan['video']):
        raise ValueError('Crossover plan differs from the declared protocol')
    head.checked_inputs(plan); verify_baseline(root,plan)
    return head.run_process(root,'crossover',plan['command'],lambda directory:validate_result(directory,root,plan))


def make_report(directory,report):
    directory=Path(directory)
    rows=read_json(directory/'metrics.json')
    page=['<!doctype html><meta charset="utf-8"><title>Wan noise versus timestep</title>',
        '<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>',
        '<h1>Input noise versus timestep conditioning</h1>',
        '<p>Replication: '+('PASS' if report['interpretation_ready'] else 'MISMATCH: review evidence before interpreting crossed conditions')+'</p>',
        '<p>EPE: lower is better, in patch units. Cosine: closer to +1 is better. Amplitude: closer to +1 is better; negative reverses direction. '
        'Hard reference and soft target readouts are shown separately. The mean is an AMF readout of the same Wan activations; this is not a generated-video comparison.</p>',
        '<p>Block 30, head 30 and mean logits; sharpening 2. Two crossed conditions deliberately mismatch input noise and model timestep. '
        'They diagnose the clean-reference failure and are not proposed generation settings. No candidate promotion or video-quality conclusion.</p>',
        '<p>Rows below use nominal textured support. metrics.csv and affine_report.html retain all offsets/supports; NPZ files retain individual pairs.</p>',
        '<table><tr>'+''.join('<th>'+k+'</th>' for k in ('Input noise','Model time','Readout','Field','Control','EPE','Cosine','Amplitude','Confidence','Entropy'))+'</tr>']
    for row in rows:
        if row['anchor_offset']!=0 or row['support']!='textured': continue
        values=[row['noise_sampling_index'],row['conditioning_sampling_index'],row['variant'],row['field'],row['control'],
                row['epe'],row['direction_cosine'],row['amplitude_ratio'],row['confidence'],row['entropy']]
        page.append('<tr>'+''.join('<td>'+html.escape('n/a' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v))+'</td>' for v in values)+'</tr>')
    page+=['</table><p>Indices: -1 means clean/timestep zero; 9 means sigma or model timestep from sampling index 9. '
           'These are denoising indices, not video-frame indices.</p>',
           '<details><summary>Audit and replication details</summary><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></details>']
    destination=directory/'crossover_report.html'; destination.write_text('\n'.join(page),encoding='utf-8')
    return destination


def summarize(root):
    root=Path(root).resolve(); plan=head.load_plan(root)
    directory=root/read_json(root/'crossover_done.json')['directory']
    report=validate_result(directory,root,plan)
    head.write_json(root/'crossover_audit.json',report)
    make_report(directory,report)
    return directory,report


archive_experiment=head.archive_experiment
