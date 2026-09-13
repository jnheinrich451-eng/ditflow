"""Fixed-temperature head screen and streaming audit of the saved affine evidence."""
import html
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from guidance_utils.wan_affine_diagnostics import field_metrics
from guidance_utils.wan_amf_calibration import CONTROLS, MODEL, THRESHOLDS, screen_candidates

BLOCKS = [20, 30]
HEADS = list(range(40))


def screen_heads(rows):
    results = []
    for variant in sorted({r['variant'] for r in rows}):
        subset = [dict(r, temperature=2.) for r in rows if r['variant'] == variant]
        for candidate in screen_candidates(subset)['candidates']:
            results.append(dict(candidate, variant=variant))
    return dict(thresholds=THRESHOLDS, candidates=results,
                note='Fixed sharpening 2; no automatic head selection. Confirm on new texture/noise before generation.')


def audit_heads(root, blocks=BLOCKS, heads=None, noise_seed=17):
    """Recompute every metric while retaining only one capture at a time."""
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    root = Path(root)
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    rows = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    config = OmegaConf.to_container(OmegaConf.load(root/'suite_config.yaml'))
    fixed = dict(model_key=MODEL, enable_model_cpu_offload=True, guidance_blocks=[], injection_blocks=[],
        target_prompt='', source_prompt='', scheduler='flowmatch', flow_shift=3., num_frames=21,
        height=480, width=832, num_inference_steps=50, seed=1, reference_only=True,
        probe=False, probe_rope=False, motion_temp=2., flow_region_masks=None)
    if any(config.get(k) != v for k,v in fixed.items()) or config.get('flow_head') is not None:
        raise ValueError('Changed head diagnostic configuration')
    if (meta['model'] != MODEL or meta['blocks'] != list(blocks) or meta['controls'] != CONTROLS
            or meta['noise_seed'] != noise_seed or meta['conditioning'] != '' or meta['temperature'] != 2.
            or meta['grid'] != [6,30,52] or not meta['cpu_offload'] or meta['mean_only']
            or meta.get('readout_heads') != heads or meta.get('readout_temperatures') is not None):
        raise ValueError('Head readout metadata differs from plan')
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
    states = [dict(noise_label='clean',sampling_index=-1,sigma=0.,timestep=0.)] + [
        dict(noise_label=f'step_{i:02d}',sampling_index=i,sigma=float(scheduler.sigmas[i]),
             timestep=float(scheduler.timesteps[i])) for i in (0,9,29)]
    if meta['noise_states'] != states:
        raise ValueError('Head readout noise schedule changed')
    variants = ['mean_logits', *[f'head_{i:02d}' for i in (HEADS if heads is None else heads)]]
    names = [f'block_{i}_attn1_processor' for i in blocks]
    expected = {(c,s['noise_label'],b,v,o,p,f) for c in CONTROLS for s in states for b in names
        for v in variants for o in (-2,0,2) for p in ('geometry','textured') for f in ('hard','soft')}
    keys = [(r['control'],r['noise_label'],r['block'],r['variant'],r['anchor_offset'],r['support'],r['field']) for r in rows]
    complete = json.loads((root/'complete.json').read_text(encoding='utf-8'))
    if set(keys) != expected or len(keys) != len(expected) or complete != dict(forward_passes=20,rows=len(expected)):
        raise ValueError('Missing, duplicated or extra head observations')
    grouped = defaultdict(list)
    for key,row in zip(keys,rows): grouped[key[:4]].append(row)
    truths = {}
    for control in CONTROLS:
        with np.load(root/control/'truth.npz',allow_pickle=False) as z:
            truths[control] = {k:z[k] for k in z.files}
    for (control,state,block,variant), items in grouped.items():
        with np.load(root/control/state/block/f'{variant}.npz',allow_pickle=False) as z:
            arrays = {k:z[k] for k in z.files}
        if set(arrays) != {'soft','hard','confidence','entropy','logit_std'}:
            raise ValueError('Missing head capture fields')
        for key,array in arrays.items():
            shape = (5,1560,2) if key in ('soft','hard') else (5,1560)
            if array.shape != shape or not np.isfinite(array).all():
                raise ValueError('Invalid head capture array')
        if state == 'step_00' and control != CONTROLS[0]:
            with np.load(root/CONTROLS[0]/state/block/f'{variant}.npz',allow_pickle=False) as first:
                if any(not np.array_equal(first[k],v) for k,v in arrays.items()):
                    raise ValueError('Pure-noise head fields differ across controls')
        truth = truths[control]
        schedule = next(s for s in states if s['noise_label']==state)
        for row in items:
            if any(row[k] != v for k,v in schedule.items()):
                raise ValueError('Metric noise label disagrees with schedule')
            offset = row['anchor_offset']; selected = truth[f'geometry_{offset}'].copy()
            if row['support']=='textured': selected &= truth[f'texture_{offset}']
            values = field_metrics(arrays[row['field']],truth[f'flow_{offset}'],selected)
            if values['patches'] <= 0 or values['finite_fraction'] != 1:
                raise ValueError('Empty support or nonfinite head prediction')
            values.update(entropy=float(arrays['entropy'][selected].mean()),
                          confidence=float(arrays['confidence'][selected].mean()))
            for key,value in values.items():
                equal = row[key] is None if value is None else row[key] is not None and np.isclose(value,row[key],rtol=1e-6,atol=1e-7)
                if not equal: raise ValueError(f'Saved metric disagrees with array: {key}')
    return dict(forward_passes=20,rows=len(rows),captures=len(grouped),recomputed=True,pure_noise_equal=True,
                note='Structural audit only; see head screening and decoded outputs.')


def make_head_report(root):
    root = Path(root)
    rows = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    screen = screen_heads(rows)
    (root/'head_screening.json').write_text(json.dumps(screen,indent=2),encoding='utf-8')
    page = ['<!doctype html><meta charset="utf-8"><title>Wan 14B individual heads</title>',
        '<style>body{font:15px system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccc}</style>',
        '<h1>Individual heads at sharpening 2</h1><p>No automatic winner. Both motion signs, amplitude and static error must pass at clean/step9/step29. '
        'Pure noise is excluded. These forward-noised controls are not sampled trajectories. '
        'Inspect affine_report.html and metrics.csv for hard/soft fields, per-pair captures and offsets.</p>',
        '<table><tr><th>Block</th><th>Readout</th><th>Screen</th><th>Failing conditions</th></tr>']
    for c in screen['candidates']:
        page.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in (
            c['block'],c['variant'],'PASS; requires confirmation' if c['screen_pass'] else 'FAIL','; '.join(c['failures'])))+'</tr>')
    (root/'head_report.html').write_text('\n'.join(page+['</table>']),encoding='utf-8')
    return screen
