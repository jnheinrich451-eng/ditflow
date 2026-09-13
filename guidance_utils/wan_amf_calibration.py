"""Detached sharpening sweep on shared Wan Q/K; no native attention changes."""
import csv
import html
import json
import math
from pathlib import Path

import numpy as np
import torch

from guidance_utils.wan_affine_diagnostics import field_metrics

CONTROLS = ['static', 'pan_right', 'pan_left', 'expand', 'contract']
TEMPERATURES = [2., 4., 8., 16.]
BLOCKS = [10, 20, 30]
MODEL = 'Wan-AI/Wan2.1-T2V-14B-Diffusers'
THRESHOLDS = dict(static_epe_max=.5, translation_cosine_min=.8, scale_cosine_min=.6,
                  amplitude_min=.5, amplitude_max=1.5)


def validate_temperatures(values):
    values = list(map(float, values))
    if not values or len(set(values)) != len(values) or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('AMF sharpening values must be distinct, finite and positive')
    return values


def variant_name(value):
    return f'temperature_{float(value):g}'


@torch.no_grad()
def shared_readouts(q, k, grid, temperatures):
    """One FP32 logit product per adjacent pair, reused for every sharpening value."""
    temperatures = validate_temperatures(temperatures)
    frames, h, w = grid
    if q.shape != k.shape or q.ndim != 4 or q.shape[1] != frames*h*w:
        raise ValueError('Invalid calibration Q/K geometry')
    heads, dim = q.shape[-2:]
    with torch.autocast(device_type=q.device.type, enabled=False):
        qf = q[-1].detach().float().reshape(frames, h*w, -1)
        kf = k[-1].detach().float().reshape(frames, h*w, -1)
        yy, xx = torch.meshgrid(torch.arange(h, device=q.device), torch.arange(w, device=q.device), indexing='ij')
        xy = torch.stack((xx.flatten(), yy.flatten()), -1).float()
        result = {t: {key: [] for key in ('soft', 'hard', 'confidence', 'entropy', 'logit_std')} for t in temperatures}
        for i in range(frames-1):
            logits = (qf[i] @ kf[i+1].T) / (heads*math.sqrt(dim))
            hard = (xy[logits.argmax(-1)]-xy).cpu().numpy()
            deviation = logits.std(-1, unbiased=False).cpu().numpy()
            for t in temperatures:
                probability = (logits*t).softmax(-1)
                result[t]['soft'].append((probability @ xy-xy).cpu().numpy())
                result[t]['hard'].append(hard)
                result[t]['confidence'].append(probability.max(-1).values.cpu().numpy())
                entropy = -(probability*probability.clamp_min(1e-30).log()).sum(-1)/math.log(max(h*w, 2))
                result[t]['entropy'].append(entropy.cpu().numpy())
                result[t]['logit_std'].append(deviation)
    return {t: {key: np.stack(value) for key, value in fields.items()} for t, fields in result.items()}


class CalibrationObserver:
    rope_enabled = False
    context = None

    def __init__(self, root, grid, temperatures, rows):
        self.root, self.grid, self.temperatures, self.rows = Path(root), grid, validate_temperatures(temperatures), rows
        self.active, self.seen = None, set()

    def attention(self, block_name, query, key, injected=False):
        if self.active is None:
            return
        if injected or block_name in self.seen:
            raise ValueError('Injected or duplicated calibration capture')
        self.seen.add(block_name)
        labels, truths = self.active
        folder = self.root/labels['control']/labels['noise_label']/block_name
        folder.mkdir(parents=True, exist_ok=True)
        for temperature, arrays in shared_readouts(query, key, self.grid, self.temperatures).items():
            variant = variant_name(temperature)
            np.savez_compressed(folder/f'{variant}.npz', **arrays)
            for offset, (truth, geometry, texture) in truths.items():
                for support, selected in [('geometry', geometry), ('textured', geometry & texture)]:
                    for field in ('hard', 'soft'):
                        self.rows.append(dict(**labels, block=block_name, variant=variant, temperature=temperature,
                            field=field, anchor_offset=offset, support=support,
                            **field_metrics(arrays[field], truth, selected),
                            entropy=float(arrays['entropy'][selected].mean()) if selected.any() else None,
                            confidence=float(arrays['confidence'][selected].mean()) if selected.any() else None))


def screen_candidates(rows):
    """Predeclared measurement screen. No pure-noise motion criterion or auto-winner."""
    nominal = [r for r in rows if r['field'] == 'soft' and r['support'] == 'textured' and r['anchor_offset'] == 0]
    candidates = sorted({(r['block'], r['temperature']) for r in nominal})
    results = []
    for block, temperature in candidates:
        failures = []
        for state in ('clean', 'step_09', 'step_29'):
            for control in CONTROLS:
                matches = [r for r in nominal if (r['block'], r['temperature'], r['noise_label'], r['control']) == (block, temperature, state, control)]
                prefix = f'{state}/{control}'
                if len(matches) != 1:
                    failures.append(prefix+': missing or duplicate row'); continue
                r = matches[0]
                if r['patches'] <= 0 or r['finite_fraction'] != 1:
                    failures.append(prefix+': invalid support/prediction'); continue
                if control == 'static':
                    if r['epe'] is None or not math.isfinite(r['epe']) or r['epe'] > THRESHOLDS['static_epe_max']:
                        failures.append(prefix+': static EPE > 0.5')
                else:
                    threshold = THRESHOLDS['translation_cosine_min'] if control.startswith('pan_') else THRESHOLDS['scale_cosine_min']
                    if r['direction_cosine'] is None or not math.isfinite(r['direction_cosine']) or r['direction_cosine'] < threshold:
                        failures.append(prefix+f': direction cosine < {threshold}')
                    if r['amplitude_ratio'] is None or not THRESHOLDS['amplitude_min'] <= r['amplitude_ratio'] <= THRESHOLDS['amplitude_max']:
                        failures.append(prefix+': amplitude outside [0.5, 1.5]')
        results.append(dict(block=block, temperature=temperature, screen_pass=not failures, failures=failures))
    return dict(thresholds=THRESHOLDS, candidates=results,
                note='Measurement screen only. Inspect offsets/pairs, confirm a fixed candidate on new texture/noise, then test actual latent gradients and decoded motion.')


def audit_calibration(root, blocks=BLOCKS, temperatures=TEMPERATURES, noise_seed=17):
    from diffusers import FlowMatchEulerDiscreteScheduler
    from omegaconf import OmegaConf
    root = Path(root)
    meta = json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    rows = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    complete = json.loads((root/'complete.json').read_text(encoding='utf-8'))
    config = OmegaConf.to_container(OmegaConf.load(root/'suite_config.yaml'))
    fixed = dict(model_key=MODEL, enable_model_cpu_offload=True, guidance_blocks=[], injection_blocks=[],
                 target_prompt='', source_prompt='', scheduler='flowmatch', flow_shift=3.,
                 num_frames=21, height=480, width=832, num_inference_steps=50, seed=1,
                 reference_only=True, probe=False, probe_rope=False, motion_temp=2.)
    if any(config.get(key) != value for key,value in fixed.items()):
        raise ValueError('Changed calibration model configuration')
    if (meta['model'] != MODEL or meta['blocks'] != list(blocks) or meta['readout_temperatures'] != list(temperatures)
            or meta['controls'] != CONTROLS or meta['noise_seed'] != noise_seed or meta['conditioning'] != ''
            or meta['grid'] != [6,30,52] or not meta['cpu_offload'] or not meta['mean_only']
            or [s['sampling_index'] for s in meta['noise_states']] != [-1,0,9,29]):
        raise ValueError('Calibration metadata differs from plan')
    scheduler = FlowMatchEulerDiscreteScheduler(shift=3.)
    scheduler.set_timesteps(50)
    for state in meta['noise_states']:
        index = state['sampling_index']
        sigma, timestep = (0.,0.) if index == -1 else (float(scheduler.sigmas[index]),float(scheduler.timesteps[index]))
        if state['sigma'] != sigma or state['timestep'] != timestep:
            raise ValueError('Calibration noise schedule differs from plan')
    names = [f'block_{b}_attn1_processor' for b in blocks]
    states = [s['noise_label'] for s in meta['noise_states']]
    expected = {(b,t,c,s,o,p,f) for b in names for t in temperatures for c in CONTROLS for s in states
                for o in (-2,0,2) for p in ('geometry','textured') for f in ('hard','soft')}
    keys = [(r['block'],r['temperature'],r['control'],r['noise_label'],r['anchor_offset'],r['support'],r['field']) for r in rows]
    if set(keys) != expected or len(keys) != len(expected) or complete != dict(forward_passes=20, rows=len(expected)):
        raise ValueError('Incomplete or duplicated calibration observations')
    captures, truths = {}, {}
    for control in CONTROLS:
        with np.load(root/control/'truth.npz', allow_pickle=False) as z:
            truths[control] = {k:z[k].copy() for k in z.files}
        for block in names:
            for state in states:
                hard = None
                for temperature in temperatures:
                    with np.load(root/control/state/block/f'{variant_name(temperature)}.npz', allow_pickle=False) as z:
                        arrays = {k:z[k].copy() for k in z.files}
                    if set(arrays) != {'soft','hard','confidence','entropy','logit_std'}:
                        raise ValueError('Missing calibration fields')
                    for key, array in arrays.items():
                        shape = (5,1560,2) if key in ('soft','hard') else (5,1560)
                        if array.shape != shape or not np.isfinite(array).all():
                            raise ValueError('Invalid calibration array')
                    if hard is not None and not np.array_equal(hard, arrays['hard']):
                        raise ValueError('Hard correspondence changed across sharpening values')
                    hard = arrays['hard']
                    captures[control,block,state,temperature] = arrays
    for block in names:
        for temperature in temperatures:
            first = captures[CONTROLS[0],block,'step_00',temperature]
            for control in CONTROLS[1:]:
                if any(not np.array_equal(first[key], captures[control,block,'step_00',temperature][key]) for key in first):
                    raise ValueError('Pure-noise fields differ across controls')
    for row in rows:
        if row['variant'] != variant_name(row['temperature']):
            raise ValueError('Mislabeled sharpening value')
        truth = truths[row['control']]; offset = row['anchor_offset']
        selected = truth[f'geometry_{offset}'].copy()
        if row['support'] == 'textured':
            selected &= truth[f'texture_{offset}']
        arrays = captures[row['control'],row['block'],row['noise_label'],row['temperature']]
        recomputed = field_metrics(arrays[row['field']], truth[f'flow_{offset}'], selected)
        if recomputed['patches'] <= 0 or recomputed['finite_fraction'] != 1:
            raise ValueError('Empty support or nonfinite metrics')
        for key, value in recomputed.items():
            if value is None:
                equal = row[key] is None
            else:
                equal = row[key] is not None and np.isclose(value,row[key],rtol=1e-6,atol=1e-7)
            if not equal:
                raise ValueError(f'Saved metric disagrees with array: {key}')
    return dict(model=MODEL, forward_passes=20, rows=len(rows), recomputed=True,
                pure_noise_equal=True, hard_equal_across_temperatures=True,
                note='Audit is not a motion-quality pass.')


def make_calibration_report(root):
    root = Path(root)
    rows = json.loads((root/'metrics.json').read_text(encoding='utf-8'))
    screen = screen_candidates(rows)
    (root/'screening.json').write_text(json.dumps(screen, indent=2), encoding='utf-8')
    with (root/'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    page = ['<!doctype html><meta charset="utf-8"><title>Wan AMF sharpening calibration</title>',
        '<style>body{font:15px system-ui;margin:24px;max-width:1600px}table{border-collapse:collapse}th,td{border:1px solid #ccc;padding:6px}th{position:sticky;top:0;background:#eee}</style>',
        '<h1>AMF sharpening calibration</h1><p>Same Q/K, multiple detached readouts. No latent update or target generation. ',
        'Higher sharpening multiplies logits more strongly. Pure noise contains no control-specific motion. ',
        'Nominal textured support is tabulated below; all pairs/offsets/supports are retained in NPZ and CSV.</p>',
        '<p>Screen: static EPE &le; 0.5; both translation cosines &ge; 0.8; both scale cosines &ge; 0.6; all moving amplitudes in [0.5,1.5], at clean/step9/step29. ',
        'These are declared measurement thresholds, not video-quality thresholds. No automatic winner.</p><ul>']
    for candidate in screen['candidates']:
        label = f"{candidate['block']} / sharpening {candidate['temperature']:g}"
        detail = 'SCREEN PASS; requires independent confirmation and gradient/decoded-response checks' if candidate['screen_pass'] else '; '.join(candidate['failures'])
        page.append('<li>'+html.escape(label+': '+detail)+'</li>')
    page.append('</ul><table><tr>'+''.join('<th>'+k+'</th>' for k in ('Block','Sharpening','Noise','Control','Field','EPE','Cosine','Amplitude','Confidence','Entropy'))+'</tr>')
    for row in rows:
        if row['anchor_offset'] != 0 or row['support'] != 'textured':
            continue
        values = [row[k] for k in ('block','temperature','noise_label','control','field','epe','direction_cosine','amplitude_ratio','confidence','entropy')]
        page.append('<tr>'+''.join('<td>'+html.escape('n/a' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v))+'</td>' for v in values)+'</tr>')
    page.append('</table>')
    destination = root/'calibration_report.html'
    destination.write_text('\n'.join(page),encoding='utf-8')
    return destination
