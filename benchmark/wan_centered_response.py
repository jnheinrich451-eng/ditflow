"""Matched two-dose response: four positive captures, no backwards or sampling."""
from pathlib import Path
import json

import numpy as np
import torch

from benchmark.wan_centered_attribution import digest, regions
from benchmark.wan_centered_gradients import (
    prepare_gradient_inputs, capture, readout_analysis, compare, write_json)
from benchmark.wan_centered_huber import frozen_model, scores, separation, torso_scores_by_pair


DOSES = (('full', 1.), ('tenth', .1))


def prepare_response_inputs(pilot, attribution, huber, gradients):
    inputs, huber_report = prepare_gradient_inputs(pilot, attribution, huber)
    root = Path(gradients)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'gradient_report.json').read_text())
    if protocol['inputs_sha256'] != inputs[4] or protocol['huber_report_sha256'] != digest(Path(huber)/'huber_report.json'):
        raise ValueError('Gradient diagnostic used different saved inputs')
    if not report['latent_unchanged'] or not report['archived_gradient_norms_match']:
        raise ValueError('Gradient diagnostic must preserve the latent and match archived gradient norms')
    if report['counts'] != dict(positive_forwards_attempted=1,latent_backwards_attempted=2,latent_backwards_completed=2):
        raise ValueError('Requires completed one-capture/two-backward diagnostic')
    sources = {**protocol['helper_sources_sha256'], 'wan_centered_gradients.py':protocol['script_sha256']}
    for name, expected in sources.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Diagnostic source changed since gradient capture: {name}')
    with np.load(root/'before_flow.npz',allow_pickle=False) as values:
        np.testing.assert_array_equal(values['flow'],inputs[5]['before'])
    with np.load(root/'readout_derivatives.npz',allow_pickle=False) as values:
        before_readout = {k:values[k].copy() for k in values.files}
    np.testing.assert_array_equal(before_readout['flow'],inputs[5]['before'])
    for key in ('same_probability','flow_jacobian_frobenius'):
        values = before_readout[key]
        if values.shape != inputs[5]['before'].shape[:-1] or not np.isfinite(values).all() or (values<0).any():
            raise ValueError(f'Invalid saved readout field: {key}')
    if (before_readout['same_probability']>1.000001).any():
        raise ValueError('Invalid saved readout probability')
    saved = {}
    for arm in ('forward','reverse'):
        with np.load(root/f'{arm}_latent_gradient.npz',allow_pickle=False) as values:
            saved[arm] = values['gradient'].copy()
        if saved[arm].dtype != np.float32 or saved[arm].shape != inputs[1]['before'].shape or not np.isfinite(saved[arm]).all():
            raise ValueError('Invalid saved FP32 latent gradient')
        norm = float(np.linalg.norm(saved[arm].astype(float)))
        if norm == 0 or not np.isclose(norm,report['latent_gradient_comparison'][arm+'_norm'],rtol=1e-6,atol=1e-8):
            raise ValueError('Saved gradient disagrees with completed report')
        np.testing.assert_array_equal(before_readout[arm+'_valid'],inputs[2][arm][1])
    provenance = {p.name:digest(p) for p in [root/'started.json',root/'gradient_report.json',
        root/'before_flow.npz',root/'readout_derivatives.npz',root/'forward_latent_gradient.npz',root/'reverse_latent_gradient.npz']}
    return inputs, saved, before_readout, provenance


def first_adam_proposal(gradient):
    """Zero initial moments, lr=.001, epsilon=1e-8, no weight decay; no optimizer."""
    return -.001*gradient/(gradient.abs()+1e-8)


def loss_response(before, after, refs, gradients, delta):
    result = {}
    for objective,(target,valid) in refs.items():
        a,b = scores(before,target,valid), scores(after,target,valid)
        predicted = float((gradients[objective].astype(np.float64)*delta.astype(np.float64)).sum())
        observed = b['huber_delta1']-a['huber_delta1']
        result[objective] = dict(before=a,after=b,actual_huber_change=observed,
            first_order_huber_change=predicted, linear_prediction_error=observed-predicted,
            observed_to_predicted_ratio=observed/predicted if predicted else None,
            normalized_prediction_error=abs(observed-predicted)/abs(predicted) if predicted else None)
    return result


def torso_response(before, after, ref, valid, common, before_readout, after_readout, arm):
    """Evaluate FIXED before-defined groups, never redefine saturation after treatment."""
    torso = regions()['torso']
    base = valid & torso
    without = common & torso
    without = without.copy(); without[2,12*52+33] = False
    groups = dict(own_valid=base, common_valid=common & torso, common_without_outlier=without,
        before_sharp=base & (before_readout['same_probability']>.99),
        before_other=base & (before_readout['same_probability']<=.99))
    result = {}
    for name,mask in groups.items():
        if not mask.any():
            result[name] = dict(count=0)
            continue
        wanted = ref.astype(float)[mask]-before[mask]
        change = after.astype(float)[mask]-before[mask]
        energy = float((wanted*wanted).sum())
        cosine_denom = float(np.linalg.norm(change)*np.linalg.norm(wanted))
        result[name] = dict(count=int(mask.sum()), before=scores(before,ref,mask), after=scores(after,ref,mask),
            actual_change_rms=float(np.sqrt((change**2).mean())),
            desired_change_rms=float(np.sqrt((wanted**2).mean())),
            change_cosine=float((change*wanted).sum()/cosine_denom) if cosine_denom else None,
            change_projection_gain=float((change*wanted).sum()/energy) if energy else None,
            before_median_sensitivity=float(np.median(before_readout['flow_jacobian_frobenius'][mask])),
            after_median_sensitivity=float(np.median(after_readout['flow_jacobian_frobenius'][mask])),
            before_same_probability_median=float(np.median(before_readout['same_probability'][mask])),
            after_same_probability_median=float(np.median(after_readout['same_probability'][mask])),
            after_same_probability_above_099_count=int((after_readout['same_probability'][mask]>.99).sum()))
    return result


@torch.no_grad()
def run_response(g, pilot, attribution, huber, gradients, output):
    from benchmark.wan_centered_pilot import clear
    from benchmark.wan_port_acceptance import difference
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Four-capture response budget already started; preserve existing artifacts')
    inputs, saved, before_readout, gradient_hashes = prepare_response_inputs(pilot,attribution,huber,gradients)
    manifest, latents, refs, _, hashes, old = inputs
    live = frozen_model(g,manifest)
    output.mkdir(parents=True,exist_ok=True)
    protocol = dict(
        hypothesis='A finite first step may leave the useful local response regime while sharply concentrated torso queries remain insensitive.',
        expected='Compare loss linearity, effective cast-visible steps, and target-aligned torso responses at two doses; inspect the same queries sensitivity in the same captures.',
        max_positive_forwards=4, max_backwards=0, max_optimizer_steps=0, max_scheduler_steps=0, max_decodes=0,
        changed_factor='Scale the SAME saved first-Adam proposal by 1 or 0.1; no new gradient or optimizer iteration.',
        fixed=dict(index=39,sigma=manifest['experiment']['guidance_sigma'],block=20,heads='all 40',multiplier=8,
            loss='Huber delta=1',base_lr=.001,epsilon=1e-8,masks='original unchanged',injection=False),
        inputs_sha256=hashes, gradient_artifacts_sha256=gradient_hashes,live=live,
        script_sha256=digest(__file__), helper_sources_sha256={name:digest(Path(__file__).with_name(name))
            for name in ('wan_centered_gradients.py','wan_centered_huber.py','wan_centered_attribution.py','wan_centered_pilot.py')},
        limits='No timing/readout-formula change. Baseline is the previous exactly replayed saved state. '
            'This tests finite response at two doses, not a full gradient finite-difference limit. '
            'Source derivatives are analytical; no backward is added by readout analysis. '
            'No universal root-cause or motion-success conclusion is automatic.')
    with (output/'started.json').open('x',encoding='utf-8') as handle:
        json.dump(protocol,handle,indent=2)
    print('HYPOTHESIS:',protocol['hypothesis'],flush=True)
    print('LIMIT: four positive captures; zero backwards, optimizer steps, scheduler steps or videos.',flush=True)
    progress = dict(positive_forwards_attempted=0,positive_forwards_completed=0)
    report = dict(port_success=False,status='Matched finite-step/readout diagnostic; no decoded-motion claim',conditions={})
    all_fields, actual_deltas, cast_deltas = {},{},{}
    common = refs['forward'][1] & refs['reverse'][1]
    try:
        prefix = torch.from_numpy(latents['before']).to(g.device)
        cast_before = prefix.to(g.dtype)
        for arm in ('forward','reverse'):
            gradient = torch.from_numpy(saved[arm]).to(g.device)
            proposal = first_adam_proposal(gradient)
            for label,scale in DOSES:
                name = f'{arm}_{label}'
                folder = output/name; folder.mkdir()
                # Every condition starts from the SAME prefix, not from the preceding condition.
                x = prefix+scale*proposal
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite proposed latent')
                delta = (x-prefix).cpu().numpy()
                cast_delta = (x.to(g.dtype).float()-cast_before.float()).cpu().numpy()
                actual_deltas[name],cast_deltas[name] = delta,cast_delta
                np.savez_compressed(folder/'actual_delta.npz',fp32=delta,after_cast=cast_delta)
                progress['positive_forwards_attempted'] += 1
                write_json(output/'progress.json',progress)
                flow,q,k = capture(g,x)
                after = flow.float().cpu().numpy().copy()
                if not np.isfinite(after).all():
                    raise RuntimeError('Nonfinite response flow')
                analysis, reconstructed = readout_analysis(q,k,refs,folder)
                if not np.allclose(after,reconstructed,rtol=0,atol=1e-4):
                    raise RuntimeError('Detached readout differs from frozen kernel')
                with np.load(folder/'readout_derivatives.npz',allow_pickle=False) as values:
                    readout = {key:values[key].copy() for key in values.files}
                np.savez_compressed(folder/'flow.npz',flow=after)
                write_json(folder/'readout_analysis.json',analysis)
                all_fields[name] = after
                condition = dict(arm=arm,scale=scale,equivalent_first_step_lr=.001*scale,
                    delta_fp32=difference(x,prefix), delta_after_cast=difference(x.to(g.dtype),cast_before),
                    cast_delta_vs_fp32_cosine=compare(delta,cast_delta)['cosine'],
                    losses=loss_response(old['before'],after,refs,saved,delta),
                    torso=torso_response(old['before'],after,*refs[arm],common,before_readout,readout,arm),
                    torso_scores_by_pair=torso_scores_by_pair(after,*refs[arm],common),
                    dominated_patch_flow=after[2,12*52+33].tolist(),
                    readout_reconstruction_max_abs=float(np.abs(after-reconstructed).max()))
                report['conditions'][name] = condition
                write_json(folder/'response.json',condition)
                progress['positive_forwards_completed'] += 1
                write_json(output/'progress.json',progress)
                print(name,'own Huber change:',condition['losses'][arm]['actual_huber_change'],
                      'cast-visible RMS:',condition['delta_after_cast']['rms'],flush=True)
                clear(g)
                del flow,q,k,x,after,readout
        np.testing.assert_array_equal(prefix.cpu().numpy(),latents['before'])
        report['by_dose'] = {}
        excluded = {a:(f,m.copy()) for a,(f,m) in refs.items()}
        for a in excluded:
            excluded[a][1][2,12*52+33] = False
        for label,scale in DOSES:
            f,r = f'forward_{label}',f'reverse_{label}'
            report['by_dose'][label] = dict(scale=scale,
                separation=separation(all_fields[f],all_fields[r],refs),
                separation_excluding_shared_patch=separation(all_fields[f],all_fields[r],excluded),
                fp32_update_comparison=compare(actual_deltas[f],actual_deltas[r]),
                cast_update_comparison=compare(cast_deltas[f],cast_deltas[r]))
        report['counts'] = progress
        report['baseline_unchanged'] = True
        report['interpretation'] = ('A more linear loss response alone is insufficient: assess common-torso target alignment '
            'and own-target errors together. Sharp groups are fixed from the BEFORE state. '
            'Smaller BF16-visible changes may be sparse rather than a smooth scaled direction. '
            'This run does not test earlier guidance timing or prove correspondence/decoded motion correctness.')
        write_json(output/'response_report.json',report)
        return report
    except Exception as error:
        write_json(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),counts=progress))
        raise
    finally:
        clear(g)
