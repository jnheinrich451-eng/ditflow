"""Fixed common-torso objective support; matched full-field NLL control.

This tests a mechanism on one saved camel state, not a general foreground method.
Prior experiment sources remain frozen. No native attention or centering is cropped.
"""
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.wan_centered_attribution import digest, regions
from benchmark.wan_centered_destination import (
    ARMS, destination_objectives, prepare_destination_inputs, summarize_target_probability)
from benchmark.wan_centered_gradients import capture, compare, readout_analysis, write_json
from benchmark.wan_centered_huber import frozen_model, scores, separation, torso_scores_by_pair
from benchmark.wan_centered_response import first_adam_proposal, torso_response


def common_torso_refs(refs):
    """The existing ROI and BOTH original masks; no target values are changed."""
    common = refs['forward'][1] & refs['reverse'][1] & regions()['torso']
    if common.shape != (5, 1560) or common.dtype != np.bool_:
        raise ValueError('Unexpected shared torso grid')
    if common.sum(1).tolist() != [70, 31, 101, 26, 96] or not common[2, 12*52+33]:
        raise ValueError('Frozen 324-query torso support changed; do not retune the ROI')
    return {a:(f, common.copy()) for a,(f,_) in refs.items()}, common


def support_nll(logp, support):
    return {a:dict(mean=float(-v[support].astype(float).mean()),
                   per_pair=[float(-v[i][support[i]].astype(float).mean()) for i in range(5)])
            for a,v in logp.items()}


def prepare_torso_inputs(pilot, attribution, huber, gradients, response, destination):
    checked = prepare_destination_inputs(pilot, attribution, huber, gradients, response)
    inputs, base, indices, _, _, provenance, geometry = checked
    _, support = common_torso_refs(inputs[2])
    root = Path(destination)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'destination_report.json').read_text())
    if protocol['inputs_sha256'] != inputs[4] or protocol['control_sha256'] != provenance:
        raise ValueError('Full-field NLL control used different saved inputs')
    if not report['baseline_unchanged'] or report['counts'] != dict(
            positive_forwards_attempted=3, positive_forwards_completed=3,
            latent_backwards_attempted=2, latent_backwards_completed=2, first_adam_proposals=2):
        raise ValueError('Requires completed three-capture full-field NLL control')
    for name, expected in {**protocol['helper_sources_sha256'],
                          'wan_centered_destination.py':protocol['script_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Full-field NLL control source changed: {name}')
    paths = [root/'started.json', root/'destination_report.json', root/'before_flow.npz',
             root/'before_target_log_probability.npz']
    with np.load(root/'before_flow.npz', allow_pickle=False) as values:
        np.testing.assert_array_equal(values['flow'], inputs[5]['before'])
    controls = {}
    for arm in ARMS:
        field_path = root/arm/'flow.npz'
        with np.load(field_path, allow_pickle=False) as values:
            field = values['flow'].copy()
        if field.shape != inputs[5]['before'].shape or not np.isfinite(field).all():
            raise ValueError('Invalid full-field control flow')
        for target in ARMS:
            actual = scores(field, *inputs[2][target])
            expected = report['arms'][arm]['original_objectives'][target]['after']
            if any(not np.isclose(actual[k], expected[k], rtol=1e-9, atol=1e-9) for k in actual):
                raise ValueError('Full-field control flow and reported errors disagree')
        log_path = root/arm/'target_log_probability.npz'
        with np.load(log_path, allow_pickle=False) as values:
            logs = {a:values[a].copy() for a in ARMS}
        if any(v.shape != support.shape or not np.isfinite(v).all() or (v > 1e-6).any()
               for v in logs.values()):
            raise ValueError('Invalid full-field target log probabilities')
        summary = summarize_target_probability(logs, inputs[2], base)
        for target in ARMS:
            for group, stats in summary[target].items():
                expected = report['arms'][arm]['target_probability'][target][group]
                for key, value in stats.items():
                    if value is None:
                        if expected[key] is not None:
                            raise ValueError('Full-field probability summary mismatch')
                    elif not np.isclose(value, expected[key], rtol=1e-9, atol=1e-12):
                        raise ValueError('Full-field probability summary mismatch')
        paths.extend([field_path, log_path])
        controls[arm] = field
    provenance = {**provenance, 'destination_artifacts':{
        str(p.relative_to(root)):digest(p) for p in paths}}
    geometry = {**geometry, 'optimization_support':dict(
        count=int(support.sum()), per_pair=support.sum(1).tolist(),
        includes_known_outlier=True, roi='Existing torso ROI: x350..590, y140..260 at 832x480',
        all_source_rows_in_centering=True, all_destination_tokens_in_softmax=True)}
    return inputs, base, indices, controls, report, provenance, geometry


def run_torso(g, pilot, attribution, huber, gradients, response, destination, output):
    from benchmark.wan_centered_pilot import clear
    from benchmark.wan_port_acceptance import difference
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Common-torso budget already started; preserve existing artifacts')
    inputs, before_readout, indices, controls, control_report, provenance, geometry = (
        prepare_torso_inputs(pilot, attribution, huber, gradients, response, destination))
    manifest, latents, refs, _, hashes, old = inputs
    optimization_refs, support = common_torso_refs(refs)
    live = frozen_model(g, manifest)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        hypothesis='Full-field NLL may obscure a useful torso response; identical common-torso objective support may recover reference specificity.',
        expected='Both arms improve old torso AMF errors and separate in requested directions across supported pairs, including initially sharp queries.',
        changed_factor='Original full-field NLL support -> the same 324 common-valid torso source queries in each arm',
        max_positive_forwards=3, max_latent_backwards=2, max_first_adam_proposals=2,
        max_scheduler_steps=0, max_decodes=0,
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], block=20, heads='all 40',
            multiplier=8, lr=.001, epsilon=1e-8, masks='original for evaluation; common-valid torso for optimization', injection=False),
        inputs_sha256=hashes, control_sha256=provenance, target_geometry=geometry, live=live,
        script_sha256=digest(__file__), helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('wan_centered_destination.py','wan_centered_response.py','wan_centered_gradients.py','wan_centered_huber.py',
             'wan_centered_attribution.py','wan_centered_pilot.py')},
        limits='Three explicit feature forwards; checkpoint blocks also recompute during backwards. '
            'Loss support and its mean normalization change; report gradients and actual Adam/cast deltas. '
            'All source rows remain in centering. This is not a spatial restriction on latent gradients. '
            'Targets are geometrically checked, not proven physical tracks. No new native parity or decoded-motion claim.')
    with (output/'started.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2)
    np.savez_compressed(output/'optimization_support.npz', common_torso=support,
        forward_original_valid=refs['forward'][1], reverse_original_valid=refs['reverse'][1])
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: 3 positive captures, 2 backwards, 2 first-Adam proposals; no sampling or videos.', flush=True)
    progress = dict(positive_forwards_attempted=0, positive_forwards_completed=0,
                    latent_backwards_attempted=0, latent_backwards_completed=0, first_adam_proposals=0)
    all_gradients, deltas, cast_deltas, after_fields = {}, {}, {}, {}
    common = refs['forward'][1] & refs['reverse'][1]
    report = dict(port_success=False, status='Common-torso loss-support diagnostic; no decoded-motion claim', arms={})

    def captured(x):
        progress['positive_forwards_attempted'] += 1
        write_json(output/'progress.json', progress)
        values = capture(g, x)
        if not torch.isfinite(values[0]).all():
            raise RuntimeError('Nonfinite captured flow')
        progress['positive_forwards_completed'] += 1
        write_json(output/'progress.json', progress)
        return values

    try:
        with torch.enable_grad():
            x = torch.from_numpy(latents['before']).to(g.device).clone().requires_grad_(True)
            flow, q, k = captured(x)
            before = flow.detach().float().cpu().numpy().copy()
            # Reuse the budgeted before capture to detect incompatible loaded state.
            if not np.allclose(before, old['before'], rtol=.005, atol=.005):
                raise RuntimeError('Saved before field does not replay; no backwards performed')
            np.savez_compressed(output/'before_flow.npz', flow=before)
            losses, before_logp = destination_objectives(q, k, optimization_refs, indices)
            np.savez_compressed(output/'before_target_log_probability.npz', **before_logp)
            report['before_target_probability'] = summarize_target_probability(before_logp, refs, before_readout)
            report['before_optimization_nll'] = support_nll(before_logp, support)
            report['before_replay_max_abs'] = float(np.abs(before-old['before']).max())
            # Drop the unused coordinate-loss graph; retain the shared Q/K NLL graph.
            del flow, q, k
            for i, arm in enumerate(ARMS):
                loss = losses[arm]
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite destination loss')
                progress['latent_backwards_attempted'] += 1
                write_json(output/'progress.json', progress)
                grad, = torch.autograd.grad(loss, x, retain_graph=i == 0)
                if not torch.isfinite(grad).all() or float(grad.norm()) == 0:
                    raise RuntimeError('Nonfinite or zero latent gradient')
                all_gradients[arm] = grad.detach().float().cpu().numpy().copy()
                np.savez_compressed(output/f'{arm}_latent_gradient.npz', gradient=all_gradients[arm])
                progress['latent_backwards_completed'] += 1
                write_json(output/'progress.json', progress)
                g._clear_kv([20])
                del grad, loss
            np.testing.assert_array_equal(x.detach().cpu().numpy(), latents['before'])
            del losses, x
        clear(g)
        with torch.no_grad():
            prefix = torch.from_numpy(latents['before']).to(g.device)
            cast_before = prefix.to(g.dtype)
            for arm in ARMS:
                folder = output/arm
                folder.mkdir()
                grad = torch.from_numpy(all_gradients[arm]).to(g.device)
                x = prefix + first_adam_proposal(grad)
                progress['first_adam_proposals'] += 1
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite proposed latent')
                delta = (x-prefix).cpu().numpy()
                cast_delta = (x.to(g.dtype).float()-cast_before.float()).cpu().numpy()
                deltas[arm], cast_deltas[arm] = delta, cast_delta
                np.savez_compressed(folder/'actual_delta.npz', fp32=delta, after_cast=cast_delta)
                flow, q, k = captured(x)
                after = flow.float().cpu().numpy().copy()
                _, logp = destination_objectives(q, k, optimization_refs, indices)
                np.savez_compressed(folder/'target_log_probability.npz', **logp)
                analysis, reconstructed = readout_analysis(q, k, refs, folder)
                if not np.allclose(after, reconstructed, rtol=0, atol=1e-4):
                    raise RuntimeError('Detached readout differs from frozen kernel')
                with np.load(folder/'readout_derivatives.npz', allow_pickle=False) as values:
                    readout = {key:values[key].copy() for key in values.files}
                np.savez_compressed(folder/'flow.npz', flow=after)
                write_json(folder/'readout_analysis.json', analysis)
                after_fields[arm] = after
                condition = dict(
                    gradient_norm=float(np.linalg.norm(all_gradients[arm].astype(float))),
                    delta_fp32=difference(x, prefix), delta_after_cast=difference(x.to(g.dtype), cast_before),
                    cast_delta_vs_fp32_cosine=compare(delta, cast_delta)['cosine'],
                    target_probability=summarize_target_probability(logp, refs, before_readout),
                    optimization_nll=support_nll(logp, support),
                    original_objectives={a:dict(before=scores(before, *refs[a]), after=scores(after, *refs[a])) for a in ARMS},
                    torso=torso_response(before, after, *refs[arm], common, before_readout, readout, arm),
                    torso_scores_by_pair=torso_scores_by_pair(after, *refs[arm], common),
                    dominated_patch_flow=after[2, 12*52+33].tolist(),
                    full_field_control={key:control_report['arms'][arm][key] for key in (
                        'gradient_norm','delta_fp32','delta_after_cast','target_probability',
                        'original_objectives','torso','torso_scores_by_pair')})
                report['arms'][arm] = condition
                write_json(folder/'response.json', condition)
                print(arm, 'old objectives:', condition['original_objectives'][arm], flush=True)
                clear(g)
                del flow, q, k, x, grad, readout
            np.testing.assert_array_equal(prefix.cpu().numpy(), latents['before'])
        excluded = {a:(f, m.copy()) for a,(f,m) in refs.items()}
        for a in ARMS:
            excluded[a][1][2, 12*52+33] = False
        report.update(
            separation=separation(after_fields['forward'], after_fields['reverse'], refs),
            separation_excluding_shared_patch=separation(after_fields['forward'], after_fields['reverse'], excluded),
            full_field_control_separation=separation(controls['forward'], controls['reverse'], refs),
            full_field_control_separation_excluding_shared_patch=separation(controls['forward'], controls['reverse'], excluded),
            latent_gradient_comparison=compare(all_gradients['forward'], all_gradients['reverse']),
            fp32_update_comparison=compare(deltas['forward'], deltas['reverse']),
            cast_update_comparison=compare(cast_deltas['forward'], cast_deltas['reverse']),
            counts=progress, baseline_unchanged=True,
            optimization_support=dict(count=int(support.sum()), per_pair=support.sum(1).tolist()),
            full_field_gradient_comparison=control_report['latent_gradient_comparison'],
            full_field_fp32_update_comparison=control_report['fp32_update_comparison'],
            full_field_cast_update_comparison=control_report['cast_update_comparison'],
            interpretation='Lower NLL alone is insufficient. Compare both arms own-target old AMF errors, '
                'fixed initially-sharp torso queries, common-support separation and per-pair direction, '
                'with and without the shared outlier. No automatic video run or motion-success gate.')
        write_json(output/'torso_report.json', report)
        return report
    except Exception as error:
        write_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), counts=progress))
        raise
    finally:
        clear(g)

