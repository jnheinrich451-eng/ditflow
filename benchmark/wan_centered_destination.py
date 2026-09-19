"""One-step destination NLL candidate; three feature captures, two backwards.

Experimental objective only. The frozen centered AMF kernel and Wan are unchanged.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmark.wan_centered_attribution import digest, regions
from benchmark.wan_centered_gradients import capture, compare, readout_analysis, write_json
from benchmark.wan_centered_huber import frozen_model, scores, separation, torso_scores_by_pair
from benchmark.wan_centered_response import prepare_response_inputs, first_adam_proposal, torso_response


ARMS = ('forward', 'reverse')


def destination_indices(flow, valid, height=30, width=52):
    """Validate hard XY displacements, preserving the original support exactly.

    Masked-out rows get a safe gather index, never a new validity decision.
    Geometric validity does NOT establish semantic correspondence correctness.
    """
    flow, valid = np.asarray(flow), np.asarray(valid)
    if flow.ndim != 3 or flow.shape[1:] != (height*width, 2) or valid.shape != flow.shape[:-1]:
        raise ValueError('Unexpected reference grid')
    if valid.dtype != np.bool_ or not valid.any() or not np.isfinite(flow[valid]).all():
        raise ValueError('Requires finite reference flow on nonempty boolean support')
    yy, xx = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    coordinates = np.stack((xx.ravel(), yy.ravel()), -1)
    destinations = flow.astype(np.float64) + coordinates
    selected = destinations[valid]
    if not np.allclose(selected, np.rint(selected), rtol=0, atol=1e-5):
        raise ValueError('Reference is not a hard integer destination')
    integer = np.rint(selected).astype(np.int64)
    if ((integer[:, 0] < 0) | (integer[:, 0] >= width) |
            (integer[:, 1] < 0) | (integer[:, 1] >= height)).any():
        raise ValueError('Reference destination lies outside the spatial grid')
    indices = np.zeros(valid.shape, np.int64)
    indices[valid] = integer[:, 1]*width + integer[:, 0]
    return indices


def destination_objectives(q, k, refs, indices, height=30, width=52):
    """Mean NLL per valid source, using exactly the frozen centered-T8 scores.

    Keep ALL source rows for centering and ALL destination tokens in log_softmax.
    Gather masks only afterwards. Stable log_softmax avoids clamped probability loss.
    Both arms share the same score graph; no additional transformer capture.
    """
    if q.ndim != 4 or q.shape != k.shape or q.shape[1] != height*width:
        raise ValueError('Expected frame, spatial, head, dimension Q/K')
    losses = {a:[] for a in ARMS}
    log_probabilities = {a:[] for a in ARMS}
    masks = {a:torch.as_tensor(refs[a][1], device=q.device) for a in ARMS}
    targets = {a:torch.as_tensor(indices[a], device=q.device) for a in ARMS}
    heads, dim = q.shape[-2:]
    with torch.autocast(device_type=q.device.type, enabled=False):
        for i in range(q.shape[0]-1):
            logits = (q[i].flatten(1) @ k[i+1].flatten(1).T)*(1/(heads*math.sqrt(dim)))
            logits = logits.float() if q.dtype != torch.float64 else logits
            logp = ((logits-logits.mean(0, keepdim=True))*8).log_softmax(-1)
            for arm in ARMS:
                chosen = logp.gather(1, targets[arm][i, :, None]).squeeze(1)
                losses[arm].append(-chosen[masks[arm][i]].sum())
                log_probabilities[arm].append(chosen.detach().cpu().numpy())
    return ({a:torch.stack(losses[a]).sum()/masks[a].sum() for a in ARMS},
            {a:np.stack(log_probabilities[a]) for a in ARMS})


def prepare_destination_inputs(pilot, attribution, huber, gradients, response):
    checked = prepare_response_inputs(pilot, attribution, huber, gradients)
    inputs, _, before_readout, gradient_hashes = checked
    refs = inputs[2]
    indices = {a:destination_indices(*refs[a]) for a in ARMS}
    root = Path(response)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'response_report.json').read_text())
    if protocol['inputs_sha256'] != inputs[4] or protocol['gradient_artifacts_sha256'] != gradient_hashes:
        raise ValueError('Response control used different saved inputs')
    if not report['baseline_unchanged'] or report['counts'] != dict(
            positive_forwards_attempted=4, positive_forwards_completed=4):
        raise ValueError('Requires completed four-capture response control')
    sources = {**protocol['helper_sources_sha256'], 'wan_centered_response.py':protocol['script_sha256']}
    for name, expected in sources.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Response control source changed: {name}')
    paths = [root/'started.json', root/'response_report.json']
    control = {}
    for arm in ARMS:
        name = arm+'_full'
        path = root/name/'flow.npz'
        with np.load(path, allow_pickle=False) as values:
            field = values['flow'].copy()
        if field.shape != inputs[5]['before'].shape or not np.isfinite(field).all():
            raise ValueError('Invalid saved full-dose control flow')
        measured = scores(field, *refs[arm])
        expected = report['conditions'][name]['losses'][arm]['after']
        if any(not np.isclose(measured[key], expected[key], rtol=1e-9, atol=1e-9) for key in measured):
            raise ValueError('Saved control flow and report disagree')
        paths.append(path)
        control[arm] = field
    common = refs['forward'][1] & refs['reverse'][1]
    geometry = dict(meaning='Integer and in-bounds only; not physical correspondence validation',
        arms={a:dict(valid_count=int(refs[a][1].sum()),
                     per_pair_valid=refs[a][1].sum(1).tolist(),
                     torso_valid=int((refs[a][1] & regions()['torso']).sum())) for a in ARMS},
        common_torso_count=int((common & regions()['torso']).sum()),
        common_torso_different_destinations=int((common & regions()['torso'] &
                                                (indices['forward'] != indices['reverse'])).sum()))
    return inputs, before_readout, indices, control, report, {
        'gradient_artifacts':gradient_hashes,
        'response_artifacts':{str(p.relative_to(root)):digest(p) for p in paths}}, geometry


def summarize_target_probability(logp, refs, before_readout):
    result = {}
    common = refs['forward'][1] & refs['reverse'][1] & regions()['torso']
    for a, (_, valid) in refs.items():
        result[a] = {}
        groups = dict(all_valid=valid, own_torso=valid & regions()['torso'],
                      common_torso=common, before_sharp_torso=valid & regions()['torso'] &
                      (before_readout['same_probability'] > .99))
        for name, mask in groups.items():
            values = logp[a][mask].astype(np.float64)
            result[a][name] = dict(count=int(mask.sum()),
                nll_mean=float(-values.mean()) if values.size else None,
                probability_mean=float(np.exp(values).mean()) if values.size else None,
                probability_median=float(np.median(np.exp(values))) if values.size else None)
    return result


def run_destination(g, pilot, attribution, huber, gradients, response, output):
    from benchmark.wan_centered_pilot import clear
    from benchmark.wan_port_acceptance import difference
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Destination budget already started; preserve existing artifacts')
    inputs, before_readout, indices, controls, control_report, provenance, geometry = (
        prepare_destination_inputs(pilot, attribution, huber, gradients, response))
    manifest, latents, refs, _, hashes, old = inputs
    live = frozen_model(g, manifest)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(
        hypothesis='Destination NLL can correct confident wrong matches whose expected-coordinate loss has weak derivatives.',
        expected='Improved target-aligned separation and own-target errors under the unchanged AMF readout, including initially sharp torso queries.',
        changed_factor='Coordinate Huber delta=1 -> mean negative log probability at the saved hard destination',
        max_positive_forwards=3, max_latent_backwards=2, max_first_adam_proposals=2,
        max_scheduler_steps=0, max_decodes=0,
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], block=20, heads='all 40',
            multiplier=8, lr=.001, epsilon=1e-8, masks='original unchanged', injection=False),
        inputs_sha256=hashes, control_sha256=provenance, target_geometry=geometry, live=live,
        script_sha256=digest(__file__), helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('wan_centered_response.py','wan_centered_gradients.py','wan_centered_huber.py',
             'wan_centered_attribution.py','wan_centered_pilot.py')},
        limits='Three explicit feature forwards; checkpoint blocks also recompute during backwards. '
            'NLL changes gradient scale as well as direction; report gradients and actual Adam/cast deltas. '
            'Targets are geometrically checked, not proven physical tracks. No new native parity or decoded-motion claim.')
    with (output/'started.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: 3 positive captures, 2 backwards, 2 first-Adam proposals; no sampling or videos.', flush=True)
    progress = dict(positive_forwards_attempted=0, positive_forwards_completed=0,
                    latent_backwards_attempted=0, latent_backwards_completed=0, first_adam_proposals=0)
    all_gradients, deltas, cast_deltas, after_fields = {}, {}, {}, {}
    common = refs['forward'][1] & refs['reverse'][1]
    report = dict(port_success=False, status='Destination-loss response diagnostic; no decoded-motion claim', arms={})

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
            losses, before_logp = destination_objectives(q, k, refs, indices)
            np.savez_compressed(output/'before_target_log_probability.npz', **before_logp)
            report['before_target_probability'] = summarize_target_probability(before_logp, refs, before_readout)
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
                _, logp = destination_objectives(q, k, refs, indices)
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
                    original_objectives={a:dict(before=scores(before, *refs[a]), after=scores(after, *refs[a])) for a in ARMS},
                    torso=torso_response(before, after, *refs[arm], common, before_readout, readout, arm),
                    torso_scores_by_pair=torso_scores_by_pair(after, *refs[arm], common),
                    dominated_patch_flow=after[2, 12*52+33].tolist(),
                    huber_control=control_report['conditions'][arm+'_full'])
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
            huber_control_separation=separation(controls['forward'], controls['reverse'], refs),
            huber_control_separation_excluding_shared_patch=separation(controls['forward'], controls['reverse'], excluded),
            latent_gradient_comparison=compare(all_gradients['forward'], all_gradients['reverse']),
            fp32_update_comparison=compare(deltas['forward'], deltas['reverse']),
            cast_update_comparison=compare(cast_deltas['forward'], cast_deltas['reverse']),
            counts=progress, baseline_unchanged=True,
            interpretation='Lower NLL alone is insufficient. Compare both arms own-target old AMF errors, '
                'fixed initially-sharp torso queries, common-support separation and per-pair direction, '
                'with and without the shared outlier. No automatic video run or motion-success gate.')
        write_json(output/'destination_report.json', report)
        return report
    except Exception as error:
        write_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), counts=progress))
        raise
    finally:
        clear(g)
