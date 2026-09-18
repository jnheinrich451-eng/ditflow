"""Frozen MSE-to-Huber ablation: two saved-state branches, no sampling."""
import json
from pathlib import Path

import numpy as np

from benchmark.wan_centered_attribution import (digest, load_inputs, decompose, regions, response_to_target)


def scores(flow, target, valid):
    residual = np.asarray(flow, np.float64)[valid]-np.asarray(target, np.float64)[valid]
    if not residual.size or not np.isfinite(residual).all():
        raise ValueError('Scores require finite residuals on nonempty support')
    absolute = np.abs(residual)
    return dict(mse=float((residual**2).mean()),
                huber_delta1=float(np.where(absolute <= 1, .5*residual**2, absolute-.5).mean()))


def huber_loss(flow, target, valid):
    import torch.nn.functional as F
    if not bool(valid.any()):
        raise ValueError('No valid targets')
    return F.huber_loss(flow[valid], target[valid], delta=1., reduction='mean')


def torso_scores_by_pair(flow, target, valid, common):
    """Own-target errors alongside separation, with explicit comparable coverage."""
    torso = regions()['torso']
    result = []
    for i in range(5):
        row = dict(pair=[i, i+1])
        for name, support in (('own_valid', valid[i] & torso),
                              ('common_valid', common[i] & torso)):
            row[name] = dict(count=int(support.sum()),
                scores=scores(flow[i], target[i], support) if support.any() else None)
        result.append(row)
    return result


def probability_summary(query, key, height, width, indices):
    """Detached logging of the SAME centered-T8 formula; never feeds optimization.

    Recomputes pair logits from captured Q/K (not another transformer forward).
    All source positions contribute to centering, including unselected queries.
    """
    import math
    import torch
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        q, k = query.detach(), key.detach()
        heads, dim = q.shape[-2:]
        logits = (q.flatten(1) @ k.flatten(1).T)*(1/(heads*math.sqrt(dim)))
        logits = logits.float() if q.dtype != torch.float64 else logits
        selected = torch.as_tensor(indices, dtype=torch.long, device=q.device)
        p = ((logits[selected]-logits.mean(dim=0, keepdim=True))*8).softmax(dim=-1)
        entropy = -(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(dim=-1)
        maximum, argmax = p.max(dim=-1)
        same = p[torch.arange(len(indices), device=q.device), selected]
        yy, xx = torch.meshgrid(torch.arange(height, device=q.device), torch.arange(width, device=q.device), indexing='ij')
        coords = torch.stack((xx.flatten(), yy.flatten()), dim=-1).to(p.dtype)
        expected = p@coords-coords[selected]
        return [dict(row=int(idx//width), column=int(idx%width), entropy_nats=float(entropy[j]),
                     max_probability=float(maximum[j]), same_spatial_probability=float(same[j]),
                     argmax_destination_index=int(argmax[j]), expected_flow_xy=expected[j].cpu().tolist())
                for j,idx in enumerate(indices)]


def improvement_concentration(before, after, target, valid):
    """Account for reductions; does not estimate latent-gradient influence."""
    before, after, target = [np.asarray(a, np.float64) for a in (before, after, target)]
    original_error = ((before-target)**2).mean(axis=-1)
    improvement = original_error-((after-target)**2).mean(axis=-1)
    result = {}
    for name, selected in (('all_valid', valid), ('torso', valid & regions()['torso'])):
        gains, original = improvement[selected], original_error[selected]
        total = float(gains.sum())
        ranks = np.argsort(original)[::-1]  # Fixed by BEFORE error, not each new outcome.
        result[name] = dict(net_mse_decrease_sum=total,
            largest_improvement_share_of_net=float(gains.max()/total) if total > 0 else None,
            baseline_highest_error_patch_share_of_net=float(gains[ranks[:1]].sum()/total) if total > 0 else None,
            baseline_top10_error_patches_share_of_net=float(gains[ranks[:10]].sum()/total) if total > 0 else None)
    return result


def separation(forward, reverse, refs):
    mask = refs['forward'][1] & refs['reverse'][1] & regions()['torso']
    delta = forward.astype(np.float64)-reverse
    target_delta = refs['forward'][0].astype(np.float64)-refs['reverse'][0]
    def measure(m):
        a, b = delta[m], target_delta[m]
        denominator = float(np.linalg.norm(a)*np.linalg.norm(b))
        energy = float((b*b).sum())
        return dict(common_count=int(m.sum()),
            cosine=float((a*b).sum()/denominator) if denominator else None,
            projection_gain=float((a*b).sum()/energy) if energy else None,
            observed_rms=float(np.sqrt((a*a).mean())) if m.any() else None,
            target_rms=float(np.sqrt((b*b).mean())) if m.any() else None)
    pairs = []
    for i in range(5):
        m = np.zeros_like(mask); m[i] = mask[i]
        pairs.append(dict(pair=[i, i+1], **measure(m)))
    return dict(all_common=measure(mask), pairs=pairs)


def prepare_inputs(pilot, attribution):
    pilot, attribution = Path(pilot), Path(attribution)
    manifest, latents, refs, traces, hashes = load_inputs(pilot)
    protocol = json.loads((attribution/'started.json').read_text())
    if hashes != protocol['inputs_sha256']:
        raise ValueError('Pilot artifacts differ from the completed attribution inputs')
    report = json.loads((attribution/'attribution.json').read_text())
    if not report['loss_replay_passed']:
        raise ValueError('Requires successful completed loss replay')
    fields = {}
    for name in ('before', 'forward', 'reverse'):
        with np.load(attribution/f'{name}_flow.npz', allow_pickle=False) as data:
            fields[name] = data['flow'].copy()
        if fields[name].shape != (5, 1560, 2) or not np.isfinite(fields[name]).all():
            raise ValueError('Invalid archived MSE flow')
    return manifest, latents, refs, traces, hashes, fields


def frozen_model(g, manifest):
    from benchmark.wan_runtime_preflight import verify_runtime
    from benchmark.wan_source_fingerprints import verify_parity_sources
    from probe_wan_response import tensor_hash
    baseline = manifest['baseline']
    root = Path(__file__).resolve().parents[1]
    runtime = verify_runtime({**baseline['packages'], 'tokenizers': '0.22.2'})
    sources = verify_parity_sources(root, baseline['source_sha256'])
    if digest(root/'guidance_utils/wan_centered_amf.py') != manifest['kernel_sha256']:
        raise ValueError('Centered kernel changed')
    if baseline['checkpoint_revision'] not in str(g.config.model_key):
        raise ValueError('Wrong checkpoint snapshot path')
    checks = dict(initial_latent_sha256=tensor_hash(g.init_latents),
                  conditioning_sha256=tensor_hash(g.guidance_embeds),
                  source_conditioning_sha256=tensor_hash(g.source_embeds),
                  rope_sha256=tensor_hash(g.transformer.init_rope))
    if checks != manifest['checks'] or list(g.config.guidance_blocks) != [20]:
        raise ValueError('Live model conditioning/noise/RoPE/blocks differ')
    if g.timesteps.cpu().tolist() != baseline['timesteps'] or g.scheduler.sigmas.tolist() != baseline['sigmas']:
        raise ValueError('Schedule differs')
    dtypes = {name:str(p.dtype) for name,p in g.transformer.named_parameters()
              if 'time_embedder' in name or name == 'patch_embedding.weight'}
    if dtypes != manifest['experiment']['model_parameter_dtypes']:
        raise ValueError('Model precision differs')
    return dict(runtime=runtime, sources=sources, checks=checks)


def target_flow(g, x, log_probability=False):
    """Same differentiable centered readout as the completed pilot; only its loss changes."""
    import torch
    from benchmark.wan_centered_pilot import clear
    from guidance_utils.wan_centered_amf import centered_pair_flow
    clear(g)
    g._set_kv_mode([20], inject=False, copy=True)
    g._forward_transformer(x, g.guidance_embeds[1:2], g.timesteps[39].expand(x.shape[0]))
    processor = g.transformer.blocks[20].attn1.processor
    if processor.query.shape != (1, 9360, 40, 128) or processor.key.shape != processor.query.shape:
        raise ValueError('Unexpected Q/K layout')
    q, k = [v[0].reshape(6, 1560, 40, 128) for v in (processor.query, processor.key)]
    flows, probabilities = [], []
    for i in range(5):
        flows.append(centered_pair_flow(q[i], k[i+1], 30, 52))
        if log_probability:
            indices = [10*52+24, 10*52+32, 14*52+24, 14*52+32]
            if i == 2:
                indices.append(12*52+33)  # Previously observed dominant patch, disclosed explicitly.
            rows = probability_summary(q[i], k[i+1], 30, 52, indices)
            for row, index in zip(rows, indices):
                row['pair'] = [i, i+1]
                row['selected_because'] = 'previous dominant patch' if i == 2 and index == 12*52+33 else 'fixed torso query'
                row['readout_check_max_abs'] = float((flows[-1][index].detach()-
                    torch.tensor(row['expected_flow_xy'], device=x.device)).abs().max())
            probabilities.extend(rows)
    flow = torch.stack(flows)
    g._clear_kv([20])
    return (flow, probabilities) if log_probability else flow


def run_huber(g, pilot, attribution, output):
    import torch
    from benchmark.wan_centered_pilot import clear
    from benchmark.wan_port_acceptance import difference
    pilot, attribution, output = Path(pilot), Path(attribution), Path(output)
    manifest, latents, refs, traces, hashes, old = prepare_inputs(pilot, attribution)
    live = frozen_model(g, manifest)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='Limiting large residual influence allows more reference-dependent torso response.',
        expected='Less shared outlier dominance and more coherent per-pair target separation versus archived MSE.',
        changed_factor='MSE -> coordinatewise standard Huber, delta 1 patch',
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], optimizer='Adam', lr=.001,
                   updates_per_arm=5, target_multiplier=8, block=20, heads='all 40', masks='original, unchanged', injection=False),
        max_target_forwards=12, max_backwards=10, max_sampling_steps=0, max_decodes=0,
        logging='Save flow after 0..5 updates, per-update latent/cast changes, and detached centered probability '
                'entropy/max/same-location summaries for four fixed torso queries per pair plus the known dominant patch. '
                'Extra pair-logit calculations only; no additional model forwards or backwards.',
        inputs_sha256=hashes, archived_attribution_sha256=digest(attribution/'attribution.json'),
        archived_fields_sha256={name:digest(attribution/f'{name}_flow.npz') for name in old},
        live=live, script_sha256=digest(__file__))
    with (output/'started.json').open('x') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: two branches, five updates each, two final reads: 12 target forwards, 10 backwards; no sampling.', flush=True)
    report = dict(port_success=False, status='AMF-response diagnostic only; no decoded claim', arms={})
    after_fields = {}
    common = refs['forward'][1] & refs['reverse'][1]
    try:
        for arm in ('forward', 'reverse'):
            folder = output/arm; folder.mkdir()
            ref, valid = refs[arm]
            target = torch.from_numpy(ref).to(g.device, torch.float32)
            mask = torch.from_numpy(valid).to(g.device)
            prefix = torch.from_numpy(latents['before']).to(g.device)
            x = prefix.clone().requires_grad_(True)
            optimizer = torch.optim.Adam([x], lr=.001)
            history, readouts = [], []
            for j in range(5):
                optimizer.zero_grad(set_to_none=True)
                flow, probabilities = target_flow(g, x, log_probability=True)
                if not torch.isfinite(flow).all():
                    raise RuntimeError('Nonfinite flow')
                flow_array = flow.detach().float().cpu().numpy()
                measured = scores(flow_array, ref, valid)
                np.savez_compressed(folder/f'flow_state_{j:02d}.npz', flow=flow_array)
                readouts.append(dict(after_updates=j, scores=measured, probability=probabilities,
                    torso_scores_by_pair=torso_scores_by_pair(flow_array,ref,valid,common),
                    improvement_concentration=improvement_concentration(old['before'],flow_array,ref,valid)))
                (folder/'readout_states.json').write_text(json.dumps(readouts, indent=2)+'\n')
                if j == 0 and not np.isclose(measured['mse'], traces[arm]['losses_before_updates'][0], rtol=.005, atol=.005):
                    (folder/'replay_failure.json').write_text(json.dumps(measured, indent=2))
                    raise RuntimeError('Starting MSE does not replay; no update performed in this branch')
                loss = huber_loss(flow, target, mask)
                loss.backward()
                if x.grad is None or not torch.isfinite(x.grad).all() or float(x.grad.norm()) == 0:
                    raise RuntimeError('Absent, nonfinite or zero latent gradient')
                previous = x.detach().clone()
                entry = dict(update=j+1, loss_evaluated_after_updates=j, **measured, gradient_norm=float(x.grad.norm()))
                optimizer.step()
                if not torch.isfinite(x).all():
                    raise RuntimeError('Nonfinite optimized latent')
                entry['actual_update_fp32'] = difference(x.detach(), previous)
                entry['actual_update_after_cast'] = difference(x.detach().to(g.dtype), previous.to(g.dtype))
                history.append(entry)
                del previous
                g._clear_kv([20])
                del flow, loss
                (folder/'optimization.json').write_text(json.dumps(history, indent=2)+'\n')
                print(arm, j+1, measured, flush=True)
            with torch.no_grad():
                final_flow, probabilities = target_flow(g, x.detach(), log_probability=True)
                if not torch.isfinite(final_flow).all():
                    raise RuntimeError('Nonfinite final flow')
                after = final_flow.float().cpu().numpy()
                del final_flow
            np.savez_compressed(folder/'flow_state_05.npz', flow=after)
            readouts.append(dict(after_updates=5, scores=scores(after,ref,valid), probability=probabilities,
                torso_scores_by_pair=torso_scores_by_pair(after,ref,valid,common),
                improvement_concentration=improvement_concentration(old['before'],after,ref,valid)))
            (folder/'readout_states.json').write_text(json.dumps(readouts, indent=2)+'\n')
            after_fields[arm] = after
            np.savez_compressed(folder/'flow.npz', flow=after)
            torch.save(x.detach().cpu(), folder/'optimized_latent.pt')
            accounted = decompose(old['before'], after, ref, valid, regions())
            errors = ((old['before'].astype(float)-ref)**2-(after.astype(float)-ref)**2).mean(axis=-1)
            torso = valid & regions()['torso']
            total = float(errors[torso].sum())
            result = dict(history=history, readout_states=readouts,
                scores=dict(before=scores(old['before'],ref,valid), archived_mse_after=scores(old[arm],ref,valid),
                            huber_after=scores(after,ref,valid)),
                mse_loss_accounting=accounted,
                common_valid_response=response_to_target(old['before'], after, ref, common, regions()),
                largest_torso_patch_share_of_net_mse_reduction=float(errors[torso].max()/total) if total > 0 else None,
                shared_outlier_flow=after[2, 12*52+33].tolist(),
                update_fp32=difference(x.detach(),prefix),
                update_after_input_cast=difference(x.detach().to(g.dtype),prefix.to(g.dtype)))
            report['arms'][arm] = result
            (folder/'result.json').write_text(json.dumps(result, indent=2)+'\n')
            del x, optimizer, target, mask, prefix
        report['separation'] = dict(archived_mse=separation(old['forward'],old['reverse'],refs),
                                    huber=separation(after_fields['forward'],after_fields['reverse'],refs))
        report['separation_by_updates'] = []
        for j in range(6):
            pair_fields = {}
            for arm in ('forward', 'reverse'):
                with np.load(output/arm/f'flow_state_{j:02d}.npz', allow_pickle=False) as values:
                    pair_fields[arm] = values['flow'].copy()
            report['separation_by_updates'].append(dict(after_updates=j,
                **separation(pair_fields['forward'],pair_fields['reverse'],refs)))
        (output/'huber_report.json').write_text(json.dumps(report, indent=2)+'\n')
    finally:
        clear(g)
    return report
