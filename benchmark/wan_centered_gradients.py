"""One frozen sampling-state capture, two Huber backward evaluations, no updates."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from benchmark.wan_centered_attribution import digest, regions
from benchmark.wan_centered_huber import prepare_inputs, frozen_model, huber_loss, scores
from guidance_utils.wan_centered_amf import centered_pair_flow


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def compare(a, b):
    """Raw gradient comparisons; half-difference and half-sum use equal scaling."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    aa, bb, ab = float((a*a).sum()), float((b*b).sum()), float((a*b).sum())
    diff, shared = float(((a-b)**2).sum()), float(((a+b)**2).sum())
    return dict(forward_norm=math.sqrt(aa), reverse_norm=math.sqrt(bb),
        cosine=ab/math.sqrt(aa*bb) if aa*bb else None,
        difference_norm=math.sqrt(diff), shared_mean_norm=.5*math.sqrt(shared),
        differential_to_shared_ratio=math.sqrt(diff/shared) if shared else None,
        formula='norm((g_forward-g_reverse)/2) / norm((g_forward+g_reverse)/2)')


def quantiles(values):
    values = np.asarray(values, np.float64)
    return dict(count=int(values.size), quantiles_0_25_50_75_100=np.quantile(values, [0,.25,.5,.75,1]).tolist()) if values.size else dict(count=0, quantiles_0_25_50_75_100=None)


def pair_derivatives(logits, height, width, targets, valid, counts, multiplier=8.):
    """Analytic derivatives at FP32 pre-centering scores, no autograd traversal.

    For F_i = sum_j p_ij d_j - d_i, dF_i/dC_ij = m p_ij (d_j-E_i).
    C_ij = S_ij - mean_r S_rj. A full objective's dL/dS is dL/dC
    minus its mean over ALL source rows. Individual F_i Jacobian norm gains
    sqrt(1-1/N) from that centering map. No region mask truncates centering.
    """
    n = height*width
    if logits.shape != (n, n) or n < 2:
        raise ValueError('Expected full square spatial score matrix')
    with torch.no_grad():
        p = ((logits-logits.mean(0, keepdim=True))*multiplier).softmax(-1)
        yy, xx = torch.meshgrid(torch.arange(height, device=p.device), torch.arange(width, device=p.device), indexing='ij')
        coords = torch.stack((xx.flatten(), yy.flatten()), -1).to(p.dtype)
        expected = p@coords
        flow = expected-coords
        # N x N x 2, bounded to one frame pair at a time.
        jac = multiplier*p[..., None]*(coords[None]-expected[:, None])
        sensitivity = jac.square().sum((1, 2)).sqrt()*math.sqrt(1-1/n)
        entropy = -(p*p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)
        fields = dict(flow=flow, same_probability=p.diagonal(), max_probability=p.max(-1).values,
            entropy_nats=entropy, flow_jacobian_frobenius=sensitivity)
        grads = {}
        for arm in ('forward', 'reverse'):
            mask = valid[arm]
            if counts[arm] <= 0 or mask.dtype != torch.bool:
                raise ValueError('Expected nonempty global boolean support')
            residual = flow-targets[arm]
            dflow = residual.clamp(-1, 1)*mask[:, None]/(2*counts[arm])
            dcentered = (jac*dflow[:, None]).sum(-1)
            draw = dcentered-dcentered.mean(0, keepdim=True)
            grads[arm] = dict(flow=dflow, centered_scores=dcentered, raw_scores=draw)
            fields[arm+'_residual_norm'] = residual.norm(dim=-1)
            # A source-indexed contribution before centering; NOT a spatial latent attribution.
            fields[arm+'_centered_score_gradient_norm'] = dcentered.norm(dim=-1)
        return fields, grads


def capture(g, x):
    from benchmark.wan_centered_pilot import clear
    clear(g)
    g._set_kv_mode([20], inject=False, copy=True)
    g._forward_transformer(x, g.guidance_embeds[1:2], g.timesteps[39].expand(x.shape[0]))
    processor = g.transformer.blocks[20].attn1.processor
    if processor.query.shape != (1, 9360, 40, 128) or processor.key.shape != processor.query.shape:
        raise ValueError('Expected native normalized post-RoPE Q/K: B,S,H,D')
    q, k = [v[0].reshape(6, 1560, 40, 128) for v in (processor.query, processor.key)]
    flow = torch.stack([centered_pair_flow(q[i], k[i+1], 30, 52) for i in range(5)])
    # Keep copy flags enabled for checkpoint recomputation in both backwards.
    g._clear_kv([20])
    return flow, q, k


def readout_analysis(q, k, refs, output, height=30, width=52):
    """Detached matrix arithmetic on captured Q/K; no further model forward/backward."""
    common = refs['forward'][1] & refs['reverse'][1]
    arrays, comparisons = {}, []
    device = q.device
    counts = {a:int(m.sum()) for a,(_,m) in refs.items()}
    for i in range(q.shape[0]-1):
        with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
            heads, dim = q.shape[-2:]
            # Exactly the frozen kernel's BF16 matmul/scale then FP32 conversion.
            # Match multiplication order, not division, for BF16 replay.
            logits = (q[i].detach().flatten(1)@k[i+1].detach().flatten(1).T)*(1/(heads*math.sqrt(dim)))
            logits = logits.float() if q.dtype != torch.float64 else logits
            targets = {a:torch.as_tensor(f[i], device=device, dtype=logits.dtype) for a,(f,_) in refs.items()}
            masks = {a:torch.as_tensor(m[i], device=device) for a,(_,m) in refs.items()}
            fields, grads = pair_derivatives(logits, height, width, targets, masks, counts)
            for name, value in fields.items():
                arrays.setdefault(name, []).append(value.cpu().numpy())
            row = dict(pair=[i,i+1], original_masks={stage:compare(grads['forward'][stage].cpu().numpy(), grads['reverse'][stage].cpu().numpy())
                for stage in ('flow', 'centered_scores', 'raw_scores')})
            del grads
            # Separate analytical control, NOT another optimized objective/backward.
            if common.any():
                cmask = torch.as_tensor(common[i], device=device)
                _, same = pair_derivatives(logits, height, width, targets,
                    {a:cmask for a in refs}, {a:int(common.sum()) for a in refs})
                row['common_mask_control'] = {stage:compare(same['forward'][stage].cpu().numpy(), same['reverse'][stage].cpu().numpy())
                    for stage in ('flow', 'centered_scores', 'raw_scores')}
                del same
            comparisons.append(row)
    arrays = {name:np.stack(values) for name,values in arrays.items()}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise RuntimeError('Nonfinite analytic readout diagnostic')
    arrays.update({a+'_valid':m for a,(_,m) in refs.items()})
    arrays['common_valid'] = common
    np.savez_compressed(Path(output)/'readout_derivatives.npz', **arrays)
    stratified = {}
    torso = regions(height, width)['torso']
    for arm, (_,valid) in refs.items():
        support = valid & torso
        strata = {}
        for name, mask in (
            ('all_valid_torso', support),
            ('same_probability_above_099', support & (arrays['same_probability'] > .99)),
            ('same_probability_at_most_099', support & (arrays['same_probability'] <= .99))):
            # Report residuals rather than labeling concentration alone a failure.
            strata[name] = {key:quantiles(arrays[key][mask]) for key in
                ('flow_jacobian_frobenius', arm+'_residual_norm', arm+'_centered_score_gradient_norm', 'same_probability')}
        stratified[arm] = strata
    return dict(per_pair_gradient_comparisons=comparisons, torso_strata=stratified,
        interpretation='Analytic derivatives at rounded FP32 pre-centering score values; actual mixed-precision latent backwards are separate. '
        '0.99 only defines a descriptive probability bin, not a pass/fail threshold. '
        'Common-mask control changes only analytical support/normalization; actual backwards use original full masks.'), arrays['flow']


def prepare_gradient_inputs(pilot, attribution, huber):
    inputs = prepare_inputs(pilot, attribution)
    huber = Path(huber)
    started = json.loads((huber/'started.json').read_text())
    report = json.loads((huber/'huber_report.json').read_text())
    if started['inputs_sha256'] != inputs[4]:
        raise ValueError('Huber used different saved pilot inputs')
    if started['archived_attribution_sha256'] != digest(Path(attribution)/'attribution.json'):
        raise ValueError('Archived attribution changed since Huber')
    for name, expected in started['archived_fields_sha256'].items():
        if digest(Path(attribution)/f'{name}_flow.npz') != expected:
            raise ValueError('Archived field changed since Huber')
    for arm in ('forward', 'reverse'):
        if len(report['arms'][arm]['history']) != 5:
            raise ValueError('Requires completed five-update Huber experiment')
        with np.load(huber/arm/'flow_state_00.npz', allow_pickle=False) as values:
            np.testing.assert_array_equal(values['flow'], inputs[5]['before'])
    return inputs, report


def run_gradients(g, pilot, attribution, huber, output):
    from benchmark.wan_centered_pilot import clear
    output = Path(output)
    # Stop before reading large inputs if any previous attempt reserved this budget.
    if (output/'started.json').exists():
        raise RuntimeError('Gradient diagnostic budget already started; preserve existing results')
    inputs, huber_report = prepare_gradient_inputs(pilot, attribution, huber)
    manifest, latents, refs, traces, hashes, old = inputs
    live = frozen_model(g, manifest)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='Concentrated centered-T8 expectations weaken useful reference-dependent derivatives.',
        expected='Locate weak reference distinction at flow-loss, centered/raw score, or latent gradient stages.',
        max_positive_forwards=1, max_latent_backwards=2, max_optimizer_steps=0, max_scheduler_steps=0, max_decodes=0,
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], block=20, heads='all 40',
            multiplier=8, loss='coordinatewise Huber delta=1', masks='original full masks', injection=False),
        inputs_sha256=hashes, huber_report_sha256=digest(Path(huber)/'huber_report.json'),
        script_sha256=digest(__file__),
        helper_sources_sha256={name:digest(Path(__file__).with_name(name)) for name in
            ('wan_centered_huber.py', 'wan_centered_attribution.py', 'wan_centered_pilot.py')}, live=live,
        caveats='One explicit capture; checkpoint blocks may recompute during each backward. No optimizer or solver intervention. '
        'Readout analysis uses detached arithmetic, not additional autograd backward traversals.')
    with (output/'started.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: 1 positive capture, 2 latent backwards, 0 updates/steps/videos.', flush=True)
    progress = dict(positive_forwards_attempted=0, latent_backwards_attempted=0, latent_backwards_completed=0)
    gradients = {}
    try:
        x = torch.from_numpy(latents['before']).to(g.device).clone().requires_grad_(True)
        progress['positive_forwards_attempted'] = 1
        write_json(output/'progress.json', progress)
        flow, q, k = capture(g, x)
        before = flow.detach().float().cpu().numpy()
        if not np.isfinite(before).all():
            raise RuntimeError('Nonfinite captured flow')
        # Same capture is both diagnostic and replay gate; no parity run is repeated.
        replay = dict(flow_max_abs=float(np.max(np.abs(before-old['before']))), arms={})
        for arm in refs:
            measured = scores(before, *refs[arm])
            expected = huber_report['arms'][arm]['readout_states'][0]['scores']
            replay['arms'][arm] = dict(actual=measured, expected=expected)
            if not all(np.isclose(measured[key], expected[key], rtol=.005, atol=.005) for key in measured):
                write_json(output/'replay_failure.json', replay)
                raise RuntimeError('Saved before losses do not replay; no backwards performed')
        # Scalar means can hide changed fields. Enforce the saved readout itself too.
        if not np.allclose(before, old['before'], rtol=.005, atol=.005):
            write_json(output/'replay_failure.json', replay)
            raise RuntimeError('Saved before field does not replay; no backwards performed')
        np.savez_compressed(output/'before_flow.npz', flow=before)
        readout, analytic_flow = readout_analysis(q, k, refs, output)
        readout['captured_flow_max_abs'] = float(np.max(np.abs(before-analytic_flow)))
        if not np.allclose(before, analytic_flow, rtol=0, atol=1e-4):
            raise RuntimeError('Analytic readout differs from captured frozen kernel')
        write_json(output/'readout_analysis.json', readout)
        del q, k
        norms = {}
        for index, arm in enumerate(('forward', 'reverse')):
            target = torch.as_tensor(refs[arm][0], device=g.device, dtype=flow.dtype)
            mask = torch.as_tensor(refs[arm][1], device=g.device)
            loss = huber_loss(flow, target, mask)
            progress['latent_backwards_attempted'] += 1
            write_json(output/'progress.json', progress)
            # No accumulation: each autograd call returns its own gradient at the same x.
            grad, = torch.autograd.grad(loss, x, retain_graph=index == 0)
            if not torch.isfinite(grad).all() or float(grad.norm()) == 0:
                raise RuntimeError('Nonfinite or zero latent gradient')
            gradients[arm] = grad.detach().float().cpu().numpy().copy()
            np.savez_compressed(output/f'{arm}_latent_gradient.npz', gradient=gradients[arm])
            actual = float(grad.norm())
            expected = huber_report['arms'][arm]['history'][0]['gradient_norm']
            norms[arm] = dict(actual=actual, archived_huber=expected,
                close_to_archived=bool(np.isclose(actual, expected, rtol=.005, atol=.005)))
            # Report a discrepancy without silently running another capture.
            progress['latent_backwards_completed'] += 1
            write_json(output/'gradient_norm_replay.json', norms)
            write_json(output/'progress.json', progress)
            print(arm, 'latent gradient norm:', actual, 'archived:', expected, flush=True)
            g._clear_kv([20])
            del grad, loss
        np.testing.assert_array_equal(x.detach().cpu().numpy(), latents['before'])
        per_frame = [dict(latent_frame=i, **compare(gradients['forward'][:,:,i], gradients['reverse'][:,:,i]))
                     for i in range(x.shape[2])]
        report = dict(port_success=False, status='Completed gradient diagnostic; no decoded-motion claim',
            counts=progress, latent_unchanged=True, replay=replay, gradient_norm_replay=norms,
            archived_gradient_norms_match=all(v['close_to_archived'] for v in norms.values()),
            latent_gradient_comparison=compare(gradients['forward'], gradients['reverse']),
            latent_gradient_by_frame=per_frame, readout=readout,
            limits='Global latent gradients use full original masks; they are not torso-only gradients. '
                'Per-frame latent statistics are temporal accounting, not tracked RGB motion. '
                'Sensitivity norms do not establish correspondence correctness or guarantee effective Adam updates.')
        write_json(output/'gradient_report.json', report)
        return report
    except Exception as error:
        write_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), counts=progress))
        raise
    finally:
        clear(g)
