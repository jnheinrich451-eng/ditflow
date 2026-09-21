"""One native capture, two local Q/K-path backwards; no latent graph or updates."""
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_destination import ARMS, destination_objectives
from benchmark.wan_centered_gradients import write_json
from benchmark.wan_centered_huber import frozen_model
from benchmark.wan_centered_qk import prepare_qk_inputs, feature_moments, moments, summarize_moments
from guidance_utils.wan_centered_amf import centered_pair_flow
from guidance_utils.wan_modules import apply_rotary_emb, _as_qk_rope


def prepare_shared_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk):
    ready = prepare_qk_inputs(pilot, attribution, huber, gradients, response, destination, torso)
    root = Path(qk)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'qk_report.json').read_text())
    if protocol['inputs_sha256'] != ready['inputs'][4] or protocol['control_sha256'] != ready['provenance']:
        raise ValueError('Q/K control used different inputs')
    for name, expected in {**protocol['helper_sources_sha256'], 'wan_centered_qk.py':protocol['script_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Q/K control source changed: {name}')
    check_values(report['counts'], dict(positive_forwards_attempted=1, positive_forwards_completed=1,
        readout_backwards_attempted=2, readout_backwards_completed=2))
    if not report['baseline_unchanged'] or report['latent_graph_created'] or report['qk_dtype'] != 'torch.bfloat16':
        raise ValueError('Requires completed native-dtype detached Q/K control')
    check_values(report['archived_latent_comparison'], ready['latent_comparison'])
    np.testing.assert_array_equal(read_npz(root/'before_flow.npz')['flow'], ready['inputs'][5]['before'])
    np.testing.assert_array_equal(read_npz(root/'optimization_support.npz')['common_torso'], ready['support'])
    logs = read_npz(root/'before_target_log_probability.npz')
    for arm in ARMS:
        np.testing.assert_array_equal(logs[arm], ready['before_logp'][arm])
    stats = read_npz(root/'gradient_moments.npz')
    for name in ('Q', 'K'):
        v = stats[name]
        if v.shape != (6, 40, 3) or not np.isfinite(v).all():
            raise ValueError('Invalid Q/K moments')
        check_values(summarize_moments(v), report['stage_comparison'][name])
    paths = ['started.json', 'qk_report.json', 'before_flow.npz', 'optimization_support.npz',
             'before_target_log_probability.npz', 'gradient_moments.npz']
    ready.update(qk_moments={n:stats[n] for n in ('Q', 'K')}, qk_report=report)
    ready['provenance'] = {**ready['provenance'], 'qk_artifacts':{n:digest(root/n) for n in paths}}
    return ready


def local_paths(attn, hidden, rotary_emb, frames, spatial):
    """Equal branch inputs, distinct identity nodes, one shared leaf for their sum.

    The branch nodes expose each contribution in a single autograd call. The shared
    leaf receives the actual native-dtype autograd sum; we do not substitute an FP32 sum.
    """
    if hidden.requires_grad or hidden.shape[0] != 1 or hidden.shape[1] != frames*spatial:
        raise ValueError('Expected one detached positive batch with the frozen token grid')
    if attn.is_cross_attention or getattr(attn, 'fused_projections', False) or attn.add_k_proj is not None:
        raise ValueError('This experiment requires unfused T2V self-attention')
    if any(p.requires_grad for p in attn.parameters()):
        raise ValueError('Attention weights must be frozen')
    shared = hidden.detach().clone().requires_grad_(True)
    q_input, k_input = shared.clone(), shared.clone()
    q = attn.norm_q(attn.to_q(q_input)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(k_input)).unflatten(2, (attn.heads, -1))
    if rotary_emb is not None:
        qr, kr = _as_qk_rope(rotary_emb)
        q, k = apply_rotary_emb(q, *qr), apply_rotary_emb(k, *kr)
    shape = (frames, spatial, attn.heads, q.shape[-1])
    return q.reshape(shape), k.reshape(shape), q_input, k_input, shared


def measure_local(attn, hidden, rotary_emb, native_q, native_k, ready, output, progress,
                  frames=6, height=30, width=52):
    """Runs inside the native processor invocation while offloaded weights are resident."""
    output = Path(output)
    with torch.enable_grad(), torch.autocast(device_type=hidden.device.type, enabled=False):
        q, k, q_input, k_input, shared = local_paths(attn, hidden, rotary_emb, frames, height*width)
        for name, actual, expected in [('Q', q, native_q), ('K', k, native_k)]:
            if not torch.equal(actual.detach().reshape_as(expected), expected):
                raise RuntimeError(f'Local {name} does not exactly replay native capture; no backwards allowed')
        with torch.no_grad():
            before = torch.stack([centered_pair_flow(q[i], k[i+1], height, width) for i in range(frames-1)]).float().cpu().numpy()
        np.testing.assert_array_equal(before, ready['inputs'][5]['before'])
        losses, logs = destination_objectives(q, k, ready['refs'], ready['indices'], height, width)
        for arm in ARMS:
            np.testing.assert_array_equal(logs[arm], ready['before_logp'][arm])
        np.savez_compressed(output/'before_flow.npz', flow=before)
        np.savez_compressed(output/'before_target_log_probability.npz', **logs)
        np.savez_compressed(output/'optimization_support.npz', common_torso=ready['support'])
        replay = dict(native_qk_exact=True, before_flow_exact=True, target_log_probability_exact=True,
                      losses={a:float(losses[a].detach()) for a in ARMS})
        write_json(output/'replay.json', replay)
        first = None
        stage_moments, within_arms, sums = {}, {}, {}
        names = ('Q', 'K', 'input_Q_path', 'input_K_path', 'input_combined')
        for index, arm in enumerate(ARMS):
            progress['local_backwards_attempted'] += 1
            write_json(output/'progress.json', progress)
            values = torch.autograd.grad(losses[arm], (q, k, q_input, k_input, shared), retain_graph=index == 0)
            if not all(torch.isfinite(v).all() for v in values):
                raise RuntimeError('Nonfinite local gradient')
            # The two clone backward nodes add in native dtype, just as a shared input does.
            if not torch.equal(values[2]+values[3], values[4]):
                raise RuntimeError('Branch gradients do not sum to the native-dtype combined gradient')
            sums[arm] = dict(native_sum_exact=True,
                fp32_sum_vs_native_max_abs=float((values[2].float()+values[3].float()-values[4].float()).abs().max()))
            cpu = tuple(v.detach().cpu() for v in values)
            del values
            a, b = [v[0].reshape(frames, height*width, -1) for v in cpu[2:4]]
            within_arms[arm] = np.stack([moments(x.float().numpy(), y.float().numpy(), axes=None) for x, y in zip(a, b)])
            if index == 0:
                first = cpu
            else:
                for name, f, r in zip(names, first, cpu):
                    if name in ('Q', 'K'):
                        stage_moments[name] = feature_moments(f, r)
                        # Same frozen readout as completed Q/K run. No full Q/K archive needed.
                        np.testing.assert_allclose(stage_moments[name], ready['qk_moments'][name], rtol=1e-6, atol=1e-12)
                    else:
                        f, r = [v[0].reshape(frames, height*width, -1) for v in (f, r)]
                        stage_moments[name] = np.stack([moments(x.float().numpy(), y.float().numpy(), axes=None) for x, y in zip(f, r)])
            progress['local_backwards_completed'] += 1
            write_json(output/'progress.json', progress)
            print(arm, 'local backward completed; stopped at detached attention input.', flush=True)
        if shared.grad is not None or any(p.grad is not None for p in attn.parameters()):
            raise RuntimeError('Unexpected accumulated input/weight gradient')
        np.testing.assert_array_equal(shared.detach().cpu().float().numpy(), hidden.detach().cpu().float().numpy())
    np.savez_compressed(output/'gradient_moments.npz', **stage_moments,
                        **{a+'_within_Q_K':v for a, v in within_arms.items()})
    summary = {name:summarize_moments(v) for name, v in stage_moments.items()}
    branch_comparison = {}
    for arm, v in within_arms.items():
        row = summarize_moments(v)
        native_norm = summary['input_combined'][arm+'_norm']
        total_norm = row['forward_norm']+row['reverse_norm']
        branch_comparison[arm] = dict(q_path_norm=row['forward_norm'], k_path_norm=row['reverse_norm'],
            cosine=row['cosine'], sum_norm_before_native_rounding=2*row['shared_mean_norm'],
            native_sum_norm=native_norm, native_sum_to_path_norms_ratio=native_norm/total_norm if total_norm else None)
    return dict(replay=replay, stage_comparison=summary,
        by_frame={n:[dict(frame=i, **summarize_moments(v)) for i, v in enumerate(m)] for n, m in stage_moments.items()},
        within_arm_branch_comparison=branch_comparison,
        branch_sum=sums, shared_input_dtype=str(hidden.dtype), shared_input_shape=list(hidden.shape))


class SharedInputProcessor:
    """Temporary observer; native attention output and processor are preserved."""
    def __init__(self, original, callback):
        self.original, self.callback, self.calls = original, callback, 0

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        self.calls += 1
        if self.calls != 1 or torch.is_grad_enabled() or hidden_states.requires_grad or encoder_hidden_states is not None:
            raise RuntimeError('Expected exactly one no-grad positive self-attention call')
        result = self.original(attn, hidden_states, encoder_hidden_states, attention_mask, rotary_emb)
        self.callback(attn, hidden_states, rotary_emb, self.original.query, self.original.key)
        return result


def run_shared_input(g, pilot, attribution, huber, gradients, response, destination, torso, qk, output):
    from benchmark.wan_centered_pilot import clear
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Shared-input diagnostic budget already started; preserve partial evidence')
    ready = prepare_shared_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk)
    manifest, latents = ready['inputs'][:2]
    live = frozen_model(g, manifest)
    if any(p.requires_grad or p.grad is not None for p in g.transformer.parameters()):
        raise ValueError('Requires frozen model without accumulated parameter gradients')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='Alignment may arise within block 20 Q/K projections and normalization, or when their input gradients combine.',
        expected='Measure Q-path, K-path and combined gradients at the same detached attention input; compare with distinct Q/K and aligned archived latent gradients.',
        max_positive_forwards=1, max_local_backwards=2, max_latent_backwards=0,
        max_optimizer_steps=0, max_scheduler_steps=0, max_decodes=0,
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], block=20, heads='all 40',
            loss='centered T8 destination NLL', support='same 324 common torso queries', injection=False),
        inputs_sha256=ready['inputs'][4], control_sha256=ready['provenance'], live=live,
        script_sha256=digest(__file__), helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('wan_centered_qk.py', 'wan_centered_torso.py', 'wan_centered_destination.py', 'wan_centered_response.py',
             'wan_centered_gradients.py', 'wan_centered_huber.py', 'wan_centered_attribution.py', 'wan_centered_pilot.py',
             'review_wan_centered_response.py')},
        limits='Local Q/K projections and norms are reevaluated once inside the no-grad native forward while weights are resident. '
               'Two backwards stop at a detached shared attention input; none traverse earlier transformer blocks. '
               'Cosines are coordinate dependent. A localization result is not a repair or decoded motion proof.')
    with (output/'started.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: one native forward, one local Q/K reconstruction, two local backwards; no updates or videos.', flush=True)
    progress = dict(positive_forwards_attempted=0, positive_forwards_completed=0,
                    local_backwards_attempted=0, local_backwards_completed=0)
    attn = g.transformer.blocks[20].attn1
    original = attn.processor
    result = {}
    def callback(*args):
        result.update(measure_local(*args, ready, output, progress))
    observer = SharedInputProcessor(original, callback)
    try:
        clear(g)
        g._set_kv_mode([20], inject=False, copy=True)
        attn.set_processor(observer)
        with torch.no_grad():
            x = torch.from_numpy(latents['before']).to(g.device)
            progress['positive_forwards_attempted'] = 1
            write_json(output/'progress.json', progress)
            g._forward_transformer(x, g.guidance_embeds[1:2], g.timesteps[39].expand(x.shape[0]))
            progress['positive_forwards_completed'] = 1
            write_json(output/'progress.json', progress)
            if observer.calls != 1 or x.requires_grad or x.grad is not None:
                raise RuntimeError('Capture count or latent graph violation')
            np.testing.assert_array_equal(x.cpu().numpy(), latents['before'])
        report = dict(port_success=False, status='Shared-input gradient localization only', counts=progress,
            baseline_unchanged=True, latent_graph_created=False, archived_latent_comparison=ready['latent_comparison'],
            archived_qk_comparison=ready['qk_report']['stage_comparison'], **result)
        write_json(output/'shared_input_report.json', report)
        return report
    except Exception as error:
        write_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), counts=progress))
        raise
    finally:
        attn.set_processor(original)
        clear(g)
