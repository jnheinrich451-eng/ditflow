"""One native capture, two local backwards through block 20's pre-attention path."""
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_destination import ARMS, destination_objectives
from benchmark.wan_centered_gradients import write_json
from benchmark.wan_centered_huber import frozen_model
from benchmark.wan_centered_qk import feature_moments, moments, summarize_moments
from benchmark.wan_centered_shared_input import prepare_shared_inputs, SharedInputProcessor
from guidance_utils.wan_centered_amf import centered_pair_flow
from guidance_utils.wan_modules import apply_rotary_emb, _as_qk_rope


STAGES = ('Q', 'K', 'attention_input', 'modulated_fp32', 'normalized_fp32',
          'norm_input_fp32', 'block_input')


def prepare_block_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared):
    ready = prepare_shared_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk)
    root = Path(shared)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'shared_input_report.json').read_text())
    if protocol['inputs_sha256'] != ready['inputs'][4] or protocol['control_sha256'] != ready['provenance']:
        raise ValueError('Shared-input control used different inputs')
    for name, expected in {**protocol['helper_sources_sha256'], 'wan_centered_shared_input.py':protocol['script_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Shared-input control source changed: {name}')
    check_values(report['counts'], dict(positive_forwards_attempted=1, positive_forwards_completed=1,
        local_backwards_attempted=2, local_backwards_completed=2))
    if not report['baseline_unchanged'] or report['latent_graph_created'] or report['port_success']:
        raise ValueError('Requires completed shared-input localization control')
    if report['shared_input_dtype'] != 'torch.bfloat16' or report['shared_input_shape'] != [1, 9360, 5120]:
        raise ValueError('Unexpected shared-input geometry/precision')
    check_values(report['archived_latent_comparison'], ready['latent_comparison'])
    check_values(report['archived_qk_comparison'], ready['qk_report']['stage_comparison'])
    np.testing.assert_array_equal(read_npz(root/'before_flow.npz')['flow'], ready['inputs'][5]['before'])
    np.testing.assert_array_equal(read_npz(root/'optimization_support.npz')['common_torso'], ready['support'])
    for a, v in read_npz(root/'before_target_log_probability.npz').items():
        np.testing.assert_array_equal(v, ready['before_logp'][a])
    stats = read_npz(root/'gradient_moments.npz')
    for name in ('Q', 'K', 'input_combined'):
        v = stats[name]
        if v.shape != ((6, 40, 3) if name in ('Q', 'K') else (6, 3)) or not np.isfinite(v).all():
            raise ValueError('Invalid shared-input gradient moments')
        check_values(summarize_moments(v), report['stage_comparison'][name])
        if name in ('Q', 'K'):
            np.testing.assert_array_equal(v, ready['qk_moments'][name])
    paths = ['started.json', 'shared_input_report.json', 'before_flow.npz', 'optimization_support.npz',
             'before_target_log_probability.npz', 'gradient_moments.npz']
    ready.update(shared_moments=stats['input_combined'], shared_report=report)
    ready['provenance'] = {**ready['provenance'], 'shared_input_artifacts':{n:digest(root/n) for n in paths}}
    return ready


def local_block_path(block, incoming, temb, rotary_emb, frames, spatial):
    """Rebuild Wan2.1's unchanged pre-attention expression on a detached input.

    Earlier transformer blocks and time conditioning are constants. Extra backward
    endpoints expose the existing casts, LayerNorm and time modulation, not new ops.
    """
    attn = block.attn1
    if incoming.requires_grad or incoming.shape[0] != 1 or incoming.shape[1] != frames*spatial:
        raise ValueError('Expected one detached positive block input with the frozen token grid')
    if temb.requires_grad or temb.ndim != 3 or temb.shape[:2] != (1, 6):
        raise ValueError('Expected frozen Wan2.1 timestep modulation (1,6,D)')
    if attn.is_cross_attention or getattr(attn, 'fused_projections', False) or attn.add_k_proj is not None:
        raise ValueError('Requires unfused T2V self-attention')
    if any(p.requires_grad for p in block.parameters()):
        raise ValueError('Block weights must be frozen')
    x = incoming.detach().clone().requires_grad_(True)
    x32 = x.float()
    normalized = block.norm1(x32)
    shift, scale, *_ = (block.scale_shift_table+temb.detach().float()).chunk(6, dim=1)
    modulated = normalized*(1+scale)+shift
    shared = modulated.type_as(x)
    q = attn.norm_q(attn.to_q(shared)).unflatten(2, (attn.heads, -1))
    k = attn.norm_k(attn.to_k(shared)).unflatten(2, (attn.heads, -1))
    if rotary_emb is not None:
        qr, kr = _as_qk_rope(rotary_emb)
        q, k = apply_rotary_emb(q, *qr), apply_rotary_emb(k, *kr)
    shape = (frames, spatial, attn.heads, q.shape[-1])
    return (q.reshape(shape), k.reshape(shape), shared, modulated, normalized, x32, x), scale.detach()


def measure_block(block, incoming, temb, hidden, rotary_emb, native_q, native_k,
                  ready, output, progress, frames=6, height=30, width=52):
    output = Path(output)
    with torch.enable_grad(), torch.autocast(device_type=incoming.device.type, enabled=False):
        endpoints, scale = local_block_path(block, incoming, temb, rotary_emb, frames, height*width)
        q, k, shared, modulated, normalized, x32, x = endpoints
        for name, actual, expected in [('attention input', shared, hidden), ('Q', q, native_q), ('K', k, native_k)]:
            if not torch.equal(actual.detach().reshape_as(expected), expected):
                raise RuntimeError(f'Local {name} does not exactly replay native capture; no backwards allowed')
        with torch.no_grad():
            before = torch.stack([centered_pair_flow(q[i], k[i+1], height, width) for i in range(frames-1)]).float().cpu().numpy()
        np.testing.assert_array_equal(before, ready['inputs'][5]['before'])
        losses, logs = destination_objectives(q, k, ready['refs'], ready['indices'], height, width)
        for a in ARMS:
            np.testing.assert_array_equal(logs[a], ready['before_logp'][a])
        np.savez_compressed(output/'before_flow.npz', flow=before)
        np.savez_compressed(output/'before_target_log_probability.npz', **logs)
        np.savez_compressed(output/'optimization_support.npz', common_torso=ready['support'])
        replay = dict(attention_input_exact=True, native_qk_exact=True, before_flow_exact=True,
                      target_log_probability_exact=True, losses={a:float(losses[a].detach()) for a in ARMS})
        write_json(output/'replay.json', replay)
        first, stage_moments, path_checks = None, {}, {}
        for index, arm in enumerate(ARMS):
            progress['local_backwards_attempted'] += 1
            write_json(output/'progress.json', progress)
            grads = torch.autograd.grad(losses[arm], endpoints, retain_graph=index == 0)
            if not all(torch.isfinite(v).all() for v in grads):
                raise RuntimeError('Nonfinite local block gradient')
            # Isolate the two casts and fixed affine map in the same backwards.
            checks = dict(attention_cast_exact=torch.equal(grads[3], grads[2].to(modulated.dtype)),
                modulation_exact=torch.equal(grads[4], (grads[3]*(1+scale)).to(normalized.dtype)),
                input_cast_exact=torch.equal(grads[6], grads[5].to(x.dtype)))
            if not all(checks.values()):
                raise RuntimeError(f'Local path derivative identity failed: {checks}')
            path_checks[arm] = checks
            cpu = tuple(v.detach().cpu() for v in grads)
            del grads
            if index == 0:
                first = cpu
            else:
                for name, f, r in zip(STAGES, first, cpu):
                    if name in ('Q', 'K'):
                        stage_moments[name] = feature_moments(f, r)
                        np.testing.assert_allclose(stage_moments[name], ready['qk_moments'][name], rtol=1e-6, atol=1e-12)
                    else:
                        f, r = [v[0].reshape(frames, height*width, -1) for v in (f, r)]
                        stage_moments[name] = np.stack([moments(a.float().numpy(), b.float().numpy(), axes=None) for a, b in zip(f, r)])
                        if name == 'attention_input':
                            np.testing.assert_allclose(stage_moments[name], ready['shared_moments'], rtol=1e-6, atol=1e-12)
            progress['local_backwards_completed'] += 1
            write_json(output/'progress.json', progress)
            print(arm, 'local backward completed; stopped at detached block 20 input.', flush=True)
        if x.grad is not None or any(p.grad is not None for p in block.parameters()):
            raise RuntimeError('Unexpected accumulated input/weight gradient')
        if not torch.equal(x.detach(), incoming):
            raise RuntimeError('Block input changed')
    np.savez_compressed(output/'gradient_moments.npz', **stage_moments)
    return dict(replay=replay, stage_comparison={n:summarize_moments(v) for n, v in stage_moments.items()},
        by_frame={n:[dict(frame=i, **summarize_moments(v)) for i, v in enumerate(m)] for n, m in stage_moments.items()},
        derivative_identity_checks=path_checks, endpoint_dtypes={n:str(v.dtype) for n, v in zip(STAGES, endpoints)},
        block_input_shape=list(incoming.shape))


def run_block_input(g, pilot, attribution, huber, gradients, response, destination, torso, qk, shared, output):
    from benchmark.wan_centered_pilot import clear
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Block-input diagnostic budget already started; preserve partial evidence')
    ready = prepare_block_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared)
    manifest, latents = ready['inputs'][:2]
    live = frozen_model(g, manifest)
    if len(g.transformer.blocks) != 40:
        raise ValueError('Expected pinned 14B model with 40 blocks')
    if any(p.requires_grad or p.grad is not None for p in g.transformer.parameters()):
        raise ValueError('Requires frozen model without accumulated parameter gradients')
    wrapper = g.transformer.blocks[20]
    block = getattr(wrapper, 'module', wrapper)
    attn, original = block.attn1, block.attn1.processor
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='Block 20 pre-attention normalization/time modulation may already align the reference gradients.',
        expected='Replay shared-input gradients; measure through casts, modulation and LayerNorm to incoming block features. '
                 'Alignment here locates a local operation; distinct block-input gradients place the change earlier.',
        max_positive_forwards=1, max_local_backwards=2, max_latent_backwards=0,
        max_optimizer_steps=0, max_scheduler_steps=0, max_decodes=0,
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], block=20, model_blocks=40,
                   heads='all 40', loss='centered T8 destination NLL', support='same 324 common torso queries', injection=False),
        inputs_sha256=ready['inputs'][4], control_sha256=ready['provenance'], live=live,
        script_sha256=digest(__file__), helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('wan_centered_shared_input.py', 'wan_centered_qk.py', 'wan_centered_torso.py', 'wan_centered_destination.py',
             'wan_centered_response.py', 'wan_centered_gradients.py', 'wan_centered_huber.py',
             'wan_centered_attribution.py', 'wan_centered_pilot.py', 'review_wan_centered_response.py')},
        native_block_class=type(block).__module__+'.'+type(block).__name__,
        native_block_forward_sha256=hashlib.sha256(inspect.getsource(type(block).forward).encode()).hexdigest(),
        limits='One no-grad native capture; one local pre-attention reconstruction; two local backwards ending at detached block input. '
               'No earlier-block or latent backward, updates or generation. Extra endpoints are logging, not extra objectives. '
               'Cosines are coordinate dependent; this is localization, not motion proof.')
    with (output/'started.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: one native forward, one local reconstruction, two local backwards; no updates or videos.', flush=True)
    progress = dict(positive_forwards_attempted=0, positive_forwards_completed=0,
                    local_backwards_attempted=0, local_backwards_completed=0)
    state, result, hook = {}, {}, None
    def capture_incoming(module, args, kwargs):
        if state or torch.is_grad_enabled():
            raise RuntimeError('Expected exactly one no-grad block-input capture')
        state['incoming'] = args[0] if args else kwargs['hidden_states']
        state['temb'] = args[2] if len(args) > 2 else kwargs['temb']
    def callback(local_attn, hidden, rope, native_q, native_k):
        if local_attn is not attn or not state:
            raise RuntimeError('Attention/block capture mismatch')
        result.update(measure_block(block, state['incoming'], state['temb'], hidden, rope, native_q, native_k,
                                    ready, output, progress))
    observer = SharedInputProcessor(original, callback)
    try:
        clear(g)
        g._set_kv_mode([20], inject=False, copy=True)
        hook = block.register_forward_pre_hook(capture_incoming, with_kwargs=True)
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
        report = dict(port_success=False, status='Block 20 incoming-feature gradient localization only', counts=progress,
            baseline_unchanged=True, latent_graph_created=False, archived_latent_comparison=ready['latent_comparison'],
            archived_shared_comparison=ready['shared_report']['stage_comparison'], **result)
        write_json(output/'block_input_report.json', report)
        return report
    except Exception as error:
        write_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), counts=progress))
        raise
    finally:
        if hook is not None:
            hook.remove()
        attn.set_processor(original)
        state.clear()
        clear(g)
