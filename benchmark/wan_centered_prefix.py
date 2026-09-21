"""Trace one fixed block-20 objective through block inputs 20/15/10/0 to the latent."""
import json
from pathlib import Path

import numpy as np
import torch

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_block_input import prepare_block_inputs
from benchmark.wan_centered_destination import ARMS, destination_objectives
from benchmark.wan_centered_gradients import capture, compare, write_json
from benchmark.wan_centered_huber import frozen_model
from benchmark.wan_centered_qk import moments, summarize_moments


BLOCKS = (0, 10, 15, 20)
ORDER = ('block_20', 'block_15', 'block_10', 'block_0', 'latent')


def prepare_prefix_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input):
    ready = prepare_block_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared)
    root = Path(block_input)
    protocol = json.loads((root/'started.json').read_text())
    report = json.loads((root/'block_input_report.json').read_text())
    if protocol['inputs_sha256'] != ready['inputs'][4] or protocol['control_sha256'] != ready['provenance']:
        raise ValueError('Block-input control used different inputs')
    for name, expected in {**protocol['helper_sources_sha256'], 'wan_centered_block_input.py':protocol['script_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Block-input control source changed: {name}')
    check_values(report['counts'], dict(positive_forwards_attempted=1, positive_forwards_completed=1,
        local_backwards_attempted=2, local_backwards_completed=2))
    if not report['baseline_unchanged'] or report['latent_graph_created'] or report['port_success']:
        raise ValueError('Requires completed local block-input control')
    check_values(report['archived_latent_comparison'], ready['latent_comparison'])
    np.testing.assert_array_equal(read_npz(root/'before_flow.npz')['flow'], ready['inputs'][5]['before'])
    np.testing.assert_array_equal(read_npz(root/'optimization_support.npz')['common_torso'], ready['support'])
    logs = read_npz(root/'before_target_log_probability.npz')
    assert logs.keys() == ready['before_logp'].keys()
    for a, v in logs.items():
        np.testing.assert_array_equal(v, ready['before_logp'][a])
    stats = read_npz(root/'gradient_moments.npz')
    for name, v in stats.items():
        if not np.isfinite(v).all():
            raise ValueError('Nonfinite block-input moments')
        check_values(summarize_moments(v), report['stage_comparison'][name])
    if stats['block_input'].shape != (6, 3):
        raise ValueError('Unexpected block-input moment shape')
    np.testing.assert_array_equal(stats['attention_input'], ready['shared_moments'])
    ready['block_moments'] = stats['block_input']
    ready['block_report'] = report
    ready['latent_gradients'] = {a:read_npz(Path(torso)/f'{a}_latent_gradient.npz')['gradient'] for a in ARMS}
    check_values(compare(*[ready['latent_gradients'][a] for a in ARMS]), ready['latent_comparison'])
    files = ['started.json', 'block_input_report.json', 'before_flow.npz', 'optimization_support.npz',
             'before_target_log_probability.npz', 'gradient_moments.npz']
    ready['provenance'] = {**ready['provenance'], 'block_input_artifacts':{n:digest(root/n) for n in files}}
    return ready


class PrefixObserver:
    """Read-only tensor hooks on original inputs; never register again on recompute."""
    def __init__(self, transformer, output, counts):
        self.output, self.counts = Path(output), counts
        self.phase = 'capture'
        self.entries = {p:{str(i):0 for i in range(40)} for p in ('capture', *ARMS)}
        self.metadata, self.current = {}, {}
        self.module_hooks, self.tensor_hooks = [], []
        for index, block in enumerate(transformer.blocks):
            self.module_hooks.append(block.register_forward_pre_hook(self.before_block(index), with_kwargs=True))

    def save_progress(self):
        write_json(self.output/'progress.json', dict(**self.counts, block_entries=self.entries))

    def before_block(self, index):
        def hook(module, args, kwargs):
            if self.phase not in self.entries:
                raise RuntimeError('Block execution outside the bounded phases')
            row = self.entries[self.phase]
            row[str(index)] += 1
            self.save_progress()
            if index > 20 or row[str(index)] > 1:
                raise RuntimeError(f'Block-entry budget exceeded: {self.phase} block {index}')
            if self.phase == 'capture' and index in BLOCKS:
                x = args[0] if args else kwargs['hidden_states']
                if not torch.is_grad_enabled() or not x.requires_grad:
                    raise RuntimeError('Selected input lacks graph; requires native non-reentrant checkpointing')
                name = f'block_{index}'
                self.metadata[name] = dict(shape=list(x.shape), dtype=str(x.dtype))
                def receive(grad):
                    if self.phase not in ARMS or name in self.current:
                        raise RuntimeError('Missing phase or repeated selected-input gradient')
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Nonfinite selected-input gradient')
                    self.current[name] = grad.detach().cpu()
                    # Returning None preserves the gradient exactly.
                self.tensor_hooks.append(x.register_hook(receive))
        return hook

    def start_backward(self, arm):
        if self.current or arm not in ARMS:
            raise RuntimeError('Unconsumed gradient capture or invalid arm')
        self.phase = arm

    def take(self):
        if set(self.current) != {f'block_{i}' for i in BLOCKS}:
            raise RuntimeError('Missing selected-input gradient; do not interpret incomplete trace')
        captured, self.current = self.current, {}
        return captured

    def close(self):
        for handle in self.tensor_hooks+self.module_hooks:
            handle.remove()
        self.current.clear()


def frames_of(name, value, frames=6):
    if name == 'latent':
        if value.ndim != 5 or value.shape[0] != 1 or value.shape[2] != frames:
            raise ValueError('Unexpected latent gradient layout')
        return value[0].movedim(1, 0).reshape(frames, -1)
    if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] % frames:
        raise ValueError('Unexpected token gradient layout')
    return value[0].reshape(frames, -1)


def stage_moments(name, a, b):
    return np.stack([moments(x.float().numpy(), y.float().numpy(), axes=None)
                     for x, y in zip(frames_of(name, a), frames_of(name, b))])


def run_prefix(g, pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input, output):
    from benchmark.wan_centered_pilot import clear
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Prefix diagnostic budget already started; preserve partial evidence')
    ready = prepare_prefix_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input)
    manifest, latents = ready['inputs'][:2]
    live = frozen_model(g, manifest)
    if len(g.transformer.blocks) != 40 or not g.transformer.gradient_checkpointing:
        raise ValueError('Requires pinned 40-block model with native gradient checkpointing enabled')
    if any(p.requires_grad or p.grad is not None for p in g.transformer.parameters()):
        raise ValueError('Requires frozen model without accumulated weight gradients')
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(hypothesis='Reference-gradient alignment may emerge in blocks 15-19, 10-14, 0-9 or latent-to-token mapping.',
        expected='Measure inputs 20,15,10,0 and latent in the same two objective backwards, replaying both endpoint controls.',
        max_positive_forwards=1, max_latent_backwards=2, max_block_entries_per_phase=21, max_total_block_entries=63,
        max_optimizer_steps=0, max_scheduler_steps=0, max_decodes=0,
        fixed=dict(index=39, sigma=manifest['experiment']['guidance_sigma'], guidance_block=20,
            logging_blocks=list(BLOCKS), loss='centered T8 destination NLL', support='same 324 common torso queries', injection=False),
        replay_tolerances=dict(block_moments_rtol=1e-6, block_moments_atol=1e-12,
            latent_gradient_rtol=5e-4, latent_gradient_atol=1e-5, latent_min_cosine=.99999),
        inputs_sha256=ready['inputs'][4], control_sha256=ready['provenance'], live=live,
        script_sha256=digest(__file__), helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in
            ('wan_centered_block_input.py','wan_centered_shared_input.py','wan_centered_qk.py','wan_centered_torso.py',
             'wan_centered_destination.py','wan_centered_response.py','wan_centered_gradients.py','wan_centered_huber.py',
             'wan_centered_attribution.py','wan_centered_pilot.py','review_wan_centered_response.py')},
        limits='One original prefix forward and two full-prefix backwards, native checkpointing unchanged. '
               'Count every block entry including checkpoint recomputation; early-stop recomputation may execute partial blocks. '
               'Hooks observe original input gradients without replacing them; no hook is added during recompute. '
               'Cosines locate alignment, not a motion-transfer result or a new guidance-layer trial.')
    with (output/'started.json').open('x', encoding='utf-8') as handle:
        json.dump(protocol, handle, indent=2)
    print('HYPOTHESIS:', protocol['hypothesis'], flush=True)
    print('LIMIT: one prefix capture, two latent backwards, at most 63 block entries including recomputation; no updates/videos.', flush=True)
    counts = dict(positive_forwards_attempted=0, positive_forwards_completed=0,
                  latent_backwards_attempted=0, latent_backwards_completed=0)
    observer = None
    replay = {}
    try:
        observer = PrefixObserver(g.transformer, output, counts)
        with torch.enable_grad():
            x = torch.from_numpy(latents['before']).to(g.device).clone().requires_grad_(True)
            counts['positive_forwards_attempted'] = 1
            observer.save_progress()
            flow, q, k = capture(g, x)
            counts['positive_forwards_completed'] = 1
            observer.save_progress()
            if any(observer.entries['capture'][str(i)] != 1 for i in range(21)) or set(observer.metadata) != {f'block_{i}' for i in BLOCKS}:
                raise RuntimeError('Initial prefix or logging endpoints incomplete')
            before = flow.detach().float().cpu().numpy().copy()
            np.testing.assert_array_equal(before, ready['inputs'][5]['before'])
            losses, logs = destination_objectives(q, k, ready['refs'], ready['indices'])
            for a in ARMS:
                np.testing.assert_array_equal(logs[a], ready['before_logp'][a])
            np.savez_compressed(output/'before_flow.npz', flow=before)
            np.savez_compressed(output/'before_target_log_probability.npz', **logs)
            np.savez_compressed(output/'optimization_support.npz', common_torso=ready['support'])
            del flow, q, k
            replay.update(before_flow_exact=True, target_log_probability_exact=True,
                          losses={a:float(losses[a].detach()) for a in ARMS}, arms={})
            write_json(output/'replay.json', replay)
            first, stats = None, {}
            for index, arm in enumerate(ARMS):
                observer.start_backward(arm)
                counts['latent_backwards_attempted'] += 1
                observer.save_progress()
                grad, = torch.autograd.grad(losses[arm], x, retain_graph=index == 0)
                counts['latent_backwards_completed'] += 1
                observer.save_progress()
                if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
                    raise RuntimeError('Nonfinite or zero latent gradient')
                current = observer.take()
                current['latent'] = grad.detach().cpu()
                del grad
                g._clear_kv([20])
                actual = current['latent'].float().numpy()
                expected = ready['latent_gradients'][arm]
                np.savez_compressed(output/f'{arm}_latent_gradient.npz', gradient=actual)
                delta = actual.astype(float)-expected
                agreement = compare(actual, expected)
                replay['arms'][arm] = dict(latent_max_abs=float(np.abs(delta).max()),
                    latent_rms=float(np.sqrt((delta**2).mean())), latent_cosine=agreement['cosine'],
                    latent_relative_rms=float(np.linalg.norm(delta)/np.linalg.norm(expected.astype(float))))
                energies = {n:[float((v.double()**2).sum()) for v in frames_of(n, t)] for n,t in current.items()}
                write_json(output/f'{arm}_squared_norms.json', energies)
                write_json(output/'replay.json', replay)
                np.testing.assert_allclose(actual, expected, rtol=5e-4, atol=1e-5)
                if agreement['cosine'] is None or agreement['cosine'] < .99999:
                    raise RuntimeError('Latent gradient direction does not replay')
                np.testing.assert_allclose(energies['block_20'], ready['block_moments'][:, index], rtol=1e-6, atol=1e-12)
                if index == 0:
                    first = current
                else:
                    stats = {n:stage_moments(n, first[n], current[n]) for n in ORDER}
                    np.savez_compressed(output/'gradient_moments.npz', **stats)
                    np.testing.assert_allclose(stats['block_20'], ready['block_moments'], rtol=1e-6, atol=1e-12)
                print(arm, 'backward and endpoint replay complete; block entries:', sum(observer.entries[arm].values()), flush=True)
            if x.grad is not None or any(p.grad is not None for p in g.transformer.parameters()):
                raise RuntimeError('Unexpected accumulated latent/weight gradient')
            np.testing.assert_array_equal(x.detach().cpu().numpy(), latents['before'])
        summary = {n:summarize_moments(v) for n,v in stats.items()}
        report = dict(port_success=False, status='Earlier-prefix gradient localization only', counts=counts,
            block_entries=observer.entries, total_block_entries=sum(sum(v.values()) for v in observer.entries.values()),
            endpoint_metadata={**observer.metadata, 'latent':dict(shape=list(latents['before'].shape), dtype=str(x.dtype))},
            replay=replay, stage_comparison=summary,
            by_frame={n:[dict(frame=i, **summarize_moments(v)) for i,v in enumerate(m)] for n,m in stats.items()},
            archived_block_input_comparison=ready['block_report']['stage_comparison']['block_input'],
            archived_latent_comparison=ready['latent_comparison'], baseline_unchanged=True,
            interpretation='Read block20 -> block15 -> block10 -> block0 -> latent in backward order. '
                'Checkpoint counters are entries, including partial recomputation. Logging blocks do not change the guidance objective. '
                'Endpoint replay is required before interpreting where reference-gradient alignment changes.')
        write_json(output/'prefix_report.json', report)
        return report
    except Exception as error:
        write_json(output/'failure.json', dict(error_type=type(error).__name__, error=str(error), counts=counts,
            block_entries=observer.entries if observer is not None else {}))
        raise
    finally:
        if observer is not None:
            observer.close()
        clear(g)
