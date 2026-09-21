"""One saved state, two backwards without gradient hooks, one with hooks."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from benchmark.review_wan_centered_response import check_values, read_npz
from benchmark.wan_centered_attribution import digest
from benchmark.wan_centered_destination import destination_objectives
from benchmark.wan_centered_gradients import capture, compare, write_json
from benchmark.wan_centered_huber import frozen_model
from benchmark.wan_centered_prefix import BLOCKS, frames_of, prepare_prefix_inputs


PASSES = ('unlogged_1', 'unlogged_2', 'logged')


def difference(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    delta = a-b
    row = compare(a,b)
    return dict(cosine=row['cosine'], actual_norm=row['forward_norm'], reference_norm=row['reverse_norm'],
        max_abs=float(np.abs(delta).max()), rms=float(np.sqrt((delta**2).mean())),
        relative_rms=float(np.linalg.norm(delta)/np.linalg.norm(b)),
        old_elementwise_gate_passed=bool(np.allclose(a,b,rtol=5e-4,atol=1e-5)),
        old_direction_gate_passed=bool(row['cosine'] is not None and row['cosine'] >= .99999))


def prepare_repeatability_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input, retry):
    ready = prepare_prefix_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input)
    root = Path(retry)
    protocol = json.loads((root/'started.json').read_text())
    failed = json.loads((root/'failure.json').read_text())
    if protocol['inputs_sha256'] != ready['inputs'][4] or protocol['control_sha256'] != ready['provenance']:
        raise ValueError('Retry used different frozen inputs')
    for name, expected in {**protocol['helper_sources_sha256'], 'wan_centered_prefix.py':protocol['script_sha256']}.items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f'Failed retry source changed: {name}')
    check_values(failed['counts'], dict(positive_forwards_attempted=1, positive_forwards_completed=1,
        latent_backwards_attempted=1, latent_backwards_completed=1))
    if failed['error_type'] != 'AssertionError':
        raise ValueError('Requires the completed first-backward replay failure, not the OOM attempt')
    np.testing.assert_array_equal(read_npz(root/'before_flow.npz')['flow'], ready['inputs'][5]['before'])
    np.testing.assert_array_equal(read_npz(root/'optimization_support.npz')['common_torso'], ready['support'])
    for a,v in read_npz(root/'before_target_log_probability.npz').items():
        np.testing.assert_array_equal(v, ready['before_logp'][a])
    grad = read_npz(root/'forward_latent_gradient.npz')['gradient']
    if grad.shape != ready['latent_gradients']['forward'].shape or not np.isfinite(grad).all():
        raise ValueError('Invalid retry gradient')
    replay = json.loads((root/'replay.json').read_text())['arms']['forward']
    actual = difference(grad, ready['latent_gradients']['forward'])
    check_values({k:actual[n] for k,n in [('latent_max_abs','max_abs'),('latent_rms','rms'),
                 ('latent_relative_rms','relative_rms'),('latent_cosine','cosine')]}, replay)
    files = ['started.json','failure.json','replay.json','before_flow.npz','before_target_log_probability.npz',
             'optimization_support.npz','forward_latent_gradient.npz','forward_squared_norms.json']
    ready['retry_gradient'] = grad
    ready['provenance'] = {**ready['provenance'], 'retry_artifacts':{n:digest(root/n) for n in files}}
    return ready


def device_details(g):
    if not torch.cuda.is_available():
        raise RuntimeError('Pretrained control requires CUDA')
    prop = torch.cuda.get_device_properties(g.device)
    if prop.total_memory < 70*1024**3:
        raise RuntimeError('Use an 80 GB-class GPU for this control; the 40 GB prefix attempt ran out of memory.')
    mm, cn = torch.backends.cuda.matmul, torch.backends.cudnn
    return dict(name=prop.name, total_memory_bytes=prop.total_memory, capability=[prop.major,prop.minor],
        uuid=str(getattr(prop,'uuid','unavailable')), torch_version=torch.__version__, cuda_version=torch.version.cuda,
        cudnn_version=cn.version(), deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=cn.deterministic, cudnn_benchmark=cn.benchmark, cudnn_allow_tf32=cn.allow_tf32,
        matmul_allow_tf32=mm.allow_tf32, bf16_reduced_precision_reduction=getattr(mm,'allow_bf16_reduced_precision_reduction',None),
        cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))


class RepeatObserver:
    """Constant block counters/input references; gradient hooks added only on pass 3."""
    def __init__(self, transformer, output, counts):
        self.output, self.counts = Path(output), counts
        self.phase = 'capture'
        self.entries = {p:{str(i):0 for i in range(40)} for p in ('capture', *PASSES)}
        self.inputs, self.current = {}, {}
        self.tensor_hooks, self.module_hooks = [], []
        for i,b in enumerate(transformer.blocks):
            self.module_hooks.append(b.register_forward_pre_hook(self.before_block(i), with_kwargs=True))

    def save(self):
        write_json(self.output/'progress.json', dict(**self.counts, block_entries=self.entries))

    def before_block(self, index):
        def hook(module,args,kwargs):
            row = self.entries[self.phase]
            row[str(index)] += 1
            self.save()
            if index > 20 or row[str(index)] > 1:
                raise RuntimeError('Repeatability block-entry budget exceeded')
            if self.phase == 'capture' and index in BLOCKS:
                x = args[0] if args else kwargs['hidden_states']
                if not torch.is_grad_enabled() or not x.requires_grad:
                    raise RuntimeError('Requires original graph inputs and non-reentrant checkpointing')
                self.inputs[f'block_{index}'] = x
        return hook

    def start(self, phase):
        self.phase = phase
        if phase == 'logged':
            if self.tensor_hooks:
                raise RuntimeError('Logging hooks already installed')
            for name,x in self.inputs.items():
                def receive(grad, name=name):
                    if self.phase != 'logged' or name in self.current:
                        raise RuntimeError('Unexpected or duplicate logged gradient')
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Nonfinite logged gradient')
                    self.current[name] = grad.detach().cpu()
                self.tensor_hooks.append(x.register_hook(receive))
        elif self.tensor_hooks:
            raise RuntimeError('Unlogged backward has gradient hooks')

    def close(self):
        for h in self.tensor_hooks+self.module_hooks:
            h.remove()
        self.inputs.clear(); self.current.clear()


def run_repeatability(g, pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input, retry, output):
    from benchmark.wan_centered_pilot import clear
    output = Path(output)
    if (output/'started.json').exists():
        raise RuntimeError('Repeatability budget already started; preserve partial evidence')
    ready = prepare_repeatability_inputs(pilot, attribution, huber, gradients, response, destination, torso, qk, shared, block_input, retry)
    live, hardware = frozen_model(g,ready['inputs'][0]), device_details(g)
    if len(g.transformer.blocks) != 40 or not g.transformer.gradient_checkpointing:
        raise ValueError('Requires pinned 40-block model with checkpointing enabled')
    if any(p.requires_grad or p.grad is not None for p in g.transformer.parameters()):
        raise ValueError('Requires frozen weights without accumulated gradients')
    output.mkdir(parents=True, exist_ok=True)
    helpers = ['wan_centered_prefix.py','wan_centered_block_input.py','wan_centered_shared_input.py','wan_centered_qk.py',
        'wan_centered_torso.py','wan_centered_destination.py','wan_centered_response.py','wan_centered_gradients.py',
        'wan_centered_huber.py','wan_centered_attribution.py','wan_centered_pilot.py','review_wan_centered_response.py']
    protocol = dict(hypothesis='Replay drift may be ordinary same-state backward variation or associated with gradient logging hooks.',
        expected='Compare two same-reference unlogged backwards, then a logged backward on the same graph/device/state.',
        max_positive_forwards=1,max_latent_backwards=3,max_total_block_entries=84,
        max_optimizer_steps=0,max_scheduler_steps=0,max_decodes=0,
        fixed=dict(index=39, sigma=ready['inputs'][0]['experiment']['guidance_sigma'],guidance_block=20,
            logging_blocks=list(BLOCKS),objective='original-reference centered T8 destination NLL', support='same 324 torso queries', injection=False),
        inputs_sha256=ready['inputs'][4],control_sha256=ready['provenance'],live=live,hardware=hardware,
        script_sha256=digest(__file__),helper_sources_sha256={n:digest(Path(__file__).with_name(n)) for n in helpers},
        limits='One graph retained for three backwards, no gradient accumulation or updates. All passes have the same module counters '
            'and retained input references; first two have no tensor gradient hooks, third adds the previous logging operation. '
            'This isolates tensor-hook presence, not all Python instrumentation. Pass order and single-repeat sampling limit causality. '
            'Old replay gates are reported, not enforced here: measuring their failure is this control objective. '
            'No tolerance is changed in the original prefix experiment; no new baseline is automatically accepted.')
    with (output/'started.json').open('x',encoding='utf-8') as f:
        json.dump(protocol,f,indent=2)
    print('HYPOTHESIS:',protocol['hypothesis'],flush=True)
    print('LIMIT: one capture, three same-reference backwards, at most 84 block entries; no updates/videos.',flush=True)
    counts = dict(positive_forwards_attempted=0,positive_forwards_completed=0,latent_backwards_attempted=0,latent_backwards_completed=0)
    observer = None
    try:
        observer = RepeatObserver(g.transformer,output,counts)
        latent = ready['inputs'][1]['before']
        with torch.enable_grad():
            x = torch.from_numpy(latent).to(g.device).clone().requires_grad_(True)
            counts['positive_forwards_attempted']=1;observer.save()
            flow,q,k = capture(g,x)
            counts['positive_forwards_completed']=1;observer.save()
            if set(observer.inputs) != {f'block_{i}' for i in BLOCKS} or any(observer.entries['capture'][str(i)] != 1 for i in range(21)):
                raise RuntimeError('Incomplete original capture')
            before = flow.detach().float().cpu().numpy().copy()
            np.testing.assert_array_equal(before,ready['inputs'][5]['before'])
            losses,logs = destination_objectives(q,k,ready['refs'],ready['indices'])
            for a in ('forward','reverse'):
                np.testing.assert_array_equal(logs[a],ready['before_logp'][a])
            loss = losses['forward']; del losses,flow,q,k
            np.savez_compressed(output/'before_flow.npz',flow=before)
            np.savez_compressed(output/'before_target_log_probability.npz',**logs)
            np.savez_compressed(output/'optimization_support.npz',common_torso=ready['support'])
            write_json(output/'replay.json',dict(before_flow_exact=True,target_log_probability_exact=True,loss=float(loss.detach())))
            saved, comparisons = {}, {}
            for i,phase in enumerate(PASSES):
                observer.start(phase)
                counts['latent_backwards_attempted']+=1;observer.save()
                grad, = torch.autograd.grad(loss,x,retain_graph=i<2)
                counts['latent_backwards_completed']+=1;observer.save()
                if not torch.isfinite(grad).all() or not torch.count_nonzero(grad):
                    raise RuntimeError('Nonfinite or zero latent gradient')
                saved[phase] = grad.detach().float().cpu().numpy().copy()
                del grad
                g._clear_kv([20])
                np.savez_compressed(output/f'{phase}_latent_gradient.npz',gradient=saved[phase])
                comparisons[phase] = dict(vs_archive=difference(saved[phase],ready['latent_gradients']['forward']),
                    vs_retry=difference(saved[phase],ready['retry_gradient']))
                if i:
                    comparisons[phase]['vs_unlogged_1']=difference(saved[phase],saved['unlogged_1'])
                if i==2:
                    comparisons[phase]['vs_unlogged_2']=difference(saved[phase],saved['unlogged_2'])
                write_json(output/'comparisons.json',comparisons)
                if i<2 and observer.current:
                    raise RuntimeError('Gradient logging occurred in an unlogged pass')
                if i==2:
                    if set(observer.current) != {f'block_{j}' for j in BLOCKS}:
                        raise RuntimeError('Missing logged gradient')
                    energies={n:[float((v.double()**2).sum()) for v in frames_of(n,t)] for n,t in observer.current.items()}
                    write_json(output/'logged_squared_norms.json',energies)
                print(phase,'completed; versus archive:',comparisons[phase]['vs_archive'],flush=True)
            if x.grad is not None or any(p.grad is not None for p in g.transformer.parameters()):
                raise RuntimeError('Unexpected accumulated latent/weight gradient')
            np.testing.assert_array_equal(x.detach().cpu().numpy(),latent)
        report = dict(status='Repeatability control completed; interpretation requires review',port_success=False,
            baseline_unchanged=True,counts=counts,block_entries=observer.entries,
            total_block_entries=sum(sum(v.values()) for v in observer.entries.values()),hardware=hardware,
            comparisons=comparisons,old_tolerances_unchanged=True,
            interpretation='All gradients use the same original reference. Compare unlogged_2 vs unlogged_1 '
                'with logged vs each unlogged pass. This is not a new forward/reverse motion comparison. '
                'Two unlogged passes provide one observed repeat difference, not a statistical tolerance bound. '
                'A logged difference can suggest an instrumentation effect but pass-order effects remain possible.')
        write_json(output/'repeatability_report.json',report)
        return report
    except Exception as error:
        write_json(output/'failure.json',dict(error_type=type(error).__name__,error=str(error),counts=counts,
            block_entries=observer.entries if observer else {}))
        raise
    finally:
        if observer:
            observer.close()
        clear(g)
