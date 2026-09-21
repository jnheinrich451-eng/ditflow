"""CPU equivalence and bounded-run gates for the block 20 input diagnostic."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from benchmark import wan_centered_block_input as experiment
from benchmark.wan_centered_destination import destination_objectives
from benchmark.wan_centered_qk import feature_moments, moments
from guidance_utils.wan_centered_amf import centered_pair_flow
from guidance_utils.wan_modules import WanInjectionProcessor


def fixture(dtype):
    torch.manual_seed(32)
    block = WanTransformerBlock(dim=8, ffn_dim=16, num_heads=2).to(dtype=dtype)
    block.requires_grad_(False)
    processor = WanInjectionProcessor('20')
    processor.copy_kv = True
    block.attn1.set_processor(processor)
    x = torch.randn(1, 24, 8, dtype=dtype)
    text = torch.randn(1, 3, 8, dtype=dtype)
    temb = torch.randn(1, 6, 8, dtype=torch.float32)
    angle = torch.randn(1, 24, 1, 2).repeat_interleave(2, -1)
    rope = (angle.cos().to(dtype), angle.sin().to(dtype))
    support = np.ones((5, 4), bool); support[:, -1] = False
    refs = {a:(np.zeros((5, 4, 2), np.float32), support.copy()) for a in experiment.ARMS}
    ids = {a:np.tile((np.arange(4)+shift)%4, (5, 1)) for a, shift in [('forward', 1), ('reverse', -1)]}
    captured = {}
    def hook(module, args):
        captured['shared'] = args[0]
    handle = block.attn1.register_forward_pre_hook(hook)
    full = x.clone().requires_grad_(True)
    block(full, text, temb, rope)
    handle.remove()
    q, k = [v.reshape(6, 4, 2, 4) for v in (processor.query, processor.key)]
    losses, logs = destination_objectives(q, k, refs, ids, 1, 4)
    grads = [torch.autograd.grad(losses[a], (q, k, captured['shared'], full), retain_graph=i == 0)
             for i, a in enumerate(experiment.ARMS)]
    qk = {n:feature_moments(grads[0][i], grads[1][i]) for i, n in enumerate(('Q', 'K'))}
    h = [v[2][0].reshape(6, 4, 8).float().numpy() for v in grads]
    shared_moments = np.stack([moments(a, b, None) for a, b in zip(*h)])
    with torch.no_grad():
        before = torch.stack([centered_pair_flow(q[i], k[i+1], 1, 4) for i in range(5)]).float().numpy()
    prefix = np.zeros((1, 2, 6, 2, 2), np.float32)
    ready = dict(inputs=({'experiment':{'guidance_sigma':.45946}}, {'before':prefix}, refs, {}, {}, {'before':before}),
        refs=refs, support=support, indices=ids, before_logp=logs, qk_moments=qk, shared_moments=shared_moments,
        latent_comparison={'cosine':.975}, provenance={}, shared_report={'stage_comparison':{}}, qk_report={'stage_comparison':{}})
    return block, x, text, temb, rope, ready, captured['shared'].detach(), grads


def algebra_checks():
    for dtype in (torch.float64, torch.bfloat16):
        block, x, text, temb, rope, ready, shared, expected = fixture(dtype)
        endpoints, scale = experiment.local_block_path(block, x, temb, rope, 6, 4)
        q, k, h, m, normalized, x32, leaf = endpoints
        torch.testing.assert_close(h, shared, rtol=0, atol=0)
        torch.testing.assert_close(q.reshape_as(block.attn1.processor.query), block.attn1.processor.query, rtol=0, atol=0)
        torch.testing.assert_close(k.reshape_as(block.attn1.processor.key), block.attn1.processor.key, rtol=0, atol=0)
        losses, logs = destination_objectives(q, k, ready['refs'], ready['indices'], 1, 4)
        for i, a in enumerate(experiment.ARMS):
            actual = torch.autograd.grad(losses[a], endpoints, retain_graph=i == 0)
            for v, wanted in zip((actual[0], actual[1], actual[2], actual[-1]), expected[i]):
                torch.testing.assert_close(v, wanted, rtol=0, atol=0)
            torch.testing.assert_close(actual[3], actual[2].to(m.dtype), rtol=0, atol=0)
            torch.testing.assert_close(actual[4], (actual[3]*(1+scale)).to(normalized.dtype), rtol=0, atol=0)
            torch.testing.assert_close(actual[6], actual[5].to(leaf.dtype), rtol=0, atol=0)
        assert leaf.grad is None and all(p.grad is None for p in block.parameters())
    print('PASS: FP64/BF16 native block Q/K, shared input and block-input derivatives match exactly; cast/modulation identities.')


def runner_checks():
    block, x, text, temb, rope, ready, _, _ = fixture(torch.bfloat16)
    # Drop the fixture's old graph before running the no-grad capture.
    block.attn1.processor.clear()
    model = SimpleNamespace(device='cpu', guidance_embeds=torch.zeros(2, 1, 1), timesteps=torch.arange(50))
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.block = block
            self.blocks = [SimpleNamespace(module=block, attn1=block.attn1)]*40
    model.transformer = Transformer()
    def set_mode(blocks, inject, copy):
        block.attn1.processor.inject_kv = inject
        block.attn1.processor.copy_kv = copy
    model._set_kv_mode = set_mode
    calls = []
    def forward(latent, embeds, timestep):
        assert not torch.is_grad_enabled() and not latent.requires_grad
        calls.append(1)
        block(x, text, temb, rope)
    model._forward_transformer = forward
    original = block.attn1.processor
    measure = experiment.measure_block
    def small(*args):
        return measure(*args, frames=6, height=1, width=4)
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(experiment, 'prepare_block_inputs', return_value=ready), \
         patch.object(experiment, 'frozen_model', return_value={}), \
         patch.object(experiment, 'measure_block', side_effect=small), \
         patch('benchmark.wan_centered_pilot.clear'):
        root = Path(tmp)
        args = (model, *[root/n for n in ('p', 'a', 'h', 'g', 'r', 'd', 't', 'q', 's')])
        report = experiment.run_block_input(*args, root/'ok')
        assert len(calls) == 1 and block.attn1.processor is original and not block._forward_pre_hooks
        assert report['counts'] == dict(positive_forwards_attempted=1, positive_forwards_completed=1,
            local_backwards_attempted=2, local_backwards_completed=2)
        assert not report['port_success'] and not report['latent_graph_created'] and report['baseline_unchanged']
        assert set(report['stage_comparison']) == set(experiment.STAGES)
        assert all(all(v.values()) for v in report['derivative_identity_checks'].values())
        with np.load(root/'ok/gradient_moments.npz') as arrays:
            assert arrays['block_input'].shape == (6, 3)
            assert all(np.isfinite(arrays[n]).all() for n in arrays.files)
        json.dumps(report, allow_nan=False)
        try:
            experiment.run_block_input(*args, root/'ok')
        except RuntimeError as error:
            assert 'budget already' in str(error)
        else:
            raise AssertionError('Repeat budget accepted')
        assert len(calls) == 1
        for name in ('bad_logp', 'bad_input', 'bad_moments'):
            if name == 'bad_logp':
                context = patch.object(experiment, 'prepare_block_inputs', return_value={**ready,
                    'before_logp':{a:v+1 for a, v in ready['before_logp'].items()}})
            elif name == 'bad_moments':
                context = patch.object(experiment, 'prepare_block_inputs', return_value={**ready,
                    'shared_moments':ready['shared_moments']*2})
            else:
                original_path = experiment.local_block_path
                def wrong(*args):
                    endpoints, scale = original_path(*args)
                    endpoints = list(endpoints); endpoints[2] = endpoints[2]+1
                    return tuple(endpoints), scale
                context = patch.object(experiment, 'local_block_path', side_effect=wrong)
            with context:
                try:
                    experiment.run_block_input(*args, root/name)
                except (RuntimeError, AssertionError):
                    pass
                else:
                    raise AssertionError('Bad replay accepted')
            failure = json.loads((root/name/'failure.json').read_text())
            assert failure['counts']['local_backwards_attempted'] == (2 if name == 'bad_moments' else 0)
            assert block.attn1.processor is original and not block._forward_pre_hooks
            assert not (root/name/'block_input_report.json').exists()
    print('PASS: one capture/two local backwards, replay and repeat guards, restored hook/processor on success/failure.')


def preflight_checks():
    _, _, _, _, _, ready, _, _ = fixture(torch.bfloat16)
    stats = {n:np.tile(v.sum((0, 1))[None, None]/240, (6, 40, 1)) for n, v in ready['qk_moments'].items()}
    ready['qk_moments'] = stats
    stats = {**stats, 'input_combined':ready['shared_moments']}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = Path(experiment.__file__).parent
        protocol = dict(inputs_sha256=ready['inputs'][4], control_sha256=ready['provenance'],
            helper_sources_sha256={}, script_sha256=experiment.digest(source/'wan_centered_shared_input.py'))
        report = dict(counts=dict(positive_forwards_attempted=1, positive_forwards_completed=1,
            local_backwards_attempted=2, local_backwards_completed=2), baseline_unchanged=True, port_success=False,
            latent_graph_created=False, shared_input_dtype='torch.bfloat16', shared_input_shape=[1,9360,5120],
            archived_latent_comparison=ready['latent_comparison'], archived_qk_comparison=ready['qk_report']['stage_comparison'],
            stage_comparison={n:experiment.summarize_moments(v) for n, v in stats.items()})
        experiment.write_json(root/'started.json', protocol)
        experiment.write_json(root/'shared_input_report.json', report)
        np.savez_compressed(root/'before_flow.npz', flow=ready['inputs'][5]['before'])
        np.savez_compressed(root/'optimization_support.npz', common_torso=ready['support'])
        np.savez_compressed(root/'before_target_log_probability.npz', **ready['before_logp'])
        np.savez_compressed(root/'gradient_moments.npz', **stats)
        with patch.object(experiment, 'prepare_shared_inputs', side_effect=lambda *a:dict(ready)):
            checked = experiment.prepare_block_inputs(*['unused']*8, root)
            assert len(checked['provenance']['shared_input_artifacts']) == 6
            stats['input_combined'] *= 2
            np.savez_compressed(root/'gradient_moments.npz', **stats)
            try:
                experiment.prepare_block_inputs(*['unused']*8, root)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupt shared-input statistics accepted')
    print('PASS: completed shared-input preflight and corrupt moment rejection.')


def notebook_checks():
    import nbformat
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root/'wan_centered_block_input.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, '<notebook cell>', 'exec')
    assert 'QK, SHARED, OUTPUT)' in nb.cells[10].source
    print('PASS: clean notebook schema, syntax and shared-input control wiring.')


if __name__ == '__main__':
    algebra_checks()
    preflight_checks()
    runner_checks()
    notebook_checks()
