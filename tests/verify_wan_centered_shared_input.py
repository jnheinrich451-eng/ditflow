"""CPU gates for native local replay, branch decomposition, and bounded failure paths."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from diffusers.models.transformers.transformer_wan import WanAttention

from benchmark import wan_centered_shared_input as experiment
from benchmark.wan_centered_destination import destination_objectives
from benchmark.wan_centered_qk import feature_moments
from guidance_utils.wan_modules import WanInjectionProcessor
from guidance_utils.wan_centered_amf import centered_pair_flow


def fixture(dtype):
    torch.manual_seed(27)
    processor = WanInjectionProcessor('20')
    processor.copy_kv = True
    attn = WanAttention(dim=8, heads=2, dim_head=4, processor=processor).to(dtype=dtype)
    attn.requires_grad_(False)
    hidden = torch.randn(1, 24, 8, dtype=dtype)
    angles = torch.randn(1, 24, 1, 2).repeat_interleave(2, -1)
    rope = (angles.cos().to(dtype), angles.sin().to(dtype))
    support = np.ones((5, 4), bool)
    support[:, -1] = False
    refs = {a:(np.zeros((5, 4, 2), np.float32), support.copy()) for a in experiment.ARMS}
    ids = {a:np.tile((np.arange(4)+shift)%4, (5, 1)) for a, shift in [('forward', 1), ('reverse', -1)]}
    with torch.no_grad():
        attn(hidden, rotary_emb=rope)
        q0, k0 = [v.reshape(6, 4, 2, 4).clone() for v in (processor.query, processor.key)]
        before = torch.stack([centered_pair_flow(q0[i], k0[i+1], 1, 4) for i in range(5)]).float().numpy()
    q, k = q0.requires_grad_(), k0.requires_grad_()
    losses, logs = destination_objectives(q, k, refs, ids, 1, 4)
    grads = [torch.autograd.grad(losses[a], (q, k), retain_graph=i == 0) for i, a in enumerate(experiment.ARMS)]
    stats = {n:feature_moments(grads[0][i], grads[1][i]) for i, n in enumerate(('Q', 'K'))}
    prefix = np.zeros((1, 2, 6, 2, 2), np.float32)
    ready = dict(inputs=({'experiment':{'guidance_sigma':.45946}}, {'before':prefix}, refs, {}, {}, {'before':before}),
        refs=refs, support=support, indices=ids, before_logp=logs, qk_moments=stats,
        latent_comparison={'cosine':.975}, provenance={}, qk_report={'stage_comparison':{}})
    return attn, hidden, rope, ready


def algebra_checks():
    for dtype in (torch.float64, torch.bfloat16):
        attn, hidden, rope, ready = fixture(dtype)
        q, k, qi, ki, shared = experiment.local_paths(attn, hidden, rope, 6, 4)
        native = attn.processor
        torch.testing.assert_close(q.reshape_as(native.query), native.query, rtol=0, atol=0)
        torch.testing.assert_close(k.reshape_as(native.key), native.key, rtol=0, atol=0)
        losses, logs = destination_objectives(q, k, ready['refs'], ready['indices'], 1, 4)
        # Independent ordinary shared-input processor graph, no branch clones/taps.
        x = hidden.detach().clone().requires_grad_(True)
        attn(x, rotary_emb=rope)
        expected, _ = destination_objectives(native.query.reshape(6, 4, 2, 4),
            native.key.reshape(6, 4, 2, 4), ready['refs'], ready['indices'], 1, 4)
        for i, a in enumerate(experiment.ARMS):
            dq, dk, ghq, ghk, gh = torch.autograd.grad(losses[a], (q, k, qi, ki, shared), retain_graph=i == 0)
            gx, = torch.autograd.grad(expected[a], (x,), retain_graph=i == 0)
            torch.testing.assert_close(gh, gx, rtol=0, atol=0)
            torch.testing.assert_close(gh, ghq+ghk, rtol=0, atol=0)
            assert ghq.count_nonzero() and ghk.count_nonzero()
            assert dq[-1].count_nonzero() == 0 and dk[0].count_nonzero() == 0
        assert shared.grad is None and all(p.grad is None for p in attn.parameters())
    print('PASS: FP64/BF16 native Q/K replay, combined gradient equals unmodified processor derivative, branch sum, no weight grads.')


def runner_checks():
    attn, hidden, rope, ready = fixture(torch.bfloat16)
    model = SimpleNamespace(device='cpu', guidance_embeds=torch.zeros(2, 1, 1), timesteps=torch.arange(50))
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = attn
            self.blocks = [SimpleNamespace(attn1=attn)]*21
    model.transformer = Transformer()
    def set_mode(blocks, inject, copy):
        attn.processor.inject_kv = inject
        attn.processor.copy_kv = copy
    model._set_kv_mode = set_mode
    calls = []
    def forward(x, embeds, timestep):
        assert not torch.is_grad_enabled() and not x.requires_grad
        calls.append(1)
        attn(hidden, rotary_emb=rope)
    model._forward_transformer = forward
    original = attn.processor
    measure = experiment.measure_local
    def small(*args):
        return measure(*args, frames=6, height=1, width=4)
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(experiment, 'prepare_shared_inputs', return_value=ready), \
         patch.object(experiment, 'frozen_model', return_value={}), \
         patch.object(experiment, 'measure_local', side_effect=small), \
         patch('benchmark.wan_centered_pilot.clear'):
        root = Path(tmp)
        args = (model, *[root/n for n in ('p', 'a', 'h', 'g', 'r', 'd', 't', 'q')])
        report = experiment.run_shared_input(*args, root/'ok')
        assert len(calls) == 1 and attn.processor is original
        assert report['counts'] == dict(positive_forwards_attempted=1, positive_forwards_completed=1,
            local_backwards_attempted=2, local_backwards_completed=2)
        assert report['baseline_unchanged'] and not report['latent_graph_created'] and not report['port_success']
        for arm in experiment.ARMS:
            branch = report['within_arm_branch_comparison'][arm]
            assert branch['q_path_norm'] > 0 and branch['k_path_norm'] > 0
            assert branch['native_sum_norm'] == report['stage_comparison']['input_combined'][arm+'_norm']
        with np.load(root/'ok/gradient_moments.npz') as arrays:
            assert arrays['input_Q_path'].shape == (6, 3)
            assert arrays['Q'].shape == (6, 2, 3)
            assert all(np.isfinite(arrays[n]).all() for n in arrays.files)
        assert len(list((root/'ok').glob('*.npz'))) == 4
        json.dumps(report, allow_nan=False)
        try:
            experiment.run_shared_input(*args, root/'ok')
        except RuntimeError as error:
            assert 'budget already' in str(error)
        else:
            raise AssertionError('Repeat budget accepted')
        assert len(calls) == 1
        for name in ('bad_logp', 'bad_qk', 'bad_moments'):
            if name == 'bad_logp':
                bad = {**ready, 'before_logp':{a:v+1 for a, v in ready['before_logp'].items()}}
                context = patch.object(experiment, 'prepare_shared_inputs', return_value=bad)
            elif name == 'bad_moments':
                bad = {**ready, 'qk_moments':{n:v*2 for n, v in ready['qk_moments'].items()}}
                context = patch.object(experiment, 'prepare_shared_inputs', return_value=bad)
            else:
                original_paths = experiment.local_paths
                def wrong_q(*args):
                    q, *rest = original_paths(*args)
                    return (q+1, *rest)
                context = patch.object(experiment, 'local_paths', side_effect=wrong_q)
            with context:
                try:
                    experiment.run_shared_input(*args, root/name)
                except (RuntimeError, AssertionError):
                    pass
                else:
                    raise AssertionError('Bad replay accepted')
            failure = json.loads((root/name/'failure.json').read_text())
            assert failure['counts']['local_backwards_attempted'] == (2 if name == 'bad_moments' else 0)
            assert not (root/name/'shared_input_report.json').exists()
            assert attn.processor is original
    print('PASS: one forward/two local backwards, compact artifacts, replay/repeat guards and processor restoration on failure.')


def preflight_checks():
    _, _, _, ready = fixture(torch.bfloat16)
    # Preflight fixes the real geometry; construct a small-value archive with its exact shape.
    stats = {n:np.tile(v.sum((0, 1))[None, None]/240, (6, 40, 1)) for n, v in ready['qk_moments'].items()}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = Path(experiment.__file__).parent
        protocol = dict(inputs_sha256=ready['inputs'][4], control_sha256=ready['provenance'],
            helper_sources_sha256={}, script_sha256=experiment.digest(source/'wan_centered_qk.py'))
        report = dict(counts=dict(positive_forwards_attempted=1, positive_forwards_completed=1,
            readout_backwards_attempted=2, readout_backwards_completed=2), baseline_unchanged=True,
            latent_graph_created=False, qk_dtype='torch.bfloat16', archived_latent_comparison=ready['latent_comparison'],
            stage_comparison={n:experiment.summarize_moments(v) for n, v in stats.items()})
        experiment.write_json(root/'started.json', protocol)
        experiment.write_json(root/'qk_report.json', report)
        np.savez_compressed(root/'before_flow.npz', flow=ready['inputs'][5]['before'])
        np.savez_compressed(root/'optimization_support.npz', common_torso=ready['support'])
        np.savez_compressed(root/'before_target_log_probability.npz', **ready['before_logp'])
        np.savez_compressed(root/'gradient_moments.npz', **stats)
        with patch.object(experiment, 'prepare_qk_inputs', side_effect=lambda *a:dict(ready)):
            checked = experiment.prepare_shared_inputs(*['unused']*7, root)
            assert len(checked['provenance']['qk_artifacts']) == 6
            stats['Q'] *= 2
            np.savez_compressed(root/'gradient_moments.npz', **stats)
            try:
                experiment.prepare_shared_inputs(*['unused']*7, root)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupt Q/K moments accepted')
    print('PASS: completed Q/K archive preflight and corruption rejection.')


def notebook_checks():
    import nbformat
    root = Path(__file__).resolve().parents[1]
    nb = nbformat.read(root/'wan_centered_shared_input.ipynb', as_version=4)
    nbformat.validate(nb)
    for cell in nb.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is None and not cell.outputs
            compile(cell.source, '<notebook cell>', 'exec')
    assert 'QK, OUTPUT)' in nb.cells[10].source
    print('PASS: clean notebook schema, syntax and prior Q/K input wiring.')


if __name__ == '__main__':
    algebra_checks()
    preflight_checks()
    runner_checks()
    notebook_checks()
