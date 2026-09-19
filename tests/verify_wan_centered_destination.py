"""CPU gates for destination geometry, saturated gradients and bounded execution."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from benchmark import wan_centered_destination as candidate
from guidance_utils.wan_centered_amf import centered_pair_flow


def geometry_checks():
    flow = np.zeros((1, 6, 2), np.float32)
    mask = np.zeros((1, 6), bool)
    mask[0, 1] = True
    flow[0, 1] = [1, 1]  # (x=1,y=0) -> (x=2,y=1), flattened destination 5.
    assert candidate.destination_indices(flow, mask, 2, 3)[0, 1] == 5
    for value in ([1.1, 1], [2, 1], [0, -1], [float('nan'), 0]):
        broken = flow.copy(); broken[0, 1] = value
        try:
            candidate.destination_indices(broken, mask, 2, 3)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid destination accepted')
    assert mask.sum() == 1
    print('PASS: XY direction/flattening, integer, bounds and finite-support gates.')


def objective_checks():
    # Confidently wrong identity correspondence. Target is the other spatial token.
    q = torch.tensor([[[[3., 0.]], [[0., 3.]]]]*2, dtype=torch.float64, requires_grad=True)
    k = q.detach().clone().requires_grad_(True)
    flow = np.array([[[1., 0.], [-1., 0.]]])
    refs = {a:(flow, np.ones((1, 2), bool)) for a in candidate.ARMS}
    indices = {a:candidate.destination_indices(*refs[a], 1, 2) for a in candidate.ARMS}
    losses, logp = candidate.destination_objectives(q, k, refs, indices, 1, 2)
    measured = centered_pair_flow(q[0], k[1], 1, 2)
    old = torch.nn.functional.huber_loss(measured, torch.from_numpy(flow[0]), delta=1.)
    old_grad, = torch.autograd.grad(old, q, retain_graph=True)
    new_grad, = torch.autograd.grad(losses['forward'], q, retain_graph=True)
    assert new_grad.norm() > 1 and old_grad.norm() < 1e-12
    assert np.isfinite(logp['forward']).all() and losses['forward'] > 30
    assert np.isclose(float(losses['forward'].detach()), -logp['forward'].mean())
    # Confirm selected target probability is that of the unchanged flow distribution.
    with torch.no_grad():
        expected_x = measured[:, 0] + torch.arange(2)
        probability_other = torch.stack((expected_x[0], 1-expected_x[1]))
        torch.testing.assert_close(probability_other, torch.from_numpy(np.exp(logp['forward'][0])), atol=1e-14, rtol=1e-10)
    # Select one query only: unselected source still receives gradient through centering.
    masked = {a:(flow, np.array([[True, False]])) for a in candidate.ARMS}
    one, _ = candidate.destination_objectives(q, k, masked, indices, 1, 2)
    grad, = torch.autograd.grad(one['forward'], q)
    assert grad[0, 1].norm() > 0
    # Two independent backwards sharing checkpointed features, no gradient accumulation.
    x = torch.randn(2, 2, 1, 2, dtype=torch.float64, requires_grad=True)
    features = checkpoint(lambda z: z.sin(), x, use_reentrant=False)
    both, _ = candidate.destination_objectives(features, features, refs, indices, 1, 2)
    a, = torch.autograd.grad(both['forward'], x, retain_graph=True)
    b, = torch.autograd.grad(both['reverse'], x)
    torch.testing.assert_close(a, b)
    assert x.grad is None
    print('PASS: stable NLL corrects saturated wrong match, unchanged probability formula, full-source centering, shared checkpoint backwards.')


def runner_checks():
    prefix = np.zeros((1, 2, 6, 2, 2), np.float32)
    before = np.full((5, 1560, 2), .05, np.float32)
    refs = {a:(np.full_like(before, v), np.ones(before.shape[:-1], bool))
            for a,v in [('forward', .25), ('reverse', -.25)]}
    readout = dict(same_probability=np.full(before.shape[:-1], .999, np.float32),
                   flow_jacobian_frobenius=np.full(before.shape[:-1], .001, np.float32))
    inputs = ({'experiment':{'guidance_sigma':.45946}}, {'before':prefix}, refs, {}, {}, {'before':before})
    checked = (inputs, readout, {}, {a:before for a in refs},
               {'conditions':{a+'_full':{} for a in refs}}, {}, {})
    g = SimpleNamespace(device='cpu', dtype=torch.bfloat16, _clear_kv=Mock())
    calls = []
    def capture(g, x):
        calls.append(x.detach().clone())
        assert x.requires_grad == (len(calls) == 1)
        return torch.from_numpy(before)+x.mean(), x, None
    def objective(q, k, refs, indices):
        losses = {a:(q.mean()-v)**2+1 for a,v in [('forward', .25), ('reverse', -.25)]}
        logp = {a:np.full(before.shape[:-1], -float(v.detach())) for a,v in losses.items()}
        return losses, logp
    def analysis(q, k, refs, output):
        field = before+float(q.mean())
        np.savez_compressed(Path(output)/'readout_derivatives.npz', flow=field, **readout)
        return {}, field
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with patch.object(candidate, 'prepare_destination_inputs', return_value=checked), \
             patch.object(candidate, 'frozen_model', return_value={}), \
             patch.object(candidate, 'capture', side_effect=capture), \
             patch.object(candidate, 'destination_objectives', side_effect=objective), \
             patch.object(candidate, 'readout_analysis', side_effect=analysis), \
             patch('benchmark.wan_centered_pilot.clear'):
            args = (g, root/'p', root/'a', root/'h', root/'g', root/'r')
            report = candidate.run_destination(*args, root/'out')
            assert len(calls) == 3 and report['counts'] == dict(
                positive_forwards_attempted=3, positive_forwards_completed=3,
                latent_backwards_attempted=2, latent_backwards_completed=2, first_adam_proposals=2)
            assert calls[1].mean() > 0 and calls[2].mean() < 0
            torch.testing.assert_close(calls[1], -calls[2])
            assert report['baseline_unchanged'] and not report['port_success']
            assert report['separation']['all_common']['projection_gain'] > 0
            json.dumps(report, allow_nan=False)
            assert len(list((root/'out').glob('*/actual_delta.npz'))) == 2
            try:
                candidate.run_destination(*args, root/'out')
            except RuntimeError as exc:
                assert 'budget already' in str(exc)
            else:
                raise AssertionError('Repeat budget accepted')
            assert len(calls) == 3
            # Corrupt before-state replay must stop before either backward/update.
            bad = list(checked)
            bad_inputs = list(inputs); bad_inputs[5] = {'before':before+1}
            bad[0] = tuple(bad_inputs)
            calls.clear()
            with patch.object(candidate, 'prepare_destination_inputs', return_value=tuple(bad)):
                try:
                    candidate.run_destination(*args, root/'bad')
                except RuntimeError as exc:
                    assert 'does not replay' in str(exc)
                else:
                    raise AssertionError('Wrong before state accepted')
            failure = json.loads((root/'bad/failure.json').read_text())
            assert failure['counts']['latent_backwards_attempted'] == 0
            assert len(calls) == 1 and not (root/'bad/destination_report.json').exists()
    print('PASS: exactly 3 captures / 2 backwards, independent branches, old-objective reporting, repeat and replay-failure guards.')


def input_checks():
    before = np.zeros((5, 1560, 2), np.float32)
    mask = np.ones(before.shape[:-1], bool)
    refs = {a:(before.copy(), mask.copy()) for a in candidate.ARMS}
    inputs = ({}, {'before':np.zeros((1,), np.float32)}, refs, {}, {'saved':'hash'}, {'before':before})
    hashes = {'gradient':'hash'}
    checked = (inputs, {}, {}, hashes)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        sources = {n:candidate.digest(Path(candidate.__file__).with_name(n)) for n in
                   ('wan_centered_gradients.py','wan_centered_huber.py','wan_centered_attribution.py','wan_centered_pilot.py')}
        protocol = dict(inputs_sha256=inputs[4], gradient_artifacts_sha256=hashes,
                        helper_sources_sha256=sources,
                        script_sha256=candidate.digest(Path(candidate.__file__).with_name('wan_centered_response.py')))
        report = dict(baseline_unchanged=True,
                      counts=dict(positive_forwards_attempted=4, positive_forwards_completed=4), conditions={})
        for a in candidate.ARMS:
            folder = root/(a+'_full'); folder.mkdir()
            np.savez_compressed(folder/'flow.npz', flow=before)
            report['conditions'][a+'_full'] = {'losses':{a:{'after':candidate.scores(before, *refs[a])}}}
        candidate.write_json(root/'started.json', protocol)
        candidate.write_json(root/'response_report.json', report)
        with patch.object(candidate, 'prepare_response_inputs', return_value=checked):
            ready = candidate.prepare_destination_inputs('p','a','h','g',root)
            assert len(ready[5]['response_artifacts']) == 4
            assert ready[-1]['common_torso_different_destinations'] == 0
            np.savez_compressed(root/'reverse_full/flow.npz', flow=before+1)
            try:
                candidate.prepare_destination_inputs('p','a','h','g',root)
            except ValueError as exc:
                assert 'report disagree' in str(exc)
            else:
                raise AssertionError('Changed control field accepted')
    print('PASS: completed-control provenance, geometry coverage, and control-field/report consistency.')


if __name__ == '__main__':
    geometry_checks()
    objective_checks()
    input_checks()
    runner_checks()
