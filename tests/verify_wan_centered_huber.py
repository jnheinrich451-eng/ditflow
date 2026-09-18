"""Check fixed Huber semantics and masked gradients; no model quality claim."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.wan_centered_huber import huber_loss, scores, probability_summary
from guidance_utils.wan_centered_amf import centered_pair_flow


flow = torch.tensor([[[.5, -2.], [21.5, 0.], [999., 999.]]], requires_grad=True)
target = torch.zeros_like(flow)
valid = torch.tensor([[True, True, False]])
loss = huber_loss(flow, target, valid)
# Four included coordinates: .125, 1.5, 21, 0. The excluded large values contribute zero.
torch.testing.assert_close(loss, torch.tensor(22.625/4))
loss.backward()
torch.testing.assert_close(flow.grad, torch.tensor([[[.125, -.25], [.25, 0.], [0., 0.]]]))
offline = scores(flow.detach().numpy(), target.detach().numpy(), valid.numpy())
np.testing.assert_allclose(offline['huber_delta1'], float(loss.detach()), atol=1e-7)
np.testing.assert_allclose(offline['mse'], (.25+4+21.5**2)/4)
try:
    huber_loss(flow, target, torch.zeros_like(valid))
except ValueError:
    pass
else:
    raise AssertionError('Empty mask accepted')
print('PASS: delta=1 standard coordinatewise Huber, fixed mask, capped residual gradients, offline scores.')

# Zero expected displacement can result from a broad distribution OR a sharp same-location peak.
uniform = torch.zeros((9, 1, 9), dtype=torch.float64)
sharp = torch.eye(9, dtype=torch.float64).reshape(9, 1, 9)*3
broad_log = probability_summary(uniform, uniform, 3, 3, [4])[0]
sharp_log = probability_summary(sharp, sharp, 3, 3, [4])[0]
np.testing.assert_allclose(broad_log['expected_flow_xy'], [0, 0], atol=1e-12)
np.testing.assert_allclose(sharp_log['expected_flow_xy'], [0, 0], atol=1e-12)
np.testing.assert_allclose(broad_log['entropy_nats'], np.log(9), atol=1e-12)
assert sharp_log['entropy_nats'] < 1e-5
assert sharp_log['same_spatial_probability'] > .999999

# Detached instrumentation must agree with the preserved readout and leave its gradients unchanged.
generator = torch.Generator().manual_seed(27)
q = torch.randn((9, 2, 3), generator=generator, dtype=torch.float64, requires_grad=True)
k = torch.randn((9, 2, 3), generator=generator, dtype=torch.float64, requires_grad=True)
f = centered_pair_flow(q,k,3,3)
reference_gradients = torch.autograd.grad(f.square().mean(), (q,k), retain_graph=True)
logged = probability_summary(q,k,3,3,[0,4,8])
np.testing.assert_allclose([v['expected_flow_xy'] for v in logged], f[[0,4,8]].detach().numpy(), atol=1e-12)
actual_gradients = torch.autograd.grad(f.square().mean(), (q,k))
for actual, expected in zip(actual_gradients, reference_gradients):
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)
print('PASS: broad versus sharp zero-flow distributions, logger/readout agreement, unchanged gradients.')

# Exercise the real runner/report loop with an analytic differentiable readout.
# Transformer capture and provenance are mocked; this is NOT pretrained evidence.
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from benchmark import wan_centered_huber as runner


def check_runner():
    shape = (5, 1560, 2)
    before = np.full(shape, .05, np.float32)
    masks = {a: np.ones(shape[:-1], bool) for a in ('forward', 'reverse')}
    masks['forward'][:, :5] = False
    masks['reverse'][:, -5:] = False
    refs = {a: (np.full(shape, sign*.25, np.float32), masks[a])
            for a, sign in (('forward', 1), ('reverse', -1))}
    traces = {a: {'losses_before_updates': [scores(before, *refs[a])['mse']]} for a in refs}
    archived = {a: before.copy() for a in ('before', 'forward', 'reverse')}
    inputs = ({'experiment': {'guidance_sigma': .4594605863}}, {'before': before},
              refs, traces, {}, archived)
    g = SimpleNamespace(device='cpu', dtype=torch.bfloat16, _clear_kv=Mock())
    calls, backwards = [], []

    def readout(g, x, log_probability=False):
        assert log_probability
        calls.append(x.detach().clone())
        if torch.is_grad_enabled():
            flow = x*1.
            flow.register_hook(lambda grad: backwards.append(grad.detach().clone()))
        else:
            flow = x.detach().clone()
        return flow, []

    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        attribution = root/'attribution'
        attribution.mkdir()
        (attribution/'attribution.json').write_text('{}')
        for name, field in archived.items():
            np.savez_compressed(attribution/f'{name}_flow.npz', flow=field)
        with patch.object(runner, 'prepare_inputs', return_value=inputs), \
             patch.object(runner, 'frozen_model', return_value={}), \
             patch.object(runner, 'target_flow', side_effect=readout), \
             patch('benchmark.wan_centered_pilot.clear') as cleanup:
            output = root/'run'
            report = runner.run_huber(g, root/'pilot', attribution, output)
            assert len(calls) == 12 and len(backwards) == 10
            torch.testing.assert_close(calls[0], calls[6], rtol=0, atol=0)
            np.testing.assert_array_equal(before, np.full(shape, .05, np.float32))
            assert not report['port_success']
            assert [s['after_updates'] for s in report['separation_by_updates']] == list(range(6))
            assert report['separation_by_updates'][0]['all_common']['observed_rms'] == 0
            assert report['separation_by_updates'][-1]['all_common']['projection_gain'] > 0
            for arm, sign in (('forward', 1), ('reverse', -1)):
                entries = report['arms'][arm]
                assert len(entries['history']) == 5 and len(entries['readout_states']) == 6
                assert len(list((output/arm).glob('flow_state_*.npz'))) == 6
                for entry in entries['history']:
                    assert entry['actual_update_fp32']['rms'] > 0
                    assert entry['actual_update_after_cast']['rms'] > 0
                for j, state in enumerate(entries['readout_states']):
                    assert state['after_updates'] == j
                    with np.load(output/arm/f'flow_state_{j:02d}.npz') as stored:
                        assert state['scores'] == scores(stored['flow'], *refs[arm])
                        pair_scores = state['torso_scores_by_pair']
                        assert len(pair_scores) == 5
                        for i, pair in enumerate(pair_scores):
                            support = masks['forward'][i] & masks['reverse'][i] & runner.regions()['torso']
                            assert pair['common_valid']['count'] == int(support.sum())
                            assert pair['common_valid']['scores'] == scores(stored['flow'][i], refs[arm][0][i], support)
                saved = torch.load(output/arm/'optimized_latent.pt', weights_only=True).numpy()
                assert ((saved-before)[masks[arm]]*sign > 0).all()
                np.testing.assert_array_equal(saved[~masks[arm]], before[~masks[arm]])
                assert entries['scores']['huber_after']['huber_delta1'] < entries['scores']['before']['huber_delta1']
            # All report numbers must be standard JSON, not NaN/Infinity.
            json.dumps(report, allow_nan=False)
            cleanup.assert_called_once_with(g)
            try:
                runner.run_huber(g, root/'pilot', attribution, output)
            except FileExistsError:
                pass
            else:
                raise AssertionError('A repeated experiment consumed another budget')
            assert len(calls) == 12

            # A discrepant starting readout must stop BEFORE any optimizer update.
            bad_traces = {a: {'losses_before_updates': [999.]} for a in refs}
            bad_inputs = (*inputs[:3], bad_traces, *inputs[4:])
            with patch.object(runner, 'prepare_inputs', return_value=bad_inputs):
                try:
                    runner.run_huber(g, root/'pilot', attribution, root/'bad_replay')
                except RuntimeError as exc:
                    assert 'Starting MSE' in str(exc)
                else:
                    raise AssertionError('Discrepant replay accepted')
            assert len(calls) == 13 and len(backwards) == 10
            assert (root/'bad_replay/forward/replay_failure.json').is_file()
            assert not (root/'bad_replay/huber_report.json').exists()

            # Failure at the final read must not produce a completed report.
            def nonfinite_final(g, x, log_probability=False):
                flow, probability = readout(g, x, log_probability)
                return (flow if torch.is_grad_enabled() else flow*float('nan')), probability

            with patch.object(runner, 'target_flow', side_effect=nonfinite_final):
                try:
                    runner.run_huber(g, root/'pilot', attribution, root/'bad_final')
                except RuntimeError as exc:
                    assert 'Nonfinite final flow' in str(exc)
                else:
                    raise AssertionError('Nonfinite final read accepted')
            assert len(calls) == 19 and len(backwards) == 15
            assert not (root/'bad_final/huber_report.json').exists()
            assert (root/'bad_final/forward/readout_states.json').is_file()
            assert not (root/'bad_final/forward/optimized_latent.pt').exists()
            assert cleanup.call_count == 3


check_runner()
print('PASS: synthetic runner integration: independent branches, 12/10 call budget, six snapshots, '
      'masked updates, report agreement, repeat/replay/nonfinite guards. No pretrained motion claim.')
