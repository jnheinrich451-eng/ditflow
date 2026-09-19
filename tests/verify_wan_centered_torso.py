"""CPU checks for frozen common support, control provenance and bounded execution."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from benchmark import wan_centered_torso as experiment


def fixture():
    before = np.full((5, 1560, 2), .05, np.float32)
    roi = np.flatnonzero(experiment.regions()['torso'])
    support = np.zeros((5, 1560), bool)
    for i,n in enumerate([70,31,101,26,96]):
        support[i,roi[:n]] = True
    assert support[2,12*52+33]
    refs = {}
    for a,v,extra in [('forward',1.,100),('reverse',-1.,200)]:
        target = np.zeros_like(before); target[:,:,0] = v
        mask = support.copy(); mask[:,extra] = True
        refs[a] = (target, mask)
    base = dict(flow=before, same_probability=np.full(support.shape,.999,np.float32),
                flow_jacobian_frobenius=np.full(support.shape,.001,np.float32))
    return before, refs, base, support


def support_checks():
    _, refs, _, expected = fixture()
    masks = {a:m.copy() for a,(_,m) in refs.items()}
    selected, support = experiment.common_torso_refs(refs)
    np.testing.assert_array_equal(support,expected)
    assert support.sum() == 324 and support[2,12*52+33]
    for a in refs:
        assert selected[a][0] is refs[a][0]
        np.testing.assert_array_equal(selected[a][1],expected)
        np.testing.assert_array_equal(refs[a][1],masks[a])
    bad = {a:(f,m.copy()) for a,(f,m) in refs.items()}
    bad['forward'][1][2,12*52+33] = False
    try:
        experiment.common_torso_refs(bad)
    except ValueError as exc:
        assert '324-query' in str(exc)
    else:
        raise AssertionError('Changed support/outlier removal accepted')
    print('PASS: exactly shared torso support, fixed per-pair counts, original masks/targets preserved, outlier retained.')


def control_fixture(root, inputs, base, provenance):
    before, refs = inputs[5]['before'], inputs[2]
    logs = {a:np.full(before.shape[:-1], -2., np.float32) for a in refs}
    summary = experiment.summarize_target_probability(logs, refs, base)
    report = dict(baseline_unchanged=True,
        counts=dict(positive_forwards_attempted=3,positive_forwards_completed=3,
                    latent_backwards_attempted=2,latent_backwards_completed=2,first_adam_proposals=2),
        latent_gradient_comparison={},fp32_update_comparison={},cast_update_comparison={},arms={})
    for a in refs:
        row = dict(gradient_norm=1.,delta_fp32={},delta_after_cast={},target_probability=summary,
            original_objectives={b:dict(before=experiment.scores(before,*refs[b]),
                                        after=experiment.scores(before,*refs[b])) for b in refs},
            torso={},torso_scores_by_pair=[])
        report['arms'][a] = row
        folder = root/a; folder.mkdir()
        np.savez_compressed(folder/'flow.npz',flow=before)
        np.savez_compressed(folder/'target_log_probability.npz',**logs)
    np.savez_compressed(root/'before_flow.npz',flow=before)
    np.savez_compressed(root/'before_target_log_probability.npz',**logs)
    source = Path(experiment.__file__).parent
    protocol = dict(inputs_sha256=inputs[4],control_sha256=provenance,
        helper_sources_sha256={'wan_centered_response.py':experiment.digest(source/'wan_centered_response.py')},
        script_sha256=experiment.digest(source/'wan_centered_destination.py'))
    experiment.write_json(root/'started.json',protocol)
    experiment.write_json(root/'destination_report.json',report)
    return report


def preflight_checks():
    before,refs,base,_ = fixture()
    inputs = ({}, {'before':np.zeros((1,),np.float32)}, refs, {}, {'input':'hash'}, {'before':before})
    previous = (inputs,base,{}, {}, {}, {'response':'hash'}, {})
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        control_fixture(root,inputs,base,previous[5])
        with patch.object(experiment,'prepare_destination_inputs',return_value=previous):
            checked = experiment.prepare_torso_inputs('p','a','h','g','r',root)
            assert checked[-1]['optimization_support']['count']==324
            assert len(checked[5]['destination_artifacts'])==8
            np.savez_compressed(root/'forward/flow.npz',flow=before+1)
            try:
                experiment.prepare_torso_inputs('p','a','h','g','r',root)
            except ValueError as exc:
                assert 'reported errors disagree' in str(exc)
            else:
                raise AssertionError('Changed control accepted')
    print('PASS: completed full-field control hashes, arrays, probabilities and support preflight; corrupt control rejected.')


def runner_checks():
    before,refs,base,support = fixture()
    prefix = np.zeros((1,2,6,2,2),np.float32)
    inputs = ({'experiment':{'guidance_sigma':.45946}}, {'before':prefix}, refs, {}, {}, {'before':before})
    g = SimpleNamespace(device='cpu',dtype=torch.bfloat16,_clear_kv=Mock())
    calls, evaluated = [], []
    def capture(g,x):
        calls.append(x.detach().clone())
        assert x.requires_grad == (len(calls)==1)
        return torch.from_numpy(before)+x.mean(), x, None
    def objective(q,k,selected,indices):
        for a in refs:
            assert selected[a][0] is refs[a][0]
            np.testing.assert_array_equal(selected[a][1],support)
        loss = {a:(q.mean()-v)**2+1 for a,v in [('forward',.25),('reverse',-.25)]}
        logs = {a:np.full(support.shape,-float(v.detach())) for a,v in loss.items()}
        return loss,logs
    def analysis(q,k,evaluation_refs,out):
        for a in refs:
            np.testing.assert_array_equal(evaluation_refs[a][1],refs[a][1])
        evaluated.append(True)
        flow = before+float(q.mean())
        np.savez_compressed(Path(out)/'readout_derivatives.npz',**{**base,'flow':flow})
        return {},flow
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        control = root/'control'; control.mkdir()
        full = control_fixture(control,inputs,base,{})
        checked = (inputs,base,{}, {a:before for a in refs}, full, {}, {})
        with patch.object(experiment,'prepare_torso_inputs',return_value=checked), \
             patch.object(experiment,'frozen_model',return_value={}), \
             patch.object(experiment,'capture',side_effect=capture), \
             patch.object(experiment,'destination_objectives',side_effect=objective), \
             patch.object(experiment,'readout_analysis',side_effect=analysis), \
             patch('benchmark.wan_centered_pilot.clear'):
            args = (g,root/'p',root/'a',root/'h',root/'g',root/'r',control)
            report = experiment.run_torso(*args,root/'out')
            assert report['counts']==full['counts'] and len(calls)==3 and len(evaluated)==2
            assert report['baseline_unchanged'] and not report['port_success']
            assert report['optimization_support']['count']==324
            assert calls[1].mean()>0 and calls[2].mean()<0
            torch.testing.assert_close(calls[1],-calls[2])
            assert report['before_target_probability']['forward']['all_valid']['count']==329
            with np.load(root/'out/optimization_support.npz') as values:
                np.testing.assert_array_equal(values['common_torso'],support)
                np.testing.assert_array_equal(values['forward_original_valid'],refs['forward'][1])
            assert (root/'out/torso_report.json').is_file()
            assert not (root/'out/destination_report.json').exists()
            json.dumps(report,allow_nan=False)
            try:
                experiment.run_torso(*args,root/'out')
            except RuntimeError as exc:
                assert 'budget already' in str(exc)
            else:
                raise AssertionError('Repeated budget accepted')
            assert len(calls)==3
            # A bad before state must stop before spending either backward.
            calls.clear()
            bad_inputs = (*inputs[:5],{'before':before+1})
            with patch.object(experiment,'prepare_torso_inputs',return_value=(bad_inputs,*checked[1:])):
                try:
                    experiment.run_torso(*args,root/'bad')
                except RuntimeError as exc:
                    assert 'does not replay' in str(exc)
                else:
                    raise AssertionError('Bad before state accepted')
            failed = json.loads((root/'bad/failure.json').read_text())
            assert failed['counts']['latent_backwards_attempted']==0 and len(calls)==1
    print('PASS: three captures/two backwards, loss-only masking, full evaluation, independent updates, saved support, repeat/replay guards.')


if __name__=='__main__':
    support_checks()
    preflight_checks()
    runner_checks()
