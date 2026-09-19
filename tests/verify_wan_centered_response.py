"""CPU checks for saved-gradient dose construction and four-capture orchestration."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from benchmark import wan_centered_response as response
from benchmark.wan_centered_huber import scores


def proposal_checks():
    torch.manual_seed(5)
    gradient = torch.randn(300,dtype=torch.float32)
    gradient[:4] = torch.tensor([0.,1e-12,-1e-9,1e-8])
    x = torch.zeros_like(gradient,requires_grad=True)
    opt = torch.optim.Adam([x],lr=.001)
    x.grad = gradient.clone()
    opt.step()
    torch.testing.assert_close(response.first_adam_proposal(gradient),x.detach(),atol=2e-10,rtol=1e-6)
    # Nonzero FP32 proposals need not survive BF16 addition at a particular prefix.
    prefix = torch.ones_like(gradient)
    small = prefix + .1*response.first_adam_proposal(gradient)
    assert (small!=prefix).any()
    assert torch.equal(small.to(torch.bfloat16),prefix.to(torch.bfloat16))
    print('PASS: first Adam proposal matches real optimizer, including epsilon behavior; explicit cast-erasure example.')


def runner_checks():
    shape = (1,2,6,2,2)
    prefix = np.zeros(shape,np.float32)
    before = np.full((5,1560,2),.05,np.float32)
    refs = {a:(np.full_like(before,v),np.ones(before.shape[:-1],bool)) for a,v in [('forward',.25),('reverse',-.25)]}
    # For f(x)=.05+mean(x), these are the exact starting Huber derivatives.
    grads = {a:np.full(shape,(.05-v)/np.prod(shape),np.float32) for a,v in [('forward',.25),('reverse',-.25)]}
    before_readout = dict(same_probability=np.full(before.shape[:-1],.999,np.float32),
        flow_jacobian_frobenius=np.full(before.shape[:-1],.001,np.float32))
    inputs = ({'experiment':{'guidance_sigma':.45946}}, {'before':prefix}, refs, {}, {}, {'before':before})
    g = SimpleNamespace(device='cpu',dtype=torch.bfloat16,_clear_kv=Mock())
    calls = []
    def capture(g,x):
        assert not torch.is_grad_enabled() and not x.requires_grad
        calls.append(x.detach().clone())
        return torch.from_numpy(before)+x.mean(), x, None
    def analyse(q,k,refs,out):
        flow = before+float(q.mean())
        fields = dict(before_readout,flow=flow)
        np.savez_compressed(Path(out)/'readout_derivatives.npz',**fields)
        return {},flow
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with patch.object(response,'prepare_response_inputs',return_value=(inputs,grads,before_readout,{})), \
             patch.object(response,'frozen_model',return_value={}), \
             patch.object(response,'capture',side_effect=capture), \
             patch.object(response,'readout_analysis',side_effect=analyse), \
             patch('benchmark.wan_centered_pilot.clear') as cleanup:
            args = (g,root/'pilot',root/'attribution',root/'huber',root/'gradients')
            report = response.run_response(*args,root/'out')
            assert len(calls)==4 and report['counts']['positive_forwards_completed']==4
            assert report['baseline_unchanged'] and not report['port_success']
            np.testing.assert_array_equal(prefix,np.zeros(shape,np.float32))
            for arm,offset in [('forward',0),('reverse',2)]:
                proposed = response.first_adam_proposal(torch.from_numpy(grads[arm]))
                for j,(label,scale) in enumerate(response.DOSES):
                    torch.testing.assert_close(calls[offset+j],scale*proposed,rtol=0,atol=0)
                    row = report['conditions'][arm+'_'+label]
                    with np.load(root/'out'/(arm+'_'+label)/'flow.npz') as data:
                        for objective in refs:
                            assert row['losses'][objective]['after'] == scores(data['flow'],*refs[objective])
                    # Squared-region Huber has an exact second-order remainder for this analytic readout.
                    residual = row['losses'][arm]['linear_prediction_error']
                    expected = .5*float(calls[offset+j].mean())**2
                    np.testing.assert_allclose(residual,expected,atol=3e-9)
                    assert row['torso']['before_sharp']['count']>0
            assert report['by_dose']['full']['separation']['all_common']['projection_gain']>0
            assert len(list((root/'out').glob('*/actual_delta.npz')))==4
            assert cleanup.call_count==5
            try:
                response.run_response(*args,root/'out')
            except RuntimeError as exc:
                assert 'budget already' in str(exc)
            else:
                raise AssertionError('Repeat budget accepted')
            assert len(calls)==4
            def bad_capture(g,x):
                f,q,k = capture(g,x)
                return f*float('nan'),q,k
            with patch.object(response,'capture',side_effect=bad_capture):
                try:
                    response.run_response(*args,root/'bad')
                except RuntimeError as exc:
                    assert 'Nonfinite response flow' in str(exc)
                else:
                    raise AssertionError('Nonfinite capture accepted')
            assert len(calls)==5
            assert (root/'bad/failure.json').is_file()
            assert not (root/'bad/response_report.json').exists()
            json.dumps(report,allow_nan=False)
    print('PASS: four fresh conditions, exact saved-gradient directions, own/cross losses, linearity accounting, reports, repeat and failure guards.')


def input_checks():
    shape=(1,2,6,2,2)
    before=np.zeros((5,1560,2),np.float32)
    refs={a:(before.copy(),np.ones(before.shape[:-1],bool)) for a in ('forward','reverse')}
    inputs=({}, {'before':np.zeros(shape,np.float32)},refs,{}, {'fixture':'hash'}, {'before':before})
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp); huber=root/'huber'; huber.mkdir()
        (huber/'huber_report.json').write_text('{}')
        saved=root/'gradients'; saved.mkdir()
        gradients={a:np.full(shape,v,np.float32) for a,v in [('forward',.1),('reverse',-.2)]}
        np.savez_compressed(saved/'before_flow.npz',flow=before)
        fields=dict(flow=before,same_probability=np.ones(before.shape[:-1],np.float32),
                    flow_jacobian_frobenius=np.ones(before.shape[:-1],np.float32),
                    forward_valid=refs['forward'][1],reverse_valid=refs['reverse'][1])
        np.savez_compressed(saved/'readout_derivatives.npz',**fields)
        for a,g in gradients.items():
            np.savez_compressed(saved/f'{a}_latent_gradient.npz',gradient=g)
        protocol=dict(inputs_sha256=inputs[4],huber_report_sha256=response.digest(huber/'huber_report.json'),
            helper_sources_sha256={n:response.digest(Path(response.__file__).with_name(n)) for n in
                ('wan_centered_huber.py','wan_centered_attribution.py','wan_centered_pilot.py')},
            script_sha256=response.digest(Path(response.__file__).with_name('wan_centered_gradients.py')))
        report=dict(latent_unchanged=True,archived_gradient_norms_match=True,
            counts=dict(positive_forwards_attempted=1,latent_backwards_attempted=2,latent_backwards_completed=2),
            latent_gradient_comparison={a+'_norm':float(np.linalg.norm(g.astype(float))) for a,g in gradients.items()})
        (saved/'started.json').write_text(json.dumps(protocol))
        (saved/'gradient_report.json').write_text(json.dumps(report))
        with patch.object(response,'prepare_gradient_inputs',return_value=(inputs,{})):
            checked=response.prepare_response_inputs(root/'pilot',root/'attribution',huber,saved)
            assert len(checked[3])==6
            fields['same_probability'][0,0]=2.
            np.savez_compressed(saved/'readout_derivatives.npz',**fields)
            try:
                response.prepare_response_inputs(root/'pilot',root/'attribution',huber,saved)
            except ValueError as exc:
                assert 'readout probability' in str(exc)
            else:
                raise AssertionError('Corrupt saved probability accepted')
    print('PASS: source/input provenance, saved gradient norms, before arrays, and invalid probability preflight guard.')


if __name__=='__main__':
    proposal_checks()
    input_checks()
    runner_checks()
