"""CPU gates for identical readout algebra, gradient endpoints and the one-capture budget."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from benchmark import wan_centered_qk as experiment
from benchmark.wan_centered_destination import destination_objectives
from guidance_utils.wan_centered_amf import centered_pair_flow


def fixture(dtype=torch.float64):
    torch.manual_seed(18)
    q=torch.randn(6,4,2,4,dtype=dtype).requires_grad_(True)
    k=torch.randn_like(q).requires_grad_(True)
    support=np.ones((5,4),bool);support[:,3]=False
    ids={a:np.tile((np.arange(4)+shift)%4,(5,1)).copy() for a,shift in [('forward',1),('reverse',-1)]}
    refs={a:(np.zeros((5,4,2),np.float32),support.copy()) for a in ids}
    return q,k,refs,ids,support


def algebra_checks():
    for dtype in (torch.float64,torch.bfloat16):
        q,k,refs,ids,_=fixture(dtype)
        old,old_logp=destination_objectives(q,k,refs,ids,1,4)
        new,new_logp,centered,raw=experiment.readout_with_taps(q,k,refs,ids,1,4)
        for a in experiment.ARMS:
            torch.testing.assert_close(new[a],old[a],rtol=0,atol=0)
            np.testing.assert_array_equal(new_logp[a],old_logp[a])
            expected=torch.autograd.grad(old[a],(q,k),retain_graph=True)
            values=torch.autograd.grad(new[a],(q,k,*centered,*raw),retain_graph=True)
            for actual,wanted in zip(values[:2],expected):
                torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
            assert torch.count_nonzero(values[0][-1])==0 and torch.count_nonzero(values[1][0])==0
            for c,r in zip(values[2:7],values[7:]):
                torch.testing.assert_close(r,c-c.mean(0,keepdim=True),rtol=1e-5,atol=1e-7)
                assert torch.count_nonzero(c[3])==0  # masked loss source
                assert torch.count_nonzero(r[3])>0   # still in centering
        assert q.grad is None and k.grad is None
    rng=np.random.default_rng(5)
    a,b=rng.normal(size=(6,4,2,4)),rng.normal(size=(6,4,2,4))
    m=experiment.moments(a,b,axes=(1,3))
    measured=experiment.summarize_moments(m)
    expected=experiment.compare(a,b)
    for key,value in measured.items():
        np.testing.assert_allclose(value,expected[key],rtol=1e-12,atol=1e-12)
    print('PASS: FP64/BF16 frozen loss and Q/K gradients exactly match; extra endpoints preserve derivatives; full centering; compact statistics reconstruct comparisons.')


def runner_checks():
    q,k,refs,ids,support=fixture(torch.float32)
    q0,k0=q.detach(),k.detach()
    with torch.no_grad():
        before=torch.stack([centered_pair_flow(q0[i],k0[i+1],1,4) for i in range(5)]).numpy()
    loss,logs,_,_=experiment.readout_with_taps(q,k,refs,ids,1,4)
    prefix=np.zeros((1,2,6,2,2),np.float32)
    inputs=({'experiment':{'guidance_sigma':.45946}},{'before':prefix},refs,{}, {},{'before':before})
    prepared=dict(inputs=inputs,refs=refs,support=support,indices=ids,before_logp=logs,
                  latent_comparison={'cosine':.975},geometry={},provenance={})
    model=SimpleNamespace(device='cpu',dtype=torch.float32,_clear_kv=Mock())
    calls=[]
    def capture(g,x):
        assert not torch.is_grad_enabled() and not x.requires_grad
        calls.append(x.clone())
        return torch.from_numpy(before),q0,k0
    native_readout=experiment.readout_with_taps
    def tapped(q,k,refs,ids):
        assert q.is_leaf and k.is_leaf and q.grad_fn is None and k.grad_fn is None
        return native_readout(q,k,refs,ids,1,4)
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        with patch.object(experiment,'prepare_qk_inputs',return_value=prepared), \
             patch.object(experiment,'frozen_model',return_value={}), \
             patch.object(experiment,'capture',side_effect=capture), \
             patch.object(experiment,'readout_with_taps',side_effect=tapped), \
             patch('benchmark.wan_centered_pilot.clear'):
            args=(model,*[root/name for name in ('p','a','h','g','r','d','t')])
            result=experiment.run_qk(*args,root/'out')
            assert len(calls)==1
            assert result['counts']==dict(positive_forwards_attempted=1,positive_forwards_completed=1,
                readout_backwards_attempted=2,readout_backwards_completed=2)
            assert result['baseline_unchanged'] and not result['latent_graph_created'] and not result['port_success']
            with np.load(root/'out/gradient_moments.npz') as arrays:
                assert arrays['Q'].shape==(6,2,3) and arrays['raw_scores'].shape==(5,3)
                for key in arrays.files:
                    assert np.isfinite(arrays[key]).all()
                assert arrays['Q'][-1].sum()==0 and arrays['K'][0].sum()==0
            assert len(list((root/'out').glob('*.npz')))==4
            json.dumps(result,allow_nan=False)
            try:
                experiment.run_qk(*args,root/'out')
            except RuntimeError as exc:
                assert 'budget already' in str(exc)
            else:
                raise AssertionError('Repeated capture budget accepted')
            assert len(calls)==1
            bad={**prepared,'before_logp':{a:v+1 for a,v in logs.items()}}
            with patch.object(experiment,'prepare_qk_inputs',return_value=bad):
                try:
                    experiment.run_qk(*args,root/'bad')
                except RuntimeError as exc:
                    assert 'do not replay' in str(exc)
                else:
                    raise AssertionError('Mismatched readout accepted')
            failed=json.loads((root/'bad/failure.json').read_text())
            assert failed['counts']['readout_backwards_attempted']==0
    print('PASS: one no-grad capture, detached Q/K, two readout-only backwards, compact exports, unused-frame zeros, replay/repeat guards.')


def preflight_checks():
    from benchmark.wan_centered_attribution import regions
    support=np.zeros((5,1560),bool)
    roi=np.flatnonzero(regions()['torso'])
    for i,n in enumerate([70,31,101,26,96]):
        support[i,roi[:n]]=True
    before=np.zeros((5,1560,2),np.float32)
    prefix=np.zeros((1,2,6,2,2),np.float32)
    refs={a:(before.copy(),support.copy()) for a in experiment.ARMS}
    inputs=({}, {'before':prefix},refs,{}, {'input':'hash'}, {'before':before})
    provenance={'earlier':'hash'}
    previous=(inputs,{}, {},{}, {},provenance,{})
    saved={a:np.full_like(prefix,v) for a,v in [('forward',1.),('reverse',2.)]}
    logs={a:np.full(support.shape,-2.,np.float32) for a in experiment.ARMS}
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        source=Path(experiment.__file__).parent
        protocol=dict(inputs_sha256=inputs[4],control_sha256=provenance,
            script_sha256=experiment.digest(source/'wan_centered_torso.py'),helper_sources_sha256={})
        report=dict(baseline_unchanged=True,counts=dict(positive_forwards_attempted=3,positive_forwards_completed=3,
            latent_backwards_attempted=2,latent_backwards_completed=2,first_adam_proposals=2),
            before_optimization_nll=experiment.support_nll(logs,support),
            latent_gradient_comparison=experiment.compare(saved['forward'],saved['reverse']))
        experiment.write_json(root/'started.json',protocol)
        experiment.write_json(root/'torso_report.json',report)
        np.savez_compressed(root/'before_flow.npz',flow=before)
        np.savez_compressed(root/'before_target_log_probability.npz',**logs)
        np.savez_compressed(root/'optimization_support.npz',common_torso=support,
                            forward_original_valid=support,reverse_original_valid=support)
        for a,v in saved.items():
            np.savez_compressed(root/f'{a}_latent_gradient.npz',gradient=v)
        with patch.object(experiment,'prepare_torso_inputs',return_value=previous):
            ready=experiment.prepare_qk_inputs('p','a','h','g','r','d',root)
            assert len(ready['provenance']['torso_artifacts'])==7
            assert ready['support'].sum()==324 and ready['latent_comparison']['cosine']==1
            np.savez_compressed(root/'reverse_latent_gradient.npz',gradient=saved['reverse']*2)
            try:
                experiment.prepare_qk_inputs('p','a','h','g','r','d',root)
            except ValueError as exc:
                assert 'gradient comparison disagrees' in str(exc)
            else:
                raise AssertionError('Changed archived gradient accepted')
    print('PASS: completed torso source/input/support/NLL preflight, archived latent-gradient consistency and corruption guard.')


if __name__=='__main__':
    algebra_checks()
    preflight_checks()
    runner_checks()
