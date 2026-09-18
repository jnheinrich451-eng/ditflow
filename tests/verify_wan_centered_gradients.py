"""Analytic derivative and retained checkpoint-graph checks; not motion evidence."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from benchmark import wan_centered_gradients as diagnostic
from benchmark.wan_centered_huber import huber_loss, scores
from guidance_utils.wan_centered_amf import centered_pair_flow


def analytic_checks():
    torch.manual_seed(12)
    n = 6
    logits = (torch.randn(n,n,dtype=torch.float64)*.2).requires_grad_()
    targets = {'forward':torch.randn(n,2,dtype=torch.float64), 'reverse':torch.randn(n,2,dtype=torch.float64)}
    masks = {'forward':torch.tensor([True,False,True,False,True,False]),
             'reverse':torch.tensor([False,True,True,True,False,False])}
    counts = {a:int(m.sum()) for a,m in masks.items()}
    coords = torch.tensor([[0,0],[1,0],[2,0],[0,1],[1,1],[2,1]],dtype=torch.float64)
    def flow(s):
        return ((s-s.mean(0,keepdim=True))*8).softmax(-1)@coords-coords
    fields, gradients = diagnostic.pair_derivatives(logits,2,3,targets,masks,counts)
    torch.testing.assert_close(fields['flow'], flow(logits), atol=1e-12,rtol=1e-12)
    jac = torch.autograd.functional.jacobian(flow,logits).reshape(n,2,-1)
    torch.testing.assert_close(fields['flow_jacobian_frobenius'],jac.square().sum((1,2)).sqrt(),atol=1e-12,rtol=1e-12)
    for arm in targets:
        real = torch.autograd.grad(huber_loss(flow(logits),targets[arm],masks[arm]),logits)[0]
        torch.testing.assert_close(real,gradients[arm]['raw_scores'],atol=1e-12,rtol=1e-12)
        assert real[~masks[arm]].norm() > 0  # All-source centering couples excluded queries.
        torch.testing.assert_close(real.sum(0),torch.zeros(n,dtype=torch.float64),atol=1e-12,rtol=0)
    broad, _ = diagnostic.pair_derivatives(torch.zeros(n,n,dtype=torch.float64),2,3,targets,masks,counts)
    sharp, _ = diagnostic.pair_derivatives(torch.eye(n,dtype=torch.float64)*10,2,3,targets,masks,counts)
    assert sharp['flow_jacobian_frobenius'].max() < broad['flow_jacobian_frobenius'].min()*1e-10
    equal = diagnostic.compare(np.arange(5),np.arange(5))
    assert equal['cosine'] == 1 and equal['differential_to_shared_ratio'] == 0
    opposite = diagnostic.compare(np.arange(5),-np.arange(5))
    assert opposite['cosine'] == -1 and opposite['differential_to_shared_ratio'] is None
    print('PASS: exact full centering Jacobian, analytic Huber derivatives, excluded-query coupling, gradient comparisons.')


def runner_checks():
    shape = (1,2,6,2,2)
    latent = np.zeros(shape,np.float32)
    before = np.full((5,1560,2),.05,np.float32)
    refs = {a:(np.full_like(before,v),np.ones(before.shape[:-1],bool)) for a,v in [('forward',.25),('reverse',-.25)]}
    saved_report = {'arms':{a:{'readout_states':[{'scores':scores(before,*refs[a])}],
        'history':[{'gradient_norm':1.}]} for a in refs}}
    inputs = ({'experiment':{'guidance_sigma':.45946}}, {'before':latent}, refs, {}, {}, {'before':before})
    g = SimpleNamespace(device='cpu',_clear_kv=Mock())
    calls, back = [], []
    def capture(g,x):
        calls.append(1)
        flow = torch.from_numpy(before)+x.mean()
        flow.register_hook(lambda grad:back.append(grad.clone()))
        return flow, None, None
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        huber = root/'huber'; huber.mkdir()
        (huber/'huber_report.json').write_text('{}')
        with patch.object(diagnostic,'prepare_gradient_inputs',return_value=(inputs,saved_report)), \
             patch.object(diagnostic,'frozen_model',return_value={}), \
             patch.object(diagnostic,'capture',side_effect=capture), \
             patch.object(diagnostic,'readout_analysis',return_value=({},before.copy())), \
             patch('benchmark.wan_centered_pilot.clear') as cleanup:
            output = root/'run'
            report = diagnostic.run_gradients(g,root/'pilot',root/'attribution',huber,output)
            assert len(calls) == 1 and len(back) == 2
            assert report['latent_unchanged'] and not report['port_success']
            assert report['counts']['latent_backwards_completed'] == 2
            assert report['latent_gradient_comparison']['cosine'] < -.999999
            for arm in refs:
                with np.load(output/f'{arm}_latent_gradient.npz') as values:
                    expected = np.clip(.05-refs[arm][0][0,0,0],-1,1)/np.prod(shape)
                    np.testing.assert_allclose(values['gradient'],expected,rtol=1e-6)
            assert not report['archived_gradient_norms_match']  # Discrepancy reported, no hidden recapture.
            assert not list(output.glob('*.mp4'))
            try:
                diagnostic.run_gradients(g,root/'pilot',root/'attribution',huber,output)
            except RuntimeError as error:
                assert 'budget already' in str(error)
            else:
                raise AssertionError('Repeated budget accepted')
            assert len(calls) == 1
            bad = json.loads(json.dumps(saved_report))
            bad['arms']['forward']['readout_states'][0]['scores']['mse'] = 100.
            with patch.object(diagnostic,'prepare_gradient_inputs',return_value=(inputs,bad)):
                try:
                    diagnostic.run_gradients(g,root/'pilot',root/'attribution',huber,root/'bad')
                except RuntimeError as error:
                    assert 'losses do not replay' in str(error)
                else:
                    raise AssertionError('Bad replay accepted')
            assert len(calls) == 2 and len(back) == 2
            assert not (root/'bad/gradient_report.json').exists()
            assert (root/'bad/failure.json').exists()
            assert cleanup.call_count == 2
    print('PASS: one capture/two backwards runner, independent gradients, unchanged latent, report and budget/replay guards.')


def tiny_wan_checks():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the retained-graph BF16/offload check')
    from verify_wan_decisive import RealSamplerTests
    from diffusers import AutoencoderKLWan, WanPipeline
    from benchmark.wan_centered_pilot import clear
    with tempfile.TemporaryDirectory() as tmp:
        g, _ = RealSamplerTests().build('amf',tmp)
        vae = AutoencoderKLWan(base_dim=4,z_dim=4,dim_mult=[1,1,1,1],num_res_blocks=1,
            latents_mean=[0.]*4,latents_std=[1.]*4).requires_grad_(False)
        pipe = WanPipeline(tokenizer=None,text_encoder=None,vae=vae,transformer=g.transformer,scheduler=g.scheduler)
        pipe.enable_model_cpu_offload(device=g.device)
        assert g.transformer.gradient_checkpointing
        assert hasattr(g.transformer,'_hf_hook')
        def capture_small(x):
            clear(g)
            g._set_kv_mode([1],inject=False,copy=True)
            g._forward_transformer(x,g.guidance_embeds[1:2],g.timesteps[39].expand(1))
            p = g.transformer.blocks[1].attn1.processor
            q,k = [v[0].reshape(2,6,4,8) for v in (p.query,p.key)]
            f = centered_pair_flow(q[0],k[1],2,3)
            g._clear_kv([1])
            return f, q, k
        x = g.init_latents.detach().clone().requires_grad_()
        initial = x.detach().clone()
        flow,q,k = capture_small(x)
        original_flow = flow.detach().clone()
        targets = [torch.full_like(flow,.3),torch.full_like(flow,-.4)]
        mask = torch.ones(flow.shape[:-1],device=g.device,dtype=torch.bool)
        refs = {a:(t.detach().float().cpu().numpy()[None],mask.cpu().numpy()[None])
                for a,t in zip(('forward','reverse'),targets)}
        readout, predicted = diagnostic.readout_analysis(q,k,refs,Path(tmp),height=2,width=3)
        np.testing.assert_allclose(predicted[0],flow.detach().cpu().numpy(),atol=1e-4,rtol=0)
        shared = []
        for i,target in enumerate(targets):
            grad, = torch.autograd.grad(huber_loss(flow,target,mask),x,retain_graph=i==0)
            shared.append(grad.detach().clone())
            assert torch.isfinite(grad).all() and grad.norm() > 0
            g._clear_kv([1])
        # Independent graph controls are tiny local tests, not extra 14B experiment calls.
        for target, expected in zip(targets, shared):
            fresh = initial.clone().requires_grad_()
            f,_,_ = capture_small(fresh)
            torch.testing.assert_close(f,original_flow,rtol=0,atol=0)
            actual, = torch.autograd.grad(huber_loss(f,target,mask),fresh)
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        torch.testing.assert_close(x,initial,rtol=0,atol=0)
        assert all(p.grad is None for p in g.transformer.parameters())
        clear(g)
    print('PASS: tiny real BF16 Wan with checkpointing and CPU offload: both retained-graph gradients equal fresh-graph controls exactly.')


if __name__ == '__main__':
    analytic_checks()
    runner_checks()
    tiny_wan_checks()
