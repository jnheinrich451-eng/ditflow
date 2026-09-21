"""Tiny real-Wan prefix tests with native non-reentrant checkpoint recomputation."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from diffusers.models.transformers.transformer_wan import WanTransformerBlock

from benchmark import wan_centered_prefix as experiment
from benchmark.wan_centered_destination import destination_objectives
from guidance_utils.wan_modules import WanInjectionProcessor
from guidance_utils.wan_centered_amf import centered_pair_flow


class TinyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = True
        self.embedding = torch.nn.Linear(2, 8)
        self.blocks = torch.nn.ModuleList([WanTransformerBlock(8, 16, 2) for _ in range(40)])
        for i, b in enumerate(self.blocks):
            p = WanInjectionProcessor(str(i)); p.copy_kv = i == 20
            b.attn1.set_processor(p)
        self.to(dtype=torch.bfloat16).requires_grad_(False)
        self.text = torch.randn(1, 3, 8, dtype=torch.bfloat16)
        self.temb = torch.randn(1, 6, 8, dtype=torch.float32)*.1
        angle = torch.randn(1, 24, 1, 2).repeat_interleave(2, -1)
        self.rope = (angle.cos().bfloat16(), angle.sin().bfloat16())

    def forward(self, x):
        h = self.embedding(x.bfloat16().permute(0,2,3,4,1).reshape(1,24,2))
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and torch.is_grad_enabled():
                h = checkpoint(block, h, self.text, self.temb, self.rope, use_reentrant=False)
            else:
                h = block(h, self.text, self.temb, self.rope)
            if i == 20:
                break
        p = self.blocks[20].attn1.processor
        q, k = [v.reshape(6,4,2,4) for v in (p.query, p.key)]
        flow = torch.stack([centered_pair_flow(q[i],k[i+1],1,4) for i in range(5)])
        return flow, q, k


def fixture():
    torch.manual_seed(41)
    transformer = TinyTransformer()
    g = SimpleNamespace(transformer=transformer, device='cpu')
    g._clear_kv = lambda blocks: [transformer.blocks[i].attn1.processor.clear() for i in blocks]
    x = torch.randn(1,2,6,2,2).requires_grad_(True)
    support = np.ones((5,4),bool); support[:,-1] = False
    refs = {a:(np.zeros((5,4,2),np.float32),support.copy()) for a in experiment.ARMS}
    ids = {a:np.tile((np.arange(4)+s)%4,(5,1)) for a,s in [('forward',1),('reverse',-1)]}
    incoming = []
    handle = transformer.blocks[20].register_forward_pre_hook(lambda m,a:incoming.append(a[0]) if not incoming else None)
    flow,q,k = transformer(x)
    losses, logs = destination_objectives(q,k,refs,ids,1,4)
    baseline = [torch.autograd.grad(losses[a],(x,incoming[0]),retain_graph=i==0) for i,a in enumerate(experiment.ARMS)]
    handle.remove()
    bm = experiment.stage_moments('block_20',baseline[0][1],baseline[1][1])
    latent = {a:baseline[i][0].float().numpy() for i,a in enumerate(experiment.ARMS)}
    before = flow.detach().float().numpy()
    ready = dict(inputs=({'experiment':{'guidance_sigma':.45946}}, {'before':x.detach().numpy()},refs,{}, {},{'before':before}),
        refs=refs,indices=ids,support=support,before_logp=logs,provenance={},block_moments=bm,
        block_report={'stage_comparison':{'block_input':experiment.summarize_moments(bm)}},
        latent_gradients=latent,latent_comparison=experiment.compare(latent['forward'],latent['reverse']),
        shared_moments=np.zeros((6,3)))
    # Verify baseline checkpoint backward against ordinary native prefix too.
    transformer.gradient_checkpointing = False
    y=x.detach().clone().requires_grad_(True)
    _,q2,k2=transformer(y)
    losses2,_=destination_objectives(q2,k2,refs,ids,1,4)
    for i,a in enumerate(experiment.ARMS):
        v,=torch.autograd.grad(losses2[a],y,retain_graph=i==0)
        np.testing.assert_array_equal(v.float().numpy(),latent[a])
    transformer.gradient_checkpointing = True
    g._clear_kv(range(40))
    return g,ready


def runner_checks():
    g,ready=fixture()
    native_objectives=experiment.destination_objectives
    def readout(q,k,refs,ids):
        return native_objectives(q,k,refs,ids,1,4)
    with tempfile.TemporaryDirectory() as tmp, \
         patch.object(experiment,'prepare_prefix_inputs',return_value=ready), \
         patch.object(experiment,'frozen_model',return_value={}), \
         patch.object(experiment,'capture',side_effect=lambda model,x:model.transformer(x)), \
         patch.object(experiment,'destination_objectives',side_effect=readout), \
         patch('benchmark.wan_centered_pilot.clear',side_effect=lambda model:model._clear_kv(range(40))):
        root=Path(tmp)
        args=(g,*[root/n for n in ('p','a','h','g','r','d','t','q','s','b')])
        report=experiment.run_prefix(*args,root/'ok')
        assert report['counts']==dict(positive_forwards_attempted=1,positive_forwards_completed=1,
            latent_backwards_attempted=2,latent_backwards_completed=2)
        assert report['total_block_entries']==63
        assert set(report['block_entries'])=={'capture','forward','reverse'}
        assert set(report['stage_comparison'])==set(experiment.ORDER)
        assert report['baseline_unchanged'] and not report['port_success']
        for phase,entries in report['block_entries'].items():
            assert [entries[str(i)] for i in range(40)]==[1]*21+[0]*19
        for a in experiment.ARMS:
            assert report['replay']['arms'][a]['latent_max_abs']==0
        assert all(not b._forward_pre_hooks for b in g.transformer.blocks)
        assert all(p.grad is None for p in g.transformer.parameters())
        json.dumps(report,allow_nan=False)
        with np.load(root/'ok/gradient_moments.npz') as data:
            assert all(data[n].shape==(6,3) for n in data.files)
        try:
            experiment.run_prefix(*args,root/'ok')
        except RuntimeError as error:
            assert 'budget already' in str(error)
        else:
            raise AssertionError('Repeated budget accepted')
        for kind in ('logp','latent','block20'):
            if kind=='logp':
                bad={**ready,'before_logp':{a:v+1 for a,v in ready['before_logp'].items()}}
            elif kind=='latent':
                bad={**ready,'latent_gradients':{a:v*2 for a,v in ready['latent_gradients'].items()}}
            else:
                bad={**ready,'block_moments':ready['block_moments']*2}
            with patch.object(experiment,'prepare_prefix_inputs',return_value=bad):
                try:
                    experiment.run_prefix(*args,root/kind)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('Bad endpoint replay accepted')
            failed=json.loads((root/kind/'failure.json').read_text())
            assert failed['counts']['latent_backwards_attempted']==(0 if kind=='logp' else 1)
            assert not (root/kind/'prefix_report.json').exists()
            assert all(not b._forward_pre_hooks for b in g.transformer.blocks)
        counts=dict(positive_forwards_attempted=0,positive_forwards_completed=0,latent_backwards_attempted=0,latent_backwards_completed=0)
        observer=experiment.PrefixObserver(g.transformer,root,counts)
        try:
            observer.phase='capture'
            try:
                observer.before_block(21)(None,(),{})
            except RuntimeError as error:
                assert 'budget exceeded' in str(error)
            else:
                raise AssertionError('Downstream block accepted')
            observer.phase='capture'
            try:
                with torch.no_grad():
                    observer.before_block(0)(None,(torch.ones(1,24,8),),{})
            except RuntimeError as error:
                assert 'lacks graph' in str(error)
            else:
                raise AssertionError('Reentrant/no-grad original endpoint accepted')
        finally:
            observer.close()
    print('PASS: real tiny Wan prefix, native checkpoint/uncheckpoint gradient equivalence, passive hooks, all five endpoints, exact replay, 63 block entries, failure/repeat/cleanup guards.')


def preflight_checks():
    # Exercise the additional evidence-chain link independently of large local archives.
    prefix=np.zeros((1,2,6,2,2),np.float32)
    flow=np.zeros((5,4,2),np.float32)
    support=np.ones((5,4),bool)
    latent={a:np.full_like(prefix,i+1.) for i,a in enumerate(experiment.ARMS)}
    stats={'block_input':np.tile([1.,4.,2.],(6,1)), 'attention_input':np.tile([.1,.4,.2],(6,1))}
    ready=dict(inputs=({}, {'before':prefix},{},{},{},{'before':flow}),support=support,
        before_logp={a:np.full((5,4),-2.,np.float32) for a in experiment.ARMS},provenance={},
        shared_moments=stats['attention_input'],latent_comparison=experiment.compare(*latent.values()))
    with tempfile.TemporaryDirectory() as tmp:
        root=Path(tmp)
        protocol=dict(inputs_sha256={},control_sha256={},helper_sources_sha256={},
            script_sha256=experiment.digest(Path(experiment.__file__).with_name('wan_centered_block_input.py')))
        report=dict(counts=dict(positive_forwards_attempted=1,positive_forwards_completed=1,
            local_backwards_attempted=2,local_backwards_completed=2),baseline_unchanged=True,latent_graph_created=False,
            port_success=False,archived_latent_comparison=ready['latent_comparison'],
            stage_comparison={n:experiment.summarize_moments(v) for n,v in stats.items()})
        for n,v in [('started',protocol),('block_input_report',report)]:
            experiment.write_json(root/(n+'.json'),v)
        np.savez_compressed(root/'before_flow.npz',flow=flow)
        np.savez_compressed(root/'optimization_support.npz',common_torso=support)
        np.savez_compressed(root/'before_target_log_probability.npz',**ready['before_logp'])
        np.savez_compressed(root/'gradient_moments.npz',**stats)
        for a,v in latent.items():
            np.savez_compressed(root/f'{a}_latent_gradient.npz',gradient=v)
        with patch.object(experiment,'prepare_block_inputs',side_effect=lambda *a:dict(ready)):
            result=experiment.prepare_prefix_inputs('p','a','h','g','r','d',root,'q','s',root)
            assert len(result['provenance']['block_input_artifacts'])==6
            stats['block_input']*=2
            np.savez_compressed(root/'gradient_moments.npz',**stats)
            try:
                experiment.prepare_prefix_inputs('p','a','h','g','r','d',root,'q','s',root)
            except AssertionError:
                pass
            else:
                raise AssertionError('Corrupt control moments accepted')
    print('PASS: completed block-input evidence preflight and corruption rejection.')


def notebook_checks():
    import nbformat
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'wan_centered_prefix.ipynb',as_version=4)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type=='code':
            assert not c.outputs and c.execution_count is None
            compile(c.source,'<cell>','exec')
    assert 'SHARED, BLOCK_INPUT, OUTPUT)' in nb.cells[10].source
    print('PASS: clean notebook schema, syntax and completed-control wiring.')


if __name__=='__main__':
    torch.set_num_threads(1)
    preflight_checks()
    runner_checks()
    notebook_checks()
