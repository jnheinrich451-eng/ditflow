"""Checks for isolated pair selection, matched starts and the actual BF16 loss path."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import copy
import hashlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from benchmark import wan_pair_pilot as pilot
from benchmark import wan_noised_reference_pilot as noised
from probe_wan_noised_reference import NoisedReferenceMixin
from probe_wan_pairs import ForwardAdjacentMixin


class PairTests(unittest.TestCase):
    def test_pair_indices_and_validity_intersection(self):
        self.assertEqual(pilot.adjacent_pairs(6).tolist(),[1,8,15,22,29])
        mask=np.ones((36,4),dtype=bool); mask[8,1]=False
        selected=pilot.selected_mask(mask,6)
        self.assertEqual(int(selected.sum()),19)
        self.assertFalse(selected[8,1]); self.assertFalse(selected[0].any()); self.assertFalse(selected[6].any())
        with self.assertRaisesRegex(ValueError,'No valid'): pilot.selected_mask(np.zeros_like(mask),6)
        with self.assertRaises(ValueError): pilot.adjacent_pairs(1)

    def test_production_mask_gives_zero_gradient_to_unselected_pairs(self):
        class Base:
            def load_attn_features(self): return self.reference
        class Owner(ForwardAdjacentMixin,Base): pass
        with tempfile.TemporaryDirectory() as tmp:
            g=Owner(); g.config=SimpleNamespace(guidance_blocks=[0],flow_pair_mode=pilot.PAIR_MODE)
            g.latent_num_frames=3; block='block_0_attn1_processor'
            g.motion_attn_masks={block:torch.ones(9,4,dtype=torch.bool)}
            g.motion_attn_masks[block][1,0]=False
            g.reference={block:torch.ones(9,4,2)}
            records=[]; g.probe=SimpleNamespace(path=Path(tmp),emit=lambda kind,**kw:records.append(dict(kind=kind,**kw)))
            reference=g.load_attn_features()[block]; mask=g.motion_attn_masks[block]
            flow=torch.full_like(reference,3.,requires_grad=True)
            loss=(flow[mask]-reference[mask]).square().mean(); loss.backward()
            self.assertEqual(float(loss.detach()),4.)  # Normalized over selected entries, not over all pairs.
            self.assertEqual(float(flow.grad[~mask].abs().sum()),0.)
            self.assertTrue(bool((flow.grad[mask]>0).all()))
            self.assertEqual(records[0]['pair_indices'],[1,5])
            torch.testing.assert_close(reference,g.reference[block],rtol=0,atol=0)

    def test_config_only_changes_pair_selection_and_protocol_label(self):
        plan=dict(generation_video='car',prompt='truck',selection=dict(block=30,head=30))
        old=noised.fixed_config(plan,'candidate','output'); new=pilot.fixed_config(plan,'output')
        differences={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
        self.assertEqual(differences,{'flow_pair_mode','visual_protocol'})
        self.assertEqual(new['optimization_steps'],5); self.assertEqual(new['lr'],[.001,.001])

    def test_reuse_rejects_runtime_or_model_source_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'model.py'; source.write_bytes(b'unchanged')
            previous=dict(python='3',packages={'torch':'same'},cuda='same',gpu=['same'],
                source_sha256={str(source):hashlib.sha256(b'unchanged').hexdigest()})
            pilot.require_reusable_environment(previous,copy.deepcopy(previous))
            for key in ('python','packages','cuda','gpu'):
                current=copy.deepcopy(previous); current[key]='changed'
                with self.assertRaisesRegex(ValueError,key): pilot.require_reusable_environment(previous,current)
            source.write_bytes(b'changed'); current=copy.deepcopy(previous)
            current['source_sha256'][str(source)]=hashlib.sha256(b'changed').hexdigest()
            with self.assertRaisesRegex(ValueError,'source changed'): pilot.require_reusable_environment(previous,current)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required by production Wan methods')
    def test_bf16_pair_loss_and_full_sampler_with_identical_pre_update_fields(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.motion_probe import MotionProbe
        from probe_wan_head_visual import capture_estimate
        from probe_report import load_trace
        class OriginalWan(NoisedReferenceMixin,WanGuidance): pass
        class AdjacentWan(ForwardAdjacentMixin,NoisedReferenceMixin,WanGuidance): pass
        with tempfile.TemporaryDirectory() as tmp:
            traces=[]
            for adjacent,cls in [(False,OriginalWan),(True,AdjacentWan)]:
                torch.manual_seed(17)
                folder=Path(tmp)/str(adjacent); folder.mkdir()
                g=cls.__new__(cls); torch.nn.Module.__init__(g)
                g.config=OmegaConf.create(dict(probe=True,probe_blocks=[1],probe_steps=[9],probe_rope=False,
                    guidance_blocks=[1],injection_blocks=[],loss_type='flow',flow_head=30,motion_temp=2.,
                    softmax_fp32=True,argmax_motion_flow=True,threshloss=True,flow_max_disp=100.,optimization_steps=5,
                    verbose=False,save_embeds=False,flow_loss='mse',reference_noise_step=9,reference_noise_seed=29,
                    flow_pair_mode=pilot.PAIR_MODE if adjacent else 'all'))
                g.device,g.dtype=torch.device('cuda'),torch.bfloat16
                g.transformer=ControlledWanTransformer(patch_size=(1,2,2),num_attention_heads=40,attention_head_dim=8,
                    in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=32,num_layers=3,
                    cross_attn_norm=True,qk_norm='rms_norm_across_heads',eps=1e-6,rope_max_seq_len=64
                    ).to(device=g.device,dtype=g.dtype).eval().requires_grad_(False)
                g.transformer.enable_gradient_checkpointing()
                g.latent_height,g.latent_width,g.patch_size=4,6,2
                g.patches_height,g.patches_width,g.latent_num_frames=2,3,2
                g.checkpoint_amf,g._guidance_scale=True,5
                g.scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); g.scheduler.set_timesteps(50,device=g.device)
                g.timesteps=g.scheduler.timesteps; g.lr_by_step={9:.001}; g.output_path=str(folder)
                g.register_guidance([1]); g.register_attention_processor([0,1,2]); g.probe=MotionProbe(g,'wan')
                g.motion_latent=torch.randn(1,4,2,4,6,device=g.device,dtype=g.dtype)
                g.transformer.init_rope=g.transformer.default_rope(g.motion_latent).to(g.device)
                rope=g.transformer.init_rope.clone()
                g.source_embeds=torch.randn(1,5,16,device=g.device,dtype=g.dtype)
                g.guidance_embeds=torch.randn(2,5,16,device=g.device,dtype=g.dtype)
                g.motion_timestep=torch.tensor([0],device=g.device); g.motion_attn_features=g.load_attn_features()
                x=torch.randn_like(g.motion_latent).float()
                with torch.no_grad(),patch('probe_wan_head_visual.decode'):
                    for i,t in enumerate(g.timesteps):
                        before=x.clone()
                        if i==9:
                            capture_estimate(g,x,9,'before')
                            with torch.enable_grad(): x,_=g.guidance_step(x,9,t,'latent','flow')
                            capture_estimate(g,x,9,'after')
                        guided=x; x=g.denoise_step(x,i,g.guidance_embeds)
                        g.probe.sampling(i,t,before,guided,x,g.scheduler)
                trace,_,events=load_trace(folder); traces.append((trace,events))
                audit=noised.audit_trace(events,OmegaConf.to_container(g.config),'candidate')
                self.assertEqual(audit['updates'],5); self.assertTrue(audit['guidance_active'])
                if adjacent:
                    selected=pilot.audit_selected_loss(trace,events,2,2,3)
                    self.assertEqual(selected['pair_indices'],[1])
                    before_metrics=pilot.score_forward_adjacent(folder,frames=2)['before']
                    self.assertAlmostEqual(before_metrics['mse'],selected['losses'][0],places=6)
                    # A diluted denominator or the old all-pair loss must fail the actual-loss audit.
                    bad=copy.deepcopy(events); next(e for e in bad if e['kind']=='full_pair_loss')['loss']+=1
                    with self.assertRaisesRegex(ValueError,'MSE disagrees'): pilot.audit_selected_loss(trace,bad,2,2,3)
                else: noised.audit_full_pairs(trace,events,'candidate',2,2,3)
                torch.testing.assert_close(rope,g.transformer.init_rope,rtol=0,atol=0)
                self.assertTrue(all(p.grad is None for p in g.transformer.parameters()))
            for kind in ('training_reference','full_pair_target'):
                a=next(e for e in traces[0][1] if e['kind']==kind)
                b=next(e for e in traces[1][1] if e['kind']==kind)
                self.assertTrue(pilot.npz_equal(traces[0][0]/a['file'],traces[1][0]/b['file']))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
