"""Weight-free checks for head selection, evidence gates and visual instrumentation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from types import SimpleNamespace, MethodType
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from guidance_utils.wan_affine_diagnostics import field_metrics, head_readouts
from guidance_utils.wan_head_diagnostics import audit_heads, screen_heads, make_head_report
from guidance_utils.wan_amf_calibration import CONTROLS, MODEL
from guidance_utils.wan_motion_flow_utils import compute_motion_flow
from benchmark import wan_head_pilot as pilot
from benchmark.wan_head_visual_pilot import audit_trace, fixed_config, require_matched_starts


class HeadTests(unittest.TestCase):
    def test_head_matches_independent_attention_and_gradient(self):
        torch.manual_seed(7)
        q = torch.randn(1,12,3,8,requires_grad=True); k = torch.randn_like(q,requires_grad=True)
        for checkpoint in (False,True):
            flow = compute_motion_flow(q,k,2,3,2,head_index=1,checkpoint_pairs=checkpoint)
            probability = (q[0,:6,1] @ k[0,6:,1].T * (2/8**.5)).softmax(-1)
            xy = torch.tensor([[x,y] for y in range(2) for x in range(3)],dtype=torch.float32)
            torch.testing.assert_close(flow[1],probability @ xy-xy)
            dq,dk = torch.autograd.grad(flow.square().sum(),(q,k))
            self.assertGreater(float(dq[:,:,1].abs().max()),0)
            self.assertGreater(float(dk[:,:,1].abs().max()),0)
            self.assertEqual(float(dq[:,:,[0,2]].abs().max()),0)
            self.assertEqual(float(dk[:,:,[0,2]].abs().max()),0)
        default = compute_motion_flow(q,k,2,3,2)
        torch.testing.assert_close(default,compute_motion_flow(q,k,2,3,2,head_index=None),rtol=0,atol=0)
        readouts = dict(head_readouts(q,k,[2,2,3],heads=[1]))
        self.assertEqual(set(readouts),{'mean_logits','head_01'})
        np.testing.assert_allclose(readouts['head_01']['soft'][0],flow[1].detach().numpy(),rtol=1e-5,atol=1e-6)
        for invalid in (-1,3,True,1.2):
            with self.assertRaises(ValueError): compute_motion_flow(q,k,2,3,2,head_index=invalid)
        for heads in ([],[1,1],[3]):
            with self.assertRaises(ValueError): list(head_readouts(q,k,[2,2,3],heads=heads))

    def test_production_reference_and_loss_use_the_same_head(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_modules import WanInjectionProcessor
        torch.manual_seed(8)
        proc = WanInjectionProcessor('block_0_attn1_processor')
        reference = torch.randn(1,12,3,8); key = torch.randn_like(reference)
        config = OmegaConf.create(dict(guidance_blocks=[0],motion_temp=2.,flow_head=1,softmax_fp32=True,
            argmax_motion_flow=True,threshloss=False,flow_max_disp=100.,flow_loss='mse'))
        g = SimpleNamespace(config=config,transformer=SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=proc))],
            config=SimpleNamespace(attention_head_dim=8)),patches_height=2,patches_width=3,latent_num_frames=2,
            checkpoint_amf=False,device='cpu',motion_latent=reference,source_embeds=None,
            guidance_embeds=torch.zeros(2,1),motion_timestep=torch.tensor([0.]),
            probe=SimpleNamespace(phase=lambda *a,**kw:nullcontext(),actual_reference=lambda *a:None,training_flow=lambda *a:None))
        def forward(x,*a,**kw): proc.query=x; proc.key=key
        g._forward_transformer=forward
        for name in ('_amf','_set_kv_mode','_clear_kv'):
            setattr(g,name,MethodType(getattr(WanGuidance,name),g))
        g.motion_attn_features=WanGuidance.load_attn_features(g)
        expected=compute_motion_flow(reference,key,2,3,2,argmax=True,head_index=1)
        torch.testing.assert_close(g.motion_attn_features[proc.block_name],expected,rtol=0,atol=0)
        x=torch.randn_like(reference,requires_grad=True)
        actual=WanGuidance.compute_motion_flow_loss(g,x,torch.tensor([930.]))
        wanted=(compute_motion_flow(x,key,2,3,2,head_index=1)-expected).square().mean()
        torch.testing.assert_close(actual,wanted,rtol=0,atol=0)
        actual.backward(); self.assertGreater(float(x.grad[:,:,1].abs().max()),0)
        self.assertEqual(float(x.grad[:,:,[0,2]].abs().max()),0)

    def test_fixed_screen_does_not_promote_best_failing_or_mean(self):
        rows=[]
        for variant in ('mean_logits','head_03'):
            for state in ('clean','step_09','step_29'):
                for control in CONTROLS:
                    rows.append(dict(block='block_20_attn1_processor',variant=variant,field='soft',support='textured',
                        anchor_offset=0,noise_label=state,control=control,patches=10,finite_fraction=1.,
                        epe=0.,direction_cosine=1.,amplitude_ratio=1.))
        self.assertTrue(all(c['screen_pass'] for c in screen_heads(rows)['candidates']))
        rows[-1]['amplitude_ratio']=0.
        report=screen_heads(rows)
        self.assertFalse(next(c for c in report['candidates'] if c['variant']=='head_03')['screen_pass'])
        with patch.object(pilot,'summarize',return_value=(Path('.'),report)):
            with self.assertRaisesRegex(ValueError,'every predeclared'): pilot.candidate_receipt('.',(20,3))

    def test_plans_freeze_selection_and_budget_before_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); inputs=root/'inputs'; inputs.mkdir(); (inputs/'manifest.csv').write_text('fixture')
            rows=[dict(clip_id=c,video_path=c) for c in ('car-turn','camel')]
            with patch.object(pilot.direction,'prepare_inputs',return_value=(inputs,rows)),patch.object(pilot,'environment_snapshot',return_value={}):
                dev=pilot.make_plan(inputs,root/'dev')
                self.assertEqual(dev['expected_rows'],19680); self.assertIsNone(dev['heads'])
                self.assertNotIn('--mean_only',dev['command']); self.assertEqual(dev['blocks'],[20,30])
                receipt=dict(candidate=dict(block='block_30_attn1_processor',variant='head_03',screen_pass=True))
                with patch.object(pilot,'candidate_receipt',return_value=receipt):
                    conf=pilot.make_plan(inputs,root/'conf',development=root/'dev',candidate=(30,3))
                self.assertEqual(conf['expected_rows'],480); self.assertEqual(conf['heads'],[3])
                self.assertEqual((conf['clip_id'],conf['noise_seed']),('camel',29))
                self.assertEqual(conf['development_receipt'],receipt)
                with self.assertRaises(ValueError): pilot.make_plan(inputs,root/'bad',candidate=(30,3))
                (root/'dev/plan.json').write_text('{}')
                with self.assertRaisesRegex(ValueError,'changed'): pilot.load_plan(root/'dev')

    def test_confirmation_cannot_switch_development_candidate(self):
        conf=dict(stage='confirmation',blocks=[30],heads=[3],development_receipt={'original':True})
        with patch.object(pilot,'load_plan',side_effect=[dict(stage='development'),conf]), \
             patch.object(pilot,'candidate_receipt',return_value={'different':True}):
            with self.assertRaisesRegex(ValueError,'frozen candidate changed'): pilot.confirmed_candidate('dev','conf')

    def test_streaming_audit_detects_metrics_noise_and_duplicate_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
            states=[dict(noise_label='clean',sampling_index=-1,sigma=0.,timestep=0.)]+[
                dict(noise_label=f'step_{i:02d}',sampling_index=i,sigma=float(scheduler.sigmas[i]),timestep=float(scheduler.timesteps[i])) for i in (0,9,29)]
            meta=dict(model=MODEL,blocks=[20],readout_heads=[3],controls=CONTROLS,noise_seed=29,conditioning='',
                temperature=2.,grid=[6,30,52],cpu_offload=True,mean_only=False,noise_states=states)
            pilot.write_json(root/'metadata.json',meta)
            config=dict(model_key=MODEL,enable_model_cpu_offload=True,guidance_blocks=[],injection_blocks=[],
                target_prompt='',source_prompt='',scheduler='flowmatch',flow_shift=3.,num_frames=21,
                height=480,width=832,num_inference_steps=50,seed=1,reference_only=True,probe=False,probe_rope=False,motion_temp=2.)
            OmegaConf.save(OmegaConf.create(config),root/'suite_config.yaml')
            truth=np.ones((5,1560,2),np.float32); valid=np.ones((5,1560),bool); rows=[]
            for control in CONTROLS:
                (root/control).mkdir()
                np.savez(root/control/'truth.npz',**{f'{key}_{offset}':value for offset in (-2,0,2)
                    for key,value in [('flow',truth),('geometry',valid),('texture',valid)]})
                for state in states:
                    folder=root/control/state['noise_label']/'block_20_attn1_processor'; folder.mkdir(parents=True)
                    arrays=dict(soft=truth,hard=truth,confidence=np.ones((5,1560)),entropy=np.zeros((5,1560)),logit_std=np.zeros((5,1560)))
                    for variant in ('mean_logits','head_03'):
                        np.savez(folder/f'{variant}.npz',**arrays)
                        for offset in (-2,0,2):
                            for support in ('geometry','textured'):
                                for field in ('hard','soft'):
                                    rows.append(dict(control=control,**state,block='block_20_attn1_processor',variant=variant,
                                        anchor_offset=offset,support=support,field=field,**field_metrics(truth,truth,valid),confidence=1.,entropy=0.))
            pilot.write_json(root/'metrics.json',rows); pilot.write_json(root/'complete.json',dict(forward_passes=20,rows=480))
            audit=lambda:audit_heads(root,[20],[3],29)
            self.assertEqual(audit()['captures'],40); make_head_report(root)
            rows[0]['epe']=3.; pilot.write_json(root/'metrics.json',rows)
            with self.assertRaisesRegex(ValueError,'metric disagrees'): audit()
            rows[0]['epe']=0.; pilot.write_json(root/'metrics.json',rows+rows[:1])
            with self.assertRaisesRegex(ValueError,'duplicated'): audit()
            pilot.write_json(root/'metrics.json',rows)
            arrays['soft']=truth*2
            np.savez(root/'pan_left/step_00/block_20_attn1_processor/head_03.npz',**arrays)
            with self.assertRaisesRegex(ValueError,'Pure-noise'): audit()

    def test_visual_configs_and_actual_trace_budgets(self):
        plan=dict(video='car',prompt='truck approaches',selection=dict(block=30,head=3))
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        from guidance_utils.wan_guidance_schedule import learning_rates
        rates=learning_rates(list(range(10)),[.002,.001],10)
        for arm in ('off','baseline','candidate'):
            config=fixed_config(plan,arm,'out')
            self.assertEqual(config['flow_head'],3 if arm=='candidate' else None)
            self.assertEqual(config['guidance_blocks'],[] if arm=='off' else [10 if arm=='baseline' else 30])
            events=[dict(kind='sampling',step=i,timestep=float(scheduler.timesteps[i]),sigma=float(scheduler.sigmas[i]),
                latent=dict(finite_fraction=1.),guidance_update=dict(abs_max=0.)) for i in range(50)]
            if arm!='off':
                events += [dict(kind='optimization',step=i,iteration=j,timestep=float(scheduler.timesteps[i]),lr=rates[i],
                    gradient=dict(finite_fraction=1.,abs_max=.1),update=dict(finite_fraction=1.,abs_max=.1),loss_before_update=1.)
                    for i in range(10) for j in range(5)]
                events.append(dict(kind='training_flow',block=f"block_{config['guidance_blocks'][0]}_attn1_processor"))
            self.assertEqual(audit_trace(events,config,arm)['updates'],0 if arm=='off' else 50)
            bad=copy.deepcopy(events); bad[0]['sigma']=.5
            with self.assertRaisesRegex(ValueError,'schedule'): audit_trace(bad,config,arm)
            with self.assertRaisesRegex(ValueError,'sampling'): audit_trace(events[1:],config,arm)
            if arm!='off':
                bad=copy.deepcopy(events); bad[50]['step']=10
                with self.assertRaisesRegex(ValueError,'optimizer count'): audit_trace(bad,config,arm)

    def test_estimates_preserve_rng_latent_scheduler_and_caches(self):
        from probe_wan_head_visual import capture_estimate
        from guidance_utils.wan_modules import WanInjectionProcessor
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        proc=WanInjectionProcessor('test'); proc.query=torch.randn(1); proc.copy_kv=True
        original_q=proc.query
        probe=SimpleNamespace(emit=lambda *a,**kw:None)
        g=SimpleNamespace(config=SimpleNamespace(guidance_blocks=[]),scheduler=scheduler,output_path='unused',probe=probe,
            transformer=SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=proc))],init_rope=torch.randn(3)))
        latent=torch.randn(2); rng=torch.get_rng_state().clone()
        def predict(*a): proc.query=torch.randn(8); return latent.clone()
        with patch('probe_wan_head_visual.predict_clean',side_effect=predict),patch('probe_wan_head_visual.decode'):
            capture_estimate(g,latent,9,'before')
        self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertIs(proc.query,original_q)
        self.assertTrue(proc.copy_kv); self.assertIsNone(scheduler.step_index)

    def test_starting_state_hashes_reject_unmatched_seed_or_conditioning(self):
        state=dict(latent_sha256='noise',reference_latent_sha256='ref',conditioning_sha256='text',
            source_conditioning_sha256='blank',rope_sha256='native',timesteps=[1,0],sigmas=[1,0],model_revision=None)
        require_matched_starts([state,dict(state)])
        for key in ('latent_sha256','conditioning_sha256','rope_sha256'):
            with self.assertRaisesRegex(ValueError,'Unmatched'): require_matched_starts([state,dict(state,**{key:'different'})])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required by production Wan methods')
    def test_actual_bf16_guidance_and_sampler_unchanged_by_estimates(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.motion_probe import MotionProbe
        from probe_wan_head_visual import capture_estimate
        outputs=[]
        for instrument in (False,True):
            torch.manual_seed(17)
            with tempfile.TemporaryDirectory() as tmp:
                g=WanGuidance.__new__(WanGuidance); torch.nn.Module.__init__(g)
                g.config=OmegaConf.create(dict(probe=True,probe_blocks=[0,1,2],probe_steps=[9],probe_rope=False,
                    guidance_blocks=[1],injection_blocks=[],loss_type='flow',flow_head=1,motion_temp=2.,
                    softmax_fp32=True,argmax_motion_flow=True,threshloss=False,optimization_steps=2,
                    verbose=False,save_embeds=False,flow_loss='mse'))
                g.device,g.dtype=torch.device('cuda'),torch.bfloat16
                g.transformer=ControlledWanTransformer(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=8,
                    in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=32,num_layers=3,
                    cross_attn_norm=True,qk_norm='rms_norm_across_heads',eps=1e-6,rope_max_seq_len=64
                    ).to(device=g.device,dtype=g.dtype).eval().requires_grad_(False)
                g.transformer.enable_gradient_checkpointing()
                g.latent_height,g.latent_width,g.patch_size=4,6,2
                g.patches_height,g.patches_width,g.latent_num_frames=2,3,2
                g.checkpoint_amf,g._guidance_scale=False,5
                g.scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); g.scheduler.set_timesteps(50,device=g.device)
                g.timesteps=g.scheduler.timesteps; g.lr_range=np.array([.001]); g.lr_by_step={9:.001}; g.output_path=tmp
                g.register_guidance([1]); g.register_attention_processor([0,1,2]); g.probe=MotionProbe(g,'wan')
                g.motion_latent=torch.randn(1,4,2,4,6,device=g.device,dtype=g.dtype)
                g.transformer.init_rope=g.transformer.default_rope(g.motion_latent).to(g.device)
                g.source_embeds=torch.randn(1,5,16,device=g.device,dtype=g.dtype)
                g.guidance_embeds=torch.randn(2,5,16,device=g.device,dtype=g.dtype)
                g.motion_timestep=torch.tensor([0],device=g.device); g.motion_attn_features=g.load_attn_features()
                x=torch.randn_like(g.motion_latent).float(); t=g.timesteps[9]
                with patch('probe_wan_head_visual.decode'):
                    if instrument: capture_estimate(g,x,9,'before')
                    optimized,rope=g.guidance_step(x,9,t,'latent','flow')
                    if instrument: capture_estimate(g,optimized,9,'after')
                result=g.denoise_step(optimized,9,g.guidance_embeds,rope)
                outputs.append((optimized.cpu(),result.cpu()))
        for a,b in zip(*outputs): torch.testing.assert_close(a,b,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
