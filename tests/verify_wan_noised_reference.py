"""No pretrained weights: independent noise, production loss, gates and matched videos."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from benchmark import wan_noised_reference_pilot as pilot
from guidance_utils.wan_amf_calibration import CONTROLS
from probe_wan_noised_reference import NoisedReferenceMixin, matched_reference_input


class ReferenceTests(unittest.TestCase):
    def test_gate_does_not_accept_good_cosine_with_wrong_magnitude(self):
        rows = [dict(block=pilot.BLOCK,variant='head_30',noise_label='step_09',field='soft',support='textured',
            anchor_offset=0,control=c,patches=10,finite_fraction=1.,epe=0.,direction_cosine=1.,amplitude_ratio=1.)
            for c in CONTROLS]
        self.assertTrue(pilot.screen_step9(rows)['passed'])
        for key,value in [('amplitude_ratio',11.6),('amplitude_ratio',-1.),('direction_cosine',.2)]:
            bad = copy.deepcopy(rows); bad[1][key]=value
            self.assertFalse(pilot.screen_step9(bad)['passed'])
        bad = copy.deepcopy(rows); bad[0]['epe']=.6
        self.assertFalse(pilot.screen_step9(bad)['passed'])
        self.assertFalse(pilot.screen_step9(rows+rows[:1])['passed'])
        # Clean failures are retained but cannot silently become a pass of the old full screen.
        clean = [dict(r,noise_label='clean',epe=99.,direction_cosine=-1.) for r in rows]
        self.assertTrue(pilot.screen_step9(rows+clean)['passed'])

    def test_reference_noise_formula_reproducibility_and_rng_isolation(self):
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        clean = torch.ones(1,4,2,4,6,dtype=torch.bfloat16)
        rng = torch.get_rng_state().clone()
        result,info = matched_reference_input(clean,scheduler,9,29)
        self.assertTrue(torch.equal(torch.get_rng_state(),rng))
        noise = torch.randn(clean.shape,generator=torch.Generator().manual_seed(29))
        sigma = float(scheduler.sigmas[9])
        expected = ((1-sigma)*clean.float()+sigma*noise).bfloat16()
        torch.testing.assert_close(result,expected,rtol=0,atol=0)
        repeated,again = matched_reference_input(clean,scheduler,9,29)
        self.assertEqual(info,again); self.assertTrue(torch.equal(result,repeated))
        changed,_ = matched_reference_input(clean,scheduler,9,1)
        self.assertFalse(torch.equal(result,changed))
        self.assertIsNone(scheduler.step_index)

    def test_reference_state_restored_after_success_and_failure(self):
        class Base:
            def load_attn_features(self):
                self.observed = (self.motion_latent.clone(),self.motion_timestep.clone())
                torch.rand(3)  # Even an incidental random operation must not alter generation RNG.
                if self.fail: raise RuntimeError('injected reference failure')
                return {'saved':True}
        class Owner(NoisedReferenceMixin,Base): pass
        with tempfile.TemporaryDirectory() as tmp:
            g = Owner(); g.output_path=tmp; g.fail=False
            g.config=SimpleNamespace(guidance_blocks=[30],reference_noise_step=9,reference_noise_seed=29)
            g.scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); g.scheduler.set_timesteps(50)
            g.timesteps=g.scheduler.timesteps
            original=torch.zeros(1,4,2,4,6,dtype=torch.bfloat16); time=torch.tensor([0])
            g.motion_latent=original; g.motion_timestep=time
            rng=torch.get_rng_state().clone()
            self.assertEqual(g.load_attn_features(),{'saved':True})
            self.assertFalse(torch.equal(g.observed[0],original))
            self.assertEqual(float(g.observed[1][0]),float(g.timesteps[9]))
            self.assertIs(g.motion_latent,original); self.assertIs(g.motion_timestep,time)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))
            g.fail=True
            with self.assertRaisesRegex(RuntimeError,'injected'): g.load_attn_features()
            self.assertIs(g.motion_latent,original); self.assertIs(g.motion_timestep,time)
            self.assertTrue(torch.equal(torch.get_rng_state(),rng))

    def test_archived_reference_input_reconstructs_and_rejects_tampering(self):
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'reference_inputs.npz'
            _,info=matched_reference_input(torch.ones(1,4,2,4,6).bfloat16(),scheduler,9,29,path)
            self.assertTrue(pilot.audit_reference_inputs(path,info)['noise_interpolation_verified'])
            with np.load(path) as data: arrays={k:data[k] for k in data.files}
            arrays['model_input'][0,0,0,0,0]+=1
            np.savez(path,**arrays)
            with self.assertRaisesRegex(ValueError,'fingerprint'): pilot.audit_reference_inputs(path,info)

    def test_frozen_configs_limit_guidance_to_measured_step(self):
        from guidance_utils.wan_guidance_schedule import window_indices,learning_rates
        plan=dict(generation_video='car',prompt='truck',selection=dict(block=30,head=30))
        for arm in pilot.ARMS:
            config=pilot.fixed_config(plan,arm,'out')
            self.assertEqual(window_indices(50,config['guidance_timestep_range']),[9])
            self.assertEqual(learning_rates([9],config['lr'],config['lr_decay_steps']),{9:.001})
            self.assertEqual(config['reference_noise_step'],9 if arm=='candidate' else None)
            self.assertEqual(config['flow_head'],30 if arm=='candidate' else None)
            self.assertEqual(config['injection_blocks'],[])

    def test_all_pair_audit_detects_nonadjacent_corruption_and_loss_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); reference=np.ones((4,6,2),dtype=np.float32)
            mask=np.ones((4,6),dtype=bool); mask[0]=False; reference[0]=0
            np.savez(root/'ref.npz',flow=reference,mask=mask)
            events=[dict(kind='training_reference',block=pilot.BLOCK,file='ref.npz')]
            for i in range(7):
                np.savez(root/f'flow{i}.npz',flow=reference+1)
                events += [dict(kind='full_pair_target',block=pilot.BLOCK,file=f'flow{i}.npz',
                    q_dtype='torch.bfloat16',k_dtype='torch.bfloat16',flow_head=30),
                    dict(kind='full_pair_loss',block=pilot.BLOCK,file=f'flow{i}.npz',loss=1.)]
            self.assertEqual(len(pilot.audit_full_pairs(root,events,'candidate',2,2,3)),7)
            bad=reference+1; bad[2,0]=np.nan  # Reverse pair; an adjacent-only check misses it.
            np.savez(root/'flow0.npz',flow=bad)
            with self.assertRaisesRegex(ValueError,'Invalid full-pair target'):
                pilot.audit_full_pairs(root,events,'candidate',2,2,3)
            np.savez(root/'flow0.npz',flow=reference+2)
            with self.assertRaisesRegex(ValueError,'MSE disagrees'):
                pilot.audit_full_pairs(root,events,'candidate',2,2,3)

    def test_failed_confirmation_still_runs_real_control_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); pilot.head.write_json(root/'confirmation_done.json',dict(directory='confirmation'))
            plan={}; calls=[]
            with patch.object(pilot,'checked_plan',return_value=plan), \
                 patch.object(pilot,'validate_confirmation',return_value=dict(gate=dict(passed=False,failures=['static']))), \
                 patch.object(pilot.head,'run_process',side_effect=lambda root,arm,*a:calls.append(arm)), \
                 patch.object(pilot,'summarize',return_value=[]):
                pilot.run_visual(root)
            self.assertEqual(calls,['off','baseline'])
            self.assertEqual(pilot.read_json(root/'candidate_skipped.json')['failures'],['static'])

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA required by production Wan methods')
    def test_actual_bf16_noised_reference_loss_gradients_and_complete_sampler(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.motion_probe import MotionProbe
        from probe_wan_head_visual import capture_estimate
        from probe_report import load_trace
        class TinyWan(NoisedReferenceMixin,WanGuidance): pass
        torch.manual_seed(17)
        with tempfile.TemporaryDirectory() as tmp:
            g=TinyWan.__new__(TinyWan); torch.nn.Module.__init__(g)
            g.config=OmegaConf.create(dict(probe=True,probe_blocks=[1],probe_steps=[9],probe_rope=False,
                guidance_blocks=[1],injection_blocks=[],loss_type='flow',flow_head=30,motion_temp=2.,
                softmax_fp32=True,argmax_motion_flow=True,threshloss=True,flow_max_disp=100.,optimization_steps=5,
                verbose=False,save_embeds=False,flow_loss='mse',reference_noise_step=9,reference_noise_seed=29))
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
            g.timesteps=g.scheduler.timesteps; g.lr_by_step={9:.001}; g.output_path=tmp
            g.register_guidance([1]); g.register_attention_processor([0,1,2]); g.probe=MotionProbe(g,'wan')
            g.motion_latent=torch.randn(1,4,2,4,6,device=g.device,dtype=g.dtype)
            clean=g.motion_latent
            g.transformer.init_rope=g.transformer.default_rope(clean).to(g.device)
            rope=g.transformer.init_rope.clone()
            g.source_embeds=torch.randn(1,5,16,device=g.device,dtype=g.dtype)
            g.guidance_embeds=torch.randn(2,5,16,device=g.device,dtype=g.dtype)
            g.motion_timestep=torch.tensor([0],device=g.device); g.motion_attn_features=g.load_attn_features()
            self.assertIs(g.motion_latent,clean); self.assertEqual(int(g.motion_timestep[0]),0)
            self.assertTrue(pilot.audit_reference_inputs(Path(tmp)/'reference_inputs.npz',
                pilot.read_json(Path(tmp)/'reference_protocol.json'))['noise_interpolation_verified'])
            x=torch.randn_like(clean).float()
            with torch.no_grad(),patch('probe_wan_head_visual.decode'):
                for i,t in enumerate(g.timesteps):
                    before=x.clone()
                    if i==9:
                        capture_estimate(g,x,9,'before')
                        with torch.enable_grad(): x,_=g.guidance_step(x,9,t,'latent','flow')
                        capture_estimate(g,x,9,'after')
                    guided=x
                    x=g.denoise_step(x,i,g.guidance_embeds)
                    g.probe.sampling(i,t,before,guided,x,g.scheduler)
            trace,_,events=load_trace(tmp)
            report=pilot.audit_trace(events,OmegaConf.to_container(g.config),'candidate')
            self.assertEqual(report['updates'],5); self.assertTrue(report['guidance_active'])
            pairs=pilot.audit_full_pairs(trace,events,'candidate',2,2,3)
            self.assertEqual(len(pairs),7); self.assertEqual(len(pairs[0]['pairs']),4)
            torch.testing.assert_close(rope,g.transformer.init_rope,rtol=0,atol=0)
            self.assertTrue(all(p.grad is None for p in g.transformer.parameters()))
            bad=copy.deepcopy(events)
            next(e for e in bad if e['kind']=='sampling' and e['step']==10)['guidance_update']['abs_max']=.1
            with self.assertRaisesRegex(ValueError,'outside'): pilot.audit_trace(bad,g.config,'candidate')


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
