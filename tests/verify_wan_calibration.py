"""CPU checks for shared-feature readouts, observation invariance and evidence audit."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from guidance_utils.motion_probe import adjacent_attention
from guidance_utils.wan_affine_diagnostics import field_metrics
from guidance_utils.wan_amf_calibration import (
    BLOCKS, CONTROLS, MODEL, TEMPERATURES, CalibrationObserver, audit_calibration,
    make_calibration_report, screen_candidates, shared_readouts, validate_temperatures, variant_name,
)
from benchmark.wan_calibration_pilot import make_plan


class CalibrationTests(unittest.TestCase):
    def test_matches_existing_observer_at_every_temperature(self):
        for dtype in (torch.float32, torch.bfloat16):
            q = torch.randn(1,18,3,8).to(dtype); k = torch.randn_like(q)
            actual = shared_readouts(q,k,[3,2,3],TEMPERATURES)
            for temperature in TEMPERATURES:
                expected = adjacent_attention(q,k,2,3,3,temperature)
                for key in expected:
                    np.testing.assert_array_equal(actual[temperature][key],expected[key])
            self.assertTrue(all(np.array_equal(actual[2.]['hard'],actual[t]['hard']) for t in TEMPERATURES))
        for invalid in ([], [0], [2,2], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError): validate_temperatures(invalid)

    def test_observation_preserves_forward_gradient_and_rng(self):
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.wan_modules import WanInjectionProcessor
        model = ControlledWanTransformer(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=8,
            in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=32,num_layers=3,
            cross_attn_norm=True,qk_norm='rms_norm_across_heads',eps=1e-6,rope_max_seq_len=64).eval().requires_grad_(False)
        for index,block in enumerate(model.blocks): block.attn1.set_processor(WanInjectionProcessor(f'block_{index}'))
        x = torch.randn(1,4,2,4,6,requires_grad=True)
        kwargs = dict(timestep=torch.tensor([930.]),encoder_hidden_states=torch.randn(1,5,16),return_dict=False)
        baseline = model(x,**kwargs)[0]
        gradient = torch.autograd.grad(baseline.square().sum(),x)[0]
        with tempfile.TemporaryDirectory() as temporary:
            rows=[]; observer=CalibrationObserver(temporary,[2,2,3],TEMPERATURES,rows)
            truth=np.zeros((1,6,2),np.float32); valid=np.ones((1,6),bool)
            observer.active=(dict(control='static',noise_label='step_09'),{0:(truth,valid,valid)})
            for b in (0,2): model.blocks[b].attn1.processor.motion_probe=observer
            rng=torch.get_rng_state().clone()
            observed=model(x,**kwargs)[0]
            observed_gradient=torch.autograd.grad(observed.square().sum(),x)[0]
            torch.testing.assert_close(observed,baseline,rtol=0,atol=0)
            torch.testing.assert_close(observed_gradient,gradient,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(observer.seen,{'block_0','block_2'})
            self.assertEqual(len(rows),32)

    def test_screen_rejects_zero_amplitude_bad_scale_and_missing_controls(self):
        rows=[]
        for state in ('clean','step_09','step_29'):
            for control in CONTROLS:
                rows.append(dict(block='block_20_attn1_processor',temperature=8.,field='soft',support='textured',
                    anchor_offset=0,noise_label=state,control=control,patches=10,finite_fraction=1.,
                    epe=0.,direction_cosine=None if control=='static' else 1.,
                    amplitude_ratio=None if control=='static' else 1.))
        self.assertTrue(screen_candidates(rows)['candidates'][0]['screen_pass'])
        expansion=next(r for r in rows if r['control']=='expand')
        for amplitude in (0.,-1.,7.):
            expansion['amplitude_ratio']=amplitude
            self.assertFalse(screen_candidates(rows)['candidates'][0]['screen_pass'])
        expansion['amplitude_ratio']=1.
        rows.pop()
        self.assertFalse(screen_candidates(rows)['candidates'][0]['screen_pass'])

    def test_plan_fixes_budget_baseline_and_confirmation_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); inputs=root/'inputs'; inputs.mkdir()
            (inputs/'manifest.csv').write_text('fixture')
            rows=[dict(clip_id=c,video_path=f'clips/{c}') for c in ('car-turn','camel')]
            with patch('benchmark.wan_calibration_pilot.direction.prepare_inputs',return_value=(inputs,rows)), \
                 patch('benchmark.wan_calibration_pilot.environment_snapshot',return_value={}):
                plan=make_plan(inputs,root/'development')
                self.assertEqual(plan['forward_passes'],20); self.assertEqual(plan['expected_rows'],2880)
                self.assertEqual(plan['model'],MODEL); self.assertEqual(plan['temperatures'],[2,4,8,16])
                self.assertTrue(plan['readout_only']); self.assertIsNone(plan['fixed_candidate'])
                followup=make_plan(inputs,root/'confirmation',clip_id='camel',noise_seed=29,candidate=(20,8))
                self.assertEqual(followup['fixed_candidate'],dict(block='block_20_attn1_processor',temperature=8.))
                with self.assertRaises(ValueError): make_plan(inputs,root/'bad',candidate=(20,8))

    def test_audit_recomputes_metrics_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
            states=[dict(noise_label='clean',sampling_index=-1,sigma=0.,timestep=0.)]+[
                dict(noise_label=f'step_{i:02d}',sampling_index=i,sigma=float(scheduler.sigmas[i]),timestep=float(scheduler.timesteps[i])) for i in (0,9,29)]
            meta=dict(model=MODEL,blocks=BLOCKS,readout_temperatures=TEMPERATURES,controls=CONTROLS,
                noise_seed=17,conditioning='',grid=[6,30,52],cpu_offload=True,mean_only=True,noise_states=states)
            (root/'metadata.json').write_text(json.dumps(meta))
            config=dict(model_key=MODEL,enable_model_cpu_offload=True,guidance_blocks=[],injection_blocks=[],
                target_prompt='',source_prompt='',scheduler='flowmatch',flow_shift=3.,num_frames=21,
                height=480,width=832,num_inference_steps=50,seed=1,reference_only=True,
                probe=False,probe_rope=False,motion_temp=2.)
            OmegaConf.save(OmegaConf.create(config),root/'suite_config.yaml')
            truth=np.ones((5,1560,2),np.float32); valid=np.ones((5,1560),bool); rows=[]
            for control in CONTROLS:
                (root/control).mkdir()
                np.savez(root/control/'truth.npz',**{f'{key}_{offset}':value for offset in (-2,0,2)
                    for key,value in [('flow',truth),('geometry',valid),('texture',valid)]})
                for block in BLOCKS:
                    for state in states:
                        folder=root/control/state['noise_label']/f'block_{block}_attn1_processor'; folder.mkdir(parents=True)
                        arrays=dict(soft=truth.copy(),hard=truth.copy(),confidence=np.ones((5,1560)),
                                    entropy=np.zeros((5,1560)),logit_std=np.zeros((5,1560)))
                        for temperature in TEMPERATURES:
                            np.savez(folder/f'{variant_name(temperature)}.npz',**arrays)
                            for offset in (-2,0,2):
                                for support in ('geometry','textured'):
                                    for field in ('hard','soft'):
                                        rows.append(dict(control=control,block=f'block_{block}_attn1_processor',
                                            noise_label=state['noise_label'],temperature=temperature,variant=variant_name(temperature),
                                            anchor_offset=offset,support=support,field=field,
                                            **field_metrics(truth,truth,valid),confidence=1.,entropy=0.))
            metrics=root/'metrics.json'; metrics.write_text(json.dumps(rows))
            (root/'complete.json').write_text(json.dumps(dict(forward_passes=20,rows=len(rows))))
            self.assertEqual(audit_calibration(root)['rows'],2880)
            self.assertTrue(make_calibration_report(root).is_file())
            rows[0]['epe']=3.; metrics.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'metric disagrees'): audit_calibration(root)
            rows[0]['epe']=0.; metrics.write_text(json.dumps(rows))
            path=root/'pan_left/step_00/block_20_attn1_processor/temperature_4.npz'
            altered=dict(arrays); altered['soft']=truth*2; np.savez(path,**altered)
            with self.assertRaisesRegex(ValueError,'Pure-noise'): audit_calibration(root)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
