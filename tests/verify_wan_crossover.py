"""CPU checks for the independent noise/time axes and replicated diagonal."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import copy
import json
import tempfile
import unittest

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from guidance_utils.wan_head_diagnostics import crossover_states,audit_heads
from guidance_utils.wan_affine_diagnostics import field_metrics
from guidance_utils.wan_amf_calibration import CONTROLS,MODEL
from benchmark import wan_head_crossover_pilot as pilot
from benchmark.wan_head_pilot import write_json
from probe_wan_affine import build_parser,noisy_input


class CrossoverTests(unittest.TestCase):
    def test_inputs_and_time_are_independent(self):
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        states=crossover_states(scheduler)
        self.assertEqual(len(states),4)
        self.assertEqual({(s['noise_sampling_index'],s['conditioning_sampling_index']) for s in states},
                         {(-1,-1),(-1,9),(9,-1),(9,9)})
        clean=torch.randn(2,3); noise=torch.randn_like(clean)
        tensors=[noisy_input(clean,noise,s['sigma']) for s in states]
        torch.testing.assert_close(tensors[0],tensors[1],rtol=0,atol=0)
        torch.testing.assert_close(tensors[2],tensors[3],rtol=0,atol=0)
        self.assertFalse(torch.equal(tensors[0],tensors[2]))
        self.assertEqual(states[1]['sigma'],0.)
        self.assertEqual(states[1]['timestep'],float(scheduler.timesteps[9]))
        self.assertEqual(states[2]['sigma'],float(scheduler.sigmas[9]))
        self.assertEqual(states[2]['timestep'],0.)

    def test_command_selects_fixed_head_and_crossover(self):
        command=pilot.command_for('car-frames')
        args=build_parser().parse_args(command[3:]+['--output_path','out'])
        self.assertEqual(args.blocks,[30]); self.assertEqual(args.readout_heads,[30])
        self.assertEqual(args.noise_steps,[9]); self.assertTrue(args.cross_noise_timestep)
        self.assertFalse(args.mean_only); self.assertIsNone(args.readout_temperatures)
        self.assertEqual(args.noise_seed,17); self.assertEqual(args.model,'14b')

    def fixture(self,root):
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        states=crossover_states(scheduler)
        records=[dict(control=c,noise_label=s['noise_label'],sigma=s['sigma'],timestep=s['timestep'],
            latent_sha256=('a' if s['noise_sampling_index']==-1 else 'b')*64) for c in CONTROLS for s in states]
        meta=dict(model=MODEL,blocks=[30],readout_heads=[30],controls=CONTROLS,noise_seed=17,conditioning='',
            temperature=2.,grid=[6,30,52],cpu_offload=True,mean_only=False,noise_states=states,
            cross_noise_timestep=True,forward_inputs=records,packages={},python='fixture',gpu='fixture',
            model_dtype='torch.bfloat16',input_base_sha256='input',noise_sha256='noise',model_revision=None,
            source_sha256={k:'same' for k in pilot.CORE_SOURCES})
        write_json(root/'metadata.json',meta)
        config=dict(model_key=MODEL,enable_model_cpu_offload=True,guidance_blocks=[],injection_blocks=[],
            target_prompt='',source_prompt='',scheduler='flowmatch',flow_shift=3.,num_frames=21,
            height=480,width=832,num_inference_steps=50,seed=1,reference_only=True,probe=False,probe_rope=False,motion_temp=2.)
        OmegaConf.save(OmegaConf.create(config),root/'suite_config.yaml')
        flow=np.ones((5,1560,2),np.float32); mask=np.ones((5,1560),bool); rows=[]
        for control in CONTROLS:
            (root/control).mkdir()
            np.savez(root/control/'truth.npz',**{f'{key}_{offset}':value for offset in (-2,0,2)
                for key,value in [('flow',flow),('geometry',mask),('texture',mask)]})
            for s in states:
                folder=root/control/s['noise_label']/'block_30_attn1_processor'; folder.mkdir(parents=True)
                arrays=dict(soft=flow,hard=flow,confidence=np.ones((5,1560)),entropy=np.zeros((5,1560)),logit_std=np.zeros((5,1560)))
                for variant in ('mean_logits','head_30'):
                    np.savez(folder/f'{variant}.npz',**arrays)
                    for offset in (-2,0,2):
                        for support in ('geometry','textured'):
                            for field in ('hard','soft'):
                                rows.append(dict(control=control,**s,block='block_30_attn1_processor',variant=variant,
                                    anchor_offset=offset,support=support,field=field,**field_metrics(flow,flow,mask),confidence=1.,entropy=0.))
        write_json(root/'metrics.json',rows); write_json(root/'complete.json',dict(forward_passes=20,rows=480))
        return meta

    def test_audit_rejects_coupling_missing_inputs_and_wrong_timestep(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); meta=self.fixture(root)
            audit=lambda:audit_heads(root,[30],[30],17,cross_noise_timestep=True)
            result=audit(); self.assertEqual(result['rows'],480); self.assertTrue(result['paired_inputs_equal'])
            self.assertIsNone(result['pure_noise_equal'])
            with self.assertRaises(ValueError): audit_heads(root,[30],[30],17)
            mutations=[lambda m:m['forward_inputs'].pop(),
                lambda m:m['forward_inputs'][1].update(latent_sha256='c'*64),
                lambda m:m['forward_inputs'][1].update(timestep=0.)]
            for mutate in mutations:
                broken=copy.deepcopy(meta); mutate(broken); write_json(root/'metadata.json',broken)
                with self.assertRaises(ValueError): audit()
            broken=copy.deepcopy(meta)
            for r in broken['forward_inputs']: r['latent_sha256']='a'*64
            write_json(root/'metadata.json',broken)
            with self.assertRaisesRegex(ValueError,'noise manipulation'): audit()

    def test_replication_checks_arrays_and_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'new'; root.mkdir(); meta=self.fixture(root)
            baseline=Path(tmp)/'old'; baseline.mkdir(); write_json(baseline/'metadata.json',meta)
            for new,old in pilot.DIAGONAL.items():
                for control in CONTROLS:
                    for variant in ('mean_logits','head_30'):
                        suffix=Path('block_30_attn1_processor')/f'{variant}.npz'
                        target=baseline/control/old/suffix; target.parent.mkdir(parents=True,exist_ok=True)
                        target.write_bytes((root/control/new/suffix).read_bytes())
            report=pilot.compare_diagonal(root,baseline)
            self.assertTrue(report['replication_pass']); self.assertEqual(report['identical'],20)
            pilot.make_report(root,dict(interpretation_ready=True,replication=report))
            broken=copy.deepcopy(meta); broken['gpu']='different'; write_json(baseline/'metadata.json',broken)
            self.assertFalse(pilot.compare_diagonal(root,baseline)['replication_pass'])
            write_json(baseline/'metadata.json',meta)
            path=baseline/'static/clean/block_30_attn1_processor/head_30.npz'
            with np.load(path) as z: arrays={k:z[k] for k in z.files}
            arrays['soft']=arrays['soft']*2; np.savez(path,**arrays)
            report=pilot.compare_diagonal(root,baseline)
            self.assertEqual(report['identical'],19); self.assertFalse(report['replication_pass'])


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
