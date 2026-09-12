"""CPU checks for a paired multi-block readout, without pretrained weights."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

from benchmark.wan_response_pilot import make_plan, run_stage, audit_readout, MODEL, CONTROLS
from guidance_utils.wan_affine_diagnostics import AffineObserver


class BlockReadoutTests(unittest.TestCase):
    def test_plan_has_one_shared_forward_command_and_no_generations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); inputs=root/'inputs'; inputs.mkdir()
            (inputs/'manifest.csv').write_text('fixture')
            rows=[dict(clip_id='car-turn',prompt='A truck turns',video_path='clips/car-turn')]
            with patch('benchmark.wan_response_pilot.direction.prepare_inputs',return_value=(inputs,rows)), \
                 patch('benchmark.wan_response_pilot.environment_snapshot',return_value={}):
                plan=make_plan(inputs,root/'run',readout_blocks=[10,20,30],readout_only=True)
            command=plan['readout_command']
            self.assertEqual(command[command.index('--blocks')+1:command.index('--controls')],['10','20','30'])
            self.assertEqual(command[command.index('--model')+1],'14b')
            self.assertNotIn('response_command',plan)
            with self.assertRaisesRegex(ValueError,'no response generations'):
                run_stage(root/'run','response')
        with self.assertRaises(ValueError):make_plan('unused','unused',readout_blocks=[10,20,30])
        with self.assertRaises(ValueError):make_plan('unused','unused',readout_blocks=[10,40],readout_only=True)

    def test_two_blocks_observed_in_one_forward_without_changing_output(self):
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.wan_modules import WanInjectionProcessor
        model=ControlledWanTransformer(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=8,
            in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=32,num_layers=3,
            cross_attn_norm=True,qk_norm='rms_norm_across_heads',eps=1e-6,rope_max_seq_len=64).eval()
        for index,block in enumerate(model.blocks):
            block.attn1.set_processor(WanInjectionProcessor(f'block_{index}'))
        x=torch.randn(1,4,2,4,6); text=torch.randn(1,5,16)
        kwargs=dict(timestep=torch.tensor([930.]),encoder_hidden_states=text,return_dict=False)
        with torch.no_grad():baseline=model(x,**kwargs)[0]
        with tempfile.TemporaryDirectory() as temporary:
            rows=[]; observer=AffineObserver(temporary,[2,2,3],2.,rows,mean_only=True)
            truth=np.zeros((1,6,2),np.float32); valid=np.ones((1,6),bool)
            observer.active=(dict(control='static',noise_label='step_09'),{0:(truth,valid,valid)})
            for b in (0,2):model.blocks[b].attn1.processor.motion_probe=observer
            rng=torch.get_rng_state().clone()
            with torch.no_grad():observed=model(x,**kwargs)[0]
            torch.testing.assert_close(observed,baseline,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(observer.seen,{'block_0','block_2'})
            self.assertEqual(len(rows),8)
            self.assertTrue(all(r['variant']=='mean_logits' for r in rows))

    def test_audit_checks_each_block_and_pure_noise_independently(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); blocks=[10,20,30]
            states=[dict(sampling_index=i,noise_label=s) for i,s in [(-1,'clean'),(0,'step_00'),(9,'step_09'),(29,'step_29')]]
            meta=dict(model=MODEL,blocks=blocks,controls=CONTROLS,mean_only=True,cpu_offload=True,
                conditioning='',grid=[6,30,52],noise_states=states)
            (root/'metadata.json').write_text(json.dumps(meta))
            rows=[]
            for block in blocks:
                name=f'block_{block}_attn1_processor'
                for control in CONTROLS:
                    for state in states:
                        folder=root/control/state['noise_label']/name; folder.mkdir(parents=True)
                        # Different blocks MAY have different pure-noise fields.
                        np.savez_compressed(folder/'mean_logits.npz',soft=np.full((5,1560,2),block,dtype=np.float32))
                        for offset in (-2,0,2):
                            for support in ('geometry','textured'):
                                for field in ('hard','soft'):
                                    rows.append(dict(block=name,control=control,noise_label=state['noise_label'],
                                        anchor_offset=offset,support=support,field=field,variant='mean_logits',finite_fraction=1))
            (root/'metrics.json').write_text(json.dumps(rows))
            (root/'complete.json').write_text(json.dumps(dict(forward_passes=20,rows=len(rows))))
            audit=audit_readout(root,blocks)
            self.assertEqual(audit['rows'],720); self.assertEqual(audit['forward_passes'],20)
            with self.assertRaises(ValueError):audit_readout(root)  # Default historical audit stays block 10 only.
            original=rows[-1]['block']; rows[-1]['block']='block_10_attn1_processor'
            (root/'metrics.json').write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'Incomplete or duplicated'):audit_readout(root,blocks)
            rows[-1]['block']=original; (root/'metrics.json').write_text(json.dumps(rows))
            np.savez_compressed(root/'pan_left/step_00/block_20_attn1_processor/mean_logits.npz',soft=np.zeros((5,1560,2)))
            with self.assertRaisesRegex(ValueError,'block_20.*pure-noise'):audit_readout(root,blocks)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
