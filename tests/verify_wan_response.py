"""Weight-free checks for target selectivity, branch isolation and trace rejection."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from probe_wan_response import compare_targets, fork_state, tensor_hash, flow_clean_estimate
from probe_wan_affine import build_parser
from guidance_utils.wan_affine_diagnostics import head_readouts
from benchmark.wan_response_pilot import audit_response, MODEL


class ResponseTests(unittest.TestCase):
    def test_clean_estimate_uses_flow_velocity_sign(self):
        clean=torch.randn(2,3); noise=torch.randn_like(clean)
        for sigma in (0.,.3,.93,1.):
            noisy=(1-sigma)*clean+sigma*noise
            torch.testing.assert_close(flow_clean_estimate(noisy,noise-clean,sigma),clean)

    def test_scheduler_forks_match_uninterrupted_sampling(self):
        scheduler=FlowMatchEulerDiscreteScheduler(shift=3.)
        scheduler.set_timesteps(50)
        latent=torch.ones(1,2,2,2,2)
        for i in range(9):
            latent=scheduler.step(latent*.1,scheduler.timesteps[i],latent,return_dict=False)[0]
        left,sl=fork_state(latent,scheduler); right,sr=fork_state(latent,scheduler)
        initial=tensor_hash(latent)
        for i in range(9,50):
            left=sl.step(left*.1,sl.timesteps[i],left,return_dict=False)[0]
        self.assertEqual(sr.step_index,9)
        self.assertEqual(scheduler.step_index,9)
        for i in range(9,50):
            right=sr.step(right*.1,sr.timesteps[i],right,return_dict=False)[0]
            latent=scheduler.step(latent*.1,scheduler.timesteps[i],latent,return_dict=False)[0]
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        torch.testing.assert_close(left,latent,rtol=0,atol=0)
        self.assertNotEqual(initial,tensor_hash(left))

    def test_selectivity_uses_common_locations_and_correct_sign(self):
        forward=torch.tensor([[[1.,0.],[100.,0.]]]); reverse=-forward
        targets={'forward':(forward,torch.tensor([[True,True]])),
                 'reverse':(reverse,torch.tensor([[True,False]]))}
        a=compare_targets(forward,targets); b=compare_targets(reverse,targets)
        self.assertEqual(a['forward_preference'],2)  # MSE averages dx and dy.
        self.assertEqual(b['forward_preference'],-2)
        self.assertEqual(a['common_positions'],1)
        targets['reverse']=(reverse,torch.zeros(1,2,dtype=torch.bool))
        with self.assertRaises(ValueError):compare_targets(forward,targets)

    def test_mean_only_does_not_change_baseline_readout(self):
        q=torch.randn(1,12,2,8); k=torch.randn_like(q)
        full=list(head_readouts(q,k,[2,2,3]))
        quick=list(head_readouts(q,k,[2,2,3],mean_only=True))
        self.assertEqual(len(quick),1); self.assertEqual(len(full),3)
        for key in full[0][1]:
            np.testing.assert_array_equal(full[0][1][key],quick[0][1][key])
        args=build_parser().parse_args(['-v','car-turn','--output_path','out','--model','14b','--low_vram','--mean_only','--blocks','39'])
        self.assertEqual(args.model,'14b'); self.assertTrue(args.low_vram)

    def test_audit_rejects_extra_updates_scheduler_leaks_and_wrong_targets(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            scheduler=FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
            config=dict(model_key=MODEL,enable_model_cpu_offload=True,guidance_blocks=[10],injection_blocks=[],
                guidance_mode='latent',loss_type='flow',flow_loss='mse',flow_max_disp=100.,motion_temp=2.,
                scheduler='flowmatch',flow_shift=3.,guidance_scale=5.,source_prompt='',seed=1,num_frames=21,
                height=480,width=832,threshloss=True,flow_min_conf=None,flow_region_masks=None,
                softmax_fp32=True,argmax_motion_flow=True)
            OmegaConf.save(OmegaConf.create(config),root/'suite_config.yaml')
            meta=dict(model=MODEL,branch_step=9,updates=5,learning_rate=.001,
                sigma=float(scheduler.sigmas[9]),timestep=float(scheduler.timesteps[9]),
                scheduler_class=type(scheduler).__name__,timesteps=scheduler.timesteps.tolist(),
                sigmas=scheduler.sigmas.tolist(),prefix=[{'step':i} for i in range(9)],shared_latent_sha256='same')
            (root/'metadata.json').write_text(json.dumps(meta))
            forward=np.ones((4,6,2),np.float32); reverse=-forward; mask=np.ones((4,6),bool)
            for name,flow in [('forward',forward),('reverse',reverse)]:
                np.savez(root/f'target_{name}.npz',flow=flow,mask=mask)
            fixtures={}
            for branch in ('off','forward','reverse'):
                folder=root/branch; folder.mkdir()
                (folder/'results.mp4').write_bytes(b'fixture-only')
                scores=compare_targets(torch.tensor(forward),{
                    'forward':(torch.tensor(forward),torch.tensor(mask)),
                    'reverse':(torch.tensor(reverse),torch.tensor(mask))})
                (folder/'response.json').write_text(json.dumps(dict(start_latent_sha256='same',
                    sampled_indices=list(range(9,50)),rope_unchanged=True,
                    evaluations=[dict(stage=s,**scores) for s in ('before_update','after_update','final_latent')])))
                for stage in ('before_update','after_update','final_latent'):
                    np.savez(folder/f'{stage}.npz',flow=forward)
                np.savez(folder/'reference.npz',flow=reverse if branch=='reverse' else forward,mask=mask)
                events=[dict(kind='training_reference',file='reference.npz')]
                if branch!='off':
                    events += [dict(kind='optimization',step=9,iteration=i,loss_before_update=1.,lr=.001,
                        gradient=dict(finite_fraction=1,rms=.1),update=dict(finite_fraction=1,rms=.01)) for i in range(5)]
                events += [dict(kind='sampling',step=i,sigma=float(scheduler.sigmas[i]),timestep=float(scheduler.timesteps[i]),
                    latent=dict(finite_fraction=1),guidance_update=dict(rms=.01 if branch!='off' and i==9 else 0.)) for i in range(9,50)]
                fixtures[branch]=(folder,{'config':config},events)
            def load(path):return fixtures[Path(path).name]
            with patch('probe_report.load_trace',side_effect=load):
                self.assertTrue(audit_response(root)['same_start'])
                fixtures['off'][2].append(dict(kind='optimization',step=9,iteration=0))
                with self.assertRaisesRegex(ValueError,'optimization'):audit_response(root)
                fixtures['off'][2].pop()
                sampled=next(e for e in fixtures['forward'][2] if e['kind']=='sampling')
                sigma=sampled['sigma']; sampled['sigma']=.5
                with self.assertRaisesRegex(ValueError,'schedule'):audit_response(root)
                sampled['sigma']=sigma
                np.savez(root/'reverse/reference.npz',flow=forward,mask=mask)
                with self.assertRaisesRegex(ValueError,'target'):audit_response(root)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
