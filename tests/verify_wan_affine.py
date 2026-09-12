"""CPU checks of affine geometry, AMF readout, noise pairing and report semantics."""
import sys
from pathlib import Path

# Tests import repo modules by their root names; make `python tests/<file>.py` work from the root checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from guidance_utils.wan_affine_diagnostics import (
    affine_controls, affine_truth, texture_support, field_metrics, head_readouts, AffineObserver, make_affine_report,
)
from probe_wan_affine import build_parser, noise_states, noisy_input


class AffineTests(unittest.TestCase):
    def setUp(self):
        self.first = np.random.default_rng(7).integers(0,256,(96,160,3),dtype=np.uint8)

    def test_rendered_translation_and_no_wrap(self):
        for kind, shift in [('pan_right',4),('pan_left',-4)]:
            frames, info = affine_controls(self.first,kind)
            if shift > 0:
                np.testing.assert_array_equal(frames[1][:,4:],self.first[:,:-4]); self.assertFalse(frames[1][:,:4].any())
            else:
                np.testing.assert_array_equal(frames[1][:,:-4],self.first[:,4:]); self.assertFalse(frames[1][:,-4:].any())
            truth, mask, anchors = affine_truth(info,[6,6,10],margin_patches=1)
            np.testing.assert_allclose(truth, np.broadcast_to([shift/4,0],truth.shape),atol=1e-6)
            self.assertEqual(anchors,[0,4,8,12,16,20]); self.assertTrue(mask.any())

    def test_rotation_and_scale_direction_independent_geometry(self):
        for kind, expected_sign in [('rotate_cw',1),('rotate_ccw',-1),('expand',1),('contract',-1)]:
            _, info = affine_controls(self.first,kind)
            field, mask, _ = affine_truth(info,[6,6,10],margin_patches=1)
            # At a patch above the image center: clockwise moves right; expansion moves up.
            at_top = field[0].reshape(6,10,2)[1,4]
            if kind.startswith('rotate'):
                self.assertGreater(at_top[0]*expected_sign,0)
            else:
                self.assertLess(at_top[1]*expected_sign,0)
            # Independent first-pair closed form around the patch-grid center.
            y,x=np.meshgrid(np.arange(6)+.5-3,np.arange(10)+.5-5,indexing='ij')
            p=np.stack([x,y],-1).reshape(-1,2)
            if kind.startswith('rotate'):
                angle=np.deg2rad(expected_sign*4)
                expected=p@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])-p
            else:expected=p*(1.25**(expected_sign*.2)-1)
            np.testing.assert_allclose(field[0],expected,atol=1e-6)
            self.assertTrue(mask.any())

    def test_rendered_marker_matches_forward_rotation_and_scale(self):
        marker=np.zeros_like(self.first); marker[44:48,118:122]=255
        yy,xx=np.indices(marker.shape[:2]); source=np.array([119.5,45.5,1.])
        for kind in ('rotate_cw','rotate_ccw','expand','contract'):
            frames,info=affine_controls(marker,kind)
            weight=frames[-1][:,:,0].astype(float)
            observed=np.array([(xx*weight).sum()/weight.sum(),(yy*weight).sum()/weight.sum()])
            # Convert integer pixel centers to PIL edge coordinates and back.
            expected=np.asarray(info['base_to_frame'][-1])@(source+np.array([.5,.5,0]))
            np.testing.assert_allclose(observed,expected[:2]-.5,atol=.2)

    def test_inverse_pair_composition_and_support(self):
        for kind in ('rotate_cw','expand','contract'):
            _,info=affine_controls(self.first,kind)
            matrices=np.asarray(info['base_to_frame'])
            point=np.array([80.,40.,1.]); source=matrices[8]@point
            relative=matrices[12]@np.linalg.inv(matrices[8])
            np.testing.assert_allclose(relative@source,matrices[12]@point)
            _,valid,anchors=affine_truth(info,[6,6,10],anchor_offset=2,margin_patches=1)
            self.assertEqual(anchors,[2,6,10,14,18,20]); self.assertFalse(valid.reshape(5,6,10)[:,0].any())
        with self.assertRaises(ValueError):affine_truth(info,[5,6,10])

    def test_metrics_do_not_reward_zero_or_reversed_flow(self):
        truth=np.broadcast_to([1.,0.],(2,6,2)); valid=np.ones((2,6),bool)
        good=field_metrics(truth,truth,valid); reverse=field_metrics(-truth,truth,valid)
        zero=field_metrics(np.zeros_like(truth),truth,valid)
        self.assertEqual(good['epe'],0); self.assertEqual(good['amplitude_ratio'],1)
        self.assertEqual(reverse['direction_cosine'],-1); self.assertEqual(zero['direction_cosine'],0)
        self.assertEqual(zero['positive_projection_fraction'],0)
        static=field_metrics(truth,np.zeros_like(truth),valid)
        self.assertIsNone(static['direction_cosine']); self.assertEqual(static['epe'],1)
        self.assertIsNone(field_metrics(truth,truth,np.zeros_like(valid))['epe'])
        frames,_=affine_controls(np.full_like(self.first,128),'static')
        self.assertFalse(texture_support(frames,[6,6,10]).any())

    def test_mean_logits_oracle_and_no_rng_or_tensor_mutation(self):
        torch.manual_seed(3); q=torch.randn(1,12,2,8); k=torch.randn_like(q)
        saved=q.clone(); rng=torch.get_rng_state().clone()
        outputs=dict(head_readouts(q,k,[2,2,3],2.))
        q0=q[0,:6]; k1=k[0,6:]
        logits=torch.stack([q0[:,h]@k1[:,h].T/(8**.5) for h in range(2)]).mean(0)
        yy,xx=torch.meshgrid(torch.arange(2),torch.arange(3),indexing='ij')
        xy=torch.stack([xx.flatten(),yy.flatten()],-1).float()
        expected=(logits*2).softmax(-1)@xy-xy
        np.testing.assert_allclose(outputs['mean_logits']['soft'][0],expected.numpy(),atol=3e-7)
        torch.testing.assert_close(q,saved,atol=0,rtol=0); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(set(outputs),{'mean_logits','head_00','head_01'})

    def test_noise_endpoints_and_scheduler_indices(self):
        latent=torch.randn(1,2,3); other=torch.randn_like(latent); noise=torch.randn_like(latent)
        torch.testing.assert_close(noisy_input(latent,noise,0),latent)
        torch.testing.assert_close(noisy_input(latent,noise,1),noisy_input(other,noise,1))
        scheduler=SimpleNamespace(sigmas=torch.tensor([1.,.8,.2]),timesteps=torch.tensor([1000.,800.,200.]))
        states=noise_states(scheduler,[0,2]); self.assertEqual(states[0]['timestep'],0)
        self.assertEqual(states[1]['sigma'],1); self.assertEqual(states[2]['timestep'],200)
        args=build_parser().parse_args(['-v','input','--output_path','out'])
        self.assertEqual(args.blocks,[10]); self.assertEqual(args.prompt,'')

    def test_observer_and_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); rows=[]; observer=AffineObserver(root,[2,2,3],2.,rows)
            truth=np.ones((1,6,2),np.float32); mask=np.ones((1,6),bool)
            observer.active=(dict(control='expand',noise_label='clean',sigma=0.,timestep=0.,sampling_index=-1),{0:(truth,mask,mask)})
            q=torch.randn(1,12,2,8)
            observer.attention('block_10',q,q)
            self.assertEqual(len(rows),12); self.assertEqual(observer.seen,{'block_10'})
            with self.assertRaises(ValueError):observer.attention('block_10',q,q)
            (root/'metrics.json').write_text(json.dumps(rows)); (root/'metadata.json').write_text('{}')
            report=make_affine_report(root)
            self.assertIn('mean_logits', (root/'metrics.csv').read_text())
            self.assertIn('not 3D turns',report.read_text(encoding='utf-8'))
            self.assertTrue((root/'expand/clean/block_10/head_01.npz').is_file())

    def test_observer_preserves_actual_wan_forward(self):
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.wan_modules import WanInjectionProcessor
        model=ControlledWanTransformer(patch_size=(1,2,2),num_attention_heads=2,attention_head_dim=8,
            in_channels=4,out_channels=4,text_dim=16,freq_dim=16,ffn_dim=32,num_layers=2,
            cross_attn_norm=True,qk_norm='rms_norm_across_heads',eps=1e-6,rope_max_seq_len=64).eval()
        for index,block in enumerate(model.blocks):
            block.attn1.set_processor(WanInjectionProcessor(f'block_{index}'))
        x=torch.randn(1,4,2,4,6); text=torch.randn(1,5,16)
        kwargs=dict(timestep=torch.tensor([500.]),encoder_hidden_states=text,return_dict=False)
        with torch.no_grad():expected=model(x,**kwargs)[0]
        with tempfile.TemporaryDirectory() as temporary:
            rows=[]; observer=AffineObserver(temporary,[2,2,3],2.,rows)
            mask=np.ones((1,6),bool); truth=np.zeros((1,6,2),np.float32)
            observer.active=(dict(control='static',noise_label='step_09',sigma=.9,timestep=500.,sampling_index=9),{0:(truth,mask,mask)})
            model.blocks[1].attn1.processor.motion_probe=observer
            rng=torch.get_rng_state().clone()
            with torch.no_grad():actual=model(x,**kwargs)[0]
            torch.testing.assert_close(actual,expected,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state())); self.assertEqual(observer.seen,{'block_1'})


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
