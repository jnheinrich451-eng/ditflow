"""CPU tests for the explicitly experimental region-weighted AMF loss."""
import sys
from pathlib import Path

# Tests import repo modules by their root names; make `python tests/<file>.py` work from the root checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from guidance_utils.wan_region_guidance import load_reference_regions, region_balanced_mse


class RegionGuidanceTests(unittest.TestCase):
    def test_background_size_does_not_dilute_subject_gradient(self):
        for background_count in (1,99):
            prediction=torch.ones(1,background_count+1,2,requires_grad=True)
            target=torch.zeros_like(prediction)
            valid=torch.ones(prediction.shape[:-1],dtype=torch.bool)
            fg=torch.zeros_like(valid); fg[0,0]=True
            loss,_=region_balanced_mse(prediction,target,valid,fg)
            loss.backward()
            self.assertEqual(float(loss.detach()),1.)
            torch.testing.assert_close(prediction.grad[0,0],torch.tensor([.5,.5]))
            self.assertAlmostEqual(float(prediction.grad[0,1:].sum()),1.,places=6)

    def test_validity_mask_preserved_and_missing_region_rejected(self):
        pred=torch.tensor([[[1.,1.],[2.,2.],[100.,100.]]],requires_grad=True)
        ref=torch.zeros_like(pred); valid=torch.tensor([[True,True,False]]); fg=torch.tensor([[True,False,True]])
        loss,_=region_balanced_mse(pred,ref,valid,fg)
        self.assertEqual(float(loss.detach()),2.5)
        loss.backward(); torch.testing.assert_close(pred.grad[0,2],torch.zeros(2))
        with self.assertRaises(ValueError):region_balanced_mse(pred,ref,valid,torch.zeros_like(fg))

    def test_region_masks_use_source_major_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            for i in range(9):
                mask=np.zeros((8,16),dtype=np.uint8)
                mask[:,8:] = 1 if i==4 else 0
                if i!=4:mask[:,:8]=1
                Image.fromarray(mask).save(root/f'{i:05d}.png')
            masks,provenance=load_reference_regions(root,9,3,1,2,'cpu')
            self.assertEqual(provenance['anchor_frames'],[0,4,8])
            expected=torch.tensor([[True,False]]*3+[[False,True]]*3+[[True,False]]*3)
            torch.testing.assert_close(masks,expected)
            (root/'00008.png').unlink()
            with self.assertRaises(ValueError):load_reference_regions(root,9,3,1,2,'cpu')

    def test_actual_loss_method_preserves_baseline_and_uses_variant(self):
        from motion_guidance_wan import WanGuidance
        g=WanGuidance.__new__(WanGuidance); torch.nn.Module.__init__(g)
        g.config=OmegaConf.create(dict(guidance_blocks=[0],flow_loss='mse'))
        proc=SimpleNamespace(block_name='block_0_attn1_processor')
        g.transformer=SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=proc))])
        g.guidance_embeds=torch.zeros(2,1,1); g.device='cpu'
        g._forward_transformer=lambda x,*args,**kwargs:setattr(proc,'query',x)
        g._amf=lambda processor:processor.query
        g._clear_kv=lambda blocks:None
        g.probe=SimpleNamespace(context=None,training_flow=lambda *args:None)
        reference=torch.arange(24,dtype=torch.float32).reshape(4,3,2)/10
        valid=torch.tensor([[1,1,0],[1,1,1],[1,1,1],[0,1,1]],dtype=torch.bool)
        fg=torch.tensor([[1,0,0]]*4,dtype=torch.bool)
        g.motion_attn_features={proc.block_name:reference}; g.motion_attn_masks={proc.block_name:valid}; g.motion_regions=fg
        for experimental in (False,True):
            g.config.flow_region_masks='synthetic' if experimental else None
            x=torch.zeros_like(reference,requires_grad=True)
            loss=g.compute_motion_flow_loss(x,torch.tensor(1.))
            expected=(region_balanced_mse(x,reference,valid,fg)[0] if experimental else torch.nn.functional.mse_loss(reference[valid],x[valid]))
            torch.testing.assert_close(loss,expected,rtol=0,atol=0)
            actual_grad=torch.autograd.grad(loss,x,retain_graph=True)[0]
            expected_grad=torch.autograd.grad(expected,x)[0]
            torch.testing.assert_close(actual_grad,expected_grad,rtol=0,atol=0)


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
