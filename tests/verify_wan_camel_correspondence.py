"""Tiny real-model check of the six-forward diagnostic; no quality claim."""
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanPipeline
from verify_wan_decisive import RealSamplerTests
from motion_guidance_wan import WanGuidance
from benchmark.wan_camel_correspondence import capture, digest, controls
from probe_wan_response import tensor_hash


def main():
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        g,_=RealSamplerTests().build('amf',str(root/'fixture'))
        g.__class__=WanGuidance
        g.num_frames,g.resolution=5,(48,32)
        g.config.video_path='original input'
        g.config.model_key='tiny random test fixture'
        g.vae=AutoencoderKLWan(base_dim=4,z_dim=4,dim_mult=[1,1,1,1],num_res_blocks=1,
                              latents_mean=[0.]*4,latents_std=[1.]*4).to(g.device).requires_grad_(False)
        g.pipe=WanPipeline(tokenizer=None,text_encoder=None,vae=g.vae,transformer=g.transformer,scheduler=g.scheduler)
        g.pipe.enable_model_cpu_offload(device=g.device)
        rng=np.random.default_rng(7)
        frames=rng.integers(0,256,(5,32,48,3),dtype=np.uint8)
        plan=dict(controls=['forward','reverse','static'],states=['clean','step_09'],block=1,
                  grid=[2,2,3],noise_seed=29,input_sha256={},roi_method='test',rgb_metric='test',first_pair_caveat='test',
                  checkpoint_revision=g.config.model_key,conditioning_sha256=tensor_hash(g.guidance_embeds),
                  sigmas=g.scheduler.sigmas.tolist(),timesteps=g.timesteps.cpu().tolist())
        for name,sequence in controls(frames).items():
            folder=root/'controls'/name; folder.mkdir(parents=True)
            for i,frame in enumerate(sequence):
                dest=folder/f'{i:05d}.png'; Image.fromarray(frame).save(dest)
                plan['input_sha256'][str(dest.relative_to(root))]=digest(dest)
        arrays={name+suffix:np.zeros((1,2,3,2),dtype=np.float32) if suffix=='_flow' else np.ones((1,2,3),dtype=bool)
                for name in plan['controls'] for suffix in ('_flow','_valid')}
        arrays.update(subject_region=np.ones((2,3),dtype=bool),background_region=np.ones((2,3),dtype=bool))
        np.savez_compressed(root/'rgb_motion.npz',**arrays)
        plan['input_sha256']['rgb_motion.npz']=digest(root/'rgb_motion.npz')
        (root/'plan.json').write_text(json.dumps(plan))
        old_path,old_output=g.config.video_path,g.output_path
        with (patch.object(g,'_forward_transformer',wraps=g._forward_transformer) as forwards,
              patch.object(g.vae,'encode',wraps=g.vae.encode) as encodes,
              patch.object(g.scheduler,'step',side_effect=AssertionError('Diagnostic must not sample'))):
            report=capture(g,root)
            assert forwards.call_count==6 and encodes.call_count==3
        assert g.config.video_path==old_path and g.output_path==old_output
        assert len(report['rows'])==36
        assert len(json.loads((root/'readouts/metadata.json').read_text())['events'])==6
        for path in (root/'readouts').glob('*.npz'):
            with np.load(path) as data:
                assert data['hard'].shape==(4,6,2) and data['soft'].shape==(4,6,2)
                assert data['query'].shape==data['key'].shape==(1,12,4,8)
                assert all(np.isfinite(data[key]).all() for key in data.files)
        try: capture(g,root)
        except RuntimeError as error: assert 'budget' in str(error)
        else: raise AssertionError('Second capture exceeded budget')
        print('PASS: 3 real VAE encodes, 6 real truncated forwards with CPU offload, no scheduler steps, saved Q/K, replay budget.')


if __name__=='__main__': main()
