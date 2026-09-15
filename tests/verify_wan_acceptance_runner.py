"""Exercise the bounded acceptance runner using a tiny random Wan and real VAE."""
import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
from verify_wan_decisive import RealSamplerTests
from benchmark.wan_port_acceptance import AcceptanceRun
from guidance_utils.wan_transformer import ControlledWanTransformer


def main(cpu_offload=False):
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory)
        g,_=RealSamplerTests().build('amf',str(root/'fixture'))
        # Use the production methods directly; this fixture's experiment mixin
        # is unnecessary for acceptance and has its own dose-count protocol.
        from motion_guidance_wan import WanGuidance
        g.__class__=WanGuidance
        g.num_frames,g.num_inference_steps,g.resolution=5,12,(48,32)
        g.config.height,g.config.width,g.config.num_frames=32,48,5
        g.config.optimization_steps=1
        g.config.save_format='mp4'
        g.scheduler=UniPCMultistepScheduler(prediction_type='flow_prediction',use_flow_sigmas=True,flow_shift=3.)
        g.scheduler.set_timesteps(12,device=g.device)
        g.timesteps=g.scheduler.timesteps
        g.guidance_steps=[9]; g.injection_steps=[]; g.lr_by_step={9:.001}
        g.vae=AutoencoderKLWan(base_dim=4,z_dim=4,dim_mult=[1,1,1,1],num_res_blocks=1,
                              latents_mean=[0.]*4,latents_std=[1.]*4).to(g.device).requires_grad_(False)
        g.pipe=WanPipeline(tokenizer=None,text_encoder=None,vae=g.vae,
                           transformer=g.transformer,scheduler=g.scheduler)
        g.pipe.set_progress_bar_config(disable=True)
        if cpu_offload:
            g.pipe.enable_model_cpu_offload(device=g.device)
            g.config.enable_model_cpu_offload=True
        reference=root/'reference'; reference.mkdir()
        rng=np.random.default_rng(19)
        for i in range(5):
            Image.fromarray(rng.integers(0,256,(32,48,3),dtype=np.uint8)).save(reference/f'{i:05d}.png')
        g.config.video_path=str(reference)
        run=AcceptanceRun(root/'run',reference,'tiny test',model=g)
        run.parity()
        if cpu_offload:
            assert hasattr(g.transformer, '_hf_hook'), 'Native parity removed the offload hook'
            assert g.transformer._old_forward.__func__ is ControlledWanTransformer.forward
        run.compare({'credible_motion':True,'notes':'Synthetic test fixture: exercises the gate, no quality claim.'})
        trace=json.loads((run.root/'forward/step_trace.json').read_text())
        assert trace['same_solver_state'] and trace['actual_sampler_received_optimized_latent']
        assert trace['update_fp32']['rms']>0 and trace['recomputed_cfg_prediction']['rms']>0
        assert trace['next_scheduler_state']['rms']>0
        assert all((run.root/arm/'final.mp4').is_file() for arm in ('native','off','forward','reverse'))
        assert json.loads((run.root/'status.json').read_text())['success'] is False
        try: run.compare({'credible_motion':True,'notes':'retry'})
        except RuntimeError: pass
        else: raise AssertionError('Run budget was not enforced')
        print('PASS: full native parity, fresh solver-state trace, two-arm budget, decoded files; tiny model only.')


if __name__=='__main__': main(cpu_offload='--cpu-offload' in sys.argv)
