"""Small real-model check of branching and solver reuse; not motion evidence."""
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from diffusers import AutoencoderKLWan, WanPipeline, UniPCMultistepScheduler
from verify_wan_decisive import RealSamplerTests
from motion_guidance_wan import WanGuidance
from benchmark.wan_centered_pilot import trace_pilot, decode_pilot, restore_state, step


def main():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        g, _ = RealSamplerTests().build('amf', str(root / 'fixture'))
        g.__class__ = WanGuidance
        g.num_frames, g.num_inference_steps, g.resolution = 5, 12, (48, 32)
        g.config.height, g.config.width, g.config.num_frames = 32, 48, 5
        g.config.source_prompt = ''
        g.config.save_format = 'mp4'
        g.scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3.)
        g.scheduler.set_timesteps(12, device=g.device)
        g.timesteps = g.scheduler.timesteps
        g.vae = AutoencoderKLWan(base_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                                latents_mean=[0.]*4, latents_std=[1.]*4).to(g.device).requires_grad_(False)
        g.pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=g.vae, transformer=g.transformer, scheduler=g.scheduler)
        g.pipe.enable_model_cpu_offload(device=g.device)
        g.config.enable_model_cpu_offload = True
        frames = list(np.random.default_rng(19).integers(0, 256, (5, 32, 48, 3), dtype=np.uint8))
        output = root / 'run'
        traces = trace_pilot(g, frames, output, index=9, updates=1)
        assert len(traces) == 2 and all(t['passed'] for t in traces)
        # Independently replay the off branch from the saved full prefix state.
        prefix = torch.load(output / 'prefix.pt', weights_only=False)
        off = torch.load(output / 'off.pt', weights_only=False)
        g.scheduler.__dict__ = restore_state(prefix['scheduler_state'], g.device)
        restored_off = restore_state(off['scheduler_state'], g.device)
        replay, _ = step(g, prefix['latent'].to(g.device), prefix['index'])
        torch.testing.assert_close(replay.cpu(), off['latent'], atol=0, rtol=0)
        assert len(g.scheduler.model_outputs) == len(restored_off['model_outputs'])
        for actual, expected in zip(g.scheduler.model_outputs, restored_off['model_outputs']):
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        try:
            trace_pilot(g, frames, output, index=9, updates=1)
        except RuntimeError as error:
            assert 'budget' in str(error)
        else:
            raise AssertionError('Repeated trace was not blocked')
        decode_pilot(g, output)
        assert all((output / arm / 'final.mp4').is_file() for arm in ('off', 'forward', 'reverse'))
        assert (output / 'comparison.mp4').is_file()
        assert json.loads((output / 'status.json').read_text())['port_success'] is False
        try:
            decode_pilot(g, output)
        except RuntimeError as error:
            assert 'budget' in str(error)
        else:
            raise AssertionError('Repeated suffix suite was not blocked')
        print('PASS: actual tiny-model gradients, identical solver replay, three decoded suffixes, offload and budget guards. No quality claim.')


if __name__ == '__main__':
    main()
