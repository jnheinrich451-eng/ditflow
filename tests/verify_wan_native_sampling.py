"""Direct sampling parity against the installed WanPipeline; no pretrained claim."""
import json
import sys
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler, WanPipeline
from diffusers.models.transformers.transformer_wan import WanTransformer3DModel
from guidance_utils.wan_modules import WanInjectionProcessor
from guidance_utils.wan_transformer import ControlledWanTransformer
from motion_guidance_wan import WanGuidance


def parity(device='cuda', dtype=torch.bfloat16, steps=4):
    torch.manual_seed(123)
    config = dict(patch_size=(1, 2, 2), num_attention_heads=2, attention_head_dim=8,
                  in_channels=4, out_channels=4, text_dim=16, freq_dim=16, ffn_dim=32,
                  num_layers=2, cross_attn_norm=True, qk_norm='rms_norm_across_heads',
                  eps=1e-6, rope_max_seq_len=64)
    # Use the real loader: Wan keeps time_embedder/norm/modulation parameters
    # in FP32. model.to(bfloat16) would erase this and miss autocast regressions.
    with tempfile.TemporaryDirectory() as directory:
        WanTransformer3DModel(**config).save_pretrained(directory)
        native = WanTransformer3DModel.from_pretrained(directory, torch_dtype=dtype).to(device).eval()
        controlled = ControlledWanTransformer.from_pretrained(directory, torch_dtype=dtype).to(device).eval()
    for i, block in enumerate(controlled.blocks):
        block.attn1.set_processor(WanInjectionProcessor(str(i)))
    vae = AutoencoderKLWan(base_dim=4, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                          latents_mean=[0.] * 4, latents_std=[1.] * 4).to(device)
    embeds = torch.randn(2, 8, 16, device=device, dtype=dtype)
    initial = torch.randn(1, 4, 3, 4, 4, device=device)
    rows = []
    for name, scheduler in (
        ('flowmatch', FlowMatchEulerDiscreteScheduler(shift=3)),
        ('unipc', UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, flow_shift=3)),
    ):
        pipe = WanPipeline(tokenizer=None, text_encoder=None, vae=vae,
                           transformer=native, scheduler=scheduler)
        pipe.set_progress_bar_config(disable=True)
        states = []
        def capture(pipe, i, t, values):
            states.append(values['latents'].clone())
            return values
        with torch.no_grad():
            pipe(prompt_embeds=embeds[1:2], negative_prompt_embeds=embeds[:1],
                 latents=initial.clone(), height=32, width=32, num_frames=9,
                 num_inference_steps=steps, guidance_scale=5., output_type='latent',
                 callback_on_step_end=capture)
        owner = WanGuidance.__new__(WanGuidance)
        torch.nn.Module.__init__(owner)
        owner.transformer, owner.device, owner.dtype = controlled, torch.device(device), dtype
        owner.scheduler = type(scheduler).from_config(scheduler.config)
        owner.scheduler.set_timesteps(steps, device=device)
        owner.timesteps = owner.scheduler.timesteps
        owner._guidance_scale = 5.
        owner.probe = SimpleNamespace(phase=lambda *a, **kw: nullcontext(), prediction=lambda *a: None)
        x = initial.clone()
        for i in range(steps):
            x = owner.denoise_step(x, i, embeds)
            delta = x.float() - states[i].float()
            rows.append(dict(scheduler=name, step=i, dtype=str(dtype),
                             max_abs=float(delta.abs().max()), rms=float(delta.square().mean().sqrt())))
    return rows


def injection_independence():
    from verify_wan_decisive import RealSamplerTests
    for guided, injected in ((False,False),(False,True),(True,False),(True,True)):
        with tempfile.TemporaryDirectory() as directory:
            g, _ = RealSamplerTests().build('amf' if guided else 'off',directory)
            g.scheduler.set_timesteps(2,device=g.device)
            g.timesteps=g.scheduler.timesteps
            g.guidance_steps=g.injection_steps=[0]
            g.lr_by_step={0:.001}
            # Deliberately overlap guidance/injection blocks.
            g.config.injection_blocks=[1] if injected else []
            g.config.optimization_steps=1
            forward, denoise=g._forward_transformer,g.denoise_step
            reference_cache={}
            calls=[]
            def observe_forward(*args,**kwargs):
                assert not any(b.attn1.processor.inject_kv for b in g.transformer.blocks)
                result=forward(*args,**kwargs)
                if kwargs.get('stop_at')=='injection':
                    reference_cache['key']=g.transformer.blocks[1].attn1.processor.key.clone()
                return result
            def observe_denoise(x,i,embeds,rope=None):
                processor=g.transformer.blocks[1].attn1.processor
                assert processor.inject_kv == (injected and i==0)
                if processor.inject_kv:
                    torch.testing.assert_close(processor.key,reference_cache['key'],rtol=0,atol=0)
                calls.append(i)
                return denoise(x,i,embeds,rope)
            g._forward_transformer,g.denoise_step=observe_forward,observe_denoise
            g.run(custom_name='test')
            assert calls==[0,1]
    return dict(combinations=4, overlapping_blocks=True, cache_overwrite=False, leakage_to_next_step=False)


if __name__ == '__main__':
    rows = parity()
    print(json.dumps(rows, indent=2))
    # Same weights, kernels, inputs, schedule and solver history should agree exactly.
    if any(row['max_abs'] != 0 for row in rows):
        raise SystemExit('Native pipeline parity failed (tiny model; no motion-quality verdict).')
    print(json.dumps(injection_independence()))
