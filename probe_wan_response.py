"""14B car-turn impulse experiment: one shared sampled latent, three continuations.

This is a diagnostic intervention, not a replacement for the 50-update baseline.
Uses the production AMF loss and Adam update. Does not modify generator code.
"""
import argparse
import copy
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

BRANCHES = ('off', 'forward', 'reverse')


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def fork_state(latent, scheduler):
    """Independent tensor AND scheduler history; no scheduler index leaks between branches."""
    return latent.detach().clone(), copy.deepcopy(scheduler)


def flow_clean_estimate(latent, velocity, sigma):
    """Flow-matching x0 estimate; a model prediction, not decoded ground truth."""
    return latent.float()-float(sigma)*velocity.float()


def compare_targets(flow, targets):
    """Compare both targets at the same positions, in addition to their native masks.

    Positive preference means closer to forward than reverse. This is an AMF
    measurement, not a heading/optical-flow quality metric.
    """
    common = targets['forward'][1] & targets['reverse'][1]
    if not common.any():
        raise ValueError('No common valid AMF positions for target selectivity')
    own, shared = {}, {}
    for name, (reference, mask) in targets.items():
        if not mask.any():
            raise ValueError(f'Empty {name} reference mask')
        error = (flow.float() - reference.float()).square().mean(-1)
        own[name] = float(error[mask].mean())
        shared[name] = float(error[common].mean())
    return dict(native_mask_mse=own, common_mask_mse=shared,
                forward_preference=shared['reverse']-shared['forward'],
                common_positions=int(common.sum()))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', required=True)
    parser.add_argument('--output_path', type=Path, required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--branch_step', type=int, default=9, choices=range(10))
    parser.add_argument('--updates', type=int, default=5, choices=range(1, 6))
    return parser


@torch.no_grad()
def read_loss_flow(guidance, latent, timestep):
    blocks = guidance.config.guidance_blocks
    guidance._set_kv_mode(blocks, inject=False, copy=True)
    try:
        guidance._forward_transformer(latent, guidance.guidance_embeds[1:2], timestep.reshape(1))
        return guidance._amf(guidance.transformer.blocks[10].attn1.processor).detach()
    finally:
        guidance._set_kv_mode(blocks, inject=False, copy=False)
        guidance._clear_kv(blocks)


@torch.no_grad()
def predict_clean(guidance, latent, step):
    """Same CFG velocity as production denoising, without advancing the scheduler."""
    t = guidance.timesteps[step]
    with torch.autocast(device_type='cuda', dtype=guidance.dtype):
        with guidance.probe.phase('estimate_cond',step,t):
            cond = guidance.transformer(hidden_states=latent.to(guidance.dtype), timestep=t.reshape(1),
                encoder_hidden_states=guidance.guidance_embeds[1:2], return_dict=False)[0].float()
        uncond = guidance.transformer(hidden_states=latent.to(guidance.dtype), timestep=t.reshape(1),
            encoder_hidden_states=guidance.guidance_embeds[:1], return_dict=False)[0].float()
    velocity = uncond+guidance.guidance_scale*(cond-uncond)
    return flow_clean_estimate(latent,velocity,guidance.scheduler.sigmas[step])


@torch.no_grad()
def decode(guidance, latent, destination):
    from diffusers.utils import export_to_video
    vae = guidance.vae
    mean = torch.tensor(vae.config.latents_mean, device=latent.device, dtype=vae.dtype).view(1,-1,1,1,1)
    std = torch.tensor(vae.config.latents_std, device=latent.device, dtype=vae.dtype).view(1,-1,1,1,1)
    frames = vae.decode(latent.to(vae.dtype)*std+mean, return_dict=False)[0]
    video = guidance.pipe.video_processor.postprocess_video(frames, output_type='pil')[0]
    export_to_video(video, str(destination), fps=16)


def main():
    args = build_parser().parse_args()
    root = args.output_path.resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh response directory. Never mix partial branch suites.')
    root.mkdir(parents=True, exist_ok=True)
    from motion_guidance_wan import MODEL_IDS, WAN_NEGATIVE_PROMPT, WanGuidance, save_video, clean_memory
    from guidance_utils.motion_probe import MotionProbe
    from guidance_utils.wan_reference_diagnostics import reference_images
    from benchmark.wan_response_pilot import environment_snapshot, audit_response

    config = OmegaConf.merge(OmegaConf.load('configs/guidance_config_wan.yaml'), dict(
        model_key=MODEL_IDS['14b'], enable_model_cpu_offload=True,
        video_path=str(Path(args.video_path).resolve()), output_path=str(root/'initialization'),
        target_prompt=args.prompt, source_prompt='', negative_prompt=WAN_NEGATIVE_PROMPT, seed=1,
        opt_mode='latent', guidance_mode='latent', loss_type='flow', save_format='mp4', save_embeds=False,
        inject_embeds=False, verbose=False, scheduler='flowmatch', flow_shift=3., num_frames=21,
        height=480, width=832, num_inference_steps=50, guidance_scale=5.,
        guidance_blocks=[10], injection_blocks=[], guidance_timestep_range=[50,40],
        lr=[.002,.001], lr_decay_steps=10, optimization_steps=args.updates,
        motion_temp=2., flow_max_disp=100., flow_min_conf=None, threshloss=True,
        argmax_motion_flow=True, flow_loss='mse', flow_region_masks=None,
        probe=False, probe_rope=False, reference_only=False))
    environment = environment_snapshot()
    (root/'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    OmegaConf.save(config, root/'suite_config.yaml')
    g = WanGuidance(config)
    block = 'block_10_attn1_processor'
    targets = {'forward': (g.motion_attn_features[block].clone(), g.motion_attn_masks[block].clone())}
    frames = reference_images(args.video_path, 21, (832,480))
    save_video(frames, str(root/'reference_forward.mp4'))
    save_video(frames[::-1], str(root/'reference_reverse.mp4'))
    # Reverse decoded frames BEFORE the causal VAE. Flipping latent time would
    # not encode the reversed video faithfully.
    with tempfile.TemporaryDirectory(prefix='wan_response_reverse_') as folder:
        for index, frame in enumerate(frames[::-1]):
            Image.fromarray(frame).save(Path(folder)/f'{index:05d}.png')
        config.video_path = folder
        with torch.no_grad():
            g.motion_latent = g.load_latent()
        g.motion_attn_features = g.load_attn_features()
    config.video_path = str(Path(args.video_path).resolve())
    targets['reverse'] = (g.motion_attn_features[block].clone(), g.motion_attn_masks[block].clone())
    if torch.equal(targets['forward'][0], targets['reverse'][0]):
        raise ValueError('Forward and reversed reference AMFs are identical; inspect readout first')
    for name, (flow, mask) in targets.items():
        np.savez_compressed(root/f'target_{name}.npz', flow=flow.float().cpu().numpy(), mask=mask.cpu().numpy())

    # Actual text-conditioned sampling prefix, NOT a forward-noised reference.
    latent = g.init_latents.clone()
    prefix = []
    for i in range(args.branch_step):
        latent = g.denoise_step(latent, i, g.guidance_embeds)
        prefix.append(dict(step=i, sigma=float(g.scheduler.sigmas[i]), latent_sha256=tensor_hash(latent)))
    shared_latent, shared_scheduler = fork_state(latent, g.scheduler)
    predicted_before = predict_clean(g,shared_latent,args.branch_step)
    decode(g,predicted_before,root/'estimated_clean_before.mp4')
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
    generator_rng = g.generator.get_state()
    rope_hash = tensor_hash(g.transformer.init_rope)
    metadata = dict(schema_version=1, model=config.model_key, grid=[g.latent_num_frames,g.patches_height,g.patches_width],
        model_revision=getattr(g.transformer.config,'_commit_hash',None), prompt=args.prompt, seed=1,
        branch_step=args.branch_step, updates=args.updates, learning_rate=float(g.lr_by_step[args.branch_step]),
        sigma=float(g.scheduler.sigmas[args.branch_step]), timestep=float(g.timesteps[args.branch_step]),
        timesteps=g.timesteps.cpu().tolist(), sigmas=g.scheduler.sigmas.cpu().tolist(), prefix=prefix,
        scheduler_class=type(g.scheduler).__name__, scheduler_config=dict(g.scheduler.config),
        shared_latent_sha256=tensor_hash(shared_latent), rope_sha256=rope_hash,
        frame_sha256=[hashlib.sha256(np.asarray(frame).tobytes()).hexdigest() for frame in frames],
        note='14B T2V impulse response, not the original full guidance window; no I2V, KV injection or RoPE optimization.')
    (root/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for branch in BRANCHES:
        print(f'[*] Response branch {branch}: shared step {args.branch_step}, {0 if branch=="off" else args.updates} updates', flush=True)
        folder = root/branch; folder.mkdir()
        latent, g.scheduler = fork_state(shared_latent, shared_scheduler)
        torch.set_rng_state(cpu_rng); torch.cuda.set_rng_state_all(cuda_rng); g.generator.set_state(generator_rng)
        g.output_path = str(folder); config.output_path = str(folder)
        config.probe = True; config.probe_blocks = [10]
        config.probe_steps = sorted({args.branch_step, args.branch_step+1, 29, 49})
        config.response_reference_order = branch
        config.response_single_step = args.branch_step
        for module in g.transformer.blocks:
            module.attn1.processor.motion_probe = None
        g.probe = MotionProbe(g, 'wan')
        reference_name = 'reverse' if branch=='reverse' else 'forward'
        g.motion_attn_features = {block:targets[reference_name][0]}
        g.motion_attn_masks = {block:targets[reference_name][1]}
        g.probe.actual_reference(block, *targets[reference_name])
        start_hash = tensor_hash(latent)
        evaluations = []

        def evaluate(label, value, timestep, step):
            with g.probe.phase(label, step, timestep):
                flow = read_loss_flow(g, value, timestep)
            np.savez_compressed(folder/f'{label}.npz', flow=flow.float().cpu().numpy())
            record = dict(stage=label, step=step, timestep=float(timestep), **compare_targets(flow, targets))
            evaluations.append(record)
            return record

        t = g.timesteps[args.branch_step]
        evaluate('before_update', latent, t, args.branch_step)
        before = latent.clone()
        if branch != 'off':
            with torch.enable_grad():
                latent, _ = g.guidance_step(latent, args.branch_step, t, mode='latent', loss_type='flow')
        evaluate('after_update', latent, t, args.branch_step)
        update_rms = float((latent-before).square().mean().sqrt())
        g._set_kv_mode([10], inject=False, copy=False); g._clear_kv([10])
        # Inspect the immediate model-predicted effect separately from retention.
        # At this high sigma an x0 estimate can be very poor; label it explicitly.
        estimate = predicted_before if branch=='off' else predict_clean(g,latent,args.branch_step)
        decode(g,estimate,folder/'estimated_clean_after.mp4')
        del estimate
        sampled = []
        for i in range(args.branch_step, len(g.timesteps)):
            pre_sample = latent
            latent = g.denoise_step(latent, i, g.guidance_embeds)
            g.probe.sampling(i, g.timesteps[i], before if i==args.branch_step else pre_sample, pre_sample, latent, g.scheduler)
            sampled.append(i)
        evaluate('final_latent', latent, g.motion_timestep[0], -1)
        if tensor_hash(g.transformer.init_rope) != rope_hash or g.transformer.trainable_rope is not None:
            raise ValueError('RoPE changed during latent-only experiment')
        if any(p.grad is not None for p in g.transformer.parameters()):
            raise ValueError('Unexpected gradient on frozen transformer weights')
        decode(g, latent, folder/'results.mp4')
        report = dict(branch=branch, start_latent_sha256=start_hash, update_rms=update_rms,
                      sampled_indices=sampled, evaluations=evaluations, rope_unchanged=True,
                      final_latent_sha256=tensor_hash(latent))
        (folder/'response.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        clean_memory()
    result = audit_response(root)
    (root/'complete.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('All three branches validated. Review decoded movement separately from AMF scores:', root, flush=True)


if __name__ == '__main__':
    main()
