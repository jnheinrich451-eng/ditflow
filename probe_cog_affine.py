"""Probe known rotation/scale/translation through CogVideoX heads at recorded noise levels.

Sibling of probe_wan_affine.py for the CogVideoX baseline. Loads the model once.
No optimization, KV injection, generation or positional-embedding edits.

It answers one question: does the AMF readout DiTFlow optimizes on CogVideoX
(block 20 of the 5B model by default) degrade with noise the way Wan's block-10
readout does? CogVideoX noises as sqrt(a)*x0 + sqrt(1-a)*eps, so every state also
records the flow-matching sigma with the same signal-to-noise ratio; by default
the two Wan guidance-window sigmas are matched to their nearest sampling steps.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

from guidance_utils.wan_affine_diagnostics import (
    CONTROLS, AffineObserver, affine_controls, affine_truth, texture_support, make_affine_report,
)
from guidance_utils.wan_reference_diagnostics import reference_images

BLOCK_COUNTS = {'5b': 42, '2b': 30}
RESOLUTION = (720, 480)
# Wan flowmatch shift 3, sampling indices 9 and 29: the end of the default guidance
# window and the later point the Wan affine suite observed.
WAN_GUIDANCE_SIGMAS = (0.9304704666137695, 0.6757655143737793)
# Nominal VAE anchors 0,4,...,20 for six latent frames, identical to the Wan suite.
TRUTH_FRAMES = 21
# motion_guidance.py's default; unused by a blank-prompt forward but recorded for fidelity.
NEGATIVE_PROMPT = 'bad quality, distortions, unrealistic, distorted image, watermark, signature'


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--video_path', required=True, help='Only the first frame supplies control texture')
    parser.add_argument('--output_path', required=True, type=Path)
    parser.add_argument('--model', default='5b', choices=sorted(BLOCK_COUNTS))
    parser.add_argument('--controls', nargs='+', choices=CONTROLS, default=list(CONTROLS))
    parser.add_argument('--blocks', nargs='+', type=int, default=None,
                        help='Observed blocks; default is the configured guidance block for the model (20 for 5b, 15 for 2b)')
    parser.add_argument('--noise_steps', nargs='+', type=int, default=[0, 9, 29],
                        help='Indices in the fixed 50-step DDIM/DPM schedule; clean t=0 is always added')
    parser.add_argument('--match_sigmas', nargs='*', type=float, default=list(WAN_GUIDANCE_SIGMAS),
                        help='Also observe the sampling indices whose equivalent flow sigma is nearest to these Wan values; '
                             'pass the flag with no values to disable')
    parser.add_argument('--video_length', type=int, default=24,
                        help='Frames encoded per control; 24 is the CogVideoX baseline working point (six latent frames)')
    parser.add_argument('--latent_sampling', default='mode', choices=('mode', 'sample'),
                        help="'mode' is deterministic; 'sample' reproduces the entry point's stochastic VAE draw")
    parser.add_argument('--noise_seed', type=int, default=17)
    parser.add_argument('--prompt', default='', help='Same text for ALL states; blank isolates noise effects and matches reference extraction')
    return parser


def equivalent_sigma(alpha_cumprod):
    """Flow-matching sigma with the same signal-to-noise ratio as sqrt(a)*x0 + sqrt(1-a)*eps."""
    a = float(alpha_cumprod)
    if not 0. <= a <= 1.:
        raise ValueError(f'alpha_cumprod outside [0, 1]: {a}')
    signal, noise = a ** .5, (1. - a) ** .5
    return noise / (signal + noise)


def schedule_table(scheduler):
    """Every sampling index with its DDPM coefficients and equivalent flow sigma; nothing is inferred."""
    alphas = scheduler.alphas_cumprod.detach().cpu().double()
    rows = []
    for index, timestep in enumerate(scheduler.timesteps.detach().cpu().tolist()):
        a = float(alphas[int(timestep)])
        rows.append(dict(sampling_index=index, timestep=int(timestep), alpha_cumprod=a,
                         signal_coefficient=a ** .5, noise_coefficient=(1. - a) ** .5, sigma=equivalent_sigma(a)))
    return rows


def nearest_indices(table, sigmas):
    return [min(table, key=lambda row: abs(row['sigma'] - float(s)))['sampling_index'] for s in sigmas]


def noise_states(table, indices):
    """Clean first, then requested sampling indices in schedule order, then a pure-noise negative control if the schedule lacks one."""
    if any(i < 0 or i >= len(table) for i in indices):
        raise ValueError(f'Sampling indices must be within 0..{len(table)-1}')
    states = [dict(noise_label='clean', sampling_index=-1, timestep=0, alpha_cumprod=1.,
                   signal_coefficient=1., noise_coefficient=0., sigma=0.)]
    states += [dict(noise_label=f'step_{index:02d}', **table[index]) for index in sorted(set(indices))]
    if not any(s['signal_coefficient'] == 0. for s in states):
        # CogVideoX rescales to zero terminal SNR, so its first sampling step is exactly
        # pure noise; other schedules need the negative control added explicitly.
        states.append(dict(noise_label='pure_noise', sampling_index=-1, timestep=table[0]['timestep'], alpha_cumprod=0.,
                           signal_coefficient=0., noise_coefficient=1., sigma=1.))
    return states


def noisy_input(latent, noise, state):
    """DDPM forward noising from the recorded coefficients; checked against scheduler.add_noise at startup."""
    if latent.shape != noise.shape:
        raise ValueError('Mismatched noise geometry')
    return state['signal_coefficient'] * latent.float() + state['noise_coefficient'] * noise.float()


class CogAffineObserver(AffineObserver):
    """CogVideoX processors pass (B, heads, text+video, D); drop the text prefix and use the Wan (B, S, heads, D) layout."""

    def attention(self, block_name, query, key, injected=False, text_prefix=0):
        if self.active is None:
            return
        super().attention(block_name, query[:, :, text_prefix:].transpose(1, 2),
                          key[:, :, text_prefix:].transpose(1, 2), injected)


def main():
    parser = build_parser(); args = parser.parse_args()
    count = BLOCK_COUNTS[args.model]
    if args.blocks is not None and any(b < 0 or b >= count for b in args.blocks):
        parser.error(f'CogVideoX-{args.model} blocks must be 0..{count-1}')
    if any(i < 0 or i >= 50 for i in args.noise_steps):
        parser.error('Sampling indices must be 0..49')
    if args.video_length < TRUTH_FRAMES:
        parser.error(f'video_length must be at least {TRUTH_FRAMES} so the nominal anchors 0..20 exist')
    root = args.output_path.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh output directory')
    root.mkdir(parents=True, exist_ok=True)
    # Import the model stack only after validating arguments. Original generation code is unchanged.
    from motion_guidance import Guidance, clean_memory, read_video_frames, save_video

    class ReadoutCog(Guidance):
        def load_attn_features(self):
            # Initialization still uses the original VAE/text/model setup. This
            # suite does not need an all-pair optimization target or reference forward.
            return {}

        @torch.no_grad()
        def load_latent(self):
            # Same decode/preprocess/scale/permute path as the entry point, with a
            # deterministic posterior mode by default so controls differ only in content.
            data_path = self.config.video_path
            if data_path.endswith('.mp4'):
                video = [Image.fromarray(f).convert('RGB').resize(self.resolution) for f in read_video_frames(data_path)]
            else:
                images = sorted([*Path(data_path).glob('*.png'), *Path(data_path).glob('*.jpg')],
                                key=lambda x: int(x.stem.split('f')[-1]))
                video = [Image.open(p).resize(self.resolution).convert('RGB') for p in images]
            video = video[:self.config.video_length]
            if len(video) != self.config.video_length:
                raise ValueError(f'{data_path}: expected {self.config.video_length} frames, got {len(video)}')
            save_video([np.array(img) for img in video], str(Path(self.config.output_path) / 'original.mp4'))
            tensor = self.pipe.video_processor.preprocess_video(video).to(self.dtype).to(self.device)
            distribution = self.vae.encode(tensor)[0]
            latents = distribution.mode() if args.latent_sampling == 'mode' else distribution.sample()
            return (self.vae.config.scaling_factor * latents).permute(0, 2, 1, 3, 4)

    first = reference_images(args.video_path, 1, RESOLUTION)[0]
    Image.fromarray(first).save(root/'base_frame.png')
    config = OmegaConf.load('configs/guidance_config.yaml')
    blocks = sorted(set(args.blocks if args.blocks is not None else list(config[f'guidance_blocks_{args.model}'])))
    config = OmegaConf.merge(config, dict(
        model_key=f'THUDM/CogVideoX-{args.model}', video_path=str(Path(args.video_path).resolve()), output_path=str(root/'initialization'),
        target_prompt=args.prompt, source_prompt='', negative_prompt=NEGATIVE_PROMPT, seed=1, video_length=args.video_length,
        opt_mode='latent', guidance_mode='latent', loss_type='flow', save_format='mp4', save_embeds=False, inject_embeds=False,
        verbose=False, injection_blocks=[], probe=False,
        # Only the early exit uses this: blocks after the last observed one pass through. No guidance runs.
        guidance_blocks=[max(blocks)], width=RESOLUTION[0], height=RESOLUTION[1]))
    OmegaConf.save(config, root/'suite_config.yaml')
    guidance = ReadoutCog(config)
    modules = guidance.transformer.transformer_blocks
    if len(modules) != count:
        raise RuntimeError(f'Expected {count} blocks for CogVideoX-{args.model}, found {len(modules)}')
    grid = [guidance.latent_num_frames, guidance.patches_height, guidance.patches_width]
    if grid[0] != (TRUTH_FRAMES-1)//4+1:
        raise RuntimeError(f'Expected six latent frames for the nominal anchors, got {grid[0]}')
    table = schedule_table(guidance.scheduler)
    matched = dict(zip(map(str, args.match_sigmas), nearest_indices(table, args.match_sigmas)))
    states = noise_states(table, sorted(set(args.noise_steps) | set(matched.values())))
    # Dedicated CPU generator; reuse exactly the same independent-frame noise for all controls.
    noise = torch.randn(guidance.motion_latent.shape, generator=torch.Generator().manual_seed(args.noise_seed), dtype=torch.float32).to(guidance.device)
    reference_latent = guidance.motion_latent.float()
    for state in states:
        if state['sampling_index'] >= 0:
            expected = guidance.scheduler.add_noise(reference_latent, noise, torch.tensor([state['timestep']], device=guidance.device))
            torch.testing.assert_close(noisy_input(reference_latent, noise, state), expected, atol=1e-4, rtol=1e-4)
    text = guidance.guidance_embeds[1:2] if args.prompt else guidance.source_embeds
    # The entry point's guidance-time early exit: later blocks and the final norms pass
    # through, so the (unused) output is not a denoising prediction.
    guidance.change_mode(train=True)
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    sources = ['probe_cog_affine.py', 'guidance_utils/wan_affine_diagnostics.py', 'guidance_utils/motion_probe.py',
               'guidance_utils/custom_modules.py', 'guidance_utils/custom_transformer.py', 'motion_guidance.py']
    metadata = dict(schema_version=1, backbone='CogVideoX', model=config.model_key, grid=grid, packages={k:version(k) for k in (
        'torch', 'diffusers', 'transformers', 'huggingface-hub', 'numpy', 'Pillow')}, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(), model_dtype=str(guidance.dtype), diagnostic_precision='fp32 detached QK',
        git_commit=commit, source_sha256={p:hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources},
        input_base_sha256=hashlib.sha256(first.tobytes()).hexdigest(), control_resolution=list(RESOLUTION), blocks=blocks,
        early_exit_after_block=max(blocks), controls=list(dict.fromkeys(args.controls)), noise_states=states,
        matched_wan_sigmas=matched, schedule_equivalence=table, scheduler=type(guidance.scheduler).__name__,
        scheduler_config=dict(guidance.scheduler.config), noise_seed=args.noise_seed,
        noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(), conditioning=args.prompt,
        text_seq_length=int(config.text_seq_length), video_length=args.video_length, truth_frames=TRUTH_FRAMES,
        latent_sampling=args.latent_sampling, temperature=float(config.motion_temp),
        model_revision=getattr(guidance.transformer.config, '_commit_hash', None),
        note='Controlled forward-noised videos, not denoising trajectories. No generation/optimization. '
             'sigma is the flow-matching sigma with equal SNR, for comparison with the Wan suite only. '
             f'All {args.video_length} frames are encoded; analytic truth uses nominal anchors 0,4,...,20 like the Wan suite. '
             'CogVideoX VAE batching (8 frames per batch) makes those anchors approximate, as they are for Wan.')
    (root/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    rows = []; observer = CogAffineObserver(root, grid, float(config.motion_temp), rows)
    for b in blocks:
        modules[b].attn1.processor.motion_probe = observer
    print(f"{len(metadata['controls'])*len(states)} observed forwards, blocks={blocks}, early exit after block {max(blocks)}; no generations", flush=True)
    print('Noise states: ' + ', '.join(f"{s['noise_label']} (t={s['timestep']}, sigma_eq={s['sigma']:.4f})" for s in states), flush=True)
    for kind in metadata['controls']:
        frames, info = affine_controls(first, kind, count=args.video_length)
        folder = root/kind; folder.mkdir(exist_ok=True)
        (folder/'control.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        truths = {}
        arrays = {}
        truth_info = dict(info, num_frames=TRUTH_FRAMES)
        for offset in (-2, 0, 2):
            truth, geometric, anchors = affine_truth(truth_info, grid, anchor_offset=offset)
            textured = texture_support(frames[:TRUTH_FRAMES], grid, anchor_offset=offset)
            truths[offset] = truth, geometric, textured
            arrays.update({f'flow_{offset}':truth, f'geometry_{offset}':geometric, f'texture_{offset}':textured, f'anchors_{offset}':anchors})
        np.savez_compressed(folder/'truth.npz', **arrays)
        # Temporary lossless inputs keep the archive compact. base_frame.png and
        # control.json reproduce every input; original.mp4 is a visual preview.
        with tempfile.TemporaryDirectory(prefix='cog_affine_') as temporary:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(Path(temporary)/f'{i:05d}.png')
            config.video_path = temporary; config.output_path = str(folder)
            latent = guidance.load_latent()
        for state in states:
            print(f"[*] {kind} | {state['noise_label']} | t={state['timestep']} | sigma_eq={state['sigma']:.6f}", flush=True)
            observer.active = (dict(control=kind, **state), truths); observer.seen.clear()
            timestep = torch.tensor([state['timestep']], device=guidance.device)
            sample = guidance.scheduler.scale_model_input(noisy_input(latent, noise, state), state['timestep'])
            try:
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=guidance.dtype):
                    guidance.transformer(hidden_states=sample.to(guidance.dtype), encoder_hidden_states=text,
                                         timestep=timestep, return_dict=False)
                expected = {modules[b].attn1.processor.block_name for b in blocks}
                if observer.seen != expected:
                    raise RuntimeError(f'Missing/extra observed blocks: {observer.seen} vs {expected}')
            finally:
                observer.active = None
            (root/'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
        clean_memory()
    print(make_affine_report(root), flush=True)
    (root/'complete.json').write_text(json.dumps(dict(forward_passes=len(states)*len(metadata['controls']), rows=len(rows))), encoding='utf-8')


if __name__ == '__main__':
    main()
