"""Probe known rotation/scale/translation through Wan heads at recorded noise levels.

Loads Wan once per suite. No optimization, KV injection, generation or RoPE edits.
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


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', required=True, help='Only the first frame supplies control texture')
    parser.add_argument('--output_path', required=True, type=Path)
    parser.add_argument('--model', choices=('1.3b', '14b'), default='1.3b')
    parser.add_argument('--low_vram', action='store_true', help='Enable model CPU offload, including for 14B')
    parser.add_argument('--mean_only', action='store_true', help='Measure the baseline mean-logit AMF only; skip per-head sweeps')
    parser.add_argument('--controls', nargs='+', choices=CONTROLS, default=list(CONTROLS))
    parser.add_argument('--blocks', nargs='+', type=int, default=[10])
    parser.add_argument('--noise_steps', nargs='+', type=int, default=[0, 9, 29], help='Indices in the fixed 50-step flowmatch schedule; clean t=0 is always added')
    parser.add_argument('--noise_seed', type=int, default=17)
    parser.add_argument('--prompt', default='', help='Same text for ALL states; blank isolates noise effects and matches reference extraction')
    parser.add_argument('--readout_temperatures', nargs='+', type=float, default=None,
                        help='Opt-in mean-logit sharpening sweep on shared Q/K; requires --mean_only')
    return parser


def noise_states(scheduler, indices):
    """Record actual sigma and timestep; do not mistake video time for noise time."""
    return [dict(noise_label='clean', sampling_index=-1, sigma=0., timestep=0.)] + [
        dict(noise_label=f'step_{index:02d}', sampling_index=index,
             sigma=float(scheduler.sigmas[index]), timestep=float(scheduler.timesteps[index]))
        for index in dict.fromkeys(indices)]


def noisy_input(latent, noise, sigma):
    if not 0 <= sigma <= 1 or latent.shape != noise.shape:
        raise ValueError('Invalid sigma or mismatched noise geometry')
    return (1-sigma)*latent.float()+sigma*noise.float()


def main():
    parser = build_parser(); args = parser.parse_args()
    if args.readout_temperatures is not None:
        from guidance_utils.wan_amf_calibration import validate_temperatures
        try:
            args.readout_temperatures = validate_temperatures(args.readout_temperatures)
        except ValueError as error:
            parser.error(str(error))
        if not args.mean_only:
            parser.error('--readout_temperatures requires --mean_only')
    layers = 30 if args.model == '1.3b' else 40
    if any(b < 0 or b >= layers for b in args.blocks) or any(i < 0 or i >= 50 for i in args.noise_steps):
        parser.error(f'Wan {args.model} blocks must be 0..{layers-1} and sampling indices 0..49')
    root = args.output_path.resolve()
    if root.exists() and any(root.iterdir()):
        parser.error('Use a fresh output directory; the notebook resumes completed clip suites')
    root.mkdir(parents=True, exist_ok=True)
    # Import the model stack only after validating arguments. Original generation code is unchanged.
    from motion_guidance_wan import MODEL_IDS, WAN_NEGATIVE_PROMPT, WanGuidance, clean_memory

    class ReadoutWan(WanGuidance):
        def load_attn_features(self):
            # Initialization still uses the original VAE/text/model setup. This
            # suite does not need an all-pair optimization target or reference forward.
            return {}

    first = reference_images(args.video_path, 1, (832, 480))[0]
    Image.fromarray(first).save(root/'base_frame.png')
    config = OmegaConf.load('configs/guidance_config_wan.yaml')
    config = OmegaConf.merge(config, dict(
        model_key=MODEL_IDS[args.model], enable_model_cpu_offload=args.low_vram,
        video_path=str(Path(args.video_path).resolve()), output_path=str(root/'initialization'),
        target_prompt=args.prompt, source_prompt='', negative_prompt=WAN_NEGATIVE_PROMPT, seed=1,
        opt_mode='latent', guidance_mode='latent', loss_type='flow', save_format='mp4', save_embeds=False,
        inject_embeds=False, verbose=False, scheduler='flowmatch', flow_shift=3., num_frames=21,
        height=480, width=832, num_inference_steps=50, guidance_blocks=[], injection_blocks=[],
        flow_region_masks=None, probe=False, probe_rope=False, reference_only=True))
    OmegaConf.save(config, root/'suite_config.yaml')
    guidance = ReadoutWan(config)
    grid = [guidance.latent_num_frames, guidance.patches_height, guidance.patches_width]
    states = noise_states(guidance.scheduler, args.noise_steps)
    # Dedicated CPU generator; reuse exactly the same independent-frame noise for all controls.
    noise = torch.randn(guidance.motion_latent.shape, generator=torch.Generator().manual_seed(args.noise_seed), dtype=torch.float32).to(guidance.device)
    text = guidance.guidance_embeds[1:2] if args.prompt else guidance.source_embeds
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    sources = ['probe_wan_affine.py', 'guidance_utils/wan_affine_diagnostics.py', 'guidance_utils/motion_probe.py',
               'guidance_utils/wan_modules.py', 'guidance_utils/wan_transformer.py', 'motion_guidance_wan.py']
    if args.readout_temperatures is not None:
        sources.append('guidance_utils/wan_amf_calibration.py')
    metadata = dict(schema_version=1, model=config.model_key, grid=grid, packages={k:version(k) for k in (
        'torch', 'diffusers', 'transformers', 'huggingface-hub', 'numpy', 'Pillow')}, python=platform.python_version(),
        gpu=torch.cuda.get_device_name(), model_dtype=str(guidance.dtype), diagnostic_precision='fp32 detached QK',
        git_commit=commit, source_sha256={p:hashlib.sha256(Path(p).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in sources},
        input_base_sha256=hashlib.sha256(first.tobytes()).hexdigest(), blocks=list(dict.fromkeys(args.blocks)),
        controls=list(dict.fromkeys(args.controls)), noise_states=states, noise_seed=args.noise_seed,
        mean_only=args.mean_only, cpu_offload=args.low_vram,
        noise_sha256=hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest(), conditioning=args.prompt,
        model_revision=getattr(guidance.transformer.config, '_commit_hash', None), temperature=float(config.motion_temp),
        note='Controlled forward-noised videos, not denoising trajectories. No generation/optimization. Native RoPE unchanged.')
    rows = []
    if args.readout_temperatures is None:
        observer = AffineObserver(root, grid, float(config.motion_temp), rows, mean_only=args.mean_only)
    else:
        from guidance_utils.wan_amf_calibration import CalibrationObserver
        metadata['readout_temperatures'] = args.readout_temperatures
        observer = CalibrationObserver(root, grid, args.readout_temperatures, rows)
    (root/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    for b in metadata['blocks']:
        guidance.transformer.blocks[b].attn1.processor.motion_probe = observer
    print(f"{len(metadata['controls'])*len(states)} observed forwards, blocks={metadata['blocks']}; no generations", flush=True)
    for kind in metadata['controls']:
        frames, info = affine_controls(first, kind)
        folder = root/kind; folder.mkdir(exist_ok=True)
        (folder/'control.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        truths = {}
        arrays = {}
        for offset in (-2, 0, 2):
            truth, geometric, anchors = affine_truth(info, grid, anchor_offset=offset)
            textured = texture_support(frames, grid, anchor_offset=offset)
            truths[offset] = truth, geometric, textured
            arrays.update({f'flow_{offset}':truth, f'geometry_{offset}':geometric, f'texture_{offset}':textured, f'anchors_{offset}':anchors})
        np.savez_compressed(folder/'truth.npz', **arrays)
        # Temporary lossless inputs keep the report archive compact. base_frame.png
        # and control.json reproduce every input; original.mp4 is a visual preview.
        with tempfile.TemporaryDirectory(prefix='wan_affine_') as temporary:
            for i, frame in enumerate(frames):
                Image.fromarray(frame).save(Path(temporary)/f'{i:05d}.png')
            config.video_path = temporary; guidance.output_path = str(folder)
            latent = guidance.load_latent()
        for state in states:
            print(f"[*] {kind} | {state['noise_label']} | sigma={state['sigma']:.6f}", flush=True)
            observer.active = (dict(control=kind, **state), truths); observer.seen.clear()
            guidance.transformer.stop_after_block = max(metadata['blocks'])
            try:
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=guidance.dtype):
                    guidance.transformer(hidden_states=noisy_input(latent, noise, state['sigma']).to(guidance.dtype),
                        timestep=torch.tensor([state['timestep']], device=guidance.device),
                        encoder_hidden_states=text, return_dict=False)
                expected = {guidance.transformer.blocks[b].attn1.processor.block_name for b in metadata['blocks']}
                if observer.seen != expected:
                    raise RuntimeError(f'Missing/extra observed blocks: {observer.seen} vs {expected}')
            finally:
                guidance.transformer.stop_after_block = None; observer.active = None
            (root/'metrics.json').write_text(json.dumps(rows, indent=2, allow_nan=False), encoding='utf-8')
        clean_memory()
    if args.readout_temperatures is None:
        print(make_affine_report(root), flush=True)
    else:
        from guidance_utils.wan_amf_calibration import make_calibration_report
        print(make_calibration_report(root), flush=True)
    (root/'complete.json').write_text(json.dumps(dict(forward_passes=len(states)*len(metadata['controls']), rows=len(rows))), encoding='utf-8')


if __name__ == '__main__':
    main()
