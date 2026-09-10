"""Load Wan once and probe RoPE on real, static, and known-motion references.

No denoising/generation. Run in the same Colab environment as the Wan baseline.
"""

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image

from guidance_utils.motion_probe import MotionProbe
from guidance_utils.wan_rope_diagnostics import make_rope_report, write_control
from guidance_utils.wan_reference_diagnostics import reference_images, make_reference_report
from motion_guidance_wan import MODEL_IDS, WAN_NEGATIVE_PROMPT, WanGuidance, clean_memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-v', '--video_path', required=True)
    parser.add_argument('--output_path', required=True, help='Use a fresh folder for each suite')
    parser.add_argument('--model', choices=list(MODEL_IDS), default='1.3b')
    parser.add_argument('--blocks', nargs='+', type=int, default=[0, 10, 15, 20])
    parser.add_argument('--num_frames', type=int, default=21)
    parser.add_argument('--guidance_blocks', nargs='+', type=int, default=None,
                        help='Blocks whose actual reference AMF target is saved (separate from diagnostic --blocks)')
    parser.add_argument('--flow_max_disp', type=float, default=None)
    parser.add_argument('--mask_dir', type=Path, default=None,
                        help='Optional aligned reference masks for offline image-motion comparison only')
    parser.add_argument('--controls', nargs='+', choices=['static', 'pan_right', 'pan_left', 'patch_right'],
                        default=['static', 'pan_right', 'pan_left', 'patch_right'])
    args = parser.parse_args()
    root = Path(args.output_path)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a fresh output_path to keep evidence from separate suites distinct')
    root.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load('configs/guidance_config_wan.yaml')
    config = OmegaConf.merge(config, dict(
        model_key=MODEL_IDS[args.model], video_path=args.video_path, output_path=str(root / 'real'),
        target_prompt='', negative_prompt=WAN_NEGATIVE_PROMPT, seed=1, opt_mode='latent', guidance_mode='latent',
        loss_type='flow', save_format='mp4', save_embeds=False, inject_embeds=False, verbose=False,
        scheduler='flowmatch', num_frames=args.num_frames, probe=True, probe_rope=True,
        probe_blocks=args.blocks, probe_steps=[], reference_only=True,
        guidance_blocks=args.guidance_blocks if args.guidance_blocks is not None else list(config['guidance_blocks_1_3b' if args.model == '1.3b' else 'guidance_blocks_14b']),
        rope_control=dict(kind='real', width=config.width, height=config.height)))
    if args.flow_max_disp is not None:
        config.flow_max_disp = args.flow_max_disp
    OmegaConf.save(config, root / 'suite_config.yaml')
    guidance = WanGuidance(config)
    runs = [str(root / 'real')]
    first = Image.fromarray(reference_images(args.video_path, 1, (config.width, config.height))[0])
    for kind in dict.fromkeys(args.controls):
        inputs = root / 'inputs' / kind
        info = write_control(first, kind, inputs, count=args.num_frames)
        config.video_path = str(inputs)
        config.output_path = str(root / kind)
        config.rope_control = info
        guidance.output_path = config.output_path
        guidance.probe = MotionProbe(guidance, 'wan')
        print(f'[*] Reference control: {kind}')
        guidance.motion_latent = guidance.load_latent()
        guidance.motion_attn_features = guidance.load_attn_features()
        runs.append(config.output_path)
        clean_memory()
    print(make_rope_report(runs, root / 'report'))
    if args.mask_dir is not None:
        print(make_reference_report(root / 'real', args.video_path, args.mask_dir, root / 'reference_report'))


if __name__ == '__main__':
    main()
