"""Decisive Wan test: decoded-motion metric, pre-registered verdict, frozen arms and the real sampler path."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from benchmark import wan_decisive_pilot as pilot
from guidance_utils import decoded_motion as dm


def texture(seed=0, size=(480, 832)):
    import cv2
    rng = np.random.default_rng(seed)
    coarse = cv2.resize(rng.random((size[0] // 16, size[1] // 16, 3)).astype(np.float32), size[::-1],
                        interpolation=cv2.INTER_CUBIC)
    fine = cv2.resize(rng.random((size[0] // 4, size[1] // 4, 3)).astype(np.float32), size[::-1],
                      interpolation=cv2.INTER_CUBIC)
    return (np.clip(.7 * coarse + .3 * fine, 0, 1) * 255).astype(np.uint8)


def panning(base, pixels_per_frame, count=21):
    return np.stack([np.roll(base, pixels_per_frame * i, axis=1) for i in range(count)])


def write_video(path, frames):
    import imageio.v2 as imageio
    imageio.mimwrite(str(path), list(frames), fps=16, macro_block_size=1, quality=9)
    return path


class MetricTests(unittest.TestCase):
    def test_flow_field_recovers_known_translation_in_patch_units(self):
        base = texture()
        for pixels, sign in ((4, 1), (-4, -1)):
            flow = dm.flow_field(panning(base, pixels))
            self.assertEqual(flow.shape, (5, 30, 52, 2))
            interior = flow[:, 3:-3, 4:-4]
            # 4 px per frame at 832 wide, 4 frames per pair = 16 px = one 16-px patch.
            self.assertAlmostEqual(float(interior[..., 0].mean()), sign * 1., delta=.15)
            self.assertLess(abs(float(interior[..., 1].mean())), .05)
        # Identical frames: Farneback leaves only numerical residue, far below the 1-patch signal.
        self.assertLess(float(np.abs(dm.flow_field(panning(base, 0))).max()), .02)

    def test_selectivity_score_gain_and_undefined_cases(self):
        rf = np.zeros((5, 30, 52, 2)); rf[..., 0] = 1
        rr = -rf
        perfect = dm.selectivity(rf, rr, rf, rr)
        self.assertAlmostEqual(perfect['score'], 1.); self.assertAlmostEqual(perfect['gain'], 1.)
        self.assertAlmostEqual(dm.selectivity(rr, rf, rf, rr)['score'], -1.)
        partial = dm.selectivity(.3 * rf, .3 * rr, rf, rr)
        self.assertAlmostEqual(partial['score'], 1.); self.assertAlmostEqual(partial['gain'], .3)
        self.assertTrue(math.isnan(dm.selectivity(rf, rf, rf, rr)['score']))
        rng = np.random.default_rng(0)
        noise = dm.selectivity(rng.normal(size=rf.shape), rng.normal(size=rf.shape), rf, rr)
        self.assertLess(abs(noise['score']), .05)
        self.assertAlmostEqual(dm.change_from(rf, np.zeros_like(rf), rf)['alignment'], 1.)

    def test_verdict_rule_is_the_pre_registered_one(self):
        def result(score, magnitude=2.):
            return dict(score=score, gain=score, separation=1., target_magnitude=magnitude)
        cases = [((.8, .1, .9), 'GO'), ((.35, .1, .9), 'GO'), ((.29, 0., .9), 'STOP'), ((.4, .3, .9), 'STOP'),
                 ((.4, -.3, .9), 'STOP'), ((float('nan'), .0, .9), 'STOP'), ((.9, .0, .4), 'INVALID'),
                 ((.9, .0, float('nan')), 'INVALID')]
        for (amf, null, positive), expected in cases:
            self.assertEqual(dm.verdict(result(amf), result(null), result(positive))['decision'], expected,
                             (amf, null, positive))
        still = dm.verdict(result(.9), result(0.), result(.9, magnitude=.05))
        self.assertEqual(still['decision'], 'INVALID')
        go = dm.verdict(result(.8), result(float('nan')), result(.9))
        self.assertEqual(go['decision'], 'GO'); self.assertEqual(go['null_size'], 0.)

    def test_scores_from_real_encoded_videos(self):
        base = texture(1)
        forward_ref = panning(base, 4)
        reverse_ref = forward_ref[::-1]
        static = panning(base, 0)
        rng = np.random.default_rng(2)
        jitter = lambda: np.clip(static.astype(int) + rng.integers(-2, 3, static.shape), 0, 255).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            def videos(**arms):
                paths = dict(reference_forward=write_video(tmp / 'rf.mp4', forward_ref),
                             reference_reverse=write_video(tmp / 'rr.mp4', reverse_ref))
                for name, frames in arms.items():
                    paths[name] = write_video(tmp / f'{name}.mp4', frames)
                return paths
            common = dict(off=static, random_a=jitter(), random_b=jitter())
            stop = pilot.score_videos(videos(**common, forward=static, reverse=jitter(),
                                             sdedit_forward=forward_ref, sdedit_reverse=reverse_ref))
            self.assertEqual(stop['verdict']['decision'], 'STOP', stop['verdict'])
            self.assertGreater(stop['selectivity']['positive_control']['score'], .9)
            go = pilot.score_videos(videos(**common, forward=forward_ref, reverse=reverse_ref,
                                           sdedit_forward=forward_ref, sdedit_reverse=reverse_ref))
            self.assertEqual(go['verdict']['decision'], 'GO', go['verdict'])
            self.assertGreater(go['change_from_off']['forward']['alignment'], .9)
            invalid = pilot.score_videos(videos(**common, forward=forward_ref, reverse=reverse_ref,
                                                sdedit_forward=static, sdedit_reverse=jitter()))
            self.assertEqual(invalid['verdict']['decision'], 'INVALID', invalid['verdict'])


class ArmTests(unittest.TestCase):
    def test_random_perturbation_matches_dose_and_is_isolated(self):
        x = torch.randn(1, 16, 6, 60, 104)
        state = torch.get_rng_state().clone()
        a = pilot.random_perturbation(x, .0042, 101, 3)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertAlmostEqual(pilot.rms(a - x), .0042, delta=.0042 * 1e-4)
        self.assertTrue(torch.equal(a, pilot.random_perturbation(x, .0042, 101, 3)))
        self.assertFalse(torch.equal(a, pilot.random_perturbation(x, .0042, 202, 3)))
        self.assertFalse(torch.equal(a, pilot.random_perturbation(x, .0042, 101, 4)))

    def test_sdedit_start_and_mid_schedule_sampling(self):
        from diffusers import FlowMatchEulerDiscreteScheduler
        reference, noise = torch.randn(2, 3), torch.randn(2, 3)
        torch.testing.assert_close(pilot.sdedit_start_latent(reference, noise, .8), .2 * reference + .8 * noise)
        scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); scheduler.set_timesteps(50)
        start = pilot.SDEDIT_START
        self.assertAlmostEqual(float(scheduler.sigmas[start]), .8139, places=3)
        scheduler.set_begin_index(start)
        x, v = torch.zeros(3), torch.ones(3)
        for offset, t in enumerate(scheduler.timesteps[start:start + 3]):
            before = float(scheduler.sigmas[start + offset]); after = float(scheduler.sigmas[start + offset + 1])
            y = scheduler.step(v, t, x, return_dict=False)[0]
            torch.testing.assert_close(y, x + (after - before) * v)
            x = y

    def test_arms_differ_only_where_the_protocol_says(self):
        plan = dict(video='fwd', reverse_video='rev', prompt='A horse walks', seed=1)
        dose = {str(i): .004 for i in range(10)}
        config = {arm: pilot.fixed_config(plan, arm, 'out', dose if pilot.KIND[arm] == 'random' else None)
                  for arm in pilot.ARMS}
        diff = lambda a, b: {k for k in config[a] if config[a][k] != config[b][k]}
        self.assertEqual(diff('forward', 'reverse'), {'video_path', 'decisive_arm'})
        self.assertEqual(diff('random_a', 'random_b'), {'decisive_arm', 'random_seed'})
        self.assertEqual(diff('forward', 'random_a'), {'decisive_arm', 'decisive_kind', 'random_seed', 'random_dose'})
        self.assertEqual(diff('sdedit_forward', 'sdedit_reverse'), {'video_path', 'decisive_arm'})
        self.assertEqual(diff('off', 'sdedit_forward'), {'decisive_arm', 'decisive_kind', 'sdedit_start'})
        amf = config['forward']
        self.assertEqual((amf['guidance_blocks'], amf['guidance_timestep_range'], amf['lr'], amf['optimization_steps'],
                          amf['injection_blocks'], amf['scheduler'], amf['flow_head'], amf['model_key']),
                         ([20], [50, 40], [.002, .001], 5, [], 'flowmatch', None, pilot.MODEL))
        self.assertEqual(config['off']['guidance_blocks'], [])
        self.assertEqual(config['sdedit_forward']['guidance_blocks'], [])
        with self.assertRaises(ValueError):
            pilot.fixed_config(plan, 'random_a', 'out')
        with self.assertRaises(ValueError):
            pilot.fixed_config(plan, 'forward', 'out', dose)

    def test_matched_dose_reads_both_amf_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arm, scale in (('forward', 1.), ('reverse', 3.)):
                (root / arm).mkdir()
                pilot.write_json(root / f'{arm}_done.json', dict(directory=arm))
                pilot.write_json(root / arm / 'dose.json', dict(update_rms={str(i): scale * (i + 1) for i in range(10)}))
            dose = pilot.matched_dose(root)
            self.assertEqual(dose['0'], 2.); self.assertEqual(dose['9'], 20.)
            pilot.write_json(root / 'reverse' / 'dose.json', dict(update_rms={str(i): 1. for i in range(9)}))
            with self.assertRaises(ValueError):
                pilot.matched_dose(root)

    def test_run_arm_writes_marker_only_after_validation_and_resumes(self):
        create = ('import pathlib, sys; p = pathlib.Path(sys.argv[-1]); p.mkdir(); (p / "ok").write_text("1")')
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, 'environment_snapshot', return_value={'e': 1}):
            root = Path(tmp)
            pilot.write_json(root / 'environment.json', {'e': 1})
            calls = []
            def validator(directory):
                calls.append(directory)
                if not (directory / 'ok').is_file():
                    raise ValueError('missing output')
                return dict(ok=True)
            first = pilot.run_arm(root, 'off', [sys.executable, '-c', create], validator)
            self.assertTrue((root / 'off_done.json').is_file())
            self.assertEqual(pilot.run_arm(root, 'off', [sys.executable, '-c', 'raise SystemExit(3)'], validator), first)
            self.assertEqual(len(calls), 2)
            with self.assertRaises(RuntimeError):
                pilot.run_arm(root, 'forward', [sys.executable, '-c', 'raise SystemExit(3)'], validator)
            self.assertFalse((root / 'forward_done.json').exists())
            self.assertEqual(len(list(root.glob('forward_*.log'))), 1)
            pilot.write_json(root / 'environment.json', {'e': 2})
            with self.assertRaisesRegex(RuntimeError, 'environment changed'):
                pilot.run_arm(root, 'reverse', [sys.executable, '-c', create], validator)

    @unittest.skipUnless((ROOT / 'probe_runs/wan_reference_inputs/manifest.csv').is_file(), 'reference-input bundle absent')
    def test_plan_from_the_real_input_bundle_and_tamper_checks(self):
        from PIL import Image
        from guidance_utils.wan_reference_diagnostics import reference_images
        with tempfile.TemporaryDirectory() as tmp, patch.object(pilot, 'environment_snapshot', return_value={'e': 1}):
            root = Path(tmp) / 'plan'
            plan = pilot.make_plan(ROOT / 'probe_runs/wan_reference_inputs', root)
            self.assertEqual((plan['clip_id'], plan['seed']), pilot.FIRST_CLIP)
            self.assertIn('horse', plan['prompt'])
            frames = reference_images(plan['video'], 21, (832, 480))
            reversed_first = np.asarray(Image.open(root / 'reference_reverse/00000.png'))
            np.testing.assert_array_equal(reversed_first, frames[-1])
            self.assertEqual(pilot.checked_plan(root)['stage'], pilot.STAGE)
            with self.assertRaises(ValueError):
                pilot.make_plan(ROOT / 'probe_runs/wan_reference_inputs', root)
            Image.fromarray(frames[0]).save(root / 'reference_reverse/00000.png')
            with self.assertRaisesRegex(ValueError, 'Reversed'):
                pilot.checked_plan(root)
            (root / 'plan.json').write_text((root / 'plan.json').read_text().replace('"seed": 1', '"seed": 2'))
            with self.assertRaisesRegex(ValueError, 'Plan changed'):
                pilot.checked_plan(root)


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by production Wan methods')
class RealSamplerTests(unittest.TestCase):
    """Tiny random BF16 Wan through the production run(), guidance_step and AMF loss."""

    def build(self, kind, tmp, dose=None):
        from PIL import Image
        from diffusers import FlowMatchEulerDiscreteScheduler
        from guidance_utils.motion_probe import MotionProbe
        from guidance_utils.wan_guidance_schedule import learning_rates
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from motion_guidance_wan import WanGuidance
        from probe_wan_decisive import DecisiveMixin

        class TinyWan(DecisiveMixin, WanGuidance):
            pass

        torch.manual_seed(17)
        g = TinyWan.__new__(TinyWan); torch.nn.Module.__init__(g)
        blocks = [1] if kind in ('amf', 'random') else []
        g.config = OmegaConf.create(dict(
            probe=False, guidance_blocks=blocks, injection_blocks=[], loss_type='flow', flow_head=None, motion_temp=2.,
            softmax_fp32=True, argmax_motion_flow=True, threshloss=True, flow_max_disp=None, flow_min_conf=None,
            optimization_steps=5, verbose=False, save_embeds=False, flow_loss='mse', flow_region_masks=None,
            guidance_mode='latent', inject_embeds=False, save_format='frames', reference_only=False,
            decisive_kind=kind, random_seed=101, random_dose=dose,
            sdedit_start=pilot.SDEDIT_START if kind == 'sdedit' else None))
        g.device, g.dtype = torch.device('cuda'), torch.bfloat16
        g.transformer = ControlledWanTransformer(
            patch_size=(1, 2, 2), num_attention_heads=4, attention_head_dim=8, in_channels=4, out_channels=4,
            text_dim=16, freq_dim=16, ffn_dim=32, num_layers=3, cross_attn_norm=True,
            qk_norm='rms_norm_across_heads', eps=1e-6, rope_max_seq_len=64,
        ).to(device=g.device, dtype=g.dtype).eval().requires_grad_(False)
        g.transformer.enable_gradient_checkpointing()
        g.latent_height, g.latent_width, g.patch_size = 4, 6, 2
        g.patches_height, g.patches_width, g.latent_num_frames = 2, 3, 2
        g.checkpoint_amf, g._guidance_scale = True, 5
        g.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); g.scheduler.set_timesteps(50, device=g.device)
        g.timesteps = g.scheduler.timesteps
        g.guidance_steps = list(pilot.GUIDANCE_STEPS); g.injection_steps = list(pilot.GUIDANCE_STEPS)
        g.lr_by_step = learning_rates(g.guidance_steps, pilot.LEARNING_RATE, None)
        g.output_path = tmp
        g.register_guidance(blocks); g.register_attention_processor([0, 1, 2])
        g.probe = MotionProbe(g, 'wan')
        generator = torch.Generator().manual_seed(5)
        g.motion_latent = torch.randn(1, 4, 2, 4, 6, generator=generator).to(g.device, g.dtype)
        g.init_latents = torch.randn(1, 4, 2, 4, 6, generator=generator).to(g.device)
        g.transformer.init_rope = g.transformer.default_rope(g.motion_latent).to(g.device)
        g.source_embeds = torch.randn(1, 5, 16, generator=generator).to(g.device, g.dtype)
        g.guidance_embeds = torch.randn(2, 5, 16, generator=generator).to(g.device, g.dtype)
        g.motion_timestep = torch.tensor([0], device=g.device)
        g.motion_attn_features = g.load_attn_features()
        decoded = {}
        def decode(z, return_dict=False):
            decoded['z'] = z.detach().clone()
            return (z,)
        g.vae = SimpleNamespace(config=SimpleNamespace(latents_mean=[0.] * 4, latents_std=[1.] * 4, z_dim=4),
                                dtype=torch.float32, decode=decode)
        g.pipe = SimpleNamespace(video_processor=SimpleNamespace(
            postprocess_video=lambda frames, output_type: [[Image.new('RGB', (6, 4))] * 2]))
        return g, decoded

    def test_amf_arm_guides_every_step_with_production_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, decoded = self.build('amf', tmp)
            rope = g.transformer.init_rope.clone()
            g.run(custom_name='final')
            records = g._records()
            self.assertEqual(sorted(records['update_rms'], key=int), [str(i) for i in range(10)])
            self.assertTrue(all(v > 0 for v in records['update_rms'].values()))
            self.assertEqual({k: len(v) for k, v in records['losses'].items()}, {str(i): 5 for i in range(10)})
            self.assertTrue(torch.isfinite(decoded['z']).all())
            torch.testing.assert_close(rope, g.transformer.init_rope, rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in g.transformer.parameters()))
            self.assertTrue((Path(tmp) / 'final').is_dir())

    def test_random_arm_matches_dose_deterministically_without_amf(self):
        dose = {str(i): .01 * (i + 1) for i in range(10)}
        outputs = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                g, decoded = self.build('random', tmp, dose)
                self.assertEqual(g.motion_attn_features, {})
                g.run(custom_name='final')
                for k, v in g._records()['update_rms'].items():
                    self.assertAlmostEqual(v, dose[k], delta=dose[k] * 1e-4)
                self.assertEqual(g._records()['losses'], {})
                outputs.append(decoded['z'])
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)

    def test_sdedit_arm_starts_from_the_noised_reference_mid_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, _ = self.build('sdedit', tmp)
            noise, original = g.init_latents.clone(), g.timesteps.clone()
            start = pilot.SDEDIT_START
            expected = pilot.sdedit_start_latent(g.motion_latent, noise, float(g.scheduler.sigmas[start]))
            calls, inner = [], g.denoise_step
            def spy(latents, i, prompt_embeds, rope=None):
                calls.append((latents.detach().clone(), float(g.timesteps[i])))
                result = inner(latents, i, prompt_embeds, rope=rope)
                calls[-1] += (g.scheduler.step_index,)
                return result
            g.denoise_step = spy
            g.run(custom_name='final')
            self.assertEqual(len(calls), 50 - start)
            torch.testing.assert_close(calls[0][0], expected)
            self.assertEqual([c[1] for c in calls], [float(t) for t in original[start:]])
            self.assertEqual((calls[0][2], calls[-1][2]), (start + 1, 50))
            self.assertEqual(g._records()['update_rms'], {})
            self.assertEqual(g._records()['sdedit_start'], start)
            with self.assertRaisesRegex(ValueError, 'must not guide'):
                g.guidance_step(noise, 0, original[0], 'latent', 'flow')


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
