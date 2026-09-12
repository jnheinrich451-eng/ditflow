"""CPU checks of the CogVideoX affine readout driver: noise schedule, observer layout, truth, parser."""
import sys
from pathlib import Path

# Tests import repo modules by their root names; make `python tests/<file>.py` work from the root checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from guidance_utils.wan_affine_diagnostics import AffineObserver, affine_controls, affine_truth, make_affine_report, texture_support
from probe_cog_affine import (
    TRUTH_FRAMES, WAN_GUIDANCE_SIGMAS, CogAffineObserver, build_parser, equivalent_sigma, nearest_indices, noise_states,
    noisy_input, schedule_table,
)


def fake_scheduler(alphas, timesteps):
    return SimpleNamespace(alphas_cumprod=torch.tensor(alphas, dtype=torch.float64), timesteps=torch.tensor(timesteps))


class CogAffineTests(unittest.TestCase):
    def test_equivalent_sigma_endpoints_and_monotonic(self):
        self.assertEqual(equivalent_sigma(1.), 0.); self.assertEqual(equivalent_sigma(0.), 1.)
        self.assertAlmostEqual(equivalent_sigma(.5), .5)
        alphas = np.linspace(1., 0., 11)
        sigmas = [equivalent_sigma(a) for a in alphas]
        self.assertTrue(all(b > a for a, b in zip(sigmas, sigmas[1:])))
        with self.assertRaises(ValueError):
            equivalent_sigma(1.5)

    def test_schedule_table_reads_coefficients_by_timestep(self):
        scheduler = fake_scheduler([.9, .5, .1, 0.], [3, 2, 1])
        table = schedule_table(scheduler)
        self.assertEqual([r['timestep'] for r in table], [3, 2, 1])
        self.assertEqual(table[0]['alpha_cumprod'], 0.); self.assertEqual(table[0]['sigma'], 1.)
        self.assertAlmostEqual(table[1]['signal_coefficient'], .1 ** .5)
        self.assertAlmostEqual(table[2]['noise_coefficient'], .5 ** .5)
        self.assertEqual(nearest_indices(table, [1., equivalent_sigma(.5), 0.]), [0, 2, 2])

    def test_noise_states_add_pure_noise_only_when_missing(self):
        with_zero = schedule_table(fake_scheduler([.9, .5, 0.], [2, 1]))
        states = noise_states(with_zero, [1, 0, 1])
        self.assertEqual([s['noise_label'] for s in states], ['clean', 'step_00', 'step_01'])
        self.assertEqual(states[0]['timestep'], 0); self.assertEqual(states[0]['sigma'], 0.)
        self.assertEqual(states[1]['sigma'], 1.)
        without_zero = schedule_table(fake_scheduler([.9, .5, .1], [2, 1]))
        states = noise_states(without_zero, [1])
        self.assertEqual([s['noise_label'] for s in states], ['clean', 'step_01', 'pure_noise'])
        self.assertEqual(states[-1]['timestep'], 2); self.assertEqual(states[-1]['signal_coefficient'], 0.)
        with self.assertRaises(ValueError):
            noise_states(without_zero, [5])

    def test_noisy_input_matches_ddpm_formula_and_endpoints(self):
        latent = torch.randn(1, 6, 16, 4, 6); noise = torch.randn_like(latent)
        table = schedule_table(fake_scheduler([.36, 0.], [1, 0]))
        clean, step0, step1 = noise_states(table, [0, 1])
        torch.testing.assert_close(noisy_input(latent, noise, clean), latent)
        torch.testing.assert_close(noisy_input(latent, noise, step0), noise)
        torch.testing.assert_close(noisy_input(latent, noise, step1), .6 * latent + .8 * noise)
        with self.assertRaises(ValueError):
            noisy_input(latent, noise[:, :3], step1)

    def test_real_cogvideox_scheduler_terminal_state_is_pure_noise(self):
        # Mirrors the CogVideoX-2b/5b configs: zero terminal SNR, trailing spacing.
        from diffusers import CogVideoXDDIMScheduler
        scheduler = CogVideoXDDIMScheduler(num_train_timesteps=1000, beta_start=.00085, beta_end=.012, beta_schedule='scaled_linear',
                                           snr_shift_scale=3., rescale_betas_zero_snr=True, timestep_spacing='trailing')
        scheduler.set_timesteps(50)
        table = schedule_table(scheduler)
        self.assertEqual(len(table), 50); self.assertEqual(table[0]['timestep'], 999); self.assertEqual(table[0]['sigma'], 1.)
        self.assertTrue(all(b['sigma'] < a['sigma'] for a, b in zip(table, table[1:])))
        latent = torch.randn(1, 6, 16, 4, 6); noise = torch.randn_like(latent)
        for index in (0, 9, 29):
            state = dict(noise_label='x', **table[index])
            expected = scheduler.add_noise(latent, noise, torch.tensor([state['timestep']]))
            torch.testing.assert_close(noisy_input(latent, noise, state), expected, atol=1e-5, rtol=1e-5)
        matched = nearest_indices(table, WAN_GUIDANCE_SIGMAS)
        self.assertEqual(len(matched), 2); self.assertLess(matched[0], matched[1])
        for sigma, index in zip(WAN_GUIDANCE_SIGMAS, matched):
            self.assertLess(abs(table[index]['sigma'] - sigma), .02)

    def test_observer_strips_text_prefix_and_matches_wan_layout(self):
        grid = [2, 2, 3]; heads, dim, prefix = 2, 8, 5
        torch.manual_seed(0)
        q_video = torch.randn(1, 12, heads, dim); k_video = torch.randn_like(q_video)
        q_cog = torch.cat([torch.randn(1, prefix, heads, dim), q_video], 1).transpose(1, 2)
        k_cog = torch.cat([torch.randn(1, prefix, heads, dim), k_video], 1).transpose(1, 2)
        truth = np.ones((1, 6, 2), np.float32); mask = np.ones((1, 6), bool)
        active = (dict(control='pan_right', noise_label='clean', sigma=0., timestep=0, sampling_index=-1), {0: (truth, mask, mask)})
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            rows_wan, rows_cog = [], []
            wan = AffineObserver(a, grid, 2., rows_wan); wan.active = active; wan.attention('block_20', q_video, k_video)
            cog = CogAffineObserver(b, grid, 2., rows_cog)
            cog.attention('block_20', q_cog, k_cog, text_prefix=prefix)  # inactive: must be a no-op
            self.assertEqual(rows_cog, [])
            cog.active = active; cog.attention('block_20', q_cog, k_cog, injected=False, text_prefix=prefix)
            self.assertEqual(len(rows_cog), len(rows_wan)); self.assertEqual(cog.seen, {'block_20'})
            for x, y in zip(rows_wan, rows_cog):
                self.assertEqual(x, y)
            with self.assertRaises(ValueError):
                cog.attention('block_20', q_cog, k_cog, injected=True, text_prefix=prefix)

    def test_truth_for_24_frame_controls_uses_nominal_anchors(self):
        first = np.random.default_rng(3).integers(0, 256, (96, 160, 3), dtype=np.uint8)
        frames, info = affine_controls(first, 'pan_right', count=24)
        self.assertEqual(len(frames), 24)
        truth, valid, anchors = affine_truth(dict(info, num_frames=TRUTH_FRAMES), [6, 6, 10], margin_patches=1)
        self.assertEqual(anchors, [0, 4, 8, 12, 16, 20])
        np.testing.assert_allclose(truth, np.broadcast_to([1., 0.], truth.shape), atol=1e-6)
        _, _, shifted = affine_truth(dict(info, num_frames=TRUTH_FRAMES), [6, 6, 10], anchor_offset=2, margin_patches=1)
        self.assertEqual(shifted, [2, 6, 10, 14, 18, 20])
        self.assertEqual(texture_support(frames[:TRUTH_FRAMES], [6, 6, 10]).shape, (5, 60))
        with self.assertRaises(ValueError):
            affine_truth(info, [6, 6, 10])

    def test_parser_defaults_and_validation(self):
        args = build_parser().parse_args(['-v', 'input', '--output_path', 'out'])
        self.assertEqual(args.model, '5b'); self.assertIsNone(args.blocks); self.assertEqual(args.video_length, 24)
        self.assertEqual(args.match_sigmas, list(WAN_GUIDANCE_SIGMAS)); self.assertEqual(args.latent_sampling, 'mode')
        self.assertEqual(build_parser().parse_args(['-v', 'i', '--output_path', 'o', '--match_sigmas']).match_sigmas, [])

    def test_report_title_uses_backbone(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); rows = []
            observer = AffineObserver(root, [2, 2, 3], 2., rows)
            truth = np.ones((1, 6, 2), np.float32); mask = np.ones((1, 6), bool)
            observer.active = (dict(control='expand', noise_label='clean', sigma=0., timestep=0, sampling_index=-1), {0: (truth, mask, mask)})
            observer.attention('block_20', torch.randn(1, 12, 2, 8), torch.randn(1, 12, 2, 8))
            (root/'metrics.json').write_text(json.dumps(rows)); (root/'metadata.json').write_text(json.dumps(dict(backbone='CogVideoX')))
            self.assertIn('<title>CogVideoX affine motion readout</title>', make_affine_report(root).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
