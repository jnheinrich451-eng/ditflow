"""Weight-free regression checks for guidance scheduling and direction metrics."""

from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from guidance_utils.wan_guidance_schedule import learning_rates, window_indices
from probe_report import retained_metrics


class ScheduleTests(unittest.TestCase):
    def test_windows_and_lr_prefix(self):
        self.assertEqual(window_indices(50, [50, 40]), list(range(10)))
        later = window_indices(50, [30, 10])
        self.assertEqual(later, list(range(20, 40)))
        rates = learning_rates(later, [.002, .001])
        self.assertEqual(rates[20], .002)
        self.assertEqual(rates[39], .001)
        baseline = learning_rates(range(10), [.002, .001], 10)
        extended = learning_rates(range(30), [.002, .001], 10)
        self.assertEqual(baseline, {i: extended[i] for i in baseline})
        self.assertTrue(all(extended[i] == .001 for i in range(10, 30)))
        self.assertEqual(window_indices(50, [0, 0]), [])
        for bad in ([20, 30], [50, -1], [100, 60]):
            with self.assertRaises(ValueError):
                window_indices(50, bad)
        with self.assertRaises(ValueError):
            learning_rates([0], [.002, .001], 0)

    def test_actual_run_loop_independent_injection_and_guidance(self):
        from motion_guidance_wan import WanGuidance

        class FinishedSampling(Exception):
            pass

        for guidance_window, injection_window in [([50, 40], [50, 20]), ([50, 20], [50, 40]),
                                                   ([30, 10], [50, 20]), ([50, 20], [0, 0])]:
            with self.subTest(guidance=guidance_window, injection=injection_window):
                g = WanGuidance.__new__(WanGuidance)
                torch.nn.Module.__init__(g)
                g.config = OmegaConf.create(dict(injection_blocks=[0], guidance_blocks=[1], inject_embeds=False,
                                                guidance_mode='latent', loss_type='flow'))
                g.guidance_steps = window_indices(50, guidance_window)
                g.injection_steps = window_indices(50, injection_window)
                g.init_latents = torch.zeros(1)
                g.motion_latent = torch.ones(1)
                g.timesteps = torch.linspace(1000, 1, 50)
                g.scheduler = SimpleNamespace()
                g.guidance_embeds = torch.zeros(2, 1)
                g.device = torch.device('cpu')
                g.probe = SimpleNamespace(enabled=False, sampling=lambda *a: None)
                state, guided, injected, caches = {}, [], [], []
                g._set_kv_mode = lambda blocks, inject, copy: state.update(injected=inject)
                g._add_noise = lambda *a: g.motion_latent
                g._forward_transformer = lambda *a, **kw: caches.append(1)

                def guide(x, i, t, **kw):
                    guided.append(i)
                    return x, None

                def denoise(x, i, *args, **kwargs):
                    if state['injected']:
                        injected.append(i)
                    if i == 49:
                        raise FinishedSampling  # Stop before VAE decoding/export.
                    return x

                g.guidance_step, g.denoise_step = guide, denoise
                with patch('motion_guidance_wan.clean_memory'), patch('motion_guidance_wan.tqdm', side_effect=lambda x, **kw: x):
                    with self.assertRaises(FinishedSampling):
                        g.run()
                self.assertEqual(guided, g.guidance_steps)
                self.assertEqual(injected, g.injection_steps)
                self.assertEqual(len(caches), len(g.injection_steps))

    def test_actual_optimizer_accepts_later_sampling_index(self):
        from motion_guidance_wan import WanGuidance
        g = WanGuidance.__new__(WanGuidance)
        torch.nn.Module.__init__(g)
        g.config = OmegaConf.create(dict(guidance_blocks=[1], optimization_steps=2, verbose=False, save_embeds=False))
        g.lr_by_step = {20: .002}
        g._set_kv_mode = lambda *a, **kw: None
        g.compute_motion_flow_loss = lambda x, t: x.square().mean()
        g.probe = SimpleNamespace(enabled=False, phase=lambda *a, **kw: nullcontext(), optimization=lambda *a: None)
        output, _ = g.guidance_step(torch.ones(1, 2), 20, torch.tensor(500), 'latent', 'flow')
        self.assertTrue(torch.isfinite(output).all())
        self.assertLess(output.square().mean().item(), 1)

    def test_empty_mask_fails_before_nan_loss(self):
        from motion_guidance_wan import WanGuidance
        g = WanGuidance.__new__(WanGuidance)
        torch.nn.Module.__init__(g)
        g.config = OmegaConf.create(dict(guidance_blocks=[0]))
        g.device = torch.device('cpu')
        g.guidance_embeds = torch.zeros(2, 1)
        g._forward_transformer = lambda *a, **kw: None
        proc = SimpleNamespace(block_name='test')
        g.transformer = SimpleNamespace(blocks=[SimpleNamespace(attn1=SimpleNamespace(processor=proc))])
        g._amf = lambda p: torch.zeros(4, 3, 2)
        g.motion_attn_features = {'test': torch.zeros(4, 3, 2)}
        g.motion_attn_masks = {'test': torch.zeros(4, 3, dtype=torch.bool)}
        with self.assertRaisesRegex(ValueError, 'empty reference AMF mask'):
            g.compute_motion_flow_loss(torch.ones(1), torch.tensor(500))

    def test_direction_metrics_detect_opposite_motion_and_empty_masks(self):
        ref = np.array([[[1., 0.], [0., 0.]]])
        mask = np.array([[True, True]])
        opposite = retained_metrics(-ref, ref, mask)
        self.assertEqual(opposite['direction_cosine'], -1)
        self.assertGreater(opposite['retained_mse'], opposite['zero_prediction_mse'])
        aligned = retained_metrics(ref, ref, mask)
        self.assertEqual(aligned['direction_cosine'], 1)
        self.assertEqual(aligned['retained_mse'], 0)
        self.assertIsNone(retained_metrics(ref, ref, ~mask)['retained_mse'])
        self.assertIsNone(retained_metrics(np.zeros_like(ref), ref, mask)['direction_cosine'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
