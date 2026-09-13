"""Causal evidence checks: observational hooks, paired states and sampler use."""
import copy
import base64
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

from benchmark import wan_control_pilot as pilot
from benchmark import wan_head_pilot as head
from benchmark import wan_noised_reference_pilot as noised
from benchmark import wan_pair_pilot as pairs
from guidance_utils.wan_control_trace import ControlRecorder, temporal_attention_mass, projected_head_stats
from probe_wan_control import ControlGuidanceMixin
from probe_wan_noised_reference import NoisedReferenceMixin
from probe_wan_pairs import ForwardAdjacentMixin


class ControlTests(unittest.TestCase):
    def test_native_mass_matches_full_video_softmax_under_autocast(self):
        torch.manual_seed(1)
        q, k = torch.randn(1, 12, 2, 4), torch.randn(1, 12, 2, 4)
        expected = (q[0, :, 1] @ k[0, :, 1].T / 2).softmax(-1).reshape(12, 3, 4).sum(-1)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            actual = temporal_attention_mass(q, k, frames=3, head=1, chunk=3)
        np.testing.assert_allclose(actual, expected.numpy(), atol=1e-7)
        np.testing.assert_allclose(actual.sum(-1), 1, atol=1e-6)
        # A framewise logit offset preserves per-frame softmax, but changes native mass.
        q = torch.ones(1, 8, 1, 1); k = torch.zeros_like(q)
        a = temporal_attention_mass(q, k, 2, 0)
        k[:, 4:] -= 12
        b = temporal_attention_mass(q, k, 2, 0)
        self.assertAlmostEqual(float(a[0, 1]), .5)
        self.assertLess(float(b[0, 1]), 1e-5)

    def test_observed_value_projection_matches_head_component(self):
        torch.manual_seed(2)
        layer = torch.nn.Linear(6, 5)
        x = torch.randn(1, 9, 6); output = layer(x)
        expected = torch.nn.functional.linear(x[..., 2:4], layer.weight[:, 2:4]).double().square().mean().sqrt()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            result = projected_head_stats(x, output, layer, head=1, head_dim=2, chunk=2)
        self.assertAlmostEqual(result['projected_head_rms'], float(expected.detach()), places=7)

    def test_reverse_decoded_frames_before_encoding(self):
        from PIL import Image
        frames = [np.full((3, 5, 3), index, dtype=np.uint8) for index in range(21)]
        with tempfile.TemporaryDirectory() as tmp, patch(
                'guidance_utils.wan_reference_diagnostics.reference_images', return_value=frames) as read:
            folder = Path(tmp)/'reverse'
            manifest = pilot.write_reversed_frames('source', folder)
            read.assert_called_once_with('source', 21, (832, 480))
            self.assertEqual(len(manifest), 21)
            for i in range(21):
                np.testing.assert_array_equal(np.asarray(Image.open(folder/f'{i:05d}.png')), frames[20-i])

    def test_config_preserves_frozen_guidance_and_start_checks(self):
        plan = dict(generation_video='forward', reverse_video='reverse', prompt='truck', selection=dict(block=30, head=30))
        forward = pilot.fixed_config(plan, 'forward', 'out')
        reverse = pilot.fixed_config(plan, 'reverse', 'out')
        self.assertEqual({k for k in forward if forward[k] != reverse[k]}, {'control_arm', 'video_path'})
        self.assertEqual(forward['guidance_blocks'], [30]); self.assertEqual(forward['optimization_steps'], 5)
        self.assertEqual(forward['lr'], [.001, .001]); self.assertEqual(forward['flow_pair_mode'], 'forward_adjacent')
        self.assertEqual(pilot.fixed_config(plan, 'off', 'out')['guidance_blocks'], [])
        start = dict(latent_sha256='same', conditioning_sha256='same', source_conditioning_sha256='same',
                     rope_sha256='same', timesteps=[1], sigmas=[1, 0], model_revision='same', reference_latent_sha256='a')
        reversed_start = dict(start, reference_latent_sha256='b')
        pilot.require_common_start(start, reversed_start)
        with self.assertRaisesRegex(ValueError, 'conditioning'):
            pilot.require_common_start(start, dict(reversed_start, conditioning_sha256='different'))

    def test_display_embeds_the_correct_distinct_video_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); expected = []
            for arm in pilot.ARMS:
                directory = root/arm; directory.mkdir()
                head.write_json(root/f'{arm}_done.json', dict(directory=arm))
                for name in ('original.mp4', 'final.mp4', 'estimated_clean_before.mp4', 'estimated_clean_after.mp4'):
                    (directory/name).write_bytes((arm+'/'+name).encode())
            old = root/'previous/candidate'; old.mkdir(parents=True)
            head.write_json(root/'previous/candidate_done.json', dict(directory='candidate'))
            (old/'final.mp4').write_bytes(b'old-all-pair-candidate')
            pilot.make_display(root, dict(response=[], selectivity=[], native=[]))
            embedded = re.findall(r'src="data:video/mp4;base64,([A-Za-z0-9+/=]+)"',
                                 (root/'control_comparison.html').read_text(encoding='utf-8'))
            expected = [root/'forward/original.mp4', root/'reverse/original.mp4',
                        *[root/arm/'final.mp4' for arm in pilot.ARMS], old/'final.mp4',
                        root/'off/estimated_clean_before.mp4', root/'forward/estimated_clean_after.mp4',
                        root/'reverse/estimated_clean_after.mp4']
            self.assertEqual([base64.b64decode(payload) for payload in embedded], [p.read_bytes() for p in expected])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by production Wan')
    def test_actual_bf16_sampler_hooks_are_observational_and_auditable(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.motion_probe import MotionProbe
        from probe_wan_head_visual import capture_estimate
        from probe_report import load_trace

        class Plain(ForwardAdjacentMixin, NoisedReferenceMixin, WanGuidance): pass
        class Observed(ControlGuidanceMixin, Plain): pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); finals = {}; starts = {}
            for arm in ('plain_forward', 'off', 'forward', 'reverse'):
                observed = arm != 'plain_forward'; guided = arm != 'off'
                torch.manual_seed(17)
                folder = root/arm; folder.mkdir()
                cls = Observed if observed else Plain
                g = cls.__new__(cls); torch.nn.Module.__init__(g)
                g.config = OmegaConf.create(dict(probe=True, probe_blocks=[1], probe_steps=[9], probe_rope=False,
                    guidance_blocks=[1] if guided else [], injection_blocks=[], loss_type='flow', flow_head=30,
                    motion_temp=2., softmax_fp32=True, argmax_motion_flow=True, threshloss=True, flow_max_disp=100.,
                    optimization_steps=5, verbose=False, save_embeds=False, flow_loss='mse',
                    reference_noise_step=9, reference_noise_seed=29, flow_pair_mode=pairs.PAIR_MODE))
                g.device, g.dtype = torch.device('cuda'), torch.bfloat16
                g.transformer = ControlledWanTransformer(patch_size=(1, 2, 2), num_attention_heads=40,
                    attention_head_dim=8, in_channels=4, out_channels=4, text_dim=16, freq_dim=16,
                    ffn_dim=32, num_layers=3, cross_attn_norm=True, qk_norm='rms_norm_across_heads',
                    eps=1e-6, rope_max_seq_len=64).to(device=g.device, dtype=g.dtype).eval().requires_grad_(False)
                g.transformer.enable_gradient_checkpointing()
                g.latent_height, g.latent_width, g.patch_size = 4, 6, 2
                g.patches_height, g.patches_width, g.latent_num_frames = 2, 3, 3
                g.checkpoint_amf, g._guidance_scale = True, 5
                g.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.)
                g.scheduler.set_timesteps(50, device=g.device)
                g.timesteps = g.scheduler.timesteps; g.lr_by_step = {9: .001}; g.output_path = str(folder)
                g.register_guidance([1]); g.register_attention_processor([0, 1, 2]); g.probe = MotionProbe(g, 'wan')
                # Distinct synthetic reference states; actual pre-VAE reversal is tested separately.
                g.motion_latent = torch.randn(1, 4, 3, 4, 6, device=g.device, dtype=g.dtype)*10
                if arm == 'reverse': g.motion_latent = g.motion_latent.flip(2)
                g.transformer.init_rope = g.transformer.default_rope(g.motion_latent).to(g.device)
                g.source_embeds = torch.randn(1, 5, 16, device=g.device, dtype=g.dtype)
                g.guidance_embeds = torch.randn(2, 5, 16, device=g.device, dtype=g.dtype)
                g.motion_timestep = torch.tensor([0], device=g.device)
                g.motion_attn_features = g.load_attn_features()
                x = torch.randn_like(g.motion_latent).float(); starts[arm] = x.clone()
                if observed: g.control = ControlRecorder(g, block=1, head=30)
                try:
                    with torch.no_grad(), patch('probe_wan_head_visual.decode'):
                        for i, t in enumerate(g.timesteps):
                            before = x.clone()
                            if i == 9 and guided:
                                if not observed: capture_estimate(g, x, i, 'before')
                                with torch.enable_grad(): x, _ = g.guidance_step(x, i, t, 'latent', 'flow')
                                if not observed: capture_estimate(g, x, i, 'after')
                            optimized = x
                            x = g.denoise_step(x, i, g.guidance_embeds)
                            g.probe.sampling(i, t, before, optimized, x, g.scheduler)
                finally:
                    if observed: g.control.close()
                finals[arm] = x.clone()
                trace, _, events = load_trace(folder)
                noised.audit_trace(events, OmegaConf.to_container(g.config), 'candidate' if guided else 'off')
                if guided: pairs.audit_selected_loss(trace, events, 3, 2, 3)
                if observed:
                    for path in (folder/'control').glob('*.npz'):
                        pilot.audit_capture(path, shape=(1, 4, 3, 4, 6), expected_block=1)
                    pilot.require_same_capture(folder/'control'/('after_09.npz' if guided else 'before_09.npz'),
                                               folder/'control/denoise_09.npz')
                    head.write_json(root/f'{arm}_done.json', dict(directory=arm))
                self.assertTrue(all(p.grad is None for p in g.transformer.parameters()))
            torch.testing.assert_close(finals['plain_forward'], finals['forward'], rtol=0, atol=0)
            self.assertFalse(torch.equal(finals['off'], finals['forward']))
            for arm in ('forward', 'reverse'):
                torch.testing.assert_close(starts['off'], starts[arm], rtol=0, atol=0)
                pilot.require_same_capture(root/arm/'control/before_09.npz', root/'off/control/before_09.npz')
            response = pilot.response_rows(root)
            self.assertEqual(len(response), 12)
            self.assertTrue(all(r['paired_euler_residual_rms'] < 1e-6 for r in response))
            self.assertEqual(len(pilot.target_selectivity(root)), 4)
            pilot.require_directional_targets(root/'reverse', root/'forward')
            self.assertEqual(len(pilot.native_rows(root)), 46)
            # A discarded update, wrong velocity or malformed checkpoint must fail.
            path = root/'forward/control/denoise_09.npz'; saved = pilot.arrays(path)
            bad = {**saved, 'next_latent': saved['next_latent']+1}
            np.savez_compressed(path, **bad)
            with self.assertRaisesRegex(ValueError, 'Sampler did not consume'):
                pilot.audit_capture(path, shape=(1, 4, 3, 4, 6), expected_block=1)
            np.savez_compressed(path, **saved)
            wrong = root/'wrong.npz'; np.savez_compressed(wrong, **{**saved, 'latent': saved['latent']+1})
            with self.assertRaisesRegex(ValueError, 'before-guidance latent'):
                pilot.require_same_capture(path, wrong)
            # Failure in diagnostic computation must restore processor flags/caches.
            from probe_wan_response import predict_clean
            failure = root/'capture_failure'; failure.mkdir(); g.output_path = str(failure)
            processor = g.transformer.blocks[1].attn1.processor
            processor.copy_kv = False; processor.query = None; processor.key = None; processor.value = None
            recorder = ControlRecorder(g, block=1, head=30)
            with self.assertRaisesRegex(RuntimeError, 'diagnostic failure'), torch.no_grad(), patch(
                    'guidance_utils.wan_control_trace.temporal_attention_mass', side_effect=RuntimeError('diagnostic failure')):
                with recorder.capture('before', 9, x): predict_clean(g, x, 9)
            self.assertIsNone(recorder.active); self.assertFalse(processor.copy_kv)
            self.assertIsNone(processor.query); self.assertIsNone(processor.key); self.assertIsNone(processor.value)
            recorder.close(); self.assertEqual(recorder.handles, [])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
