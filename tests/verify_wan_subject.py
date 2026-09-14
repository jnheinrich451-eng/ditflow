"""Geometry, archive isolation, and actual tiny-BF16 guidance checks; no weights downloaded."""
import base64
import json
import tempfile
import unittest
import zipfile
import re
from pathlib import Path
from unittest.mock import patch, Mock
import sys

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import wan_subject_pilot as pilot
from benchmark import wan_control_pilot as control
from benchmark import wan_head_pilot as head
from benchmark import wan_pair_pilot as pairs
from guidance_utils.wan_subject_alignment import aligned_targets, alignment_loss
from guidance_utils.wan_control_trace import ControlRecorder
from probe_wan_control import ControlGuidanceMixin
from probe_wan_noised_reference import NoisedReferenceMixin
from probe_wan_pairs import ForwardAdjacentMixin
from probe_wan_subject import SubjectLossMixin


class GeometryTests(unittest.TestCase):
    def fixture(self, flow=(1., -.5)):
        field = np.broadcast_to(np.array(flow, np.float32), (9, 64, 2)).copy()
        valid = np.ones((9, 64), dtype=bool)
        regions = np.zeros((3, 8, 8), dtype=bool); regions[:, 1:3, 1:3] = True
        reference = [[1, 1, 3, 3]]*3
        generated = [[4, 2, 8, 6], [2, 4, 6, 8], [1, 1, 5, 5]]
        return field, valid, regions, reference, generated, (8, 8)

    def test_spatial_relocation_scales_displacement_and_removes_old_subject(self):
        result = aligned_targets(*self.fixture())
        np.testing.assert_array_equal(result['flow'][result['subject']], np.tile([2., -1.], (result['subject'].sum(), 1)))
        self.assertTrue(result['subject'][1].reshape(8, 8)[2:6, 4:8].all())
        self.assertFalse(result['background'][1].reshape(8, 8)[1:3, 1:3].any())
        self.assertFalse(np.any(result['subject'] & result['background']))
        self.assertEqual(set(np.flatnonzero(result['mask'].any(-1))), {1, 5})
        np.testing.assert_array_equal(result['flow'][result['background']], np.tile([1., -.5], (result['background'].sum(), 1)))

    def test_zero_flow_does_not_inherit_moving_generated_boxes(self):
        result = aligned_targets(*self.fixture((0., 0.)))
        self.assertTrue(np.array_equal(result['flow'], np.zeros_like(result['flow'])))

    def test_direction_sign_survives_alignment(self):
        forward = aligned_targets(*self.fixture((1., 2.)))
        reverse = aligned_targets(*self.fixture((-1., -2.)))
        np.testing.assert_array_equal(forward['flow'], -reverse['flow'])

    def test_invalid_reference_entries_are_not_reintroduced(self):
        args = list(self.fixture()); args[1][1, 9] = False
        result = aligned_targets(*args)
        self.assertFalse(np.any(result['source_index'][1] == 9))
        self.assertFalse(result['subject'][1].reshape(8, 8)[2:4, 4:6].any())

    def test_balancing_uses_separate_means_and_gradients_exclude_unused_support(self):
        value = torch.tensor([[2., 2.], [4., 4.], [4., 4.], [8., 8.]], requires_grad=True)
        target = torch.zeros_like(value)
        fg = torch.tensor([True, False, False, False]); bg = torch.tensor([False, True, True, False])
        uniform, _, _, w = alignment_loss(value, target, fg, bg, 'uniform')
        balanced, _, _, _ = alignment_loss(value, target, fg, bg, 'balanced')
        self.assertAlmostEqual(w, 1/3); self.assertEqual(float(uniform.detach()), 12.)
        self.assertEqual(float(balanced.detach()), 10.)
        balanced.backward(); self.assertEqual(float(value.grad[-1].abs().sum()), 0.)
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            alignment_loss(value, target, fg, fg, 'balanced')

    def test_archive_restores_outer_control_preserving_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'control.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('control/plan.json', json.dumps(dict(stage=control.STAGE)))
                for name in ('plan.sha256', 'environment.json', 'off_done.json', 'forward_done.json', 'reverse_done.json'):
                    bundle.writestr('control/'+name, '{}')
                bundle.writestr('control/previous/plan.json', json.dumps(dict(stage='noised_reference_step9_visual')))
                bundle.writestr('control/sampler_response.csv', 'important')
            destination = root/'existing'; destination.mkdir(); (destination/'keep.txt').write_text('keep')
            restored = pilot.restore_control_archive(archive, destination)
            self.assertNotEqual(restored, destination)
            self.assertEqual((destination/'keep.txt').read_text(), 'keep')
            self.assertEqual((restored/'sampler_response.csv').read_text(), 'important')
            self.assertEqual(pilot.restore_control_archive(archive, destination), restored)

    def test_archive_rejects_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); archive = root/'bad.zip'
            with zipfile.ZipFile(archive, 'w') as bundle:
                bundle.writestr('../outside.txt', 'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                pilot.restore_control_archive(archive, root/'result')
            self.assertFalse((root/'outside.txt').exists())

    def test_fresh_reference_readout_must_match_frozen_source(self):
        class Parent:
            def load_attn_features(self):
                self.motion_attn_masks = {'block_30_attn1_processor': torch.ones((4, 2), dtype=torch.bool)}
                return {'block_30_attn1_processor': torch.zeros((4, 2, 2))}
        class Subject(SubjectLossMixin, Parent): pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = root/'target.npz'
            fg = np.zeros((4, 2), bool); fg[1, 0] = True
            data = dict(source_flow=np.zeros((4, 2, 2), np.float32), source_mask=np.ones((4, 2), bool),
                        flow=np.ones((4, 2, 2), np.float32), subject=fg, background=~fg)
            np.savez_compressed(path, **data)
            g = Subject(); g.device = torch.device('cpu'); g.probe = Mock(path=root)
            g.config = OmegaConf.create(dict(guidance_blocks=[30], alignment_target=str(path), alignment_mode='uniform'))
            g.load_attn_features()
            self.assertTrue(torch.equal(g.alignment['flow'], torch.ones((4, 2, 2))))
            np.savez_compressed(path, **{**data, 'source_flow': data['source_flow']+1})
            with self.assertRaisesRegex(ValueError, 'Fresh reference'):
                g.load_attn_features()

    def test_portable_report_embeds_the_correct_nine_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); prior = root/'previous'; prior.mkdir()
            for base, arms in [(prior, control.ARMS), (root, pilot.ARMS)]:
                for arm in arms:
                    directory = base/arm; directory.mkdir()
                    head.write_json(base/f'{arm}_done.json', dict(directory=arm))
                    (directory/'original.mp4').write_bytes(('ref/'+arm).encode())
                    (directory/'final.mp4').write_bytes(('final/'+arm).encode())
            report = dict(arms={a: dict(losses=[dict(subject_mse=2, background_mse=3), dict(subject_mse=1, background_mse=2)]) for a in pilot.ARMS})
            pilot.make_display(root, report)
            encoded = re.findall(r'src="data:video/mp4;base64,([A-Za-z0-9+/=]+)"', (root/'subject_comparison.html').read_text())
            expected = [prior/'forward/original.mp4', prior/'reverse/original.mp4',
                        *[prior/a/'final.mp4' for a in control.ARMS], *[root/a/'final.mp4' for a in pilot.ARMS]]
            self.assertEqual([base64.b64decode(s) for s in encoded], [p.read_bytes() for p in expected])


class SamplingTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required by production Wan')
    def test_real_bf16_optimizer_and_sampler_with_component_gradients(self):
        from motion_guidance_wan import WanGuidance
        from guidance_utils.wan_transformer import ControlledWanTransformer
        from guidance_utils.motion_probe import MotionProbe
        from probe_report import load_trace

        class SubjectWan(ControlGuidanceMixin, SubjectLossMixin, ForwardAdjacentMixin, NoisedReferenceMixin, WanGuidance):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); finals = {}; losses = {}
            for mode in ('uniform', 'balanced'):
                for recorded in (False, True):
                    name = mode+('_recorded' if recorded else '_plain'); folder = root/name; folder.mkdir()
                    torch.manual_seed(17)
                    g = SubjectWan.__new__(SubjectWan); torch.nn.Module.__init__(g)
                    g.config = OmegaConf.create(dict(probe=True, probe_blocks=[1], probe_steps=[9], probe_rope=False,
                        guidance_blocks=[1], injection_blocks=[], loss_type='flow', flow_head=30,
                        motion_temp=2., softmax_fp32=True, argmax_motion_flow=True, threshloss=True, flow_max_disp=100.,
                        optimization_steps=5, verbose=False, save_embeds=False, flow_loss='mse',
                        reference_noise_step=9, reference_noise_seed=29, flow_pair_mode=pairs.PAIR_MODE,
                        alignment_mode=mode, record_region_gradients=recorded))
                    g.device, g.dtype = torch.device('cuda'), torch.bfloat16
                    g.transformer = ControlledWanTransformer(patch_size=(1, 2, 2), num_attention_heads=40,
                        attention_head_dim=8, in_channels=4, out_channels=4, text_dim=16, freq_dim=16,
                        ffn_dim=32, num_layers=3, cross_attn_norm=True, qk_norm='rms_norm_across_heads',
                        eps=1e-6, rope_max_seq_len=64).to(device=g.device, dtype=g.dtype).eval().requires_grad_(False)
                    g.transformer.enable_gradient_checkpointing()
                    g.latent_height, g.latent_width, g.patch_size = 4, 6, 2
                    g.patches_height, g.patches_width, g.latent_num_frames = 2, 3, 3
                    g.checkpoint_amf, g._guidance_scale = True, 5
                    g.scheduler = FlowMatchEulerDiscreteScheduler(shift=3.); g.scheduler.set_timesteps(50, device=g.device)
                    g.timesteps = g.scheduler.timesteps; g.lr_by_step = {9: .001}; g.output_path = str(folder)
                    g.register_guidance([1]); g.register_attention_processor([0, 1, 2]); g.probe = MotionProbe(g, 'wan')
                    g.motion_latent = torch.randn(1, 4, 3, 4, 6, device=g.device, dtype=g.dtype)*10
                    g.transformer.init_rope = g.transformer.default_rope(g.motion_latent).to(g.device)
                    g.source_embeds = torch.randn(1, 5, 16, device=g.device, dtype=g.dtype)
                    g.guidance_embeds = torch.randn(2, 5, 16, device=g.device, dtype=g.dtype)
                    g.motion_timestep = torch.tensor([0], device=g.device)
                    g.motion_attn_features = ForwardAdjacentMixin.load_attn_features(g)
                    block = 'block_1_attn1_processor'
                    valid = g.motion_attn_masks[block].cpu().numpy()
                    subject = np.zeros_like(valid); positions = np.argwhere(valid)
                    for pair, pos in positions[:max(1, len(positions)//3)]: subject[pair, pos] = True
                    background = valid & ~subject
                    field = g.motion_attn_features[block].float().cpu().numpy().copy()
                    field[subject] += np.array([1., -.5], dtype=np.float32)
                    bundle = dict(flow=field, mask=valid, subject=subject, background=background)
                    g.alignment = {k: torch.as_tensor(bundle[k], device=g.device) for k in ('flow', 'subject', 'background')}
                    target_path = folder/'target.npz'; np.savez_compressed(target_path, **bundle)
                    np.savez_compressed(g.probe.path/'aligned_reference.npz', **bundle)
                    g.probe.emit('aligned_reference', block=block, file='aligned_reference.npz', mode=mode)
                    x = torch.randn_like(g.motion_latent).float(); g.control = ControlRecorder(g, block=1, head=30)
                    try:
                        with torch.no_grad(), patch('probe_wan_head_visual.decode'):
                            for i, t in enumerate(g.timesteps):
                                before = x.clone()
                                if i == 9:
                                    with torch.enable_grad(): x, _ = g.guidance_step(x, i, t, 'latent', 'flow')
                                optimized = x
                                x = g.denoise_step(x, i, g.guidance_embeds)
                                g.probe.sampling(i, t, before, optimized, x, g.scheduler)
                    finally:
                        g.control.close()
                    finals[name] = x.clone()
                    _, _, events = load_trace(folder)
                    control.noised.audit_trace(events, OmegaConf.to_container(g.config), 'candidate')
                    losses[name] = pilot.audit_losses(folder, target_path, mode)
                    for path in (folder/'control').glob('*.npz'):
                        control.audit_capture(path, shape=(1, 4, 3, 4, 6), expected_block=1)
                    if recorded:
                        rows = pilot.gradient_rows(folder, mode)
                        self.assertEqual(len(rows), 5)
                        self.assertTrue(all(r['subject_gradient_rms'] > 0 and r['background_gradient_rms'] > 0 for r in rows))
                        # Corrupt a captured iterate: the trajectory check must reject it.
                        path = folder/'region_gradients/iteration_02.npz'; saved = pilot.arrays(path)
                        np.savez_compressed(path, **{**saved, 'latent': saved['latent']+1})
                        with self.assertRaisesRegex(ValueError, 'trajectory'):
                            pilot.gradient_rows(folder, mode)
                    self.assertTrue(all(p.grad is None for p in g.transformer.parameters()))
                torch.testing.assert_close(finals[mode+'_plain'], finals[mode+'_recorded'], rtol=0, atol=0)
                self.assertEqual(losses[mode+'_plain'], losses[mode+'_recorded'])
            self.assertFalse(torch.equal(finals['uniform_plain'], finals['balanced_plain']))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
