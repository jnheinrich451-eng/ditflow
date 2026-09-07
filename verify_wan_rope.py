"""Independent complex-number RoPE oracle and known pixel-motion controls."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed
from guidance_utils.wan_modules import apply_rotary_emb
from guidance_utils.motion_probe import adjacent_attention
from guidance_utils.wan_rope_diagnostics import make_control_frames, evaluation_mask, make_rope_report


def complex_oracle(q, frames, height, width):
    """Independent t/y/x phase construction, following the original Wan formula."""
    d = q.shape[-1]
    spatial = 2 * (d // 6)
    dims = [d - 2 * spatial, spatial, spatial]
    t, y, x = torch.meshgrid(torch.arange(frames), torch.arange(height), torch.arange(width), indexing='ij')
    phases = []
    for position, channels in zip((t, y, x), dims):
        inverse = 10000. ** (-torch.arange(0, channels, 2, dtype=torch.float64) / channels)
        phases.append(position.reshape(-1, 1) * inverse)
    angle = torch.cat(phases, dim=-1)[None, :, None, :]
    z = torch.view_as_complex(q.double().reshape(*q.shape[:-1], -1, 2))
    return torch.view_as_real(z * torch.polar(torch.ones_like(angle), angle)).flatten(-2)


class RopeTests(unittest.TestCase):
    def test_rotation_against_independent_complex_oracle(self):
        # Real head dimension, non-square grid, multiple temporal positions.
        torch.manual_seed(7)
        f, h, w, d = 6, 30, 52, 128
        rope = WanRotaryPosEmbed(d, (1, 2, 2), 1024)
        cos, sin = rope(torch.zeros(1, 1, f, h * 2, w * 2))
        q = torch.randn(1, f*h*w, 1, d, dtype=torch.float64)
        actual = apply_rotary_emb(q, cos, sin)
        expected = complex_oracle(q, f, h, w)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(actual.square().sum(-1), q.square().sum(-1), atol=2e-5, rtol=2e-6)
        inverse = apply_rotary_emb(actual, cos, -sin)
        torch.testing.assert_close(inverse, q, atol=2e-6, rtol=2e-6)

    def test_content_translation_sign_before_and_after_rope(self):
        torch.manual_seed(1)
        f, h, w, d = 3, 3, 5, 128
        identity = torch.randn(h*w, d, dtype=torch.float64)
        identity = identity / identity.norm(dim=-1, keepdim=True) * 20
        q = identity.repeat(f, 1)[None, :, None, :]
        # Each frame's content shifts one spatial token to the right.
        q = torch.stack([torch.roll(identity, i, 0) for i in range(f)]).reshape(1, f*h*w, 1, d)
        k = q.clone()
        rope = WanRotaryPosEmbed(d, (1, 2, 2), 64)
        cos, sin = rope(torch.zeros(1, 1, f, h*2, w*2))
        for a, b in [(q, k), (apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin))]:
            fields = adjacent_attention(a, b, h, w, f, 2)['hard']
            valid = np.arange(h*w) % w < w-1
            np.testing.assert_array_equal(fields[:, valid], np.broadcast_to([1, 0], fields[:, valid].shape))

    def test_pixel_controls_and_evaluation_regions(self):
        first = np.random.default_rng(3).integers(0, 256, (480, 832, 3), dtype=np.uint8)
        static, _ = make_control_frames(first, 'static')
        np.testing.assert_array_equal(static[0], static[-1])
        for name, shift in [('pan_right', 4), ('pan_left', -4)]:
            frames, info = make_control_frames(first, name)
            np.testing.assert_array_equal(frames[1], np.roll(first, shift, axis=1))
            self.assertEqual(info['expected_dx_per_video_frame'], shift)
            mask = evaluation_mask(info, 0, 30, 52).reshape(30, 52)
            self.assertFalse(mask[:, :5].any())
            self.assertFalse(mask[:, -5:].any())
        patch, info = make_control_frames(first, 'patch_right')
        for i in range(5):
            self.assertTrue(evaluation_mask(info, i, 30, 52).any())
        np.testing.assert_array_equal(patch[0][:100], patch[-1][:100])
        x0, y0, x1, y1 = info['frame_boxes'][0]
        np.testing.assert_array_equal(patch[0][y0:y1, x0:x1], patch[1][y0:y1, x0+4:x1+4])

    def test_rope_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = dict(grid=[2, 2, 3], packages={}, config=dict(model_key='synthetic', width=48, height=32,
                        rope_control=dict(kind='pan_right', width=48, height=32, expected_dx_per_video_frame=4)))
            (root/'metadata.json').write_text(json.dumps(meta))
            events = []
            for variant in ['pre_rope', 'temporal_only', 'spatial_only', 'full_rope']:
                filename = variant + '.npz'
                np.savez(root/filename, hard=np.broadcast_to([1., 0.], (1, 6, 2)))
                events.append(dict(kind='rope_attention', stage='reference', block='block_15', variant=variant, file=filename))
            (root/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
            report = make_rope_report([root], root/'report')
            self.assertIn('full_rope', report.read_text(encoding='utf-8'))
            self.assertTrue((root/'report/rope_metrics.csv').exists())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
