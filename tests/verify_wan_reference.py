"""CPU checks for reference image-motion diagnostics; no pretrained weights."""
import sys
from pathlib import Path

# Tests import repo modules by their root names; make `python tests/<file>.py` work from the root checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from guidance_utils.wan_reference_diagnostics import comparison, image_motion, reference_images


class ReferenceDiagnosticsTests(unittest.TestCase):
    def textured(self):
        a = np.random.default_rng(4).integers(0, 256, (160, 256, 3), dtype=np.uint8)
        return cv2.GaussianBlur(a, (5, 5), 0)

    def test_translation_sign_and_patch_units(self):
        first = self.textured()
        for shift in (-8, 8):
            second = np.roll(first, shift, axis=1)
            flow, valid = image_motion(first, second, (20, 32))
            selected = valid.copy()
            selected[:, :5] = selected[:, -5:] = False
            self.assertGreater(selected.sum(), 200)
            np.testing.assert_allclose(np.median(flow[selected], axis=0), [shift / 8, 0], atol=.15)

    def test_static_texture_and_unsupported_flat_image(self):
        first = self.textured()
        flow, valid = image_motion(first, first, (20, 32))
        self.assertGreater(valid.sum(), 300)
        self.assertLess(float(np.median(np.linalg.norm(flow[valid], axis=-1))), .02)
        _, valid = image_motion(np.zeros_like(first), np.zeros_like(first), (20, 32))
        self.assertFalse(valid.any())

    def test_zero_amf_is_not_dropped_from_direction_score(self):
        measured = np.array([[1., 0.], [1., 0.], [0., 0.]])
        amf = np.array([[1., 0.], [0., 0.], [1., 0.]])
        result = comparison(amf, measured, np.ones(3, dtype=bool))
        self.assertEqual(result['moving_patches'], 2)
        self.assertEqual(result['cosine'], .5)
        self.assertIsNone(comparison(amf, measured, np.zeros(3, dtype=bool))['cosine'])

    def test_background_subtraction_distinguishes_relative_direction(self):
        measured = np.array([[-1., 0.], [-3., 0.]])  # Subject right relative to camera-tracked background.
        amf = np.array([[-4., 0.], [-3., 0.]])
        subject = np.array([True, False])
        self.assertEqual(comparison(amf, measured, subject)['cosine'], 1.)
        self.assertEqual(comparison(amf-amf[1], measured-measured[1], subject)['cosine'], -1.)

    def test_packed_input_sorting_and_short_input(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index in (10, 2, 0):
                Image.fromarray(np.full((16, 16, 3), index, dtype=np.uint8)).save(root / f'f{index}.png')
            frames = reference_images(root, 3, (16, 16))
            self.assertEqual([int(f[0, 0, 0]) for f in frames], [0, 2, 10])
            with self.assertRaises(ValueError): reference_images(root, 4, (16, 16))


if __name__ == '__main__':
    unittest.main()
