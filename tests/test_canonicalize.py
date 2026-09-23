"""benchmark/canonicalize.py: crop geometry, exactness, and the lossless round trip.

Run:  python tests/test_canonicalize.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import canonicalize as cz                              # noqa: E402

H, W = 480, 720


def test_crop_box_davis_no_resize():
    l, t, r, b = cz.crop_box(480, 854, H, W)
    assert (l, t, r, b) == (67, 0, 787, 480)          # 720 wide, centred, no vertical crop
    fr = np.zeros((2, 480, 854, 3), np.uint8)
    fr[:, :, 67] = 255                                # first kept column
    out, rec = cz.canonicalize_frames(fr, H, W)
    assert out.shape == (2, H, W, 3) and rec["resize_factor"] == 1.0
    assert out[:, :, 0].max() == 255 and out[:, :, 1].max() == 0


def test_crop_box_miradata_exact_two_thirds():
    l, t, r, b = cz.crop_box(720, 1280, H, W)
    assert (r - l, b - t) == (1080, 720) and l == 100  # 3:2 crop at native res, centred
    fr = np.random.default_rng(0).integers(0, 255, (1, 720, 1280, 3), np.uint8)
    out, rec = cz.canonicalize_frames(fr, H, W)
    assert out.shape == (1, H, W, 3)
    assert abs(rec["resize_factor"] - 2 / 3) < 1e-12   # exact, because crop came first


def test_taller_source_crops_height():
    l, t, r, b = cz.crop_box(1000, 1000, H, W)         # square -> crop rows, keep width
    assert (l, r) == (0, 1000) and (b - t) == 666 and t == 167


def test_generated_shape_passes_through_unchanged():
    fr = np.random.default_rng(1).integers(0, 255, (3, H, W, 3), np.uint8)
    out, rec = cz.canonicalize_frames(fr, H, W)
    assert np.array_equal(out, fr) and rec["crop_ltrb"] == [0, 0, W, H]


def test_lossless_round_trip():
    fr = np.random.default_rng(2).integers(0, 255, (4, H, W, 3), np.uint8)  # worst case for codecs
    with tempfile.TemporaryDirectory() as d:
        dst = Path(d) / "x.mp4"
        cz.write_lossless(fr, dst, 12.0)
        back = cz.read_back(dst)
    assert back.shape == fr.shape and np.array_equal(back, fr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
