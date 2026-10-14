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


def test_cut_window_takes_the_right_frames():
    """Frame k of the cut must be source frame start+k (each source frame encodes its index)."""
    src = np.zeros((40, 72, 128, 3), np.uint8)
    for i in range(40):
        src[i, :, : 3 * i + 3] = 255                     # a white bar whose width encodes the index
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "src.mp4"
        cz.write_lossless(src, p, 30.0)
        cut = cz.cut_window(p, 4, 21)
        try:
            cz.cut_window(p, 30, 21)
        except SystemExit:
            pass
        else:
            raise AssertionError("short video was accepted")
    widths = (cut[:, 0, :, 0] > 127).sum(1)
    assert cut.shape[0] == 21 and list(widths) == [3 * (4 + k) + 3 for k in range(21)], widths


def test_full_frame_resize_lands_on_reference_geometry():
    """Source 1280x720 with a marker; route A = reference (crop then resize), route B = Wan-style
    full-frame resize to 832x480 then from_full_frame_resize. The marker must land in the same place."""
    from PIL import Image
    src = np.zeros((1, 720, 1280, 3), np.uint8)
    src[0, 300:360, 600:660] = 255                                # 60x60 white square
    a, rec = cz.canonicalize_frames(src, H, W)
    wan = np.asarray(Image.fromarray(src[0]).resize((832, 480), Image.LANCZOS))[None]
    b, rb = cz.from_full_frame_resize(wan, (720, 1280), rec["crop_ltrb"], H, W)
    assert rb["crop_in_generated"] == [65.0, 0.0, 767.0, 480.0], rb
    ca = np.argwhere(a[0, :, :, 0] > 127).mean(0); cb = np.argwhere(b[0, :, :, 0] > 127).mean(0)
    assert np.abs(ca - cb).max() < 1.0, (ca, cb)                  # same place within a pixel


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
