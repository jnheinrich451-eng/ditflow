"""object_metrics + cameractrl_errors on synthetic inputs with known answers.

Run:  python tests/test_object_metrics.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import camera_metrics as cm                           # noqa: E402
from benchmark import object_metrics as om                           # noqa: E402
from tests.test_camera_metrics import rot, traj, T                   # noqa: E402

H, W = 48, 72


def fake_extract(d: Path, box_x0: np.ndarray, root: np.ndarray, zbg=5.0):
    """Minimal extract-v1 output: a 10x8 subject box at x0[t], and a world root trajectory."""
    d.mkdir(parents=True, exist_ok=True)
    n = len(box_x0)
    inst = np.zeros((n, H, W), np.uint8)
    for t, x in enumerate(box_x0):
        inst[t, 20:28, int(x):int(x) + 10] = 1
    np.savez(d / "raw.npz", instances=inst)
    np.savez(d / "derived.npz", Z_bg_med=np.full(n, zbg, np.float32))
    (d / "derived.json").write_text(json.dumps({"instances": {"1": {"world_traj": root.tolist()}}}))


def test_identical_pair_scores_perfect():
    with tempfile.TemporaryDirectory() as tmp:
        x = np.arange(10) * 3.0
        root = np.c_[np.arange(10) * 0.1, np.zeros(10), np.full(10, 5.0)]
        fake_extract(Path(tmp, "r"), x, root); fake_extract(Path(tmp, "g"), x, root)
        e = om.object_errors(Path(tmp, "r"), Path(tmp, "g"), range(10), range(10), cam_scale=1.0)
    assert abs(e["box_iou"] - 1) < 1e-12 and e["objmc_px"] == 0 and e["obj_err3d"] == 0


def test_shift_and_scale_known_answers():
    with tempfile.TemporaryDirectory() as tmp:
        x = np.arange(10) * 3.0
        root = np.c_[np.arange(10) * 0.1, np.zeros(10), np.full(10, 5.0)]
        fake_extract(Path(tmp, "r"), x, root)
        fake_extract(Path(tmp, "g"), x + 5, 0.5 * root + 7.0)      # 5 px right; half-scale, offset world
        e = om.object_errors(Path(tmp, "r"), Path(tmp, "g"), range(10), range(10), cam_scale=2.0)
    assert abs(e["objmc_px"] - 5.0) < 1e-9                         # image centroid moved 5 px
    assert e["objmc_motion_px"] < 1e-9                             # ...but a constant offset is not motion
    assert abs(e["box_iou"] - 5 / 15) < 1e-9                       # 10-wide boxes offset by 5: 5/15
    assert e["obj_err3d"] < 1e-9                                   # same motion once scaled; offset cancels


def test_frame_correspondence_is_used():
    with tempfile.TemporaryDirectory() as tmp:
        root = np.c_[np.arange(24) * 0.1, np.zeros(24), np.full(24, 5.0)]
        fake_extract(Path(tmp, "r"), np.arange(24) * 2.0, root)
        fake_extract(Path(tmp, "g"), np.arange(24)[::2] * 2.0, root[::2])   # generated = every other ref frame
        e = om.object_errors(Path(tmp, "r"), Path(tmp, "g"), range(0, 24, 2), range(12), cam_scale=1.0)
    assert e["objmc_px"] == 0 and e["obj_err3d"] < 1e-12


def test_cameractrl_identity_and_uniform_scale():
    E = traj()
    a = cm.cameractrl_errors(E, E)
    assert a["RotErr_sum_rad"] < 1e-9 and a["TransErr_sum_sq"] < 1e-18 and abs(a["rescale"] - 1) < 1e-12
    Es = E.copy(); Es[:, :3, 3] *= 0.3
    b = cm.cameractrl_errors(Es, E)                              # first-two-frame rescale undoes a uniform scale
    assert abs(b["rescale"] - 1 / 0.3) < 1e-9 and b["TransErr_sum_sq"] < 1e-18


def test_cameractrl_rotation_is_summed_radians():
    E = traj()
    Er = E.copy()
    for t in range(T):
        Er[t, :3, :3] = E[t, :3, :3] @ rot([0, 1, 0], 2.0)        # 2 deg body-frame offset on every frame
    Er[0] = E[0]                                                  # frame 0 is the reference: identical
    a = cm.cameractrl_errors(Er, E)
    assert abs(a["RotErr_sum_rad"] - (T - 1) * np.radians(2.0)) < 1e-6, a["RotErr_sum_rad"]


def test_cameractrl_rescale_is_fragile():
    """Documents why CameraCtrl's numbers are reported for comparability only: a small error on
    frame 1 alone rescales the whole trajectory; least-squares scale_only barely moves."""
    E = traj()
    En = E.copy()
    En[1, :3, 3] += (En[1, :3, 3] - En[0, :3, 3]) * 0.3           # 30% error on ONE frame's step
    a = cm.cameractrl_errors(En, E)
    b = cm.camera_errors(cm.to_camera0(En), cm.to_camera0(E), align="scale_only")
    assert abs(a["rescale"] - 1) > 0.2                           # whole trajectory mis-scaled
    assert abs(b["scale"] - 1) < 0.05                            # robust alternative


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and fn.__module__ == __name__:
            fn()
            print(f"PASS {name}")
