"""T0.4 of the Q1 protocol: camera_metrics on synthetic trajectories with known answers.

Run:  python tests/test_camera_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import camera_metrics as cm                          # noqa: E402

T = 20
rng = np.random.default_rng(0)


def rot(axis, deg):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.radians(deg)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def traj(seed=0, static=False):
    """Smooth c2w OpenCV trajectory: arc translation + slow yaw."""
    r = np.random.default_rng(seed)
    E = np.repeat(np.eye(4)[None], T, 0)
    for t in range(T):
        E[t, :3, :3] = rot([0, 1, 0], 1.5 * t) @ rot(r.normal(size=3), 0.3 * t)
        E[t, :3, 3] = 0 if static else [0.1 * t, 0.02 * t ** 2 / T, 0.05 * np.sin(t / 3)]
    return E


def se3(seed):
    r = np.random.default_rng(seed)
    G = np.eye(4)
    G[:3, :3] = rot(r.normal(size=3), r.uniform(10, 170))
    G[:3, 3] = r.normal(size=3) * 3
    return G


def test_identity():
    E = traj()
    e = cm.camera_errors(E, E)
    for k in ("ate", "rot_mean_deg", "trans_mean", "rpe_rot_deg", "rpe_trans"):
        assert abs(e[k]) < 1e-9, (k, e[k])
    assert abs(e["scale"] - 1) < 1e-9 and e["align_used"] == "sim3"


def test_known_scale():
    E = traj()
    Es = E.copy()
    Es[:, :3, 3] *= 0.37
    e = cm.camera_errors(Es, E)
    assert abs(e["scale"] - 1 / 0.37) < 1e-9, e["scale"]
    assert e["ate"] < 1e-9 and e["rot_mean_deg"] < 1e-9


def test_rigid_offset():
    E = traj()
    e = cm.camera_errors(se3(3) @ E, E)
    assert e["ate"] < 1e-9 and e["rot_mean_deg"] < 1e-9 and abs(e["scale"] - 1) < 1e-9


def test_known_rotation_offset():
    E = traj()
    Er = E.copy()
    for t in range(T):                                   # per-frame 3 deg about a fixed body axis
        Er[t, :3, :3] = E[t, :3, :3] @ rot([0.3, 1, 0.2], 3.0)
    e = cm.camera_errors(Er, E, align="none")
    assert abs(e["rot_mean_deg"] - 3.0) < 1e-6, e["rot_mean_deg"]
    assert abs(e["rot_sum_deg"] - 3.0 * T) < 1e-5
    assert e["trans_mean"] < 1e-9


def test_known_translation_offset():
    E = traj()
    Et = E.copy()
    off = rng.normal(size=(T, 3))
    Et[:, :3, 3] += off
    e = cm.camera_errors(Et, E, align="none")
    want = np.linalg.norm(off, axis=1)
    assert abs(e["trans_mean"] - want.mean()) < 1e-9
    assert abs(e["trans_sum"] - want.sum()) < 1e-9        # summed form = mean x frames


def test_noise_grows_and_rpe_ignores_global_drift():
    E = traj()
    ates = []
    for sig in (0.01, 0.05, 0.2):
        En = E.copy()
        En[:, :3, 3] += np.random.default_rng(1).normal(scale=sig, size=(T, 3))
        ates.append(cm.camera_errors(En, E)["ate"])
    assert ates[0] < ates[1] < ates[2], ates
    En = E.copy()
    En[:, :3, 3] += np.random.default_rng(1).normal(scale=0.05, size=(T, 3))
    a, b = cm.camera_errors(En, E, align="none"), cm.camera_errors(se3(7) @ En, E, align="none")
    assert abs(a["rpe_rot_deg"] - b["rpe_rot_deg"]) < 1e-9 and abs(a["rpe_trans"] - b["rpe_trans"]) < 1e-9


def test_frame_subset():
    E, En = traj(), traj()
    En[:, :3, 3] += rng.normal(scale=0.05, size=(T, 3))
    idx = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    a = cm.camera_errors(En, E, idx=idx, align="none")
    b = cm.camera_errors(En[idx], E[idx], align="none")
    for k in ("ate", "rot_mean_deg", "trans_mean", "rpe_rot_deg", "rpe_trans"):
        assert abs(a[k] - b[k]) < 1e-12, k
    assert a["n"] == 10


def test_convention_round_trip():
    E = traj()
    gl = E.copy()
    gl[:, :3, :3] = E[:, :3, :3] @ np.diag([1, -1, -1])   # what a c2w OpenGL export looks like
    assert np.allclose(cm.convert_convention(gl, "c2w_opengl"), E, atol=1e-12)
    w2c = np.linalg.inv(E)
    assert np.allclose(cm.convert_convention(w2c, "w2c_opencv"), E, atol=1e-9)
    w2c_gl = np.linalg.inv(gl)
    assert np.allclose(cm.convert_convention(w2c_gl, "w2c_opengl"), E, atol=1e-9)
    assert np.allclose(cm.convert_convention(E, "c2w_opencv"), E)


def test_convention_mismatch_is_loud():
    E = traj()
    gl = E.copy()
    gl[:, :3, :3] = E[:, :3, :3] @ np.diag([1, -1, -1])
    e = cm.camera_errors(gl, E, align="none")             # OpenGL fed as if OpenCV
    assert e["rot_mean_deg"] > 179.9, e["rot_mean_deg"]   # 180 deg about x on every frame
    e2 = cm.camera_errors(gl, E)                          # even after sim(3) it cannot hide
    assert e2["rot_mean_deg"] > 179.9


def test_degenerate_static_camera():
    E = traj(static=True)
    try:
        cm.umeyama_sim3(E[:, :3, 3], E[:, :3, 3])
    except cm.DegenerateError:
        pass
    else:
        raise AssertionError("sim3 accepted a static trajectory")
    Er = E.copy()
    for t in range(T):
        Er[t, :3, :3] = rot([0, 0, 1], 2.0) @ E[t, :3, :3]   # a global rotation offset
    e = cm.camera_errors(Er, E)
    assert e["align_used"] == "rot_only"
    assert e["rot_mean_deg"] < 1e-9                        # rot-only alignment removes it
    assert np.isfinite(e["rpe_rot_deg"])


def test_length_dependence():
    E = traj()
    Et = E.copy()
    Et[:, :3, 3] += 0.1
    a = cm.camera_errors(Et[:10], E[:10], align="none")
    b = cm.camera_errors(Et, E, align="none")
    assert abs(a["trans_mean"] - b["trans_mean"]) < 1e-12
    assert abs(b["trans_sum"] - 2 * a["trans_sum"]) < 1e-12


def test_to_camera0():
    E = traj()
    E0 = cm.to_camera0(se3(5) @ E)
    assert np.allclose(E0[0], np.eye(4), atol=1e-12)
    assert cm.camera_errors(E0, E)["ate"] < 1e-9


def test_scale_only_on_anchored_near_linear_path():
    """Both trajectories anchored at frame 0 and the path nearly a line: a free
    sim(3) rotation is unconstrained and leaks into RotErr; scale_only must not."""
    E = np.repeat(np.eye(4)[None], T, 0)
    for t in range(T):                                    # straight line + tiny wobble, slow yaw
        E[t, :3, 3] = [0.1 * t, 1e-4 * np.sin(t), 0]
        E[t, :3, :3] = rot([0, 1, 0], 0.5 * t)
    Es = E.copy()
    Es[:, :3, 3] *= 0.11                                  # D4RT-like scale
    Es[:, :3, 3] += np.random.default_rng(2).normal(scale=2e-3, size=(T, 3))
    Es[0, :3, 3] = 0
    a = cm.camera_errors(Es, E, align="scale_only")
    assert abs(a["scale"] - 1 / 0.11) < 0.05, a["scale"]
    assert a["rot_mean_deg"] < 1e-9, a["rot_mean_deg"]  # orientations were identical
    b = cm.camera_errors(Es, E, align="sim3")
    assert b["rot_mean_deg"] > a["rot_mean_deg"]         # the artefact sim(3) introduces


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
