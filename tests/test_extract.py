"""extract-v1 packer CPU tests on synthetic geometry (closed-form scenes, no
model anywhere — CLAUDE.md §3.2).

Run:  python tests/test_extract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.extract import pack                            # noqa: E402

F, H, W = 12, 64, 64
K0 = np.array([[60.0, 0, 31.5], [0, 60.0, 31.5], [0, 0, 1]])


def rotz_y(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def scene(trans=0.05, yaw=0.0, obj_speed=0.1, seed=0):
    """Camera moves along +x by `trans`/frame and yaws by `yaw`/frame; 150
    static world points + 20 points of one object moving along +x."""
    rng = np.random.default_rng(seed)
    G = np.stack([np.eye(4) for _ in range(F)])
    for t in range(F):
        G[t, :3, :3] = rotz_y(yaw * t)
        G[t, :3, 3] = [trans * t, 0, 0]
    Xs = np.c_[rng.uniform(-3, 3, 150), rng.uniform(-2, 2, 150), rng.uniform(4, 12, 150)]
    Xo0 = np.c_[rng.uniform(-.3, .3, 20), rng.uniform(-.3, .3, 20), rng.uniform(5, 5.5, 20)]
    Xw = np.zeros((170, F, 3))
    Xw[:150] = Xs[:, None]
    Xw[150:] = Xo0[:, None] + np.arange(F)[None, :, None] * np.array([obj_speed, 0, 0])
    Xl = np.stack([(Xw[:, t] - G[t, :3, 3]) @ G[t, :3, :3] for t in range(F)], 1)
    K = np.repeat(K0[None], F, 0)
    P, z = pack.project(K[None], Xl)
    Q = (z > 0) & (P[..., 0] > 0) & (P[..., 0] < W - 1) & (P[..., 1] > 0) & (P[..., 1] < H - 1)
    Q[:, 0] = True
    D = np.repeat(np.linspace(4, 12, H)[None, :, None], F, 0) * np.ones((F, H, W))
    M = np.zeros((F, H, W), bool)
    M[:, 28:36, 28:36] = True
    D[M] = 5.0
    sub = np.arange(150, 160)                       # sparse subject tracks
    y_dyn = np.zeros(170, bool)
    y_dyn[150:] = True
    cid = -np.ones(170, int)
    cid[150:160], cid[160:170] = 3, 7               # 160..169: another cluster
    d = {"K": K, "G": G, "D": D.astype(np.float16), "X_local": Xl[:170],
         "P": P, "Q": Q, "anchors": np.zeros((170, 3)),
         "prov": {"native_hw": [480, 640]},
         "qc": {"sim3_scale_per_frame": [1.0] * F, "window_boundaries": []},
         "M": M, "present": np.ones(F, bool), "t_c": 0, "channel": "x0",
         "B": {"y_dyn": y_dyn, "cluster_id": cid, "o_src_members": sub},
         "C5": {"X_local_dense": Xl[150:155], "P_dense": P[150:155],
                "Q_dense": Q[150:155], "anchors_dense": np.zeros((5, 3))}}
    return d


CFG = {"bg_depth_percentiles": [5, 95]}


def test_world_lift_and_labels():
    d = scene()
    r = pack.build_raw(d, 16.0, "{}")
    Xw = r["track_xyz"][:150]
    assert np.abs(Xw - Xw[:, :1]).max() < 1e-4, "static points must be fixed in world"
    lab = r["track_label"]
    assert (lab[:150] == 0).all() and (lab[150:160] == 1).all()
    assert (lab[160:170] == 2).all() and (lab[170:] == 1).all()    # dense -> subject
    assert r["instance_ids"].tolist() == [1, 2]
    assert r["instance_has_mask"].tolist() == [True, False]
    assert r["instance_stage_b_cluster"].tolist() == [-1, 7]
    assert np.allclose(r["cam_extrinsics"][0], np.eye(4))


def test_flow_cam_explains_static_and_isolates_object():
    d = scene(trans=0.05, yaw=0.01, obj_speed=0.1)
    r = pack.build_raw(d, 16.0, "{}")
    der, js = pack.build_derived(r, CFG)
    v = der["flow_cam_valid"] & der["flow_obs_valid"]
    res = der["flow_obs"] - der["flow_cam"]
    assert np.abs(res[:150][v[:150]]).max() < 1e-3, "static residual must vanish"
    idx = der["residual_obj_idx"]
    assert idx.tolist() == list(range(150, r["track_label"].size))
    ro = der["residual_obj"][der["residual_obj_valid"]]
    assert np.median(np.abs(ro[:, 0])) > 0.5, "object motion must survive in residual"
    assert js["instances"]["1"]["salient"] and not js["instances"]["2"]["salient"]
    traj = np.array(js["instances"]["2"]["world_traj"])
    assert np.allclose(np.diff(traj[:, 0]), 0.1, atol=1e-4), "world object velocity"


def test_pure_rotation_has_no_parallax():
    d = scene(trans=0.0, yaw=0.01)
    r = pack.build_raw(d, 16.0, "{}")
    der, js = pack.build_derived(r, CFG)
    assert js["delta_parallax_px"] < 1e-6
    assert abs(js["rho"] - 1.0) < 1e-3, js["rho"]
    assert np.allclose(der["flow_rot"], der["flow_cam"], atol=1e-3)


def test_pure_translation_parallax_formula():
    d = scene(trans=0.05, yaw=0.0)
    r = pack.build_raw(d, 16.0, "{}")
    der, js = pack.build_derived(r, CFG)
    assert js["rho"] < 1e-6
    bg = ~d["M"][0]
    Zlo, Zhi = np.percentile(d["D"][0][bg].astype(float), [5, 95])
    want = (F - 1) * 60.0 * 0.05 * (1 / Zlo - 1 / Zhi)
    assert abs(js["delta_parallax_px"] - want) < 1e-3 * want
    assert abs(js["instances"]["1"]["obj_bg_disparity_px"]
               - (F - 1) * 60.0 * 0.05 * abs(1 / 5.0 - 1 / np.median(d["D"][0][bg].astype(float)))) < 1e-3
    assert np.allclose(js["cam_centers"][-1], [0.05 * (F - 1), 0, 0])


def test_refuses_nonfinite():
    d = scene()
    d["G"] = d["G"].copy()
    d["G"][3, 0, 3] = np.nan
    try:
        pack.build_raw(d, 16.0, "{}")
    except AssertionError:
        return
    raise AssertionError("non-finite extrinsics were packed")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
