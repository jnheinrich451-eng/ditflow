"""extract-v1 packer: Module-1 caches -> raw + derived (pure numpy, no model).

The measuring device of the experiment chapter (docs/experiment-pipeline-v1.md
S0). It turns a video into camera, depth, masks and tracks and NOTHING else:
no metric, no threshold, no reference to a target. Errors against an intended
camera/object live in the scorer.

Every field is re-expressed from the frozen Stage A contract
(docs/contract_stage_a_v1.md) plus the Stage B / C / C.5 caches; conventions
are the MEASURED ones encoded there (G_{0<-t}, z-depth, OpenCV axes,
process-resolution pixel-centre K), never re-guessed here.
"""
from __future__ import annotations

import numpy as np

SCHEMA = "extract-v1"
CAM_CONVENTION = (
    "c2w_opencv_world_is_camera0: X_world = E[t,:3,:3] @ X_cam_t + E[t,:3,3]; "
    "world = camera-0 (E[0] = I); camera axes x right, y down, z forward; "
    "units = D4RT model units (similarity-ambiguous, NOT metres)")
# track_label values
STATIC, SUBJECT, DYN_UNASSIGNED = 0, 1, -1


# ------------------------------------------------------------------ geometry
def lift_world(G, Xl):
    """[N,F,3] camera-t coords -> world (camera-0) via G_{0<-t}. f64 math."""
    G = np.asarray(G, np.float64)
    Xl = np.asarray(Xl, np.float64)
    return np.einsum("tij,ntj->nti", G[:, :3, :3], Xl) + G[None, :, :3, 3]


def project(K, X):
    """Pinhole projection, pixel-centre convention (same as Stage A / C.5).
    K [3,3] or broadcastable [...,3,3]; X [...,3]. -> (uv [...,2], z [...])."""
    z = X[..., 2]
    zs = np.where(np.abs(z) > 1e-9, z, 1e-9)
    u = K[..., 0, 0] * X[..., 0] / zs + K[..., 0, 2]
    v = K[..., 1, 1] * X[..., 1] / zs + K[..., 1, 2]
    return np.stack([u, v], -1), z


def rel_pose(G, t):
    """G_{t+1<-t}: camera-t coords -> camera-(t+1) coords."""
    G = np.asarray(G, np.float64)
    return np.linalg.inv(G[t + 1]) @ G[t]


# --------------------------------------------------------------------- raw
def subject_indices(stage_b: dict, channel: str) -> np.ndarray:
    """Stage B's own subject set, verbatim (no re-decision here): follow-cam
    branch -> quiet blob (subject + glued halo, measured); x0 -> winner."""
    key = "fc_blob_idx" if channel == "branch" else "o_src_members"
    return np.asarray(stage_b[key], int)


def sparse_labels(stage_b: dict, subj: np.ndarray, N: int):
    """0 static, 1 subject, 2.. other Stage-B dynamic clusters (ranked by
    Stage B cluster id), -1 dynamic but unclustered. -> (labels, id->cluster)."""
    lab = np.zeros(N, np.int16)
    y_dyn = np.asarray(stage_b["y_dyn"]).astype(bool)
    cid = np.asarray(stage_b["cluster_id"], int)
    is_subj = np.zeros(N, bool)
    is_subj[subj] = True
    other = y_dyn & ~is_subj
    lab[other & (cid < 0)] = DYN_UNASSIGNED
    id_map = {}
    for k, c in enumerate(sorted(set(cid[other & (cid >= 0)].tolist()))):
        lab[other & (cid == c)] = 2 + k
        id_map[2 + k] = int(c)
    lab[is_subj] = SUBJECT
    return lab, id_map


def in_mask_counts(P, Q, M):
    """Per track: (# valid frames, # valid frames projecting inside M)."""
    F, H, W = M.shape
    u = np.clip(np.round(P[..., 0]).astype(int), 0, W - 1)
    v = np.clip(np.round(P[..., 1]).astype(int), 0, H - 1)
    t = np.broadcast_to(np.arange(F)[None], Q.shape)
    inside = M[t, v, u] & Q
    return Q.sum(1).astype(np.int32), inside.sum(1).astype(np.int32)


def build_raw(d: dict, fps: float, provenance_json: str) -> dict:
    """d = loaded caches (scripts/plot_thesis_tracks.load layout)."""
    K = np.asarray(d["K"], np.float32)
    G = np.asarray(d["G"], np.float32)
    D = np.asarray(d["D"], np.float32)          # f16 in the bundle; upcast is lossless
    F, H, W = D.shape
    M = np.asarray(d["M"], bool)
    assert M.shape == (F, H, W), f"mask {M.shape} vs depth {(F, H, W)}"
    B, c5 = d["B"], d["C5"]

    N_s = d["Q"].shape[0]
    subj = subject_indices(B, d["channel"])
    lab_s, id_map = sparse_labels(B, subj, N_s)
    N_d = c5["Q_dense"].shape[0]

    Xl = np.concatenate([d["X_local"], c5["X_local_dense"]]).astype(np.float64)
    P = np.concatenate([d["P"], c5["P_dense"]]).astype(np.float32)
    Q = np.concatenate([d["Q"], c5["Q_dense"]]).astype(bool)
    assert P.shape[1] == F and Q.shape == P.shape[:2], "track / frame counts differ"
    label = np.concatenate([lab_s, np.full(N_d, SUBJECT, np.int16)])
    source = np.concatenate([np.zeros(N_s, np.uint8), np.ones(N_d, np.uint8)])
    spawn = np.concatenate([d["anchors"][:, 2], c5["anchors_dense"][:, 2]]).astype(np.int32)
    n_valid, n_in = in_mask_counts(P, Q, M)

    ids = sorted(int(i) for i in set(label.tolist()) if i > 0)
    has_mask = np.array([i == SUBJECT for i in ids], bool)
    raw = {
        "schema": np.array(SCHEMA),
        "cam_extrinsics": G,
        "cam_convention": np.array(CAM_CONVENTION),
        "cam_intrinsics": K,
        "cam_is_metric": np.array(False),
        "cam_scale_drift": np.asarray(d["qc"]["sim3_scale_per_frame"], np.float32),
        "window_boundaries": np.asarray(d["qc"].get("window_boundaries", []), np.int32),
        "depth": D,
        "depth_valid": np.isfinite(D) & (D > 0),
        "instances": M.astype(np.uint8) * SUBJECT,
        "instance_ids": np.asarray(ids, np.int16),
        "instance_has_mask": has_mask,
        "instance_stage_b_cluster": np.asarray(
            [id_map.get(i, -1) for i in ids], np.int32),
        "subject_present": np.asarray(d["present"], bool),
        "t_c": np.array(int(d["t_c"]), np.int32),
        "track_xy": P,
        "track_visible": Q,
        "track_label": label,
        "track_source": source,
        "track_spawn": spawn,
        "track_n_valid": n_valid,
        "track_n_in_mask": n_in,
        "track_xyz": lift_world(G, Xl).astype(np.float32),
        "image_hw": np.array([H, W], np.int32),
        "native_hw": np.asarray(d["prov"]["native_hw"], np.int32),
        "fps": np.array(float(fps), np.float64),
        "channel": np.array(d["channel"]),
        "provenance_json": np.array(provenance_json),
    }
    check_raw(raw)
    return raw


def check_raw(r: dict) -> None:
    F, H, W = r["depth"].shape
    N = r["track_xy"].shape[0]
    assert r["cam_extrinsics"].shape == (F, 4, 4)
    assert r["cam_intrinsics"].shape == (F, 3, 3)
    assert r["instances"].shape == (F, H, W) and r["depth_valid"].shape == (F, H, W)
    assert r["track_xy"].shape == (N, F, 2) and r["track_xyz"].shape == (N, F, 3)
    for k in ("track_visible", "track_label", "track_source", "track_spawn"):
        assert r[k].shape[0] == N, k
    assert r["cam_scale_drift"].shape == (F,) and r["subject_present"].shape == (F,)
    assert r["fps"] > 0, "fps must be positive"
    assert np.allclose(r["cam_extrinsics"][:, 3], [0, 0, 0, 1]), "E not homogeneous"
    for k, v in r.items():
        if np.issubdtype(np.asarray(v).dtype, np.floating):
            assert np.isfinite(v).all(), f"non-finite values in {k}"


# ----------------------------------------------------------------- derived
def camera_flows(r: dict):
    """Per track and step t->t+1: observed flow, camera-induced flow (the
    motion a STATIC point at the track's depth would show from pose alone),
    and rotation-only flow (infinite-depth homography K' R K^-1)."""
    E = r["cam_extrinsics"].astype(np.float64)
    K = r["cam_intrinsics"].astype(np.float64)
    Q = r["track_visible"]
    N, F = Q.shape
    Xw = r["track_xyz"].astype(np.float64)
    P = r["track_xy"].astype(np.float64)
    obs = np.zeros((N, F - 1, 2))
    cam = np.zeros((N, F - 1, 2))
    rot = np.zeros((N, F - 1, 2))
    v_cam = np.zeros((N, F - 1), bool)
    for t in range(F - 1):
        Ei = np.linalg.inv(E[t])
        Xt = Xw[:, t] @ Ei[:3, :3].T + Ei[:3, 3]           # camera-t coords
        Rr = rel_pose(E, t)
        Xn = Xt @ Rr[:3, :3].T + Rr[:3, 3]                 # same point, camera-(t+1)
        p0, z0 = project(K[t], Xt)
        p1, z1 = project(K[t + 1], Xn)
        cam[:, t] = p1 - p0
        Hr = K[t + 1] @ Rr[:3, :3] @ np.linalg.inv(K[t])
        h = np.c_[p0, np.ones(N)] @ Hr.T
        rot[:, t] = h[:, :2] / np.where(np.abs(h[:, 2:]) > 1e-9, h[:, 2:], 1e-9) - p0
        obs[:, t] = P[:, t + 1] - P[:, t]
        v_cam[:, t] = Q[:, t] & (z0 > 0) & (z1 > 0)
    v_obs = Q[:, :-1] & Q[:, 1:]
    return obs, cam, rot, v_cam, v_obs


def build_derived(r: dict, cfg_d: dict):
    """-> (arrays for derived.npz, dict for derived.json). Descriptors only."""
    E = r["cam_extrinsics"].astype(np.float64)
    K = r["cam_intrinsics"].astype(np.float64)
    D = r["depth"].astype(np.float64)
    F = D.shape[0]
    lab = r["track_label"]
    subj_mask = r["instances"] == SUBJECT
    lo, hi = (float(x) for x in cfg_d["bg_depth_percentiles"])

    obs, cam, rot, v_cam, v_obs = camera_flows(r)
    v_both = v_cam & v_obs
    obj_idx = np.flatnonzero(lab != STATIC)

    # background depth spread per frame: every valid pixel outside the subject
    # mask (other movers have no mask in extract-v1; stated in the manifest)
    bg = r["depth_valid"] & ~subj_mask
    Zlo = np.array([np.percentile(D[t][bg[t]], lo) for t in range(F)])
    Zhi = np.array([np.percentile(D[t][bg[t]], hi) for t in range(F)])
    Zbg = np.array([np.median(D[t][bg[t]]) for t in range(F)])
    fg = subj_mask & r["depth_valid"]
    has_obj = fg.reshape(F, -1).any(1)
    Zobj = np.array([np.median(D[t][fg[t]]) if has_obj[t] else 0.0 for t in range(F)])

    # per-step lateral camera translation in camera-t coords, focal at t
    c = E[:, :3, 3]
    Tperp = np.zeros(F - 1)
    for t in range(F - 1):
        cn = np.linalg.inv(E[t]) @ np.r_[c[t + 1], 1.0]
        Tperp[t] = np.linalg.norm(cn[:2])
    f = 0.5 * (K[:, 0, 0] + K[:, 1, 1])
    dpar = f[:-1] * Tperp * (1.0 / Zlo[:-1] - 1.0 / Zhi[:-1])
    obj_ok = has_obj[:-1]
    dobj = np.where(obj_ok, f[:-1] * Tperp * np.abs(
        1.0 / np.where(obj_ok, Zobj[:-1], 1.0) - 1.0 / Zbg[:-1]), 0.0)

    # rho: ratio of per-step medians over static tracks, summed over steps
    # (sum, not per-step ratio: weights steps by camera motion, so a near-
    # static camera does not produce 0/0 noise)
    st = (lab == STATIC)[:, None] & v_cam
    n_rot = np.linalg.norm(rot, axis=-1)
    n_cam = np.linalg.norm(cam, axis=-1)
    med_rot = np.array([np.median(n_rot[st[:, t], t]) if st[:, t].any() else 0.0
                        for t in range(F - 1)])
    med_cam = np.array([np.median(n_cam[st[:, t], t]) if st[:, t].any() else 0.0
                        for t in range(F - 1)])
    steps_static = st.any(0)

    arrays = {
        "schema": np.array(SCHEMA),
        "flow_obs": obs.astype(np.float32), "flow_obs_valid": v_obs,
        "flow_cam": cam.astype(np.float32), "flow_cam_valid": v_cam,
        "flow_rot": rot.astype(np.float32),
        "residual_obj_idx": obj_idx.astype(np.int32),
        "residual_obj": (obs - cam)[obj_idx].astype(np.float32),
        "residual_obj_valid": v_both[obj_idx],
        "Z_bg_lo": Zlo.astype(np.float32), "Z_bg_hi": Zhi.astype(np.float32),
        "Z_bg_med": Zbg.astype(np.float32),
        "Z_obj_med": Zobj.astype(np.float32), "Z_obj_valid": has_obj,
        "T_perp_step": Tperp.astype(np.float32),
        "delta_parallax_step": dpar.astype(np.float32),
        "obj_bg_disparity_step": dobj.astype(np.float32),
        "rho_num_step": med_rot.astype(np.float32),
        "rho_den_step": med_cam.astype(np.float32),
        "rho_step_valid": steps_static,
    }
    for k, v in arrays.items():
        if np.issubdtype(np.asarray(v).dtype, np.floating):
            assert np.isfinite(v).all(), f"non-finite values in derived {k}"

    Xw = r["track_xyz"].astype(np.float64)
    Q = r["track_visible"]
    inst = {}
    for i, has in zip(r["instance_ids"].tolist(), r["instance_has_mask"].tolist()):
        sel = lab == i
        traj, n_sup = [], []
        for t in range(F):
            m = sel & Q[:, t]
            n_sup.append(int(m.sum()))
            traj.append(np.median(Xw[m, t], 0).round(6).tolist() if m.any() else None)
        rec = {"world_traj": traj, "n_support": n_sup,
               "salient": bool(i == SUBJECT), "has_mask": bool(has)}
        if has:
            rec["area_frac"] = [round(float(a), 6) for a in
                                (r["instances"] == i).reshape(F, -1).mean(1)]
            rec["obj_bg_disparity_px"] = round(float(dobj.sum()), 4)
        inst[str(i)] = rec

    steps = np.linalg.norm(np.diff(c, axis=0), axis=1)
    den = float(med_cam[steps_static].sum())
    js = {
        "schema": SCHEMA,
        "dt_s": 1.0 / float(r["fps"]),
        "units": "px for image quantities; D4RT model units for 3D (not metres)",
        "cam_centers": c.round(6).tolist(),
        "cam_path_length": float(steps.sum()),
        "cam_net_displacement": float(np.linalg.norm(c[-1] - c[0])),
        "cam_path_over_Zbg": float(steps.sum() / np.median(Zbg)),
        "Z_p_lo": float(np.median(Zlo)), "Z_p_hi": float(np.median(Zhi)),
        "Z_percentiles": [lo, hi],
        "delta_parallax_px": float(dpar.sum()),
        "rho": (float(med_rot[steps_static].sum() / den) if den > 0 else None),
        "rho_steps_used": int(steps_static.sum()),
        "instances": inst,
    }
    return arrays, js
