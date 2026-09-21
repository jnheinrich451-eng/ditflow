"""Local/shared 3D tracks + validity Q (SKILL §1.2.6-1.2.8, §5).

The shared-coordinate pass re-queries every anchor with t_cam=0 (Eq 22) —
X0 is never produced by transforming X_local through G; that transform exists
only as the Eq (24) consistency CHECK computed here for QC.

NOTE on q: the four-factor validity is reconstructed from the SKILL (§5, §7
A-CH) as visibility, confidence, cheirality, in-frame projection. If the
research_pipeline_v1_2 doc defines §1.2.6-1.2.8 differently, update here.
"""

from __future__ import annotations

import numpy as np


def project(K: np.ndarray, X: np.ndarray) -> np.ndarray:
    """K [T,3,3], X [N,T,3] -> P [N,T,2] process-resolution pixels."""
    x = np.einsum("tij,ntj->nti", K, X)
    return x[..., :2] / np.clip(x[..., 2:3], 1e-9, None)


def validity(V, C, X_local, P, cfg, process_hw) -> np.ndarray:
    """Four-factor q_{i,t}: visibility, confidence, cheirality, in-frame."""
    th = cfg["thresholds"]
    Hm, Wm = process_hw
    in_frame = ((P[..., 0] >= 0) & (P[..., 0] <= Wm - 1)
                & (P[..., 1] >= 0) & (P[..., 1] <= Hm - 1))
    return ((V >= th["tau_v"]) & (C >= th["tau_c"])
            & (X_local[..., 2] > 0) & in_frame)


def eq24_residual_px(K, G, X_local, X0, Q) -> np.ndarray:
    """Per-entry Eq (24) residual in frame-0 pixels: ||pi_0(X0) - pi_0(G_t X_local)||.
    NaN where Q=0. This is the CHECK, not the path (SKILL §5). Since Task 2
    this is VISUALIZATION-ONLY — the gated quantity is eq24_relative below
    (pixel error scales with f/Z, so it is not clip-invariant; CLAUDE.md §4)."""
    Xw = np.einsum("tij,ntj->nti", G[:, :3, :3], X_local) + G[None, :, :3, 3]
    pa = project(np.repeat(K[:1], K.shape[0], 0), X0)
    pb = project(np.repeat(K[:1], K.shape[0], 0), Xw)
    res = np.linalg.norm(pa - pb, axis=-1).astype(np.float32)
    res[~Q] = np.nan
    return res


def eq24_relative(G, X_local, X0, Q) -> np.ndarray:
    """A-G2 primary statistic (Task 2): per-entry RELATIVE 3D residual
    ||X0 - G_t X_local|| / max(Z, Z_min), with Z = (X_local)_z. Clip-invariant:
    depth and focal length cancel out of the comparison across clips.

    Z_min = 5% of the clip's median valid depth — a degenerate-denominator
    guard that is part of the statistic's definition (not a tunable), so the
    quantity stays unit-free and needs no per-clip configuration. NaN at Q=0.
    """
    Xw = np.einsum("tij,ntj->nti", G[:, :3, :3], X_local) + G[None, :, :3, 3]
    d3 = np.linalg.norm(X0 - Xw, axis=-1).astype(np.float32)
    Z = X_local[..., 2]
    z_med = float(np.median(Z[Q])) if Q.any() else 1.0
    rel = d3 / np.maximum(Z, 0.05 * z_med)
    rel[~Q] = np.nan
    return rel.astype(np.float32)


def build_tracks(adapter, anchors: np.ndarray, cfg: dict, cached: dict | None = None) -> dict:
    """Dual query pass over all anchors [N,3] = (u, v, s_i). `cached` (from
    discover_anchors) holds trajectories already obtained through the very
    same adapter.track_batch Eq 22 path — passing it skips the duplicate
    queries; it is a dedup, not a different data path."""
    K, G, sim3 = adapter.cameras()
    N, T = len(anchors), adapter.T
    if cached is not None:
        Xl, X0 = cached["X_local"], cached["X0"]
        V, C = cached["V"], cached["C"]
        assert Xl.shape == (N, T, 3), "track cache does not match anchors"
    else:
        Xl = np.zeros((N, T, 3), np.float32)
        X0 = np.zeros((N, T, 3), np.float32)
        V = np.zeros((N, T), np.float32)
        C = np.zeros((N, T), np.float32)
        for s in np.unique(anchors[:, 2]).astype(int):
            idx = np.flatnonzero(anchors[:, 2].astype(int) == s)
            tr = adapter.track_batch(anchors[idx, :2], int(s))
            Xl[idx], X0[idx] = tr["X_local"], tr["X0"]
            V[idx], C[idx] = tr["V"], tr["C"]
    P = project(K, Xl).astype(np.float32)
    Q = validity(V, C, Xl, P, cfg, adapter.process_hw)
    return {"anchors": anchors.astype(np.float32), "K": K, "G": G,
            "X_local": Xl, "X0": X0, "P": P, "V": V, "C": C, "Q": Q,
            "eq24_px": eq24_residual_px(K, G, Xl, X0, Q),
            "eq24_rel": eq24_relative(G, Xl, X0, Q),
            "sim3_scale": sim3}
