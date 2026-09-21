"""Stage C.5 mask-guided densification — pure CPU geometry (SKILL v1.1 §2).
Seeding/dedup/occupancy here; D4RT queries live in the runner (GPU)."""
from __future__ import annotations

import numpy as np


def dilate_mask(mask, px):
    import cv2
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def mark(occ, uv, rad):
    """Mark occupancy grid [H,W] around points uv (float px)."""
    H, W = occ.shape
    for u, v in np.asarray(uv).reshape(-1, 2):
        x0, x1 = max(int(u - rad), 0), min(int(u + rad) + 1, W)
        y0, y1 = max(int(v - rad), 0), min(int(v + rad) + 1, H)
        occ[y0:y1, x0:x1] = True
    return occ


def candidates(mask_t, stride, occ):
    """Stride-grid pixels inside mask_t and not occupied -> [N,2] (u,v)."""
    H, W = mask_t.shape
    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    us, vs = us.ravel(), vs.ravel()
    keep = mask_t[vs, us] & ~occ[vs, us]
    return np.stack([us[keep], vs[keep]], 1).astype(np.float64)


def eta_of(X0, Xl, G, Q, eps=1e-12):
    Xw = np.einsum("tij,ntj->nti", G[:, :3, :3].astype(np.float64),
                   Xl.astype(np.float64)) + G[None, :, :3, 3].astype(np.float64)
    e = np.linalg.norm(X0.astype(np.float64) - Xw, axis=-1)
    med = np.median(e[Q]) if Q.any() else 0.0
    sigma = 1.4826 * np.median(np.abs(e[Q] - med)) + eps if Q.any() else eps
    return np.exp(-e / sigma), float(sigma), e


def inside_mask_frac(P, Q, M):
    """Per-track fraction of valid frames projecting inside the mask."""
    N, F = Q.shape
    H, W = M.shape[1:]
    out = np.full(N, np.nan)
    for i in range(N):
        ts = np.flatnonzero(Q[i])
        if not len(ts):
            continue
        u = np.clip(P[i, ts, 0].round().astype(int), 0, W - 1)
        v = np.clip(P[i, ts, 1].round().astype(int), 0, H - 1)
        out[i] = float(M[ts, v, u].mean())
    return out
