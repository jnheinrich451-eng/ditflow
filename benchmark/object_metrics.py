"""Object-motion metrics for one (reference, generated) pair, from two extract-v1 outputs.

Three numbers, from cheapest/most-literature-compatible to most specific:

  Box-IoU      SymphoMotion's object metric: per frame, IoU of the subject's bounding box in
               the generated video vs the reference; mean over frames. SymphoMotion seeds SAM2
               with the GT first-frame mask; here both boxes come from the extractor's own
               Stage C subject mask (`instances == 1`), because in motion TRANSFER the subject
               changes (horse rider -> walking man) and a reference mask would not fit it.
               Rewards matching position AND size, so it penalises a subject swap that changes
               shape even when the motion is right. Report it, don't lean on it.
  ObjMC        MotionCtrl's object metric: mean Euclidean distance (px) between the generated
               and reference subject trajectories in the image. Here the trajectory is the mask
               centroid per frame. Image-plane, so it MIXES camera and object motion: a method
               that copies the camera correctly gets part of this right for free.
  ObjErr3D     ours: the subject's world-frame root trajectory (camera-compensated, from
               derived.json instances[1].world_traj), as displacement from its own frame 0, the
               generated one scaled into the reference's units by the camera alignment scale,
               error per frame divided by the reference's median background depth -> unitless.
               This is the only one of the three that separates object motion from camera.

All three take `idx_ref`, `idx_gen`: the frame correspondence. Never match by position unless
the correspondence is the identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SUBJECT = 1


def load(d: Path):
    d = Path(d)
    raw = dict(np.load(d / "raw.npz", allow_pickle=False))
    der = json.loads((d / "derived.json").read_text(encoding="utf-8"))
    return raw, der


def boxes_from_masks(masks: np.ndarray) -> np.ndarray:
    """(T,H,W) bool -> (T,4) x0,y0,x1,y1 (inclusive pixels); NaN row where the mask is empty."""
    out = np.full((len(masks), 4), np.nan)
    for t, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            out[t] = [xs.min(), ys.min(), xs.max(), ys.max()]
    return out


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row IoU of (T,4) boxes; NaN where either box is missing."""
    ix0, iy0 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    ix1, iy1 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(ix1 - ix0 + 1, 0, None) * np.clip(iy1 - iy0 + 1, 0, None)
    area = lambda x: (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)
    with np.errstate(invalid="ignore"):
        return inter / (area(a) + area(b) - inter)


def mask_centroids(masks: np.ndarray) -> np.ndarray:
    """(T,H,W) bool -> (T,2) x,y centroid; NaN where empty."""
    out = np.full((len(masks), 2), np.nan)
    for t, m in enumerate(masks):
        ys, xs = np.nonzero(m)
        if len(xs):
            out[t] = [xs.mean(), ys.mean()]
    return out


def world_root(der: dict) -> np.ndarray:
    """Subject root trajectory in the camera-0 world frame, (T,3); NaN where unsupported."""
    tr = der["instances"][str(SUBJECT)]["world_traj"]
    return np.array([p if p is not None else [np.nan] * 3 for p in tr], np.float64)


def object_errors(ref_dir, gen_dir, idx_ref, idx_gen, cam_scale: float) -> dict:
    """cam_scale: s from camera_errors(E_gen, E_ref, align="scale_only")['scale'] -- the factor
    that maps generated world units onto reference world units."""
    raw_r, der_r = load(ref_dir)
    raw_g, der_g = load(gen_dir)
    idx_ref, idx_gen = np.asarray(idx_ref, int), np.asarray(idx_gen, int)
    assert len(idx_ref) == len(idx_gen), "frame correspondence must be one-to-one"
    m_r = raw_r["instances"][idx_ref] == SUBJECT
    m_g = raw_g["instances"][idx_gen] == SUBJECT
    assert m_r.shape[1:] == m_g.shape[1:], f"mask shapes differ {m_r.shape} {m_g.shape}: canonicalise first"

    iou = box_iou(boxes_from_masks(m_g), boxes_from_masks(m_r))
    c_r, c_g = mask_centroids(m_r), mask_centroids(m_g)
    d2 = np.linalg.norm(c_g - c_r, axis=1)

    X_r, X_g = world_root(der_r)[idx_ref], world_root(der_g)[idx_gen]
    D_r = X_r - X_r[0]                                   # motion only: own frame-0 offset removed
    D_g = (X_g - X_g[0]) * cam_scale
    with np.load(Path(ref_dir) / "derived.npz") as z:     # reference background depth = unit
        zbg = float(np.median(z["Z_bg_med"][idx_ref]))
    e3 = np.linalg.norm(D_g - D_r, axis=1) / zbg

    ok = lambda v: v[np.isfinite(v)]
    return {
        "n": int(len(idx_ref)),
        "box_iou": float(ok(iou).mean()) if ok(iou).size else float("nan"),
        "box_frames_missing": int((~np.isfinite(iou)).sum()),
        "objmc_px": float(ok(d2).mean()) if ok(d2).size else float("nan"),
        "obj_err3d": float(ok(e3).mean()) if ok(e3).size else float("nan"),
        "obj_err3d_final": float(e3[-1]) if np.isfinite(e3[-1]) else float("nan"),
        "ref_path_len_3d": float(np.nansum(np.linalg.norm(np.diff(X_r, axis=0), axis=1)) / zbg),
    }
