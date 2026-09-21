"""Follow-cam branch (SKILL §12 addendum, human-approved 2026-08-09).

Measured basis (diag_absorption + diag_camera_channel on the calibration
clips, 2026-08-09):
- A followed object's world motion is ABSENT from X0 (car: 0.09x the camera
  step; below the static noise floor at 0.31x): the cross-frame head
  predicts "static" for content with no image motion.
- Per-point kinematic reconstruction G·X_local is structurally capped at
  SNR ~2 by the lift floor and measured 0.016 vs a 0.045 bar, under both
  the bundle G and a stream-only decontaminated refit. Dead end.
- The object IS cleanly detectable as the camera-frame-quiet, spatially
  compact population (car + bleed halo; horse + shore ring).
- Object-vs-halo membership is NOT geometrically separable (tracker motion
  bleed writes the subject's motion onto neighboring pixels) — membership
  is finalized by SAM2 in Stage C from the seed emitted here.
- World motion is composed at the RIGID level downstream: camera trajectory
  (source decided in the retargeting SKILL: bundle G vs the A.5 ViPE
  record) ∘ the relative camera-frame rigid motion measured here.

This module only MEASURES and packages: the firing decision against the
X0-clustering path (starvation predicate) belongs to the runner. The
per-frame rigid fit on the blob is provisional — it includes the halo by
construction and is recomputed downstream on SAM2-refined membership.

Pure NumPy, CPU-only (SKILL §10.8).
"""

from __future__ import annotations

import numpy as np

from .scoring import score_frames_mask


def _pair_valid(Q, boundaries, exclusion):
    N, F = Q.shape
    tmask = score_frames_mask(F, boundaries, exclusion)
    v = np.zeros((N, F), bool)
    v[:, 1:] = Q[:, :-1] & Q[:, 1:]
    return v & tmask[None, :]


def _per_track_median(mag, valid, min_frames=4):
    out = np.full(mag.shape[0], np.nan)
    for i in range(mag.shape[0]):
        m = valid[i] & np.isfinite(mag[i])
        if m.sum() >= min_frames:
            out[i] = float(np.median(mag[i][m]))
    return out


def _kabsch(A, B):
    """Rigid fit B ~= R (A - muA) + muB over paired [n,3] sets (f64)."""
    muA, muB = A.mean(0), B.mean(0)
    H = (A - muA).T @ (B - muB)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, muA, muB


def detect(X_local, X0, G, Q, P, anchors, boundaries, cfg, seed=0):
    """Detector + measurements for one clip. Returns a record dict.

    Criteria (provisional values in cfg['follow_cam']; final gates frozen by
    the human at calibration):
      quiet     cam-frame speed < quiet_frac x clip median
      compact   within compact_frac x Zmed of the quiet median position
      fired     n_blob >= min_quiet  AND  camera step / world floor >=
                travel_floor_mult (a static camera cannot fire the branch)
    """
    fc = cfg["follow_cam"]
    excl = int(cfg["scoring"]["boundary_score_exclusion"])
    Xl = X_local.astype(np.float64)
    N, F = Q.shape
    valid = _pair_valid(Q, boundaries, excl)

    dXl = np.full((N, F, 3), np.nan)
    dXl[:, 1:] = Xl[:, 1:] - Xl[:, :-1]
    cs = _per_track_median(np.linalg.norm(dXl, axis=-1), valid)
    med_cs = float(np.nanmedian(cs))
    quiet = np.nan_to_num(cs, nan=np.inf) < float(fc["quiet_frac"]) * med_cs
    stream = np.isfinite(cs) & ~quiet

    dX0 = np.full((N, F, 3), np.nan)
    X0d = X0.astype(np.float64)
    dX0[:, 1:] = X0d[:, 1:] - X0d[:, :-1]
    ws = _per_track_median(np.linalg.norm(dX0, axis=-1), valid)
    floor = float(np.nanmedian(ws[stream])) if stream.any() else float("nan")
    c = G[:, :3, 3].astype(np.float64)
    step_med = float(np.median(np.linalg.norm(np.diff(c, axis=0), axis=1)))

    Zmed = float(np.nanmedian(np.where(Q, Xl[..., 2], np.nan)))
    pos = np.full((N, 3), np.nan)
    for i in np.flatnonzero(quiet):
        m = Q[i]
        if m.any():
            pos[i] = np.median(Xl[i, m], axis=0)
    if quiet.any():
        center = np.nanmedian(pos[quiet], axis=0)
    else:
        center = np.full(3, np.nan)
    dist = np.linalg.norm(pos - center, axis=1)
    blob = quiet & (np.nan_to_num(dist, nan=np.inf)
                    < float(fc["compact_frac"]) * Zmed)

    step_over_floor = (step_med / floor
                       if np.isfinite(floor) and floor > 0 else float("nan"))
    fired = bool(int(blob.sum()) >= int(fc["min_quiet"])
                 and np.isfinite(step_over_floor)
                 and step_over_floor >= float(fc["travel_floor_mult"]))

    rec = {"fired": fired, "n_quiet": int(quiet.sum()),
           "n_blob": int(blob.sum()), "med_cs": round(med_cs, 5),
           "quiet_cs": round(float(np.nanmedian(cs[quiet])), 5)
           if quiet.any() else None,
           "world_floor": round(floor, 5) if np.isfinite(floor) else None,
           "cam_step_med": round(step_med, 5),
           "step_over_floor": round(step_over_floor, 2)
           if np.isfinite(step_over_floor) else None,
           "compact_med": round(float(np.nanmedian(dist[blob]) / Zmed), 3)
           if blob.any() else None,
           "Zmed": round(Zmed, 3),
           "quiet_mask": quiet, "blob_idx": np.flatnonzero(blob),
           "v0_med": center}
    # blob CORE (centroid-proximal core_frac): the subject-pure region —
    # arbitration v3 (2026-08-21, corpus-measured) gates winner overlap
    # against THIS, not the halo-inclusive blob: 21-clip evidence separated
    # legit laterals (core ovl 0.103-0.187) from halo-defeated winners
    # (0.000-0.044) with an empty zone between.
    bidx = rec["blob_idx"]
    if len(bidx):
        order = np.argsort(np.nan_to_num(dist[bidx], nan=np.inf))
        n_core = max(int(len(bidx) * float(fc.get("core_frac", 0.35))), 4)
        rec["blob_core_idx"] = bidx[order[:n_core]]
    else:
        rec["blob_core_idx"] = bidx
    if not fired:
        return rec

    # SAM2 seed (Stage C consumes; SAM2 itself is out of scope here §10.6):
    # frame of maximal blob validity; bbox + subsampled points over the blob
    # INCLUDING the halo — the semantic cut is exactly what SAM2 is for.
    bcount = Q[blob].sum(axis=0)
    seed_frame = int(np.argmax(bcount))
    mem = np.flatnonzero(blob & Q[:, seed_frame])
    uv = (P[mem, seed_frame].astype(np.float64) if P is not None
          else anchors[mem, :2].astype(np.float64))
    qlo, qhi = (float(x) for x in fc.get("seed_bbox_q", [0.05, 0.95]))
    bbox = [float(np.quantile(uv[:, 0], qlo)), float(np.quantile(uv[:, 1], qlo)),
            float(np.quantile(uv[:, 0], qhi)), float(np.quantile(uv[:, 1], qhi))]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(mem), min(int(fc["seed_points_cap"]), len(mem)),
                      replace=False)
    rec.update({"seed_frame": seed_frame, "seed_bbox": bbox,
                "seed_points": uv[pick].astype(np.float32)})

    # provisional relative rigid motion of the blob, CAMERA frame, centroid-
    # anchored per CLAUDE.md rule 9: tau_rel[t] = centroid displacement vs
    # the seed frame; world-origin tau is never stored.
    R_rel = np.full((F, 3, 3), np.nan)
    tau_rel = np.full((F, 3), np.nan)
    rel_valid = np.zeros(F, bool)
    bidx = rec["blob_idx"]
    for t in range(F):
        common = bidx[Q[bidx, t] & Q[bidx, seed_frame]]
        if len(common) < int(fc["min_common"]):
            continue
        R, muA, muB = _kabsch(Xl[common, seed_frame], Xl[common, t])
        R_rel[t], tau_rel[t] = R, muB - muA
        rel_valid[t] = True
    rec.update({"ref_frame": seed_frame, "R_rel": R_rel.astype(np.float32),
                "tau_rel": tau_rel.astype(np.float32), "rel_valid": rel_valid,
                "c_rel": rec["v0_med"].astype(np.float32)})
    return rec
