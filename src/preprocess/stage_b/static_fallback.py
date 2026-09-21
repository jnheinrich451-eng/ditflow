"""Pass-2 static-set fallback (SKILL §4.3) — the ORIGINAL policy behind a
frozen interface. Alternative policies (e.g. follow-cam priors) are future
work implementing the same signature; nothing else may change (§10.7).

Reading of "lowest-motion quartile" (harness-measured, 2026-08-07): the
pipeline invokes this fallback precisely when Pass-1 is DISTRUSTED, so the
motion prior must be fit-free. Using Pass-1's fitted scores is self-defeating
under full inversion (the movers score lowest — measured recall 0.0 on the
frame-dominant harness case). The prior is therefore raw, uncorrected
shared-frame displacement, depth-normalized: median_t ||dX0_{i,t}|| / Z.
"""

from __future__ import annotations

import numpy as np

BORDER_FRAC = 0.08   # image-border margin as a fraction of process size
QUARTILE = 0.25      # lowest-raw-motion quartile


def select(ctx: dict, provisional_labels: np.ndarray, cfg: dict):
    """Original policy: image-border tracks + lowest RAW-motion quartile.

    ctx (enriched by two_pass): anchors [N,3], X0 [N,F,3], X_local [N,F,3],
    Q [N,F], process_hw (H,W); also m (pass-1 scores, unused by this policy
    but part of the frozen interface for future ones).
    Returns an iterable of track indices forming the Pass-2 static set."""
    anchors = ctx["anchors"]
    X0, X_local, Q = ctx["X0"], ctx["X_local"], ctx["Q"]
    H, W = ctx["process_hw"]

    du = np.minimum(anchors[:, 0], W - 1 - anchors[:, 0])
    dv = np.minimum(anchors[:, 1], H - 1 - anchors[:, 1])
    border = (du < BORDER_FRAC * W) | (dv < BORDER_FRAC * H)

    # fit-free raw-motion prior
    N, F = Q.shape
    pair = Q[:, :-1] & Q[:, 1:]
    d = np.linalg.norm(np.diff(X0.astype(np.float64), axis=1), axis=-1)
    Z = X_local[..., 2].astype(np.float64)
    z_min = float(cfg["scoring"]["z_min_frac"]) * float(np.median(Z[Q]))
    u_raw = np.where(pair, d / np.maximum(Z[:, 1:], z_min), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN tracks
        motion = np.nanmedian(u_raw, axis=1)
    finite = np.isfinite(motion)
    q = np.nanquantile(motion[finite], QUARTILE) if finite.any() else np.inf
    low_motion = finite & (motion <= q)
    return list(np.flatnonzero(border | low_motion))
