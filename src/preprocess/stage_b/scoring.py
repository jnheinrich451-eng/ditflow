"""Reliability weights + dynamic scoring (SKILL §5, appendix 12.1/12.3).

Pure NumPy. Weighting is eta-only per the dead-heads contract: V collapsed,
C saturated on this checkpoint (contract v1 §5.3-5.4) — the retired
sqrt(v*c) formula must not appear anywhere (SKILL §10.4).
"""

from __future__ import annotations

import numpy as np


def eta_weights(X0, X_local, G, Q, eps=1e-12) -> np.ndarray:
    """eta_{i,t} = exp(-e_G2 / sigma_eta), appendix 12.1.

    e_G2 = per-point local/shared consistency residual (3D, bundle units);
    sigma_eta = 1.4826 * MAD over q=1 entries, per clip, at load time."""
    Xw = np.einsum("tij,ntj->nti", G[:, :3, :3].astype(np.float64),
                   X_local.astype(np.float64)) + G[None, :, :3, 3].astype(np.float64)
    e = np.linalg.norm(X0.astype(np.float64) - Xw, axis=-1)
    med = np.median(e[Q])
    sigma = 1.4826 * np.median(np.abs(e[Q] - med)) + eps
    return np.exp(-e / sigma)


def pair_weights(eta: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Adjacent-pair weight w_{i,t} = q_{t-1} q_t sqrt(eta_{t-1} eta_t)
    (appendix 12.1); w[:, 0] = 0 by construction."""
    w = np.zeros_like(eta)
    w[:, 1:] = (Q[:, :-1] & Q[:, 1:]) * np.sqrt(eta[:, :-1] * eta[:, 1:])
    return w


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    """Weighted median: smallest x with cumulative weight >= half the total."""
    x, w = np.asarray(x, np.float64), np.asarray(w, np.float64)
    order = np.argsort(x)
    cw = np.cumsum(w[order])
    if cw[-1] <= 0:
        return float("nan")
    return float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])


def score_frames_mask(F: int, boundaries, exclusion: int) -> np.ndarray:
    """Frames eligible for scoring: t >= 1 and dist(t, boundaries) > exclusion."""
    tmask = np.ones(F, bool)
    tmask[0] = False
    for b in boundaries:
        tmask[max(0, b - exclusion):min(F, b + exclusion + 1)] = False
    return tmask


def score_tracks(r, X_local, Q, w, boundaries, cfg, tau_z, null_mask=None):
    """Appendix 12.3: u -> robust z -> per-track m_i (Q75) -> y_dyn.

    r: [N,F,3] corrected residuals (nan at t=0 / invalid pairs).
    null_mask [N] (harness-measured revision): the tracks whose entries
    define the NULL statistics (mu, MAD). z is a score against the static
    null distribution — with a frame-dominant mover, all-entries statistics
    ARE the mover's motion (measured: statics z=-3.9, movers z=0.65, zero
    labels). Pass 1 uses all tracks (bootstrap); Pass 2 the static set.
    Returns (m [N], y_dyn [N], labeled [N]); tau_z None -> y_dyn all False."""
    scfg = cfg["scoring"]
    eps = float(cfg["eps"])
    N, F = Q.shape
    pair_ok = np.zeros((N, F), bool)
    pair_ok[:, 1:] = Q[:, :-1] & Q[:, 1:]
    tmask = score_frames_mask(F, boundaries, int(scfg["boundary_score_exclusion"]))
    sel = pair_ok & tmask[None, :]
    if null_mask is None:
        null_mask = np.ones(N, bool)
    nsel = sel & null_mask[:, None]

    Z = X_local[..., 2].astype(np.float64)
    z_min = float(scfg["z_min_frac"]) * float(np.median(Z[Q]))
    u = np.linalg.norm(r, axis=-1) / np.maximum(Z, z_min)

    # Band structure for the null statistics. TIME bands (per window segment,
    # measured revision 2026-08-08): X0 noise grows with temporal distance and
    # steps at window boundaries, so one global noise floor mislabels quiet
    # late-window background as dynamic — the same unfairness the pipeline's
    # depth-band fallback addresses along the depth axis. Bands compose.
    band_id = np.zeros((N, F), int)
    n_bands = 1
    if scfg.get("time_bands", "none") == "segment":
        seg_of_t = np.zeros(F, int)
        for bd in boundaries:
            seg_of_t[int(bd):] += 1
        band_id = band_id * (seg_of_t.max() + 1) + seg_of_t[None, :]
        n_bands *= seg_of_t.max() + 1
    d_bands = int(scfg["depth_bands"])
    if d_bands > 1:
        lz = np.log(np.maximum(Z, z_min))
        edges = np.quantile(lz[nsel], np.linspace(0, 1, d_bands + 1))
        edges[0] -= 1e-9
        edges[-1] += 1e-9
        db = np.clip(np.digitize(lz, edges) - 1, 0, d_bands - 1)
        band_id = band_id * d_bands + db
        n_bands *= d_bands

    z = np.full((N, F), np.nan)
    # global stats as the fallback for starved bands
    mu_g = weighted_median(u[nsel], w[nsel])
    mad_g = weighted_median(np.abs(u[nsel] - mu_g), w[nsel])
    for bid in np.unique(band_id[sel]):
        band = band_id == bid
        nb, sb = nsel & band, sel & band
        if nb.sum() >= 30:
            mu = weighted_median(u[nb], w[nb])
            mad = weighted_median(np.abs(u[nb] - mu), w[nb])
        else:
            mu, mad = mu_g, mad_g
        z[sb] = (u[sb] - mu) / (1.4826 * mad + eps)

    counts = sel.sum(axis=1)
    labeled = counts >= int(scfg["f_score_min"])
    m = np.full(N, np.nan)
    # Track quantile: the appendix annotates m_i as UNWEIGHTED Q75, but the
    # synthetic harness measured a failure mode: one outlier ENTRY corrupts
    # two adjacent pairs, so heavy-tail tracks get >25% corrupted frames and
    # the unweighted Q75 lands on corruption (precision 0.76 at tau_z=2.5).
    # eta already scores those entries ~0 — the weighted quantile uses that.
    # Config-selectable; deviation flagged for human sign-off.
    weighted = bool(scfg.get("track_quantile_weighted", True))
    for i in np.flatnonzero(labeled):
        zi = z[i, sel[i]]
        if weighted:
            wi = w[i, sel[i]]
            order = np.argsort(zi)
            cw = np.cumsum(wi[order])
            if cw[-1] <= 0:
                continue
            m[i] = float(zi[order][min(np.searchsorted(cw, 0.75 * cw[-1]),
                                       len(zi) - 1)])
        else:
            m[i] = float(np.nanquantile(zi, 0.75))
    y = np.zeros(N, bool)
    if tau_z is not None:
        y[labeled] = m[labeled] > float(tau_z)
    return m, y, labeled
