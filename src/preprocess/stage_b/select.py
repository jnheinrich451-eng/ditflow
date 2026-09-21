"""Cluster selection Score(C_k) (pipeline §1.3.3, appendix 12.5)."""

from __future__ import annotations

import numpy as np


def score_clusters(cluster_id, m, Q, lambdas, n_min, eta=None):
    """Score(C_k) = l1*|C_k|/N_dyn + l2*TemporalCoverage + l3*MotionMagnitude
    (normalized per appendix 12.5). Returns (O_src cluster id or None,
    per-cluster diagnostics list).

    eta [N,F] (eta-referee, 2026-08-09b): when given, Score is multiplied by
    the cluster's normalized median track reliability, so a cluster of
    correlated-noise tracks (coherent but geometrically unreliable) cannot
    win selection over a genuinely tracked object."""
    F = Q.shape[1]
    ids = sorted(c for c in np.unique(cluster_id) if c >= 0)
    if not ids:
        return None, []
    n_dyn = int((cluster_id >= 0).sum())
    l1, l2, l3 = (float(v) for v in lambdas)
    eta_track = None
    if eta is not None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            eta_track = np.nanmean(np.where(Q, eta, np.nan), axis=1)
    rows = []
    mags = {}
    for c in ids:
        members = np.flatnonzero(cluster_id == c)
        cov = float(((Q[members].sum(axis=0)) > int(n_min)).mean())
        mag = float(np.nanmedian(m[members]))
        mags[c] = mag
        row = {"cluster": int(c), "size": int(len(members)),
               "coverage": cov, "motion_mag": mag}
        if eta_track is not None:
            row["eta_med"] = float(np.nanmedian(eta_track[members]))
        rows.append(row)
    mag_max = max(abs(v) for v in mags.values()) or 1.0
    eta_max = (max(r["eta_med"] for r in rows) or 1.0) if eta_track is not None else 1.0
    for row in rows:
        base = (l1 * row["size"] / max(n_dyn, 1)
                + l2 * row["coverage"]
                + l3 * row["motion_mag"] / mag_max)
        rel = (row["eta_med"] / eta_max) if eta_track is not None else 1.0
        row["score"] = base * rel
    best = max(rows, key=lambda r: r["score"])
    return int(best["cluster"]), rows
