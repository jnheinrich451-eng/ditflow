"""Segment-wise similarity drift correction (SKILL §4, appendix 12.2).

Model per frame t >= 1 over the static-candidate set:
    dX_{i,t} ~= a_t * X0_{i,t-1} + b_t
fitted by IRLS (Huber), eta-only weights. NO smoothing across window
boundaries (SKILL §10.2); optional median-of-3 within segments.
"""

from __future__ import annotations

import numpy as np

from .scoring import score_tracks


def _fit_frame(dX, Xprev, w, cfg):
    """IRLS similarity fit for one frame. Returns (a, b[3]) or None."""
    eps = float(cfg["eps"])
    dcfg = cfg["drift"]
    n = len(dX)
    if n < 4:
        return None
    a, b = 0.0, np.zeros(3)
    w_rob = np.ones(n)
    for _ in range(int(dcfg["irls_iters"])):
        ww = w * w_rob
        sw = ww.sum()
        if sw <= eps:
            return None
        # normal equations for theta = [a, b1, b2, b3]; rows: 3 eqs per point
        # sum_i ww_i (X_i.X_i) a + sum ww_i X_i . b = sum ww_i X_i.dX_i
        # sum_i ww_i X_i a + (sum ww_i) b = sum ww_i dX_i
        Sxx = float((ww * (Xprev * Xprev).sum(1)).sum())
        Sx = (ww[:, None] * Xprev).sum(0)
        SxD = float((ww * (Xprev * dX).sum(1)).sum())
        SD = (ww[:, None] * dX).sum(0)
        A = np.zeros((4, 4))
        A[0, 0] = Sxx
        A[0, 1:] = Sx
        A[1:, 0] = Sx
        A[1:, 1:] = np.eye(3) * sw
        rhs = np.concatenate([[SxD], SD])
        try:
            theta = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            return None
        a_new, b_new = float(theta[0]), theta[1:]
        res = np.linalg.norm(dX - a_new * Xprev - b_new, axis=1)
        med = np.median(res)
        sigma = 1.4826 * np.median(np.abs(res - med)) + eps
        c = float(dcfg["huber_c"]) * sigma
        rr = np.maximum(res, eps)
        w_rob = np.where(res <= c, 1.0, c / rr)
        if abs(a_new - a) < float(dcfg["irls_tol"]) and \
                np.linalg.norm(b_new - b) < float(dcfg["irls_tol"]):
            a, b = a_new, b_new
            break
        a, b = a_new, b_new
    return a, b


def fit_drift(X0, Q, w, static_mask, boundaries, cfg):
    """Per-frame (a_t, b_t) over static-candidate tracks; residuals for all.

    Returns a[F] (nan at t=0 and unfit frames), b[F,3], r[N,F,3] (nan where
    the adjacent pair is invalid)."""
    X0 = X0.astype(np.float64)
    N, F = Q.shape
    a = np.full(F, np.nan)
    b = np.full((F, 3), np.nan)
    r = np.full((N, F, 3), np.nan)
    dX_all = X0[:, 1:] - X0[:, :-1]
    for t in range(1, F):
        pair = Q[:, t - 1] & Q[:, t]
        fit_set = pair & static_mask
        got = _fit_frame(dX_all[fit_set, t - 1], X0[fit_set, t - 1],
                         w[fit_set, t], cfg) if fit_set.sum() >= 4 else None
        if got is None:
            continue
        a[t], b[t] = got
        r[pair, t] = dX_all[pair, t - 1] - a[t] * X0[pair, t - 1] - b[t]

    if cfg["drift"]["segment_smoothing"] == "median3":
        seg_edges = [1] + [int(x) for x in boundaries] + [F]
        a_s = a.copy()
        b_s = b.copy()
        for s0, s1 in zip(seg_edges[:-1], seg_edges[1:]):
            for t in range(s0 + 1, s1 - 1):
                a_s[t] = np.nanmedian(a[t - 1:t + 2])
                b_s[t] = np.nanmedian(b[t - 1:t + 2], axis=0)
        a, b = a_s, b_s
        for t in range(1, F):
            pair = Q[:, t - 1] & Q[:, t]
            if np.isfinite(a[t]):
                r[pair, t] = dX_all[pair, t - 1] - a[t] * X0[pair, t - 1] - b[t]
    return a, b, r


def gauge_curve(a, F):
    """Cumulative gauge g_t reconstructed from fitted a_t (g_0 = 1):
    X0 / g_t is the de-gauged coordinate. Frames with no fit inherit the
    previous gauge. This is the §4.4 de-gauging signal in curve form."""
    g = np.ones(F)
    for t in range(1, F):
        step = 1.0 + (a[t] if np.isfinite(a[t]) else 0.0)
        g[t] = g[t - 1] * step
    return g


def segment_scales(a, boundaries, F):
    """Cumulative per-segment relative scale from a_t (SKILL §4.4): the
    de-gauging signal. Segment k spans [edges[k], edges[k+1]); scale is the
    cumulative product of (1 + a_t) at its boundary-crossing entry frames,
    normalized to segment 0 = 1."""
    edges = [0] + [int(x) for x in boundaries] + [F]
    scales = [1.0]
    for b_t in boundaries:
        step = 1.0 + (a[int(b_t)] if np.isfinite(a[int(b_t)]) else 0.0)
        scales.append(scales[-1] * step)
    return [{"start": int(s), "end": int(e), "rel_scale": float(sc)}
            for s, e, sc in zip(edges[:-1], edges[1:], scales)]


def two_pass(X0, X_local, Q, w, boundaries, cfg, thresholds, fallback_fn,
             fallback_ctx):
    """Pipeline §1.3.1 two-pass scheme with the rho breakdown gate
    (appendix 12.2). fallback_fn(ctx, provisional_labels, cfg) -> index set.
    thresholds: dict with 'rho' and 'tau_z' (tau_z None -> ungated labels;
    rho gate then cannot fire meaningfully and pass 2 uses pass-1 statics)."""
    N, F = Q.shape
    all_static = np.ones(N, bool)
    a1, b1, r1 = fit_drift(X0, Q, w, all_static, boundaries, cfg)
    m1, y1, lab1 = score_tracks(r1, X_local, Q, w, boundaries, cfg,
                                thresholds.get("tau_z"))
    rho = float(y1[lab1].mean()) if lab1.any() else float("nan")

    # Breakdown detection, two arms (harness-measured): the rho>gate arm
    # catches PARTIAL contamination, but full inversion (fit locks onto a
    # frame-dominant mover, labels flip) yields a SMALL rho — its signature
    # is pipeline G3: fitted b_t carrying object velocity instead of
    # residual drift. Either arm fires the fallback.
    bmag1 = float(np.nanmedian(np.linalg.norm(b1, axis=1)))
    used_fallback = False
    rho_gate = thresholds.get("rho")
    tau_b = thresholds.get("tau_b")
    breakdown = ((rho_gate is not None and np.isfinite(rho)
                  and rho > float(rho_gate))
                 or (tau_b is not None and np.isfinite(bmag1)
                     and bmag1 > float(tau_b)))
    if breakdown:
        # ctx enriched with the data any policy may need — bundle arrays are
        # fit-free; m (pass-1 scores) included for the frozen interface but
        # the original policy deliberately ignores it (see static_fallback).
        static = np.zeros(N, bool)
        static[list(fallback_fn({**fallback_ctx, "m": m1, "X0": X0,
                                 "X_local": X_local, "Q": Q}, y1, cfg))] = True
        used_fallback = True
    else:
        static = ~y1

    a2, b2, r2 = fit_drift(X0, Q, w, static, boundaries, cfg)
    m2, y2, lab2 = score_tracks(r2, X_local, Q, w, boundaries, cfg,
                                thresholds.get("tau_z"), null_mask=static)
    return {"a": a2, "b": b2, "r": r2, "m": m2, "y_dyn": y2, "labeled": lab2,
            "rho_pass1": rho, "b_med_pass1": bmag1,
            "b_med_pass2": float(np.nanmedian(np.linalg.norm(b2, axis=1))),
            "used_fallback": used_fallback, "static_set": static,
            "segment_scales": segment_scales(a2, boundaries, F),
            "pass1": {"a": a1, "b": b1, "m": m1, "y_dyn": y1}}
