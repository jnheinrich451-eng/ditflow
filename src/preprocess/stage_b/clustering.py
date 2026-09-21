"""Rung-0 coherent-motion clustering (pipeline §1.3.3, appendix 12.4).

Threshold graph on pairwise rigidity + direction + proximity + overlap,
connected components. Rung-1 (affinity/spectral) is explicitly out of scope.
"""

from __future__ import annotations

import numpy as np

from .scoring import score_frames_mask


def _track_summaries(X0, r, Q, boundaries, cfg):
    """Per track over valid non-boundary frames (appendix 12.4):
    r_bar  — component-wise median residual motion (direction summary);
    X_bar  — median position;
    s_bar  — median residual SPEED, median_t ||r_t||. The speed is the
    moving/not-moving quantity: ||r_bar|| cancels for direction-changing
    motion (measured 2026-08-08: the turning car's median vector ~0 while
    its speed was large — the norm-of-median floor disqualified the car)."""
    N, F = Q.shape
    tmask = score_frames_mask(F, boundaries,
                              int(cfg["scoring"]["boundary_score_exclusion"]))
    seg_of_t = np.zeros(F, int)
    for bd in boundaries:
        seg_of_t[int(bd):] += 1
    n_seg = seg_of_t.max() + 1
    min_seg = int(cfg["clustering"].get("min_seg_frames", 4))
    r_bar = np.full((N, 3), np.nan)
    r_seg = np.full((N, n_seg, 3), np.nan)
    X_bar = np.full((N, 3), np.nan)
    s_bar = np.full(N, np.nan)
    for i in range(N):
        tv = Q[i] & tmask
        if tv.any():
            X_bar[i] = np.median(X0[i, tv], axis=0)
        rv = tv & np.isfinite(r[i, :, 0])
        if rv.any():
            r_bar[i] = np.median(r[i, rv], axis=0)
            s_bar[i] = np.median(np.linalg.norm(r[i, rv], axis=-1))
            for s in range(n_seg):
                sv = rv & (seg_of_t == s)
                if sv.sum() >= min_seg:
                    r_seg[i, s] = np.median(r[i, sv], axis=0)
    return r_bar, X_bar, s_bar, r_seg


def _stratified_subsample(idx, anchors, X_bar, cap, grid, rng):
    """Image-region x depth strata (pipeline §1.3.3 compute cap)."""
    if len(idx) <= cap:
        return idx
    gx, gy, gz = grid
    u = anchors[idx, 0]
    v = anchors[idx, 1]
    z = X_bar[idx, 2]
    bx = np.clip(np.digitize(u, np.quantile(u, np.linspace(0, 1, gx + 1))[1:-1]), 0, gx - 1)
    by = np.clip(np.digitize(v, np.quantile(v, np.linspace(0, 1, gy + 1))[1:-1]), 0, gy - 1)
    bz = np.clip(np.digitize(z, np.quantile(z, np.linspace(0, 1, gz + 1))[1:-1]), 0, gz - 1)
    strat = bx * gy * gz + by * gz + bz
    keep = []
    for s in np.unique(strat):
        members = idx[strat == s]
        quota = max(1, int(round(cap * len(members) / len(idx))))
        keep.extend(rng.choice(members, size=min(quota, len(members)),
                               replace=False))
    return np.asarray(sorted(keep[:cap]), int)


def _edge_ok(e, cosd, prox, overlap, th):
    return (e < th["tau_r"] and cosd > th["tau_d"]
            and prox < th["tau_x"] and overlap >= th["f_min"])


def cluster_tracks(X0, r, Q, anchors, y_dyn, boundaries, cfg, th, seed=0,
                   a=None, eta=None):
    """Returns cluster_id[N] int (-1 = static/unassigned) and diagnostics.

    a: fitted per-frame drift (drift.py). When given and clustering.de_gauge
    is true (default), X0 is DE-GAUGED (divided by the cumulative gauge g_t)
    before any pairwise geometry — measured 2026-08-09: static-pair e_ij grew
    +63% from near to far pairs purely from the gauge, poisoning rigidity.

    eta [N,F]: reliability weights. ETA REFEREE (measured 2026-08-09b):
    OpenD4RT's far-field errors are CORRELATED — neighboring bad tracks form
    fake rigid, coherent clusters that no motion statistic can reject (the
    sky blob that beat the car). But those tracks are exactly the ones whose
    local/shared geometry disagrees, which eta grades: candidates below
    eta_gate_frac x the labeled-population median are excluded from the
    graph and from post-hoc assignment."""
    eps = float(cfg["eps"])
    N, F = Q.shape
    if a is not None and cfg["clustering"].get("de_gauge", True):
        from .drift import gauge_curve
        X0 = X0 / gauge_curve(a, F)[None, :, None]
    eta_track = None
    if eta is not None:
        with np.errstate(invalid="ignore"):
            eta_track = np.where(Q, eta, np.nan)
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                eta_track = np.nanmean(eta_track, axis=1)
    cluster_id = np.full(N, -1, int)
    dyn_idx = np.flatnonzero(y_dyn)
    n_dyn_raw = len(dyn_idx)
    eta_gate = None
    if eta_track is not None:
        frac = float(cfg["clustering"].get("eta_gate_frac", 0.8))
        eta_gate = frac * float(np.nanmedian(eta_track))
        dyn_idx = dyn_idx[np.nan_to_num(eta_track[dyn_idx]) >= eta_gate]
    n_eta_gated = n_dyn_raw - len(dyn_idx)
    if len(dyn_idx) < 2:
        return cluster_id, {"n_dyn": int(n_dyn_raw), "n_sampled": 0,
                            "n_clusters": 0, "n_eta_gated": int(n_eta_gated)}
    r_bar, X_bar, s_bar, r_seg = _track_summaries(X0, r, Q, boundaries, cfg)
    rng = np.random.default_rng(seed)
    sample = _stratified_subsample(dyn_idx, anchors, X_bar,
                                   int(cfg["clustering"]["subsample_cap"]),
                                   cfg["clustering"]["strata_grid"], rng)
    S = len(sample)
    Xs = X0[sample].astype(np.float64)          # [S,F,3]
    Qs = Q[sample]

    # pairwise inter-track distances per frame, then e_ij (median |d_t - d_{t-1}|)
    valid_pair_t = Qs[:, None, :] & Qs[None, :, :]          # [S,S,F]
    adj_valid = valid_pair_t[:, :, 1:] & valid_pair_t[:, :, :-1]
    d = np.linalg.norm(Xs[:, None, :, :] - Xs[None, :, :, :], axis=-1)  # [S,S,F]
    dd = np.abs(d[:, :, 1:] - d[:, :, :-1])
    dd = np.where(adj_valid, dd, np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN pairs -> nan
        e = np.nanmedian(dd, axis=2)
    overlap = valid_pair_t.sum(axis=2)

    rb, xb = r_bar[sample], X_bar[sample]
    nrm = np.linalg.norm(rb, axis=1)
    # Direction similarity (measured revision 2026-08-09): whole-track r_bar
    # cancels for turning objects, and a SMALL vector's direction is noise —
    # car-car cosines went random and the car lost its edges while lateral
    # movers kept theirs. Default "segment" mode compares per-window-segment
    # mean motions (direction is locally stable even through a turn) and
    # takes the median cosine over segments both tracks cover.
    if cfg["clustering"].get("direction_mode", "segment") == "segment":
        rs = r_seg[sample]                       # [S, n_seg, 3]
        seg_nrm = np.linalg.norm(rs, axis=-1)    # [S, n_seg]
        dots = np.einsum("ikc,jkc->ijk", np.nan_to_num(rs), np.nan_to_num(rs))
        cos_seg = dots / (seg_nrm[:, None, :] * seg_nrm[None, :, :] + eps)
        cos_seg[np.isnan(seg_nrm)[:, None, :] | np.isnan(seg_nrm)[None, :, :]] = np.nan
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cosd = np.nanmedian(cos_seg, axis=2)
        cosd = np.nan_to_num(cosd, nan=-1.0)     # no common segment -> no edge
    else:                                        # "r_bar": appendix-literal
        cosd = (rb @ rb.T) / (nrm[:, None] * nrm[None, :] + eps)
    prox = np.linalg.norm(xb[:, None] - xb[None, :], axis=-1)

    # Direction floor (measured revisions 2026-08-08, twice): tracks at the
    # static noise floor have random r_bar direction — the cosine test
    # degenerates to a coin flip and percolates the graph. The floor is on
    # median residual SPEED (s_bar), NOT ||r_bar||: a turning object's median
    # vector cancels toward zero while its speed stays large (the drifting
    # car lost every edge under the ||r_bar|| version — O_src fell to 5
    # noise tracks on distant trees).
    stat_idx = np.flatnonzero(~y_dyn)
    floor = (float(np.nanmedian(s_bar[stat_idx])) if len(stat_idx) else 0.0)
    mult = float(cfg["clustering"].get("dir_floor_mult", 2.0))
    moving = np.nan_to_num(s_bar[sample]) > mult * floor
    dir_ok = (cosd > float(th["tau_d"])) & moving[:, None] & moving[None, :]

    crit = {"rigidity": np.nan_to_num(e, nan=np.inf) < float(th["tau_r"]),
            "direction": dir_ok,
            "proximity": prox < float(th["tau_x"]),
            "overlap": overlap >= int(th["f_min"])}
    edges = crit["rigidity"] & crit["direction"] & crit["proximity"] & crit["overlap"]
    np.fill_diagonal(edges, False)
    # edge autopsy: per-criterion pass fractions over sampled pairs — the
    # standing diagnostic for "why did the subject not cluster"
    iu = np.triu_indices(S, 1)
    edge_diag = {k: round(float(v[iu].mean()), 3) for k, v in crit.items()}
    edge_diag.update({"joint": round(float(edges[iu].mean()), 3),
                      "moving_frac": round(float(moving.mean()), 3),
                      "speed_floor": round(floor, 5),
                      "tau_used": {k: (round(float(th[k]), 4)
                                       if isinstance(th[k], (int, float)) else th[k])
                                   for k in ("tau_r", "tau_d", "tau_x", "f_min")}})

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n_comp, labels = connected_components(csr_matrix(edges), directed=False)
    # components of size 1 are noise -> unassigned
    sizes = np.bincount(labels)
    remap = {}
    for c in range(n_comp):
        if sizes[c] >= 2:
            remap[c] = len(remap)
    for k, i in enumerate(sample):
        if labels[k] in remap:
            cluster_id[i] = remap[labels[k]]

    # post-hoc assignment of non-sampled dynamic tracks (appendix 12.4);
    # the direction floor applies here too. Assignment source + vote counts
    # are recorded (2026-08-10 instrumentation, non-behavioral): a single
    # passing edge currently suffices to join — the suspected stray-ring
    # entry path on the articulated clip.
    posthoc = np.zeros(N, bool)
    posthoc_votes = np.zeros(N, np.int32)
    rest = [i for i in dyn_idx if i not in set(sample)]
    for i in rest:
        if np.nan_to_num(s_bar[i]) <= mult * floor:
            continue
        if eta_gate is not None and np.nan_to_num(eta_track[i]) < eta_gate:
            continue
        votes = {}
        for k, j in enumerate(sample):
            cid = cluster_id[j]
            if cid < 0:
                continue
            tv = Q[i] & Q[j]
            ov = int(tv.sum())
            adj = tv[1:] & tv[:-1] & Q[i, :-1] & Q[j, :-1]
            if ov < int(th["f_min"]) or not adj.any():
                continue
            di = np.linalg.norm(X0[i] - X0[j], axis=-1)
            e_ij = float(np.nanmedian(np.abs(np.where(adj, di[1:] - di[:-1], np.nan))))
            if cfg["clustering"].get("direction_mode", "segment") == "segment":
                c_all = np.sum(r_seg[i] * r_seg[j], -1) / (
                    np.linalg.norm(r_seg[i], axis=-1)
                    * np.linalg.norm(r_seg[j], axis=-1) + eps)
                c_all = c_all[np.isfinite(c_all)]
                cd = float(np.median(c_all)) if c_all.size else -1.0
            else:
                cd = float(np.dot(r_bar[i], r_bar[j])
                           / (np.linalg.norm(r_bar[i]) * np.linalg.norm(r_bar[j]) + eps))
            px = float(np.linalg.norm(X_bar[i] - X_bar[j]))
            if _edge_ok(e_ij, cd, px, ov, {k2: float(th[k2]) for k2 in
                                           ("tau_r", "tau_d", "tau_x")} | {"f_min": int(th["f_min"])}):
                votes[cid] = votes.get(cid, 0) + 1
        if votes:
            best = max(votes.values())
            tied = [c for c, v in votes.items() if v == best]
            if len(tied) == 1:
                cluster_id[i] = tied[0]
            else:  # tie -> nearest cluster centroid by X_bar
                cents = {c: X_bar[[j for j in sample if cluster_id[j] == c]].mean(0)
                         for c in tied}
                cluster_id[i] = min(tied, key=lambda c: np.linalg.norm(X_bar[i] - cents[c]))
            posthoc[i] = True
            posthoc_votes[i] = int(votes[cluster_id[i]])

    n_clusters = int(cluster_id.max() + 1) if cluster_id.max() >= 0 else 0
    edge_diag["n_eta_gated"] = int(n_eta_gated)
    edge_diag["n_posthoc"] = int(posthoc.sum())
    return cluster_id, {"n_dyn": int(n_dyn_raw), "n_sampled": int(S),
                        "n_clusters": n_clusters, "edge_diag": edge_diag,
                        "posthoc_mask": posthoc,
                        "posthoc_votes": posthoc_votes}
