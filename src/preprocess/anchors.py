"""Anchor discovery via the occupancy tensor (SKILL §1.2.5, §5).

Occupancy at full process resolution, seeded on a stride grid (dense per-pixel
seeding is prohibited, §10.10). Frames are swept in order; a stride-grid pixel
spawns an anchor iff unoccupied; each new anchor's trajectory then marks all
target pixels where its projection is valid (q=1) within a stride/2 radius.
"""

from __future__ import annotations

import numpy as np

from .tracks import project, validity

MAX_ANCHORS = 20_000  # runaway guard (SKILL §5)


def discover_anchors(adapter, cfg: dict, log=print):
    """Returns (anchors [N,3], track_cache) — the cache holds each kept
    anchor's full trajectory from the discovery queries (same Eq 22 path
    build_tracks would use), so the track pass need not re-query."""
    stride = int(cfg["thresholds"]["anchor_stride"])
    r = stride // 2
    Hm, Wm = adapter.process_hw
    T = adapter.T
    K, _, _ = adapter.cameras()
    occ = np.zeros((T, Hm, Wm), bool)
    out: list[tuple[float, float, float]] = []
    rows: list[tuple] = []  # per kept anchor: (X_local, X0, V, C) slices

    th = cfg["thresholds"]
    ys = np.arange(r, Hm - r, stride)
    xs = np.arange(r, Wm - r, stride)
    for s in range(T):
        free = [(float(u), float(v)) for v in ys for u in xs if not occ[s, int(v), int(u)]]
        if not free:
            continue
        uv = np.asarray(free, np.float32)
        tr = adapter.track_batch(uv, s)
        P = project(K, tr["X_local"])
        q = validity(tr["V"], tr["C"], tr["X_local"], P, cfg, (Hm, Wm))
        # per-factor pass rates over all (anchor, frame) entries — the runaway
        # failure mode is "q=1 only at the spawn frame", and this shows which
        # factor is responsible (SKILL §9: validity gates vs marking radius).
        fV = float((tr["V"] >= th["tau_v"]).mean())
        fC = float((tr["C"] >= th["tau_c"]).mean())
        fZ = float((tr["X_local"][..., 2] > 0).mean())
        fq = float(q.mean())
        keep = np.flatnonzero(q[:, s])  # a seed invalid at its own frame is noise
        for i in keep:
            out.append((uv[i, 0], uv[i, 1], float(s)))
            rows.append((tr["X_local"][i], tr["X0"][i], tr["V"][i], tr["C"][i]))
            for t2 in np.flatnonzero(q[i]):
                u2, v2 = int(round(P[i, t2, 0])), int(round(P[i, t2, 1]))
                occ[t2, max(0, v2 - r):v2 + r + 1, max(0, u2 - r):u2 + r + 1] = True
        log(f"anchors: frame {s}: +{len(keep)}/{len(free)} seeds (total {len(out)}) | "
            f"pass-rates V {fV:.0%} C {fC:.0%} z>0 {fZ:.0%} q {fq:.0%}")
        if len(out) >= MAX_ANCHORS:
            raise RuntimeError(
                f"anchor runaway: {len(out)} at frame {s} (SKILL §5 guard).\n"
                f"Cross-frame pass rates this batch: V>={th['tau_v']}: {fV:.0%}, "
                f"C>={th['tau_c']}: {fC:.0%}, cheirality: {fZ:.0%}, overall q: {fq:.0%}.\n"
                "A factor near 0% means that threshold disqualifies tracks away from "
                "their spawn frame, so occupancy never fills and every frame re-seeds "
                "(SKILL §9 row 6). Check tau_v/tau_c against the probe P6 ranges.")

    if not out:
        raise RuntimeError("no anchors discovered — check tau_v/tau_c against P6 ranges")
    cache = {"X_local": np.stack([r[0] for r in rows]),
             "X0": np.stack([r[1] for r in rows]),
             "V": np.stack([r[2] for r in rows]),
             "C": np.stack([r[3] for r in rows])}
    return np.asarray(out, np.float32), cache
