#!/usr/bin/env python
"""Thesis figures for ONE clip, read straight from the cached Module-1
artifacts (Stage A bundle, Stage B, Stage C mask, Stage C.5 dense tracks,
Stage B.5 scene up / subject centroid). Read-only: no model is loaded and
nothing is recomputed except viz-only geometry (projection, occlusion cull).

    python scripts/plot_thesis_tracks.py --clip-id 000000000035.1.003

Writes <drive_root>/viz/thesis/<clip_id>/:
  fig_camera_trajectory.{pdf,png}  camera centres + frustums (3D), top-down
                                   view, translation / orientation vs t
  fig_sparse_tracks.{pdf,png}      sparse global tracks: image trails at t_ref
                                   + bird's-eye world map of X0
  fig_dense_tracks.{pdf,png}       dense mask tracks: overlay + full-clip
                                   close-up coloured by t
  fig_filmstrip.{pdf,png}          mask + sparse + dense over 5 frames
  figures.md                       caption facts (counts, frames, provenance)

Frames: scene frame (right, forward, up) built from B.5 u_scene when present,
else camera-0 with up = -y (OpenCV). Units are D4RT model units (up to a
similarity), never metres.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.bundle import read_bundle          # noqa: E402
from src.preprocess.config import load_config          # noqa: E402
from src.preprocess.stage_c import finalize            # noqa: E402

# Reference dataviz palette (light / print). Categorical slots 1-3 validate
# all-pairs; background tracks use neutral ink so hue always means identity.
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SUBJ, DENSE, MASK = "#2a78d6", "#eb6834", "#1baf7a"
TIME_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
OCC_SLACK = 1.08   # viz-only z-buffer cull, same rule as inspect_bundle.py
WASH = 0.35        # blend video frames toward white so marks read in print
DPI = 300


# ---------------------------------------------------------------- loading
def load(cfg: dict, c: str) -> dict:
    root = Path(str(cfg["paths"]["drive_root"]))
    b = read_bundle(cfg, c)
    d = {k: b[k] for k in ("K", "G", "D", "anchors", "X_local", "X0", "P", "Q")}
    d["prov"], d["qc"] = b["provenance"], b["qc"]
    bdir = Path(cfg["paths"]["stage_b_cache"])
    with np.load(bdir / f"{c}.npz") as z:
        d["B"] = {k: z[k] for k in z.files}
    qb = json.loads((bdir / f"{c}_qc_b.json").read_text(encoding="utf-8"))
    d["channel"] = "branch" if (qb.get("follow_cam") or {}).get("branch_active") else "x0"
    cdir = Path(cfg["paths"]["stage_c_cache"]) / c
    assert (cdir / "qc_c.json").exists(), f"no Stage C record for {c}"
    d["M"], d["present"], d["t_c"] = finalize.read_mask(cdir)
    c5 = root / "cache/stage_c5" / f"{c}.npz"
    assert c5.exists(), f"no Stage C.5 dense tracks for {c}: {c5}"
    with np.load(c5) as z:
        d["C5"] = {k: z[k] for k in z.files}
    d["B5"] = {}
    b5 = root / "cache/stage_b5" / f"{c}.npz"
    if b5.exists():
        with np.load(b5) as z:
            d["B5"] = {k: z[k] for k in z.files}
    return d


def read_frames(path: str, idx) -> dict:
    import imageio.v2 as iio
    if not Path(path).exists():
        raise SystemExit(f"source clip not on this VM ({path}) - run bootstrap first")
    want, out = set(int(i) for i in idx), {}
    rd = iio.get_reader(path)
    for i, f in enumerate(rd):
        if i in want:
            out[i] = f[..., :3].copy()
        if i >= max(want):
            break
    rd.close()
    missing = want - set(out)
    assert not missing, f"video shorter than bundle: frames {sorted(missing)} missing"
    return out


# --------------------------------------------------------------- geometry
def scene_basis(u_scene):
    """Rows (right, forward, up) in camera-0 coords; coords_scene = B @ X."""
    up = np.asarray(u_scene, np.float64) if u_scene is not None else np.array([0., -1., 0.])
    up = up / np.linalg.norm(up)
    fwd = np.array([0., 0., 1.]) - up[2] * up
    assert np.linalg.norm(fwd) > 1e-6, "u_scene parallel to camera-0 optical axis"
    fwd /= np.linalg.norm(fwd)
    B = np.stack([np.cross(fwd, up), fwd, up])
    assert abs(np.linalg.det(B) - 1.0) < 1e-6, "scene basis not right-handed"
    return B


def visible(D, Xl, P, Q):
    """Q minus points sitting >8% behind the depth map at their own pixel
    (amodal tracks behind an occluder). Viz-only; Q itself is untouched."""
    F, Hm, Wm = D.shape
    u = np.clip(P[..., 0].round().astype(int), 0, Wm - 1)
    v = np.clip(P[..., 1].round().astype(int), 0, Hm - 1)
    d_at = D.astype(np.float32)[np.arange(F)[None, :], v, u]
    return Q & ~(Xl[..., 2] > d_at * OCC_SLACK)


def frustum(K, R, o, s, Hp):
    """Apex + 4 image-corner points at depth s, camera-0 coords."""
    uv = np.array([[0, 0], [Hp, 0], [Hp, Hp], [0, Hp]], np.float64)
    rays = np.linalg.solve(K, np.c_[uv, np.ones(4)].T).T      # z = 1
    return o, o + (rays * s) @ R.T


def track_segments(P, vis, idx, t0, t1, sx, sy):
    """Segments between consecutive visible frames in [t0, t1] -> (segs, t)."""
    segs, ts = [], []
    for i in idx:
        for t in range(t0, t1):
            if vis[i, t] and vis[i, t + 1]:
                segs.append([(P[i, t, 0] * sx, P[i, t, 1] * sy),
                             (P[i, t + 1, 0] * sx, P[i, t + 1, 1] * sy)])
                ts.append(t)
    return np.asarray(segs, np.float64).reshape(-1, 2, 2), np.asarray(ts)


# ------------------------------------------------------------------ style
def style(plt):
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 8.5, "axes.titlesize": 9,
        "axes.labelsize": 8.5, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2, "text.color": INK, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "legend.frameon": False,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42})


def washed(frame):
    return (frame.astype(np.float32) * (1 - WASH) + 255 * WASH).astype(np.uint8)


def image_ax(ax, frame, title, loc="left"):
    ax.imshow(washed(frame))
    ax.set_title(title, loc=loc)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def draw_mask(ax, M_t, sx, sy, fill=True):
    Hp, Wp = M_t.shape
    ext = (0, Wp * sx, Hp * sy, 0)
    if fill:
        from matplotlib.colors import ListedColormap
        ax.imshow(np.ma.masked_where(~M_t, M_t), cmap=ListedColormap([MASK]),
                  alpha=0.28, extent=ext, interpolation="nearest")
    if M_t.any():
        xs = (np.arange(Wp) + 0.5) * sx
        ys = (np.arange(Hp) + 0.5) * sy
        ax.contour(xs, ys, M_t.astype(float), levels=[0.5], colors=[MASK], linewidths=1.2)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=DPI)


# ---------------------------------------------------------------- figures
def fig_camera(d, B, Hp, every, out, plt):
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    G, K = d["G"].astype(np.float64), d["K"].astype(np.float64)
    F = len(G)
    assert np.linalg.norm(G[0, :3, 3]) < 1e-2, "o_0 != 0 - G convention broken"
    o = G[:, :3, 3] @ B.T
    fwd = G[:, :3, 2] @ B.T                          # optical axis, scene coords
    yaw = np.degrees(np.unwrap(np.arctan2(fwd[:, 0], fwd[:, 1])))
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 2], -1, 1)))
    xb = d["B5"].get("X_bar_track")
    xb = None if xb is None else xb.astype(np.float64) @ B.T
    okb = None if xb is None else np.isfinite(xb).all(1)
    core = o if xb is None else np.vstack([o, xb[okb]])
    s = 0.07 * max(float(np.ptp(core, axis=0).max()), 1e-6)
    keys = sorted(set(range(0, F, every)) | {F - 1})
    fr = {k: frustum(K[k], G[k, :3, :3], G[k, :3, 3], s, Hp) for k in keys}
    cmap = TIME_CMAP
    norm = Normalize(0, F - 1)

    fig = plt.figure(figsize=(7.4, 4.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.3, 1], hspace=0.75, wspace=0.28)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    lc = Line3DCollection(np.stack([o[:-1], o[1:]], 1), cmap=cmap, norm=norm, lw=2)
    lc.set_array(np.arange(F - 1))
    ax.add_collection(lc)
    allp = [core]
    for k, (apex, cr) in fr.items():
        apex, cr = apex @ B.T, cr @ B.T
        col = cmap(norm(k))
        for c in cr:
            ax.plot(*zip(apex, c), color=col, lw=0.6)
        ax.plot(*np.vstack([cr, cr[:1]]).T, color=col, lw=0.8)
        allp.append(cr)
    if xb is not None:
        ax.plot(*xb[okb].T, color=DENSE, lw=1.2, ls="--")
    ax.scatter(*o[0], color=INK, s=12, zorder=5)
    ax.text(*o[0], "  t=0", fontsize=7, color=INK2)
    allp = np.vstack(allp)
    lo, hi = allp.min(0), allp.max(0)
    rng = np.maximum(hi - lo, 0.2 * (hi - lo).max())    # equal units, no flat axis
    mid = (hi + lo) / 2
    ax.set_xlim(mid[0] - rng[0] / 2, mid[0] + rng[0] / 2)
    ax.set_ylim(mid[1] - rng[1] / 2, mid[1] + rng[1] / 2)
    ax.set_zlim(mid[2] - rng[2] / 2, mid[2] + rng[2] / 2)
    ax.set_box_aspect(rng)
    ax.set_xlabel("right"); ax.set_ylabel("forward"); ax.set_zlabel("up")
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(MaxNLocator(3))
    ax.tick_params(labelsize=6.5, pad=0)
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1, 1, 1, 0))
        pane._axinfo["grid"].update(color=GRID, linewidth=0.5)
    ax.view_init(elev=22, azim=-60)
    ax.set_title("(a) camera centres and frustums", loc="left")

    a2 = fig.add_subplot(gs[0, 1])
    lc2 = LineCollection(np.stack([o[:-1, :2], o[1:, :2]], 1), cmap=cmap, norm=norm, lw=2)
    lc2.set_array(np.arange(F - 1))
    a2.add_collection(lc2)
    h = fwd[keys, :2] / np.maximum(np.linalg.norm(fwd[keys, :2], axis=1, keepdims=True), 1e-9)
    a2.quiver(o[keys, 0], o[keys, 1], h[:, 0], h[:, 1], color=[cmap(norm(k)) for k in keys],
              angles="xy", scale_units="xy", scale=1 / (1.5 * s), width=0.006)
    if xb is not None:
        a2.plot(xb[okb, 0], xb[okb, 1], color=DENSE, lw=1.2, ls="--")
    lo2, hi2 = core[:, :2].min(0), core[:, :2].max(0)
    half = 0.55 * max(float((hi2 - lo2).max()), 1e-6)
    m2 = (hi2 + lo2) / 2
    a2.set_xlim(m2[0] - half, m2[0] + half); a2.set_ylim(m2[1] - half, m2[1] + half)
    a2.set_aspect("equal", adjustable="box")
    a2.set(xlabel="right", ylabel="forward")
    a2.set_title("(b) top-down (ground plane)", loc="left")

    a3 = fig.add_subplot(gs[1, 1])
    for j, (lbl, ls) in enumerate((("right", "-"), ("forward", "--"), ("up", ":"))):
        y = o[:, j] - o[0, j]
        a3.plot(y, color=INK2, lw=1.4, ls=ls)
        a3.annotate(lbl, (F - 1, y[-1]), xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    a3.set(xlabel="frame t", ylabel="translation")
    a3.set_xlim(0, F - 1)
    a3.set_title("(c) camera translation vs t", loc="left")

    a4 = fig.add_subplot(gs[2, 1])
    for lbl, y, ls in (("yaw", yaw - yaw[0], "-"), ("pitch", pitch - pitch[0], "--")):
        a4.plot(y, color=INK2, lw=1.4, ls=ls)
        a4.annotate(lbl, (F - 1, y[-1]), xytext=(3, 0), textcoords="offset points",
                    fontsize=7, color=INK2, va="center")
    a4.set(xlabel="frame t", ylabel="degrees")
    a4.set_xlim(0, F - 1)
    a4.set_title("(d) camera orientation vs t", loc="left")

    cb = fig.colorbar(ScalarMappable(norm, cmap), ax=ax, shrink=0.5, pad=0.0,
                      location="bottom", aspect=30)
    cb.set_label("frame t", color=INK2); cb.outline.set_visible(False)
    hand = [Line2D([], [], color=TIME_RAMP[3], lw=2, label="camera centre o_t (colour = t)")]
    if xb is not None:
        hand.append(Line2D([], [], color=DENSE, lw=1.2, ls="--", label="subject centroid"))
    fig.legend(handles=hand, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04))
    save(fig, out, "fig_camera_trajectory")
    plt.close(fig)
    return {"frustum_frames": keys, "yaw_range_deg": float(np.ptp(yaw)),
            "pitch_range_deg": float(np.ptp(pitch)),
            "path_length": float(np.linalg.norm(np.diff(o, axis=0), axis=1).sum())}


def fig_sparse(d, B, frame, t_ref, trail, max_sparse, subj, out, plt):
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    P, vis, Q = d["P"], d["vis"], d["Q"]
    N, F = Q.shape
    Hp = d["D"].shape[1]
    H, W = frame.shape[:2]
    sx, sy = W / Hp, H / Hp
    is_subj = np.zeros(N, bool); is_subj[subj] = True
    is_bg = ~is_subj & ~d["B"]["y_dyn"].astype(bool)
    rng = np.random.default_rng(0)
    pick = lambda idx, n: np.sort(rng.choice(idx, min(len(idx), n), replace=False))
    bg = pick(np.flatnonzero(is_bg & vis[:, t_ref]), max_sparse)
    sj = pick(np.flatnonzero(is_subj & vis[:, t_ref]), max_sparse // 2)
    t0 = max(t_ref - trail, 0)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                 gridspec_kw={"width_ratios": [W / H, 1.0]})
    image_ax(a1, frame, f"(a) sparse tracks at t = {t_ref}, trails over {t_ref - t0} frames")
    for idx, col, lw in ((bg, INK2, 0.6), (sj, SUBJ, 0.9)):
        segs, _ = track_segments(P, vis, idx, t0, t_ref, sx, sy)
        a1.add_collection(LineCollection(segs, colors=col, linewidths=lw, alpha=0.8))
        a1.scatter(P[idx, t_ref, 0] * sx, P[idx, t_ref, 1] * sy, s=5, color=col,
                   edgecolors="white", linewidths=0.3, zorder=3)
    a1.set_xlim(0, W); a1.set_ylim(H, 0)

    # (b) bird's-eye map of the world-frame (X0) sparse tracks at first valid frame
    ok = Q.any(1) & is_bg
    first = Q.argmax(1)
    Xw = d["X0"][np.arange(N), first][ok].astype(np.float64) @ B.T
    Xw = Xw[pick(np.arange(len(Xw)), 6000)]
    G = d["G"].astype(np.float64)
    o = G[:, :3, 3] @ B.T
    norm = Normalize(0, F - 1)
    a2.scatter(Xw[:, 0], Xw[:, 1], s=0.6, color=MUTED, alpha=0.45, linewidths=0,
               rasterized=True)
    lc = LineCollection(np.stack([o[:-1, :2], o[1:, :2]], 1), cmap=TIME_CMAP, norm=norm, lw=2)
    lc.set_array(np.arange(F - 1)); a2.add_collection(lc)
    a2.scatter(*o[t_ref, :2], s=24, color=TIME_CMAP(norm(t_ref)), edgecolors="white",
               linewidths=1, zorder=4)
    xb = d["B5"].get("X_bar_track")
    if xb is not None:
        xb = xb.astype(np.float64) @ B.T
        okb = np.isfinite(xb).all(1)
        a2.plot(xb[okb, 0], xb[okb, 1], color=DENSE, lw=1.2, ls="--")
    pts = np.vstack([Xw[:, :2], o[:, :2]])
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    lo, hi = np.minimum(lo, o[:, :2].min(0)), np.maximum(hi, o[:, :2].max(0))
    pad = 0.05 * (hi - lo)
    a2.set_xlim(lo[0] - pad[0], hi[0] + pad[0]); a2.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    a2.set_aspect("equal", adjustable="box")
    a2.set(xlabel="right", ylabel="forward")
    a2.set_title("(b) world frame X0, bird's-eye view", loc="left")
    cb = fig.colorbar(ScalarMappable(norm, TIME_CMAP), ax=a2, shrink=0.7, pad=0.02)
    cb.set_label("frame t", color=INK2); cb.outline.set_visible(False)
    hand = [Line2D([], [], color=INK2, lw=1, marker="o", ms=3, label="static sparse tracks"),
            Line2D([], [], color=SUBJ, lw=1, marker="o", ms=3,
                   label="subject sparse tracks (Stage B)"),
            Line2D([], [], color=TIME_RAMP[3], lw=2, label="camera path")]
    if xb is not None:
        hand.append(Line2D([], [], color=DENSE, lw=1.2, ls="--", label="subject centroid"))
    fig.legend(handles=hand, loc="lower center", ncol=len(hand), bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, out, "fig_sparse_tracks")
    plt.close(fig)
    return {"n_bg_drawn": int(len(bg)), "n_subj_drawn": int(len(sj)),
            "n_bg_total": int(is_bg.sum()), "n_subj_total": int(is_subj.sum()),
            "n_other_dynamic_not_drawn": int((~is_subj & ~is_bg).sum()),
            "n_world_points": int(len(Xw))}


def fig_dense(d, frame, t_ref, trail, out, plt):
    from matplotlib.cm import ScalarMappable
    from matplotlib.collections import LineCollection
    from matplotlib.colors import Normalize
    P, vis = d["C5"]["P_dense"], d["vis_dense"]
    N, F = vis.shape
    M = d["M"]
    Hp = M.shape[1]
    H, W = frame.shape[:2]
    sx, sy = W / Hp, H / Hp
    idx = np.arange(N)
    now = idx[vis[:, t_ref]]
    t0 = max(t_ref - trail, 0)

    # (b) crop = mask bbox at t_ref U full-clip dense trajectories, padded
    segs_all, ts = track_segments(P, vis, idx, 0, F - 1, sx, sy)
    if M[t_ref].any():
        vv, uu = np.nonzero(M[t_ref])
        x0, x1, y0, y1 = uu.min() * sx, (uu.max() + 1) * sx, vv.min() * sy, (vv.max() + 1) * sy
    else:
        x0, x1, y0, y1 = 0, W, 0, H
    if len(segs_all):
        x0, x1 = min(x0, segs_all[..., 0].min()), max(x1, segs_all[..., 0].max())
        y0, y1 = min(y0, segs_all[..., 1].min()), max(y1, segs_all[..., 1].max())
    # pad 15%, then grow the short side to the frame aspect so both panels match
    cw, ch = 1.3 * (x1 - x0), 1.3 * (y1 - y0)
    cw, ch = max(cw, ch * W / H), max(ch, cw * H / W)
    cw, ch = min(cw, W), min(ch, H)
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    cx0 = float(np.clip(mx - cw / 2, 0, W - cw)); cx1 = cx0 + cw
    cy0 = float(np.clip(my - ch / 2, 0, H - ch)); cy1 = cy0 + ch

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 2.9))
    image_ax(a1, frame, f"(a) mask + dense tracks, t = {t_ref}")
    draw_mask(a1, M[t_ref], sx, sy)
    segs, _ = track_segments(P, vis, now, t0, t_ref, sx, sy)
    a1.add_collection(LineCollection(segs, colors=DENSE, linewidths=0.8, alpha=0.9))
    a1.scatter(P[now, t_ref, 0] * sx, P[now, t_ref, 1] * sy, s=7, color=DENSE,
               edgecolors="white", linewidths=0.4, zorder=3)
    a1.set_xlim(0, W); a1.set_ylim(H, 0)

    segs = segs_all
    image_ax(a2, frame, "(b) dense trajectories, full clip")
    draw_mask(a2, M[t_ref], sx, sy, fill=False)
    norm = Normalize(0, F - 1)
    lc = LineCollection(segs, cmap=TIME_CMAP, norm=norm, linewidths=0.9)
    lc.set_array(ts); a2.add_collection(lc)
    a2.set_xlim(cx0, cx1); a2.set_ylim(cy1, cy0)
    cb = fig.colorbar(ScalarMappable(norm, TIME_CMAP), ax=a2, shrink=0.8, pad=0.02)
    cb.set_label("frame t", color=INK2); cb.outline.set_visible(False)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hand = [Patch(facecolor=MASK, alpha=0.4, edgecolor=MASK, label="subject mask (Stage C)"),
            Line2D([], [], color=DENSE, lw=1, marker="o", ms=3,
                   label=f"dense tracks (Stage C.5), trails over {t_ref - t0} frames")]
    fig.legend(handles=hand, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, out, "fig_dense_tracks")
    plt.close(fig)
    return {"n_dense": int(N), "n_dense_visible_t_ref": int(len(now)),
            "dense_valid_frac": float(d["C5"]["Q_dense"].mean()),
            "inside_mask_frac_median": float(np.nanmedian(d["C5"]["inside_mask_frac"]))}


def fig_film(d, frames, film, subj, max_sparse, out, plt):
    from matplotlib.lines import Line2D
    P, vis = d["P"], d["vis"]
    Pd, vd = d["C5"]["P_dense"], d["vis_dense"]
    N = P.shape[0]
    Hp = d["M"].shape[1]
    H, W = frames[film[0]].shape[:2]
    sx, sy = W / Hp, H / Hp
    is_subj = np.zeros(N, bool); is_subj[subj] = True
    is_bg = ~is_subj & ~d["B"]["y_dyn"].astype(bool)
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(film), figsize=(7.4, 7.4 / len(film) * H / W + 0.6))
    for ax, t in zip(np.atleast_1d(axes), film):
        image_ax(ax, frames[t], f"t = {t}", loc="center")
        draw_mask(ax, d["M"][t], sx, sy, fill=False)
        bg = np.flatnonzero(is_bg & vis[:, t])
        bg = rng.choice(bg, min(len(bg), max_sparse), replace=False)
        sj = np.flatnonzero(is_subj & vis[:, t])
        sj = rng.choice(sj, min(len(sj), max_sparse // 2), replace=False)
        dn = np.flatnonzero(vd[:, t])
        ax.scatter(P[bg, t, 0] * sx, P[bg, t, 1] * sy, s=1.5, color=INK2, linewidths=0)
        ax.scatter(P[sj, t, 0] * sx, P[sj, t, 1] * sy, s=2.5, color=SUBJ, linewidths=0)
        ax.scatter(Pd[dn, t, 0] * sx, Pd[dn, t, 1] * sy, s=3, color=DENSE,
                   edgecolors="white", linewidths=0.2)
        ax.set_xlim(0, W); ax.set_ylim(H, 0)
    hand = [Line2D([], [], color=MASK, lw=1.2, label="subject mask (Stage C)"),
            Line2D([], [], ls="", marker="o", ms=3, color=INK2, label="static sparse"),
            Line2D([], [], ls="", marker="o", ms=3, color=SUBJ, label="subject sparse"),
            Line2D([], [], ls="", marker="o", ms=4, color=DENSE, label="dense (Stage C.5)")]
    fig.legend(handles=hand, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    save(fig, out, "fig_filmstrip")
    plt.close(fig)


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="000000000035.1.003")
    ap.add_argument("--t-ref", type=int, default=None, help="default: Stage C canonical t_c")
    ap.add_argument("--trail", type=int, default=16, help="trail length in frames")
    ap.add_argument("--max-sparse", type=int, default=600, help="sparse tracks drawn per class")
    ap.add_argument("--frustum-every", type=int, default=10)
    ap.add_argument("--n-film", type=int, default=5)
    ap.add_argument("--out", default=None, help="default: <drive_root>/viz/thesis/<clip>")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    global TIME_CMAP
    TIME_CMAP = LinearSegmentedColormap.from_list("time", TIME_RAMP)
    style(plt)

    cfg = load_config("configs/stage_a.yaml")
    c = args.clip_id
    d = load(cfg, c)
    Hp = int(cfg["video"]["process_height"])
    N, F = d["Q"].shape
    assert d["D"].shape[1:] == (Hp, Hp) and d["M"].shape == (F, Hp, Hp), "resolution mismatch"
    d["vis"] = visible(d["D"], d["X_local"], d["P"], d["Q"])
    d["vis_dense"] = visible(d["D"], d["C5"]["X_local_dense"], d["C5"]["P_dense"],
                             d["C5"]["Q_dense"])
    subj = (d["B"]["fc_blob_idx"] if d["channel"] == "branch"
            else d["B"]["o_src_members"]).astype(int)
    t_ref = d["t_c"] if args.t_ref is None else int(args.t_ref)
    assert 0 <= t_ref < F, f"t_ref {t_ref} outside [0, {F})"
    film = [int(t) for t in np.linspace(0, F - 1, args.n_film).round()]
    B = scene_basis(d["B5"].get("u_scene"))
    up_src = "B.5 u_scene" if "u_scene" in d["B5"] else "camera-0 -y (no B.5 record)"

    root = Path(str(cfg["paths"]["drive_root"]))
    out = Path(args.out) if args.out else root / "viz" / "thesis" / c
    out.mkdir(parents=True, exist_ok=True)
    frames = read_frames(d["prov"]["clip_path"], sorted({t_ref, *film}))

    cam = fig_camera(d, B, Hp, args.frustum_every, out, plt)
    sp = fig_sparse(d, B, frames[t_ref], t_ref, args.trail, args.max_sparse, subj, out, plt)
    dn = fig_dense(d, frames[t_ref], t_ref, args.trail, out, plt)
    fig_film(d, frames, film, subj, args.max_sparse, out, plt)

    subj_name = ("follow-cam blob fc_blob_idx (subject + halo)" if d["channel"] == "branch"
                 else "X0-path winner o_src_members")
    md = [f"# Thesis figures - {c}", "",
          f"- config hash `{d['prov'].get('config_hash')}`, channel **{d['channel']}**, "
          f"F = {F}, N = {N} sparse, N_d = {dn['n_dense']} dense, process res {Hp}px, "
          f"video {frames[t_ref].shape[1]}x{frames[t_ref].shape[0]}",
          f"- scene frame (right, forward, up) from {up_src}; units = D4RT model units",
          f"- t_ref = {t_ref} (Stage C t_c = {d['t_c']}), filmstrip frames {film}, "
          f"trail {args.trail} frames",
          f"- subject sparse set = {subj_name}: {sp['n_subj_total']} tracks "
          f"({sp['n_subj_drawn']} drawn at t_ref); static background {sp['n_bg_total']} "
          f"({sp['n_bg_drawn']} drawn); other dynamic-labelled, not drawn: "
          f"{sp['n_other_dynamic_not_drawn']}",
          f"- camera: path length {cam['path_length']:.3f}, yaw range "
          f"{cam['yaw_range_deg']:.1f} deg, pitch range {cam['pitch_range_deg']:.1f} deg; "
          f"frustums at {cam['frustum_frames']}",
          f"- dense: {dn['n_dense_visible_t_ref']} visible at t_ref, valid fraction "
          f"{dn['dense_valid_frac']:.3f}, median inside-mask fraction "
          f"{dn['inside_mask_frac_median']:.3f}",
          "- viz-only occlusion cull: a point is hidden when > 8% behind the depth map "
          "at its pixel (Q untouched)",
          "- on branch clips the dense X0 carries no object motion; the subject centroid "
          "drawn is B.5 X_bar_track (camera trajectory composed with relative motion)"]
    (out / "figures.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {out}")
    return 0


TIME_CMAP = None

if __name__ == "__main__":
    sys.exit(main())
