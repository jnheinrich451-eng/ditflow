#!/usr/bin/env python
"""Human-review visualizations (SKILL §8) — the "check the effect" deliverable.

Called by run_stage_a.py after gates pass; also runs standalone on a cached
bundle:  python scripts/inspect_bundle.py --clip-id <id>
Outputs to viz_dir/<clip_id>/: tracks_overlay.mp4, depth.mp4, camera_traj.png,
x0_static_drift.png, consistency_hist.png.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# ALL src.preprocess imports must happen at module load — after a D4RTAdapter
# exists, `src` resolves to the model repo's package (name collision) and any
# lazy `from src.preprocess...` import inside a function raises.
from src.preprocess.bundle import read_bundle  # noqa: E402
from src.preprocess.config import load_config  # noqa: E402
from src.preprocess.tracks import eq24_relative, eq24_residual_px  # noqa: E402

UPSCALE = 2  # viz-only upscale of the 256px process frames
FPS = 12


def _colors(n: int) -> np.ndarray:
    rng = np.random.default_rng(0)  # stable ids across renders
    return rng.integers(64, 255, size=(n, 3)).astype(np.uint8)


def render(cfg: dict, clip_id: str, frames_u8: np.ndarray, b: dict) -> Path:
    """frames_u8: process-resolution video [F,H,W,3]; b: contract arrays
    (+ optional eq24_px, recomputed if absent)."""
    import cv2
    import imageio.v2 as iio
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(cfg["paths"]["viz_dir"]) / clip_id
    out.mkdir(parents=True, exist_ok=True)
    K, G, D = b["K"], b["G"], b["D"]
    P, V, C, Q = b["P"], b["V"], b["C"], b["Q"]
    N, F = Q.shape
    colors = _colors(N)

    # 1) tracks_overlay.mp4 — P scatter, color by id, alpha=V*C, Q=0 hidden.
    # Alpha is normalized to the clip's own max: this checkpoint's V head sits
    # ~0.05 cross-frame, which would render raw V*C invisible; sqrt-gamma keeps
    # the relative dimming visible without washing it out.
    # Z-BUFFER OCCLUSION CULLING (viz-only, Q untouched): the V head cannot
    # flag occlusion reliably (tau_v is a token filter), so amodal tracks —
    # e.g. road points passing behind a car — would be drawn on top of the
    # occluder. A point is hidden when it sits >8% behind the depth map at
    # its own pixel.
    Xl = b["X_local"]
    Hm, Wm = D.shape[1:]
    u_idx = np.clip(P[..., 0].round().astype(int), 0, Wm - 1)
    v_idx = np.clip(P[..., 1].round().astype(int), 0, Hm - 1)
    d_at = D.astype(np.float32)[np.arange(F)[None, :], v_idx, u_idx]
    occluded = Xl[..., 2] > d_at * 1.08
    draw = Q & ~occluded
    A = V * C
    A = np.sqrt(np.clip(A / max(float(A[Q].max()) if Q.any() else 1.0, 1e-6), 0, 1))
    with iio.get_writer(out / "tracks_overlay.mp4", fps=FPS) as w:
        for t in range(F):
            frame = cv2.resize(frames_u8[t], None, fx=UPSCALE, fy=UPSCALE,
                               interpolation=cv2.INTER_NEAREST)
            for i in np.flatnonzero(draw[:, t]):
                a = float(A[i, t])
                if a < 0.08:  # ultra-dim points render as black speckle — skip
                    continue
                x, y = (P[i, t] * UPSCALE).round().astype(int)
                cv2.circle(frame, (x, y), 2, tuple(int(c * a) for c in colors[i]), -1)
            w.append_data(frame)

    # 2) depth.mp4 — fixed colormap scale across ALL frames (no flicker).
    d = D.astype(np.float32)
    lo, hi = np.percentile(d[np.isfinite(d)], [2, 98])
    cmap = plt.get_cmap("turbo")
    with iio.get_writer(out / "depth.mp4", fps=FPS) as w:
        for t in range(F):
            norm = np.clip((d[t] - lo) / max(hi - lo, 1e-9), 0, 1)
            w.append_data((cmap(norm)[..., :3] * 255).astype(np.uint8))

    # 3) camera_traj.png — with G_{0<-t}, o_t is simply G[:, :3, 3]; o_0 ~ 0.
    # Right panel: per-axis curves vs t. G_t is estimated independently per
    # frame from grid queries inside quantized windows, so two artifact shapes
    # are possible: per-frame jitter (noise) and step discontinuities where
    # the canonical window block switches — distinguish them here before
    # judging the 3D shape.
    o = G[:, :3, 3]
    assert np.linalg.norm(o[0]) < 1e-2, "o_0 != 0 — G convention broken (§8.3)"
    fig = plt.figure(figsize=(11, 5))
    ax = fig.add_subplot(121, projection="3d")
    ax.plot(o[:, 0], o[:, 1], o[:, 2], "-o", ms=2)
    ax.scatter(*o[0], c="r", s=40, label="t=0")
    ax.set_title(f"camera centers o_t = G[:3,3]  ({clip_id})\n"
                 "axes = camera-0 frame (OpenCV): x=right, y=DOWN, z=forward")
    ax.set_xlabel("x (right)")
    ax.set_ylabel("y (down)")
    ax.set_zlabel("z (forward)")
    ax.legend()
    ax2 = fig.add_subplot(122)
    for k, lbl in enumerate(("x (right)", "y (down)", "z (forward)")):
        ax2.plot(o[:, k], label=lbl, lw=1)
    ax2.set(xlabel="t", ylabel="camera center [model units, camera-0 axes]",
            title="per-axis: z ramp = forward travel (NOT altitude);\n"
                  "steps = window-switch artifact, fuzz = estimation noise")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(out / "camera_traj.png", dpi=120)
    plt.close(fig)

    # 4) x0_static_drift.png — 50 lowest-motion tracks: ||X0_t - X0_ref|| vs t.
    full = Q.mean(axis=1) >= 0.9
    X0 = b["X0"]
    motion = np.nanstd(np.where(Q[..., None], X0, np.nan), axis=1).sum(-1)
    cand = np.flatnonzero(full)
    pick = cand[np.argsort(motion[cand])[:50]] if cand.size else np.array([], int)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in pick:
        ref = X0[i, int(b["anchors"][i, 2])]
        drift = np.linalg.norm(X0[i] - ref, axis=-1)
        drift[~Q[i]] = np.nan
        ax.plot(drift, lw=0.6, alpha=0.5)
    ax.set(xlabel="t", ylabel="||X0_t - X0_ref||",
           title="static-track drift (flat = good; slope = ego-motion leakage)")
    fig.savefig(out / "x0_static_drift.png", dpi=120)
    plt.close(fig)

    # 5) consistency_hist.png — Eq (24) residual, two panels: pixel space
    # (visualization only since Task 2) and the GATED relative 3D residual.
    res_px = b.get("eq24_px")
    if res_px is None:
        res_px = eq24_residual_px(K, G, b["X_local"], X0, Q)
    res_rel = b.get("eq24_rel")
    if res_rel is None:
        res_rel = eq24_relative(G, b["X_local"], X0, Q)
    fig, (axp, axr) = plt.subplots(1, 2, figsize=(11, 4))
    vals = res_px[np.isfinite(res_px)]
    axp.hist(vals, bins=60)
    tau_px = cfg["thresholds"].get("tau_g_px")
    if tau_px is not None:
        axp.axvline(tau_px, color="r", ls="--", label=f"legacy marker {tau_px}")
        axp.legend()
    axp.set(xlabel="Eq(24) residual [px]", ylabel="count",
            title=f"pixel space — VIZ ONLY (median {np.median(vals):.2f} px)")
    rvals = res_rel[np.isfinite(res_rel)]
    axr.hist(rvals, bins=60)
    med = float(np.median(rvals))
    axr.axvline(5 * med, color="orange", ls=":", label="5x median (inlier edge)")
    tau_rel = cfg["thresholds"].get("tau_g_rel")
    if tau_rel is not None:
        axr.axvline(tau_rel, color="r", ls="--", label=f"tau_g_rel={tau_rel}")
    axr.legend()
    inl = float((rvals <= 5 * med).mean()) if rvals.size else float("nan")
    axr.set(xlabel="relative 3D residual ||X0-G·Xl||/max(Z,Zmin)", ylabel="count",
            title=f"GATED statistic (median {med:.4f}, inliers {inl:.1%})")
    fig.tight_layout()
    fig.savefig(out / "consistency_hist.png", dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", required=True)
    ap.add_argument("--config", default="configs/stage_a.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    b = read_bundle(cfg, args.clip_id)

    # Re-derive process-resolution frames from the source clip in provenance.
    import cv2
    import imageio.v2 as iio
    clip_path = b["provenance"].get("clip_path", "")
    if not Path(clip_path).exists():
        raise SystemExit(f"source clip not on this VM ({clip_path}) — re-download "
                         "data or run inspect right after run_stage_a.py")
    Hm = int(cfg["video"]["process_height"])
    rd = iio.get_reader(clip_path)
    frames = []
    for i, fr in enumerate(rd):
        if i >= b["K"].shape[0]:
            break
        frames.append(cv2.resize(fr[..., :3], (Hm, Hm), interpolation=cv2.INTER_AREA))
    rd.close()
    out = render(cfg, args.clip_id, np.stack(frames).astype(np.uint8), b)
    print(f"review package -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
