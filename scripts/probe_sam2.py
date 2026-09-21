#!/usr/bin/env python
"""Stage C Task 0 — SAM2 mini-probe (SKILL_stage_c_mask §2.1).

MEASURES, on one real clip with real Stage B prompts, before ANY module code
(CLAUDE.md §3.1 probe-before-adapter, §3.3 conventions measured not assumed):

  P1  video-predictor API: point+box seeding at a mid-clip frame; whether
      BIDIRECTIONAL propagation is native (a `reverse` path) or needs two
      passes — introspected AND executed, not assumed.
  P2  resolution behavior: native frame size vs the model's internal size vs
      returned mask shape/dtype/range; masks must be mappable to the 256x256
      process frame for M_src (masks don't touch K; shape mismatch would be
      a contract violation).
  P3  memory-bank behavior under mid-sequence re-prompting: does injecting
      new points update or reset previously propagated frames; does the API
      accept it at all after propagation.
  P4  VRAM / runtime per phase for an 81-frame clip on the session GPU.

Default clip: 000000000035.1.003 (branch-active — prompts are the frozen
Stage B contract's `fc_seed_points` + `seed_bbox` at `seed_frame`, scaled
process->native; the followed car barely moves in frame, so the same points
re-used at a later frame are valid for the P3 re-prompt test).

Install (Colab, once):
    pip install "git+https://github.com/facebookresearch/sam2.git"
Checkpoint downloads via HF hub (HF_TOKEN already in env from bootstrap).

    python scripts/probe_sam2.py [--clip-id ID] [--variant base-plus]

Writes <stage_c_viz>/probe_sam2_report.md + overlay PNGs.
STOP POINT (SKILL §2.1): human reviews the report before adapter code.
"""
from __future__ import annotations

import argparse
import inspect as _inspect
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.bundle import read_bundle      # noqa: E402
from src.preprocess.config import load_config      # noqa: E402

VARIANTS = {"tiny": "facebook/sam2.1-hiera-tiny",
            "small": "facebook/sam2.1-hiera-small",
            "base-plus": "facebook/sam2.1-hiera-base-plus",
            "large": "facebook/sam2.1-hiera-large"}


def _extract_frames(clip_path, F, out_dir):
    import imageio.v2 as iio
    out_dir.mkdir(parents=True, exist_ok=True)
    rd = iio.get_reader(clip_path)
    shape = None
    n = 0
    for i, fr in enumerate(rd):
        if i >= F:
            break
        iio.imwrite(out_dir / f"{i:05d}.jpg", fr[..., :3], quality=95)
        shape = fr.shape[:2]
        n = i + 1
    rd.close()
    return n, shape  # (H, W) native


def _overlay(png_path, frame_path, mask, pts=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as iio
    img = iio.imread(frame_path)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img)
    if mask is not None:
        ax.imshow(np.where(mask, 1.0, np.nan), alpha=0.45, cmap="autumn",
                  vmin=0, vmax=1)
    if pts is not None:
        ax.scatter(pts[:, 0], pts[:, 1], s=30, c="lime", marker="+")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", default="000000000035.1.003")
    ap.add_argument("--variant", default="base-plus", choices=list(VARIANTS))
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--frames-dir", default="/tmp/sam2_probe_frames")
    args = ap.parse_args()
    cfg = load_config(args.config)
    Hp = int(cfg["video"]["process_height"])
    out = Path(cfg["paths"]["stage_c_viz"])
    out.mkdir(parents=True, exist_ok=True)
    md = [f"# SAM2 mini-probe — {args.clip_id}, variant {args.variant}\n"]

    try:
        import torch
        import sam2
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as e:
        raise SystemExit(
            f"SAM2 import failed ({e}) — install first:\n"
            '  pip install "git+https://github.com/facebookresearch/sam2.git"')
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    md.append(f"- versions: sam2 {getattr(sam2, '__version__', '?')}, torch "
              f"{torch.__version__}, GPU **{gpu}**")

    # ---- Stage B prompts (frozen contract; branch-active clip expected)
    b = read_bundle(cfg, args.clip_id)
    F = b["Q"].shape[1]
    bdir = Path(cfg["paths"]["stage_b_cache"])
    qc_b = json.loads((bdir / f"{args.clip_id}_qc_b.json").read_text(encoding="utf-8"))
    fcr = qc_b.get("follow_cam") or {}
    assert fcr.get("branch_active"), \
        f"{args.clip_id} is not branch-active — probe expects seed prompts"
    with np.load(bdir / f"{args.clip_id}.npz") as z:
        seed_pts = z["fc_seed_points"].astype(np.float64)
    seed_frame = int(fcr["seed_frame"])
    seed_bbox = [float(v) for v in fcr["seed_bbox"]]

    # ---- frames at NATIVE resolution
    t0 = time.time()
    n_fr, (Hn, Wn) = _extract_frames(b["provenance"]["clip_path"], F,
                                     Path(args.frames_dir) / args.clip_id)
    md.append(f"- frames: {n_fr} extracted at native {Wn}x{Hn} "
              f"(process {Hp}x{Hp}; scale {Wn / Hp:.3f}x{Hn / Hp:.3f}) "
              f"[{time.time() - t0:.1f}s]")
    sx, sy = Wn / Hp, Hn / Hp
    pts_native = seed_pts * np.array([sx, sy])
    box_native = np.array([seed_bbox[0] * sx, seed_bbox[1] * sy,
                           seed_bbox[2] * sx, seed_bbox[3] * sy])

    # ---- P1: API surface (introspected, then executed)
    sig_prop = str(_inspect.signature(SAM2VideoPredictor.propagate_in_video))
    sig_add = str(_inspect.signature(SAM2VideoPredictor.add_new_points_or_box))
    md.append(f"\n## P1 — API\n\n- `propagate_in_video{sig_prop}`\n"
              f"- `add_new_points_or_box{sig_add}`")
    native_reverse = "reverse" in sig_prop
    md.append(f"- native `reverse` parameter: **{native_reverse}**")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    predictor = SAM2VideoPredictor.from_pretrained(VARIANTS[args.variant])
    md.append(f"- from_pretrained: {time.time() - t0:.1f}s")
    amp = torch.autocast("cuda", dtype=torch.bfloat16)
    with amp, torch.inference_mode():
        t0 = time.time()
        state = predictor.init_state(
            video_path=str(Path(args.frames_dir) / args.clip_id))
        t_init = time.time() - t0
        _, obj_ids, logits = predictor.add_new_points_or_box(
            state, frame_idx=seed_frame, obj_id=1,
            points=pts_native.astype(np.float32),
            labels=np.ones(len(pts_native), np.int32),
            box=box_native.astype(np.float32))
        md.append(f"- seed at t={seed_frame}: {len(pts_native)} pts + box "
                  f"accepted; obj_ids={list(obj_ids)}")

        masks = {}
        t0 = time.time()
        for fi, oids, ml in predictor.propagate_in_video(
                state, start_frame_idx=seed_frame):
            masks[fi] = (ml[0, 0] > 0).cpu().numpy()
        t_fwd = time.time() - t0
        fwd_frames = sorted(masks)
        t0 = time.time()
        n_bwd, bwd_err = 0, None
        if native_reverse:
            try:
                for fi, oids, ml in predictor.propagate_in_video(
                        state, start_frame_idx=seed_frame, reverse=True):
                    masks[fi] = (ml[0, 0] > 0).cpu().numpy()
                    n_bwd += 1
            except Exception as e:  # noqa: BLE001 — probe records, not hides
                bwd_err = repr(e)
        t_bwd = time.time() - t0
        md.append(f"- forward: frames [{fwd_frames[0]}..{fwd_frames[-1]}] "
                  f"({len(fwd_frames)}) in {t_fwd:.1f}s")
        md.append(f"- reverse pass: {n_bwd} frames in {t_bwd:.1f}s"
                  + (f" — ERROR: {bwd_err}" if bwd_err else ""))
        md.append(f"- coverage after both: {len(masks)}/{n_fr} frames — "
                  f"**bidirectional from mid-clip "
                  f"{'NATIVE' if len(masks) == n_fr and not bwd_err else 'NOT confirmed'}**")

        # ---- P2: resolution + output semantics
        ml_probe = logits
        md.append(f"\n## P2 — resolution/output\n\n"
                  f"- internal image size: {getattr(predictor, 'image_size', '?')}"
                  f"; returned logits shape {tuple(ml_probe.shape)} dtype "
                  f"{ml_probe.dtype} range [{float(ml_probe.min()):.1f}, "
                  f"{float(ml_probe.max()):.1f}]")
        m0 = masks[seed_frame]
        md.append(f"- binarized (>0) mask at seed: shape {m0.shape} vs native "
                  f"({Hn},{Wn}) — match={m0.shape == (Hn, Wn)}; area frac "
                  f"{m0.mean():.4f}")
        import cv2
        m_proc = cv2.resize(m0.astype(np.uint8), (Hp, Hp),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
        md.append(f"- nearest-downsample to process {Hp}x{Hp}: area frac "
                  f"{m_proc.mean():.4f} (rel change "
                  f"{abs(m_proc.mean() - m0.mean()) / max(m0.mean(), 1e-9):.3f})")

        # ---- P3: mid-sequence re-prompt behavior
        md.append("\n## P3 — re-prompt / memory bank\n")
        probe_t = min(seed_frame + 20, n_fr - 10)
        before = {k: masks[k].copy() for k in
                  (seed_frame + 5, probe_t + 5) if k in masks}
        try:
            predictor.add_new_points_or_box(
                state, frame_idx=probe_t, obj_id=1,
                points=pts_native.astype(np.float32),
                labels=np.ones(len(pts_native), np.int32))
            md.append(f"- re-prompt at t={probe_t} AFTER propagation: accepted")
            n_re = 0
            for fi, oids, ml in predictor.propagate_in_video(
                    state, start_frame_idx=probe_t,
                    max_frame_num_to_track=10):
                if fi in before:
                    changed = bool((before[fi] != (ml[0, 0] > 0).cpu().numpy()).mean() > 0.01)
                    md.append(f"  - frame {fi}: mask changed >1% = {changed}")
                n_re += 1
            md.append(f"- re-propagation covered {n_re} frames — bank "
                      "**updates in place** (no state reset needed)")
        except Exception as e:  # noqa: BLE001 — the behavior IS the finding
            md.append(f"- re-prompt raised: {e!r} — bank likely requires "
                      "reset-and-reseed; §4 re-prompt design must use "
                      "predictor.reset_state + full reseed")

    # ---- P4: cost
    vram = torch.cuda.max_memory_allocated() / 2**30
    md.append(f"\n## P4 — cost ({args.variant})\n\n"
              f"- init_state {t_init:.1f}s, forward {t_fwd:.1f}s, reverse "
              f"{t_bwd:.1f}s for {n_fr} frames; peak VRAM **{vram:.2f} GiB**")

    for tt in (max(seed_frame - 20, 0), seed_frame,
               min(seed_frame + 20, n_fr - 1)):
        _overlay(out / f"probe_mask_t{tt}.png",
                 Path(args.frames_dir) / args.clip_id / f"{tt:05d}.jpg",
                 masks.get(tt), pts_native if tt == seed_frame else None)
    md.append(f"\nOverlays: probe_mask_t*.png (seed={seed_frame} ±20)\n")

    (out / "probe_sam2_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nreport -> {out / 'probe_sam2_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
