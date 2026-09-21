#!/usr/bin/env python
"""Stage A end-to-end: clip -> gate-checked frozen-contract bundle + §8 viz.

    python scripts/run_stage_a.py --clip /content/data/<clip>.mp4

A failed gate prints the QC report and exits nonzero WITHOUT writing a bundle
(SKILL §7). Requires thresholds.tau_g_px to be frozen from 3-clip probe runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Import ALL of our src.* modules BEFORE the adapter is constructed: creating
# it evicts our `src` package from sys.modules (name collision with the model
# repo's package of the same name).
from src.preprocess.config import load_config          # noqa: E402
from src.preprocess.d4rt_adapter import D4RTAdapter, window_boundaries  # noqa: E402
from src.preprocess.anchors import discover_anchors    # noqa: E402
from src.preprocess.tracks import build_tracks         # noqa: E402
from src.preprocess.bundle import run_gates, write_bundle  # noqa: E402
import scripts.inspect_bundle as inspect_bundle        # noqa: E402


def load_clip(path: str, max_frames: int) -> np.ndarray:
    import imageio.v2 as iio
    rd = iio.get_reader(path)
    frames = []
    for i, fr in enumerate(rd):
        if i >= max_frames:
            break
        frames.append(fr[..., :3])
    rd.close()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return np.stack(frames).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True, help="path to the reference clip")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config, require_filled=("repo", "d4rt"))
    if not Path(args.clip).is_file():
        raise SystemExit(
            f"--clip must be the path of one video FILE, got: {args.clip!r}\n"
            "e.g. /content/data/videos/clip_001.mp4 — list candidates with:\n"
            "  !find /content/data -name '*.mp4'")
    clip_id = Path(args.clip).stem
    video = load_clip(args.clip, int(cfg["video"]["max_frames"]))
    print(f"[stage_a] {clip_id}: {video.shape[0]} frames "
          f"{video.shape[2]}x{video.shape[1]} (config {cfg['_meta']['config_hash']})")

    t0 = time.time()
    adapter = D4RTAdapter(cfg)
    adapter.open(video, clip_path=str(args.clip))
    K, G, sim3 = adapter.cameras()
    print(f"[stage_a] cameras derived ({time.time() - t0:.0f}s); "
          f"sim3 scale range [{sim3.min():.3f}, {sim3.max():.3f}]")

    anchors, track_cache = discover_anchors(adapter, cfg)
    print(f"[stage_a] {len(anchors)} anchors ({time.time() - t0:.0f}s)")

    tr = build_tracks(adapter, anchors, cfg, cached=track_cache)
    D = np.stack([adapter.depth_map(t, int(cfg["video"]["depth_stride"]))
                  for t in range(adapter.T)])
    print(f"[stage_a] tracks + depth done ({time.time() - t0:.0f}s)")

    qc = run_gates(tr, cfg)
    # Task 0 (Stage B SKILL §1, human-signed): scale re-gauge frames for the
    # drift model — from the adapter's actual window schedule, additive field.
    qc["window_boundaries"] = window_boundaries(adapter.T, adapter.clip_frames)
    for name, g in qc["gates"].items():
        tag = "UNGATED" if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
        print(f"[gate] {tag} {name}: {g}")
    if qc["ungated_pending_calibration"]:
        print(f"[gate] note: {qc['ungated_pending_calibration']} report statistics "
              "only — thresholds pending human calibration (CLAUDE.md §4)")
    viz_dir = Path(cfg["paths"]["viz_dir"]) / clip_id
    if not qc["all_pass"]:
        viz_dir.mkdir(parents=True, exist_ok=True)
        # Stamp hash + cost so the calibration table can include failed clips
        # (CLAUDE.md §4: the table carries EVERY clip's statistics).
        qc_fail = {**qc, "config_hash": cfg["_meta"]["config_hash"],
                   "runtime_s": round(time.time() - t0, 1),
                   "peak_vram_gb": round(adapter.torch.cuda.max_memory_allocated() / 2**30
                                         if adapter.torch.cuda.is_available() else 0.0, 2)}
        (viz_dir / "qc_failed.json").write_text(json.dumps(qc_fail, indent=2),
                                                encoding="utf-8")
        print(f"[stage_a] GATES FAILED — no bundle written; report: "
              f"{viz_dir / 'qc_failed.json'} (SKILL §7)")
        return 1

    # a qc_failed.json from an earlier failed run is now stale — remove it so
    # the viz folder never shows a failure report next to passing artifacts
    (viz_dir / "qc_failed.json").unlink(missing_ok=True)

    arrays = {"K": tr["K"], "G": tr["G"], "D": D, "anchors": tr["anchors"],
              "X_local": tr["X_local"], "X0": tr["X0"], "P": tr["P"],
              "V": tr["V"], "C": tr["C"], "Q": tr["Q"]}
    peak_vram_gb = (adapter.torch.cuda.max_memory_allocated() / 2**30
                    if adapter.torch.cuda.is_available() else 0.0)
    prov = {**adapter.provenance(), "clip_path": str(args.clip),
            "frames": int(adapter.T),
            "runtime_s": round(time.time() - t0, 1),
            "peak_vram_gb": round(peak_vram_gb, 2)}
    out = write_bundle(cfg, clip_id, arrays, prov, qc)
    print(f"[stage_a] bundle -> {out}")

    inspect_bundle.render(cfg, clip_id, adapter.video_model, {**arrays,
                          "eq24_px": tr["eq24_px"], "eq24_rel": tr["eq24_rel"]})
    print(f"[stage_a] review package -> {viz_dir}  ({time.time() - t0:.0f}s total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
