#!/usr/bin/env python
"""extract-v1: ONE video -> camera, depth, masks, tracks (Module 1 as an
instrument; docs/experiment-pipeline-v1.md S0).

    python scripts/extract.py --video /content/data/videos/<clip>.mp4 --fps 16 \
        [--subject-text bear] [--pack-only] [--force]

Runs the existing, gated Module-1 stages A -> B -> C -> C.5 (each its own
subprocess, skipped when its cache matches the current config hash), then
repacks their caches into

    <extract.out_dir>/<clip_id>/raw.npz       slow model outputs (see pack.build_raw)
    <extract.out_dir>/<clip_id>/derived.npz   per-track / per-step arrays
    <extract.out_dir>/<clip_id>/derived.json  scalars + per-instance trajectories
    <extract.out_dir>/<clip_id>/manifest.md   field table for this clip

Rules: identical code path for reference, generated and dataset videos (no
branch on the source; ground truth never enters). No metric, no threshold.
A stage gate failure is a finding: the run stops and says where.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                              # noqa: E402
from src.preprocess.config import _interpolate, load_config  # noqa: E402
from src.extract import pack                             # noqa: E402
from scripts.plot_thesis_tracks import load              # noqa: E402
from scripts.run_pipeline import current, run            # noqa: E402

FIELD_DOC = {
    "cam_extrinsics": "E[t] camera-t -> world (camera-0), see cam_convention",
    "cam_intrinsics": "K[t], process-resolution px, pixel-centre convention",
    "cam_is_metric": "False: D4RT units are similarity-ambiguous -> scorer fits scale",
    "cam_scale_drift": "per-frame sim3 scale X_local vs X0 grid fits (Stage A, measured)",
    "window_boundaries": "encode-window re-gauge frames (scale steps possible here)",
    "depth": "z-depth, model units, process res", "depth_valid": "finite & > 0",
    "instances": "0 background, 1 subject (Stage C mask); other movers have no mask",
    "instance_ids": "ids appearing in track_label (> 0)",
    "instance_has_mask": "whether the id has pixels in `instances`",
    "instance_stage_b_cluster": "Stage B cluster id behind ids >= 2 (-1 = subject)",
    "subject_present": "Stage C honest presence flag per frame",
    "t_c": "Stage C canonical / prompt frame",
    "track_xy": "pixel tracks, process px (meaningless where not visible)",
    "track_visible": "Stage A four-factor validity Q (not an occlusion probability)",
    "track_label": "0 static, 1 subject, 2.. other Stage-B clusters, -1 dynamic unclustered",
    "track_source": "0 sparse scene-wide (Stage A), 1 dense subject (Stage C.5)",
    "track_spawn": "frame the track was seeded at",
    "track_n_valid": "# visible frames", "track_n_in_mask": "# visible frames inside subject mask",
    "track_xyz": "world (camera-0) = E[t] @ X_local (NOT Stage A X0: contract §5.1/#8)",
    "image_hw": "process (H, W)", "native_hw": "video (H, W)", "fps": "input fps",
    "channel": "Stage B channel: x0 | branch (follow-cam)",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def probe_video(path: Path):
    import imageio.v2 as iio
    rd = iio.get_reader(str(path))
    meta = rd.get_meta_data()
    n, hw = 0, None
    for fr in rd:
        hw = hw or fr.shape[:2]
        n += 1
    rd.close()
    return n, hw, meta.get("fps")


def git_head() -> str:
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                           capture_output=True, text=True).stdout.strip()
    return (r.stdout.strip() or "unknown") + ("+dirty" if dirty else "")


def load_cfg_x(cfg: dict, path="configs/extract.yaml") -> dict:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = copy.deepcopy(raw)
    root["paths"] = cfg["paths"]
    out = _interpolate(raw, root)
    out["_hash"] = hashlib.sha256(json.dumps(raw, sort_keys=True).encode()).hexdigest()[:12]
    return out


def run_stages(cfg, cfg_x, video: Path, clip_id: str, force: bool) -> None:
    root = Path(str(cfg["paths"]["drive_root"]))
    h = cfg["_meta"]["config_hash"]
    done = {"A": root / "cache/stage_a" / clip_id / "provenance.json",
            "B": root / "cache/stage_b" / f"{clip_id}_qc_b.json",
            "C": root / "cache/stage_c" / clip_id / "qc_c.json",
            "C5": root / "cache/stage_c5" / f"{clip_id}_qc_c5.json"}
    cmd = {"A": ["scripts/run_stage_a.py", "--clip", str(video)],
           "B": ["scripts/run_stage_b.py", "--clip-id", clip_id],
           "C": ["scripts/run_stage_c.py", "--clips", clip_id],
           "C5": ["scripts/run_stage_c5.py", "--clip-id", clip_id]}
    for s in cfg_x["extract"]["stages"]:
        if current(done[s], h) and not force:
            print(f"[extract] {s}: cache current, skipped")
            continue
        ok, err = run(cmd[s])
        # C.5 exits 0 on a skipped clip: the cache record is the ground truth
        if not ok or not current(done[s], h):
            raise SystemExit(f"[extract] stage {s} produced no current output for "
                             f"{clip_id}: {err or 'see the stage log'} — a failed "
                             "gate is a finding; nothing packed (CLAUDE.md §3.4)")
        print(f"[extract] {s}: done")


def manifest(clip_id, raw, der, js) -> str:
    md = [f"# extract-v1 — {clip_id}", "", "## raw.npz", "",
          "| key | shape | dtype | meaning |", "|---|---|---|---|"]
    for k, v in raw.items():
        a = np.asarray(v)
        md.append(f"| `{k}` | `{'x'.join(map(str, a.shape)) or 'scalar'}` | {a.dtype} "
                  f"| {FIELD_DOC.get(k, '')} |")
    md += ["", "## derived.npz", "", "| key | shape | dtype |", "|---|---|---|"]
    for k, v in der.items():
        a = np.asarray(v)
        md.append(f"| `{k}` | `{'x'.join(map(str, a.shape)) or 'scalar'}` | {a.dtype} |")
    lab = raw["track_label"]
    md += ["", "## derived.json scalars", ""]
    md += [f"- `{k}` = {js[k]}" for k in ("delta_parallax_px", "rho", "rho_steps_used",
                                          "Z_p_lo", "Z_p_hi", "cam_path_length",
                                          "cam_path_over_Zbg")]
    md += ["", f"- tracks: {len(lab)} ({int((raw['track_source'] == 0).sum())} sparse, "
           f"{int((raw['track_source'] == 1).sum())} dense); labels "
           + ", ".join(f"{int(i)}: {int((lab == i).sum())}" for i in np.unique(lab)),
           f"- channel `{raw['channel']}`; on `branch` the subject label is Stage B's "
           "quiet blob (subject + glued halo) — `track_n_in_mask / track_n_valid` "
           "lets the scorer re-select without a threshold living here",
           "- background depth excludes only the subject mask; unmasked movers "
           "(labels >= 2, -1) are inside it"]
    return "\n".join(md) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="canonicalised video file")
    ap.add_argument("--fps", type=float, required=True, help="frames per second (dt = 1/fps)")
    ap.add_argument("--subject-text", default=None,
                    help="recorded in provenance; NOT used for masks in extract-v1")
    ap.add_argument("--pack-only", action="store_true", help="do not run stages, pack caches")
    ap.add_argument("--force", action="store_true", help="re-run every stage")
    ap.add_argument("--pass-tag", default=None,
                    help="second-pass mode (C4 noise floor): re-run every stage and write to "
                         "<out_dir>_<tag>/<clip_id>/; refuses to overwrite an existing tag. "
                         "Stage caches are scratch and ARE overwritten; pass 1's packed "
                         "output under <out_dir>/ is untouched")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--config-x", default="configs/extract.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg_x = load_cfg_x(cfg, args.config_x)
    video = Path(args.video)
    if not video.is_file():
        raise SystemExit(f"--video must be a video file, got {video}")
    clip_id = video.stem                       # run_stage_a keys caches by stem
    n, hw, fps_meta = probe_video(video)
    max_f = int(cfg["video"]["max_frames"])
    if n > max_f:
        raise SystemExit(f"{clip_id}: {n} frames > video.max_frames={max_f}; Stage A would "
                         "silently truncate — trim/resample upstream and record it")
    can = cfg_x["canonical"]
    frozen = all(can[k] is not None for k in ("frames", "height", "width"))
    if frozen and (n, hw[0], hw[1]) != (can["frames"], can["height"], can["width"]):
        raise SystemExit(f"{clip_id}: {n}x{hw[0]}x{hw[1]} is not the canonical "
                         f"{can['frames']}x{can['height']}x{can['width']} — normalise upstream")
    if fps_meta and abs(float(fps_meta) - args.fps) > 1e-3:
        print(f"[extract] note: --fps {args.fps} differs from container fps {fps_meta}; "
              "using --fps, both recorded")

    digest = sha256(video)
    out_root = Path(cfg_x["extract"]["out_dir"])
    force = args.force
    if args.pass_tag:
        out_root = out_root.with_name(f"{out_root.name}_{args.pass_tag}")
        force = True                               # a second pass must be a real second pass
        if (out_root / clip_id / "raw.npz").exists():
            raise SystemExit(f"pass tag {args.pass_tag!r} already exists for {clip_id}: "
                             f"{out_root / clip_id} — choose a new tag, never overwrite a pass")
    out = out_root / clip_id
    # the stage caches are keyed by file STEM: refuse to pack a different video's
    # caches under this name (generated videos must get unique file names)
    bprov = Path(cfg["paths"]["cache_dir"]) / clip_id / "provenance.json"
    if bprov.exists():
        prev = json.loads(bprov.read_text(encoding="utf-8")).get("clip_path")
        if prev and Path(prev).resolve() != video.resolve() and not force:
            raise SystemExit(f"clip id collision: cache '{clip_id}' was built from {prev}, "
                             f"not {video}. Rename the video or pass --force")
    rec = out / "raw.npz"
    if rec.exists() and not force:
        with np.load(rec) as z:
            old = json.loads(str(z["provenance_json"])).get("video_sha256")
        if old != digest:
            raise SystemExit(f"{rec} was extracted from a different video (sha {old[:12]}); "
                             "pass --force to re-run every stage on this one")

    if not args.pack_only:
        run_stages(cfg, cfg_x, video, clip_id, force)
    d = load(cfg, clip_id)

    root = Path(str(cfg["paths"]["drive_root"]))
    qc = {s: json.loads(p.read_text(encoding="utf-8")) for s, p in (
        ("B", Path(cfg["paths"]["stage_b_cache"]) / f"{clip_id}_qc_b.json"),
        ("C", Path(cfg["paths"]["stage_c_cache"]) / clip_id / "qc_c.json"),
        ("C5", root / "cache/stage_c5" / f"{clip_id}_qc_c5.json"))}
    cfg_c = yaml.safe_load(Path("configs/stage_c.yaml").read_text(encoding="utf-8"))
    prov = {
        "schema": pack.SCHEMA, "clip_id": clip_id, "video_path": str(video),
        "video_sha256": digest, "frames": n, "native_hw": list(hw),
        "fps": args.fps, "container_fps": fps_meta,
        "subject_text": args.subject_text,
        "mask_source": "Stage C auto (track-prompted SAM2); subject_text unused in v1",
        "canonical": can if frozen else "unfrozen",
        "repo_commit": git_head(),
        "config_hash_module1": cfg["_meta"]["config_hash"],
        "config_hash_extract": cfg_x["_hash"],
        "components": {"d4rt_commit": d["prov"].get("model_commit"),
                       "d4rt_checkpoint": d["prov"].get("checkpoint"),
                       "sam2_variant": cfg_c["sam2"]["variant"],
                       "gpu": d["prov"].get("gpu")},
        "gates": {"A": {k: g.get("pass") for k, g in d["qc"]["gates"].items()},
                  "B_follow_cam": (qc["B"].get("follow_cam") or {}).get("branch_active"),
                  "C": qc["C"].get("gates"), "C5": qc["C5"].get("gates")},
        "seed": "none exposed by stages A-C.5; run-to-run spread is measured by C4",
        "pass_tag": args.pass_tag,
    }
    raw = pack.build_raw(d, args.fps, json.dumps(prov, default=str))
    der, js = pack.build_derived(raw, cfg_x["derived"])
    js = {**js, "clip_id": clip_id, "video_sha256": digest,
          "config_hash_extract": cfg_x["_hash"],
          "config_hash_module1": cfg["_meta"]["config_hash"]}

    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "raw.npz", **raw)
    np.savez_compressed(out / "derived.npz", **der)
    (out / "derived.json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    (out / "manifest.md").write_text(manifest(clip_id, raw, der, js), encoding="utf-8")
    print(f"[extract] {clip_id}: delta_parallax={js['delta_parallax_px']:.2f}px "
          f"rho={js['rho']} tracks={raw['track_xy'].shape[0]} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
