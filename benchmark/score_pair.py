"""One table row: score a generated video against its reference, both already run through
scripts/extract.py. Camera and object, ours and the literature's definitions side by side.

    python benchmark/score_pair.py --ref <extract-v1>/<ref_clip> --gen <extract-v1>/<gen_id> \
        --label ditflow_seed0 --idx-ref 0:21 --idx-gen 0:21 --out results/pairs.csv

--idx-ref / --idx-gen are the frame correspondence (python slice "a:b[:step]" or a comma list);
default is the identity over the shorter video. Appends one row to --out and prints it.

Columns
  CamRotErr_deg       ours, primary: geodesic rotation error, both anchored at frame 0, scale-only, mean per frame
  RPE_rot_deg         alignment-free relative rotation error per frame (robust on a near-static camera)
  ATE_sim3            sim(3)-aligned centre error, reference units (Q1 primary endpoint)
  CC_RotErr_rad_sum   CameraCtrl Eq.4 as written (SymphoMotion's CamRotErr): sum over frames, radians
  CC_TransErr_sq_sum  CameraCtrl Eq.5 + App. D.5 (SymphoMotion's CamTransErr): sum of squared L2, first-two-frame rescale
  BoxIoU              SymphoMotion's object metric, from the extractor's subject masks, higher better
  ObjMC_px            MotionCtrl's object metric: mean centroid distance in px (camera-entangled)
  ObjErr3D            ours: camera-compensated world-frame root error / ref background depth (unitless)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import camera_metrics as cm                           # noqa: E402
from benchmark import object_metrics as om                           # noqa: E402


def parse_idx(s: str | None, n: int) -> np.ndarray:
    if s is None:
        return np.arange(n)
    if ":" in s:
        parts = [int(p) if p else None for p in s.split(":")]
        return np.arange(n)[slice(*parts)]
    return np.array([int(v) for v in s.split(",")])


def score(ref: Path, gen: Path, idx_ref, idx_gen) -> dict:
    with np.load(ref / "raw.npz") as z:
        E_r = z["cam_extrinsics"].astype(np.float64)
    with np.load(gen / "raw.npz") as z:
        E_g = z["cam_extrinsics"].astype(np.float64)
    Er, Eg = cm.to_camera0(E_r[idx_ref]), cm.to_camera0(E_g[idx_gen])
    so = cm.camera_errors(Eg, Er, align="scale_only")
    s3 = cm.camera_errors(Eg, Er, align="sim3")
    cc = cm.cameractrl_errors(Eg, Er)
    ob = om.object_errors(ref, gen, idx_ref, idx_gen, cam_scale=so["scale"])
    return {
        "n_frames": so["n"],
        "CamRotErr_deg": so["rot_mean_deg"], "RPE_rot_deg": so["rpe_rot_deg"],
        "ATE_sim3": s3["ate"], "cam_scale": so["scale"],
        "CC_RotErr_rad_sum": cc["RotErr_sum_rad"], "CC_TransErr_sq_sum": cc["TransErr_sum_sq"],
        "CC_rescale": cc["rescale"],
        "BoxIoU": ob["box_iou"], "ObjMC_px": ob["objmc_px"], "ObjErr3D": ob["obj_err3d"],
        "box_frames_missing": ob["box_frames_missing"], "ref_path_len_3d": ob["ref_path_len_3d"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, type=Path)
    ap.add_argument("--gen", required=True, type=Path)
    ap.add_argument("--label", required=True)
    ap.add_argument("--idx-ref", default=None)
    ap.add_argument("--idx-gen", default=None)
    ap.add_argument("--out", default="results/pairs.csv", type=Path)
    a = ap.parse_args()
    with np.load(a.ref / "raw.npz") as z:
        n_r = len(z["cam_extrinsics"])
    with np.load(a.gen / "raw.npz") as z:
        n_g = len(z["cam_extrinsics"])
    n = min(n_r, n_g)
    ir, ig = parse_idx(a.idx_ref, n_r)[:n] if a.idx_ref else np.arange(n), \
             parse_idx(a.idx_gen, n_g)[:n] if a.idx_gen else np.arange(n)
    row = {"label": a.label, "ref": a.ref.name, "gen": a.gen.name} | score(a.ref, a.gen, ir, ig)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    new = not a.out.exists()
    with open(a.out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    for k, v in row.items():
        print(f"{k:20s} {v:.4f}" if isinstance(v, float) else f"{k:20s} {v}")


if __name__ == "__main__":
    main()
