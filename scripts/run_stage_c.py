#!/usr/bin/env python
"""Stage C runner (SKILL §2.3): variant family -> SAM2 -> gate referee ->
QC re-prompt -> presence -> finalized bundle + calibration table.

  --calibrate   provisional gate values from configs/stage_c.yaml, full
                stats table for the human freeze (writes no thresholds)
  default       production: requires c_* lock thresholds

    python scripts/run_stage_c.py --calibrate [--clips ID ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import yaml                                                # noqa: E402
from src.preprocess.bundle import read_bundle              # noqa: E402
from src.preprocess.config import load_config              # noqa: E402
from src.preprocess.stage_b.scoring import eta_weights     # noqa: E402
from src.preprocess.stage_c import finalize, mask as maskmod, presence, prompts  # noqa: E402
from probe_sam2 import _extract_frames                     # noqa: E402

C_GATE_KEYS = ("c_tau_iou", "c_size_lo", "c_size_hi", "c_tau_temp",
               "c_abs_area_max")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--config-c", default="configs/stage_c.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg_c = yaml.safe_load(Path(args.config_c).read_text(encoding="utf-8"))
    th = {k: cfg["thresholds"].get(k) for k in C_GATE_KEYS}
    if args.calibrate:
        sel = cfg_c["select"]
        th = {"c_tau_iou": th["c_tau_iou"] or sel["tau_iou_provisional"],
              "c_size_lo": th["c_size_lo"] or sel["area_hull_lo"],
              "c_size_hi": th["c_size_hi"] or sel["area_hull_hi"],
              "c_tau_temp": th["c_tau_temp"] or sel["tau_temp_provisional"],
              "c_abs_area_max": th["c_abs_area_max"] or sel["abs_area_max"]}
    elif any(v is None for v in th.values()):
        raise SystemExit(f"production refused: unset lock thresholds "
                         f"{[k for k, v in th.items() if v is None]}")
    clips = args.clips or cfg_c["smoke"]["clips"]
    Hp = int(cfg["video"]["process_height"])
    viz = Path(cfg["paths"]["stage_c_viz"])
    viz.mkdir(parents=True, exist_ok=True)
    runner = maskmod.Sam2Runner(cfg_c["sam2"]["variant"])

    md = [f"# Stage C {'calibration' if args.calibrate else 'production'} "
          f"table\n", f"Config hash `{cfg['_meta']['config_hash']}`; "
          f"thresholds {th}\n",
          "| clip | channel | variant | inband | t_c | med IoU | area/hull | "
          "temp | reprompts | present | poor |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for clip_id in clips:
        # tolerate video FILENAMES (the natural paste): strip .mp4 if the
        # extension-less bundle exists; otherwise skip with a did-you-mean
        # instead of killing the run (mirrors run_stage_b)
        cdir = Path(cfg["paths"]["cache_dir"])
        if not (cdir / clip_id).exists():
            alt = clip_id[:-4] if clip_id.endswith(".mp4") else None
            if alt and (cdir / alt).exists():
                print(f"note: '{clip_id}' -> '{alt}' (video filename given)")
                clip_id = alt
            else:
                import difflib
                known = sorted(d.name for d in cdir.glob("*") if d.is_dir())
                near = difflib.get_close_matches(clip_id, known, n=1)
                msg = (f"SKIP {clip_id}: no Stage A bundle"
                       + (f" — did you mean '{near[0]}'?" if near else ""))
                md.append(f"| {clip_id} | — | SKIPPED | | | | | | | | |")
                print(msg)
                continue
        b = read_bundle(cfg, clip_id)
        Q, P = b["Q"], b["P"]
        F = Q.shape[1]
        bdir = Path(cfg["paths"]["stage_b_cache"])
        qc_b = json.loads((bdir / f"{clip_id}_qc_b.json").read_text(encoding="utf-8"))
        with np.load(bdir / f"{clip_id}.npz") as z:
            sb = {k: z[k] for k in z.files}
        eta = eta_weights(b["X0"], b["X_local"], b["G"], Q)
        t_c, core, variants, prov = prompts.build_variants(b, sb, qc_b,
                                                           cfg_c, eta)
        fdir = Path(cfg_c["smoke"]["frames_dir"]) / clip_id
        n_fr, (Hn, Wn) = _extract_frames(b["provenance"]["clip_path"], F, fdir)
        scale = np.array([Wn / Hp, Hn / Hp])

        results, states = {}, {}
        for name, (pts, labels, box) in variants.items():
            box_n = None if box is None else [box[0] * scale[0],
                                              box[1] * scale[1],
                                              box[2] * scale[0],
                                              box[3] * scale[1]]
            m, st = runner.run(fdir, n_fr, t_c, pts * scale, labels, box_n)
            results[name] = m
            states[name] = (st, pts * scale, labels)
        stats = {n: maskmod.score_masks(m, core, P, Q, scale, n_fr)
                 for n, m in results.items()}
        chosen, inband = maskmod.referee(stats, th)
        masks = results[chosen]
        st, pts_n, labels = states[chosen]

        # §4 QC re-prompt on failing frames (bounded, logged)
        rlog = []
        fails = [t for t in range(n_fr)
                 if t in masks and len(core[Q[core, t]]) >= 3
                 and np.nan_to_num(maskmod.hull_iou(
                     masks[t], P[core[Q[core, t]], t] * scale)) < float(th["c_tau_iou"])]
        for t in fails[:int(cfg_c["reprompt"]["max_events"])]:
            mem = core[Q[core, t]]
            pp = P[mem, t].astype(np.float64) * scale
            before = maskmod.hull_iou(masks[t], pp)
            masks = runner.reprompt(st, masks, t, pp,
                                    np.ones(len(pp), np.int32))
            rlog.append({"t": int(t), "iou_before": round(float(before), 3),
                         "iou_after": round(float(
                             maskmod.hull_iou(masks[t], pp)), 3)})

        pres, masks = presence.presence(masks, core, Q, n_fr,
                                        cfg_c["prompts"]["n_pos_min"])
        abs_pass = bool(stats[chosen]["area_med"]
                        <= float(th["c_abs_area_max"]))
        qc_c = finalize.write_stage_c(cfg, clip_id, masks, pres, t_c, prov,
                                      {**stats[chosen], "inband": inband,
                                       "c_abs_pass": abs_pass,
                                       "all_variants": stats},
                                      rlog, chosen,
                                      b["provenance"].get("config_hash"))
        s = stats[chosen]
        md.append(f"| {clip_id} | {prov['channel']} | {chosen} | {inband} "
                  f"| {t_c} | {s['med_iou']:.3f} | {s['area_hull_ratio']:.2f} "
                  f"| {s['temp_med']:.2f} | {len(rlog)} | {int(pres.sum())} "
                  f"| {prov['prompt_poor']} |")
        print(md[-1])

        # review package: mask overlay video (§8)
        import cv2
        import imageio.v2 as iio
        with iio.get_writer(viz / f"mask_{clip_id}.mp4", fps=12) as w:
            for t in range(n_fr):
                img = iio.imread(fdir / f"{t:05d}.jpg").copy()
                if t in masks and masks[t].any():
                    img[masks[t]] = (0.55 * img[masks[t]]
                                     + 0.45 * np.array([255, 210, 0])).astype(np.uint8)
                w.append_data(cv2.resize(img, (Wn // 2, Hn // 2)))

    out = viz / ("calibration_stage_c.md" if args.calibrate else "run_stage_c.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
