#!/usr/bin/env python
"""Stage B runner: Stage A bundle -> motion proposal (§9 contract).

Modes:
  default      production run — refuses null lock thresholds (CLAUDE.md §4)
  --calibrate  calibration run: sweeps tau_z candidates (configs/stage_b.yaml),
               uses DATA-DERIVED provisional clustering thresholds (formulas
               documented below, reported in the table — the human freezes the
               lock from the evidence; this script writes no thresholds)
  --validate   checks fitted a_t against the archived A.5 sigma_t oracle on
               the calibration clips (SKILL §6) — READ-ONLY use of the record

CPU-only, seconds per clip. Window boundaries come from qc.json ONLY (§10.3).

    python scripts/run_stage_b.py --clip-id <id> [--calibrate | --validate]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                                        # noqa: E402
from src.preprocess.bundle import read_bundle                      # noqa: E402
from src.preprocess.config import load_config, load_stage_a5       # noqa: E402
from src.preprocess.stage_b import follow_cam, static_fallback     # noqa: E402
from src.preprocess.stage_b.clustering import cluster_tracks       # noqa: E402
from src.preprocess.stage_b.drift import two_pass                  # noqa: E402
from src.preprocess.stage_b.scoring import eta_weights, pair_weights  # noqa: E402
from src.preprocess.stage_b.select import score_clusters           # noqa: E402

GATE_KEYS = ("rho", "tau_b", "tau_z", "tau_r", "tau_d", "tau_x",
             "f_min", "n_min", "lambda_1", "lambda_2", "lambda_3")
# Adaptive lock alternatives (threshold proposal 2026-08-09): the lock may
# freeze a clip-adaptive multiplier instead of an absolute value — the
# measured noise floors differ per clip (car tau_r 0.0104-0.0130 vs horse
# 0.0087-0.0123), so absolutes cannot serve both (CLAUDE.md §4).
ADAPTIVE_ALTS = {"tau_r": "tau_r_floor_mult", "tau_x": "tau_x_depth_frac"}
# Follow-cam gate keys (SKILL §12.4): lock values override the provisional
# follow_cam: config block when present.
FC_LOCK_KEYS = {"fc_quiet_frac": "quiet_frac", "fc_min_quiet": "min_quiet",
                "fc_compact_frac": "compact_frac",
                "fc_travel_floor_mult": "travel_floor_mult",
                "fc_min_winner_frac": "min_winner_frac",
                "fc_min_winner_blob_overlap": "min_winner_blob_overlap",
                "fc_min_winner_core_overlap": "min_winner_core_overlap"}


def load_cfg_b(path="configs/stage_b.yaml"):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def provisional_cluster_th(b, res, th_base, cfg_b):
    """DATA-DERIVED provisional clustering thresholds for calibration runs
    only — documented formulas, reported alongside results; never written to
    the lock. Revision 2026-08-09, from the measured diagnosis:
    - tau_r is anchored to the STATIC-PAIR NOISE FLOOR on DE-GAUGED X0
      (floor_mult x median static-pair e_ij) — the earlier self-referential
      Q60-of-local-pairs landed BELOW the noise floor on both clips (car
      0.0026 vs floor 0.0079), guaranteeing zero edges.
    - tau_x from scene scale (depth fraction), raised: 0.25 x Zmed measured
      smaller than the car itself.
    - tau_d provisional lowered per measured within-object segment cosines
      (horse torso-torso median 0.25 at tau_d=0.5 -> no same-object edges)."""
    from src.preprocess.stage_b.drift import gauge_curve
    cal = cfg_b["calibration"]
    th = dict(th_base)
    Q = b["Q"]
    Zmed = float(np.median(b["X_local"][..., 2][Q]))
    th.setdefault("tau_x", float(cal.get("tau_x_depth_frac", 0.6)) * Zmed)

    g = gauge_curve(res["a"], Q.shape[1])
    X0d = b["X0"].astype(np.float64) / g[None, :, None]
    rng = np.random.default_rng(0)
    stat = np.flatnonzero(res["labeled"] & ~res["y_dyn"])
    if len(stat) >= 20 and "tau_r" not in th:
        idx = rng.choice(stat, min(150, len(stat)), replace=False)
        Xs = X0d[idx]
        Qs = Q[idx]
        vp = Qs[:, None, :] & Qs[None, :, :]
        d = np.linalg.norm(Xs[:, None] - Xs[None, :], axis=-1)
        dd = np.abs(np.diff(d, axis=2))
        dd = np.where(vp[:, :, 1:] & vp[:, :, :-1], dd, np.nan)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            e = np.nanmedian(dd, axis=2)
        ev = e[np.triu_indices(len(idx), 1)]
        ev = ev[np.isfinite(ev)]
        if ev.size >= 50:
            th["tau_r"] = float(cal.get("tau_r_floor_mult", 1.5)
                                * np.median(ev))
    th.setdefault("tau_r", 0.1)
    th.setdefault("tau_d", float(cal.get("tau_d_provisional", 0.15)))
    th.setdefault("f_min", max(20, Q.shape[1] // 4))
    th.setdefault("n_min", 10)
    return th


def winner_blob_overlap(cid, o_src, fc, core=False):
    """Fraction of the X0 winner's members inside the detected quiet blob
    (core=True: inside the blob CORE — arbitration v3). None when there is
    nothing to compare."""
    if o_src is None or fc is None or not fc.get("fired"):
        return None
    win = np.flatnonzero(cid == o_src)
    if not len(win):
        return None
    key = "blob_core_idx" if (core and "blob_core_idx" in fc) else "blob_idx"
    blob = set(int(x) for x in fc[key])
    return float(np.mean([int(i) in blob for i in win]))


def starved(o_src, rows, res, cfg_b, cid=None, fc=None):
    """SKILL §12 starvation predicate v2 (approved 2026-08-10): when the
    detector fired, the X0 path owns the clip only if its winner covers the
    DETECTED SUBJECT — winner∩blob overlap >= min_winner_blob_overlap. The
    v1 size ratio failed on 000000000284.0.006: a 6.8% mislabeled-background
    winner cleared the 5% bar and blocked the branch (measured overlaps:
    horse 0.23 = subject, car/284 both 0.00 = artifact; gate 0.1 sits in the
    empty zone, and the articulated-subject failure direction is safe — it
    keeps the X0 path). Size arm retained only when the detector is silent."""
    if o_src is None:
        return True
    # v3 (approved 2026-08-21): overlap vs blob CORE — a halo-contaminated
    # winner touches the blob (24.0.006 whole-blob 0.197, 299.1.006 even
    # 1.00) but not its subject-pure core (both <= 0.044); legit laterals
    # keep core overlap >= 0.103. Whole-blob path kept only for legacy fc
    # records lacking blob_core_idx.
    ovl_c = winner_blob_overlap(cid, o_src, fc, core=True)
    if ovl_c is not None and fc is not None and "blob_core_idx" in fc:
        return ovl_c < float(cfg_b["follow_cam"].get(
            "min_winner_core_overlap", 0.07))
    ovl = winner_blob_overlap(cid, o_src, fc)
    if ovl is not None:
        return ovl < float(cfg_b["follow_cam"].get("min_winner_blob_overlap",
                                                   0.1))
    size = next((r["size"] for r in rows if r["cluster"] == o_src), 0)
    n_dyn = int(res["y_dyn"].sum())
    return size < float(cfg_b["follow_cam"]["min_winner_frac"]) * max(n_dyn, 1)


def run_clip(cfg, cfg_b, clip_id, thresholds, lambdas):
    b = read_bundle(cfg, clip_id)
    boundaries = b["qc"].get("window_boundaries")
    assert boundaries is not None, \
        f"{clip_id}: qc.json lacks window_boundaries — run refresh_qc_boundaries.py (Task 0)"
    eta = eta_weights(b["X0"], b["X_local"], b["G"], b["Q"])
    w = pair_weights(eta, b["Q"])
    Hp = int(cfg["video"]["process_height"])
    ctx = {"anchors": b["anchors"], "process_hw": (Hp, Hp)}
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, boundaries, cfg_b,
                   thresholds, static_fallback.select, ctx)
    cid, cdiag = cluster_tracks(b["X0"], res["r"], b["Q"], b["anchors"],
                                res["y_dyn"], boundaries, cfg_b, thresholds,
                                a=res["a"], eta=eta)
    o_src, rows = score_clusters(cid, res["m"], b["Q"], lambdas,
                                 thresholds["n_min"], eta=eta)
    return b, res, cid, cdiag, o_src, rows, boundaries


def write_outputs(cfg, cfg_b, clip_id, b, res, cid, o_src, rows, boundaries,
                  thresholds, tag="", cdiag=None, fc=None, branch_active=False,
                  winner_ovl=None, winner_ovl_core=None):
    out = Path(cfg["paths"]["stage_b_cache"])
    out.mkdir(parents=True, exist_ok=True)
    stem = clip_id + (f"__{tag}" if tag else "")
    o_members = np.flatnonzero(cid == o_src) if o_src is not None else np.array([], int)
    extra = {}
    if cdiag is not None and "posthoc_mask" in cdiag:
        cdiag = dict(cdiag)
        extra["posthoc_mask"] = cdiag.pop("posthoc_mask")
        extra["posthoc_votes"] = cdiag.pop("posthoc_votes")
    if fc is not None and fc.get("fired"):
        extra.update({"fc_blob_idx": fc["blob_idx"].astype(np.int32),
                      "fc_blob_core_idx": np.asarray(
                          fc.get("blob_core_idx", fc["blob_idx"]),
                          np.int32),
                      "fc_R_rel": fc["R_rel"], "fc_tau_rel": fc["tau_rel"],
                      "fc_rel_valid": fc["rel_valid"],
                      "fc_seed_points": fc["seed_points"]})
    np.savez_compressed(out / f"{stem}.npz",
                        y_dyn=res["y_dyn"], m=res["m"].astype(np.float32),
                        cluster_id=cid.astype(np.int32),
                        o_src_members=o_members.astype(np.int32),
                        a=res["a"].astype(np.float32),
                        b=res["b"].astype(np.float32), **extra)
    qc_b = {"clip_id": clip_id, "tag": tag,
            "follow_cam": None if fc is None else {
                **{k: fc.get(k) for k in
                   ("fired", "n_quiet", "n_blob", "med_cs", "quiet_cs",
                    "world_floor", "cam_step_med", "step_over_floor",
                    "compact_med", "Zmed", "seed_frame", "seed_bbox")},
                "winner_blob_overlap": winner_ovl,
                "winner_core_overlap": winner_ovl_core,
                "branch_active": bool(branch_active)},
            "rho_pass1": res["rho_pass1"],
            "b_med_pass1": res["b_med_pass1"], "b_med_pass2": res["b_med_pass2"],
            "used_fallback": bool(res["used_fallback"]),
            "n_dyn": int(res["y_dyn"].sum()),
            "n_labeled": int(res["labeled"].sum()),
            "o_src": None if o_src is None else int(o_src),
            "clusters": rows, "clustering_diag": cdiag,
            "segment_scales": res["segment_scales"],
            "window_boundaries": [int(x) for x in boundaries],
            "thresholds_used": {k: thresholds.get(k) for k in GATE_KEYS},
            "config_hash": cfg["_meta"]["config_hash"],
            "upstream_bundle_hash": b["provenance"].get("config_hash")}
    (out / f"{stem}_qc_b.json").write_text(json.dumps(qc_b, indent=2),
                                           encoding="utf-8")
    return qc_b


def validate_against_sigma(cfg, clip_id, res, boundaries):
    """SKILL §6: characterize fitted a_t against the archived sigma_t curve.
    For static points a_t ~= -d(sigma)/sigma (appendix 12.2 sign note)."""
    a5 = load_stage_a5()
    rec_path = Path(a5["paths"]["out_dir"]) / "calibration_record.json"
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if clip_id not in rec:
        return {"note": f"{clip_id} not in calibration record — skipped"}
    sig = np.array([np.nan if v is None else v for v in rec[clip_id]["sigma_abs"]],
                   float)
    a = res["a"]
    T = min(len(sig), len(a))
    pred = -np.diff(sig[:T]) / sig[:T - 1]          # expected a_t at frame t=1..
    fit = a[1:T]
    tm = np.ones(T - 1, bool)
    for bd in boundaries:
        for k in (bd - 1, bd, bd + 1):
            if 0 <= k - 1 < T - 1:
                tm[k - 1] = False
    ok = tm & np.isfinite(pred) & np.isfinite(fit)
    corr = float(np.corrcoef(fit[ok], pred[ok])[0, 1]) if ok.sum() > 4 else float("nan")
    jumps = []
    for bd in boundaries:
        if bd < T and np.isfinite(a[bd]) and np.isfinite(pred[bd - 1]):
            jumps.append({"t": int(bd), "fitted_a": round(float(a[bd]), 4),
                          "sigma_step": round(float(pred[bd - 1]), 4)})
    return {"corr_nonboundary": corr,
            "fitted_a_median_abs": float(np.nanmedian(np.abs(fit[ok]))),
            "sigma_pred_median_abs": float(np.nanmedian(np.abs(pred[ok]))),
            "jumps": jumps, "n_frames_compared": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-id", nargs="*", default=None)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--config", default="configs/stage_a.yaml")
    ap.add_argument("--config-b", default="configs/stage_b.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg_b = load_cfg_b(args.config_b)
    th_lock = {k: cfg["thresholds"].get(k) for k in GATE_KEYS}

    clips = args.clip_id or sorted(
        d.name for d in Path(cfg["paths"]["cache_dir"]).glob("*")
        if (d / "bundle.npz").exists())
    viz = Path(cfg["paths"]["stage_b_viz"])
    viz.mkdir(parents=True, exist_ok=True)

    # lock fc_* keys override the provisional follow_cam config (both modes)
    for lk, ck in FC_LOCK_KEYS.items():
        if cfg["thresholds"].get(lk) is not None:
            cfg_b["follow_cam"][ck] = float(cfg["thresholds"][lk])
    # lock adaptive multipliers override the calibration formula parameters
    for alt in ADAPTIVE_ALTS.values():
        if cfg["thresholds"].get(alt) is not None:
            cfg_b["calibration"][alt] = float(cfg["thresholds"][alt])

    if not args.calibrate:
        missing = [k for k in GATE_KEYS
                   if th_lock.get(k) is None
                   and cfg["thresholds"].get(ADAPTIVE_ALTS.get(k, "")) is None]
        if missing:
            raise SystemExit(
                f"production run refused: lock thresholds unset: {missing} — "
                "run --calibrate, review the table, freeze the lock (CLAUDE.md §4)")

    report = ["# Stage B run report\n",
              f"Config hash `{cfg['_meta']['config_hash']}` · mode: "
              + ("calibrate" if args.calibrate else "production")
              + (" + validate" if args.validate else "") + "\n"]
    skipped = []
    known = sorted(d.name for d in Path(cfg["paths"]["cache_dir"]).glob("*")
                   if d.is_dir())
    for clip_id in clips:
        # A stale bundle is refused per clip (CLAUDE.md §4), never allowed to
        # kill the batch: orphans from older config eras get listed + skipped.
        # A nonexistent id (typo) gets a did-you-mean instead of "malformed".
        if not (Path(cfg["paths"]["cache_dir"]) / clip_id).exists():
            import difflib
            near = difflib.get_close_matches(clip_id, known, n=1)
            msg = (f"SKIP {clip_id}: no such bundle"
                   + (f" — did you mean '{near[0]}'?" if near else
                      f"; known ids: {known}"))
            skipped.append(clip_id)
            report.append(f"\n- {msg}")
            print(msg)
            continue
        try:
            read_bundle(cfg, clip_id)
        except (RuntimeError, AssertionError) as e:
            skipped.append(clip_id)
            msg = f"SKIP {clip_id}: {e}"
            report.append(f"\n- {msg}")
            print(msg)
            continue
        if args.calibrate:
            lam = cfg_b["calibration"]["lambda_defaults"]
            report.append(f"\n## {clip_id}\n")
            b0 = read_bundle(cfg, clip_id)
            fc = follow_cam.detect(b0["X_local"], b0["X0"], b0["G"], b0["Q"],
                                   b0["P"], b0["anchors"],
                                   b0["qc"]["window_boundaries"], cfg_b)
            report.append(
                f"- follow-cam detector: fired={fc['fired']} "
                f"n_quiet={fc['n_quiet']} n_blob={fc['n_blob']} "
                f"step/floor={fc['step_over_floor']} "
                f"compact={fc['compact_med']} "
                f"seed_frame={fc.get('seed_frame')} "
                f"seed_bbox={[None if v is None else round(v, 1) for v in fc['seed_bbox']] if fc.get('seed_bbox') else None}")
            print(report[-1])
            for tau_z in cfg_b["calibration"]["tau_z_candidates"]:
                th = {**{k: v for k, v in th_lock.items() if v is not None},
                      "tau_z": float(tau_z)}
                th.setdefault("rho", 0.4)
                b, res, cid, cdiag, o_src, rows, bd = run_clip(
                    cfg, cfg_b, clip_id, provisional_cluster_th_wrap(cfg, cfg_b, clip_id, th),
                    lam)
                active = bool(starved(o_src, rows, res, cfg_b, cid, fc)
                              and fc["fired"])
                ovl = winner_blob_overlap(cid, o_src, fc)
                ovl_c = winner_blob_overlap(cid, o_src, fc, core=True)
                qc_b = write_outputs(cfg, cfg_b, clip_id, b, res, cid, o_src,
                                     rows, bd, th, tag=f"tz{tau_z}",
                                     cdiag=cdiag, fc=fc, branch_active=active,
                                     winner_ovl=ovl, winner_ovl_core=ovl_c)
                # diagnostic: dynamic rate by spawn segment — the measured
                # late-window false-positive signature shows up here
                s_i = b["anchors"][:, 2].astype(int)
                seg_of = np.zeros(b["Q"].shape[1], int)
                for x in bd:
                    seg_of[int(x):] += 1
                segs = seg_of[np.clip(s_i, 0, len(seg_of) - 1)]
                lab = res["labeled"]
                rates = [round(float(res["y_dyn"][lab & (segs == s)].mean()), 2)
                         if (lab & (segs == s)).sum() else None
                         for s in range(seg_of.max() + 1)]
                report.append(
                    f"- tau_z={tau_z}: rho1={qc_b['rho_pass1']:.3f} "
                    f"b_med1={qc_b['b_med_pass1']:.4f} n_dyn={qc_b['n_dyn']}"
                    f"/{qc_b['n_labeled']} clusters={len(rows)} "
                    f"O_src={qc_b['o_src']} "
                    f"(size {next((r['size'] for r in rows if r['cluster'] == qc_b['o_src']), 0)}) "
                    f"fallback={qc_b['used_fallback']} "
                    f"fc-branch={active} "
                    f"ovl={'-' if ovl is None else round(ovl, 2)} "
                    f"dyn-rate-by-spawn-seg={rates}")
                print(report[-1])
                ed = (cdiag or {}).get("edge_diag")
                if ed:
                    report.append(
                        f"  - edges: rig {ed['rigidity']} dir {ed['direction']} "
                        f"prox {ed['proximity']} ovl {ed['overlap']} "
                        f"joint {ed['joint']} | moving {ed['moving_frac']} "
                        f"floor {ed['speed_floor']} | tau {ed['tau_used']}")
                    print(report[-1])
        else:
            th = {k: v for k, v in th_lock.items() if v is not None}
            if "tau_r" not in th or "tau_x" not in th:
                # adaptive lock keys: compute the per-clip values with the
                # LOCKED multiplier (overlaid above), documented formulas
                th = provisional_cluster_th_wrap(cfg, cfg_b, clip_id, th)
            b, res, cid, cdiag, o_src, rows, bd = run_clip(
                cfg, cfg_b, clip_id, th,
                (th["lambda_1"], th["lambda_2"], th["lambda_3"]))
            fc = follow_cam.detect(b["X_local"], b["X0"], b["G"], b["Q"],
                                   b["P"], b["anchors"], bd, cfg_b)
            active = bool(starved(o_src, rows, res, cfg_b, cid, fc)
                          and fc["fired"])
            ovl = winner_blob_overlap(cid, o_src, fc)
            ovl_c = winner_blob_overlap(cid, o_src, fc, core=True)
            qc_b = write_outputs(cfg, cfg_b, clip_id, b, res, cid, o_src, rows,
                                 bd, th, cdiag=cdiag, fc=fc,
                                 branch_active=active, winner_ovl=ovl,
                                 winner_ovl_core=ovl_c)
            report.append(f"\n## {clip_id}: n_dyn={qc_b['n_dyn']} "
                          f"O_src={qc_b['o_src']} fallback={qc_b['used_fallback']} "
                          f"fc-branch={active} (fired={fc['fired']}, "
                          f"n_blob={fc['n_blob']}, "
                          f"ovl={'-' if ovl is None else round(ovl, 2)}, "
                          f"ovlc={'-' if ovl_c is None else round(ovl_c, 2)})")
            print(report[-1])
        if args.validate:
            _, res_v, _, _, _, _, bd = run_clip(
                cfg, cfg_b, clip_id,
                provisional_cluster_th_wrap(cfg, cfg_b, clip_id,
                                            {"rho": th_lock.get("rho") or 0.4,
                                             "tau_b": th_lock.get("tau_b"),
                                             "tau_z": None}),
                cfg_b["calibration"]["lambda_defaults"])
            v = validate_against_sigma(cfg, clip_id, res_v, bd)
            report.append(f"- sigma_t oracle: {json.dumps(v)}")
            print(report[-1])

    if skipped:
        report.append(f"\n\n**Skipped (stale/malformed): {skipped}** — re-run "
                      "Stage A on them to include, or delete the orphan dirs.")
    out = viz / ("calibration_stage_b.md" if args.calibrate else "run_stage_b.md")
    out.write_text("\n".join(report), encoding="utf-8")
    print(f"\nreport -> {out}")
    return 0


def provisional_cluster_th_wrap(cfg, cfg_b, clip_id, th):
    """Fill clustering thresholds for calibration/validation runs from data-
    derived provisional formulas (documented in provisional_cluster_th)."""
    b = read_bundle(cfg, clip_id)
    boundaries = b["qc"]["window_boundaries"]
    eta = eta_weights(b["X0"], b["X_local"], b["G"], b["Q"])
    w = pair_weights(eta, b["Q"])
    Hp = int(cfg["video"]["process_height"])
    res = two_pass(b["X0"], b["X_local"], b["Q"], w, boundaries, cfg_b, th,
                   static_fallback.select,
                   {"anchors": b["anchors"], "process_hw": (Hp, Hp)})
    return provisional_cluster_th(b, res, th, cfg_b)


if __name__ == "__main__":
    sys.exit(main())
