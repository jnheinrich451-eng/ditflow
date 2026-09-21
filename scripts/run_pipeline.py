#!/usr/bin/env python
"""One-click pipeline over a curated tier (or explicit clips):
A (D4RT bundle) -> B (motion proposal) -> C (SAM2 mask) -> C.5 (dense
tracks) -> B.5 (heading d_src + scene up u_scene). Each stage runs as its
own subprocess per clip (models load/unload per call — the memory-disjoint
rule holds by construction), skips when its output already matches the
current config hash, and a failing stage marks the clip and moves on.
Writes <stage_c_viz>/../pipeline_summary.md.

    python scripts/run_pipeline.py --curated tier_a_good [--videos-dir D] [--force]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import yaml                                    # noqa: E402
from src.preprocess.config import load_config  # noqa: E402


def _json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def current(p, h):
    d = _json(p)
    return bool(d) and d.get("config_hash") == h


def run(cmd):
    r = subprocess.run([sys.executable] + cmd, capture_output=True, text=True)
    tail = (r.stderr or r.stdout).strip().splitlines()[-1:] if r.returncode else []
    return r.returncode == 0, (tail[0] if tail else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated", default="tier_a_good")
    ap.add_argument("--clip-id", nargs="*", default=[])
    ap.add_argument("--videos-dir", default="/content/data/videos")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stages", default="A,B,C,C5,B5")
    args = ap.parse_args()
    cfg = load_config("configs/stage_a.yaml")
    h = cfg["_meta"]["config_hash"]
    root = Path(str(cfg["paths"]["drive_root"]))
    clips = list(args.clip_id)
    if not clips:
        cur = yaml.safe_load(Path("configs/curated_clips.yaml").read_text(encoding="utf-8"))
        tiers = list(cur) if args.curated == "all" else [args.curated]
        clips = [c for t in tiers for c in cur[t]]
    stages = [s.strip() for s in args.stages.split(",")]
    P = {"A": lambda c: root / "cache/stage_a" / c / "provenance.json",
         "B": lambda c: root / "cache/stage_b" / f"{c}_qc_b.json",
         "C": lambda c: root / "cache/stage_c" / c / "qc_c.json",
         "C5": lambda c: root / "cache/stage_c5" / f"{c}_qc_c5.json",
         "B5": lambda c: root / "cache/stage_b5" / f"{c}_qc_b5.json"}
    CMD = {"A": lambda c: ["scripts/run_stage_a.py", "--clip",
                           str(Path(args.videos_dir) / f"{c}.mp4")],
           "B": lambda c: ["scripts/run_stage_b.py", "--clip-id", c],
           "C": lambda c: ["scripts/run_stage_c.py", "--clips", c],
           "C5": lambda c: ["scripts/run_stage_c5.py", "--clip-id", c],
           "B5": lambda c: ["scripts/run_stage_b5.py", "--clip-id", c]}
    md = ["# Pipeline summary\n",
          "| clip | A | B ch | C variant/inband/abs | C5 n / G2b / purity "
          "| B5 heading / sigma_theta / up / kappa | status |",
          "|---|---|---|---|---|---|---|"]
    for c in clips:
        status = "ok"
        for s in stages:
            done = (P[s](c).exists() if s == "B5" else current(P[s](c), h))
            if done and not args.force:
                continue
            ok, err = run(CMD[s](c))
            if not ok:
                status = f"FAILED at {s}: {err[:90]}"
                break
            print(f"{c}: {s} done")
        a = "ok" if (root / "cache/stage_a" / c / "bundle.npz").exists() else "-"
        qb = _json(P["B"](c)) or {}
        fc = qb.get("follow_cam") or {}
        ch = "branch" if fc.get("branch_active") else ("x0" if qb else "-")
        qc = _json(P["C"](c)) or {}
        g = qc.get("gates", {})
        cc = (f"{qc.get('chosen_variant')}/{g.get('inband')}/{g.get('c_abs_pass')}"
              if qc else "-")
        q5 = _json(P["C5"](c)) or {}
        g5 = q5.get("gates", {})
        c5 = (f"{q5.get('n_dense')} / {g5.get('C5-G2b', {}).get('px')}px / "
              f"{g5.get('C5-PURITY', {}).get('med')}" if q5 else "-")
        q6 = _json(P["B5"](c)) or {}
        b5 = (f"{'def' if q6.get('heading_defined') else 'undef'} / "
              f"{q6.get('sigma_theta', float('nan')):.3f} / {q6.get('up_source')} / "
              f"{q6.get('kappa_basis', float('nan')):.2f}" if q6 else "-")
        md.append(f"| {c} | {a} | {ch} | {cc} | {c5} | {b5} | {status} |")
        print(md[-1])
    out = root / "viz" / "pipeline_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nsummary -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
