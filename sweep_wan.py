"""Grid sweep over DiTFlow-Wan guidance hyperparameters, scored with MF and CLIP.

Runs `motion_guidance_wan.py` once per grid point, then evaluates each output
with the paper's own metrics -- `eval/motion_fidelity_score.py` (motion
fidelity) and `eval/clip_score.py` (prompt adherence) -- and aggregates into a
comparison table.

    python sweep_wan.py \
        --video_path ./assets/bmx-trees.mp4 \
        --prompt "Leopard running up a snowy hill in a forest" \
        --guidance_blocks 10 12 15 18 20 \
        --motion_temp 1 2 4 8

Design notes:

  * **Each run is a subprocess.** Wan is loaded and torn down per grid point;
    doing that in-process leaks VRAM across runs and eventually OOMs. Process
    isolation is the only reliable way to get comparable peak-VRAM numbers too.
  * **Resumable.** A grid point whose output video already exists is skipped, so
    a sweep killed by a dying Colab VM resumes where it left off. `--force`
    re-runs everything.
  * **Results are written after every run**, not at the end, so a crashed sweep
    still leaves a usable table.
  * **Metrics degrade gracefully.** MF needs the `cotracker` package (via
    torch.hub) and CLIP needs `clip`; a missing one is recorded as `n/a` and the
    sweep continues rather than losing the generations.

Recommended sweep order (see README_WAN.md "Tuning"): motion_temp first --
Wan applies RMSNorm to q/k, so attention logits sit on a different scale than
CogVideoX's and the AMF softmax sharpness is the highest-leverage knob --
then guidance_blocks, then guidance_timestep_range, then lr.
"""

import argparse
import csv
import itertools
import json
import re
import subprocess
import sys
from pathlib import Path

PY = sys.executable
REPO = Path(__file__).resolve().parent


def _run(cmd, log_path=None):
    """Run a subprocess, tee output to a log, return (returncode, stdout+stderr)."""
    proc = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if log_path:
        Path(log_path).write_text(out, encoding="utf-8")
    return proc.returncode, out


def _grab(pattern, text, cast=float):
    m = re.search(pattern, text)
    if not m:
        return None
    try:
        return cast(m.group(1))
    except (TypeError, ValueError):
        return None


def generate(point, args):
    """Run one grid point. Returns dict with video path, seconds, peak VRAM."""
    out_dir = Path(args.output_root) / point["tag"]
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, "motion_guidance_wan.py",
        "--video_path", args.video_path,
        "--prompt", args.prompt,
        "--model", args.model,
        "--loss_type", args.loss_type,
        "--opt_mode", args.opt_mode,
        "--output_path", str(out_dir),
        "--seed", str(args.seed),
    ]
    if args.num_frames is not None:
        cmd += ["--num_frames", str(args.num_frames)]
    if point.get("no_guidance"):
        cmd += ["--no_guidance"]
    if point.get("no_injection"):
        cmd += ["--no_injection"]
    if point.get("guidance_blocks") is not None:
        cmd += ["--guidance_blocks"] + [str(b) for b in point["guidance_blocks"]]
    if point.get("motion_temp") is not None:
        cmd += ["--motion_temp", str(point["motion_temp"])]
    if point.get("optimization_steps") is not None:
        cmd += ["--optimization_steps", str(point["optimization_steps"])]
    if point.get("lr") is not None:
        cmd += ["--lr", str(point["lr"][0]), str(point["lr"][1])]
    cmd += args.extra

    rc, out = _run(cmd, out_dir / "generate.log")
    if rc != 0:
        tail = "\n".join(out.strip().splitlines()[-6:])
        return {"ok": False, "error": tail}

    return {
        "ok": True,
        "video": _grab(r"\|\s*([^|\n]+\.mp4)\s*$", out, str) or _find_output(out_dir),
        "seconds": _grab(r"Finished in ([\d.]+)s", out),
        "peak_vram_gb": _grab(r"peak VRAM ([\d.]+)GB", out),
        "seq_len": _grab(r"seq_len=(\d+)", out, int),
    }


def _find_output(out_dir):
    """Fallback: newest .mp4 in out_dir that isn't the reference copy."""
    cands = [p for p in Path(out_dir).glob("*.mp4") if p.name != "original.mp4"]
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None


def score(video, out_dir, args):
    """Evaluate one output with the paper's MF and CLIP metrics."""
    scores = {"mf": None, "clip": None, "mf_note": "", "clip_note": ""}
    reference = Path(out_dir) / "original.mp4"  # resized/truncated to match the generation

    if not args.skip_mf:
        rc, out = _run(
            [PY, "eval/motion_fidelity_score.py",
             "--video_path", str(video),
             "--original_video_path", str(reference),
             "--output_path", str(out_dir)],
            Path(out_dir) / "eval_mf.log",
        )
        scores["mf"] = _grab(r"Tracklets score:\s*([\d.eE+-]+)", out)
        if scores["mf"] is None:
            scores["mf_note"] = "cotracker missing" if "cotracker" in out.lower() else f"rc={rc}"

    if not args.skip_clip:
        rc, out = _run(
            [PY, "eval/clip_score.py",
             "--video_path", str(video),
             "--prompt", args.prompt,
             "--output_path", str(out_dir)],
            Path(out_dir) / "eval_clip.log",
        )
        scores["clip"] = _grab(r"CLIP score:\s*([\d.eE+-]+)", out)
        if scores["clip"] is None:
            scores["clip_note"] = "clip missing" if "clip" in out.lower() and "No module" in out else f"rc={rc}"

    return scores


def build_grid(args):
    points = []
    if args.include_baselines:
        points.append({"tag": "baseline_backbone", "no_guidance": True, "no_injection": True, "label": "backbone (no guidance/injection)"})
        points.append({"tag": "baseline_injection", "no_guidance": True, "label": "injection only"})

    blocks = args.guidance_blocks or [None]
    temps = args.motion_temp or [None]
    steps = args.optimization_steps or [None]

    for gb, mt, os_ in itertools.product(blocks, temps, steps):
        parts, label = [], []
        if gb is not None:
            parts.append(f"gb{'-'.join(str(b) for b in gb)}")
            label.append(f"blocks={gb}")
        if mt is not None:
            parts.append(f"t{mt:g}")
            label.append(f"temp={mt:g}")
        if os_ is not None:
            parts.append(f"s{os_}")
            label.append(f"steps={os_}")
        points.append({
            "tag": "_".join(parts) or "default",
            "label": ", ".join(label) or "config default",
            "guidance_blocks": gb, "motion_temp": mt, "optimization_steps": os_,
        })
    return points


def write_results(rows, root):
    root = Path(root)
    fields = ["tag", "label", "mf", "clip", "seconds", "peak_vram_gb", "seq_len", "video", "note"]
    with open(root / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def fmt(v, spec=".4f"):
        return "n/a" if v is None else (format(v, spec) if isinstance(v, float) else str(v))

    # Rank by MF where available; unscored rows sink to the bottom.
    ranked = sorted(rows, key=lambda r: (r.get("mf") is None, -(r.get("mf") or 0)))
    lines = [
        "# DiTFlow-Wan sweep results", "",
        "MF = motion fidelity (higher is better). CLIP = prompt adherence (higher is better).",
        "Sorted by MF.", "",
        "| config | MF | CLIP | time (s) | peak VRAM (GB) | note |",
        "|---|---|---|---|---|---|",
    ]
    for r in ranked:
        lines.append(
            f"| {r['label']} | {fmt(r.get('mf'))} | {fmt(r.get('clip'))} | "
            f"{fmt(r.get('seconds'), '.0f')} | {fmt(r.get('peak_vram_gb'), '.1f')} | {r.get('note', '')} |"
        )
    (root / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (root / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Hyperparameter sweep for DiTFlow-Wan, scored with MF + CLIP")
    p.add_argument("-v", "--video_path", required=True)
    p.add_argument("-p", "--prompt", required=True)
    p.add_argument("--output_root", default="./sweeps/run")
    p.add_argument("--model", default="1.3b", choices=["1.3b", "14b"])
    p.add_argument("--loss_type", default="flow", choices=["flow", "moft", "smm"])
    p.add_argument("--opt_mode", default="latent", choices=["latent", "emb"])
    p.add_argument("--num_frames", type=int, default=None)
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--guidance_blocks", type=int, nargs="+", action="append",
                   help="Repeat to sweep: --guidance_blocks 12 --guidance_blocks 15")
    p.add_argument("--motion_temp", type=float, nargs="+", default=None)
    p.add_argument("--optimization_steps", type=int, nargs="+", default=None)

    p.add_argument("--include_baselines", action="store_true", help="Also run backbone / injection-only references")
    p.add_argument("--skip_mf", action="store_true")
    p.add_argument("--skip_clip", action="store_true")
    p.add_argument("--force", action="store_true", help="Re-run grid points that already have output")
    p.add_argument("--dry_run", action="store_true", help="Print the grid and exit")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra args passed through to motion_guidance_wan.py")
    args = p.parse_args()

    grid = build_grid(args)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    print(f"{len(grid)} grid point(s) -> {root}")
    for pt in grid:
        print(f"  - {pt['tag']:28s} {pt['label']}")
    if args.dry_run:
        return

    rows = []
    for i, pt in enumerate(grid, 1):
        out_dir = root / pt["tag"]
        existing = _find_output(out_dir) if out_dir.exists() else None
        note = ""

        print(f"\n[{i}/{len(grid)}] {pt['label']}")
        if existing and not args.force:
            print(f"  reusing existing output: {existing}")
            gen = {"ok": True, "video": existing, "seconds": None, "peak_vram_gb": None, "seq_len": None}
            note = "reused"
        else:
            gen = generate(pt, args)
            if not gen["ok"]:
                print(f"  FAILED: {gen['error']}")
                rows.append({**pt, "note": f"generation failed: {gen['error'][:120]}"})
                write_results(rows, root)
                continue
            print(f"  generated in {gen['seconds']:.0f}s, peak {gen['peak_vram_gb']:.1f}GB"
                  if gen["seconds"] else "  generated")

        if not gen["video"] or not Path(gen["video"]).exists():
            rows.append({**pt, "note": "output video not found"})
            write_results(rows, root)
            continue

        sc = score(gen["video"], out_dir, args)
        notes = " ".join(n for n in (note, sc["mf_note"], sc["clip_note"]) if n)
        print(f"  MF={sc['mf'] if sc['mf'] is not None else 'n/a'}  "
              f"CLIP={sc['clip'] if sc['clip'] is not None else 'n/a'}  {notes}")

        rows.append({**pt, **{k: gen[k] for k in ("video", "seconds", "peak_vram_gb", "seq_len")},
                     "mf": sc["mf"], "clip": sc["clip"], "note": notes})
        write_results(rows, root)

    print(f"\nDone. Table: {root / 'results.md'}")
    scored = [r for r in rows if r.get("mf") is not None]
    if scored:
        best = max(scored, key=lambda r: r["mf"])
        print(f"Best MF: {best['mf']:.4f} at {best['label']}")


if __name__ == "__main__":
    main()
