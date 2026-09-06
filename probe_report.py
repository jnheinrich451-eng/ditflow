"""Render shared Wan/CogVideoX probe traces without loading either model.

    python probe_report.py results_lucia_euler results_lucia_cog --output_dir probe_comparison

In a notebook: from probe_report import plot_capture; plot_capture(run, block=15,
stage="denoise_cond", step=10). Paths accept a run root or a timestamped trace.
"""

import argparse
import base64
import csv
import html
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_trace(run):
    path = Path(run)
    if not (path / "metadata.json").exists():
        candidates = sorted((path / "probes").glob("*/metadata.json"))
        if not candidates:
            raise FileNotFoundError(f"No probe traces in {path}; generate with --probe first")
        path = candidates[-1].parent
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (path / "events.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return path, metadata, events


def plot_capture(run, block, stage="reference", step=None, iteration=None, kind="hard"):
    """Plot forward-adjacent fields. Hard=argmax, soft=expected displacement.

    Every arrow uses normalized image coordinates, with a fixed scale of one.
    Confidence is peak attention probability, not calibrated physical accuracy.
    """
    path, meta, events = load_trace(run)
    name = f"block_{block}_attn1_processor"
    matches = [e for e in events if e["kind"] == "attention" and e["block"] == name
               and e["stage"] == stage and (step is None or e["step"] == step)
               and (iteration is None or e["iteration"] == iteration)]
    if not matches:
        raise ValueError(f"No capture for {name}, {stage}, step={step}, iteration={iteration}")
    event = matches[-1]
    with np.load(path / event["file"]) as data:
        flow, confidence = data[kind].copy(), data["confidence"].copy()
    frames, h, w = meta["grid"]
    fig, axes = plt.subplots(1, frames - 1, figsize=(3.8 * (frames - 1), 3.4), squeeze=False)
    yy, xx = np.meshgrid((np.arange(h) + 0.5) / h, (np.arange(w) + 0.5) / w, indexing="ij")
    stride = max(1, int(np.ceil(max(h, w) / 16)))
    sample = (slice(None, None, stride), slice(None, None, stride))
    for i, ax in enumerate(axes[0]):
        vectors = flow[i].reshape(h, w, 2) / [w, h]
        ax.imshow(confidence[i].reshape(h, w), extent=(0, 1, 1, 0), vmin=0, vmax=1, cmap="Greys", alpha=0.4)
        ax.quiver(xx[sample], yy[sample], vectors[..., 0][sample], vectors[..., 1][sample],
                  color="#1f77b4", angles="xy", scale_units="xy", scale=1, width=0.006)
        ax.set(xlim=(0, 1), ylim=(1, 0), title=f"latent {i} → {i + 1}", xlabel="x / width")
        ax.set_aspect("equal")
    axes[0, 0].set_ylabel("y / height")
    fig.suptitle(f"{meta['model']} block {block} | {stage} step {event['step']} | {kind} AMF")
    fig.tight_layout()
    return fig


def _inline_figure(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return '<img alt="Motion probe plot" src="data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode() + '">'


def make_report(runs, output_dir):
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    traces = [load_trace(run) for run in runs]
    page = ['<!doctype html><meta charset="utf-8"><title>Motion probe comparison</title>',
            '<style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px}img{max-width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>',
            '<h1>Motion probe comparison</h1>',
            '<p>AMF estimates correspondence; these plots do not measure camera pose or physical optical flow. '
            'Attention diagnostics use fp32 and forward-adjacent latent pairs. Actual training losses and masks '
            'are saved separately. No subject/background segmentation is inferred.</p>',
            '<p>Compare direction patterns and within-run trends. Different models have different noise schedules, '
            'VAE temporal support, grids, conditioning, and logit distributions. Neither identical step indices '
            'nor absolute loss values establish equivalent conditions. Arrow scale is fixed in fractions of image width/height.</p>']
    csv_rows = []
    for run_i, (path, meta, events) in enumerate(traces):
        label = f"{meta['model']} — {path.parent.parent.name}"
        page.append(f"<h2>{html.escape(label)}</h2><p>{html.escape(str(path))}</p>")
        cfg = meta["config"]
        settings = {k: cfg.get(k) for k in ("model_key", "seed", "video_path", "target_prompt", "height", "width",
                                            "num_frames", "video_length", "guidance_blocks", "injection_blocks",
                                            "guidance_timestep_range", "motion_temp", "threshloss", "flow_max_disp", "flow_min_conf")}
        settings.update(scheduler=meta["scheduler"], probe_blocks=meta["blocks"], packages=meta["packages"])
        page.append("<pre>" + html.escape(json.dumps(settings, indent=2)) + "</pre>")
        reference_masks = [e for e in events if e["kind"] == "training_reference"]
        if reference_masks:
            page.append("<p>Actual training-reference mask coverage: " + ", ".join(
                f"{html.escape(e['block'])}: {100 * e['kept_fraction']:.2f}%" for e in reference_masks) + ".</p>")
        opts = [e for e in events if e["kind"] == "optimization"]
        samples = [e for e in events if e["kind"] == "sampling"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        axes[0, 0].plot(range(len(opts)), [e["loss_before_update"] for e in opts])
        axes[0, 0].set(title="Actual guidance loss (before each update)", xlabel="Optimizer iteration", ylabel="Configured loss")
        axes[0, 1].plot(range(len(opts)), [(e.get("gradient") or {}).get("rms") for e in opts])
        axes[0, 1].set(title="Gradient RMS (unscaled)", xlabel="Optimizer iteration")
        for key, label_ in (("guidance_update", "Guidance"), ("sampling_update", "Sampling")):
            axes[1, 0].plot([e["step"] for e in samples], [e[key]["rms"] for e in samples], label=label_)
        axes[1, 0].set(title="Latent update RMS", xlabel="Sampling step")
        axes[1, 0].legend()
        for block in meta["blocks"]:
            name = f"block_{block}_attn1_processor"
            captures = [e for e in events if e["kind"] == "attention" and e["block"] == name
                        and e["stage"] == "denoise_cond" and "mse_to_reference_hard_normalized" in e]
            axes[1, 1].plot([e["step"] for e in captures], [e["mse_to_reference_hard_normalized"] for e in captures],
                            marker="o", label=f"block {block}")
        axes[1, 1].set(title="Diagnostic AMF MSE to hard reference (all adjacent patches)", xlabel="Sampling step", ylabel="Normalized MSE")
        if axes[1, 1].lines:
            axes[1, 1].legend()
        fig.tight_layout()
        fig.savefig(destination / f"run_{run_i}_curves.png", dpi=150)
        page.append(_inline_figure(fig))
        for block in meta["blocks"]:
            for stage in ("reference", "final_latent"):
                try:
                    fig = plot_capture(path, block, stage=stage)
                except ValueError:
                    page.append(f"<p>Missing {stage} capture at block {block}; run may be incomplete.</p>")
                    continue
                fig.savefig(destination / f"run_{run_i}_block_{block}_{stage}.png", dpi=150)
                page.append(_inline_figure(fig))
        for e in events:
            if e["kind"] != "attention":
                continue
            for pair in e["hard_pairs"]:
                csv_rows.append({"run": str(path), "model": meta["model"], "stage": e["stage"], "step": e["step"],
                                 "timestep": e["timestep"], "iteration": e["iteration"], "block": e["block"],
                                 "mean_confidence": e["mean_confidence"], "mean_entropy": e["mean_entropy"], **pair})
    if csv_rows:
        with (destination / "attention_pairs.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
    report = destination / "comparison.html"
    report.write_text("\n".join(page), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output_dir", default="probe_comparison")
    args = parser.parse_args()
    print(make_report(args.runs, args.output_dir))
