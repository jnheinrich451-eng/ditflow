"""Inline video display and result download helpers, for Colab or Jupyter.

    from colab_utils import show_videos, show_run, zip_results, summarize_run

    show_run("results_wan")          # reference vs result, side by side
    zip_results("results_wan")       # -> results_wan.zip, downloads in Colab

Colab's filesystem is temporary: anything under /content disappears when the
runtime recycles. Download or copy to Drive before you lose it.
"""

import base64
import os
import shutil
from pathlib import Path

VIDEO_EXTS = (".mp4", ".gif", ".webm")


def _tag(path, width=420):
    """One <video> (or <img> for gif) as a self-contained data URI."""
    path = Path(path)
    data = base64.b64encode(path.read_bytes()).decode()
    if path.suffix.lower() == ".gif":
        return f'<img src="data:image/gif;base64,{data}" width="{width}">'
    mime = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    return (
        f'<video width="{width}" controls loop autoplay muted playsinline>'
        f'<source src="data:{mime};base64,{data}" type="{mime}"></video>'
    )


def show_videos(paths, labels=None, width=420):
    """Render videos side by side. Returns an IPython HTML object."""
    from IPython.display import HTML

    paths = [Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths)]
    missing = [p for p in paths if not p.exists()]
    if missing:
        return HTML(f"<pre>missing: {', '.join(str(m) for m in missing)}</pre>")

    labels = labels or [p.stem for p in paths]
    cells = "".join(
        f'<figure style="margin:0 12px 0 0;text-align:center">{_tag(p, width)}'
        f'<figcaption style="font:13px/1.6 system-ui;opacity:.75">{lab}</figcaption></figure>'
        for p, lab in zip(paths, labels)
    )
    total_mb = sum(p.stat().st_size for p in paths) / 1024**2
    note = ""
    if total_mb > 40:
        note = (f'<p style="font:12px system-ui;color:#c00">~{total_mb:.0f} MB inlined as base64; '
                f"large notebooks get slow. Use zip_results() instead.</p>")
    return HTML(f'<div style="display:flex;flex-wrap:wrap;align-items:flex-start">{cells}</div>{note}')


def find_outputs(output_dir="results_wan"):
    """(reference, [generated...]) for a run directory."""
    d = Path(output_dir)
    if not d.exists():
        return None, []
    vids = sorted((p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS),
                  key=lambda p: p.stat().st_mtime)
    reference = next((p for p in vids if p.name == "original.mp4"), None)
    return reference, [p for p in vids if p.name != "original.mp4"]


def show_run(output_dir="results_wan", width=420):
    """Reference beside every generated video in a run directory."""
    from IPython.display import HTML

    reference, generated = find_outputs(output_dir)
    if reference is None and not generated:
        return HTML(f"<pre>no videos in {output_dir}/ -- did the run finish?</pre>")
    paths = ([reference] if reference else []) + generated
    labels = (["reference"] if reference else []) + [p.stem[:40] for p in generated]
    return show_videos(paths, labels, width)


def summarize_run(output_dir="results_wan"):
    """Did guidance actually run? Prints what the directory says."""
    d = Path(output_dir)
    if not d.exists():
        print(f"{d}/ does not exist")
        return
    reference, generated = find_outputs(d)
    print(f"{d}/")
    print(f"  reference : {reference.name if reference else 'MISSING'}")
    for g in generated:
        print(f"  generated : {g.name}  ({g.stat().st_size/1024**2:.1f} MB)")

    embeds = d / "embeds"
    saved = sorted(embeds.glob("*.pt")) if embeds.exists() else []
    if saved:
        kinds = sorted({p.name.split("_")[0] for p in saved})
        print(f"  guidance  : {len(saved)} optimised tensor(s) saved, kind(s)={kinds}")
        print("              -> guidance DID run on those timesteps")
    else:
        print("  guidance  : no embeds/*.pt saved")
        print("              -> guidance did NOT run (--no_guidance, or the")
        print("                 guidance_timestep_range matched no timestep)")

    cfg = d / "config.yaml"
    if cfg.exists():
        import yaml
        c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
        keys = ("guidance_blocks", "motion_temp", "num_frames", "guidance_timestep_range",
                "optimization_steps", "loss_type", "guidance_mode")
        print("  config    : " + ", ".join(f"{k}={c[k]}" for k in keys if k in c))


def zip_results(output_dir="results_wan", zip_path=None, download=True, include_embeds=False):
    """Zip a run directory and, in Colab, trigger a browser download.

    embeds/*.pt are excluded by default -- they are large and only useful for
    --inject_embeds, not for looking at results.
    """
    src = Path(output_dir)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")

    zip_path = Path(zip_path or f"{src.name}.zip")
    staged = src
    if not include_embeds and (src / "embeds").exists():
        staged = Path(f"/tmp/_zip_{src.name}")
        if staged.exists():
            shutil.rmtree(staged)
        shutil.copytree(src, staged, ignore=shutil.ignore_patterns("embeds"))

    archive = shutil.make_archive(str(zip_path.with_suffix("")), "zip", staged)
    if staged is not src:
        shutil.rmtree(staged, ignore_errors=True)

    print(f"{archive}  ({os.path.getsize(archive)/1024**2:.1f} MB)")
    if download:
        try:
            from google.colab import files  # noqa: PLC0415

            files.download(archive)
        except ImportError:
            print("(not in Colab -- copy the file yourself)")
    return archive


def save_to_drive(output_dir="results_wan", drive_subdir="ditflow_wan"):
    """Copy a run directory to Google Drive, which survives runtime recycling."""
    from google.colab import drive  # noqa: PLC0415

    if not Path("/content/drive").exists():
        drive.mount("/content/drive")
    dest = Path("/content/drive/MyDrive") / drive_subdir / Path(output_dir).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(output_dir, dest)
    print(f"copied -> {dest}")
    return str(dest)
