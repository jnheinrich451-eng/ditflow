#!/usr/bin/env python
"""Find the model versions Module 1 was VALIDATED with, to fill the pins in
configs/stage_a.yaml (d4rt.commit, d4rt.checkpoint_revision, sam2.git_commit,
sam2.weights_revision). Run on Colab with Drive mounted; read-only.

    python scripts/find_pins.py [--pipeline-root /content/drive/MyDrive/motion_transfer]

Evidence, strongest first:
  d4rt.commit            MEASURED: model_commit in the pipeline repo's validated
                         Stage A provenance.json files (the adapter records it).
  checkpoint_revision,   INFERRED: the HF commit that was `main` on the date the
  sam2.weights_revision  stage was validated (downloads were unpinned).
  sam2.git_commit        INFERRED: the sam2 default-branch commit on the Stage C
                         date (pip installed an unpinned git HEAD).
Inferred pins must be confirmed by reproduction: re-extract a calibration clip
and compare its qc gate statistics with the pipeline repo's.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def before(commits, day):
    cut = datetime.fromisoformat(day + "T23:59:59").replace(tzinfo=timezone.utc)
    ok = [c for c in commits if c.created_at <= cut]
    return max(ok, key=lambda c: c.created_at) if ok else None


def hf_commits(repo, day):
    from huggingface_hub import HfApi
    cs = HfApi().list_repo_commits(repo)
    print(f"\n{repo}: {len(cs)} commits (latest 6)")
    for c in sorted(cs, key=lambda c: c.created_at)[-6:]:
        print(f"  {c.commit_id}  {c.created_at:%Y-%m-%d}  {c.title[:60]}")
    pick = before(cs, day)
    print(f"  -> main on {day}: {pick.commit_id if pick else 'NONE (repo newer than date)'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-root", default="/content/drive/MyDrive/motion_transfer")
    ap.add_argument("--stage-a-date", default="2026-08-03", help="Stage A sign-off")
    ap.add_argument("--stage-c-date", default="2026-08-22", help="Stage C sign-off")
    args = ap.parse_args()

    provs = sorted(Path(args.pipeline_root, "cache/stage_a").glob("*/provenance.json"))
    cnt = collections.Counter()
    for p in provs:
        d = json.loads(p.read_text(encoding="utf-8"))
        cnt[(d.get("model_commit"), d.get("checkpoint"))] += 1
    print(f"Stage A bundles read: {len(provs)}")
    for (commit, ck), n in cnt.most_common():
        print(f"  d4rt.commit={commit}  checkpoint_subdir={ck}  ({n} bundles)")
    if len(cnt) > 1:
        print("  WARNING: bundles disagree - the pipeline caches mix D4RT versions")

    hf_commits("Lijiaxin0111/OpenD4RT", args.stage_a_date)
    hf_commits("facebook/sam2.1-hiera-base-plus", args.stage_c_date)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--no-checkout",
                        "https://github.com/facebookresearch/sam2.git", tmp], check=True)
        sha = subprocess.run(["git", "-C", tmp, "rev-list", "-1",
                              f"--before={args.stage_c_date}T23:59:59Z", "HEAD"],
                             capture_output=True, text=True, check=True).stdout.strip()
    print(f"\nfacebookresearch/sam2 default branch on {args.stage_c_date}: {sha}")
    try:
        from importlib.metadata import distribution
        u = json.loads(distribution("sam2").read_text("direct_url.json") or "{}")
        print(f"sam2 installed on THIS VM: {u.get('vcs_info', {}).get('commit_id')}")
    except Exception as e:  # noqa: BLE001
        print(f"sam2 not installed on this VM ({type(e).__name__})")
    print("\nFill these into configs/stage_a.yaml, then run scripts/bootstrap_extract.sh.")


if __name__ == "__main__":
    main()
