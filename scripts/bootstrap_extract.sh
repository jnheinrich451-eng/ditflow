#!/usr/bin/env bash
# extract-v1 session bootstrap (Module 1 instrument: Open-d4rt + SAM2).
# Run in a Colab runtime DEDICATED to extraction — never in the same runtime as
# requirements.txt (CogVideoX/Wan pins numpy 1.26 / pillow 9.5 / xformers):
#   bash scripts/bootstrap_extract.sh
#
# Differs from scripts/bootstrap.sh (the pipeline-repo copy) in exactly this:
#   - installs requirements-extract.txt, not requirements.txt
#   - every model is PINNED from configs/stage_a.yaml (d4rt.commit,
#     d4rt.checkpoint_revision, d4rt.checkpoint_subdir, sam2.git_commit,
#     sam2.weights_revision) and refuses to run while a pin is '<...>'
#   - Open-d4rt is checked out at the pin, never `git pull`ed
#   - no dataset download unless repo.hf_dataset_id is set
# Idempotent. Assumes Drive is mounted at /content/drive (notebook cell).
set -euo pipefail

DRIVE=/content/drive/MyDrive
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"
[ -d "$DRIVE" ] || { echo "FATAL: Drive not mounted at /content/drive"; exit 1; }

# ---- secrets -> process env only (never printed, never persisted) ----
load_secret () {  # $1 = dotenv file, $2 = env var name
  local file="$1" var="$2" line val
  [ -f "$file" ] || { echo "FATAL: missing secret file $file"; exit 1; }
  line="$(grep -m1 -E '^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' "$file")" || { echo "FATAL: no KEY=VALUE line in $file"; exit 1; }
  val="${line#*=}"
  val="${val#"${val%%[![:space:]]*}"}"
  val="${val%"${val##*[![:space:]]}"}"
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
  export "$var"="$val"
}
[ -n "${HF_TOKEN:-}" ] || load_secret "$DRIVE/secrets/hf_token.env" HF_TOKEN
export HF_TOKEN

pip install -q -r requirements-extract.txt

# ---- pins + paths from configs/stage_a.yaml (refuse unfilled pins) ----
eval "$(python - <<'PY'
from src.preprocess.config import load_config
c = load_config(require_filled=("d4rt", "sam2"))
d, s, p = c["d4rt"], c["sam2"], c["paths"]
for k, v in {"D4RT_REPO": d["repo"], "D4RT_CKPT": d["checkpoint"],
             "D4RT_COMMIT": d["commit"], "D4RT_CKPT_REV": d["checkpoint_revision"],
             "D4RT_CKPT_SUB": d["checkpoint_subdir"],
             "SAM2_COMMIT": s["git_commit"], "SAM2_W_REV": s["weights_revision"],
             "HF_DATASET_ID": c["repo"].get("hf_dataset_id") or "",
             "DRIVE_ROOT": p["drive_root"], "D4RT_DIR": p["d4rt_dir"],
             "CKPT_DIR": p["ckpt_dir"], "DATA_DIR": p["data_dir"]}.items():
    print(f"export {k}='{v}'")
PY
)"
mkdir -p "$DRIVE_ROOT/cache/stage_a" "$DRIVE_ROOT/viz/stage_a" "$DATA_DIR"

# ---- Open-d4rt at the pinned commit ----
[ -d "$D4RT_DIR/.git" ] || git clone "$D4RT_REPO" "$D4RT_DIR"
git -C "$D4RT_DIR" fetch -q origin || true
git -C "$D4RT_DIR" checkout -q "$D4RT_COMMIT"
HEAD_SHA="$(git -C "$D4RT_DIR" rev-parse HEAD)"
[[ "$HEAD_SHA" == "$D4RT_COMMIT"* ]] || { echo "FATAL: Open-d4rt HEAD $HEAD_SHA != pin $D4RT_COMMIT"; exit 1; }
# same order as the validated environment: theirs after ours
[ -f "$D4RT_DIR/requirements.txt" ] && pip install -q -r "$D4RT_DIR/requirements.txt" || true

# ---- OpenD4RT checkpoint: pinned HF revision, validated subfolder only ----
python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=os.environ["D4RT_CKPT"], repo_type="model",
                      revision=os.environ["D4RT_CKPT_REV"],
                      allow_patterns=[os.environ["D4RT_CKPT_SUB"] + "/*"],
                      local_dir=os.environ["CKPT_DIR"])
sub = os.path.join(p, os.environ["D4RT_CKPT_SUB"])
for f in ("opend4rt.ckpt", "model.yaml"):
    assert os.path.isfile(os.path.join(sub, f)), f"checkpoint incomplete: {sub}/{f} missing"
print("checkpoint ->", sub)
PY

# ---- SAM2 code at the pinned commit (skip if already that commit) ----
python - <<'PY' || pip install -q "git+https://github.com/facebookresearch/sam2.git@${SAM2_COMMIT}"
import json, os, sys
from importlib.metadata import distribution
u = json.loads(distribution("sam2").read_text("direct_url.json") or "{}")
sys.exit(0 if u.get("vcs_info", {}).get("commit_id", "").startswith(os.environ["SAM2_COMMIT"]) else 1)
PY

# ---- SAM2 weights at the pinned revision (into the HF cache Stage C reads) ----
python - <<'PY'
import os, yaml
from huggingface_hub import hf_hub_download
from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES
from src.preprocess.stage_c.mask import VARIANTS
variant = yaml.safe_load(open("configs/stage_c.yaml", encoding="utf-8"))["sam2"]["variant"]
_, ck = HF_MODEL_ID_TO_FILENAMES[VARIANTS[variant]]
print("sam2 weights ->", hf_hub_download(VARIANTS[variant], ck, revision=os.environ["SAM2_W_REV"]))
PY

# ---- optional dataset ----
if [ -n "$HF_DATASET_ID" ]; then
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
print("videos ->", snapshot_download(repo_id=os.environ["HF_DATASET_ID"], repo_type="dataset",
                                     local_dir=os.environ["DATA_DIR"]))
PY
fi

# ---- environment summary (copied into every bundle's provenance) ----
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo no-gpu)"
TORCH="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo no-torch)"
REPO_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo none)"
SUMMARY="gpu=$GPU | torch=$TORCH | repo=$REPO_SHA | d4rt=${HEAD_SHA:0:7} | d4rt_ckpt=${D4RT_CKPT_REV:0:7} | sam2=${SAM2_COMMIT:0:7} | sam2_w=${SAM2_W_REV:0:7} | utc=$(date -u +%FT%TZ)"
echo "$SUMMARY" | tee /content/env_summary.txt
echo "bootstrap_extract OK"
