#!/usr/bin/env bash
# Stage A session bootstrap (SKILL §1.5). Idempotent — run at the top of every
# Colab session, from the repo root:   bash scripts/bootstrap.sh
#
# Assumes Drive is already mounted at /content/drive (colab.ipynb cell 0 does
# that; it needs an interactive auth flow bash can't do). Everything else —
# secrets, git credentials, deps, model repo, checkpoint, videos — happens here.
set -euo pipefail

DRIVE=/content/drive/MyDrive
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

[ -d "$DRIVE" ] || { echo "FATAL: Drive not mounted at /content/drive — run the notebook mount cell first."; exit 1; }

# ---- secrets -> process env only (SKILL §1.3: never print, never persist) ----
# colab.ipynb cell 0 already exports GH_TOKEN/HF_TOKEN; parsing here is the
# fallback for running this script standalone (e.g. VS Code remote terminal).
load_secret () {  # $1 = dotenv file, $2 = canonical env var name
  local file="$1" var="$2" line val
  [ -f "$file" ] || { echo "FATAL: missing secret file $file"; exit 1; }
  # Read whatever single KEY=VALUE line the file defines; do not guess-rename.
  # Tolerates spaces around '=' (e.g. "GH_TOKEN = abc") and quoted values.
  line="$(grep -m1 -E '^[A-Za-z_][A-Za-z0-9_]*[[:space:]]*=' "$file")" || { echo "FATAL: no KEY=VALUE line in $file"; exit 1; }
  val="${line#*=}"
  val="${val#"${val%%[![:space:]]*}"}"   # ltrim
  val="${val%"${val##*[![:space:]]}"}"   # rtrim
  val="${val%\"}"; val="${val#\"}"; val="${val%\'}"; val="${val#\'}"
  export "$var"="$val"
}
[ -n "${GH_TOKEN:-}" ] || load_secret "$DRIVE/secrets/gh_token.env" GH_TOKEN
[ -n "${HF_TOKEN:-}" ] || load_secret "$DRIVE/secrets/hf_token.env" HF_TOKEN
export GH_TOKEN HF_TOKEN   # huggingface_hub picks HF_TOKEN up implicitly

# ---- ephemeral git credentials: /tmp dies with the VM (that is the point) ----
git config --global credential.helper 'store --file /tmp/.git-credentials'
printf 'https://x-access-token:%s@github.com\n' "$GH_TOKEN" > /tmp/.git-credentials
chmod 600 /tmp/.git-credentials
git pull --ff-only 2>/dev/null || true   # no-op if just cloned / no upstream yet

# ---- python deps (torch ships with Colab) ----
pip install -q -r requirements.txt

# ---- pull the fields we need out of configs/stage_a.yaml ----
eval "$(python - <<'PY'
from src.preprocess.config import load_config
c = load_config()
print(f"export HF_DATASET_ID='{c['repo']['hf_dataset_id']}'")
print(f"export D4RT_REPO='{c['d4rt']['repo']}'")
print(f"export D4RT_CKPT='{c['d4rt']['checkpoint']}'")
print(f"export DRIVE_ROOT='{c['paths']['drive_root']}'")
print(f"export D4RT_DIR='{c['paths']['d4rt_dir']}'")
print(f"export CKPT_DIR='{c['paths']['ckpt_dir']}'")
print(f"export DATA_DIR='{c['paths']['data_dir']}'")
PY
)"
mkdir -p "$DRIVE_ROOT/cache/stage_a" "$DRIVE_ROOT/viz/stage_a" "$DATA_DIR"

# ---- D4RT model repo -> local disk ----
if [[ "$D4RT_REPO" == \<* ]]; then
  echo "WARN: d4rt.repo not filled in configs/stage_a.yaml — skipping model repo clone."
else
  if [ ! -d "$D4RT_DIR/.git" ]; then git clone --depth 1 "$D4RT_REPO" "$D4RT_DIR"
  else git -C "$D4RT_DIR" pull --ff-only || true; fi
  [ -f "$D4RT_DIR/requirements.txt" ] && pip install -q -r "$D4RT_DIR/requirements.txt" || true
fi

# ---- checkpoint (HF model id like org/name, or an absolute path) ----
if [[ "$D4RT_CKPT" == \<* ]]; then
  echo "WARN: d4rt.checkpoint not filled — skipping checkpoint download."
elif [[ "$D4RT_CKPT" == /* ]]; then
  echo "checkpoint: using local path $D4RT_CKPT"
else
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=os.environ["D4RT_CKPT"], repo_type="model",
                      local_dir=os.environ["CKPT_DIR"])
print("checkpoint ->", p)
PY
fi

# ---- reference videos -> local disk, NOT Drive (read speed) ----
if [[ "$HF_DATASET_ID" == \<* ]]; then
  echo "WARN: repo.hf_dataset_id not filled — skipping video download."
else
  python - <<'PY'
import os
from huggingface_hub import snapshot_download
p = snapshot_download(repo_id=os.environ["HF_DATASET_ID"], repo_type="dataset",
                      local_dir=os.environ["DATA_DIR"])
print("videos ->", p)
PY
fi

# ---- one-line environment summary (copied into every bundle's provenance) ----
GPU="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1 || echo no-gpu)"
TORCH="$(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo no-torch)"
REPO_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo none)"
D4RT_SHA="$(git -C "$D4RT_DIR" rev-parse --short HEAD 2>/dev/null || echo none)"
SUMMARY="gpu=$GPU | torch=$TORCH | repo=$REPO_SHA | d4rt=$D4RT_SHA | utc=$(date -u +%FT%TZ)"
echo "$SUMMARY" | tee /content/env_summary.txt
echo "bootstrap OK"
