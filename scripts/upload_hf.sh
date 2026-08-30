#!/usr/bin/env bash
set -euo pipefail

repo_id=${1:?usage: upload_hf.sh OWNER/REPO MODEL_DIR}
model_dir=${2:?usage: upload_hf.sh OWNER/REPO MODEL_DIR}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
model_dir=$(cd -- "$model_dir" && pwd -P)

command -v hf >/dev/null || { echo "Hugging Face CLI 'hf' is required" >&2; exit 2; }
python3 "$script_dir/validate_upload.py" "$model_dir"
hf auth whoami >/dev/null
hf repos create "$repo_id" --type model --exist-ok

# hf_xet makes this multi-commit upload resumable and deduplicated. Re-run this
# command after an interruption; already committed content will be skipped.
export HF_XET_HIGH_PERFORMANCE=${HF_XET_HIGH_PERFORMANCE:-1}
hf upload "$repo_id" "$model_dir" . \
  --type model \
  --commit-message "Publish reproducible Intel W4A16 + FP8 PLE + MTP3 checkpoint"

python3 - "$repo_id" <<'PY'
import sys
from huggingface_hub import HfApi

repo = sys.argv[1]
info = HfApi().model_info(repo)
print(f"published_model.repo={repo}")
print(f"published_model.revision={info.sha}")
print("Record both values in repro.lock.json before tagging the GitHub release.")
PY
