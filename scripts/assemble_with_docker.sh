#!/usr/bin/env bash
set -euo pipefail

data_root=${1:?usage: assemble_with_docker.sh DATA_ROOT [OUTPUT_NAME]}
output_name=${2:-upload}
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)
data_root=$(cd -- "$data_root" && pwd -P)
image=${IMAGE:-qwen38-flash-next-2x3090:locked}

[[ -d "$data_root/intel-autoround" ]] || {
  echo "missing $data_root/intel-autoround; run download_sources.sh first" >&2
  exit 2
}
[[ -d "$data_root/radixark-nvfp4" ]] || {
  echo "missing $data_root/radixark-nvfp4; run download_sources.sh first" >&2
  exit 2
}
[[ "$output_name" != */* && "$output_name" != .* ]] || {
  echo "OUTPUT_NAME must be a simple directory name" >&2
  exit 2
}

exec docker run --rm \
  --user "$(id -u):$(id -g)" \
  --entrypoint /repo/scripts/assemble_hf_repo.sh \
  -e KEEP_WORK="${KEEP_WORK:-0}" \
  -v "$repo_dir:/repo:ro" \
  -v "$data_root:/data" \
  "$image" \
  /data/intel-autoround \
  /data/radixark-nvfp4 \
  "/data/$output_name" \
  /data/work
