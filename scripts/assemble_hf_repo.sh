#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: assemble_hf_repo.sh INTEL_SOURCE PLE_SOURCE OUTPUT WORK_DIR" >&2
  exit 2
}

[[ $# -eq 4 ]] || usage
intel_source=$(cd -- "$1" && pwd -P)
ple_source=$(cd -- "$2" && pwd -P)
output=$3
work_root=$4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_dir=$(cd -- "$script_dir/.." && pwd)

[[ ! -e "$output" ]] || { echo "output already exists: $output" >&2; exit 2; }
mkdir -p "$work_root"
work_root=$(cd -- "$work_root" && pwd -P)
build_dir=$(mktemp -d "$work_root/qwen38-mtp-build.XXXXXX")
keep_work=${KEEP_WORK:-0}
cleanup() {
  if [[ "$keep_work" == 1 ]]; then
    echo "retained intermediate MTP checkpoint: $build_dir" >&2
  else
    rm -rf -- "$build_dir"
  fi
}
trap cleanup EXIT

python3 "$script_dir/build_intel_fp8ple_hybrid.py" \
  --intel-source "$intel_source" \
  --ple-source "$ple_source" \
  --output "$output"

python3 "$script_dir/build_mtp_int4.py" \
  "$ple_source" "$build_dir/development-mtp" \
  --group-size 32

mkdir -p "$output/runtime"
python3 "$script_dir/compact_mtp_checkpoint.py" \
  "$build_dir/development-mtp" \
  "$output/runtime/mtp-int4-g32" \
  --copy-experts

cp "$repo_dir/packaging/model-card.md" "$output/README.md"
cp "$repo_dir/packaging/runtime-README.md" "$output/runtime/README.md"
cp "$repo_dir/docker/Dockerfile" "$output/runtime/Dockerfile"
cp "$repo_dir/LICENSE" "$output/runtime/LICENSE"
cp "$repo_dir/repro.lock.json" "$output/runtime/repro.lock.json"
cp "$repo_dir/configs/static_hot_cache_rankings.json" \
  "$output/runtime/static_hot_cache_rankings.json"
cp "$script_dir/serve-container.sh" "$output/runtime/serve-container.sh"
cp "$script_dir/validate_upload.py" "$output/runtime/validate_upload.py"
cp "$repo_dir/runtime/install_overlay.py" "$output/runtime/install_overlay.py"
cp -R "$repo_dir/runtime/vllm-overlay" "$output/runtime/vllm-overlay"

python3 "$script_dir/validate_hybrid.py" "$output"
python3 "$script_dir/validate_compact_mtp.py" \
  "$build_dir/development-mtp" "$output/runtime/mtp-int4-g32"
python3 "$script_dir/validate_upload.py" "$output"

printf 'upload tree ready: %s\n' "$(cd -- "$output" && pwd -P)"
