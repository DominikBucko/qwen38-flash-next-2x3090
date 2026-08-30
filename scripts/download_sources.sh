#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: download_sources.sh DESTINATION_ROOT}
command -v hf >/dev/null || {
  echo "Hugging Face CLI 'hf' is required" >&2
  exit 2
}

mkdir -p "$destination"
destination=$(cd -- "$destination" && pwd -P)

hf download Intel/Qwen3.8-Flash-Next-W4A16-AutoRound \
  --revision 861536dda5bcb208376fc4cd879b2bf76bece9fe \
  --exclude model-00016-of-00017.safetensors \
  --exclude model_extra_tensors.safetensors \
  --local-dir "$destination/intel-autoround"

hf download RadixArk/Qwen3.8-Flash-Next-NVFP4 \
  --revision 7b719225242aacd3dbd3f9407468c2ee9a9d2594 \
  --local-dir "$destination/radixark-nvfp4"

printf 'Intel source: %s\nPLE/MTP source: %s\n' \
  "$destination/intel-autoround" "$destination/radixark-nvfp4"
