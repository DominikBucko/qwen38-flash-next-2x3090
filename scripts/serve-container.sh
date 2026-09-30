#!/usr/bin/env bash
set -euo pipefail

profile=/opt/qwen38/configs/2x3090-128gb.env
[[ -f "$profile" ]] || { echo "missing runtime profile: $profile" >&2; exit 2; }
# shellcheck source=/dev/null
source "$profile"

case "$ENABLE_VISION" in
  0) modality_args=(--language-model-only) ;;
  1)
    if ! [[ "$VISION_MAX_IMAGES" =~ ^[1-9][0-9]*$ && "$VISION_MAX_PIXELS" =~ ^[1-9][0-9]*$ ]]; then
      echo "VISION_MAX_IMAGES and VISION_MAX_PIXELS must be positive decimal integers" >&2
      exit 2
    fi
    if (( VISION_MAX_PIXELS < 65536 || VISION_MAX_PIXELS > 16777216 )); then
      echo "VISION_MAX_PIXELS must be between 65536 and 16777216" >&2
      exit 2
    fi
    modality_args=(
      --limit-mm-per-prompt "{\"image\":$VISION_MAX_IMAGES,\"video\":0}"
      --mm-processor-kwargs "{\"min_pixels\":65536,\"max_pixels\":$VISION_MAX_PIXELS}"
      --mm-encoder-tp-mode weights
    )
    ;;
  *) echo "ENABLE_VISION must be 0 or 1" >&2; exit 2 ;;
esac

case "$DISABLE_CUSTOM_ALL_REDUCE" in
  0) custom_all_reduce_arg= ;;
  1) custom_all_reduce_arg=--disable-custom-all-reduce ;;
  *)
    echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1" >&2
    exit 2
    ;;
esac

case "${QWEN38_ASYNC_SCHEDULING:-0}" in
  0) async_scheduling_arg=--no-async-scheduling ;;
  1) async_scheduling_arg=--async-scheduling ;;
  *) echo "QWEN38_ASYNC_SCHEDULING must be 0 or 1" >&2; exit 2 ;;
esac

allocator_config=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
expandable_segments=
IFS=',' read -r -a allocator_options <<< "$allocator_config"
for option in "${allocator_options[@]}"; do
  compact_option=${option//[[:space:]]/}
  case "$compact_option" in
    expandable_segments:True|expandable_segments:true)
      expandable_segments=True
      ;;
    expandable_segments:False|expandable_segments:false)
      expandable_segments=False
      ;;
  esac
done
if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == 0 && "$expandable_segments" == True ]]; then
  echo "DISABLE_CUSTOM_ALL_REDUCE=0 is incompatible with expandable_segments:True in this pinned runtime; set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False or keep DISABLE_CUSTOM_ALL_REDUCE=1" >&2
  exit 2
fi

model=${1:-/model}
mtp_model=${2:-"$model/runtime/mtp-int4-g32"}
rankings=/workspace/static_hot_cache_rankings.json

[[ -f "$model/model.safetensors.index.json" ]] || {
  echo "model checkpoint not found at $model" >&2
  exit 2
}
[[ -f "$rankings" ]] || { echo "missing hot-cache rankings: $rankings" >&2; exit 2; }
[[ -f "$mtp_model/model.safetensors.index.json" ]] || {
  echo "compact MTP checkpoint not found at $mtp_model" >&2
  exit 2
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PYTORCH_CUDA_ALLOC_CONF=$allocator_config
export VLLM_PLE_CPU_OFFLOAD=1
export VLLM_WNA16_STATIC_HOT_CACHE_FILE=$rankings
export VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1

hot_only=${QWEN38_HOT_ONLY:-0}
[[ "$hot_only" =~ ^[0-9]+$ ]] || { echo "QWEN38_HOT_ONLY must be a number of experts" >&2; exit 2; }
if (( hot_only > 0 )); then
  # 64 GB profile (configs/2x3090-64gb.env): each GPU owns QWEN38_HOT_ONLY experts per layer outright; the other
  # experts of its half live once in a RAM arena. Decode computes them on the CPU (plus a GPU share over PCIe),
  # prefill streams them to the GPU. The PLE table is read in place from the checkpoint on NVMe.
  [[ "$MAX_NUM_SEQS" == 1 ]] || { echo "MAX_NUM_SEQS must be 1 with QWEN38_HOT_ONLY" >&2; exit 2; }
  [[ "$MTP_DEPTH" =~ ^[0-3]$ ]] || { echo "MTP_DEPTH must be 0-3 with QWEN38_HOT_ONLY" >&2; exit 2; }
  export QWEN38_CPU_EXPERTS=1 QWEN38_CPU_EXPERTS_MODEL="$model"
  export QWEN38_PLE_MMAP=1 QWEN38_PLE_MMAP_DIR="$model" QWEN38_PLE_PREFAULT=0
  export QWEN38_STREAM_STAGE=0 QWEN38_STAGE_OVERLAP=0
  export VLLM_WNA16_DYNAMIC_LRU=0 VLLM_WNA16_MIXED_VMM_HOT_CACHE=0 VLLM_WNA16_STATIC_HOT_CACHE_SIZE=0
  offload_args=()
  # Host plan: resolve the "auto" CPU pools (one per GPU) and arena size once for every serving process.
  host_py=${QWEN38_HOST_PY:-$(python3 -c 'import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec("vllm").origin), "qwen38_host.py"))')}
  while IFS='=' read -r key value; do
    export "$key=$value"
  done < <(python3 "$host_py" --env --ranks 2)
  python3 "$host_py" --ranks 2 | sed 's/^/[qwen38-64gb] /'
else
  export VLLM_WNA16_DYNAMIC_LRU=1
  export VLLM_WNA16_MIXED_VMM_HOT_CACHE=1
  offload_args=(--offload-backend uva --cpu-offload-gb "$CPU_OFFLOAD_GB" --cpu-offload-params experts)
fi

exec vllm serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --moe-backend humming \
  --dtype bfloat16 \
  "${modality_args[@]}" \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS" \
  ${offload_args[@]+"${offload_args[@]}"} \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype "${KV_CACHE_DTYPE:-auto}" \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  "$async_scheduling_arg" \
  ${custom_all_reduce_arg:+"$custom_all_reduce_arg"} \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config \
  "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH,\"use_local_argmax_reduction\":true,\"model\":\"$mtp_model\"}"
