#!/usr/bin/env bash
set -euo pipefail

profile=/opt/qwen38/configs/2x3090-128gb.env
[[ -f "$profile" ]] || { echo "missing runtime profile: $profile" >&2; exit 2; }
# shellcheck source=/dev/null
source "$profile"

case "$DISABLE_CUSTOM_ALL_REDUCE" in
  0) custom_all_reduce_arg= ;;
  1) custom_all_reduce_arg=--disable-custom-all-reduce ;;
  *)
    echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1" >&2
    exit 2
    ;;
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
export VLLM_WNA16_DYNAMIC_LRU=1
export VLLM_WNA16_STATIC_HOT_CACHE_FILE=$rankings
export VLLM_WNA16_MIXED_VMM_HOT_CACHE=1
export VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1

exec vllm serve "$model" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 2 \
  --enable-expert-parallel \
  --all2all-backend allgather_reducescatter \
  --moe-backend humming \
  --dtype bfloat16 \
  --language-model-only \
  --load-format safetensors \
  --safetensors-load-strategy lazy \
  --max-parallel-loading-workers "$MAX_PARALLEL_LOADING_WORKERS" \
  --offload-backend uva \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --cpu-offload-params experts \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --kv-cache-dtype auto \
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --mamba-cache-mode align \
  --no-async-scheduling \
  ${custom_all_reduce_arg:+"$custom_all_reduce_arg"} \
  --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --speculative-config \
  "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP_DEPTH,\"use_local_argmax_reduction\":true,\"model\":\"$mtp_model\"}"
