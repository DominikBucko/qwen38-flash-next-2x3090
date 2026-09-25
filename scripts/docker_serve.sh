#!/usr/bin/env bash
set -euo pipefail

image=${IMAGE:-qwen38-flash-next-2x3090:locked}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the assembled/downloaded model directory}
port=${PORT:-8000}
qsa_exact=${VLLM_QSA_EXACT_TOPK:-0}

if [[ ${DISABLE_CUSTOM_ALL_REDUCE+x} ]]; then
  case "$DISABLE_CUSTOM_ALL_REDUCE" in
    0|1) ;;
    *)
      echo "DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1" >&2
      exit 2
      ;;
  esac
fi

docker_env=(
  -e "PORT=$port"
  -e "VLLM_QSA_EXACT_TOPK=$qsa_exact"
)
for name in \
  SERVED_MODEL_NAME \
  MAX_MODEL_LEN \
  MAX_NUM_BATCHED_TOKENS \
  MAX_PARALLEL_LOADING_WORKERS \
  MAX_NUM_SEQS \
  KV_CACHE_MEMORY_BYTES \
  CPU_OFFLOAD_GB \
  VLLM_PLE_OFFLOAD_READY_TIMEOUT \
  VLLM_WNA16_STATIC_HOT_CACHE_SIZE \
  VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS \
  VLLM_PREFIX_CACHE_RETENTION_INTERVAL \
  DISABLE_CUSTOM_ALL_REDUCE \
  PYTORCH_CUDA_ALLOC_CONF \
  ENABLE_VISION \
  VISION_MAX_IMAGES \
  VISION_MAX_PIXELS \
  MTP_DEPTH \
  QWEN38_ASYNC_SCHEDULING \
  QWEN38_STREAM_STAGE \
  QWEN38_STREAM_STAGE_MIN_TOKENS \
  QWEN38_STAGE_OVERLAP \
  QWEN38_TRITON_SKINNY \
  QWEN38_PLE_PREFAULT \
  QWEN38_PLE_PREFAULT_RESERVE_GIB \
  VLLM_MTP_DRAFT_VOCAB_RANGES
do
  if declare -p "$name" &>/dev/null; then
    docker_env+=(-e "$name=${!name}")
  fi
done

# Optional persistent Humming/Triton JIT caches: avoids recompiling kernels on
# every start and inside the first long request.
cache_mounts=()
if [[ -n "${JIT_CACHE_DIR:-}" ]]; then
  mkdir -p "$JIT_CACHE_DIR/humming" "$JIT_CACHE_DIR/triton"
  jit_cache_dir=$(cd -- "$JIT_CACHE_DIR" && pwd -P)
  cache_mounts=(
    -v "$jit_cache_dir/humming:/root/.humming"
    -v "$jit_cache_dir/triton:/root/.triton"
  )
fi

model_dir=$(cd -- "$model_dir" && pwd -P)
[[ -f "$model_dir/model.safetensors.index.json" ]] || {
  echo "MODEL_DIR is not a model checkpoint: $model_dir" >&2
  exit 2
}

exec docker run --rm \
  --name qwen38-flash-next \
  --gpus all \
  --ipc host \
  --cap-add SYS_PTRACE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p "127.0.0.1:$port:$port" \
  "${docker_env[@]}" \
  ${cache_mounts[@]+"${cache_mounts[@]}"} \
  -v "$model_dir:/model:ro" \
  "$image" /model
