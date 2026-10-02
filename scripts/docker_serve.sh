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
  VLLM_MTP_DRAFT_VOCAB_RANGES \
  KV_CACHE_DTYPE \
  VLLM_API_KEY
do
  if declare -p "$name" &>/dev/null; then
    docker_env+=(-e "$name=${!name}")
  fi
done
# The 64 GB profile's QWEN38_* settings (hot-only experts, CPU pools, arena) are passed through as well.
while IFS= read -r name; do
  case " ${docker_env[*]} " in *" $name="*) continue ;; esac
  docker_env+=(-e "$name=${!name}")
done < <(compgen -e | grep -E '^QWEN38_[A-Z0-9_]+$' || true)

# Optional container limits. The 64 GB profile sets MEMORY_LIMIT=auto: installed RAM (MemTotal rounded up to a
# multiple of 8 GiB) minus HOST_RESERVE_GIB (default 8) for the OS; the runtime sizes its expert arenas from it.
# Unset keeps the 128 GB profiles unlimited (their loader relies on swap).
limit_args=()
if [[ -n "${MEMORY_LIMIT:-}" ]]; then
  memory_limit=$MEMORY_LIMIT
  if [[ "$memory_limit" == auto ]]; then
    mem_total_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    installed_gib=$(( (mem_total_kib + 8388607) / 8388608 * 8 ))
    memory_limit="$(( installed_gib - ${HOST_RESERVE_GIB:-8} ))g"
  fi
  limit_args+=(--memory "$memory_limit" --memory-swap "$memory_limit")
fi
[[ -n "${CPUSET:-}" ]] && limit_args+=(--cpuset-cpus "$CPUSET")

# Optional persistent JIT caches (Humming, Triton, TorchInductor): avoids
# recompiling kernels on every start and inside the first long request.
cache_mounts=()
if [[ -n "${JIT_CACHE_DIR:-}" ]]; then
  mkdir -p "$JIT_CACHE_DIR/humming" "$JIT_CACHE_DIR/triton" "$JIT_CACHE_DIR/inductor"
  jit_cache_dir=$(cd -- "$JIT_CACHE_DIR" && pwd -P)
  cache_mounts=(
    -v "$jit_cache_dir/humming:/root/.humming"
    -v "$jit_cache_dir/triton:/root/.triton"
    # TorchInductor-managed kernels (most of the torch.compile'd graph) cache
    # their Triton artifacts under $TORCHINDUCTOR_CACHE_DIR; without this they
    # recompile on every container restart.
    -v "$jit_cache_dir/inductor:/root/.torchinductor"
    -v "$jit_cache_dir/inductor:/tmp/torchinductor_root"
  )
  docker_env+=(-e "TORCHINDUCTOR_CACHE_DIR=/root/.torchinductor")
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
  ${limit_args[@]+"${limit_args[@]}"} \
  -p "127.0.0.1:$port:$port" \
  "${docker_env[@]}" \
  ${cache_mounts[@]+"${cache_mounts[@]}"} \
  -v "$model_dir:/model:ro" \
  "$image" /model
