#!/usr/bin/env bash
set -euo pipefail

image=${IMAGE:-qwen38-flash-next-2x3090:locked}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the assembled/downloaded model directory}
port=${PORT:-8000}
qsa_exact=${VLLM_QSA_EXACT_TOPK:-0}

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
  MTP_DEPTH
do
  if [[ -v "$name" ]]; then
    docker_env+=(-e "$name=${!name}")
  fi
done

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
  -v "$model_dir:/model:ro" \
  "$image" /model
