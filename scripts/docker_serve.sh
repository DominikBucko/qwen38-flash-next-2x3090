#!/usr/bin/env bash
set -euo pipefail

image=${IMAGE:-qwen38-flash-next-2x3090:locked}
model_dir=${MODEL_DIR:?Set MODEL_DIR to the assembled/downloaded model directory}
port=${PORT:-8000}
qsa_exact=${VLLM_QSA_EXACT_TOPK:-0}

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
  -e "PORT=$port" \
  -e "VLLM_QSA_EXACT_TOPK=$qsa_exact" \
  -v "$model_dir:/model:ro" \
  "$image" /model
