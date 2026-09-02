#!/usr/bin/env bash
set -euo pipefail

fail=0
warn() { printf 'WARN: %s\n' "$*" >&2; }
error() { printf 'ERROR: %s\n' "$*" >&2; fail=1; }

[[ "$(uname -s)" == Linux ]] || error "serving requires Linux"
[[ "$(uname -m)" == x86_64 ]] || error "the packaged image was validated on x86_64"
command -v docker >/dev/null || error "docker is not installed"
command -v nvidia-smi >/dev/null || error "nvidia-smi is not installed"

if command -v nvidia-smi >/dev/null; then
  gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
  [[ "$gpu_count" -ge 2 ]] || error "two CUDA GPUs are required by this profile"
  nvidia-smi --query-gpu=index,name,memory.total \
    --format=csv,noheader
fi

if [[ -r /proc/meminfo ]]; then
  mem_kib=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
  [[ "$mem_kib" -ge 125000000 ]] || error "this profile needs approximately 128 GiB RAM"
  swap_kib=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
  if [[ "$swap_kib" -lt 33554432 ]]; then
    warn "only $((swap_kib / 1024 / 1024)) GiB swap is configured; use at least 32 GiB of fast NVMe swap (48-64 GiB recommended) for load-time headroom"
  elif [[ "$swap_kib" -lt 50331648 ]]; then
    warn "$((swap_kib / 1024 / 1024)) GiB swap is configured; 48-64 GiB gives safer load-time headroom"
  fi
fi

if [[ -r /proc/sys/kernel/yama/ptrace_scope ]]; then
  ptrace_scope=$(< /proc/sys/kernel/yama/ptrace_scope)
  [[ "$ptrace_scope" -eq 0 ]] || warn \
    "ptrace_scope=$ptrace_scope; --cap-add SYS_PTRACE usually suffices, but PLE IPC may require a temporary scope=0 on some hosts"
fi

if command -v docker >/dev/null && ! docker info >/dev/null 2>&1; then
  error "Docker daemon is unavailable to the current user"
fi

(( fail == 0 )) || exit 1
printf 'preflight passed\n'
