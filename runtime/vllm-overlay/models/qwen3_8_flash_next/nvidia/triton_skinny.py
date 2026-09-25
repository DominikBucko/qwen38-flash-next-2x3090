# SPDX-License-Identifier: Apache-2.0
"""Experimental SM86 skinny-M BF16 GEMM for decode (M <= 8), Triton tensor cores.

Pads M to 16 rows and streams each weight byte once; FP32 accumulation, BF16
output; deterministic split-K (fixed-order reduction). Used only through the
existing Qwen3.8 low-latency GEMM hook when QWEN38_TRITON_SKINNY=1.
"""
import math
import torch
import triton
import triton.language as tl

MAX_M = 8
_workspace = {}
_configs = {}


@triton.jit
def _reduce_kernel(part_ptr, out_ptr, M, N, stride_om, BLOCK_M: tl.constexpr,
                   BLOCK_N: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for s in range(SPLIT_K):
        acc += tl.load(part_ptr + s * (BLOCK_M * N) + offs_m[:, None] * N + offs_n[None, :],
                       mask=mask, other=0.0)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
             acc.to(out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _skinny_dot_kernel(x_ptr, w_ptr, out_ptr, part_ptr, M, N, K,
                       stride_xm, stride_wn, stride_om,
                       K_PER_SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
                       BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, STAGES: tl.constexpr):
    # Tensor-core variant: M padded to 16 rows; W tile [BLOCK_N, BLOCK_K].
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    k_start = pid_k * K_PER_SPLIT
    for k0 in tl.range(0, K_PER_SPLIT, BLOCK_K, num_stages=STAGES):
        offs_k = k_start + k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :],
                    mask=n_mask[:, None] & k_mask[None, :], other=0.0,
                    eviction_policy='evict_first')
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :],
                    mask=m_mask[:, None] & k_mask[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
    if SPLIT_K == 1:
        tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
                 acc.to(out_ptr.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])
    else:
        base = part_ptr + pid_k * (16 * N)
        tl.store(base + offs_m[:, None] * N + offs_n[None, :], acc,
                 mask=m_mask[:, None] & n_mask[None, :])



def config_for(n, k):
    key = (n, k)
    cfg = _configs.get(key)
    if cfg is not None:
        return cfg
    if n >= 2048:
        bk = 64 if k < 1024 else 128
        cfg = (32, bk, 1, 3 if n < 4096 else 2)
    else:
        bn, bk = 16, 128
        tiles = triton.cdiv(n, bn)
        split = 1
        target = 96.0 / tiles
        if target > 1.5:
            split = min(8, 1 << int(round(math.log2(target))))
        while split > 1 and k < 2 * bk * split:
            split //= 2
        cfg = (bn, bk, split, 4)
    _configs[key] = cfg
    return cfg


def skinny_mm(x, w):
    m, k = x.shape
    n = w.shape[0]
    bn, bk, split, stages = config_for(n, k)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    k_per_split = triton.cdiv(triton.cdiv(k, split), bk) * bk
    part = out
    if split > 1:
        key = (x.device, split * 16 * n)
        part = _workspace.get(key)
        if part is None:
            part = torch.empty(split * 16 * n, dtype=torch.float32, device=x.device)
            _workspace[key] = part
    _skinny_dot_kernel[(triton.cdiv(n, bn), split)](
        x, w, out, part, m, n, k, x.stride(0), w.stride(0), out.stride(0),
        K_PER_SPLIT=k_per_split, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=split,
        STAGES=stages, num_warps=4)
    if split > 1:
        _reduce_kernel[(triton.cdiv(n, 128),)](part, out, m, n, out.stride(0), BLOCK_M=16,
                                                BLOCK_N=128, SPLIT_K=split, num_warps=4)
    return out
