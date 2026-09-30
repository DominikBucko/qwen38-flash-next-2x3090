# SPDX-License-Identifier: Apache-2.0
"""Decode (T <= 4, hot-only layers): the GPU computes a share of each layer's cold experts itself, straight from
the pinned host arena over PCIe, while the CPU computes the rest.

Per layer, inside the CUDA graph: a one-program selection kernel takes up to ceil(FRAC x n_cold) of the step's
distinct cold experts (first-appearance order; only "protected" arena experts, whose slots the dynamic arena
never evicts), and writes a per-(token, k) mask that the bridge applies to the CPU request (masked ids become -1,
which the CPU skips). Two Triton kernels then read those experts' GPTQ INT4 weights and F16 scales from host
memory (UVA addresses), compute SiLU(x Wg) * (x Wu) and the down projection with FP32 accumulation, and add the
router-weighted result into an FP32 buffer that the bridge's wait kernel adds together with the CPU's part.
"""
import os

import torch

from vllm.triton_utils import tl, triton

FRAC = float(os.environ.get("QWEN38_GPU_SHARE_FRAC", "0"))
K_MAX = int(os.environ.get("QWEN38_GPU_SHARE_KMAX", "8"))
H, I, TOPK, MAXT = 2560, 640, 10, 4
# slot layout (expert_store.PARTS): gate q, up q, down q, gate s, up s, down s
OFF_GQ, OFF_UQ, OFF_DQ, OFF_GS, OFF_US, OFF_DS = 0, 819200, 1638400, 2457600, 2483200, 2508800


def enabled() -> bool:
    return FRAC > 0 and K_MAX > 0


@triton.jit
def _select_kernel(ids_ptr, hot_ptr, addr_ptr, sel_ptr, mask_ptr, n_pairs, frac,
                   P: tl.constexpr, K_MAX: tl.constexpr):
    offs = tl.arange(0, P)
    valid = offs < n_pairs
    e = tl.load(ids_ptr + offs, mask=valid, other=-1).to(tl.int32)
    es = tl.maximum(e, 0)
    hot = tl.load(hot_ptr + es, mask=valid, other=0) >= 0
    addr = tl.load(addr_ptr + es, mask=valid, other=0)
    cold = valid & (e >= 0) & (hot == 0)
    earlier = offs[None, :] < offs[:, None]
    dup = tl.sum(((e[:, None] == e[None, :]) & cold[None, :] & earlier).to(tl.int32), axis=1) > 0
    first = cold & (dup == 0)
    n_cold = tl.sum(first.to(tl.int32), axis=0)
    elig = first & (addr != 0)
    n_elig = tl.sum(elig.to(tl.int32), axis=0)
    k = tl.minimum(tl.minimum((n_cold.to(tl.float32) * frac + 0.999).to(tl.int32), K_MAX), n_elig)
    rank = tl.cumsum(elig.to(tl.int32), axis=0) - 1
    chosen = elig & (rank < k)
    slot = tl.arange(0, K_MAX)
    tl.store(sel_ptr + slot, tl.full((K_MAX,), -1, tl.int32))
    tl.debug_barrier()
    tl.store(sel_ptr + rank, e, mask=chosen)
    chosen_e = tl.where(chosen, e, -2)
    m = tl.sum(((e[:, None] == chosen_e[None, :]) & cold[:, None]).to(tl.int32), axis=1) > 0
    tl.store(mask_ptr + offs, m.to(tl.int8), mask=valid)


@triton.jit
def _share_up_kernel(x_ptr, sel_ptr, addr_ptr, h_ptr, T,
                     H: tl.constexpr, I: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, MT: tl.constexpr):
    k = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(sel_ptr + k)
    if e >= 0:
        base = tl.load(addr_ptr + e)
        gq = (base + 0).to(tl.pointer_type(tl.int32), bitcast=True)
        uq = (base + 819200).to(tl.pointer_type(tl.int32), bitcast=True)
        gs = (base + 2457600).to(tl.pointer_type(tl.float16), bitcast=True)
        us = (base + 2483200).to(tl.pointer_type(tl.float16), bitcast=True)
        cols = pid_n * BN + tl.arange(0, BN)
        rows = tl.arange(0, BR)
        tok = tl.arange(0, MT)
        kk = tl.arange(0, 8 * BR)
        sh = tl.arange(0, 8) * 4
        acc_g = tl.zeros((MT, BN), tl.float32)
        acc_u = tl.zeros((MT, BN), tl.float32)
        for r0 in range(0, H // 8, BR):
            # BR packed rows = 8*BR consecutive k values, all in one 128-wide scale group (BR divides 16).
            # Each packed int32 is read from host memory once and expanded to its 8 nibbles in registers.
            qidx = (r0 + rows)[:, None] * I + cols[None, :]
            qg = tl.reshape((tl.load(gq + qidx)[:, None, :] >> sh[None, :, None]) & 0xF, (8 * BR, BN))
            qu = tl.reshape((tl.load(uq + qidx)[:, None, :] >> sh[None, :, None]) & 0xF, (8 * BR, BN))
            sg = tl.load(gs + (r0 // 16) * I + cols).to(tl.float32)
            su = tl.load(us + (r0 // 16) * I + cols).to(tl.float32)
            wg = (qg.to(tl.float32) - 8.0) * sg[None, :]
            wu = (qu.to(tl.float32) - 8.0) * su[None, :]
            xt = tl.load(x_ptr + tok[:, None] * H + (8 * r0 + kk)[None, :], mask=(tok < T)[:, None], other=0.0)
            xt = xt.to(tl.float16)
            acc_g = tl.dot(xt, wg.to(tl.float16), acc_g)
            acc_u = tl.dot(xt, wu.to(tl.float16), acc_u)
        hval = acc_g / (1.0 + tl.exp(-acc_g)) * acc_u
        tl.store(h_ptr + (k * MT + tok[:, None]) * I + cols[None, :], hval, mask=(tok < T)[:, None])


@triton.jit
def _share_down_kernel(h_ptr, sel_ptr, addr_ptr, ids_ptr, w_ptr, y_ptr, T,
                       H: tl.constexpr, I: tl.constexpr, TOPK: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr,
                       MT: tl.constexpr):
    k = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(sel_ptr + k)
    if e >= 0:
        base = tl.load(addr_ptr + e)
        dq = (base + 1638400).to(tl.pointer_type(tl.int32), bitcast=True)
        ds = (base + 2508800).to(tl.pointer_type(tl.float16), bitcast=True)
        cols = pid_n * BN + tl.arange(0, BN)
        rows = tl.arange(0, BR)
        tok = tl.arange(0, MT)
        kk = tl.arange(0, 8 * BR)
        sh = tl.arange(0, 8) * 4
        acc = tl.zeros((MT, BN), tl.float32)
        for r0 in range(0, I // 8, BR):
            qidx = (r0 + rows)[:, None] * H + cols[None, :]
            q = tl.reshape((tl.load(dq + qidx)[:, None, :] >> sh[None, :, None]) & 0xF, (8 * BR, BN))
            s = tl.load(ds + (r0 // 16) * H + cols).to(tl.float32)
            w = (q.to(tl.float32) - 8.0) * s[None, :]
            ht = tl.load(h_ptr + (k * MT + tok[:, None]) * I + (8 * r0 + kk)[None, :], mask=(tok < T)[:, None],
                         other=0.0)
            acc = tl.dot(ht.to(tl.float16), w.to(tl.float16), acc)
        # router weight of expert e for each token (0 if the token did not pick it)
        kt = tl.arange(0, 16)
        pm = (tok[:, None] < T) & (kt[None, :] < TOPK)
        tid = tl.load(ids_ptr + tok[:, None] * TOPK + kt[None, :], mask=pm, other=-1)
        tw = tl.load(w_ptr + tok[:, None] * TOPK + kt[None, :], mask=pm, other=0.0).to(tl.float32)
        wt = tl.sum(tl.where(tid == e, tw, 0.0), axis=1)
        tl.atomic_add(y_ptr + tok[:, None] * H + cols[None, :], acc * wt[:, None], mask=(tok < T)[:, None])


class Share:
    def __init__(self, device, addr: torch.Tensor):
        self.addr = addr.to(device)  # [L, E] int64: UVA address of protected arena slots, 0 otherwise
        self.sel = torch.full((K_MAX,), -1, dtype=torch.int32, device=device)
        self.mask = torch.zeros(64, dtype=torch.int8, device=device)
        self.h = torch.zeros(K_MAX, 16, I, dtype=torch.float32, device=device)
        self.y = torch.zeros(MAXT, H, dtype=torch.float32, device=device)
        # Side stream: the share kernels (PCIe-bound, few SMs) run next to the hot-expert GEMM.
        self.stream = torch.cuda.Stream(device=device) if os.environ.get("QWEN38_GPU_SHARE_STREAM", "1") == "1" \
            else None

    def launch(self, layer: int, x, topk_ids, topk_weights) -> None:
        """Fork the share onto the side stream (join with join() before the bridge wait)."""
        if self.stream is None:
            self.compute(layer, x, topk_ids, topk_weights)
            return
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        with torch.cuda.stream(self.stream):
            self.compute(layer, x, topk_ids, topk_weights)

    def join(self) -> None:
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

    def select(self, layer: int, topk_ids: torch.Tensor, hot_map: torch.Tensor) -> None:
        n = topk_ids.numel()
        _select_kernel[(1,)](topk_ids, hot_map, self.addr[layer], self.sel, self.mask, n, FRAC,
                             P=64, K_MAX=K_MAX, num_warps=2)

    def compute(self, layer: int, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor) -> None:
        T = x.shape[0]
        self.y.zero_()
        addr = self.addr[layer]
        _share_up_kernel[(K_MAX, I // 64)](x, self.sel, addr, self.h, T, H=H, I=I, BN=64, BR=4, MT=16,
                                            num_warps=4, num_stages=3)
        _share_down_kernel[(K_MAX, H // 128)](self.h, self.sel, addr, topk_ids, topk_weights, self.y, T,
                                              H=H, I=I, TOPK=TOPK, BN=128, BR=4, MT=16, num_warps=4,
                                              num_stages=3)


_SHARE = None


def get(device=None):
    global _SHARE
    if _SHARE is None and enabled():
        from . import expert_store
        store = expert_store.get()
        _SHARE = Share(device or torch.device("cuda"), store.gpu_addr)
    return _SHARE
