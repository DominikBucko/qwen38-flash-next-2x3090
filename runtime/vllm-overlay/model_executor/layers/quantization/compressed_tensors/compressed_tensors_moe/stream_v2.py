# SPDX-License-Identifier: Apache-2.0
"""Prefill for hot-only layers: stream the cold experts from the host arena and convert them on the GPU.

Per MoE layer the GPU computes its hot experts from its own weights, then the cold experts in groups of
QWEN38_STREAM_GROUP: a copy stream DMAs a group's raw checkpoint bytes (GPTQ int32 K-first qweight + F16 scales,
see expert_store.py) into one of two raw buffers, the compute stream converts them to Humming's layout exactly as
the loader does (K-first -> N-first, weight repack, BF16 scales with the kernel's scale permutation), and runs the
original Humming GEMMs with an expert map that selects only that group. Every group, hot and cold, adds its
weighted expert outputs into one FP32 accumulator, which is rounded to BF16 once, as in the all-GPU path.
DMAs run up to two groups ahead, across layer and step boundaries.
"""
import copy
import os
from dataclasses import replace
from types import SimpleNamespace

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from . import expert_store
from .expert_store import SLOT

logger = init_logger(__name__)

GROUP = int(os.environ.get("QWEN38_STREAM_GROUP", "128"))


def _wait_event(ev) -> None:
    """Wait for a CUDA event by polling: Event.synchronize() spin-waits on a whole core by default, and on machines
    with 16 cores or fewer that core is needed by the thread that issues the prefill kernels."""
    import time
    while not ev.query():
        time.sleep(0.0002)
H, I, GS = 2560, 640, 128
_layers: dict[int, SimpleNamespace] = {}
_state = None


@triton.jit
def _mul_sum_acc_kernel(inputs_ptr, weights_ptr, ids_ptr, emap_ptr, acc_ptr, num_tokens,
                        top_k: tl.constexpr, size: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_k = tl.program_id(0)
    pid_m = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    m_mask = offs_m < num_tokens
    mask = m_mask[:, None] & (offs_k < size)[None, :]
    acc_ptrs = acc_ptr + (offs_m * size)[:, None] + offs_k[None, :]
    acc = tl.load(acc_ptrs, mask=mask, other=0.0)
    for n in tl.static_range(top_k):
        w = tl.load(weights_ptr + offs_m * top_k + n, mask=m_mask, other=0.0).to(tl.float32)
        ids = tl.load(ids_ptr + offs_m * top_k + n, mask=m_mask, other=-1)
        # ids < 0 are unrouted pairs (vLLM's profile run sets every id to -1 under expert parallelism)
        ok = tl.load(emap_ptr + ids, mask=m_mask & (ids >= 0), other=-1) >= 0
        a = tl.load(inputs_ptr + (offs_m * (top_k * size) + n * size)[:, None] + offs_k[None, :],
                    mask=mask & ok[:, None], other=0.0).to(tl.float32)
        acc += a * w[:, None]
    tl.store(acc_ptrs, acc, mask=mask)


def _mul_sum_acc(down, topk_weights, topk_ids, emap, acc):
    """acc[m] += sum_n w[m,n] * down[m,n] over the (m,n) whose expert emap keeps (FP32)."""
    m, top_k, size = down.shape
    block_m, block_k = 4, 512
    grid = (triton.cdiv(size, block_k), triton.cdiv(m, block_m))
    _mul_sum_acc_kernel[grid](down, topk_weights, topk_ids, emap, acc, m, top_k=top_k, size=size,
                              BLOCK_M=block_m, BLOCK_K=block_k, num_warps=4)


def _run_experts(experts, x, topk_ids, topk_weights, w13, w2, emap, acc, activation):
    """The Humming indexed schedule of stream_stage/tiered, reduced into the FP32 accumulator."""
    m, top_k = topk_ids.shape
    ws1, ws2, _ = experts.make_workspaces(m, top_k, activation)
    buffers = experts.prepare_buffers(ws1, ws2, m, top_k, activation)
    kwargs1, kwargs2 = experts.prepare_humming_moe_kwargs(topk_ids=topk_ids, expert_map=emap,
                                                          expert_tokens_meta=None)
    inputs, input_scale = experts.quantize_input("w13", inputs=x,
                                                 quanted_input=buffers.get("quanted_gate_up_input", None))
    experts.humming_forward("w13", inputs=inputs, weight=w13, input_scale=input_scale,
                            outputs=buffers["gate_up_output"], **kwargs1)
    experts.apply_activation(activation=activation, input=buffers["gate_up_output"],
                             output=buffers["activation_output"])
    inputs, input_scale = experts.quantize_input("w2", inputs=buffers["activation_output"],
                                                 quanted_input=buffers.get("quanted_down_input", None))
    experts.humming_forward("w2", inputs=inputs, weight=w2, input_scale=input_scale,
                            outputs=buffers["down_output"].view(-1, x.size(-1)), **kwargs2)
    _mul_sum_acc(buffers["down_output"].view(m, top_k, -1), topk_weights, topk_ids, emap, acc)


def register(method, layer, index: int, hot_ids: list[int]) -> None:
    """Called once per hot-only target layer after its weights are converted."""
    base = method.moe_kernel.fused_experts
    for name in ("w13", "w2"):
        cfg = base.humming_configs[name]
        if cfg.has_zero_point or cfg.use_fused_e8m0_scale or cfg.has_bias or cfg.mma_type.value.lower() != "mma":
            raise RuntimeError(f"stream_v2: unsupported Humming config for {name}: {cfg}")
    staged = copy.copy(base)
    staged.num_experts = GROUP
    staged.humming_configs = {name: replace(cfg, num_experts=GROUP) for name, cfg in base.humming_configs.items()}
    _layers[index] = SimpleNamespace(index=index, method=method, layer=layer, base=base, staged=staged,
                                     hot_ids=list(hot_ids), groups=None)


def _perm(to_apply_on_c: bool, device) -> torch.Tensor:
    # humming.transform.transform_humming_weight_scale, precomputed once
    perm = [0, 1, 8, 9, 16, 17, 24, 25] if to_apply_on_c else [0, 8, 16, 24, 32, 40, 48, 56]
    count = sum(x < 8 for x in perm)
    out = []
    for i in range(8 // count):
        out += [x + count * i for x in perm]
    return torch.tensor(out, dtype=torch.long, device=device)


class Converter:
    """Checkpoint slot bytes -> the loader's Humming tensors, for up to `rows` experts at a time."""

    def __init__(self, device, rows: int, apply_on_c13: bool, apply_on_c2: bool):
        from humming.kernel.repack_weight import RepackWeightKernel
        self.rows = rows
        self.perm13 = _perm(apply_on_c13, device)
        self.perm2 = _perm(apply_on_c2, device)
        self.repack = RepackWeightKernel(weight_bits=4, activation_bits=16, is_weight_packed=True,
                                         should_preprocess_for_int2fp=False, should_preprocess_with_zp=False,
                                         use_wgmma=False, use_fused_e8m0_scale=False, group_size_zp=0,
                                         use_packed_k_layout=False)
        S = rows
        # Used in order on one stream only.
        self.w13n = torch.empty(S, 2 * I, H // 8, dtype=torch.int32, device=device)
        self.w2n = torch.empty(S, H, I // 8, dtype=torch.int32, device=device)
        self.w13h = torch.empty(S, H // 16, 2 * I * 2, dtype=torch.int32, device=device)
        self.w2h = torch.empty(S, I // 16, H * 2, dtype=torch.int32, device=device)
        self.s13k = torch.empty(S, H // GS, 2 * I, dtype=torch.bfloat16, device=device)
        self.s2k = torch.empty(S, I // GS, H, dtype=torch.bfloat16, device=device)
        self.s13h = torch.empty_like(self.s13k)
        self.s2h = torch.empty_like(self.s2k)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.w13n, self.w2n, self.w13h, self.w2h,
                                                           self.s13k, self.s2k, self.s13h, self.s2h))

    def __call__(self, raw: torch.Tensor, rows: int):
        """Rows beyond `rows` keep stale data; their experts are never selected."""
        r = raw.view(-1, SLOT)[:rows]
        gq = r[:, 0:819200].view(torch.int32).view(rows, H // 8, I)
        uq = r[:, 819200:1638400].view(torch.int32).view(rows, H // 8, I)
        dq = r[:, 1638400:2457600].view(torch.int32).view(rows, I // 8, H)
        gs = r[:, 2457600:2483200].view(torch.float16).view(rows, H // GS, I)
        us = r[:, 2483200:2508800].view(torch.float16).view(rows, H // GS, I)
        ds = r[:, 2508800:2534400].view(torch.float16).view(rows, I // GS, H)
        # vLLM's loader: w13 = [gate | up] along N, K-first; the K-first schema transposes to N-first.
        self.w13n[:rows, :I].copy_(gq.transpose(1, 2))
        self.w13n[:rows, I:].copy_(uq.transpose(1, 2))
        self.w2n[:rows].copy_(dq.transpose(1, 2))
        self.repack(inputs=self.w13n, outputs=self.w13h, zero_point=None, interleave_mode=3)
        self.repack(inputs=self.w2n, outputs=self.w2h, zero_point=None, interleave_mode=3)
        # Scales: F16 -> BF16 (the parameter dtype), then Humming's permutation of groups of N.
        self.s13k[:rows, :, :I].copy_(gs)
        self.s13k[:rows, :, I:].copy_(us)
        self.s2k[:rows].copy_(ds)
        torch.index_select(self.s13k.view(-1, self.perm13.numel()), 1, self.perm13,
                           out=self.s13h.view(-1, self.perm13.numel()))
        torch.index_select(self.s2k.view(-1, self.perm2.numel()), 1, self.perm2,
                           out=self.s2h.view(-1, self.perm2.numel()))
        return self.w13h, self.w2h, self.s13h, self.s2h


class _TailReader:
    """Reads a layer's tail experts (those not in the arena) from the checkpoint into pinned bounce buffers, in
    slot layout, a few layers ahead of the GPU. Reads go through the page cache, the tail's RAM tier. Requests
    carry the step's plan generation and the expert list, because the dynamic arena changes the tail per step."""

    def __init__(self, store, nbuf: int = 4, threads: int = 8):
        import mmap
        import queue
        import threading
        self.store = store
        self.nbuf = nbuf
        self.maxc = max(1, max(len(t) for t in store.tail))
        self.size = self.maxc * SLOT
        self.mem = mmap.mmap(-1, nbuf * self.size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                             prot=mmap.PROT_READ | mmap.PROT_WRITE)
        self.mem.madvise(mmap.MADV_HUGEPAGE)
        self.bytes = torch.frombuffer(self.mem, dtype=torch.uint8)
        self.bytes.zero_()
        err = torch.cuda.cudart().cudaHostRegister(self.bytes.data_ptr(), nbuf * self.size, 0)
        if int(err) != 0:
            raise RuntimeError(f"tail reader: cudaHostRegister failed ({err})")
        self.owner = [None] * nbuf     # (gen, layer) whose tail the buffer holds
        self.pending = [None] * nbuf
        self.done_ev = [None] * nbuf
        self.cv = threading.Condition()
        self.q = queue.Queue()
        from vllm import qwen38_host
        from vllm.model_executor.layers.fused_moe.hot_only import rank_and_size
        from .cpu_experts import extension
        spec = os.environ.get("QWEN38_TAIL_READER_CPUS")
        rank, ranks = rank_and_size()
        self.cpus = set(qwen38_host.parse_cpus(spec) if spec else qwen38_host.pool_cpus(rank=rank, ranks=ranks))
        self.threads = threads
        self.ext = extension()      # tail_read: one GIL-free call per layer
        self.addr = self.bytes.data_ptr()
        self.read_s = 0.0
        self.thread = threading.Thread(target=self._run, name="qwen38-tail-reader", daemon=True)
        self.thread.start()

    def request(self, gen: int, layer: int, experts: list[int], keep=()) -> None:
        if not experts:
            return
        if len(experts) > self.maxc:
            raise RuntimeError(f"tail reader: {len(experts)} tail experts exceed the buffer ({self.maxc})")
        b = layer % self.nbuf
        key = (gen, layer)
        with self.cv:
            if self.pending[b] == key or self.owner[b] == key:
                return
            self.pending[b] = key
        self.q.put((gen, layer, list(experts), set(keep)))

    DROP = os.environ.get("QWEN38_TAIL_DROP_CACHE", "1") == "1"

    def _parts(self, layer: int, experts: list[int], keep: set, base: int) -> torch.Tensor:
        """[fd, file offset, bytes, destination offset, drop page cache] for every part of the listed experts.
        Streamed parts leave the page cache (it holds PLE rows and decode-time tail reads); kept ones stay."""
        drop = 1 if self.DROP else 0
        loc = self.store.loc[layer]
        rows = []
        for i, e in enumerate(experts):
            d = 0 if e in keep else drop
            dst = base + i * SLOT
            for fd, off, size, soff in loc[e]:
                rows.append((fd, off, size, dst + soff, d))
        return torch.tensor(rows, dtype=torch.int64)

    def _run(self):
        import time
        os.sched_setaffinity(0, self.cpus)  # the extension's read threads inherit this
        while True:
            gen, layer, experts, keep = self.q.get()
            b = layer % self.nbuf
            ev = self.done_ev[b]
            if ev is not None:
                _wait_event(ev)  # the DMA that reads this buffer's previous content has finished
            t0 = time.perf_counter()
            self.ext.tail_read(self.addr, self._parts(layer, experts, keep, b * self.size), self.threads)
            self.read_s += time.perf_counter() - t0
            with self.cv:
                self.owner[b] = (gen, layer)
                if self.pending[b] == (gen, layer):
                    self.pending[b] = None
                self.cv.notify_all()

    def ready(self, gen: int, layer: int) -> bool:
        return self.owner[layer % self.nbuf] == (gen, layer)

    def wait(self, gen: int, layer: int, count: int) -> torch.Tensor:
        b = layer % self.nbuf
        with self.cv:
            while self.owner[b] != (gen, layer):
                self.cv.wait()
        return self.bytes[b * self.size:b * self.size + count * SLOT]


def _runs(slots: list[int]) -> list[tuple[int, int]]:
    """Consecutive slot runs of an ascending slot list: [(first_slot, count)]."""
    out = []
    for sl in slots:
        if out and out[-1][0] + out[-1][1] == sl:
            out[-1] = (out[-1][0], out[-1][1] + 1)
        else:
            out.append((sl, 1))
    return out


class _State:
    def __init__(self, device):
        import numpy as np
        self.device = device
        self.store = expert_store.get()
        any_layer = _layers[min(_layers)]
        cfg13, cfg2 = any_layer.base.humming_configs["w13"], any_layer.base.humming_configs["w2"]
        S = GROUP
        self.convert = Converter(device, S, cfg13.should_apply_bs_on_c, cfg2.should_apply_bs_on_c)
        self.stream = torch.cuda.Stream(device=device)
        self.raw = [torch.empty(S * SLOT, dtype=torch.uint8, device=device) for _ in range(2)]
        self.copy_done = [torch.cuda.Event() for _ in range(2)]
        self.raw_free = [torch.cuda.Event() for _ in range(2)]
        self.loaded = [None, None]
        self.order = sorted(_layers)
        # Per-layer arena and tail sizes never change (the dynamic arena replaces within a layer), so the number
        # of groups per step is fixed; their members come from a snapshot of the mapping taken at step start.
        rows = sum((len(self.store.cold[i]) + S - 1) // S + 1 for i in self.order)
        self.emap_host = [torch.full((rows, 512), -1, dtype=torch.int32).pin_memory() for _ in range(2)]
        self.emap_host_np = [t.numpy() for t in self.emap_host]
        self.emap_copied = [None, None]
        self.emap_dev = torch.full((rows, 512), -1, dtype=torch.int32, device=device)
        self.gen = 0
        self.items = []
        self.tail = None
        if any(self.store.tail):
            self.tail = _TailReader(self.store, nbuf=TAIL_BUFFERS)
            if self.tail.maxc > S:
                raise RuntimeError(f"stream_v2: {self.tail.maxc} tail experts per layer exceed the group size {S}")
            self.raw_tail = torch.empty(self.tail.maxc * SLOT, dtype=torch.uint8, device=device)
            self.tail_stream = torch.cuda.Stream(device=device)  # no head-of-line blocking with arena DMAs
            self.tail_copy_done = torch.cuda.Event()
            self.tail_raw_free = torch.cuda.Event()
            self.tail_issued = None
        self.prefill_event = None
        self.staged_calls = 0
        self.plan()
        mib = (sum(t.numel() for t in self.raw) + self.convert.nbytes()
               + (self.raw_tail.numel() if self.tail is not None else 0)) / 2**20
        logger.info("Stream v2: %d layers, %d groups of <= %d cold experts, GPU buffers %.0f MiB",
                    len(_layers), len(self.items), S, mib)

    def plan(self) -> None:
        """Snapshot the arena mapping (stable: the CPU server does not promote during a prefill step) and build
        this step's DMA groups, tail lists and expert maps."""
        import numpy as np
        S = GROUP
        self.gen += 1
        hb = self.gen % 2
        if self.emap_copied[hb] is not None:
            _wait_event(self.emap_copied[hb])  # its previous H2D copy has finished (a whole step ago)
        host = self.emap_host_np[hb]
        host.fill(-1)
        slot_of = self.store.slot_of.numpy().copy()
        counts = _usage.keep_sets() if _usage is not None else None
        self.items = []
        row = 0
        for index in self.order:
            entry = _layers[index]
            cold = self.store.cold_all[index]
            so = slot_of[index]
            arena = sorted((int(so[e]), e) for e in cold if so[e] >= 0)
            tail = [e for e in cold if so[e] < 0]
            entry.groups = []
            for first in range(0, len(arena), S):
                chunk = arena[first:first + S]
                experts = [e for _, e in chunk]
                host[row, experts] = np.arange(len(chunk), dtype=np.int32)
                entry.groups.append(SimpleNamespace(pos=len(self.items), row=row, count=len(chunk)))
                self.items.append((index, _runs([sl for sl, _ in chunk]), len(chunk)))
                row += 1
            entry.tail_list = tail
            entry.tail_keep = set()
            if counts is not None and tail and TAIL_KEEP > 0:
                c = counts[index]
                entry.tail_keep = {e for e in sorted(tail, key=lambda e: -c[e])[:TAIL_KEEP] if c[e] > 0}
            entry.tail_row = None
            if tail:
                host[row, tail] = np.arange(len(tail), dtype=np.int32)
                entry.tail_row = row
                row += 1
        self.emap_dev[:row].copy_(self.emap_host[hb][:row], non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.emap_copied[hb] = ev
        self.loaded = [None, None]
        if self.tail is not None:
            self.tail_issued = None

    def issue(self, pos: int) -> None:
        if _prof is not None:
            import time
            t0 = time.perf_counter()
            self._issue(pos)
            _prof.host_issue += time.perf_counter() - t0
        else:
            self._issue(pos)

    def _issue(self, pos: int) -> None:
        if pos >= len(self.items):
            return  # no prefetch into the next step: its plan (and the arena mapping) is not known yet
        b = pos % 2
        if self.loaded[b] == pos:
            return
        index, runs, count = self.items[pos]
        self.stream.wait_event(self.raw_free[b])
        with torch.cuda.stream(self.stream):
            if _prof is not None:
                start = torch.cuda.Event(enable_timing=True)
                start.record(self.stream)
            off = 0
            for first, n in runs:
                self.raw[b][off:off + n * SLOT].copy_(self.store.bytes[first * SLOT:(first + n) * SLOT],
                                                     non_blocking=True)
                off += n * SLOT
            self.copy_done[b].record(self.stream)
            if _prof is not None:
                end = torch.cuda.Event(enable_timing=True)
                end.record(self.stream)
                _prof.copies.append((start, end))
        self.loaded[b] = pos

    def request_tail(self, index: int) -> None:
        entry = _layers[index]
        if entry.tail_list:
            self.tail.request(self.gen, index, entry.tail_list, entry.tail_keep)

    def issue_tail(self, index: int, block: bool) -> bool:
        """DMA layer `index`'s tail group (bounce buffer -> raw_tail); returns False if not ready and not blocking."""
        key = (self.gen, index)
        if self.tail_issued == key:
            return True
        if not block and not self.tail.ready(self.gen, index):
            return False
        src = self.tail.wait(self.gen, index, len(_layers[index].tail_list))
        self.tail_stream.wait_event(self.tail_raw_free)
        with torch.cuda.stream(self.tail_stream):
            self.raw_tail[:src.numel()].copy_(src, non_blocking=True)
            self.tail_copy_done.record(self.tail_stream)
            done = torch.cuda.Event()
            done.record(self.tail_stream)
        self.tail.done_ev[index % self.tail.nbuf] = done
        self.tail_issued = key
        return True


def _get_state(device) -> _State:
    global _state
    if _state is None:
        _state = _State(device)
        _verify(_state)
    return _state


def prepare(device) -> None:
    _get_state(device)


def _read_checkpoint_expert(layer: int, expert: int) -> torch.Tensor:
    """One expert's slot bytes straight from the checkpoint (for hot experts, which are not in the arena)."""
    import json
    import struct
    root = os.environ.get("QWEN38_CPU_EXPERTS_MODEL", "/model")
    weight_map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
    out = torch.zeros(SLOT, dtype=torch.uint8)
    prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
    for proj, kind, nbytes in expert_store.PARTS:
        name = prefix + proj + "." + kind
        path = os.path.join(root, weight_map[name])
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
            start, end = header[name]["data_offsets"]
            fh.seek(8 + n + start)
            data = fh.read(end - start)
        off = expert_store.OFFSETS[(proj, kind)]
        out[off:off + nbytes] = torch.frombuffer(bytearray(data), dtype=torch.uint8)
    return out


def _verify(state: _State) -> None:
    """Byte-exact check: converting hot experts' checkpoint bytes must reproduce the loaded Humming weights."""
    for index in (min(_layers), sorted(_layers)[len(_layers) // 2], max(_layers)):
        entry = _layers[index]
        layer = entry.layer
        for local in (0, len(entry.hot_ids) - 1):
            raw = _read_checkpoint_expert(index, entry.hot_ids[local]).to(state.device)
            w13h, w2h, s13h, s2h = state.convert(raw, 1)
            pairs = ((w13h[0], layer.w13_weight_packed[local], "w13"), (w2h[0], layer.w2_weight_packed[local], "w2"),
                     (s13h[0], layer.w13_weight_scale[local], "w13 scale"),
                     (s2h[0], layer.w2_weight_scale[local], "w2 scale"))
            for got, want, name in pairs:
                if got.shape != want.shape or got.dtype != want.dtype or not torch.equal(got, want):
                    raise RuntimeError(f"stream_v2 conversion mismatch: layer {index} expert "
                                       f"{entry.hot_ids[local]} {name}: {tuple(got.shape)} {got.dtype} vs "
                                       f"{tuple(want.shape)} {want.dtype}")
    logger.info("Stream v2: conversion verified byte-exact against loaded hot experts (3 layers x 2 experts)")


PROFILE = os.environ.get("QWEN38_STREAM_PROFILE") == "1"
# The profiler's CUDA events (recorded where the host may block next: before DMA waits, conversions and the NVMe
# tail wait) and its end-of-step synchronize also keep the GPU fed. Without them, a host limited to 16 cores often
# left the GPU idle for seconds per long prompt (131K-token prefill ~1,460 instead of ~2,100 tok/s in 5 of 6 runs;
# with them in 1 of 6). On 32 cores long prompts were fast either way. On by default, without the timing and the
# log line; QWEN38_STREAM_MARKERS=0 turns them off.
MARKERS = PROFILE or os.environ.get("QWEN38_STREAM_MARKERS", "1") == "1"


class _Prof:
    """QWEN38_STREAM_PROFILE=1: per prefill step, GPU time of hot GEMMs, DMA stalls, conversion and cold GEMMs,
    the host time spent in forward(), and the copy stream's busy time. Synchronizes once per step. With log=False
    (the default markers) it records the same events and synchronizes, but computes and logs nothing."""

    def __init__(self, log: bool = True):
        self.log = log
        self.marks = []      # (kind, start_event, end_event) on the compute stream
        self.copies = []     # (start, end) on the copy stream
        self.host = 0.0
        self.host_issue = 0.0
        self.host_tail = 0.0
        self.reader0 = None
        self.t0 = None

    def mark(self, kind):
        ev = torch.cuda.Event(enable_timing=True)
        ev.record()
        self.marks.append((kind, ev))

    def report(self, tokens):
        torch.cuda.synchronize()
        if not self.log:
            self.__init__(log=False)
            return
        tot = {}
        for (kind, a), (_, b) in zip(self.marks, self.marks[1:]):
            tot[kind] = tot.get(kind, 0.0) + a.elapsed_time(b)
        busy = sum(a.elapsed_time(b) for a, b in self.copies)
        span = self.marks[0][1].elapsed_time(self.marks[-1][1]) if len(self.marks) > 1 else 0.0
        logger.info("Stream v2 profile: tokens=%d moe span %.0f ms | %s | copy busy %.0f ms | host in forward %.0f ms (DMA issue %.0f, tail wait %.0f) | "
                    "tail reads %.0f ms", tokens, span, " ".join(f"{k} {v:.0f}" for k, v in sorted(tot.items())), busy,
                    self.host * 1e3, self.host_issue * 1e3, self.host_tail * 1e3,
                    (_state.tail.read_s - (self.reader0 or 0.0)) * 1e3 if _state is not None and _state.tail else 0.0)
        reader0 = _state.tail.read_s if _state is not None and _state.tail else None
        self.__init__()
        self.reader0 = reader0


_prof = _Prof(log=PROFILE) if MARKERS else None


CPU_PREFILL_MAX = int(os.environ.get("QWEN38_CPU_PREFILL_MAX", "384"))
USAGE_DECAY = float(os.environ.get("QWEN38_PREFILL_USAGE_DECAY", "0.5"))
TAIL_KEEP = int(os.environ.get("QWEN38_TAIL_KEEP", "8"))
# Pinned bounce buffers for NVMe-tier experts, one layer each; the reader works up to this many layers ahead.
TAIL_BUFFERS = max(2, int(os.environ.get("QWEN38_TAIL_BUFFERS", "4")))


class _Usage:
    """Per-layer routing counts of prefill steps (both paths), decayed per step, on the GPU; copied to pinned host
    memory at every prefill step end. The CPU server turns them into promotion hints after a prompt, and the tail
    reader keeps the most-used tail experts in the page cache."""

    def __init__(self, device):
        self.order = sorted(_layers)
        self.counts = torch.zeros(48, 512, dtype=torch.float32, device=device)
        self.host = torch.zeros(48, 512, dtype=torch.float32).pin_memory()
        self.ev = None

    def step_start(self):
        self.counts.mul_(USAGE_DECAY)

    def add(self, index: int, topk_ids: torch.Tensor):
        ids = topk_ids.reshape(-1)
        valid = (ids >= 0) & (ids < 512)  # the profile run's dummy routing can hold out-of-range ids
        self.counts[index].index_add_(0, ids.clamp(0, 511).long(), valid.to(torch.float32))

    def step_end(self, svc):
        self.host.copy_(self.counts, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.ev = ev
        if svc is not None and getattr(svc, "dynamic", False):
            svc.ext.usage_ready(ev.cuda_event, self.host.data_ptr(), TAIL_KEEP, 24)

    def keep_sets(self):
        """Per layer: the TAIL_KEEP most-used experts of the previous prefill steps (or None if not ready)."""
        if self.ev is None or not self.ev.query():
            return None
        return self.host.numpy()


_usage = None


def _usage_for(device):
    global _usage
    if _usage is None:
        _usage = _Usage(device)
    return _usage



def forward(index: int, x, topk_ids, topk_weights, activation) -> torch.Tensor:
    """Routed-expert output of a hot-only layer for a prefill batch (hot + all cold experts).

    Up to QWEN38_CPU_PREFILL_MAX tokens the CPU computes the cold experts (streaming every cold expert costs a
    fixed ~2.2 s per step); larger chunks stream them to the GPU.
    """
    import time
    t_host = time.perf_counter()
    entry = _layers[index]
    layer = entry.layer
    usage = _usage_for(x.device)
    first, last = index == usage.order[0], index == usage.order[-1]
    if first:
        usage.step_start()
    usage.add(index, topk_ids)
    if x.shape[0] <= CPU_PREFILL_MAX:
        from . import cpu_experts
        if cpu_experts.enabled():
            svc = cpu_experts.service()
            if x.shape[0] <= svc.big_max:
                if last:
                    usage.step_end(svc)
                svc.submit_big(x, topk_ids, topk_weights, index)
                acc = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
                w13 = getattr(layer, "w13_weight", layer.w13_weight_packed)
                w2 = getattr(layer, "w2_weight", layer.w2_weight_packed)
                _run_experts(entry.base, x, topk_ids, topk_weights, w13, w2, layer.expert_map, acc, activation)
                svc.wait_add_f32(acc, index)
                return acc.to(x.dtype)
    state = _get_state(x.device)
    compute = torch.cuda.current_stream(x.device)
    order = state.order
    at = order.index(index)
    svc = None
    if at == 0:
        # A prefill step starts: stop arena promotions (the CPU server finishes any in progress), then snapshot
        # the mapping this step's DMAs will read.
        from . import cpu_experts
        svc = cpu_experts.service() if cpu_experts.enabled() else None
        if svc is not None and getattr(svc, "dynamic", False):
            svc.ext.arena_begin_prefill()
        state.plan()
    groups = entry.groups
    prof = _prof
    # Start this layer's first DMAs before the hot GEMM so they overlap it.
    state.issue(groups[0].pos)
    state.issue(groups[0].pos + 1)
    if state.tail is not None:
        for ahead in range(TAIL_BUFFERS):
            if at + ahead < len(order):
                state.request_tail(order[at + ahead])
        if entry.tail_list:
            state.issue_tail(index, block=False)
    if prof:
        prof.mark("hot")
    acc = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float32, device=x.device)
    w13 = getattr(layer, "w13_weight", layer.w13_weight_packed)
    w2 = getattr(layer, "w2_weight", layer.w2_weight_packed)
    _run_experts(entry.base, x, topk_ids, topk_weights, w13, w2, layer.expert_map, acc, activation)
    staged = entry.staged
    q = entry.base.quant_config
    staged.quant_config = replace(q, _w1=replace(q._w1, scale=state.convert.s13h),
                                  _w2=replace(q._w2, scale=state.convert.s2h))
    for g in groups:
        b = g.pos % 2
        state.issue(g.pos)
        if prof:
            prof.mark("stall")
        compute.wait_event(state.copy_done[b])
        if prof:
            prof.mark("convert")
        w13h, w2h, _, _ = state.convert(state.raw[b], g.count)
        state.raw_free[b].record(compute)
        state.issue(g.pos + 2)  # into the buffer just converted
        if prof:
            prof.mark("cold")
        _run_experts(staged, x, topk_ids, topk_weights, w13h, w2h, state.emap_dev[g.row], acc, activation)
    if state.tail is not None and entry.tail_list:
        if prof:
            prof.mark("tail_wait")
            t_tail = time.perf_counter()
        state.issue_tail(index, block=True)
        if prof:
            prof.host_tail += time.perf_counter() - t_tail
        compute.wait_event(state.tail_copy_done)
        if prof:
            prof.mark("convert")
        w13h, w2h, _, _ = state.convert(state.raw_tail, len(entry.tail_list))
        state.tail_raw_free.record(compute)
        state.tail_issued = None
        if at + 1 < len(order) and _layers[order[at + 1]].tail_list:
            state.issue_tail(order[at + 1], block=False)
        if prof:
            prof.mark("cold")
        _run_experts(staged, x, topk_ids, topk_weights, w13h, w2h, state.emap_dev[entry.tail_row], acc,
                     activation)
    if prof:
        prof.mark("reduce")
    out = acc.to(x.dtype)
    state.staged_calls += 1
    if state.staged_calls <= 2:
        logger.info("Stream v2 active: layer=%d tokens=%d groups=%d", index, x.shape[0], len(groups))
    if at == len(order) - 1:
        # Step done: hand the server an event after this step's last arena/tail DMA; promotions resume once
        # it has completed.
        from . import cpu_experts
        svc = cpu_experts.service() if cpu_experts.enabled() else None
        if svc is not None and getattr(svc, "dynamic", False):
            if state.tail is not None:
                state.stream.wait_stream(state.tail_stream)
            ev = torch.cuda.Event()
            ev.record(state.stream)
            state.prefill_event = ev
            svc.ext.arena_end_prefill(ev.cuda_event)
        usage.step_end(svc)
    if prof:
        prof.mark("gap")
        prof.host += time.perf_counter() - t_host
        if index == max(_layers):
            prof.report(x.shape[0])
    return out
