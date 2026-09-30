# SPDX-License-Identifier: Apache-2.0
"""Cold MoE experts on the CPU for single-GPU decode (QWEN38_CPU_EXPERTS=1).

The GPU keeps a fixed hot set per layer; every other routed expert is computed by a pinned CPU thread
pool straight from the checkpoint's GPTQ INT4 tensors (memory-mapped: the page cache is the RAM tier,
NVMe backs it). Per layer the GPU writes a request to pinned host memory, computes its hot experts, then
waits for the CPU's partial sum and adds it (CUDA-graph safe; see bridge_cuda.cu).
"""
import json
import mmap
import os
import struct
import threading

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)
L, E = 48, 512
REQ_BYTES, RESP_BYTES = 41472, 41088
_PARTS = (("gate_proj", "qweight"), ("gate_proj", "scales"), ("up_proj", "qweight"),
          ("up_proj", "scales"), ("down_proj", "qweight"), ("down_proj", "scales"))
_HOT: dict[int, list[int]] = {}
_OWNED: dict[int, list[int]] = {}     # expert parallelism: this rank's experts per layer
_LOCK = threading.Lock()
_SVC = None


def enabled() -> bool:
    return os.environ.get("QWEN38_CPU_EXPERTS") == "1"


def register_hot_set(layer: int, global_ids, owned=None) -> None:
    _HOT[int(layer)] = [int(g) for g in global_ids]
    if owned is not None:
        _OWNED[int(layer)] = [int(g) for g in owned]


def _cpu_list(spec: str) -> list[int]:
    from vllm.qwen38_host import parse_cpus
    return parse_cpus(spec)


def build_extension(verbose: bool = False):
    """Compile (or load the cached build of) the CPU kernel and the GPU bridge. The image builds it at build time
    (QWEN38_CPU_EXPERTS_BUILD); x86-64 with AVX2, FMA and F16C; sm_86 code plus PTX for newer GPUs."""
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    build = os.environ.get("QWEN38_CPU_EXPERTS_BUILD", "/root/.cache/qwen38/cpu_experts")
    os.makedirs(build, exist_ok=True)
    return load("qwen38_cpu_experts", [os.path.join(here, "cpu_moe.cpp"), os.path.join(here, "bridge_cuda.cu")],
                extra_cflags=["-O3", "-mavx2", "-mfma", "-mf16c", "-DPF_ROWS=6", "-DWITH_BRIDGE"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_86,code=sm_86",
                                   "-gencode=arch=compute_86,code=compute_86"],
                build_directory=build, verbose=verbose)


_EXT = None
_EXT_LOCK = threading.Lock()


def extension():
    """The loaded extension, shared by the CPU expert service and the prefill tail reader."""
    global _EXT
    with _EXT_LOCK:
        if _EXT is None:
            _EXT = build_extension()
    return _EXT


class _Service:
    def __init__(self):
        from vllm import qwen38_host
        self.ext = extension()
        if sorted(_HOT) != list(range(L)):
            raise RuntimeError(f"CPU experts: hot sets registered for {len(_HOT)} of {L} layers")
        self._maps = {}
        self.dynamic = False
        from vllm.model_executor.layers.fused_moe import hot_only
        if hot_only.size() > 0:
            # Hot-only serving: cold experts live in the host arena (expert_store.py).
            from . import expert_store
            store = expert_store.get()
            self.table = store.table
            if any(store.tail):
                self.ext.set_tail_mask(store.tail_mask)
                # Dynamic arena: decode-used tail experts replace their layer's least recently used arena expert.
                per_step = int(os.environ.get("QWEN38_EXPERT_PROMOTE", "8"))
                if per_step > 0:
                    from .expert_store import SLOT
                    self.ext.dyn_init(store.table, store.tail_mask, store.slot_of, store.owner, store.slot_layer,
                                      store.last_use, store.file_addr, store.base, SLOT, per_step, L - 1,
                                      store.slot_protected)
                self.dynamic = per_step > 0
        else:
            root = os.environ.get("QWEN38_CPU_EXPERTS_MODEL", "/model")
            wm = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
            self.table = torch.zeros(L, E, 6, dtype=torch.int64)
            for l in range(L):
                for e in range(E):
                    prefix = f"model.language_model.layers.{l}.mlp.experts.{e}."
                    for i, (proj, kind) in enumerate(_PARTS):
                        self.table[l, e, i] = self._addr(root, wm, prefix + proj + "." + kind)
        self.skip = torch.zeros(L, E, dtype=torch.uint8)
        for l, ids in _HOT.items():
            self.skip[l, ids] = 1
        for l, owned in _OWNED.items():          # the other rank computes its own experts
            mine = torch.zeros(E, dtype=torch.bool)
            mine[owned] = True
            self.skip[l, ~mine] = 1
        self.req = torch.zeros(L * REQ_BYTES, dtype=torch.uint8).pin_memory()
        self.resp = torch.zeros(L * RESP_BYTES, dtype=torch.uint8).pin_memory()
        self.req_addr, self.resp_addr = self.req.data_ptr(), self.resp.data_ptr()
        # One large slot for prefill chunks of up to big_max tokens (eager only, shared by all layers).
        req_big, resp_big, self.big_max = self.ext.big_layout()
        self.req_big = torch.zeros(req_big, dtype=torch.uint8).pin_memory()
        self.resp_big = torch.zeros(resp_big, dtype=torch.uint8).pin_memory()
        self.counter = torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device())
        # Pool and serving-process CPUs: QWEN38_CPU_EXPERTS_CPUS / QWEN38_MAIN_CPUS, or derived from the host
        # topology (qwen38_host.py: the pool spans every L3 domain, since bandwidth scales with CCDs, not cores;
        # with several GPU ranks each rank gets its own pool).
        from vllm.model_executor.layers.fused_moe.hot_only import rank_and_size
        rank, ranks = rank_and_size()
        cpus = qwen38_host.pool_cpus(rank=rank, ranks=ranks)
        slices = os.environ.get("QWEN38_CPU_EXPERTS_SLICES")
        if slices:
            self.ext.set_slices([int(v) for v in slices.split(",")])
        main = qwen38_host.main_cpus(cpus) if ranks == 1 else qwen38_host.main_cpus(ranks=ranks)
        os.sched_setaffinity(0, set(main))
        self.ext.set_stats_every(int(os.environ.get("QWEN38_CPU_EXPERTS_STATS", "0")))
        self.ext.start_server(self.table, self.skip, self.req_addr, self.resp_addr, L, len(cpus), cpus,
                              self.req_big.data_ptr(), self.resp_big.data_ptr())
        logger.info("CPU experts (rank %d/%d): %d layers, %d hot experts/layer on GPU, %d pool threads on CPUs %s, "
                    "serving threads on CPUs %s, %d files mapped", rank, ranks, L, len(_HOT[0]), len(cpus),
                    qwen38_host.format_cpus(cpus), qwen38_host.format_cpus(main), len(self._maps))

    def _addr(self, root, wm, name):
        rel = wm[name]
        if rel not in self._maps:
            path = os.path.join(root, rel)
            fd = os.open(path, os.O_RDONLY)
            mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
            os.close(fd)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            self._maps[rel] = (mm, np.frombuffer(mm, dtype=np.uint8).ctypes.data, 8 + n, hdr)
        mm, base, data0, hdr = self._maps[rel]
        return base + data0 + hdr[name]["data_offsets"][0]

    def submit(self, x, topk_ids, topk_weights, layer: int, mask=None):
        """mask: optional int8 [T*topk] on the GPU; masked (token, k) pairs are left to the GPU share."""
        w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
        self.ext.bridge_submit(x.contiguous(), topk_ids.contiguous(), w.contiguous(),
                               self.req_addr + REQ_BYTES * layer, self.counter, layer, layer == 0,
                               0 if mask is None else mask.data_ptr())

    def wait_add(self, out, layer: int, extra=None):
        """extra: optional FP32 [T, H] on the GPU, added together with the CPU result."""
        self.ext.bridge_wait_add(out, self.resp_addr + RESP_BYTES * layer, self.counter, layer,
                                 0 if extra is None else extra.data_ptr())

    def submit_big(self, x, topk_ids, topk_weights, layer: int):
        """Prefill chunk of up to big_max tokens; pair with wait_add_f32 (FP32 accumulator)."""
        w = topk_weights if topk_weights.dtype == torch.float32 else topk_weights.float()
        self.ext.bridge_submit_big(x.contiguous(), topk_ids.contiguous(), w.contiguous(), self.req_big.data_ptr(),
                                   self.counter, layer, layer == 0)

    def wait_add_f32(self, acc, layer: int):
        self.ext.bridge_wait_add_f32(acc, self.resp_big.data_ptr(), self.counter, layer)


def service() -> _Service:
    global _SVC
    if _SVC is None:
        with _LOCK:
            if _SVC is None:
                _SVC = _Service()
    return _SVC
