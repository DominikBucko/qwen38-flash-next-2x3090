# SPDX-License-Identifier: Apache-2.0
"""Host copy of the cold routed experts for hot-only single-GPU serving (QWEN38_HOT_ONLY).

One private anonymous arena (huge pages requested) holds every cold expert's checkpoint tensors in a fixed slot
layout: gate, up and down GPTQ qweight (int32, K-first), then gate, up and down F16 group scales. The qzeros are
not copied (all 0x77777777: symmetric, zero point 8). Layer l's cold experts occupy consecutive slots in expert-id
order, so a group of them is one contiguous range. The CPU decode kernel reads the slots in place; the arena is
registered with CUDA so prefill can DMA a group of slots to the GPU and convert it there.

The arena is anonymous memory: it is charged to the serving cgroup like any other process memory (a
cudaHostAlloc'd pool would not be), and after the fill the checkpoint's page cache for these ranges is dropped.
"""
import json
import mmap
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

L, E = 48, 512
# (checkpoint projection, tensor, bytes) in slot order
PARTS = (("gate_proj", "qweight", 819200), ("up_proj", "qweight", 819200), ("down_proj", "qweight", 819200),
         ("gate_proj", "scales", 25600), ("up_proj", "scales", 25600), ("down_proj", "scales", 25600))
OFFSETS = {}
_off = 0
for _proj, _kind, _n in PARTS:
    OFFSETS[(_proj, _kind)] = _off
    _off += _n
USED = _off                                   # 2,534,400
SLOT = (USED + 4095) // 4096 * 4096           # 2,535,424: page-aligned slots
# The CPU kernel's per-expert pointer order (cpu_experts._PARTS)
CPU_ORDER = (("gate_proj", "qweight"), ("gate_proj", "scales"), ("up_proj", "qweight"),
             ("up_proj", "scales"), ("down_proj", "qweight"), ("down_proj", "scales"))

_HOT: dict[int, list[int]] = {}
_OWNED: dict[int, set[int]] = {}      # expert parallelism: this rank's experts per layer (default: all)
_LOCK = threading.Lock()
_STORE = None


def register_hot_set(layer: int, global_ids, owned=None) -> None:
    _HOT[int(layer)] = [int(g) for g in global_ids]
    _OWNED[int(layer)] = set(range(E)) if owned is None else {int(g) for g in owned}


def hot_sets() -> dict[int, list[int]]:
    return _HOT


class ExpertStore:
    def __init__(self):
        if sorted(_HOT) != list(range(L)):
            raise RuntimeError(f"expert store: hot sets registered for {len(_HOT)} of {L} layers")
        root = os.environ.get("QWEN38_CPU_EXPERTS_MODEL", "/model")
        weight_map = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        headers = {}

        def locate(name):
            rel = weight_map[name]
            if rel not in headers:
                with open(os.path.join(root, rel), "rb") as fh:
                    n = struct.unpack("<Q", fh.read(8))[0]
                    headers[rel] = (8 + n, json.loads(fh.read(n)))
            data0, header = headers[rel]
            meta = header[name]
            start, end = meta["data_offsets"]
            return rel, data0 + start, end - start, meta

        # Arena size (QWEN38_EXPERT_ARENA_SLOTS, default "auto": what the memory budget holds, at most every cold
        # expert; see vllm/qwen38_host.py). With fewer slots than cold experts each layer keeps its
        # best-ranked cold experts in the arena; the rest (the "tail") stay on NVMe behind the page cache: the CPU
        # reads them through a file mapping, prefill streams them through pinned bounce buffers.
        ranked = _ranked_ids(root)
        # This rank's cold experts: owned (all of them without expert parallelism) and not hot on its GPU.
        cold_all = [[e for e in ranked[l] if e in _OWNED[l] and e not in set(_HOT[l])] for l in range(L)]
        total = sum(map(len, cold_all))
        from vllm import qwen38_host
        from vllm.model_executor.layers.fused_moe.hot_only import rank_and_size
        self.rank, self.ranks = rank_and_size()
        # QWEN38_EXPERT_ARENA_SLOTS (per rank), default: this rank's share of the RAM budget
        want = qwen38_host.arena_slots(total, L, ranks=self.ranks)
        per_layer = [want // L + (1 if l < want % L else 0) for l in range(L)]
        self.cold = [sorted(cold_all[l][:per_layer[l]]) for l in range(L)]
        self.tail = [sorted(cold_all[l][per_layer[l]:]) for l in range(L)]
        self.first_slot = []
        self.slot_of = torch.full((L, E), -1, dtype=torch.int32)
        slots = 0
        for l in range(L):
            self.first_slot.append(slots)
            for i, e in enumerate(self.cold[l]):
                self.slot_of[l, e] = slots + i
            slots += len(self.cold[l])
        self.num_slots = slots
        self.nbytes = slots * SLOT
        # MAP_PRIVATE anonymous: eligible for transparent huge pages (a shared anonymous map is shmem).
        self.arena = mmap.mmap(-1, self.nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                               prot=mmap.PROT_READ | mmap.PROT_WRITE)
        for advice in ("MADV_HUGEPAGE", "MADV_DONTFORK"):
            if hasattr(mmap, advice):
                self.arena.madvise(getattr(mmap, advice))
        self.base = torch.frombuffer(self.arena, dtype=torch.uint8).data_ptr()
        # Fault the whole arena in first (huge-page faults compact memory as needed), before the page cache
        # of the fill fragments it further.
        t_pop = time.perf_counter()
        _madvise_chunks(self.base, self.nbytes, _MADV_POPULATE_WRITE)
        huge_pop = _anon_huge_kib()
        t_pop = time.perf_counter() - t_pop

        jobs: dict[str, list[tuple[int, int, int]]] = {}
        for l in range(L):
            for e in self.cold[l]:
                slot = int(self.slot_of[l, e])
                prefix = f"model.language_model.layers.{l}.mlp.experts.{e}."
                for proj, kind, nbytes in PARTS:
                    rel, offset, size, meta = locate(prefix + proj + "." + kind)
                    if size != nbytes:
                        raise RuntimeError(f"expert store: {prefix}{proj}.{kind} is {size} bytes, expected {nbytes}")
                    jobs.setdefault(rel, []).append((slot * SLOT + OFFSETS[(proj, kind)], offset, size))
        t0 = time.perf_counter()
        view = memoryview(self.arena)
        drop = os.environ.get("QWEN38_EXPERT_STORE_DROP_CACHE", "1") == "1"

        def fill(rel):
            fd = os.open(os.path.join(root, rel), os.O_RDONLY)
            try:
                done = 0
                for dst, offset, size in sorted(jobs[rel], key=lambda job: job[1]):
                    got = os.preadv(fd, [view[dst:dst + size]], offset)
                    if got != size:
                        raise RuntimeError(f"expert store: short read {got}/{size} from {rel}@{offset}")
                    if drop:
                        os.posix_fadvise(fd, offset, size, os.POSIX_FADV_DONTNEED)
                    done += size
                return done
            finally:
                os.close(fd)

        threads = int(os.environ.get("QWEN38_EXPERT_STORE_THREADS", "16"))
        with ThreadPoolExecutor(max_workers=threads) as pool:
            filled = sum(pool.map(fill, sorted(jobs)))
        t1 = time.perf_counter()
        # Whatever still sits on 4K pages: collapse into huge pages now (MADV_COLLAPSE copies into freshly
        # compacted 2 MB pages), before cudaHostRegister pins the physical pages.
        if os.environ.get("QWEN38_EXPERT_STORE_COLLAPSE", "1") == "1":
            _madvise_chunks(self.base, self.nbytes, _MADV_COLLAPSE)
        t_col = time.perf_counter() - t1
        t1 = time.perf_counter()
        err = torch.cuda.cudart().cudaHostRegister(self.base, self.nbytes, 0)
        if int(err) != 0:
            raise RuntimeError(f"expert store: cudaHostRegister failed ({err})")
        t2 = time.perf_counter()
        self.bytes = torch.frombuffer(self.arena, dtype=torch.uint8)
        # Pointer table for the CPU kernel; hot experts stay 0 (they are skipped).
        self.table = torch.zeros(L, E, 6, dtype=torch.int64)
        for l in range(L):
            for e in self.cold[l]:
                slot = self.base + int(self.slot_of[l, e]) * SLOT
                for i, part in enumerate(CPU_ORDER):
                    self.table[l, e, i] = slot + OFFSETS[part]
        # File mappings and ranges of every cold expert: tail experts are read through them (the page cache is
        # their RAM tier), and with the dynamic arena any cold expert can become a tail expert.
        import numpy as np
        self._maps = {}
        self._fds = {}
        self.cold_all = cold_all
        self.tail_mask = torch.zeros(L, E, dtype=torch.uint8)
        self.file_addr = torch.zeros(L, E, 6, dtype=torch.int64)
        self.loc = [dict() for _ in range(L)]   # per layer: expert -> [(fd, offset, size, slot_offset) x 6]
        for l in range(L):
            for e in cold_all[l]:
                prefix = f"model.language_model.layers.{l}.mlp.experts.{e}."
                parts = []
                for proj, kind, nbytes in PARTS:
                    rel, offset, size, meta = locate(prefix + proj + "." + kind)
                    if rel not in self._maps:
                        fd = os.open(os.path.join(root, rel), os.O_RDONLY)
                        self._fds[rel] = fd
                        mm = mmap.mmap(fd, 0, access=mmap.ACCESS_READ)
                        # address of a read-only mapping (ctypes cannot take from_buffer of it; numpy can)
                        self._maps[rel] = (mm, np.frombuffer(mm, dtype=np.uint8).ctypes.data)
                    parts.append((self._fds[rel], offset, size, OFFSETS[(proj, kind)]))
                    addr = self._maps[rel][1] + offset
                    for i, part in enumerate(CPU_ORDER):
                        if part == (proj, kind):
                            self.file_addr[l, e, i] = addr
                self.loc[l][e] = parts
            for e in self.tail[l]:
                self.table[l, e] = self.file_addr[l, e]
                self.tail_mask[l, e] = 1
        self.tail_loc = [[(e, self.loc[l][e]) for e in self.tail[l]] for l in range(L)]
        # Dynamic arena bookkeeping (cpu_moe.cpp dyn_*): slots stay with their layer; LRU stamps start below every
        # real decode step, lower-ranked experts older, so the first evictions take the least likely experts.
        self.owner = torch.zeros(slots, dtype=torch.int32)
        self.slot_layer = torch.zeros(slots, dtype=torch.int32)
        self.last_use = torch.zeros(slots, dtype=torch.int64)
        # The best-ranked QWEN38_GPU_SHARE_PROTECT arena experts per layer are never evicted, so the GPU can read
        # them over PCIe (gpu_share.py) from addresses that stay valid; gpu_addr is their UVA slot address.
        protect = int(os.environ.get("QWEN38_GPU_SHARE_PROTECT", "200"))
        self.slot_protected = torch.zeros(slots, dtype=torch.uint8)
        self.gpu_addr = torch.zeros(L, E, dtype=torch.int64)
        for l in range(L):
            rank = {e: r for r, e in enumerate(cold_all[l])}
            for e in self.cold[l]:
                sl = int(self.slot_of[l, e])
                self.owner[sl] = e
                self.slot_layer[sl] = l
                self.last_use[sl] = -1 - rank[e]
                if rank[e] < protect:
                    self.slot_protected[sl] = 1
                    self.gpu_addr[l, e] = self.base + sl * SLOT
        tail_n = sum(map(len, self.tail))
        huge = _anon_huge_kib()
        logger.info("Expert store (rank %d/%d): %d cold experts in the arena (%d-%d per layer), %.2f GiB; huge "
                    "pages %.1f GiB after prefault (%.1f s), %.1f GiB after fill + collapse (%.1f s); filled %.2f GiB "
                    "in %.1f s, pinned in %.1f s; %d tail experts (%.2f GiB) on NVMe",
                    self.rank, self.ranks, slots, min(map(len, self.cold)), max(map(len, self.cold)),
                    self.nbytes / 2**30,
                    huge_pop / 2**20, t_pop, huge / 2**20, t_col, filled / 2**30, t1 - t0 - t_col, t2 - t1,
                    tail_n, tail_n * USED / 2**30)

    def layer_range(self, layer: int, first: int, count: int) -> torch.Tensor:
        """uint8 view of `count` consecutive cold slots of `layer` starting at cold index `first`."""
        start = (self.first_slot[layer] + first) * SLOT
        return self.bytes[start:start + count * SLOT]


def _ranked_ids(root: str) -> list[list[int]]:
    """Per layer: the static rankings (hottest first), then the never-ranked experts in id order."""
    path = os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_FILE") or os.path.join(os.getcwd(), "static_hot_cache_rankings.json")
    rankings = json.load(open(path))
    out = []
    for l in range(L):
        seen, ids = set(), []
        for g in rankings.get(str(l), []):
            g = int(g)
            if 0 <= g < E and g not in seen:
                ids.append(g)
                seen.add(g)
        ids += [e for e in range(E) if e not in seen]
        out.append(ids)
    return out


_MADV_POPULATE_WRITE, _MADV_COLLAPSE = 23, 25
_libc = None


def _madvise_chunks(addr: int, nbytes: int, advice: int, chunk: int = 1 << 30) -> int:
    """madvise(advice) over [addr, addr + nbytes) in 1 GiB steps; returns the number of failed chunks."""
    import ctypes
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _libc.madvise.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int)
    failed = 0
    for off in range(0, nbytes, chunk):
        if _libc.madvise(addr + off, min(chunk, nbytes - off), advice) != 0:
            failed += 1
    return failed


def _anon_huge_kib() -> int:
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                if line.startswith("AnonHugePages:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def get() -> ExpertStore:
    global _STORE
    if _STORE is None:
        with _LOCK:
            if _STORE is None:
                _STORE = ExpertStore()
    return _STORE
