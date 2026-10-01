# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dedicated CPU-offload process for PLE embedding layers.

This module implements a standalone process that:
1. Loads only the :class:`PleOffloadLayer` weights into CPU memory.
2. Accepts per-step computation requests from GPU worker processes.
3. Runs ``forward_impl()`` on CPU, copies results to every TP worker's GPU
   output buffer for the requesting DP rank, and signals the corresponding
   IPC semaphore.

The TP workers within one DP rank receive identical inputs, so the CPU result
is computed once per DP rank and fanned out to all of its TP ranks.

Class structure mirrors the GPU worker pattern in multiproc_executor.py:

  PleOffloadWorkerHandle -- handle held by the spawning GPU worker
  PleOffloadWorker       -- process lifecycle and READY handshake
  PleOffloadRunner       -- owns weights and serves inference requests
"""

import contextlib
import os
import json
import mmap as _mmap
import multiprocessing.process
import os
import pickle
import signal
import tempfile
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, cast

import msgspec
import torch
import torch.distributed as dist
import zmq

import vllm.envs as envs
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    PleOffloadLayer,
    mark_as_offload_worker,
)
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import initialize_dummy_weights
from vllm.utils.system_utils import decorate_logs, get_mp_context
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.ple_offload.protocol import (
    _PLE_OFFLOAD_REQUEST_DECODER,
    PleOffloadRegistration,
    PleOffloadRequest,
)

logger = init_logger(__name__)


@dataclass
class PleOffloadOutputTarget:
    """Shared CPU output and handshake flags for one TP worker."""

    tp_rank: int
    cpu_output_buffer: torch.Tensor
    ready_flag: torch.Tensor
    consumed_flag: torch.Tensor


@dataclass
class PleOffloadInputBuffers:
    """Shared-memory input buffers registered for one DP rank."""

    input_ids_buf: torch.Tensor  # int32 (max_num_tokens,)
    query_start_loc_buf: torch.Tensor  # int32 (max_num_reqs + 1,)
    ngram_context_buf: torch.Tensor | None  # int32 (max_num_reqs, ngram_context_len)


@dataclass
class PleOffloadWorkerHandle:
    """Resources owned by the GPU worker that spawned the offload process."""

    proc: Any
    death_writer: Connection | None
    ready_pipe_reader: Connection | None

    def close(self) -> None:
        """Release all process resources. Safe to call more than once."""
        if self.ready_pipe_reader is not None:
            self.ready_pipe_reader.close()
            self.ready_pipe_reader = None
        if self.death_writer is not None:
            self.death_writer.close()
            self.death_writer = None
        # First allow the child to exit after observing the closed death pipe.
        if self.proc.is_alive():
            self.proc.join(timeout=5)
        # Fall back to SIGTERM if graceful shutdown times out.
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=5)
        # Use SIGKILL as the final fallback for a stuck child.
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=5)


def _init_offload_distributed() -> None:
    """Initialize the single-rank Gloo world required by TP-aware layers."""
    if dist.is_initialized():
        return

    # VocabParallelEmbedding reads the TP process group during construction.
    # The offload process owns the full embedding table, so it uses an isolated
    # TP1/PP1 Gloo world and never joins the GPU workers' NCCL groups.
    store_dir = tempfile.mkdtemp(prefix="vllm_ple_offload_")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{store_dir}/store",
        local_rank=0,
        backend="gloo",
    )
    # initialize_model_parallel reads the active VllmConfig in the current
    # vLLM version. Explicitly configure DP1/TP1/PP1 to match the isolated
    # world, regardless of any DP environment variables inherited from the GPU
    # worker. The real DP/TP configuration is used later for model construction,
    # registration, and request routing.
    offload_config = VllmConfig()
    offload_parallel_config = offload_config.parallel_config
    offload_parallel_config.data_parallel_size = 1
    offload_parallel_config.data_parallel_size_local = 1
    offload_parallel_config.data_parallel_rank = 0
    offload_parallel_config.data_parallel_rank_local = 0
    offload_parallel_config.data_parallel_index = 0
    offload_parallel_config.tensor_parallel_size = 1
    offload_parallel_config.pipeline_parallel_size = 1
    offload_parallel_config.prefill_context_parallel_size = 1
    offload_parallel_config.decode_context_parallel_size = 1
    offload_parallel_config.world_size = 1
    offload_parallel_config.nnodes = 1
    offload_parallel_config.node_rank = 0
    with set_current_vllm_config(offload_config):
        ensure_model_parallel_initialized(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            backend="gloo",
        )
    logger.info(
        "Distributed environment initialized (backend=gloo, rank=0, world_size=1)."
    )


class PleOffloadWorker:
    """Manage process creation, READY handshake, and the child entry point."""

    READY_STR = "READY"

    @staticmethod
    def make_process(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
    ) -> PleOffloadWorkerHandle:
        """Spawn one CPU offload process for all local DP and TP workers."""
        context = get_mp_context()
        ready_reader, ready_writer = context.Pipe(duplex=False)
        death_reader, death_writer = context.Pipe(duplex=False)
        proc = context.Process(
            target=PleOffloadWorker.proc_main,
            kwargs={
                "vllm_config": vllm_config,
                "num_workers": num_workers,
                "ipc_addr": ipc_addr,
                "ready_pipe": (ready_reader, ready_writer),
                "death_pipe": death_reader,
            },
            name="PleOffloadWorker",
            daemon=True,
        )

        # Python normally forbids a daemon WorkerProc from spawning children.
        # vLLM owns this process through death_pipe and explicit shutdown, so
        # temporarily clear the daemon flag while the child is created.
        parent = multiprocessing.process._current_process  # type: ignore[attr-defined]
        saved_daemon = parent._config.get("daemon")
        parent._config["daemon"] = False
        try:
            proc.start()
        finally:
            parent._config["daemon"] = saved_daemon
        ready_writer.close()
        return PleOffloadWorkerHandle(
            proc=proc,
            death_writer=death_writer,
            ready_pipe_reader=ready_reader,
        )

    @staticmethod
    def wait_for_ready(handle: PleOffloadWorkerHandle) -> None:
        """Wait until weights and all GPU registrations are ready to serve."""
        reader = handle.ready_pipe_reader
        if reader is None:
            return
        if not reader.poll(envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT):
            raise TimeoutError(
                "PLE offload worker did not become ready within "
                f"{envs.VLLM_PLE_OFFLOAD_READY_TIMEOUT}s."
            )
        try:
            message = reader.recv()
        except EOFError as error:
            raise RuntimeError("PLE offload worker exited during startup") from error
        finally:
            reader.close()
            handle.ready_pipe_reader = None
        if message.get("status") != PleOffloadWorker.READY_STR:
            raise RuntimeError(
                "PLE offload worker failed during startup: "
                f"{message.get('error', 'unknown error')}"
            )
        layer_names = message["layer_names"]
        logger.info(
            "Worker ready - %d PleOffloadLayer(s): %s",
            len(layer_names),
            layer_names,
        )

    @staticmethod
    def proc_main(
        vllm_config: VllmConfig,
        num_workers: int,
        ipc_addr: str,
        ready_pipe: tuple[Connection, Connection],
        death_pipe: Connection,
    ) -> None:
        """Load PLE weights, accept registrations, and run the request loop."""
        decorate_logs("PleOffloadWorker")
        if os.environ.get("QWEN38_CPU_EXPERTS") == "1":
            # Keep PLE lookups off the SMT siblings of the CPU expert pool's cores (vllm/qwen38_host.py).
            from vllm import qwen38_host
            os.sched_setaffinity(0, set(qwen38_host.main_cpus()))
            # The row gathers are page-fault bound (MADV_WILLNEED already overlaps the NVMe reads), so a few
            # intra-op threads suffice; one spinning OpenMP thread per CPU took the cores that the GPU worker
            # needs to issue prefill kernels on machines with 16 cores or fewer.
            torch.set_num_threads(max(1, int(os.environ.get("QWEN38_PLE_THREADS", "4"))))
        ready_reader, ready_writer = ready_pipe
        ready_reader.close()
        shutdown_event = threading.Event()

        def monitor_parent() -> None:
            try:
                death_pipe.recv()
            except EOFError:
                logger.info("Parent exited, shutting down.")
                shutdown_event.set()

        def handle_signal(_signum: int, _frame: object) -> None:
            shutdown_event.set()

        signal.signal(signal.SIGTERM, handle_signal)
        signal.signal(signal.SIGINT, handle_signal)
        threading.Thread(
            target=monitor_parent,
            daemon=True,
            name="PleOffloadDeathMonitor",
        ).start()

        zmq_context: zmq.Context | None = None
        pull_socket: zmq.Socket | None = None
        try:
            # The flag lets PleOffloadLayer subclasses execute their complete
            # constructors instead of becoming empty GPU-worker placeholders.
            mark_as_offload_worker()

            # Initialize Gloo before installing the real VllmConfig. This keeps
            # the CPU process in an isolated rank-zero, world-size-one group.
            _init_offload_distributed()

            # Model components read the active VllmConfig while the meta model
            # is constructed, so keep the context around runner initialization.
            with set_current_vllm_config(vllm_config):
                runner = PleOffloadRunner(vllm_config)

            zmq_context = zmq.Context()
            pull_socket = zmq_context.socket(zmq.PULL)
            pull_socket.bind(ipc_addr)
            logger.info(
                "Bound IPC address %s; waiting for %d GPU worker registration(s).",
                ipc_addr,
                num_workers,
            )

            # READY means that the process can immediately serve requests. Wait
            # for every DP/TP worker to register before notifying the parent.
            runner.accept_registrations(pull_socket, num_workers)
            ready_writer.send(
                {
                    "status": PleOffloadWorker.READY_STR,
                    "layer_names": sorted(runner.layer_names),
                }
            )
            ready_writer.close()
            ready_writer = None  # type: ignore[assignment]

            if os.environ.get("QWEN38_PLE_PREFAULT", "0") == "1":
                # GPU workers register after their checkpoint load, which is
                # what evicts this table to swap. Re-read it in the background
                # while the engine finishes initialization.
                threading.Thread(
                    target=_prefault_tables,
                    args=(runner, shutdown_event),
                    daemon=True,
                    name="PleOffloadPrefault",
                ).start()

            runner.busy_loop(pull_socket, shutdown_event)
        except Exception as error:
            logger.exception("Unexpected failure in PLE offload worker.")
            if ready_writer is not None:
                with contextlib.suppress(Exception):
                    ready_writer.send({"status": "FAILURE", "error": repr(error)})
            raise
        finally:
            if pull_socket is not None:
                pull_socket.close(linger=0)
            if zmq_context is not None:
                zmq_context.term()
            if ready_writer is not None:
                ready_writer.close()
            death_pipe.close()


def _ple_disk_shard_of(mapped_name: str) -> str | None:
    """"<layer>.a.b.shard_3.weight" -> "<layer>.a.b" (the parameter the shard fills)."""
    import re

    m = re.match(r"^(.*)\.shard_\d+\.weight$", mapped_name)
    return m.group(1) if m else None


class _PleQuantTable:
    """Shard-mmapped quantized n-gram table; gathers dequantize to BF16."""

    ROWS_PER_SHARD = 2_500_012

    def __init__(self, quant_dir: str, total_rows: int, width: int) -> None:
        import json
        import os

        from safetensors import safe_open

        meta = json.load(open(os.path.join(quant_dir, "META.json")))
        self.layout = meta["layout"]
        assert meta["rows"] == total_rows and meta["width"] == width, (
            f"sidecar built for {meta['rows']}x{meta['width']}, "
            f"table is {total_rows}x{width}"
        )
        n_shards = meta["shards"]
        assert n_shards * self.ROWS_PER_SHARD == total_rows, "non-uniform shards"
        # Order matters: the e2m1 layout string contains "e4m3" (its scale dtype).
        if "e2m1" in self.layout:
            key = "weight_e2m1"
        elif "e4m3" in self.layout:
            key = "weight_fp8"
        else:
            key = "weight_i4"
        self._q, self._s, self._s2 = [], [], []
        for n in range(n_shards):
            f = safe_open(os.path.join(quant_dir, f"shard_{n}.safetensors"),
                          framework="pt")
            self._q.append(f.get_tensor(key))
            self._s.append(f.get_tensor("weight_scale"))
            self._s2.append(
                f.get_tensor("weight_scale_2").item()
                if "weight_scale_2" in f.keys() else 1.0
            )
        self.width = width
        self._lut = None
        # nvfp4: e4m3 scales are 1 byte, and a packed nvfp4 table may carry a
        # PER-SHARD weight_scale_2 (vLLM PR #56273 assumes one global), so the
        # scalar rides the wire as 4 raw fp32 bytes per row. int4 layout
        # unchanged.
        _PACKED_NVFP4 = "group16_e2m1_e4m3scale_lownibblefirst"
        _PACKED_INT4 = "group16_int4_fp16scale_lownibblefirst"
        self._nvfp4 = self.layout == _PACKED_NVFP4
        self.packed = (
            os.environ.get("VLLM_PLE_PACKED", "0") == "1"
            and self.layout in (_PACKED_INT4, _PACKED_NVFP4)
        )
        if self.packed:
            logger.info("PLE quant table: PACKED transport active (raw nibble+%s "
                        "rows; GPU-side dequant%s).",
                        "e4m3-scale+fp32-global" if self._nvfp4 else "fp16-scale",
                        ", nvfp4 e2m1" if self._nvfp4 else "")
        elif os.environ.get("VLLM_PLE_PACKED", "0") == "1":
            raise RuntimeError(f"VLLM_PLE_PACKED=1 but layout {self.layout!r}"
                               " is not a packed-capable int4/nvfp4 layout")
        logger.info("PLE quant table: %s, %d shards mmapped from %s",
                    self.layout, n_shards, quant_dir)

    def gather_into(self, ids: torch.Tensor, out: torch.Tensor) -> None:
        ids = ids.long()
        shard = ids // self.ROWS_PER_SHARD
        local = ids - shard * self.ROWS_PER_SHARD
        order = torch.argsort(shard)
        s_sorted, l_sorted = shard[order], local[order]
        uniq, counts = torch.unique_consecutive(s_sorted, return_counts=True)
        pos = 0
        if self.packed:
            pb = self.width // 2                      # packed nibble bytes
            nsc = self._s[0].shape[1]                 # groups = width // 16
            # int4: 16-bit scales, no per-row global (folded into scale).
            # nvfp4: 8-bit e4m3 scales + a per-row 32-bit fp32 global.
            sc_b, gl_b = (1, 4) if self._nvfp4 else (2, 0)
            need = pb + sc_b * nsc + gl_b
            assert out.dtype == torch.uint8 and out.shape[1] >= need, (
                f"packed PLE transport expects uint8 rows >= {need}B, "
                f"got {out.dtype} x {out.shape[1]}")
            for s, c in zip(uniq.tolist(), counts.tolist()):
                sel = l_sorted[pos:pos + c]
                tmp = torch.empty(c, out.shape[1], dtype=torch.uint8)
                tmp[:, :pb] = self._q[s].index_select(0, sel)
                tmp[:, pb:pb + sc_b * nsc] = (
                    self._s[s].index_select(0, sel).view(torch.uint8)
                    .reshape(c, sc_b * nsc))
                if gl_b:
                    # weight_scale_2 is per-shard; broadcast to each row.
                    # NB: 1-D tensor — a 0-dim tensor cannot .view() dtype.
                    gb = torch.tensor(
                        [float(self._s2[s])], dtype=torch.float32
                    ).view(torch.uint8)
                    tmp[:, pb + sc_b * nsc:pb + sc_b * nsc + gl_b] = gb
                out[order[pos:pos + c]] = tmp
                pos += c
            return
        for s, c in zip(uniq.tolist(), counts.tolist()):
            sel = l_sorted[pos:pos + c]
            rows = self._dequant(s, sel)
            out[order[pos:pos + c]] = rows.to(out.dtype)
            pos += c

    def _dequant(self, s: int, sel: torch.Tensor) -> torch.Tensor:
        if "e2m1" in self.layout:
            if self._lut is None:
                mags = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
                self._lut = torch.tensor(mags + [-m for m in mags],
                                         dtype=torch.float32)
            packed = self._q[s].index_select(0, sel)
            lo = (packed & 0xF).long()
            hi = (packed >> 4).long()
            nib = torch.stack((lo, hi), dim=-1).view(packed.shape[0], self.width)
            scale = self._s[s].index_select(0, sel).to(torch.float32)
            g = self.width // scale.shape[1]
            return (self._lut[nib]
                    * scale.repeat_interleave(g, dim=1)
                    * self._s2[s])
        if "e4m3" in self.layout:
            q = self._q[s].index_select(0, sel).to(torch.float32)
            return q * self._s[s].index_select(0, sel)[:, None]
        packed = self._q[s].index_select(0, sel)
        lo = (packed & 0xF).to(torch.int16)
        hi = (packed >> 4).to(torch.int16)
        nib = torch.stack((lo, hi), dim=-1).view(packed.shape[0], self.width)
        scale = self._s[s].index_select(0, sel).to(torch.float32)
        g = self.width // scale.shape[1]
        return (nib.to(torch.float32) - 8) * scale.repeat_interleave(g, dim=1)


def _ple_quant_dir() -> str | None:
    import os

    return os.environ.get("VLLM_PLE_QUANT_DIR") or None


def _ple_quant_attach(layer_name: str, layer: torch.nn.Module,
                      quant_dir: str) -> str | None:
    """Swap the layer's table for a sidecar-backed quant store.

    Returns the stubbed parameter's name, or None when the layer has no
    parameter large enough to be a table (>= 1 GiB).
    """
    named = sorted(layer.named_parameters(), key=lambda kv: kv[1].numel(), reverse=True)
    if not named or named[0][1].numel() * named[0][1].element_size() < (1 << 30):
        return None
    pname, param = named[0]
    rows, width = param.shape
    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    owner._ple_quant = _PleQuantTable(quant_dir, rows, width)
    # Stub before anything writes the parameter. Replace the Parameter object rather
    # than assigning .data: with a quantized sidecar the table is built on the meta
    # device (no 95 GB virtual reservation for the kernel's overcommit heuristic to
    # refuse), and set_data() rejects a meta -> cpu swap. vLLM's weight attributes
    # (weight_loader, output_dim, ...) are carried over to the stub.
    old = getattr(owner, parts[-1])
    stub = torch.nn.Parameter(torch.empty(0, width, dtype=param.dtype), requires_grad=False)
    for k, v in vars(old).items():
        setattr(stub, k, v)
    setattr(owner, parts[-1], stub)
    logger.info("PLE quant: %s.%s stubbed, gathers served from sidecar.",
                layer_name, pname)
    return pname


def _ple_disk_dir() -> str | None:
    import os

    return os.environ.get("VLLM_PLE_DISK_OFFLOAD_DIR") or None


_PLE_DISK_MAPS: dict[str, object] = {}


def _disk_backed_tensor(path: str, shape: tuple[int, ...], dtype: torch.dtype,
                        writable: bool) -> torch.Tensor:
    """Map ``path`` as a tensor of ``shape``/``dtype``.

    numpy has no bfloat16, so the file is mapped with a same-width integer dtype
    and reinterpreted. ``writable`` selects a shared read-write mapping (first
    boot, shard writes must reach the file) versus copy-on-write (steady state).
    MADV_RANDOM is applied either way: gathers are random-access and readahead
    only evicts useful pages.
    """
    import numpy as np

    _NP = {torch.bfloat16: (np.uint16, torch.uint16), torch.float16: (np.uint16, torch.uint16),
           torch.float32: (np.uint32, torch.uint32), torch.float8_e4m3fn: (np.uint8, torch.uint8)}
    np_dtype, torch_int = _NP[dtype]
    arr = np.memmap(path, dtype=np_dtype, mode="r+" if writable else "c", shape=shape)
    with contextlib.suppress(Exception):
        arr._mmap.madvise(_mmap.MADV_RANDOM)  # noqa: SLF001 - numpy has no public madvise
    _PLE_DISK_MAPS[path] = arr
    return torch.from_numpy(arr).view(dtype)


def _ple_disk_attach(layer_name: str, layer: torch.nn.Module,
                     disk_dir: str) -> tuple[str, bool] | None:
    """Swap the layer's largest parameter (the n-gram table) for a disk-backed map.

    Returns ``(param_name, file_complete)`` or ``None`` when the layer has no
    parameter large enough to be worth spilling (>= 1 GiB).
    """
    import os

    named = sorted(layer.named_parameters(), key=lambda kv: kv[1].numel(), reverse=True)
    if not named or named[0][1].numel() * named[0][1].element_size() < (1 << 30):
        return None
    pname, param = named[0]
    shape, dtype = tuple(param.shape), param.dtype
    nbytes = param.numel() * param.element_size()
    os.makedirs(disk_dir, exist_ok=True)
    base = os.path.join(disk_dir, layer_name.replace("/", "_") + "." + pname)
    bin_path, done_path = base + ".bin", base + ".done.json"

    complete = False
    if (os.path.exists(done_path) and os.path.exists(bin_path)
            and os.path.getsize(bin_path) == nbytes):
        meta = json.load(open(done_path))
        complete = meta.get("shape") == list(shape) and meta.get("dtype") == str(dtype)
    if not complete:
        # SAFETY: never clobber a foreign table file. Truncate-means-write-
        # through could silently fp8-ify a BF16 table, so an existing
        # mismatched-size file aborts the boot with the full truth.
        if os.path.exists(bin_path) and os.path.getsize(bin_path) != nbytes:
            raise RuntimeError(
                f"PLE disk: {bin_path} exists at {os.path.getsize(bin_path)} "
                f"bytes but model expects {nbytes} (param dtype {param.dtype}, "
                "shape "
                f"{tuple(param.shape)}). Refusing to overwrite a foreign "
                "table — align PLE_BF16_TABLE/dtype or remove the file."
            )
        with contextlib.suppress(FileNotFoundError):
            os.remove(done_path)
        with open(bin_path, "ab") as f:
            f.truncate(nbytes)

    mapped = _disk_backed_tensor(bin_path, shape, dtype, writable=not complete)
    # Replace the parameter data in place; module structure and names are unchanged,
    # so load_weights and the gather path are untouched.
    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    getattr(owner, parts[-1]).data = mapped
    logger.info(
        "PLE disk offload: %s.%s -> %s (%.1f GiB, %s)",
        layer_name, pname, bin_path, nbytes / (1 << 30),
        "reusing finished file" if complete else "first boot, writing through",
    )
    return pname, complete


def _ple_disk_finalize(layer_name: str, layer: torch.nn.Module, pname: str,
                       disk_dir: str) -> None:
    """Flush the written mapping, record completion, and remap copy-on-write."""
    import os

    owner = layer
    parts = pname.split(".")
    for p in parts[:-1]:
        owner = getattr(owner, p)
    param = getattr(owner, parts[-1])
    base = os.path.join(disk_dir, layer_name.replace("/", "_") + "." + pname)
    arr = _PLE_DISK_MAPS.get(base + ".bin")
    if arr is not None:
        with contextlib.suppress(Exception):
            arr.flush()
    json.dump({"shape": list(param.shape), "dtype": str(param.dtype)},
              open(base + ".done.json", "w"))
    param.data = _disk_backed_tensor(base + ".bin", tuple(param.shape), param.dtype,
                                     writable=False)
    logger.info("PLE disk offload: %s.%s finalized and remapped copy-on-write.",
                layer_name, pname)


def _mem_available_bytes() -> int:
    with open("/proc/meminfo") as meminfo:
        for line in meminfo:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0


def _prefault_tables(runner: "PleOffloadRunner", shutdown_event: threading.Event) -> None:
    """Read one byte per page of the large host tables so decode lookups do not
    fault them in from swap. Read-only; stops when free memory runs low."""
    reserve = int(float(os.environ.get("QWEN38_PLE_PREFAULT_RESERVE_GIB", "4")) * 2**30)
    chunk_bytes = 64 * 2**20
    page = 4096
    started = time.monotonic()
    touched = 0
    reason = "complete"
    with torch.inference_mode():
        for layer_name, layer in runner._layers.items():
            for tensor in layer.state_dict().values():
                if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                        or not tensor.is_contiguous()
                        or tensor.numel() * tensor.element_size() < 2**30):
                    continue
                flat = tensor.reshape(-1).view(torch.uint8)
                for offset in range(0, flat.numel(), chunk_bytes):
                    if shutdown_event.is_set():
                        reason = "shutdown"
                        break
                    if _mem_available_bytes() < reserve:
                        reason = "memory reserve reached"
                        break
                    int(flat[offset:offset + chunk_bytes:page].sum())
                    touched += min(chunk_bytes, flat.numel() - offset)
                if reason != "complete":
                    break
            if reason != "complete":
                break
    logger.info("PLE prefault %s: %.1f GiB in %.1f s", reason, touched / 2**30,
                time.monotonic() - started)

class PleOffloadRunner:
    """Own all discovered PLE tables and serve every local DP rank."""

    def __init__(self, vllm_config: VllmConfig) -> None:
        self.vllm_config = vllm_config
        self._clamp_input_ids = (
            getattr(vllm_config, "speculative_config", None) is not None
        )
        # name -> PleOffloadLayer (CPU)
        self._layers: dict[str, PleOffloadLayer] = {}
        # dp_rank -> layer_name -> one destination per TP rank
        self._worker_targets: dict[int, dict[str, list[PleOffloadOutputTarget]]] = {}
        # Shared-memory inputs are registered once per DP rank by TP rank zero.
        self._input_bufs: dict[int, PleOffloadInputBuffers] = {}
        self._load_weights()

    @property
    def layer_names(self) -> list[str]:
        """Return PleOffloadLayer names in model traversal order."""
        return list(self._layers)

    def _load_weights(self) -> None:
        """Load only :class:`PleOffloadLayer` subtrees into CPU memory.

        Strategy:
        1. Build the entire model on ``meta`` so non-offloaded parameters use no
           physical memory. PleOffloadLayer constructors explicitly target CPU.
        2. Discover all PleOffloadLayer modules from the complete model.
        3. Stream the checkpoint through a prefix filter so only matching PLE
           tensors are materialized and passed to ``model.load_weights``.
        4. Run post-load processing only on the CPU-owned PLE subtrees.
        """
        model_config = self.vllm_config.model_config
        load_config = self.vllm_config.load_config

        # Step 1: build complete structure, while only PLE subtrees allocate CPU
        # memory. All transformer, MoE, and vision parameters remain on meta.
        logger.info("Initializing model structure for PLE weight discovery ...")
        model_dtype = cast(torch.dtype, model_config.dtype)
        with set_default_torch_dtype(model_dtype), torch.device("meta"):
            model = initialize_model(
                vllm_config=self.vllm_config,
                model_config=model_config,
            )

        # Step 2: preserve named_modules DFS order so CPU execution follows the
        # same layer order as the GPU model forward.
        offload_layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not offload_layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but no PleOffloadLayer "
                "was found in the initialized model"
            )
        logger.info(
            "Found %d PleOffloadLayer(s): %s",
            len(offload_layers),
            sorted(offload_layers),
        )
        offload_prefixes = tuple(f"{name}." for name in offload_layers)

        quant_dir = _ple_quant_dir()
        disk_dir = _ple_disk_dir() if quant_dir is None else None
        disk_attached: dict[str, str] = {}
        disk_complete_params: set[str] = set()
        disk_complete_tables: tuple[str, ...] = ()
        if quant_dir is not None:
            table_prefixes = []
            for layer_name, layer in offload_layers.items():
                pname = _ple_quant_attach(layer_name, layer, quant_dir)
                if pname is None:
                    continue
                full = f"{layer_name}.{pname}"
                disk_complete_params.add(full)
                table_prefixes.append(full.rsplit(".", 1)[0])
            disk_complete_tables = tuple(table_prefixes)
        if disk_dir is not None:
            table_prefixes = []
            for layer_name, layer in offload_layers.items():
                attached = _ple_disk_attach(layer_name, layer, disk_dir)
                if attached is None:
                    continue
                pname, complete = attached
                disk_attached[layer_name] = pname
                if complete:
                    full = f"{layer_name}.{pname}"
                    disk_complete_params.add(full)
                    # "...ngram_embedding.weight" -> "...ngram_embedding": the module
                    # whose shard_N.weight checkpoint tensors fill this table.
                    table_prefixes.append(full.rsplit(".", 1)[0])
            # Shard tensors that land inside an already-finished table are not
            # re-read from the checkpoint: mapping the file replaces them.
            disk_complete_tables = tuple(table_prefixes)

        # Step 3: filter checkpoint tensors before model.load_weights(). The
        # conditional-generation checkpoint uses HF names such as
        # ``model.language_model.*`` while named_modules exposes mapped vLLM
        # names such as ``language_model.model.*``. Apply the model mapper only
        # for matching, then yield the original pair so load_weights performs
        # its normal single mapping pass.
        mapper = getattr(model, "hf_to_vllm_mapper", None)
        matched_checkpoint_tensors = 0

        def offload_only_iter(
            weights: Iterable[tuple[str, torch.Tensor]],
        ) -> Iterable[tuple[str, torch.Tensor]]:
            nonlocal matched_checkpoint_tensors
            for weight_name, tensor in weights:
                mapped_name: str | None = weight_name
                if mapper is not None:
                    mapped_names = mapper.apply_list([weight_name])
                    mapped_name = mapped_names[0] if mapped_names else None
                if mapped_name is not None and mapped_name.startswith(offload_prefixes):
                    matched_checkpoint_tensors += 1
                    if disk_complete_tables:
                        table = _ple_disk_shard_of(mapped_name)
                        if table is None and mapped_name.endswith(
                            (".weight", ".weight_scale", ".weight_scale_2", ".input_scale")
                        ):
                            # Checkpoints that ship the table as one tensor plus a
                            # global scale (the FP8 revision and the NVFP4 builds
                            # derived from it) instead of shard_N.weight parts. The
                            # sidecar owns those rows, so drop them before the loader
                            # tries to write a 51.2B tensor into the stub.
                            table = mapped_name.rsplit(".", 1)[0]
                        if table is not None and table.startswith(disk_complete_tables):
                            if (
                                disk_attached
                                and mapped_name.endswith(".weight_scale")
                                and os.environ.get("PLE_BF16_TABLE") != "1"
                            ):
                                # exp/disk8: a raw disk table (.bin) carries the
                                # weight rows only; the global weight_scale stays a
                                # small real parameter that load_weights must fill.
                                # Quant-sidecar mode (disk_attached empty) keeps
                                # the original drop-weight-and-scale behavior.
                                yield weight_name, tensor
                                continue
                            continue
                    yield weight_name, tensor

        loader = get_model_loader(load_config)
        if isinstance(loader, DummyModelLoader):
            logger.info(
                "Initializing dummy weights for %d PleOffloadLayer(s) ...",
                len(offload_layers),
            )
            for layer in offload_layers.values():
                initialize_dummy_weights(layer, model_config)
        elif isinstance(loader, DefaultModelLoader):
            all_weights = loader.get_all_weights(model_config, model)
            loaded_params = model.load_weights(offload_only_iter(all_weights))
            if matched_checkpoint_tensors == 0:
                raise RuntimeError(
                    "PLE offload checkpoint filter matched no weights for "
                    f"layers: {sorted(offload_layers)}"
                )

            expected_offload_params = {
                f"{layer_name}.{param_name}"
                for layer_name, layer in offload_layers.items()
                for param_name, _ in layer.named_parameters()
            }
            loaded_offload_entries = {
                name for name in loaded_params if name.startswith(offload_prefixes)
            }
            loaded_expected_params = expected_offload_params.intersection(loaded_params)
            missing_offload_params = sorted(
                expected_offload_params.difference(loaded_expected_params)
                - disk_complete_params
            )
            if missing_offload_params:
                raise RuntimeError(
                    "PLE offload checkpoint did not load all materialized "
                    f"parameters: {missing_offload_params}"
                )
            logger.info(
                "PLE offload matched %d checkpoint tensor(s), loaded %d "
                "offload entries, and verified %d/%d materialized "
                "parameter(s) for layers: %s",
                matched_checkpoint_tensors,
                len(loaded_offload_entries),
                len(loaded_expected_params),
                len(expected_offload_params),
                sorted(offload_layers),
            )
        else:
            raise NotImplementedError(
                "PLE offload requires the default or dummy model loader, got "
                f"{type(loader).__name__}"
            )

        # Step 4: post-load processing is restricted to CPU-owned PLE modules;
        # the remainder of the model is still on meta and must not be visited.
        for layer in offload_layers.values():
            process_weights_after_loading(layer, model_config, torch.device("cpu"))

        if disk_dir is not None:
            for layer_name, pname in disk_attached.items():
                if f"{layer_name}.{pname}" not in disk_complete_params:
                    _ple_disk_finalize(layer_name, offload_layers[layer_name],
                                       pname, disk_dir)

        self._layers.update(offload_layers)
        del model
        logger.info("PLE weight loading complete.")

    def accept_registrations(
        self,
        pull_socket: zmq.Socket,
        num_workers: int,
    ) -> None:
        """Receive every local DP/TP worker's IPC and shared-memory buffers."""
        logger.info("Waiting for %d GPU worker registration(s) ...", num_workers)
        registrations: list[PleOffloadRegistration] = []
        for index in range(num_workers):
            item = pickle.loads(pull_socket.recv())
            if not isinstance(item, PleOffloadRegistration):
                raise RuntimeError(
                    "Expected PleOffloadRegistration during setup, got "
                    f"{type(item).__name__} ({index + 1}/{num_workers})"
                )
            registrations.append(item)
            logger.info(
                "GPU worker %d registered (dp_rank=%d, tp_rank=%d, layers=%s).",
                item.worker_id,
                item.dp_rank,
                item.tp_rank,
                sorted(item.cpu_output_buffers),
            )

        dp_size = self.vllm_config.parallel_config.data_parallel_size
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if num_workers != dp_size * tp_size:
            raise RuntimeError(
                f"Expected {dp_size * tp_size} registrations for DP={dp_size}, "
                f"TP={tp_size}, got {num_workers}"
            )

        registrations_by_dp: dict[int, list[PleOffloadRegistration]] = {}
        for registration in registrations:
            registrations_by_dp.setdefault(registration.dp_rank, []).append(
                registration
            )
        if set(registrations_by_dp) != set(range(dp_size)):
            raise RuntimeError(
                f"Expected DP ranks {set(range(dp_size))}, "
                f"got {set(registrations_by_dp)}"
            )
        for dp_rank, dp_registrations in registrations_by_dp.items():
            tp_ranks = {registration.tp_rank for registration in dp_registrations}
            if tp_ranks != set(range(tp_size)):
                raise RuntimeError(
                    f"DP rank {dp_rank} expected TP ranks {set(range(tp_size))}, "
                    f"got {tp_ranks}"
                )

        for registration in registrations:
            if set(registration.cpu_output_buffers) != set(self.layer_names):
                raise RuntimeError(
                    "Registered PLE layers do not match CPU layers: "
                    f"registered={sorted(registration.cpu_output_buffers)}, "
                    f"cpu={sorted(self.layer_names)}"
                )
            if (
                set(registration.cpu_output_ready_flags) != set(self.layer_names)
                or set(registration.cpu_output_consumed_flags)
                != set(self.layer_names)
            ):
                raise RuntimeError("Registered PLE handshake flags do not match layers")
            targets_for_dp = self._worker_targets.setdefault(registration.dp_rank, {})
            for layer_name, cpu_buffer in registration.cpu_output_buffers.items():
                ready_flag = registration.cpu_output_ready_flags[layer_name]
                consumed_flag = registration.cpu_output_consumed_flags[layer_name]
                if (
                    cpu_buffer.device.type != "cpu"
                    or not cpu_buffer.is_shared()
                    or ready_flag.device.type != "cpu"
                    or not ready_flag.is_shared()
                    or ready_flag.dtype != torch.int32
                    or ready_flag.numel() != 1
                    or consumed_flag.device.type != "cpu"
                    or not consumed_flag.is_shared()
                    or consumed_flag.dtype != torch.int32
                    or consumed_flag.numel() != 1
                ):
                    raise RuntimeError("PLE outputs must be shared CPU tensors")
                target = PleOffloadOutputTarget(
                    tp_rank=registration.tp_rank,
                    cpu_output_buffer=cpu_buffer,
                    ready_flag=ready_flag,
                    consumed_flag=consumed_flag,
                )
                targets_for_dp.setdefault(layer_name, []).append(target)
            # All TP ranks in one DP group receive the same input, so buffers
            # registered by TP rank zero are sufficient for that DP rank.
            if registration.tp_rank == 0:
                self._input_bufs[registration.dp_rank] = PleOffloadInputBuffers(
                    input_ids_buf=registration.input_ids_buf,
                    query_start_loc_buf=registration.query_start_loc_buf,
                    ngram_context_buf=registration.ngram_context_buf,
                )

        if set(self._input_bufs) != set(range(dp_size)):
            raise RuntimeError(
                "TP rank zero did not register PLE input buffers for every DP "
                f"rank: expected={set(range(dp_size))}, got={set(self._input_bufs)}"
            )

        for dp_rank, layer_targets in self._worker_targets.items():
            for layer_name, targets in layer_targets.items():
                if len(targets) != tp_size:
                    raise RuntimeError(
                        f"PLE layer {layer_name} for DP rank {dp_rank} received "
                        f"{len(targets)} targets, expected {tp_size}"
                    )
                targets.sort(key=lambda target: target.tp_rank)
        logger.info(
            "Registrations complete (dp_size=%d, tp_size=%d, layers=%s).",
            dp_size,
            tp_size,
            sorted(self.layer_names),
        )

    @torch.inference_mode()
    def busy_loop(
        self,
        pull_socket: zmq.Socket,
        shutdown_event: threading.Event,
    ) -> None:
        """Decode and batch available requests by DP rank until shutdown."""
        logger.info("Busy-loop started.")
        poller = zmq.Poller()
        poller.register(pull_socket, zmq.POLLIN)
        while not shutdown_event.is_set():
            if pull_socket not in dict(poller.poll(timeout=100)):
                continue

            requests = []
            try:
                requests.append(_PLE_OFFLOAD_REQUEST_DECODER.decode(pull_socket.recv()))
                while True:
                    requests.append(
                        _PLE_OFFLOAD_REQUEST_DECODER.decode(
                            pull_socket.recv(zmq.NOBLOCK)
                        )
                    )
            except zmq.Again:
                pass
            except msgspec.DecodeError as error:
                raise RuntimeError("Unexpected PLE offload request") from error

            self._handle_requests(requests)

    def _handle_requests(self, requests: list[PleOffloadRequest]) -> None:
        """Run requests layer-first so each DP rank can resume promptly."""
        requests_by_dp: dict[int, PleOffloadRequest] = {}
        for request in requests:
            if request.dp_rank not in self._worker_targets:
                logger.warning(
                    "No PLE output targets for dp_rank=%d; skipping request.",
                    request.dp_rank,
                )
                continue
            if request.dp_rank in requests_by_dp:
                logger.warning(
                    "Duplicate PLE request for dp_rank=%d; skipping duplicate.",
                    request.dp_rank,
                )
                continue
            requests_by_dp[request.dp_rank] = request

        # Speculative placeholders are not vocabulary IDs. Normalize each DP
        # input once before all PLE layers consume the shared buffer.
        if self._clamp_input_ids:
            for dp_rank, request in requests_by_dp.items():
                self._input_bufs[dp_rank].input_ids_buf[
                    : request.num_tokens
                ].clamp_min_(0)

        for layer_name, layer in self._layers.items():
            for dp_rank, request in requests_by_dp.items():
                targets = self._worker_targets[dp_rank][layer_name]

                # The captured GPU stream sets consumed=1 only after H2D has
                # finished, and resets ready before doing so. Claim every TP
                # destination before computing the next shared result.
                for target in targets:
                    while int(target.consumed_flag.item()) != 1:
                        time.sleep(0)
                for target in targets:
                    target.ready_flag.zero_()
                    target.consumed_flag.zero_()

                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (
                    input_bufs.ngram_context_buf[: request.num_reqs]
                    if input_bufs.ngram_context_buf is not None
                    else None
                )
                result = layer.forward_impl(
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.input_ids_buf[: request.num_tokens],
                    input_bufs.query_start_loc_buf[: request.num_reqs + 1],
                    ngram_context,
                    output_buffer=targets[0].cpu_output_buffer,
                )

                # The result is identical on every TP rank. Fan it out in host
                # memory, then publish readiness only after all stores finish.
                slices = tuple(slice(0, size) for size in result.shape)
                for target in targets[1:]:
                    target.cpu_output_buffer[slices].copy_(result[slices])
                for target in targets:
                    target.ready_flag.fill_(1)
