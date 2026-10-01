# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention.
"""

from __future__ import annotations

import contextlib
import os
import queue
import struct
import threading
import zlib

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096


class PersistIntegrityError(RuntimeError):
    """A persisted slot failed its CRC check on load.

    Raised in the load-thread; the worker re-raises it from get_finished()
    (main thread) so vLLM's normal worker-failure surface fires. Serving the
    bytes is never an option: fail loud, never confidently wrong.
    """

    def __init__(self, message: str, slot: int | None = None) -> None:
        super().__init__(message)
        self.slot = slot


def _alloc_aligned(num_slots: int, bpb: int) -> torch.Tensor:
    """Allocate a staging buffer whose base address is O_DIRECT aligned.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and return an aligned view. The view keeps the backing
    storage alive.
    """
    nbytes = num_slots * bpb
    raw = torch.zeros(nbytes + _ALIGNMENT, dtype=torch.int8, device="cpu")
    offset = -raw.data_ptr() % _ALIGNMENT
    return raw[offset : offset + nbytes].view(num_slots, bpb)


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while preadv/pwritev slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._persist: bool = False
        self._crc_fd: int = -1
        self._crc_dirty = threading.Event()
        self._thread_failure: BaseException | None = None
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        self._store_buffer_caches: dict[str, torch.Tensor] = {}
        self._load_buffer_caches: dict[str, torch.Tensor] = {}
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int = 2,
        use_page_cache: bool = False,
        persist: bool = False,
        crc_path: str | None = None,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        assert total_block_bytes % _ALIGNMENT == 0, (
            f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
        )

        # Separate buffer pools for store and load threads
        self._store_buffer_caches = {}
        self._load_buffer_caches = {}
        for name, gpu_t in gpu_caches.items():
            bpb = gpu_t.stride(0) * gpu_t.element_size()
            store_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(store_buf)
            self._store_buffer_caches[name] = store_buf
            load_buf = _alloc_aligned(num_buffer_slots, bpb)
            pin_tensor(load_buf)
            self._load_buffer_caches[name] = load_buf

        # Pre-built iovec views per slot (avoid per-transfer .numpy() calls)
        self._store_slot_views = [
            [
                memoryview(self._store_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            [
                memoryview(self._load_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]

        self._store_params = build_params(
            gpu_caches,
            self._store_buffer_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            self._load_buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        expected_size = num_disk_slots * total_block_bytes
        self._persist = persist
        if persist:
            # Persistent arm: reopen the existing tier file as-is. Contents
            # are trusted ONLY through the CRC + journal + fingerprint gates;
            # a geometry mismatch is a hard stop, never a silent truncate.
            flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
            if not use_page_cache:
                flags |= O_DIRECT
            try:
                existing = os.stat(disk_path).st_size
            except FileNotFoundError:
                existing = 0
            if existing not in (0, expected_size):
                raise RuntimeError(
                    f"DiskBackend[persist]: {disk_path} has size {existing}, "
                    f"expected {expected_size} ({num_disk_slots} slots x "
                    f"{total_block_bytes}B) -- tier layout changed, refuse to "
                    "reuse or truncate. Wipe the tier dir for a cold start."
                )
            self._fd = os.open(disk_path, flags, 0o600)
            if existing == 0:
                os.ftruncate(self._fd, expected_size)
                logger.info("DiskBackend[persist]: creating cold tier %s", disk_path)
            if crc_path is None:
                raise RuntimeError("persist mode requires crc_path")
            self._crc_fd = os.open(
                crc_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            crc_size = num_disk_slots * 4
            if os.fstat(self._crc_fd).st_size < crc_size:
                os.ftruncate(self._crc_fd, crc_size)
        else:
            # Legacy in-boot arm: slot contents never outlive the process, so
            # unlink then O_EXCL rather than reopening: a pre-existing file
            # would otherwise keep its own (possibly world-readable) mode,
            # and blocks may encode user prompts.
            with contextlib.suppress(FileNotFoundError):
                os.unlink(disk_path)
            # O_DIRECT by default: page cache would consume the very host DRAM
            # this backend exists to conserve, and doubles the copy on the
            # store path.
            flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
            if not use_page_cache:
                flags |= O_DIRECT
            self._fd = os.open(disk_path, flags, 0o600)
            os.ftruncate(self._fd, expected_size)
        self._disk_path = disk_path

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes"
            " (page_cache=%s)",
            disk_path,
            num_disk_slots,
            (num_disk_slots * total_block_bytes) / (1024**3),
            num_buffer_slots,
            total_block_bytes,
            use_page_cache,
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        if self._fd < 0:
            return
        if self._persist:
            # Persistent arm: keep the name and the bytes; the next boot
            # revalidates through fingerprint + journal + CRC.
            logger.info(
                "DiskBackend[persist]: keeping tier file %s across shutdown",
                self._disk_path,
            )
        else:
            # Slot contents can encode user prompts, so drop the name now rather
            # than leaving them readable until the next run overwrites the file.
            # Unlinking only removes the directory entry: any thread still holding
            # the fd keeps writing to the (now anonymous) inode, which the kernel
            # frees once the last fd goes away.
            with contextlib.suppress(OSError):
                os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        os.close(self._fd)
        self._fd = -1

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            try:
                self._do_store(src_blocks, dst_blocks, stream)
            except Exception as exc:  # noqa: BLE001 - escalate via get_finished
                self.fail_thread(exc)
                continue
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def fail_thread(self, exc: BaseException) -> None:
        """Record a fatal IO/integrity error for the main thread to re-raise."""
        if self._thread_failure is None:
            self._thread_failure = exc
        slot = getattr(exc, "slot", None)
        if isinstance(slot, int) and self._persist:
            self._write_poison(slot)
        logger.critical(
            "DiskBackend IO thread failure (escalating): %s", exc, exc_info=True
        )

    def _write_poison(self, slot: int) -> None:
        """Mark a slot untrustworthy for the NEXT boot.

        Without this a corrupt slot is a landmine: the journal entry survives
        the crash, every later boot replays it, and the same request kills the
        lane again. The scheduler consumes these records at boot and journals
        a FREE for them, degrading a poison slot into a plain cache miss.
        """
        try:
            with open(f"{self._disk_path}.poison", "ab") as f:
                f.write(struct.pack("<I", slot))
                f.flush()
                os.fsync(f.fileno())
            logger.critical(
                "DiskBackend[persist]: slot %d marked poisoned "
                "(next boot frees it -> miss instead of repeat crash)", slot
            )
        except OSError:
            logger.exception("DiskBackend[persist]: failed to write poison record")

    def take_failure(self) -> BaseException | None:
        exc, self._thread_failure = self._thread_failure, None
        return exc

    def _slot_crc(self, slot_views: list[memoryview]) -> int:
        crc = 0
        for mv in slot_views:
            crc = zlib.crc32(mv, crc)
        return crc & 0xFFFFFFFF

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        written = os.pwritev(self._fd, self._store_slot_views[buf_slot], file_offset)
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )
        if self._crc_fd >= 0:
            # Durability order: data (O_DIRECT, device-acked here) -> CRC
            # (fsync'd before the batch's DMA event completes, see
            # _do_store) -> scheduler journals ADD only after the event.
            slot = file_offset // self._total_block_bytes
            crc = struct.pack("<I", self._slot_crc(self._store_slot_views[buf_slot]))
            os.pwrite(self._crc_fd, crc, slot * 4)
            self._crc_dirty.set()

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        bytes_read = os.preadv(self._fd, self._load_slot_views[buf_slot], file_offset)
        if bytes_read < self._total_block_bytes:
            raise OSError(
                f"Short read: expected {self._total_block_bytes} bytes, "
                f"read {bytes_read}"
            )
        if self._crc_fd >= 0:
            slot = file_offset // self._total_block_bytes
            stored = os.pread(self._crc_fd, 4, slot * 4)
            if len(stored) < 4:
                raise PersistIntegrityError(
                    f"persist tier: no CRC record for slot {slot}", slot=slot
                )
            want = struct.unpack("<I", stored)[0]
            got = self._slot_crc(self._load_slot_views[buf_slot])
            if want != got:
                raise PersistIntegrityError(
                    f"persist tier: slot {slot} CRC mismatch "
                    f"(stored {want:#010x} != recomputed {got:#010x}) -- "
                    "stale or torn bytes, refusing to DMA",
                    slot=slot,
                )

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            try:
                self._do_load(src_blocks, dst_blocks, stream)
            except Exception as exc:  # noqa: BLE001 - escalate via get_finished
                # Failed load: DO NOT record the completion event — the
                # request must never be treated as loaded until the worker
                # main thread re-raises and vLLM fails it.
                self.fail_thread(exc)
                continue
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        assert self._store_params is not None
        n = self._num_buffer_slots
        # (DMA event, file offset) of the block already staged in each slot.
        pending: list[tuple[torch.Event, int] | None] = [None] * n

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                prev[0].synchronize()
                self._writev_slot(buf_slot, prev[1])

            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (ev, disk_slot * self._total_block_bytes)

        for slot, last in enumerate(pending):
            if last is not None:
                last[0].synchronize()
                self._writev_slot(slot, last[1])

        if self._crc_fd >= 0 and self._crc_dirty.is_set():
            # One fsync per store batch, before the completion event is
            # recorded: guarantees CRC durability precedes the scheduler's
            # journal ADD for every block in this batch.
            os.fsync(self._crc_fd)
            self._crc_dirty.clear()

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_params is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            self._readv_slot(buf_slot, disk_slot * self._total_block_bytes)

            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev
