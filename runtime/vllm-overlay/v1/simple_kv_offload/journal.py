# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistence layer for the disk KV tier: add-only journal + fingerprint.

Design (see docs/kv-tier-persistence.md):

- The tier files are fixed-slot arrays: slot i lives at offset i*bpb in each
  per-rank file. The only thing lost across a restart was the in-RAM index
  mapping block_hash(+group) -> slot. This module persists exactly that.

- ADD-only journal. Each record is (kind, key, slot, seq):
    ADD:  key(36B hash+group) now legitimately occupies slot.
    FREE: slot bytes were overwritten by a store that never completed
          caching (abandoned/reset path) -- any older ADD for this slot is
          untrustworthy.
  Replay applies records in order: last ADD per slot wins (slot reuse),
  last ADD per key wins, FREE clears a slot's key. A hash's data bytes are
  only invalidated by an overwrite of its slot; eviction alone leaves bytes
  intact and re-readable -- so no explicit evict records are needed.

- Ordering contract with the workers: slot data is written O_DIRECT (device
  ack) and the per-slot CRC is pwrite+fsync'd BEFORE the worker's DMA event
  completes; the scheduler only appends the ADD record after the event
  completes. Losing the tail of J on power loss therefore degrades to a
  cache miss, never to a stale-serve. A torn last record fails its per-record
  CRC and is dropped.

- Load-time verification is the backstop against device/FTL lies: the worker
  recomputes the slot CRC after every pread and refuses to DMA on mismatch.

This module is stdlib-only so both the scheduler manager and tests can use
it without pulling GPU/torch dependencies.
"""

import json
import os
import struct
import zlib
from typing import Iterable


# --------------------------------------------------------------------------
# Journal records
# --------------------------------------------------------------------------

KIND_ADD = 1
KIND_FREE = 2

_REC_BODY = struct.Struct(
    "<BI36sII"  # kind, reserved, key(32B hash + 4B group id), slot, seq
)
_REC_CRC = struct.Struct("<I")
RECORD_SIZE = _REC_BODY.size + _REC_CRC.size  # 49 -> padded below
RECORD_SIZE = ((RECORD_SIZE + 7) // 8) * 8  # 56
KEY_LEN = 36  # BlockHash (32B) + group id (4B big-endian), as in vLLM v1


def encode_add(seq: int, key: bytes, slot: int) -> bytes:
    assert len(key) == KEY_LEN, f"unexpected block-hash key length {len(key)}"
    body = _REC_BODY.pack(KIND_ADD, 0, key, slot, seq & 0xFFFFFFFF)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    rec = body + _REC_CRC.pack(crc)
    return rec + b"\x00" * (RECORD_SIZE - len(rec))


def encode_free(seq: int, slot: int) -> bytes:
    body = _REC_BODY.pack(KIND_FREE, 0, b"\x00" * KEY_LEN, slot, seq & 0xFFFFFFFF)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    rec = body + _REC_CRC.pack(crc)
    return rec + b"\x00" * (RECORD_SIZE - len(rec))


class JournalWriter:
    """Append-only journal writer. Buffered + flush(); NO fsync required:

    Durability ordering (data -> CRC -> ADD, enforced by the workers) makes
    losing unsynced journal tails a harmless miss. fsync here would only add
    scheduler-side latency to a hot step loop.
    """

    def __init__(self, path: str):
        self.path = path
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        self._fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW,
                           0o600)
        self._seq = 0

    def add_batch(self, pairs: Iterable[tuple[bytes, int]]) -> None:
        if not pairs:
            return
        buf = bytearray()
        for key, slot in pairs:
            self._seq += 1
            buf += encode_add(self._seq, key, slot)
        # Deliberately no fsync (see class docstring): losing this tail on
        # power loss must degrade to a miss, and durability ordering
        # (data -> CRC -> ADD) already guarantees that direction.
        os.write(self._fd, bytes(buf))

    def free_batch(self, slots: Iterable[int]) -> None:
        buf = bytearray()
        for slot in slots:
            self._seq += 1
            buf += encode_free(self._seq, slot)
        if buf:
            os.write(self._fd, bytes(buf))

    def flush(self) -> None:
        # os.write already handed bytes to the page cache; nothing buffered in
        # Python. Kept as an explicit seam for callers/tests.
        pass

    def close(self) -> None:
        try:
            os.close(self._fd)
        except OSError:
            pass


class JournalReader:
    """Replay a journal into (key -> slot) with last-ADD-wins semantics."""

    @staticmethod
    def replay(path: str, num_slots: int) -> tuple[dict[bytes, int], dict]:
        stats = {"records": 0, "torn_tail": 0, "bad_crc": 0, "out_of_range": 0,
                 "freed": 0}
        slot_to_key: dict[int, bytes] = {}
        if not os.path.exists(path):
            return {}, stats
        data = open(path, "rb").read()
        n_full = len(data) // RECORD_SIZE
        tail = len(data) - n_full * RECORD_SIZE
        if tail:
            stats["torn_tail"] = tail
        for i in range(n_full):
            rec = data[i * RECORD_SIZE:(i + 1) * RECORD_SIZE]
            body = rec[:_REC_BODY.size]
            (crc_stored,) = _REC_CRC.unpack(rec[_REC_BODY.size:_REC_BODY.size + 4])
            if zlib.crc32(body) & 0xFFFFFFFF != crc_stored:
                stats["bad_crc"] += 1
                continue
            kind, _rsv, key, slot, _seq = _REC_BODY.unpack(body)
            stats["records"] += 1
            if slot >= num_slots:
                stats["out_of_range"] += 1
                continue
            if kind == KIND_FREE:
                old = slot_to_key.pop(slot, None)
                if old is not None:
                    stats["freed"] += 1
            elif kind == KIND_ADD:
                slot_to_key[slot] = key
        key_to_slot = {k: s for s, k in slot_to_key.items()}
        return key_to_slot, stats


# --------------------------------------------------------------------------
# Fingerprint: fail closed to COLD, never reuse bytes across a semantic change
# --------------------------------------------------------------------------

FP_VERSION = 1
# Bump when the journal record format or its semantics change: an old journal
# must never be replayed by code that reads a different layout.
JOURNAL_FORMAT = 1

# Fields that decide whether persisted slot bytes are still VALID FOR THE SAME
# TOKENS. Everything here is a byte-semantics field; connector file hashes are
# deliberately NOT compared (a bookkeeping fix must not cold-start the tier) --
# they ride along in "debug" for forensics. Journal format is compared instead.
COMPARE_FIELDS = (
    "model_config_sha",
    "hf_overrides",
    "kv_cache_dtype",
    "tp_size",
    "hash_seed",
    "capacity_bytes",
    "vllm_version",
    "journal_format",
    # PLE table identity: the table feeds per-layer token embeddings, so a
    # table TIER switch (bf16/fp8/int4/nvfp4) changes the KV bytes for the same
    # tokens. The hand rule "change disk_path with the table tier" is enforced
    # here automatically.
    "ple_table",
)


def compute_runtime_fields(model_dir: str, vllm_config, kv_transfer_config,
                           module_files: list[str]) -> dict:
    """Fields the engine can witness itself at scheduler-init time."""
    import hashlib

    def _sha_file(p):
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        except OSError:
            return None

    import vllm as _vllm

    extra = getattr(kv_transfer_config, "kv_connector_extra_config", {}) or {}

    # PLE table identity (see COMPARE_FIELDS): flags + dir + done-marker hash.
    ple_dir = os.environ.get("VLLM_PLE_DISK_OFFLOAD_DIR") or ""
    done_sha = None
    if ple_dir and os.path.isdir(ple_dir):
        import glob
        markers = sorted(glob.glob(os.path.join(ple_dir, "*.done.json")))
        if markers:
            done_sha = {os.path.basename(m): _sha_file(m) for m in markers}
    ple_table = {
        "bf16_flag": os.environ.get("PLE_BF16_TABLE"),
        "arm_int4": os.environ.get("PLE_ARM_INT4"),
        "arm_nvfp4": os.environ.get("PLE_ARM_NVFP4"),
        "dir": ple_dir,
        "done_json": done_sha,
    }

    fields = {
        "model_config_sha": _sha_file(os.path.join(model_dir, "config.json")),
        "hf_overrides": os.environ.get("HF_OVERRIDES_JSON"),
        "kv_cache_dtype": str(
            getattr(vllm_config.cache_config, "dtype", None)),
        "block_size": getattr(vllm_config.cache_config, "block_size", None),
        "tp_size": getattr(vllm_config.parallel_config, "tensor_parallel_size",
                           None),
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
        "capacity_bytes": extra.get("disk_capacity_bytes"),
        "vllm_version": getattr(_vllm, "__version__", "?"),
        "journal_format": JOURNAL_FORMAT,
        # PLE table identity (compared; see COMPARE_FIELDS)
        "ple_table": ple_table,
        # diagnostics only -- never compared (see COMPARE_FIELDS)
        "connector_shas": {os.path.basename(p): _sha_file(p)
                           for p in module_files},
    }
    return fields


def compute_layout_fields(cpu_kv_cache_config) -> dict:
    """Slot geometry the journal's ids are only meaningful under."""
    return {
        "num_blocks": cpu_kv_cache_config.num_blocks,
        "groups": repr([
            (g.kv_cache_group_id if hasattr(g, "kv_cache_group_id") else i,
             repr(g.kv_cache_spec))
            for i, g in enumerate(cpu_kv_cache_config.kv_cache_groups)
        ]),
    }


def load_fp(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def save_fp(path: str, fp: dict) -> None:
    """Write the fingerprint atomically (tmp + fsync + rename)."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(fp, f, sort_keys=True, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
