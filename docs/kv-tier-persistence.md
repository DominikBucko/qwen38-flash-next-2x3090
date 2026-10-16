# Persistent disk KV tier (`"persist": true`)

`SimpleCPUOffloadConnector` with `kv_offload_backend: "disk"` already writes
every completed prefix block to NVMe while the request runs (eager mode), and
restores it on re-lookup — within one boot. A restart wipes that:
`DiskBackend.init()` unlinks the tier file and `shutdown()` unlinks it again,
the `block_hash -> slot` index lives only in scheduler RAM, and block hashing
is re-randomized every boot. The persistence arm removes all three walls so a
prefix computed once is never re-prefilled again across restarts. Inert —
byte-identical legacy behavior — unless the connector config says
`"persist": true`.

## Why it pays

Restore is I/O, re-prefill is compute. Measured on a community rig
(4x RTX 5060 Ti 16 GB, TP4, fp8 KV, block=800): a full 13.7 GB session-tier
restore reads in **~2.5 s**, against **4.5–6.5 min** to re-prefill the same
tokens (2,500 tok/s class prefill); a 110K-token restore across a full process
stop/start came in at **2.61 s**. With a 25 GiB tier (~2400 slots/rank at
5.5 MB/slot) that is roughly four full 455K sessions' prefixes surviving
forever — across deploys, reboots and lane switches.

## Why naive persistence is dangerous — and the three gates

Cross-request KV contamination is a known genre for local-disk KV backends
(cf. LMCache issue #4385). Wrong KV does not crash: it silently answers with
someone else's context. Every accept in this arm requires **journal ∧ CRC ∧
fingerprint**, and every ambiguous case fails closed to a cache miss:

1. **Add-only journal** (`journal.py`, stdlib-only, scheduler-side): 56-byte
   fixed records `(ADD|FREE, key=hash[36B]+group[4B], slot, seq)` with
   per-record CRC. Replay: last ADD per slot wins, last ADD per key wins, FREE
   kills a claim. No EVICT records are needed — eviction leaves bytes intact
   and valid for their hash; only a slot overwrite invalidates, and every
   overwrite is followed by either ADD (completed store) or FREE (abandoned
   store). Durability ordering is the crash-safety contract:
   `data (O_DIRECT pwritev, device-acked) -> slot CRC (one fsync per store
   batch) -> worker DMA completion -> scheduler appends ADD`. Losing the
   journal tail to power loss is a miss (safe); a journaled slot whose bytes or
   CRC disagree is refused (safe).
2. **Load-time CRC** (`disk_backend._readv_slot`): recompute after every read;
   mismatch raises `PersistIntegrityError`, the load thread records no
   completion, and the error re-raises on the worker main thread into vLLM's
   normal loud surface — never painted over. The slot is also marked poisoned
   so the next boot frees it: a corrupt block becomes a plain miss, not a
   landmine.
3. **Fingerprint gate** (`manager._persist_replay`): `<stem>.fp.json` holds
   the byte-semantics set — model `config.json` sha256, `HF_OVERRIDES_JSON`
   (rope changes rewrite position semantics), KV dtype, TP size,
   `PYTHONHASHSEED`, tier capacity, engine version, journal format, plus the
   derived layout (num_blocks, group specs). Any drift refuses the replay,
   quarantines the journal, and boots cold — the tier itself stays mounted and
   keeps accumulating fresh stores. The PLE table identity is part of the
   fingerprint: switching the n-gram table tier (bf16/fp8/int4/nvfp4) changes
   the KV bytes for identical tokens, so the old "change disk_path with the
   table tier" hand rule is enforced automatically here.

The known residual: a CRC-and-bytes pair can only both lie via an SSD FTL lie
after power loss — the same trust class the machine's model storage already
lives with.

## The hash-seed wall

`kv_cache_utils.init_none_hash()` seeds the block-hash chain from
`os.urandom(32)` whenever `PYTHONHASHSEED` is unset, so every boot mints new
hashes for identical prefixes and cross-boot keys can never match. The persist
arm therefore **refuses to boot without a fixed `PYTHONHASHSEED`** in the
container environment (vLLM's own warning recommends pinning it for
reproducibility). Pin it to a constant and keep it pinned; changing it is a
cold-boot event by construction.

## Usage

```json
KV_TIER_JSON={"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"kv_offload_backend":"disk",
    "disk_path":"/var/lib/qwen38/kv-tier","disk_capacity_bytes":26843545600,
    "disk_buffer_slots":16,"persist":true}}
PYTHONHASHSEED=1888275146
```

`scripts/serve-container.sh` passes it as `--kv-transfer-config` when set.
Expected sequence on first enable: boot #1 is cold by design and writes
`fp.json`; after a clean restart boot #2 logs `replayed=N/M slots`. Good
acceptance probes: a long session across a restart answering "what did I say
earlier?", plus a fresh-prefix control that must miss normally.
`lazy_offload` is rejected in persist mode (lazy-store bookkeeping is not
journaled); eager mode is the supported configuration.

## Limits, honestly

- Persist mode is eager-only (guarded at config parse, loud).
- A pre-existing upstream FIXME (`_prepare_eager_store_specs`) about stale
  `num_stored_blocks` under eviction becomes reachable when a tier fills;
  worst case is a skipped store (miss). Worth a review follow-up.
- Endurance: a tier on a consumer NVMe next to the model tables sees a
  write-heavy all-prefill workload; the arm inherits the disk's own endurance
  budget. This is a placement decision, not a code property.
