# VMM + LRU hybrid — one expert tier, two access paths (golden-v2.0)

Status: PROMOTED 2026-09-24 as the :8000 boot default (tier `golden-v2.0`, arm
`PREFILL_VMM_ARM=2`). Supersedes the `PREFILL_VMM_ARM=0|1` either/or choice.

## 1. What this is

A 125B-class MoE model (Qwen3.8-Flash-Next, W4A16 int4 experts + FP8 PLE) is served on
4x 16 GB cards. The int4 expert weights are ~63 GiB and cannot fit in VRAM, so they live
in host RAM and only a small VRAM working set is kept hot. Two mechanisms existed for
choosing what stays in VRAM:

| arm | mechanism | prefill | decode |
|---|---|---|---|
| 0 (golden) | full expert set pinned in host RAM (UVA) + a 44-row VRAM **mirror** rotated by a dynamic LRU | ~1.26-1.41 k tok/s | ~55-61 tok/s |
| 1 (VMM) | one contiguous CUDA-VMM tensor per layer: first 44 rows **device-backed**, rest host-backed, rows pre-permuted hot-first; no mirror, no LRU | ~1.67-1.92 k tok/s | ~28-38 tok/s |

Arm 1's prefill win comes from the contiguous layout with hot rows already in VRAM and
zero cache bookkeeping on the big-batch path; its decode loss comes from having no
mirror at all, so every token routing to a cold expert pays a PCIe read.

golden-v2.0 keeps **one** tier and serves both access paths from it: the VMM device
prefix *is* the LRU mirror, so prefill keeps arm 1's contiguous read and decode gets
golden's rotation back.

Measured on the same rig, same window, same bench shapes (`bench-3k500.py` +
`needle-probe.py`), cold boots:

| shape | arm 0 | arm 1 | **v2.0** |
|---|---|---|---|
| decode, 3k probe | 55.41 | 37.59 | **55.95** |
| decode, sustained 4k | 61.25 | 27.74 | **59.66** |
| prefill 3k | 1,258.0 | 1,670.0 | **1,727.2** |
| prefill 32k | 1,411.8 | 1,838.9 | **1,917.5** |
| prefill 190K | 1,345 | 1,731-1,775 | **1,770-1,822** |
| recall @190K | pass | pass | **pass** |

i.e. **prefill +33..37% over golden with decode at golden parity** (+49% / +115% over
arm 1, which is what the lane ran before this promotion).

## 2. The insight

The VMM device prefix (44 rows, ranking order) and the golden mirror (44 rows, ranking
order) hold **the same experts in the same order**, both built from
`configs/static_hot_cache_rankings.json`. So the prefix can *be* the mirror — no second
VRAM tier (which would not fit: a 44-row tier costs ~5 GiB/card) and no extra VRAM copy.

## 3. Why it was not free (and the flaw in the first attempt)

In golden, **every** local expert keeps a host-resident row: the mirror is a *copy*, the
original stays in the pinned UVA tensor. In the arm-1 layout the 44 hot rows existed
**only** in device memory. An LRU eviction would therefore overwrite an expert's only
copy and drop it from both maps — a silent, cumulative loss of routed experts (never a
crash, never a visible error; just a slowly wrong model).

The fix: the VMM tensor is allocated with `capacity + N` rows —

```
        VRAM per card (device-mapped prefix)         host RAM (host-mapped region)
   ┌────────────────────────────────────┐    ┌──────────────────────────────────────┐
   │ rows [0, capacity)                 │    │ rows [capacity, capacity + N)        │
   │ = the 44 ranked-hot experts        │    │ = the FULL permuted expert set; its  │
   │ = the decode mirror AND the fast   │    │   first `capacity` rows duplicate    │
   │   prefill rows                     │    │   the device prefix                  │
   └────────────────────────────────────┘    └──────────────────────────────────────┘
```

so `row(capacity + old_to_new[local])` is a permanent home for **every** local expert and
an eviction is always recoverable — exactly golden's semantics.

### Boot-time fill

Two fills in one permute-kernel launch: output row `r` is written from source row
`fill[r]`, with `fill = cat([new_to_old[:capacity], new_to_old])`. The first `capacity`
entries lay the hot prefix (device rows), the remainder lay the full permutation (host
region), so the host region *begins* with the duplicate of the prefix. A boot canary
compares the two regions byte-for-byte and refuses to serve if they differ.

## 4. How the two access paths coexist at runtime

The forward pass already forks on query size, which is what makes this cheap:

| batch | path | reads |
|---|---|---|
| **> 16 tokens (prefill)** | plain MoE over the whole tensor with the dynamic map | hot/promoted experts from device rows, everything else from host rows |
| **<= 16 tokens (decode)** | LRU: check slots, copy misses up, evict oldest, run MoE over the `capacity`-row prefix | misses copied from their host row into a prefix slot; the evicted expert's row is untouched (host rows are written once at boot) |

Both paths read **one shared map** (`rot_map`: expert -> row that currently holds its
bytes). A promotion points the promoted expert at its slot and the evicted expert back at
its host row, so a prefill step and a decode step can never disagree about where an expert
lives. Golden's static `hot_map` (expert -> slot) and `cold_map` (expert -> host row) are
kept as the LRU's own bookkeeping; `rot_map` is the fused view the plain path consumes.

## 5. Code changes (all in one module, applied as an out-of-tree overlay)

`fn-ext/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_wna16.py`
(bind-mounted over the installed package; no image rebuild). v2.0 = image module +
F1/F2 fixes + canary (sha256 `8f24042b…`).

1. `__init__`: `self._vmm_lru_hybrid` from `VLLM_WNA16_VMM_LRU_HYBRID`.
2. `maybe_init_static_hot_cache`: new branch — when VMM **and** dynamic LRU **and** the
   hybrid flag are all on, route to the hybrid init instead of returning early (the old
   hard gate `VMM and not LRU` stays for arm 1).
3. `maybe_init_mixed_vmm_hot_cache`: records the computed permutation + installed tensors
   so the hybrid init can reuse them (no duplicated logic); behaviour identical for arm 1.
4. `maybe_init_mixed_vmm_lru_hybrid` (new): builds the LRU cache over **zero-copy
   `t[:capacity]` prefix views** of the same 10 tensors golden rotates; `hot_map` =
   slot for the ranked-hot experts; `cold_map` = `capacity + old_to_new[local]` uniformly
   for every local expert; `rot_map` = clone of the installed permuted map; LRU state
   (`slot_global_ids`, ages, clock, miss buffers) exactly as golden; cache kernel built by
   the same construction golden uses (so the AutoGPTQ name re-alias path is handled);
   witness line `[hybrid] VMM+LRU cache: layer=… capacity=… tensor_rows=… device_prefix=…
   host_duplicate=…`.
5. `_allocate_mixed_vmm_tensor`: `host_duplicate` mode (rows `capacity + N`, layout above),
   plus the boot canary.
6. `_update_lru_expert_map_kernel`: takes `rot_map_ptr` and an `HYBRID: tl.constexpr`;
   under HYBRID it writes `rot_map` on promotion and restores the evicted expert's host row.
   Golden launches pass a dummy buffer with the flag off (compiled out).
7. `apply()`: plain-path call sites use `cache.rot_map` when present, else
   `layer.expert_map`; the golden static-mirror tail keeps `cache.cold_map`.
8. The expert tensors are extended to the same `capacity + N` row space, so Humming's
   `num_experts` / `num_local_experts` are rebound at bind time (established mechanism,
   already used for the legacy mirror). Reviewed as correctness-safe; tuning-bucket drift
   is perf-only and was closed empirically (prefill parity measured, see §1).

## 6. Invariants that make it correct

1. **No gather aliasing**: promotion sources are host rows (`>= capacity`), destinations are
   slots (`< capacity`) — disjoint for hot and cold experts alike.
2. **One row space** for all expert tensors (weights, scales, zero points, g_idx, sort), so a
   single `cold_map` addresses all of them.
3. **Prefix content == ranking order == `slot_global_ids`** at init (no fill needed for the
   mirror; the prefix is already correct).
4. **Every local expert has exactly one permanent host row**, written once at boot and never
   overwritten — so an eviction can never lose weights.
5. **`rot_map` starts equal to the arm-1 permutation**, so prefill before any decode step is
   identical to arm 1.
6. **Byte-level canary at boot**: map-arithmetic asserts cannot see a fill-order mistake
   (see §7); only comparing bytes can.

## 7. Failure modes met on the way (keep these in mind for any similar work)

- **v1 — no host home for the hot rows** (design flaw in the first spec): evictions silently
  deleted experts from both maps. It made the lane *faster* (less MoE work), which is why
  speed numbers from that build are worthless: run the correctness gate before believing any
  timing.
- **v2 — fill order**: the duplicate was written at the tensor's *tail* while every map
  addressed it at `capacity`, so every cold expert served a **different expert's weights**
  from token zero. Silent, plausible garbage: `434` instead of `437`, empty replies, and a
  needle probe that answered by continuing the prompt's filler text. All map-arithmetic
  asserts passed. Only a byte comparison (the canary) catches this class.
- **golden static-mirror tail**: an unrelated call site must keep golden's `cold_map` or the
  static arm double-counts hot experts. The hybrid paths never reach it, but the file must
  stay correct for that arm.

## 8. Cost model (measured)

- host RAM: +~19-22 GiB **pinned** (the duplicate; free RAM 63G -> 41G at boot). Side
  effect: it squeezes the PLE table's page cache (the 95 GiB PLE table is NVMe-mapped and
  fast because the page cache holds its hot rows).
- VRAM: +~240 MiB/card (extending the small per-expert tensors to the same row space);
  the device prefix itself is unchanged (~19.9 GiB across 4 ranks).
- boot: +~34% one-time copy volume (N + capacity rows instead of N). No per-step cost.
- rollback: set `PREFILL_VMM_ARM=0` (golden) or `1` (VMM prefill arm) in `.env` + cold boot.

## 9. Gates to run after any change to this mechanism

1. boot witnesses: `[hybrid] VMM+LRU cache` per layer-rank, `device_prefix=` and
   `host_duplicate=` byte counts, canary silent, no tracebacks.
2. `curl :8000/health` -> 200; `GPU KV cache size` banner unchanged.
3. math gate: `19*23` -> 437 (temperature 0).
4. needle recall at depth (`needle-probe.py` / `niah-probe.py`): must pass, and must still
   pass **after** heavy traffic (rotation must not degrade quality).
5. prefill: `bench-3k500.py --in 3072` and `--in 32768` (the >=30k shape is the real gate);
   decode: `--in 128 --out 4096`.
6. the boot log must report the expected arm, dtype and mount set — see the boot-witness list in the PR description.

Repo copy of the experiment record: `the project log`.
Independent review of the fix: `docs/review-hybrid-v2.md`.