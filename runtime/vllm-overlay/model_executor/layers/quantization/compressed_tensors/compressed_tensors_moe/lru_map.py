# SPDX-License-Identifier: Apache-2.0
"""Experimental integer-only LRU update. Preserve the released victim policy."""
import triton
import triton.language as tl
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import _update_lru_expert_map_kernel as _serial


@triton.jit
def update_kernel(global_ids_ptr, source_map_ptr, cache_map_ptr,
                  slot_global_ids_ptr, slot_ages_ptr, clock_ptr,
                  miss_local_ids_ptr, miss_slots_ptr,
                  num_ids: tl.constexpr, global_num_experts: tl.constexpr,
                  capacity: tl.constexpr, id_block: tl.constexpr,
                  capacity_block: tl.constexpr):
    r = tl.arange(0, id_block)
    s = tl.arange(0, capacity_block)
    valid_r = r < num_ids
    valid_s = s < capacity
    requested = tl.load(global_ids_ptr + r, valid_r, -1)
    old_ids = tl.load(slot_global_ids_ptr + s, valid_s, -1)
    ages = tl.load(slot_ages_ptr + s, valid_s, 0)
    clock = tl.load(clock_ptr)
    global_ok = valid_r & (requested >= 0) & (requested < global_num_experts)
    safe_ids = tl.minimum(tl.maximum(requested, 0), global_num_experts - 1)
    local = tl.load(source_map_ptr + safe_ids, global_ok, -1)
    is_local = global_ok & (local >= 0)
    initial_slots = tl.load(cache_map_ptr + safe_ids, is_local, -1)
    earlier = (requested[:, None] == requested[None, :]) & (r[None, :] < r[:, None])
    first = tl.sum(earlier.to(tl.int32), axis=1) == 0
    misses = is_local & (initial_slots < 0) & first
    matches = (old_ids[:, None] == requested[None, :]) & valid_r[None, :]
    protected = ~valid_s | (tl.sum(matches.to(tl.int32), axis=1) > 0)
    num_misses = tl.sum(misses.to(tl.int32), axis=0)
    available = tl.sum((~protected).to(tl.int32), axis=0)

    # The original argmin uses INT_MAX for protected slots. Preserve its rare
    # tie/over-capacity behavior, including empty-cache tests and clock wrap.
    fallback = (num_misses > available) | (tl.sum((valid_s & (ages == 0x7fffffff)).to(tl.int32), axis=0) > 0)
    if fallback:
        _serial(global_ids_ptr, source_map_ptr, cache_map_ptr,
                slot_global_ids_ptr, slot_ages_ptr, clock_ptr,
                miss_local_ids_ptr, miss_slots_ptr, num_ids,
                global_num_experts, capacity, id_block, capacity_block)
    else:
        # Signed age first, then slot index: tl.argmin's leftmost tie rule.
        keys = (ages.to(tl.int64) + 2147483648) * capacity_block + s
        keys = tl.where(protected, 0x7fffffffffffffff, keys)
        ordered = tl.sort(keys, descending=False)
        rank = tl.cumsum(misses.to(tl.int32), axis=0) - 1
        victims = (tl.gather(ordered, tl.maximum(rank, 0), axis=0) % capacity_block).to(tl.int32)
        assigned = misses[None, :] & (s[:, None] == victims[None, :])
        has_new = tl.sum(assigned.to(tl.int32), axis=1) > 0
        new_ids = tl.sum(tl.where(assigned, requested[None, :], 0), axis=1)
        final_ids = tl.where(has_new, new_ids, old_ids)
        used = (final_ids[:, None] == requested[None, :]) & is_local[None, :]
        last = tl.max(tl.where(used, r[None, :], -1), axis=1)
        final_ages = tl.where(last >= 0, clock + last + 1, ages)
        tl.store(cache_map_ptr + old_ids, -1, valid_s & has_new & (old_ids >= 0))
        tl.store(cache_map_ptr + safe_ids, victims, misses)
        tl.store(slot_global_ids_ptr + s, final_ids, valid_s)
        tl.store(slot_ages_ptr + s, final_ages, valid_s)
        tl.store(miss_local_ids_ptr + r, tl.where(misses, local, -1), valid_r)
        tl.store(miss_slots_ptr + r, tl.where(misses, victims, -1), valid_r)
        tl.store(clock_ptr, clock + num_ids)
