# SPDX-License-Identifier: Apache-2.0
"""Experimental startup-only compilation of all supported hot84 request sizes."""
import functools
import time
import torch
import triton
from vllm.logger import init_logger

if __package__:
    from .lru_map import update_kernel
else:
    from lru_map import update_kernel

_warmed = set()


def warmup_once(method, layer):
    cache = method._static_hot_cache
    if cache is None or not cache.dynamic_lru:
        return
    if cache.hot_map.numel() != 512 or not 64 <= cache.slot_global_ids.numel() <= 96:
        raise RuntimeError('unchecked LRU warmup geometry')
    if method._static_hot_cache_max_tokens != 16:
        raise RuntimeError('unchecked LRU token threshold')
    top_k = layer.top_k
    if top_k != 10:
        raise RuntimeError(f'expected checkpoint top-k10, got {top_k}')
    device = cache.hot_map.device
    pairs = ((layer.w13_weight_packed, cache.w13_weight),
             (layer.w2_weight_packed, cache.w2_weight),
             (layer.w13_weight_scale, cache.w13_scale),
             (layer.w2_weight_scale, cache.w2_scale))
    signature = (str(device), cache.slot_global_ids.numel(), top_k, tuple((tuple(s.shape[1:]), s.dtype) for s, _ in pairs))
    if signature in _warmed:
        return
    names = ('hot_map', 'slot_global_ids', 'slot_ages', 'clock')
    originals = [getattr(cache, name) for name in names]
    if any(t.dtype != torch.int32 for t in (*originals, cache.cold_map)):
        raise RuntimeError('unchecked LRU metadata dtype')
    snapshots = [t.clone() for t in originals]
    started = time.monotonic()
    # Read the real routed layer: Qwen3.8 Flash Next selects ten experts/token.
    # A three-token boundary uses30 IDs, not24. Above84 IDs the cache falls back.
    counts = tuple(range(top_k, cache.slot_global_ids.numel()+1, top_k))
    for count in counts:
        # Never warm up on live replacement state. Nonlocal IDs also make all
        # byte-copy launches no-ops, with the actual weight/scale pointer types.
        ids = torch.full((count,), -1, dtype=torch.int32, device=device)
        state = [t.clone() for t in snapshots]
        miss_ids, miss_slots = torch.empty_like(ids), torch.empty_like(ids)
        update_kernel[(1,)](ids, cache.cold_map, *state, miss_ids, miss_slots,
            num_ids=count, global_num_experts=512, capacity=cache.slot_global_ids.numel(),
            id_block=triton.next_power_of_2(count), capacity_block=128, num_warps=4)
        for source, output in pairs:
            method._gather_lru_rows(source, output, miss_ids, miss_slots, count)
    torch.cuda.synchronize(device)
    if any(not torch.equal(a, b) for a, b in zip(originals, snapshots)):
        raise RuntimeError('LRU warmup changed the real cache state')
    _warmed.add(signature)
    init_logger(__name__).info('LRU startup warmup: device=%s counts=%s seconds=%.3f',
                              device, counts, time.monotonic()-started)


def install(cls):
    original = cls.maybe_init_static_hot_cache

    @functools.wraps(original)
    def initialize(self, layer):
        result = original(self, layer)
        warmup_once(self, layer)
        return result

    cls.maybe_init_static_hot_cache = initialize
