"""Experimental M4-only target staging overlap. Never changes expert selection."""
import functools
from types import MethodType, SimpleNamespace

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
from .staging_copy import copy_rows

_streams = {}


def initialize(method, layer):
    cache = method._static_hot_cache
    if cache is None or not cache.dynamic_lru:
        return
    # Marlin draft and other unsupported backends retain their original path.
    hot = cache.kernel.fused_experts
    base = method.moe_kernel.fused_experts
    if not isinstance(hot, HummingIndexedExperts):
        return
    if method.num_bits != 4 or method.group_size != 128:
        raise RuntimeError('staging overlap only checked for target INT4 group128')
    if (hot is base or not 64 <= hot.num_experts <= 96 or base.num_experts != 256
            or layer.top_k != 10 or cache.hot_map.numel() != 512
            or method._static_hot_cache_max_tokens != 16):
        raise RuntimeError('unchecked target staging geometry')
    if hasattr(method, '_qwen38_staging'):
        if method._qwen38_staging.cache is not cache:
            raise RuntimeError('staging cache identity changed')
        return
    device = cache.w2_weight.device
    with torch.cuda.device(device):
        if device not in _streams:
            _streams[device] = torch.cuda.Stream(device=device)
    state = SimpleNamespace(cache=cache, stream=_streams[device], active=False,
                            forked=False, python_calls=0)
    original_gather = method._gather_lru_rows
    original_forward = hot.humming_forward
    down_metadata = tuple(getattr(cache, name, None) for name in
                          ('w2_scale', 'w2_zp', 'w2_g_idx', 'w2_sort'))

    def gather(source, output, miss_ids, miss_slots, num_ids):
        if not state.active or output is None:
            return original_gather(source, output, miss_ids, miss_slots, num_ids)
        if output is cache.w2_weight:
            if state.forked:
                raise RuntimeError('duplicate down-weight staging in one apply')
            state.stream.wait_stream(torch.cuda.current_stream(device))
            state.forked = True
            with torch.cuda.stream(state.stream):
                return copy_rows(source, output, miss_ids, miss_slots, num_ids,
                                 cta_cap=8)
        if any(output is tensor for tensor in down_metadata):
            if not state.forked:
                raise RuntimeError('down metadata staged before packed weights')
            with torch.cuda.stream(state.stream):
                return original_gather(source, output, miss_ids, miss_slots, num_ids)
        return original_gather(source, output, miss_ids, miss_slots, num_ids)

    def forward(self, sublayer_name, *args, **kwargs):
        if state.active and sublayer_name == 'w2':
            if not state.forked:
                raise RuntimeError('down GEMM reached without staging fork')
            torch.cuda.current_stream(device).wait_stream(state.stream)
        return original_forward(sublayer_name, *args, **kwargs)

    # Compile only the new copy signature, using an inert all-hit list. The
    # existing startup warmup still covers the original LRU and metadata copies.
    ids = torch.full((40,), -1, dtype=torch.int32, device=device)
    copy_rows(layer.w2_weight_packed, cache.w2_weight, ids, ids, 40, cta_cap=8)
    torch.cuda.synchronize(device)
    method._gather_lru_rows = gather
    hot.humming_forward = MethodType(forward, hot)
    method._qwen38_staging = state
    init_logger(__name__).info('M4 target staging overlap initialized: layer=%s',
                               method.layer_name)


def install(cls):
    if hasattr(cls, '_qwen38_staging_original_apply'):
        raise RuntimeError('staging overlap installed twice')
    original_init = cls.maybe_init_static_hot_cache
    original_apply = cls.apply
    cls._qwen38_staging_original_apply = original_apply

    @functools.wraps(original_init)
    def init(self, layer):
        result = original_init(self, layer)
        initialize(self, layer)
        return result

    @functools.wraps(original_apply)
    def apply(self, layer, x, topk_weights, topk_ids, shared_experts,
              shared_experts_input):
        state = getattr(self, '_qwen38_staging', None)
        if state is None or x.shape[0] != 4 or topk_ids.shape != (4, 10):
            return original_apply(self, layer, x, topk_weights, topk_ids,
                                  shared_experts, shared_experts_input)
        if state.active or state.cache is not self._static_hot_cache:
            raise RuntimeError('reentrant or replaced staging cache')
        state.active, state.forked = True, False
        state.python_calls += 1  # capture/eager diagnostic only, not replay count
        try:
            return original_apply(self, layer, x, topk_weights, topk_ids,
                                  shared_experts, shared_experts_input)
        finally:
            # Close the capture fork and protect all shared slot/miss buffers
            # before any next layer or request can reuse them.
            if state.forked:
                torch.cuda.current_stream(state.cache.w2_weight.device).wait_stream(state.stream)
            state.active, state.forked = False, False

    cls.maybe_init_static_hot_cache = init
    cls.apply = apply
