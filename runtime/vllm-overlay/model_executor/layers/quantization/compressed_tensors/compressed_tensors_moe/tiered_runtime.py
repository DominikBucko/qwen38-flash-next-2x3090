# SPDX-License-Identifier: Apache-2.0
"""Experimental layer-by-layer integration. Not installed in the release."""
import copy
from dataclasses import replace
from types import SimpleNamespace
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.experts.fused_humming_moe import HummingIndexedExperts
from .tiered_vmm import allocate_tiered


def initialize(method,layer):
    cache = method._static_hot_cache
    if cache is None or not cache.dynamic_lru:
        return
    base = method.moe_kernel.fused_experts
    hot = cache.kernel.fused_experts
    if not isinstance(base,HummingIndexedExperts) or not isinstance(hot,HummingIndexedExperts):
        raise RuntimeError('tiered prefill requires the checked Humming indexed path')
    if hasattr(base,'_qwen38_tiered_prefill'):
        raise RuntimeError('tiered prefill already initialized')
    capacity = cache.slot_global_ids.numel()
    local_count = base.num_experts
    local_ids = layer.expert_map[cache.slot_global_ids.long()].long()
    if local_count!=256 or not 64<=capacity<=96 or local_ids.unique().numel()!=capacity:
        raise RuntimeError('unchecked tiered expert geometry')
    for name in ('w1_zp','w2_zp','w1_bias','w2_bias','g1_alphas','g2_alphas'):
        if getattr(base.quant_config,name) is not None:
            raise RuntimeError(f'unhandled tiered quantization field {name}')
    from vllm.model_executor.offloader.exact_pinned import extension
    pinned_before = extension().live_bytes()
    # Driver VMM allocations cannot reuse idle blocks in PyTorch's allocator.
    # Release only unused cached blocks at this load-time boundary; all live
    # model/cache tensors stay allocated. This is never called during inference.
    torch.cuda.empty_cache()
    combined_weights,allocations,combined_scales = [],[],[]
    for parameter_name,cache_name in (('w13_weight_packed','w13_weight'),('w2_weight_packed','w2_weight')):
        parameter = getattr(layer,parameter_name)
        cached = getattr(cache,cache_name)
        combined,allocation = allocate_tiered(parameter,local_ids)
        if not torch.equal(combined[:capacity],cached) or not torch.equal(combined[capacity:],parameter):
            raise RuntimeError(f'tiered packed-byte mismatch: {parameter_name}')
        # Do not keep the original60GB host backing in addition to the new pool.
        # Rebind the existing Parameter object so aliases see the source suffix.
        parameter.data = combined[capacity:]
        setattr(cache,cache_name,combined[:capacity])
        combined_weights.append(combined)
        allocations.append(allocation)
    for parameter_name,cache_name in (('w13_weight_scale','w13_scale'),('w2_weight_scale','w2_scale')):
        parameter = getattr(layer,parameter_name)
        cached = getattr(cache,cache_name)
        combined = torch.cat((cached,parameter),dim=0).contiguous()
        if not torch.equal(combined[:capacity],cached) or not torch.equal(combined[capacity:],parameter):
            raise RuntimeError(f'tiered scale-byte mismatch: {parameter_name}')
        parameter.data = combined[capacity:]
        setattr(cache,cache_name,combined[:capacity])
        combined_scales.append(combined)
    base.quant_config = replace(base.quant_config,
        _w1=replace(base.quant_config._w1,scale=layer.w13_weight_scale),
        _w2=replace(base.quant_config._w2,scale=layer.w2_weight_scale))
    method.moe_quant_config = base.quant_config
    hot.quant_config = replace(hot.quant_config,
        _w1=replace(hot.quant_config._w1,scale=cache.w13_scale),
        _w2=replace(hot.quant_config._w2,scale=cache.w2_scale))
    # Only humming_forward uses this view. Keep the original owner's tuning,
    # metadata and prepare/finalize path; do not construct a second MoE workspace.
    tiered = copy.copy(base)
    tiered.humming_configs = {name:replace(config,num_experts=local_count+capacity)
                              for name,config in base.humming_configs.items()}
    tiered.quant_config = replace(base.quant_config,
        _w1=replace(base.quant_config._w1,scale=combined_scales[0]),
        _w2=replace(base.quant_config._w2,scale=combined_scales[1]))
    base._qwen38_tiered_prefill = SimpleNamespace(
        experts=tiered,cache=cache,w13=combined_weights[0],w2=combined_weights[1],
        scales=combined_scales,allocations=allocations,
        max_decode_tokens=method._static_hot_cache_max_tokens)
    torch.cuda.synchronize(layer.w13_weight_packed.device)
    pinned_after = extension().live_bytes()
    expected_release = sum(w[capacity:].numel()*w.element_size() for w in combined_weights)
    if pinned_before-pinned_after!=expected_release:
        raise RuntimeError(f'old tiered source backing retained: before={pinned_before},after={pinned_after},expected_release={expected_release}')
    get_accum = sum(a.gpu_bytes for a in allocations)
    logical_hot = sum(w[:capacity].numel()*w.element_size() for w in combined_weights)
    init_logger(__name__).info(
        'Tiered prefill: layer=%s source_bytes_released=%d gpu_weight_bytes=%d gpu_page_rounding_bytes=%d host_weight_bytes=%d',
        method.layer_name,expected_release,get_accum,get_accum-logical_hot,
        sum(a.host_bytes for a in allocations))
    import os as _os
    if _os.environ.get('QWEN38_STREAM_STAGE', '0') == '1':
        from .stream_stage import register as _stream_register
        _stream_register(method, layer, base, base._qwen38_tiered_prefill)
