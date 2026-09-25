# SPDX-License-Identifier: Apache-2.0
"""Unvalidated VMM layout: dynamic GPU hot prefix + immutable host source pool.

Use only in an isolated GPU test. This creates owned CUDA mappings; keep the
returned allocation alive until all tensor views and captured graphs are gone.
The allocation lasts for the CUDA context, like the released static VMM helper.
"""
import math
import torch
import triton
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
    CompressedTensorsWNA16MoEMethod, _MixedVMMAllocation,
    _MIXED_VMM_ALLOCATIONS, _permute_expert_rows_kernel,
)
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum


def allocate_tiered(source, hot_local_ids):
    """Copy exact packed bytes; map only the hot prefix (rounded to pages) on GPU."""
    from cuda.bindings import driver
    check = CompressedTensorsWNA16MoEMethod._cuda_driver_check
    assert source.is_cuda and source.is_contiguous() and source.ndim>0
    device_index = source.device.index
    check(driver.cuInit(0),'cuInit')
    device = check(driver.cuDeviceGet(device_index),'cuDeviceGet')
    numa = check(driver.cuDeviceGetAttribute(
        driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,device),'host NUMA')

    def prop(kind,index):
        value = driver.CUmemAllocationProp()
        value.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        value.location.type = kind
        value.location.id = index
        value.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
        return value

    gpu_prop = prop(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,device_index)
    host_prop = prop(driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,numa)
    minimum = driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    alignment = math.lcm(check(driver.cuMemGetAllocationGranularity(gpu_prop,minimum),'GPU granularity'),
                         check(driver.cuMemGetAllocationGranularity(host_prop,minimum),'host granularity'))
    capacity = hot_local_ids.numel()
    rows = source.shape[0]+capacity
    shape = (rows,*source.shape[1:])
    row_bytes = source[0].numel()*source.element_size()
    mapped_bytes = triton.cdiv(rows*row_bytes,alignment)*alignment
    gpu_bytes = triton.cdiv(capacity*row_bytes,alignment)*alignment
    host_bytes = mapped_bytes-gpu_bytes
    address = check(driver.cuMemAddressReserve(mapped_bytes,alignment,0,0),'reserve')
    handles = []
    mapped = []
    try:
        for offset,size,properties in ((0,gpu_bytes,gpu_prop),(gpu_bytes,host_bytes,host_prop)):
            if size:
                handle = check(driver.cuMemCreate(size,properties,0),'allocate tier')
                handles.append(handle)
                check(driver.cuMemMap(int(address)+offset,size,0,handle,0),'map tier')
                mapped.append((int(address)+offset,size))
        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        check(driver.cuMemSetAccess(address,mapped_bytes,[access],1),'set access')
        storage = torch._C._construct_storage_from_data_pointer(int(address),source.device,mapped_bytes)
        metadata = dict(nbytes=mapped_bytes,data_ptr=int(address),size=shape,
                        stride=tuple(source.stride()),dtype=source.dtype,
                        device=source.device,storage_offset=0)
        output = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata,storage)
        permutation = torch.cat((hot_local_ids.to(torch.int64),
                                 torch.arange(source.shape[0],device=source.device,dtype=torch.int64)))
        _permute_expert_rows_kernel[(rows,triton.cdiv(source[0].numel(),1024))](
            source,output,permutation,row_size=source[0].numel(),block_size=1024,num_warps=8)
        torch.cuda.synchronize(source.device)
    except Exception:
        # Only owned mappings/handles are touched. A failed CUDA context may
        # reject cleanup; preserve the original error and let process exit reap it.
        for ptr,size in reversed(mapped):
            driver.cuMemUnmap(ptr,size)
        for handle in handles:
            driver.cuMemRelease(handle)
        driver.cuMemAddressFree(address,mapped_bytes)
        raise
    owner = _MixedVMMAllocation(storage=storage,address=address,handles=tuple(handles),
                               mapped_bytes=mapped_bytes,gpu_bytes=gpu_bytes,host_bytes=host_bytes)
    _MIXED_VMM_ALLOCATIONS.append(owner)
    return output,owner


def apply_tiered(owner, tiered, cache, *, output, hidden_states, w1, w2,
                 topk_weights, topk_ids, activation, global_num_experts,
                 expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                 expert_tokens_meta, apply_router_weight_on_input):
    """One GEMM per stage, same global block schedule and final reduction."""
    assert not apply_router_weight_on_input
    assert a1q_scale is None and a2_scale is None
    assert cache.dynamic_lru and expert_map is not None
    hidden_states = hidden_states.view(-1,hidden_states.size(-1))
    buffers = owner.prepare_buffers(workspace13,workspace2,topk_ids.size(0),topk_ids.size(1),activation)
    kwargs1,kwargs2 = owner.prepare_humming_moe_kwargs(topk_ids,expert_map,expert_tokens_meta)
    capacity = cache.slot_global_ids.numel()
    local_to_tiered = torch.arange(owner.num_experts,device=topk_ids.device,dtype=torch.int32)+capacity
    hot_locals = expert_map[cache.slot_global_ids.long()].long()
    local_to_tiered.scatter_(0,hot_locals,torch.arange(capacity,device=topk_ids.device,dtype=torch.int32))
    original_ids = kwargs1['expert_ids']
    # Unused tail metadata is uninitialized. Only remap valid local IDs.
    mapped_ids = local_to_tiered[original_ids.clamp(0,owner.num_experts-1).long()]
    kwargs1 = dict(kwargs1,expert_ids=mapped_ids)
    kwargs2 = dict(kwargs2,expert_ids=mapped_ids)
    inputs,input_scale = owner.quantize_input('w13',hidden_states,None)
    tiered.humming_forward('w13',inputs=inputs,weight=w1,input_scale=input_scale,
                           outputs=buffers['gate_up_output'],**kwargs1)
    owner.apply_activation(activation=activation,input=buffers['gate_up_output'],output=buffers['activation_output'])
    inputs,input_scale = owner.quantize_input('w2',buffers['activation_output'],None)
    tiered.humming_forward('w2',inputs=inputs,weight=w2,input_scale=input_scale,
                           outputs=buffers['down_output'].view(-1,hidden_states.size(-1)),**kwargs2)
    moe_fused_mul_sum(inputs=buffers['down_output'].view(*topk_ids.shape,-1),
                      topk_weights=topk_weights,topk_ids=topk_ids,expert_map=expert_map,outputs=output)
