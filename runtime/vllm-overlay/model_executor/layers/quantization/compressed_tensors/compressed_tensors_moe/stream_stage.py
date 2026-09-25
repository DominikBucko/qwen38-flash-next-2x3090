# SPDX-License-Identifier: Apache-2.0
"""Experimental streamed expert staging for large target prefill chunks.

Large prefill chunks touch nearly every routed expert. The tiered path reads
cold experts straight from host memory inside the GEMM, so PCIe sits idle
during attention/GDN/dense work and the GEMM stalls on host reads.

This path instead DMA-copies a layer's complete local source pool (the exact
immutable host rows, unchanged bytes) into one of two GPU staging buffers on a
side stream, one or two MoE layers ahead, and runs the original non-cached
Humming schedule (original expert map, original source scales) on the staged
copy. Routing, weights, scales, and reduction are those of the original
all-source path; only the physical location of the weight bytes differs.

The LRU hot cache is neither read nor modified here.
"""
import os
import re
from types import SimpleNamespace

import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.moe_fused_mul_sum import moe_fused_mul_sum

logger = init_logger(__name__)

MIN_TOKENS = int(os.environ.get('QWEN38_STREAM_STAGE_MIN_TOKENS', '3000'))
PREFETCH_NEXT_STEP = os.environ.get('QWEN38_STREAM_STAGE_PREFETCH_NEXT', '1') == '1'
# Runtime A/B switch: if this file exists (host-mounted dir), use the original
# tiered path. Checked once per forward step at the first MoE layer.
OFF_FLAG = os.environ.get('QWEN38_STREAM_STAGE_OFF_FLAG', '/tmp/qwen38-profile/stream-stage-off')
# Borrow the GPU hot-cache pages of the last layers as staging slots instead of
# allocating dedicated slots. Their hot rows are restored from the immutable
# source before any step that can read the hot cache.
BORROW = os.environ.get('QWEN38_STREAM_STAGE_BORROW', '1') == '1'
BORROW_PER_SLOT = int(os.environ.get('QWEN38_STREAM_STAGE_BORROW_PER_SLOT', '4'))

_devices = {}   # device -> SimpleNamespace(stream, slots, layers, ...)
_runner_hooked = False


def mark_dirty():
    """A step that can update the LRU (<=16 tokens) was scheduled."""
    for state in _devices.values():
        state.dirty = True


def _make_wrapper(original):
    import functools
    import time
    step_log = os.environ.get('QWEN38_STEP_LOG')
    log_file = open(f'{step_log}.{os.getpid()}', 'a', buffering=1) if step_log else None

    @functools.wraps(original)
    def execute_model(self, scheduler_output, *args, **kwargs):
        tokens = getattr(scheduler_output, 'total_num_scheduled_tokens', 0) or 0
        t0 = time.perf_counter()
        try:
            if 0 < tokens < MIN_TOKENS:
                restore_borrowed()
            if 0 < tokens <= 16:
                mark_dirty()
            return original(self, scheduler_output, *args, **kwargs)
        finally:
            if log_file is not None:
                log_file.write(f'{t0:.6f} {time.perf_counter() - t0:.6f} {tokens}\n')

    return execute_model


def _hook_runner():
    # The hot set changes only in target MoE calls with <=16 tokens, which run
    # inside replayed CUDA graphs. execute_model runs in Python for every step,
    # in stream order, so it can flag those steps without a device sync.
    global _runner_hooked
    if _runner_hooked:
        return
    import functools
    import importlib
    for name in ('vllm.v1.worker.gpu.model_runner', 'vllm.v1.worker.gpu_model_runner'):
        try:
            cls = importlib.import_module(name).GPUModelRunner
        except Exception:
            continue
        if getattr(cls.execute_model, '_qwen38_stream_stage', False):
            continue
        original = cls.execute_model

        execute_model = _make_wrapper(original)
        execute_model._qwen38_stream_stage = True
        cls.execute_model = execute_model
        logger.info('Stream staging: hooked %s.GPUModelRunner.execute_model', name)
    _runner_hooked = True
_by_owner = {}  # id(owner experts) -> (device state, position)


def _layer_index(name):
    match = re.search(r'layers\.(\d+)\.', name or '')
    if match is None:
        raise RuntimeError(f'cannot parse layer index from {name!r}')
    return int(match.group(1))


def register(method, layer, base, tiered_ns):
    """Called after tiered initialization of one target MoE layer."""
    capacity = tiered_ns.cache.slot_global_ids.numel()
    src13 = tiered_ns.w13[capacity:]
    src2 = tiered_ns.w2[capacity:]
    if not (src13.is_contiguous() and src2.is_contiguous()):
        raise RuntimeError('stream staging requires contiguous source pools')
    if src13.data_ptr() != layer.w13_weight_packed.data_ptr() or src2.data_ptr() != layer.w2_weight_packed.data_ptr():
        raise RuntimeError('stream staging source is not the layer source pool')
    device = src13.device
    state = _devices.get(device)
    if state is None:
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
        _hook_runner()
        state = SimpleNamespace(device=device, stream=stream, entries={}, order=None, dirty=True,
                                hot_clobbered=False, restores=0,
                                slots=None, loaded=[None, None], copy_done=[None, None],
                                gemm_done=[None, None], staged_calls=0, copies=0,
                                shapes=(tuple(src13.shape), src13.dtype, tuple(src2.shape), src2.dtype))
        _devices[device] = state
    if state.shapes != (tuple(src13.shape), src13.dtype, tuple(src2.shape), src2.dtype):
        raise RuntimeError('stream staging requires identical layer geometry')
    index = _layer_index(method.layer_name)
    if index in state.entries:
        raise RuntimeError(f'duplicate stream-staging layer {index}')
    cache = tiered_ns.cache
    local_of_global = layer.expert_map.detach().to('cpu').tolist()
    state.entries[index] = SimpleNamespace(index=index, owner=base, src13=src13, src2=src2,
                                           cache=cache, local_of_global=local_of_global,
                                           hot13=cache.w13_weight, hot2=cache.w2_weight,
                                           hot_locals=None, runs=None, borrowed=False,
                                           allocations=tuple(tiered_ns.allocations))
    state.order = None
    if state.slots is None and not BORROW:
        # Allocate both slots now so a startup OOM is visible before READY.
        state.slots = [(torch.empty_like(src13, device=device), torch.empty_like(src2, device=device))
                       for _ in range(2)]
        for i in range(2):
            state.copy_done[i] = torch.cuda.Event()
            state.gemm_done[i] = torch.cuda.Event()
        slot_bytes = src13.numel() * src13.element_size() + src2.numel() * src2.element_size()
        logger.info('Stream staging: device=%s slot_bytes=%d total_bytes=%d min_tokens=%d',
                    device, slot_bytes, 2 * slot_bytes, MIN_TOKENS)
    _by_owner[id(base)] = state


def _positions(state):
    if state.order is None:
        state.order = sorted(state.entries)
        for pos, index in enumerate(state.order):
            state.entries[index].pos = pos
            _by_owner[id(state.entries[index].owner)] = state
    return state.order


def _map_borrowed(handles, sizes, device, shape, dtype):
    """Map existing GPU physical handles back-to-back into a new VA range."""
    from cuda.bindings import driver
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_wna16 import (
        CompressedTensorsWNA16MoEMethod)
    check = CompressedTensorsWNA16MoEMethod._cuda_driver_check
    total = sum(sizes)
    need = 1
    for d in shape:
        need *= d
    need *= torch.empty((), dtype=dtype).element_size()
    if total < need:
        raise RuntimeError(f'borrowed pages too small: {total} < {need}')
    address = check(driver.cuMemAddressReserve(total, 2 * 1024 * 1024, 0, 0), 'reserve borrow VA')
    offset = 0
    for handle, size in zip(handles, sizes):
        check(driver.cuMemMap(int(address) + offset, size, 0, handle, 0), 'map borrowed')
        offset += size
    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device.index
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    check(driver.cuMemSetAccess(address, total, [access], 1), 'borrow access')
    storage = torch._C._construct_storage_from_data_pointer(int(address), device, total)
    stride = []
    acc = 1
    for d in reversed(shape):
        stride.insert(0, acc)
        acc *= d
    meta = dict(nbytes=total, data_ptr=int(address), size=tuple(shape), stride=tuple(stride),
                dtype=dtype, device=device, storage_offset=0)
    tensor = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(meta, storage)
    return tensor, (storage, address, total)


def _build_slots(state):
    order = _positions(state)
    if state.slots is not None:
        return
    count = 2 * BORROW_PER_SLOT
    if len(order) < count + 2:
        raise RuntimeError('not enough layers to borrow staging pages')
    borrowed = [state.entries[i] for i in order[-count:]]
    slots, keep = [], []
    for s in range(2):
        group = borrowed[s * BORROW_PER_SLOT:(s + 1) * BORROW_PER_SLOT]
        views = []
        for which, src_name in ((0, 'src13'), (1, 'src2')):
            handles = [e.allocations[which].handles[0] for e in group]
            sizes = [e.allocations[which].gpu_bytes for e in group]
            src = getattr(group[0], src_name)
            view, owner = _map_borrowed(handles, sizes, state.device, src.shape, src.dtype)
            views.append(view)
            keep.append(owner)
        slots.append(tuple(views))
    for e in borrowed:
        e.borrowed = True
    state.slots = slots
    state.borrow_owners = keep
    for i in range(2):
        state.copy_done[i] = torch.cuda.Event()
        state.gemm_done[i] = torch.cuda.Event()
    state.borrowed_entries = borrowed
    logger.info('Stream staging: borrowed hot pages of layers %s as two slots on %s',
                [e.index for e in borrowed], state.device)


def restore_borrowed():
    """Rewrite borrowed layers' hot rows from the immutable source pool."""
    for state in _devices.values():
        if not state.hot_clobbered:
            continue
        entries = state.borrowed_entries
        ids = torch.stack([e.cache.slot_global_ids for e in entries]).to('cpu').tolist()
        stream = state.stream
        stream.wait_stream(torch.cuda.current_stream(state.device))
        with torch.cuda.stream(stream):
            for e, row in zip(entries, ids):
                for slot, g in enumerate(row):
                    local = e.local_of_global[g]
                    if local < 0:
                        raise RuntimeError(f'non-local hot expert in layer {e.index}')
                    e.hot13[slot].copy_(e.src13[local], non_blocking=True)
                    e.hot2[slot].copy_(e.src2[local], non_blocking=True)
            done = torch.cuda.Event()
            done.record(stream)
        torch.cuda.current_stream(state.device).wait_event(done)
        if state.restores == 0:
            # One-time exact check of every restored row against its source row.
            for e, row in zip(entries, ids):
                for slot, g in enumerate(row):
                    local = e.local_of_global[g]
                    if not (torch.equal(e.hot13[slot], e.src13[local])
                            and torch.equal(e.hot2[slot], e.src2[local])):
                        raise RuntimeError(f'borrowed restore mismatch layer {e.index} slot {slot}')
            logger.info('Stream staging: restore verified for %d layers on %s', len(entries), state.device)
        state.loaded = [None, None]
        state.hot_clobbered = False
        state.restores += 1
        if state.restores <= 3:
            logger.info('Stream staging: restored %d borrowed layers on %s', len(entries), state.device)


def _snapshot(state):
    """Read every layer's current hot set (one sync per staged forward step).

    The LRU only changes during decode steps, which may replay CUDA graphs
    without running Python, so the hot set cannot be tracked on the host.
    """
    order = _positions(state)
    ids = torch.stack([state.entries[i].cache.slot_global_ids for i in order]).to('cpu').tolist()
    state.snapshot_done = torch.cuda.Event()
    state.snapshot_done.record(torch.cuda.current_stream(state.device))
    changed = []
    for pos, (index, row) in enumerate(zip(order, ids)):
        entry = state.entries[index]
        locals_ = tuple(entry.local_of_global[g] for g in row)
        if entry.hot_locals != locals_:
            if min(locals_) < 0 or len(set(locals_)) != len(locals_):
                raise RuntimeError(f'invalid hot set for layer {index}')
            hot = set(locals_)
            runs, start = [], None
            for local in range(entry.src13.shape[0] + 1):
                cold = local < entry.src13.shape[0] and local not in hot
                if cold and start is None:
                    start = local
                elif not cold and start is not None:
                    runs.append((start, local))
                    start = None
            entry.hot_locals = locals_
            entry.hot_index = torch.tensor(locals_, dtype=torch.long, device=state.device)
            entry.runs = runs
            changed.append(pos)
    for slot, pos in enumerate(state.loaded):
        if pos is not None and pos in changed:
            state.loaded[slot] = None  # stale hot rows: re-stage


def _issue(state, pos):
    """Stage the layer at `pos` into slot pos % 2: cold rows by DMA, hot rows on-GPU."""
    order = _positions(state)
    slot = pos % 2
    if state.loaded[slot] == pos:
        return
    entry = state.entries[order[pos]]
    dst13, dst2 = state.slots[slot]
    stream = state.stream
    # The previous occupant's GEMM must have finished reading the slot, and the
    # hot rows must not be read before earlier compute-stream work that could
    # have updated them (LRU refills) has finished.
    stream.wait_event(state.gemm_done[slot])
    stream.wait_event(state.snapshot_done)
    state.hot_clobbered = state.hot_clobbered or BORROW
    with torch.cuda.stream(stream):
        if entry.borrowed:
            # Its own hot rows live in the borrowed slots: never read them here.
            dst13.copy_(entry.src13, non_blocking=True)
            dst2.copy_(entry.src2, non_blocking=True)
        else:
            for a, b in entry.runs:
                dst13[a:b].copy_(entry.src13[a:b], non_blocking=True)
                dst2[a:b].copy_(entry.src2[a:b], non_blocking=True)
            entry.hot_index.record_stream(stream)
            dst13.index_copy_(0, entry.hot_index, entry.hot13)
            dst2.index_copy_(0, entry.hot_index, entry.hot2)
        state.copy_done[slot].record(stream)
    state.loaded[slot] = pos
    state.copies += 1


def verify(state, pos):
    """Exact byte check of one staged layer against its immutable source pool."""
    _snapshot(state)
    state.loaded = [None, None]
    _issue(state, pos)
    torch.cuda.current_stream(state.device).wait_event(state.copy_done[pos % 2])
    entry = state.entries[_positions(state)[pos]]
    dst13, dst2 = state.slots[pos % 2]
    ok = torch.equal(dst13, entry.src13) and torch.equal(dst2, entry.src2)
    state.loaded = [None, None]
    if not ok:
        raise RuntimeError(f'stream staging byte mismatch at position {pos}')
    logger.info('Stream staging verified: device=%s layer=%d runs=%d hot=%d',
                state.device, entry.index, len(entry.runs), len(entry.hot_locals))


def maybe_apply(owner, *, output, hidden_states, topk_weights, topk_ids, activation,
                expert_map, a1q_scale, a2_scale, workspace13, workspace2,
                expert_tokens_meta, apply_router_weight_on_input):
    """Return True if the staged path handled this call."""
    state = _by_owner.get(id(owner))
    if state is None or hidden_states.shape[0] < MIN_TOKENS:
        return False
    assert not apply_router_weight_on_input
    assert a1q_scale is None and a2_scale is None
    order = _positions(state)
    entry = state.by_owner_id[id(owner)] if getattr(state, 'by_owner_id', None) else None
    if entry is None:
        state.by_owner_id = {id(e.owner): e for e in state.entries.values()}
        entry = state.by_owner_id[id(owner)]
    pos = entry.pos
    if pos == 0:
        state.disabled = os.path.exists(OFF_FLAG)
        if not state.disabled:
            _build_slots(state)
            if not getattr(state, 'verified', False):
                verify(state, 0)
                verify(state, len(order) - 1)
                state.verified = True
            if state.dirty:
                _snapshot(state)
                state.dirty = False
                state.snapshots = getattr(state, 'snapshots', 0) + 1
    if getattr(state, 'disabled', False) or getattr(state, 'snapshot_done', None) is None:
        return False
    slot = pos % 2
    compute = torch.cuda.current_stream(state.device)
    # Make sure this layer is (being) copied; issue the next one behind it so
    # the copy engine stays busy while this layer and the following
    # attention/GDN work run.
    _issue(state, pos)
    if pos == 0 and len(order) > 1:
        _issue(state, 1)
    compute.wait_event(state.copy_done[slot])
    w1, w2 = state.slots[slot]

    hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    buffers = owner.prepare_buffers(workspace13, workspace2, topk_ids.size(0), topk_ids.size(1), activation)
    buffers['output'] = output
    kwargs1, kwargs2 = owner.prepare_humming_moe_kwargs(topk_ids=topk_ids, expert_map=expert_map,
                                                        expert_tokens_meta=expert_tokens_meta)
    inputs, input_scale = owner.quantize_input('w13', inputs=hidden_states,
                                               quanted_input=buffers.get('quanted_gate_up_input', None))
    owner.humming_forward('w13', inputs=inputs, weight=w1, input_scale=input_scale,
                          outputs=buffers['gate_up_output'], **kwargs1)
    owner.apply_activation(activation=activation, input=buffers['gate_up_output'],
                           output=buffers['activation_output'])
    inputs, input_scale = owner.quantize_input('w2', inputs=buffers['activation_output'],
                                               quanted_input=buffers.get('quanted_down_input', None))
    owner.humming_forward('w2', inputs=inputs, weight=w2, input_scale=input_scale,
                          outputs=buffers['down_output'].view(-1, hidden_states.size(-1)), **kwargs2)
    # Last reader of the staged weights: release the slot for the copy stream.
    state.gemm_done[slot].record(compute)
    nxt = pos + 2
    if nxt < len(order):
        _issue(state, nxt)
    elif PREFETCH_NEXT_STEP:
        # Speculatively stage the first layers for a following prefill chunk.
        _issue(state, nxt - len(order))
    moe_fused_mul_sum(inputs=buffers['down_output'].view(*topk_ids.shape, -1),
                      topk_weights=topk_weights, topk_ids=topk_ids, expert_map=expert_map,
                      outputs=output)
    state.staged_calls += 1
    if pos == 0 and state.staged_calls <= 2 * len(order):
        logger.info('Stream staging active: device=%s tokens=%d copies=%d snapshots=%d',
                    state.device, hidden_states.shape[0], state.copies, getattr(state, 'snapshots', 0))
    return True
