# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental QSA score specialization with unchanged selection semantics.

Keep the released compressed cache, rollback metadata, approximate/exact top-k
switch and causal expansion. Only scoring tiles and query/key reuse change.
"""
import os
import torch
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from .qsa_special_kernels import (
    _qsa_mqa_paged_uniform_kernel, _qsa_mqa_paged_prefill_kernel)
from . import qsa as legacy

WORKSPACE_BYTES = 64 * 1024**2
TILE_R = int(os.environ.get('VLLM_QSA_PREFILL_TILE_R', '16'))
if TILE_R not in (4, 8, 16, 32, 64):
    raise ValueError('QSA prefill tile must be 4,8,16,32 or64')


@triton.jit
def _visible_kernel(positions, requests, seq_lens, output, count,
                    num_reqs, RATIO: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    active = row < count
    req = tl.load(requests+row, mask=active, other=-1)
    position = tl.load(positions+row, mask=active, other=-1)
    valid = active & (req >= 0) & (req < num_reqs) & (position >= 0)
    length = tl.load(seq_lens+req, mask=valid, other=0)
    visible = tl.minimum((position+1)//RATIO, length//RATIO)
    tl.store(output+row, tl.where(valid, visible, 0), mask=active)


def visible_blocks(positions, requests, seq_lens, ratio):
    output = torch.empty(positions.shape, device=positions.device, dtype=torch.int32)
    if positions.numel():
        _visible_kernel[(triton.cdiv(positions.numel(), 128),)](
            positions, requests, seq_lens, output, positions.numel(),
            seq_lens.numel(), RATIO=ratio, BLOCK=128)
    return output


def decode_logits(q, cache, table, qsl, visible, max_query_len):
    columns = table.shape[1]*cache.shape[1]
    logits = torch.empty((q.shape[0], columns), device=q.device, dtype=torch.float32)
    programs = table.shape[0]*triton.cdiv(columns, 64)
    tiles = 1 if programs < 16384 else 2 if programs < 32768 else 4 if programs < 131072 else 8
    _qsa_mqa_paged_uniform_kernel[(table.shape[0], triton.cdiv(columns, 64*tiles))](
        q, cache, table, qsl, visible, logits, *q.stride()[:-1],
        *cache.stride()[:2], table.stride(0), logits.stride(0),
        PAGE_SIZE=cache.shape[1], PAGE_TABLE_WIDTH=table.shape[1],
        NUM_PAGES=cache.shape[0], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
        # The kernel reads the live length from qsl and already pads3 to4.
        # Reuse the four-row specialization at a three-token context tail.
        DECODE_QUERY_LEN=triton.next_power_of_2(max_query_len), BLOCK_N=64, TILES_PER_PROG=tiles,
        STAGES=2, num_warps=2)
    return logits


def prefill_logits(q, cache, table, qsl, visible, max_query_len,
                   query_offset, num_queries, tile_r=TILE_R):
    columns = table.shape[1]*cache.shape[1]
    logits = torch.empty((num_queries, columns), device=q.device, dtype=torch.float32)
    grid = (table.shape[0], triton.cdiv(min(num_queries, max_query_len), tile_r),
            triton.cdiv(columns, 64*16))
    _qsa_mqa_paged_prefill_kernel[grid](
        q, cache, table, qsl, visible, logits, *q.stride()[:-1],
        *cache.stride()[:2], table.stride(0), logits.stride(0),
        num_queries, query_offset,
        PAGE_SIZE=cache.shape[1], PAGE_TABLE_WIDTH=table.shape[1],
        NUM_PAGES=cache.shape[0], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
        TILE_R=tile_r, BLOCK_N=64, K_TILES=16, STAGES=2, num_warps=4)
    return logits


def topk(logits, visible, blocks, workspace, k):
    columns = logits.shape[1]
    if legacy._QSA_TOPK_MODE == '1':
        legacy._qsa_exact_topk(logits, visible, blocks, k, columns)
        return
    if legacy._QSA_TOPK_MODE == 'fill':
        legacy._qsa_mask_invisible_(logits, visible, columns)
    cooperative = (blocks.shape[0] <= 32 and logits.stride(0) % 4 == 0
                   and current_platform.has_device_capability(90)
                   and not current_platform.is_device_capability_family(120))
    op = torch.ops._C.cooperative_topk if cooperative else torch.ops._C.persistent_topk
    op(logits, visible, blocks, workspace, k, columns)


def select(q, cache, metadata, token_topk, ratio, out=None):
    rows = q.shape[0]
    width = token_topk+ratio-1
    if out is None:
        out = torch.empty((rows, width), device=q.device, dtype=torch.int32)
    if out.shape != (rows, width) or token_topk % ratio:
        raise ValueError('invalid QSA selection shape')
    if not rows:
        return out
    if q.stride(-1) != 1 or cache.stride(-1) != 1 or metadata.block_table.stride(-1) != 1:
        raise ValueError('specialized QSA needs contiguous head dimensions and page rows')
    table = metadata.block_table
    columns = table.shape[1]*cache.shape[1]
    positions = metadata.logical_positions[:rows]
    requests = metadata.token_to_req[:rows]
    visible = visible_blocks(positions, requests, metadata.seq_lens, ratio)
    qsl = metadata.query_start_loc
    max_query_len = metadata.max_query_len
    if max_query_len <= 0:
        raise ValueError('QSA maximum query length missing from metadata')
    rows_per_chunk = max(1, WORKSPACE_BYTES//max(columns*4, 1))
    blocks = torch.empty((min(rows, rows_per_chunk), token_topk//ratio),
                         device=q.device, dtype=torch.int32)
    workspace = torch.empty((1024**2,), device=q.device, dtype=torch.uint8)
    # Ragged short requests are valid here: the kernel reads live qsl boundaries.
    decode = max_query_len <= 4 and rows <= rows_per_chunk
    for start in range(0, rows, rows_per_chunk):
        end = min(start+rows_per_chunk, rows)
        part = slice(start, end)
        if decode:
            logits = decode_logits(q, cache, table, qsl, visible, max_query_len)
        else:
            logits = prefill_logits(q, cache, table, qsl, visible,
                                     max_query_len, start, end-start)
        selected = blocks[:end-start]
        topk(logits, visible[part], selected, workspace, token_topk//ratio)
        legacy.expand_qsa_block_indices_cuda(selected, positions[part],
                                             metadata.seq_lens, requests[part],
                                             ratio, token_topk, out[part])
        # Never retain one chunk while allocating the next 64MiB score tensor.
        del logits
    return out
