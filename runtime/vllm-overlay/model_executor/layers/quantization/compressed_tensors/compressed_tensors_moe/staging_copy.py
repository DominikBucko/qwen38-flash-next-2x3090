# SPDX-License-Identifier: Apache-2.0
"""Unvalidated bounded-grid copy for the existing expert LRU.

Keep the routing, replacement policy, source bytes and destination slots intact.
The old copy launches one CTA per 1,024 elements even for cache hits. Bound
CTAs per requested expert; misses loop over their row, hits return immediately.
"""
from vllm.triton_utils import tl, triton


@triton.jit
def _copy_rows(source, output, miss_ids, miss_slots,
               ROW_SIZE: tl.constexpr, CTAS: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    source_row = tl.load(miss_ids+request)
    output_row = tl.load(miss_slots+request)
    if source_row < 0 or output_row < 0:
        return
    offsets = tl.arange(0, BLOCK)
    for tile in tl.range(tl.program_id(1), tl.cdiv(ROW_SIZE, BLOCK), CTAS,
                         num_stages=2):
        columns = tile*BLOCK+offsets
        values = tl.load(source+source_row.to(tl.int64)*ROW_SIZE+columns,
                         mask=columns < ROW_SIZE)
        tl.store(output+output_row.to(tl.int64)*ROW_SIZE+columns, values,
                 mask=columns < ROW_SIZE)


def copy_rows(source, output, miss_ids, miss_slots, num_ids,
              cta_cap=64, block=1024, warps=4):
    if source is None:
        assert output is None
        return
    assert output is not None
    if source.ndim == 0 or output.ndim == 0:
        return
    if not source.is_contiguous() or not output.is_contiguous():
        raise ValueError('expert rows must be contiguous')
    if source.dtype != output.dtype or source.shape[1:] != output.shape[1:]:
        raise ValueError('expert row layout differs')
    row_size = source[0].numel()
    if num_ids == 0 or row_size == 0:
        return
    ctas = min(triton.cdiv(row_size, block), cta_cap)
    _copy_rows[(num_ids, ctas)](source, output, miss_ids, miss_slots,
                              ROW_SIZE=row_size, CTAS=ctas, BLOCK=block,
                              num_warps=warps)
