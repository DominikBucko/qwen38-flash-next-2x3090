# SPDX-License-Identifier: Apache-2.0
"""Screened experiment, not a demonstrated upgrade: restricted MTP vocabulary.

Only the draft head is restricted. Target logits and rejection sampling stay
unchanged. Missing draft tokens can reduce acceptance, especially across
languages. Ranges let us use views of the existing TP-sharded head, rather
than allocate another head beside an already tight full-context KV cache.
"""
import json

import torch
import torch.nn.functional as F


def parse_ranges(value: str, vocab_size: int) -> tuple[tuple[int, int], ...]:
    ranges = json.loads(value)
    if not isinstance(ranges, list) or not ranges:
        raise ValueError("draft vocabulary needs a non-empty list of [start, end) ranges")
    result = []
    previous_end = -1
    for pair in ranges:
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(index) is not int for index in pair)):
            raise ValueError("draft vocabulary range boundaries must be integers")
        start, end = pair
        if not 0 <= start < end <= vocab_size or start < previous_end:
            raise ValueError("draft vocabulary ranges must be sorted, disjoint, and in bounds")
        result.append((start, end))
        previous_end = end
    return tuple(result)


def local_top_pair(hidden_states, weight, shard_start, shard_end, ranges,
                   *, scale=1.0, soft_cap=None):
    """Return (logit, global token ID) per row; exclude shard padding."""
    if scale <= 0:
        raise ValueError("greedy draft reduction requires a positive scale")
    values = indices = None
    for start, end in ranges:
        start, end = max(start, shard_start), min(end, shard_end)
        if start >= end:
            continue
        # A view, never index_select/contiguous: no duplicate head allocation.
        selected_weight = weight[start - shard_start:end - shard_start]
        if (hidden_states.shape[0] <= 8 and hidden_states.dtype == torch.bfloat16
                and selected_weight.is_contiguous() and hidden_states.is_contiguous()):
            from .triton_skinny import skinny_mm
            logits = skinny_mm(hidden_states, selected_weight)
        else:
            logits = F.linear(hidden_states, selected_weight)
        if soft_cap is not None:
            logits = torch.tanh(logits / soft_cap) * soft_cap
        if scale != 1.0:
            logits = logits * scale
        part_values, part_indices = logits.max(dim=-1)
        part_indices = part_indices + start
        if values is None:
            values, indices = part_values, part_indices
        else:
            # Sorted ranges plus strict > preserve the lowest-ID tie break.
            better = part_values > values
            values = torch.where(better, part_values, values)
            indices = torch.where(better, part_indices, indices)
    if values is None:
        values = hidden_states.new_full((hidden_states.shape[0],), -float("inf"))
        indices = torch.zeros(hidden_states.shape[0], dtype=torch.int64,
                              device=hidden_states.device)
    return torch.stack((values.float(), indices.float()), dim=-1)


def reduce_top_pairs(gathered, tp_size):
    """Match the pinned local-argmax collective's rank and token tie order."""
    pairs = gathered.reshape(-1, tp_size, 2)
    winner = pairs[:, :, 0].argmax(dim=-1, keepdim=True)
    return pairs[:, :, 1].gather(-1, winner).squeeze(-1).to(torch.int64)


def get_top_tokens(processor, lm_head, hidden_states, ranges, all_gather):
    from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

    if not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod):
        raise ValueError("reduced draft vocabulary requires an unquantized LM head")
    if processor.head_dtype not in (None, hidden_states.dtype):
        raise ValueError("reduced draft vocabulary does not support a separate head dtype")
    if hidden_states.ndim != 2:
        raise ValueError("expected [tokens, hidden] draft states")
    shard = lm_head.shard_indices
    pair = local_top_pair(
        hidden_states, lm_head.weight, shard.org_vocab_start_index,
        min(shard.org_vocab_end_index, processor.org_vocab_size), ranges,
        scale=processor.scale, soft_cap=processor.soft_cap,
    )
    if lm_head.tp_size > 1:
        pair = all_gather(pair, dim=-1)
    return reduce_top_pairs(pair, lm_head.tp_size)
