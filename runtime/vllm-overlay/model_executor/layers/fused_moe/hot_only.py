# SPDX-License-Identifier: Apache-2.0
"""Hot-only expert placement (QWEN38_HOT_ONLY=<experts per layer and GPU>).

Each GPU owns only its hot experts of every target layer: the expert map sends the hot global ids to local slots
0..H-1 and every other id to -1, so the loader skips cold experts and the kernels skip their tokens. Cold experts
are computed from a host copy instead (CPU for decode, streamed to the GPU for prefill; see
compressed_tensors_moe/expert_store.py). With expert parallelism each rank takes its hot set from the experts it
owns (the first H of them in the static rankings), and its host copy holds only its own cold experts, so nothing
is stored twice. The MTP draft layer is not affected.
"""
import json
import os
import re

import torch

_RANKINGS = None


def size() -> int:
    return int(os.environ.get("QWEN38_HOT_ONLY", "0"))


def layer_index(layer_name: str) -> int | None:
    if "language_model.model.layers." not in layer_name:
        return None
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    return int(match.group(1)) if match else None


def _rankings() -> dict:
    global _RANKINGS
    if _RANKINGS is None:
        path = os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_FILE") or os.path.join(os.getcwd(), "static_hot_cache_rankings.json")
        with open(path) as fh:
            _RANKINGS = json.load(fh)
    return _RANKINGS


def hot_ids(layer: int, global_num_experts: int, owned=None) -> list[int]:
    """The first H ranked experts of `layer` among `owned` (default: all), H = QWEN38_HOT_ONLY."""
    count = size()
    owned = set(range(global_num_experts)) if owned is None else set(owned)
    ids, seen = [], set()
    for raw in _rankings().get(str(layer), []):
        g = int(raw)
        if 0 <= g < global_num_experts and g in owned and g not in seen:
            ids.append(g)
            seen.add(g)
            if len(ids) == count:
                break
    if len(ids) != count:
        raise RuntimeError(f"hot-only: layer {layer} ranks {len(ids)} of this rank's experts, need {count}")
    return ids


def rank_and_size() -> tuple[int, int]:
    """(rank, size) of this worker in the tensor-parallel group, which is also the expert-parallel group here."""
    try:
        from vllm.distributed import parallel_state
        group = parallel_state.get_tp_group()
        return group.rank_in_group, group.world_size
    except Exception:
        return 0, 1


def apply_placement(manager, layer_name: str, global_num_experts: int) -> None:
    """Turn an ExpertMapManager into a hot-only one for target MoE layers (one GPU, or one expert-parallel rank)."""
    if size() <= 0:
        return
    layer = layer_index(layer_name)
    if layer is None:
        return
    if manager.num_fused_shared_experts:
        raise RuntimeError("hot-only placement does not support fused shared experts")
    if manager.expert_map is None:
        owned = list(range(global_num_experts))
    else:                                       # expert parallelism: only this rank's experts
        owned = torch.nonzero(manager.expert_map >= 0).flatten().tolist()
    ids = hot_ids(layer, global_num_experts, owned)
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    expert_map[torch.tensor(ids, dtype=torch.long)] = torch.arange(len(ids), dtype=torch.int32)
    manager._local_num_experts = len(ids)
    manager._expert_map = expert_map
    manager._qwen38_hot_ids = ids
    manager._qwen38_owned_ids = owned
