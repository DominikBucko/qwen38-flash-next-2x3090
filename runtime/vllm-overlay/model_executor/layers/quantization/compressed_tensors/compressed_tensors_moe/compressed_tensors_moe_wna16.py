# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import re
from dataclasses import dataclass, replace
from typing import Any

import torch
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEExpertsModular,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    WNA16MoEBackend,
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    make_wna16_moe_quant_config,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_TYPES_MAP,
    WNA16_ZP_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_config,
    get_marlin_input_dtype,
    marlin_make_workspace_new,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4Static32GroupScale,
    kInt4StaticGroupScale,
    kInt8StaticGroupScale,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)


@triton.jit
def _prepare_static_stage_map_kernel(
    global_ids_ptr,
    original_map_ptr,
    stage_map_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < num_ids
    global_ids = tl.load(global_ids_ptr + offsets, mask=mask, other=-1)
    valid_ids = mask & (global_ids >= 0) & (global_ids < global_num_experts)
    global_safe = tl.minimum(tl.maximum(global_ids, 0), global_num_experts - 1)
    local_ids = tl.load(original_map_ptr + global_safe, mask=valid_ids, other=-1)
    tl.store(
        stage_map_ptr + global_safe,
        offsets,
        mask=valid_ids & (local_ids >= 0),
    )


@triton.jit
def _hybrid_gather_expert_rows_kernel(
    cold_ptr,
    hot_ptr,
    output_ptr,
    global_ids_ptr,
    original_map_ptr,
    hot_map_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    column_mask = offsets < row_size
    global_id = tl.load(global_ids_ptr + row)
    valid_id = (global_id >= 0) & (global_id < global_num_experts)
    global_safe = tl.minimum(tl.maximum(global_id, 0), global_num_experts - 1)
    local_id = tl.load(original_map_ptr + global_safe, mask=valid_id, other=-1)
    hot_id = tl.load(hot_map_ptr + global_safe, mask=valid_id, other=-1)
    local_safe = tl.maximum(local_id, 0)
    hot_safe = tl.maximum(hot_id, 0)
    hot_values = tl.load(
        hot_ptr + hot_safe * row_size + offsets,
        mask=column_mask & (local_id >= 0) & (hot_id >= 0),
        other=0,
    )
    cold_values = tl.load(
        cold_ptr + local_safe * row_size + offsets,
        mask=column_mask & (local_id >= 0) & (hot_id < 0),
        other=0,
    )
    tl.store(
        output_ptr + row * row_size + offsets,
        hot_values + cold_values,
        mask=column_mask & (local_id >= 0),
    )


@triton.jit
def _permute_expert_rows_kernel(
    source_ptr,
    output_ptr,
    new_to_old_ptr,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    new_row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    mask = offsets < row_size
    old_row = tl.load(new_to_old_ptr + new_row)
    values = tl.load(source_ptr + old_row * row_size + offsets, mask=mask)
    tl.store(output_ptr + new_row * row_size + offsets, values, mask=mask)


@triton.jit
def _update_lru_expert_map_kernel(
    global_ids_ptr,
    source_map_ptr,
    cache_map_ptr,
    slot_global_ids_ptr,
    slot_ages_ptr,
    clock_ptr,
    miss_local_ids_ptr,
    miss_slots_ptr,
    rot_map_ptr,
    lru_dummy_ptr,
    num_ids: tl.constexpr,
    global_num_experts: tl.constexpr,
    capacity: tl.constexpr,
    id_block: tl.constexpr,
    capacity_block: tl.constexpr,
    HYBRID: tl.constexpr,
):
    id_offsets = tl.arange(0, id_block)
    requested = tl.load(
        global_ids_ptr + id_offsets, mask=id_offsets < num_ids, other=-1
    )
    slot_offsets = tl.arange(0, capacity_block)
    valid_slots = slot_offsets < capacity
    slot_global_ids = tl.load(
        slot_global_ids_ptr + slot_offsets, mask=valid_slots, other=-1
    )
    slot_ages = tl.load(slot_ages_ptr + slot_offsets, mask=valid_slots, other=0)
    protected = ~valid_slots
    for request_index in tl.static_range(0, num_ids):
        requested_global_id = tl.sum(
            tl.where(id_offsets == request_index, requested, 0), axis=0
        )
        protected = protected | (slot_global_ids == requested_global_id)

    clock = tl.load(clock_ptr)
    for request_index in tl.static_range(0, num_ids):
        global_id = tl.sum(
            tl.where(id_offsets == request_index, requested, 0), axis=0
        )
        valid_global = (global_id >= 0) & (global_id < global_num_experts)
        global_safe = tl.minimum(
            tl.maximum(global_id, 0), global_num_experts - 1
        )
        local_id = tl.load(
            source_map_ptr + global_safe, mask=valid_global, other=-1
        )
        is_local = valid_global & (local_id >= 0)
        cache_slot = tl.load(
            cache_map_ptr + global_safe, mask=is_local, other=-1
        )
        is_hit = is_local & (cache_slot >= 0)
        appeared_before = tl.sum(
            (requested == global_id) & (id_offsets < request_index)
        ) > 0

        victim_ages = tl.where(protected, 0x7FFFFFFF, slot_ages)
        victim = tl.argmin(victim_ages, axis=0)
        is_miss = is_local & ~is_hit & ~appeared_before
        assigned_slot = tl.where(is_hit, cache_slot, victim)
        new_age = clock + request_index + 1

        old_global_id = tl.sum(
            tl.where(slot_offsets == victim, slot_global_ids, 0), axis=0
        )
        old_valid = is_miss & (old_global_id >= 0)
        tl.store(cache_map_ptr + old_global_id, -1, mask=old_valid)
        tl.store(cache_map_ptr + global_safe, victim, mask=is_miss)
        if HYBRID:
            # rot_map keeps the prefill access path pointing at each owned
            # expert's best live copy: the mirror slot while cached (the slot
            # is VRAM, so prefill rides the rotation too), and the evicted
            # expert's host-backed row (its cold_map value) once its slot is
            # handed to the new expert.  lru_dummy_ptr is the non-hybrid
            # stand-in and is deliberately never dereferenced.
            tl.store(
                rot_map_ptr + global_safe,
                tl.where(is_hit, cache_slot, victim),
                mask=is_hit | is_miss,
            )
            old_cold_row = tl.load(
                source_map_ptr
                + tl.minimum(
                    tl.maximum(old_global_id, 0), global_num_experts - 1
                ),
                mask=old_global_id >= 0,
                other=-1,
            )
            tl.store(
                rot_map_ptr + old_global_id,
                old_cold_row,
                mask=old_valid,
            )

        slot_global_ids = tl.where(
            is_miss & (slot_offsets == victim), global_id, slot_global_ids
        )
        slot_ages = tl.where(
            (is_hit | is_miss) & (slot_offsets == assigned_slot),
            new_age,
            slot_ages,
        )
        protected = protected | (is_miss & (slot_offsets == victim))
        tl.store(
            miss_local_ids_ptr + request_index,
            tl.where(is_miss, local_id, -1),
        )
        tl.store(
            miss_slots_ptr + request_index,
            tl.where(is_miss, victim, -1),
        )

    tl.store(slot_global_ids_ptr + slot_offsets, slot_global_ids, mask=valid_slots)
    tl.store(slot_ages_ptr + slot_offsets, slot_ages, mask=valid_slots)
    tl.store(clock_ptr, clock + num_ids)


@triton.jit
def _gather_lru_expert_rows_kernel(
    source_ptr,
    cache_ptr,
    miss_local_ids_ptr,
    miss_slots_ptr,
    row_size: tl.constexpr,
    num_ids: tl.constexpr,
    block_size: tl.constexpr,
):
    request_index = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * block_size + tl.arange(0, block_size)
    source_row = tl.load(miss_local_ids_ptr + request_index)
    cache_row = tl.load(miss_slots_ptr + request_index)
    mask = (source_row >= 0) & (cache_row >= 0) & (offsets < row_size)
    values = tl.load(source_ptr + source_row * row_size + offsets, mask=mask)
    tl.store(cache_ptr + cache_row * row_size + offsets, values, mask=mask)


@dataclass
class _StaticStageBuffers:
    """Process-global scratch reused sequentially by every target MoE layer."""

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_zp: torch.Tensor | None
    w2_zp: torch.Tensor | None
    w13_g_idx: torch.Tensor | None
    w2_g_idx: torch.Tensor | None
    w13_sort: torch.Tensor | None
    w2_sort: torch.Tensor | None
    expert_map: torch.Tensor
    capacity: int


@dataclass
class _StaticHotExpertCache:
    """Fixed VRAM mirror for frequently routed UVA-resident experts.

    The original slot-indexed tensors remain in mapped host memory and serve
    as the cold path.  A second Marlin invocation evaluates only ``hot_map``;
    ``cold_map`` masks those experts from the original invocation.  Both maps
    are immutable, which keeps the path compatible with CUDA graphs and avoids
    a device-to-host routing synchronization on every layer and decode step.
    """

    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_scale: torch.Tensor
    w2_scale: torch.Tensor
    w13_zp: torch.Tensor | None
    w2_zp: torch.Tensor | None
    w13_g_idx: torch.Tensor
    w2_g_idx: torch.Tensor
    w13_sort: torch.Tensor
    w2_sort: torch.Tensor
    hot_map: torch.Tensor
    cold_map: torch.Tensor
    kernel: Any
    global_ids: tuple[int, ...]
    stage: _StaticStageBuffers | None
    dynamic_lru: bool = False
    slot_global_ids: torch.Tensor | None = None
    slot_ages: torch.Tensor | None = None
    clock: torch.Tensor | None = None
    miss_local_ids: torch.Tensor | None = None
    miss_slots: torch.Tensor | None = None
    # Hybrid only: global-id -> current row of the VMM tensor (mirror slot
    # while cached, host-backed row once evicted).  None on the golden arm.
    rot_map: torch.Tensor | None = None
    # Shared 1-element stand-in for the kernel's rot_map argument on
    # non-hybrid caches, so the signature change stays call-site compatible.
    lru_dummy: torch.Tensor | None = None


@dataclass
class _MixedVMMAllocation:
    """Keep CUDA VMM mappings and their non-owning PyTorch storage alive."""

    storage: Any
    address: Any
    handles: tuple[Any, ...]
    mapped_bytes: int
    gpu_bytes: int
    host_bytes: int
    # v2 hybrid only: data bytes of the host-backed duplicate of the hot
    # prefix (``hot_experts`` rows appended after the full permutation).
    # Zero on the arm-1 layout.
    duplicate_data_bytes: int = 0


_STATIC_STAGE_BUFFERS: dict[tuple[Any, ...], _StaticStageBuffers] = {}
_MIXED_VMM_ALLOCATIONS: list[_MixedVMMAllocation] = []


class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        # Extract properties from weight_quant
        self.symmetric = weight_quant.symmetric
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.group_size = weight_quant.group_size
        self.actorder = weight_quant.actorder
        self.layer_name = layer_name or ""
        self._weight_scale_refine = int(
            os.getenv("VLLM_WNA16_TP_SCALE_REFINE", "1")
        )
        if (
            self._weight_scale_refine > 1
            and self.group_size > 0
            and self.moe.intermediate_size_per_partition % self.group_size == 0
        ):
            # The draft checkpoint's group-32 scales already align with its
            # TP shard.  Refine only layers whose checkpoint groups actually
            # cross a shard boundary (group-128 target experts here).
            self._weight_scale_refine = 1
        if self._weight_scale_refine > 1:
            parallel = self.moe.moe_parallel_config
            if parallel.use_ep or parallel.tp_size <= 1:
                raise ValueError(
                    "VLLM_WNA16_TP_SCALE_REFINE is only valid for tensor-parallel "
                    "MoE without expert parallelism."
                )
            if not self.symmetric:
                raise ValueError(
                    "VLLM_WNA16_TP_SCALE_REFINE currently supports symmetric "
                    "quantization only."
                )
            if self.group_size <= 0 or self.group_size % self._weight_scale_refine:
                raise ValueError(
                    "Checkpoint WNA16 group size must be positive and divisible "
                    "by VLLM_WNA16_TP_SCALE_REFINE."
                )
            checkpoint_group_size = self.group_size
            self.group_size //= self._weight_scale_refine
            self.weight_quant = weight_quant.model_copy(
                update={"group_size": self.group_size}
            )
            logger.info_once(
                "Losslessly refining WNA16 expert scales from group-%d to "
                "group-%d for TP%d by duplicating checkpoint scales %dx.",
                checkpoint_group_size,
                self.group_size,
                parallel.tp_size,
                self._weight_scale_refine,
                scope="local",
            )
        self._static_hot_cache: _StaticHotExpertCache | None = None
        self._static_hot_cache_max_tokens = int(
            os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS", "16")
        )
        self._static_stage_enabled = os.getenv(
            "VLLM_WNA16_STATIC_STAGE_CACHE", "0"
        ).lower() in ("1", "true", "yes")
        self._static_stage_capacity = int(
            os.getenv("VLLM_WNA16_STATIC_STAGE_CAPACITY", "128")
        )
        self._dynamic_lru_enabled = os.getenv(
            "VLLM_WNA16_DYNAMIC_LRU", "0"
        ).lower() in ("1", "true", "yes")
        self._mixed_vmm_enabled = os.getenv(
            "VLLM_WNA16_MIXED_VMM_HOT_CACHE", "0"
        ).lower() in ("1", "true", "yes")
        self._vmm_lru_hybrid = os.getenv("VLLM_WNA16_VMM_LRU_HYBRID", "0") == "1"
        self._mixed_vmm_active = False
        # Permutation state recorded by maybe_init_mixed_vmm_hot_cache so the
        # hybrid initializer can reuse it instead of duplicating the logic.
        self._mixed_vmm_state: dict[str, Any] | None = None

        # Extract quant_type and create weight key for oracle selection
        self.quant_type = (
            WNA16_SUPPORTED_TYPES_MAP[self.num_bits]
            if self.symmetric
            else WNA16_ZP_SUPPORTED_TYPES_MAP[self.num_bits]
        )

        if self.num_bits == 4:
            if self.group_size == 32:
                scale = kInt4Static32GroupScale
            else:
                scale = kInt4StaticGroupScale
        elif self.num_bits == 8:
            assert self.group_size == -1
            scale = kInt8StaticGroupScale
        else:
            raise ValueError(
                "CompressedTensorsWNA16MoEMethod only supports int4 and int8 now."
            )

        weight_key = QuantKey(self.quant_type, scale, symmetric=self.symmetric)

        is_actorder = self.strategy == QuantizationStrategy.GROUP and self.actorder in (
            ActivationOrdering.GROUP,
            ActivationOrdering.DYNAMIC,
        )

        # Select WNA16 MoE backend via oracle.
        selection_moe = self.moe
        if self.moe.moe_backend == "humming" and self.group_size == 32:
            # Humming on Ampere supports the target model's symmetric INT4
            # group-128 experts, but not the separately quantized group-32 MTP
            # draft.  Keep the target on Humming and select Marlin for only the
            # draft instead of rejecting the whole speculative configuration.
            selection_moe = replace(self.moe, moe_backend="marlin")
            logger.info_once(
                "Using Marlin for group-32 WNA16 MoE while Humming remains "
                "selected for supported target layers.",
                scope="local",
            )

        self.wna16_backend, self.experts_cls = select_wna16_moe_backend(
            config=selection_moe,
            weight_key=weight_key,
            quant_config=self.weight_quant,
            may_have_zp=not self.symmetric,
            may_have_bias=False,
            allow_tile_padding=not is_actorder,
        )

        self.is_marlin = self.wna16_backend in [
            WNA16MoEBackend.MARLIN,
            WNA16MoEBackend.BATCHED_MARLIN,
        ]
        self.is_transposed = self.wna16_backend != WNA16MoEBackend.FLASHINFER_TRTLLM

        if self.is_marlin:
            assert check_moe_marlin_supports_config(
                self.moe, self.group_size, allow_tile_padding=not is_actorder
            )
            self.input_dtype = get_marlin_input_dtype(layer_name)
        else:
            # channelwise is not supported by this kernel
            assert weight_quant.strategy == "group"
            # grouped actorder isn't supported by this kernel
            assert weight_quant.actorder != "group"

            assert self.symmetric, "Only symmetric quantization is supported for MoE"

            # Non-Marlin WNA16 always uses bf16/fp16 inputs
            self.input_dtype = torch.bfloat16

    def get_weight_shape(
        self,
        weight_name: str,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        num_groups_w2: int | None = None,
        num_groups_w13: int | None = None,
    ) -> tuple[int, int, int]:
        """
        Get the shape of the weight based on the weight name, number of experts
        hidden size, intermediate size per partition, number of groups for w2,
        and number of groups for w13. Pass in num_groups_w2 and num_groups_w13
        for weight scales/zero_points.
        """
        if weight_name in ("w13_scale", "w13_zp"):
            assert num_groups_w13 is not None, (
                "num_groups_w13 must be provided for weight scales/zero_points"
            )
        if weight_name in ("w2_scale", "w2_zp"):
            assert num_groups_w2 is not None, (
                "num_groups_w2 must be provided for weight scales/zero_points"
            )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        shape_map = {
            "w13_weight": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    hidden_size // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    hidden_size // self.packed_factor,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w13_scale": {
                "Flashinfer": (
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    num_groups_w13,
                ),
                "Marlin": (
                    num_experts,
                    num_groups_w13,
                    w13_num_shards * intermediate_size_per_partition,
                ),
            },
            "w13_zp": {
                "Marlin": (
                    num_experts,
                    num_groups_w13,
                    w13_num_shards
                    * intermediate_size_per_partition
                    // self.packed_factor,
                ),
            },
            "w2_weight": {
                "Flashinfer": (
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition // self.packed_factor,
                ),
                "Marlin": (
                    num_experts,
                    intermediate_size_per_partition // self.packed_factor,
                    hidden_size,
                ),
            },
            "w2_scale": {
                "Flashinfer": (num_experts, hidden_size, num_groups_w2),
                "Marlin": (num_experts, num_groups_w2, hidden_size),
            },
            "w2_zp": {
                "Marlin": (
                    num_experts,
                    num_groups_w2,
                    hidden_size // self.packed_factor,
                ),
            },
        }
        backend_key = "Marlin" if self.is_transposed else "Flashinfer"
        return shape_map[weight_name][backend_key]

    @staticmethod
    def _w2_scale_sharding(
        actorder,
        group_size: int,
        intermediate_size_per_partition: int,
        intermediate_size_full: int,
    ) -> tuple[bool, int, bool]:
        """Decide how to shard w2 group scales across TP for WNA16 Marlin MoE.

        Only ``actorder="group"`` permutes activations by ``g_idx`` at runtime
        and therefore needs the full-K (unsharded) w2 scales plus ``is_k_full``.
        ``actorder="weight"``/``"static"`` (and ``None``) reorder weights at
        quantization time, so scales shard normally per TP rank.
        """
        load_full_w2 = (actorder == "group") and group_size != -1
        w2_scales_size = (
            intermediate_size_full if load_full_w2 else intermediate_size_per_partition
        )
        is_k_full = (actorder != "group") or (
            intermediate_size_per_partition == intermediate_size_full
        )
        return load_full_w2, w2_scales_size, is_k_full

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        intermediate_size_full = extra_weight_attrs.pop("intermediate_size_full")

        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": self.is_transposed, "quant_method": self.strategy}
        )

        w13_weight = torch.nn.Parameter(
            torch.empty(
                *self.get_weight_shape(
                    "w13_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                *self.get_weight_shape(
                    "w2_weight",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                ),
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        load_full_w2, w2_scales_size, self.is_k_full = self._w2_scale_sharding(
            self.actorder,
            self.group_size,
            intermediate_size_per_partition,
            intermediate_size_full,
        )

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            if hidden_size % self.group_size != 0:
                raise ValueError(
                    "CompressedTensors WNA16 MoE requires hidden_size "
                    f"({hidden_size}) to be divisible by group_size "
                    f"({self.group_size})."
                )
            if (
                not load_full_w2
                and intermediate_size_per_partition % self.group_size != 0
            ):
                raise ValueError(
                    "CompressedTensors WNA16 MoE with static group "
                    "scales requires the MoE intermediate size per "
                    "tensor-parallel partition "
                    f"({intermediate_size_per_partition}) to be divisible by "
                    f"group_size ({self.group_size}). Scale groups would "
                    "otherwise cross TP shard boundaries; use a compatible TP "
                    "size or enable expert parallelism."
                )
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        layer.num_groups_w13 = num_groups_w13
        layer.num_groups_w2 = num_groups_w2

        w13_scale = torch.nn.Parameter(
            torch.ones(
                *self.get_weight_shape(
                    "w13_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w13=num_groups_w13,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)
        set_weight_attrs(
            w13_scale, {"weight_scale_refine": self._weight_scale_refine}
        )

        w2_scale = torch.nn.Parameter(
            torch.ones(
                *self.get_weight_shape(
                    "w2_scale",
                    num_experts,
                    hidden_size,
                    intermediate_size_per_partition,
                    num_groups_w2=num_groups_w2,
                ),
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"weight_scale_refine": self._weight_scale_refine})
        set_weight_attrs(w2_scale, {"load_full_w2": load_full_w2})

        if not self.symmetric:
            w13_zp = torch.nn.Parameter(
                torch.zeros(
                    *self.get_weight_shape(
                        "w13_zp",
                        num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        num_groups_w13=num_groups_w13,
                    ),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_zero_point", w13_zp)
            set_weight_attrs(w13_zp, extra_weight_attrs)

            w2_zp = torch.nn.Parameter(
                torch.zeros(
                    *self.get_weight_shape(
                        "w2_zp",
                        num_experts,
                        hidden_size,
                        intermediate_size_per_partition,
                        num_groups_w2=num_groups_w2,
                    ),
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_zero_point", w2_zp)
            set_weight_attrs(w2_zp, extra_weight_attrs)

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def _setup_kernel(self, layer: RoutedExperts):
        assert self.experts_cls is not None
        if self.wna16_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming_utils import (
                get_humming_moe_quant_config,
            )

            self.moe_quant_config = get_humming_moe_quant_config(
                layer,
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
            )
        else:
            self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.moe_quant_config is not None

        # Add Marlin-specific arguments
        marlin_args: dict[str, Any] = {}
        if self.is_marlin:
            marlin_args = {
                "w13_g_idx": layer.w13_weight_g_idx,
                "w2_g_idx": layer.w2_weight_g_idx,
                "w13_g_idx_sort_indices": layer.w13_g_idx_sort_indices,
                "w2_g_idx_sort_indices": layer.w2_g_idx_sort_indices,
                "is_k_full": self.is_k_full,
            }

        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            backend=self.wna16_backend,
            routing_tables=layer._expert_routing_tables(),
            **marlin_args,
        )

    @staticmethod
    def _cache_expert_rows(
        tensor: torch.Tensor | None,
        local_ids: torch.Tensor,
        local_num_experts: int,
    ) -> torch.Tensor | None:
        if tensor is None or tensor.ndim == 0:
            return tensor
        if tensor.shape[0] != local_num_experts:
            return tensor
        return torch.index_select(tensor, 0, local_ids).contiguous()

    @staticmethod
    def _stage_signature(tensor: torch.Tensor | None) -> tuple[Any, ...] | None:
        if tensor is None:
            return None
        return (tuple(tensor.shape[1:]), tensor.dtype)

    @staticmethod
    def _allocate_stage_rows(
        tensor: torch.Tensor | None,
        capacity: int,
        local_num_experts: int,
    ) -> torch.Tensor | None:
        if tensor is None:
            return None
        if tensor.ndim == 0 or tensor.shape[0] != local_num_experts:
            raise ValueError(
                "Static WNA16 staging requires expert-indexed quant tensors; "
                f"got shape={tuple(tensor.shape)}, experts={local_num_experts}"
            )
        return torch.empty(
            (capacity, *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    def _get_static_stage_buffers(
        self,
        original_map: torch.Tensor,
        local_num_experts: int,
        w13: torch.Tensor,
        w2: torch.Tensor,
        w13_scale: torch.Tensor,
        w2_scale: torch.Tensor,
        w13_zp: torch.Tensor | None,
        w2_zp: torch.Tensor | None,
        w13_g_idx: torch.Tensor,
        w2_g_idx: torch.Tensor,
        w13_sort: torch.Tensor,
        w2_sort: torch.Tensor,
    ) -> _StaticStageBuffers:
        capacity = self._static_stage_capacity
        tensors = (
            w13,
            w2,
            w13_scale,
            w2_scale,
            w13_zp,
            w2_zp,
            w13_g_idx,
            w2_g_idx,
            w13_sort,
            w2_sort,
        )
        key = (
            str(w13.device),
            capacity,
            tuple(original_map.shape),
            original_map.dtype,
            *(self._stage_signature(tensor) for tensor in tensors),
        )
        stage = _STATIC_STAGE_BUFFERS.get(key)
        if stage is not None:
            return stage

        allocated = tuple(
            self._allocate_stage_rows(tensor, capacity, local_num_experts)
            for tensor in tensors
        )
        (
            stage_w13,
            stage_w2,
            stage_w13_scale,
            stage_w2_scale,
            stage_w13_zp,
            stage_w2_zp,
            stage_w13_g_idx,
            stage_w2_g_idx,
            stage_w13_sort,
            stage_w2_sort,
        ) = allocated
        assert stage_w13 is not None and stage_w2 is not None
        assert stage_w13_scale is not None and stage_w2_scale is not None
        assert stage_w13_g_idx is not None and stage_w2_g_idx is not None
        assert stage_w13_sort is not None and stage_w2_sort is not None
        stage = _StaticStageBuffers(
            w13_weight=stage_w13,
            w2_weight=stage_w2,
            w13_scale=stage_w13_scale,
            w2_scale=stage_w2_scale,
            w13_zp=stage_w13_zp,
            w2_zp=stage_w2_zp,
            w13_g_idx=stage_w13_g_idx,
            w2_g_idx=stage_w2_g_idx,
            w13_sort=stage_w13_sort,
            w2_sort=stage_w2_sort,
            expert_map=torch.empty_like(original_map),
            capacity=capacity,
        )
        _STATIC_STAGE_BUFFERS[key] = stage
        stage_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in allocated
            if tensor is not None
        ) + stage.expert_map.numel() * stage.expert_map.element_size()
        logger.info(
            "Allocated shared WNA16 staging buffers: experts=%d bytes=%.2f MiB",
            capacity,
            stage_bytes / (1024 * 1024),
        )
        return stage

    @staticmethod
    def _gather_static_stage_rows(
        cold: torch.Tensor | None,
        hot: torch.Tensor | None,
        output: torch.Tensor | None,
        global_ids: torch.Tensor,
        original_map: torch.Tensor,
        hot_map: torch.Tensor,
    ) -> None:
        if cold is None:
            assert hot is None and output is None
            return
        assert hot is not None and output is not None
        row_size = cold[0].numel()
        num_ids = global_ids.numel()
        grid = (num_ids, triton.cdiv(row_size, 1024))
        _hybrid_gather_expert_rows_kernel[grid](
            cold,
            hot,
            output,
            global_ids,
            original_map,
            hot_map,
            num_ids=num_ids,
            global_num_experts=original_map.numel(),
            row_size=row_size,
            block_size=1024,
            num_warps=8,
        )

    @staticmethod
    def _gather_lru_rows(
        source: torch.Tensor | None,
        output: torch.Tensor | None,
        miss_local_ids: torch.Tensor,
        miss_slots: torch.Tensor,
        num_ids: int,
    ) -> None:
        if source is None:
            assert output is None
            return
        assert output is not None
        if source.ndim == 0 or output.ndim == 0:
            return
        row_size = source[0].numel()
        grid = (num_ids, triton.cdiv(row_size, 1024))
        _gather_lru_expert_rows_kernel[grid](
            source,
            output,
            miss_local_ids,
            miss_slots,
            row_size=row_size,
            num_ids=num_ids,
            block_size=1024,
            num_warps=8,
        )

    @staticmethod
    def _cuda_driver_check(result: tuple[Any, ...], operation: str) -> Any:
        from cuda.bindings import driver

        error, *values = result
        if error != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed: {error}")
        if not values:
            return None
        return values[0] if len(values) == 1 else tuple(values)

    @classmethod
    def _allocate_mixed_vmm_tensor(
        cls,
        source: torch.Tensor,
        new_to_old: torch.Tensor,
        hot_experts: int,
        host_duplicate: bool = False,
    ) -> tuple[torch.Tensor, _MixedVMMAllocation]:
        """Build one contiguous tensor with a VRAM prefix and host suffix.

        Experts are permuted hot-first, so Marlin keeps its original tensor
        geometry and expert count while the CUDA page tables select the
        physical memory tier.  CUDA VMM mappings use a 2 MiB granularity on
        Ampere; padding lives only in storage and is outside the tensor shape.

        With ``host_duplicate`` (v2 hybrid layout) the tensor carries
        ``hot_experts`` extra rows, so the geometry is:

        - rows ``[0, capacity)``: DEVICE-mapped hot prefix, the capacity
          statically-hot experts in ranking order (the LRU mirror / decode
          path reads and rotates these rows in place);
        - rows ``[capacity, capacity + N)``: HOST-mapped copy of the FULL
          permutation, i.e. row ``capacity + k`` holds the expert at
          permuted position ``k``.  Its first ``capacity`` rows duplicate
          the hot prefix, so EVERY local expert keeps a permanent
          host-backed row that the LRU can gather from and re-promote;
          evicting a hot slot is no longer lossy.

        The ``gpu_bytes`` prefix keeps the same formula (aligned UP, never
        down), so the first ``capacity`` rows are always entirely inside
        the device-mapped region.  Without ``host_duplicate`` the function
        produces exactly the arm-1 layout (N rows, no suffix copy).
        """
        from cuda.bindings import driver

        if not source.is_cuda or not source.is_contiguous() or source.ndim == 0:
            raise ValueError(
                "Mixed WNA16 VMM requires a contiguous CUDA/UVA tensor; "
                f"got device={source.device}, shape={tuple(source.shape)}"
            )
        device_index = source.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        cls._cuda_driver_check(driver.cuInit(0), "cuInit")
        device = cls._cuda_driver_check(
            driver.cuDeviceGet(device_index), "cuDeviceGet"
        )
        numa_id = cls._cuda_driver_check(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
                device,
            ),
            "cuDeviceGetAttribute(HOST_NUMA_ID)",
        )

        def allocation_prop(location_type: Any, location_id: int) -> Any:
            prop = driver.CUmemAllocationProp()
            prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
            prop.location.type = location_type
            prop.location.id = location_id
            prop.requestedHandleTypes = (
                driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
            )
            return prop

        device_prop = allocation_prop(
            driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
            device_index,
        )
        host_prop = allocation_prop(
            driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,
            numa_id,
        )
        minimum = (
            driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
        )
        device_granularity = cls._cuda_driver_check(
            driver.cuMemGetAllocationGranularity(device_prop, minimum),
            "cuMemGetAllocationGranularity(device)",
        )
        host_granularity = cls._cuda_driver_check(
            driver.cuMemGetAllocationGranularity(host_prop, minimum),
            "cuMemGetAllocationGranularity(host)",
        )
        alignment = max(device_granularity, host_granularity)
        row_bytes = source[0].numel() * source.element_size()
        total_rows = source.shape[0] + (hot_experts if host_duplicate else 0)
        original_bytes = total_rows * row_bytes
        mapped_bytes = ((original_bytes + alignment - 1) // alignment) * alignment
        gpu_data_bytes = hot_experts * row_bytes
        gpu_bytes = ((gpu_data_bytes + alignment - 1) // alignment) * alignment
        gpu_bytes = min(gpu_bytes, mapped_bytes)
        host_bytes = mapped_bytes - gpu_bytes

        address = cls._cuda_driver_check(
            driver.cuMemAddressReserve(mapped_bytes, alignment, 0, 0),
            "cuMemAddressReserve",
        )
        handles: list[Any] = []
        if gpu_bytes:
            gpu_handle = cls._cuda_driver_check(
                driver.cuMemCreate(gpu_bytes, device_prop, 0),
                "cuMemCreate(device)",
            )
            handles.append(gpu_handle)
            cls._cuda_driver_check(
                driver.cuMemMap(address, gpu_bytes, 0, gpu_handle, 0),
                "cuMemMap(device)",
            )
        if host_bytes:
            host_handle = cls._cuda_driver_check(
                driver.cuMemCreate(host_bytes, host_prop, 0),
                "cuMemCreate(host)",
            )
            handles.append(host_handle)
            cls._cuda_driver_check(
                driver.cuMemMap(
                    int(address) + gpu_bytes,
                    host_bytes,
                    0,
                    host_handle,
                    0,
                ),
                "cuMemMap(host)",
            )

        access = driver.CUmemAccessDesc()
        access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        access.location.id = device_index
        access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        cls._cuda_driver_check(
            driver.cuMemSetAccess(address, mapped_bytes, [access], 1),
            "cuMemSetAccess",
        )

        storage = torch._C._construct_storage_from_data_pointer(
            int(address), source.device, mapped_bytes
        )
        metadata = {
            "nbytes": mapped_bytes,
            "data_ptr": int(address),
            "size": (total_rows, *source.shape[1:]),
            # source is contiguous (checked above), so the row strides do
            # not depend on the row count.
            "stride": tuple(source.stride()),
            "dtype": source.dtype,
            "device": source.device,
            "storage_offset": 0,
        }
        output = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
            metadata, storage
        )
        row_size = source[0].numel()
        if host_duplicate:
            # Boot-time one-off fill of BOTH regions in a single pass:
            # rows [0, capacity) take the hot permutation (the device
            # prefix), rows [capacity, capacity + N) take the SAME full
            # permutation again (the host duplicate region).  Total copied
            # volume is N + capacity rows, the same order as the old N-row
            # fill; nothing here runs per step.
            fill_new_to_old = torch.cat([new_to_old[:hot_experts], new_to_old])
        else:
            fill_new_to_old = new_to_old
        grid = (total_rows, triton.cdiv(row_size, 1024))
        _permute_expert_rows_kernel[grid](
            source,
            output,
            fill_new_to_old,
            row_size=row_size,
            block_size=1024,
            num_warps=8,
        )
        if host_duplicate:
            # Row-content canary (reviewer F3): every map addresses the host
            # region as if it began with a copy of the device prefix.  Map
            # arithmetic asserts cannot see a fill-order mistake, so compare
            # bytes here, once per layer-rank at boot.
            prefix_rows = output[:hot_experts]
            duplicate_rows = output[hot_experts : 2 * hot_experts]
            if prefix_rows.shape[0] and not torch.equal(prefix_rows, duplicate_rows):
                raise RuntimeError(
                    "Mixed VMM host duplicate does not mirror the device prefix "
                    f"(prefix {tuple(prefix_rows.shape)} vs duplicate "
                    f"{tuple(duplicate_rows.shape)})"
                )
        # The source parameter can be replaced immediately after this returns.
        torch.cuda.synchronize(source.device)
        allocation = _MixedVMMAllocation(
            storage=storage,
            address=address,
            handles=tuple(handles),
            mapped_bytes=mapped_bytes,
            gpu_bytes=gpu_bytes,
            host_bytes=host_bytes,
            duplicate_data_bytes=hot_experts * row_bytes if host_duplicate else 0,
        )
        _MIXED_VMM_ALLOCATIONS.append(allocation)
        return output, allocation

    def maybe_init_mixed_vmm_hot_cache(self, layer: RoutedExperts) -> None:
        capacity = int(os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_SIZE", "0"))
        cache_file = os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_FILE")
        if not cache_file:
            cache_file = os.path.join(os.getcwd(), "static_hot_cache_rankings.json")
        # The allocation is backend-agnostic: every supported modular WNA16
        # layout keeps experts in dimension 0 and consumes ordinary contiguous
        # CUDA pointers.  Marlin was the first consumer, but restricting the
        # VMM hot/cold placement to Marlin prevents fair low-latency backend
        # comparisons (notably Triton and Humming) with the same memory tiering.
        if capacity <= 0 or self.wna16_backend not in (
            WNA16MoEBackend.MARLIN,
            WNA16MoEBackend.BATCHED_MARLIN,
            WNA16MoEBackend.TRITON,
            WNA16MoEBackend.HUMMING,
        ):
            return
        if "language_model.model.layers." not in self.layer_name:
            return
        if self._mixed_vmm_active:
            raise RuntimeError(f"Mixed VMM cache already initialized: {self.layer_name}")
        if not os.path.isfile(cache_file):
            raise FileNotFoundError(
                f"Mixed WNA16 VMM rankings not found: {cache_file}"
            )
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", self.layer_name)
        if match is None:
            return
        with open(cache_file) as source_file:
            rankings = json.load(source_file)
        ranked_global_ids = rankings.get(match.group(1), [])
        if not ranked_global_ids:
            return

        w13 = layer.w13_weight_packed
        w2 = layer.w2_weight_packed
        if not (
            getattr(w13, "_vllm_is_uva_offloaded", False)
            and getattr(w2, "_vllm_is_uva_offloaded", False)
        ):
            return
        local_num_experts = w13.shape[0]
        original_map = layer.expert_map
        if original_map is None:
            original_map = torch.arange(
                layer.global_num_experts, dtype=torch.int32, device=w13.device
            )
        original_map_cpu = [int(value) for value in original_map.detach().cpu().tolist()]
        hot_local_ids: list[int] = []
        # Same filtered ranking order as hot_local_ids; the hybrid mirror's
        # slot i initial occupant is hot_global_ids[i].
        hot_global_ids: list[int] = []
        seen: set[int] = set()
        for raw_global_id in ranked_global_ids:
            global_id = int(raw_global_id)
            if global_id < 0 or global_id >= len(original_map_cpu):
                continue
            local_id = original_map_cpu[global_id]
            if local_id < 0 or local_id in seen:
                continue
            hot_local_ids.append(local_id)
            hot_global_ids.append(global_id)
            seen.add(local_id)
            if len(hot_local_ids) == capacity:
                break
        if not hot_local_ids:
            return
        new_to_old_list = hot_local_ids + [
            local_id
            for local_id in range(local_num_experts)
            if local_id not in seen
        ]
        if len(new_to_old_list) != local_num_experts:
            raise RuntimeError("Mixed VMM expert permutation is not bijective")
        old_to_new = [0] * local_num_experts
        for new_id, old_id in enumerate(new_to_old_list):
            old_to_new[old_id] = new_id
        # v2 hybrid layout: every tensor gets `capacity` extra host-backed
        # duplicate rows so that EVERY local expert (hot or cold) keeps a
        # permanent host row, exactly like the golden dynamic-LRU arm where
        # cold_map[g] is the layer's own local id for all locals.  The flag
        # mirrors the gate in maybe_init_static_hot_cache, so arm-1
        # (MIXED_VMM without LRU/hybrid) keeps today's N-row layout.
        capacity = len(hot_local_ids)
        host_duplicate = bool(
            self._dynamic_lru_enabled
            and getattr(
                self,
                "_vmm_lru_hybrid",
                os.getenv("VLLM_WNA16_VMM_LRU_HYBRID", "0") == "1",
            )
        )
        total_rows = local_num_experts + (capacity if host_duplicate else 0)

        def _served_row(new_row: int) -> int:
            # Row that serves the expert at permuted position `new_row`:
            # hot rows stay in the device prefix; cold experts are served
            # from the host duplicate region at `capacity + new_row`.
            if host_duplicate and new_row >= capacity:
                return capacity + new_row
            return new_row

        new_map_cpu = [
            -1 if old_id < 0 else _served_row(old_to_new[old_id])
            for old_id in original_map_cpu
        ]
        new_map = torch.tensor(
            new_map_cpu, dtype=original_map.dtype, device=original_map.device
        )
        new_to_old = torch.tensor(
            new_to_old_list, dtype=torch.long, device=w13.device
        )
        if host_duplicate:
            # Same extension for the <64 MiB regular tensors so ALL expert
            # tensors share one row space [0, total_rows) and one cold_map
            # addresses all of them (invariant 4 of the spec, extended).
            permute_index = torch.cat([new_to_old[:capacity], new_to_old])
        else:
            permute_index = new_to_old

        tensor_names = (
            "w13_weight_packed",
            "w2_weight_packed",
            "w13_weight_scale",
            "w2_weight_scale",
            "w13_weight_zero_point",
            "w2_weight_zero_point",
            "w13_weight_g_idx",
            "w2_weight_g_idx",
            "w13_g_idx_sort_indices",
            "w2_g_idx_sort_indices",
            "w13_weight_shape",
            "w2_weight_shape",
        )
        installed_tensors: dict[str, torch.Tensor] = {}
        vmm_gpu_bytes = 0
        vmm_host_bytes = 0
        vmm_dup_bytes = 0
        regular_gpu_bytes = 0
        with torch.no_grad():
            for name in tensor_names:
                tensor = getattr(layer, name, None)
                if (
                    tensor is None
                    or tensor.ndim == 0
                    or tensor.shape[0] != local_num_experts
                ):
                    continue
                if tensor.numel() * tensor.element_size() >= 64 * 1024 * 1024:
                    reordered, allocation = self._allocate_mixed_vmm_tensor(
                        tensor,
                        new_to_old,
                        len(hot_local_ids),
                        host_duplicate=host_duplicate,
                    )
                    vmm_gpu_bytes += allocation.gpu_bytes
                    vmm_host_bytes += allocation.host_bytes
                    vmm_dup_bytes += allocation.duplicate_data_bytes
                else:
                    reordered = (
                        torch.index_select(tensor, 0, permute_index)
                        .contiguous()
                    )
                    regular_gpu_bytes += reordered.numel() * reordered.element_size()
                    if host_duplicate:
                        # The duplicated hot rows of a regular (small)
                        # tensor: capacity rows of the source geometry.
                        vmm_dup_bytes += capacity * tensor[0].numel() * tensor.element_size()
                replace_parameter(
                    layer,
                    name,
                    torch.nn.Parameter(reordered, requires_grad=False),
                )
                installed_tensors[name] = reordered
            # RoutedExperts exposes expert_map as a read-only property backed
            # by the registered _expert_map buffer.  Keep the manager in sync
            # as well so a later consumer cannot observe the pre-permutation
            # local slot numbering.
            layer._expert_map = new_map
            layer.expert_map_manager._expert_map = new_map
            layer.w13_weight = layer.w13_weight_packed
            layer.w2_weight = layer.w2_weight_packed
            if host_duplicate:
                # Extended row space: the Humming experts assert
                # w1.size(0) == humming_configs["*"].num_experts on every
                # forward (moe_problem_size) and size their grouped-permute
                # buffers from moe_config.num_local_experts, so the MAIN
                # kernel must be bound with the total row count in hybrid
                # mode.  The mirror kernel still binds `capacity` slots via
                # _make_hot_cache_kernel's own replace().  The swap is
                # permanent on purpose: the fn-ext AutoGPTQ VMM shim calls
                # self._setup_kernel(layer) a second time after re-aliasing,
                # outside this function's scope.
                self.moe = replace(self.moe, num_local_experts=total_rows)
                humming_configs = getattr(layer, "humming_configs", None)
                if humming_configs is not None:
                    layer.humming_configs = {
                        name: replace(config, num_experts=total_rows)
                        for name, config in humming_configs.items()
                    }
            self._setup_kernel(layer)
        # UVA offloading obtains its backing through PyTorch's pinned-host
        # allocator.  Replacing those parameters releases the tensors, but the
        # allocator caches the physical blocks.  Return them to the OS now so
        # the host-backed VMM suffix does not transiently double model RAM.
        del w13, w2
        torch.accelerator.empty_host_cache()
        self._mixed_vmm_active = True
        # Stash everything the VMM+LRU hybrid needs so it can mount the LRU
        # mirror on this GPU prefix without recomputing the permutation.
        self._mixed_vmm_state = {
            "original_map": original_map,
            "new_map": new_map,
            "hot_global_ids": hot_global_ids,
            "hot_local_ids": hot_local_ids,
            "new_to_old_list": new_to_old_list,
            "old_to_new": old_to_new,
            "local_num_experts": local_num_experts,
            "tensors": installed_tensors,
            # v2 hybrid additions.
            "host_duplicate": host_duplicate,
            "total_rows": total_rows,
            "vmm_gpu_bytes": vmm_gpu_bytes,
            "vmm_dup_bytes": vmm_dup_bytes,
        }
        logger.info(
            "Mixed WNA16 VMM cache: layer=%s hot=%d gpu=%.2f MiB "
            "host=%.2f MiB regular_gpu=%.2f MiB",
            self.layer_name,
            len(hot_local_ids),
            vmm_gpu_bytes / (1024 * 1024),
            vmm_host_bytes / (1024 * 1024),
            regular_gpu_bytes / (1024 * 1024),
        )

    def _make_hot_cache_kernel(
        self,
        layer: RoutedExperts,
        mirror_w13_scale: torch.Tensor | None,
        mirror_w2_scale: torch.Tensor | None,
        mirror_w13_zp: torch.Tensor | None,
        mirror_w2_zp: torch.Tensor | None,
        kernel_w13_scale: torch.Tensor | None,
        kernel_w2_scale: torch.Tensor | None,
        kernel_w13_zp: torch.Tensor | None,
        kernel_w2_zp: torch.Tensor | None,
        kernel_w13_g_idx: torch.Tensor | None,
        kernel_w2_g_idx: torch.Tensor | None,
        kernel_w13_sort: torch.Tensor | None,
        kernel_w2_sort: torch.Tensor | None,
        num_slots: int,
    ) -> Any:
        """Build the second MoE kernel invocation that evaluates only the
        mirror rows (``hot_map``).

        Shared verbatim between the golden static mirror (mirror tensors are
        dedicated VRAM copies, possibly with staged scale/zp/g_idx/sort
        buffers) and the VMM+LRU hybrid (mirror tensors are zero-copy VMM
        prefix views and every ``kernel_*`` argument equals its ``mirror_*``
        counterpart).  The HUMMING branch keeps the original pairing: the
        quant config points at the mirror scale/zero-point tensors while the
        kernel receives the (possibly staged) g_idx/sort tensors.
        """
        cache_moe_config = self.moe
        if self.wna16_backend == WNA16MoEBackend.HUMMING:
            from vllm.model_executor.layers.quantization.utils.humming_utils import (
                get_humming_moe_quant_config,
            )

            cache_humming_configs = {
                name: replace(config, num_experts=num_slots)
                for name, config in layer.humming_configs.items()
            }
            cache_quant_config = get_humming_moe_quant_config(
                layer,
                humming_configs=cache_humming_configs,
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
            )
            cache_quant_config = replace(
                cache_quant_config,
                _w1=replace(
                    cache_quant_config._w1,
                    scale=mirror_w13_scale,
                    zp=mirror_w13_zp,
                ),
                _w2=replace(
                    cache_quant_config._w2,
                    scale=mirror_w2_scale,
                    zp=mirror_w2_zp,
                ),
            )
            cache_moe_config = replace(
                self.moe, num_local_experts=num_slots
            )
        else:
            cache_quant_config = make_wna16_moe_quant_config(
                w1_scale=kernel_w13_scale,
                w2_scale=kernel_w2_scale,
                group_size=self.group_size,
                num_bits=self.num_bits,
                w1_zp=kernel_w13_zp,
                w2_zp=kernel_w2_zp,
                gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
                gemm1_alpha=getattr(layer, "swiglu_alpha", None),
                gemm1_beta=getattr(layer, "swiglu_beta", None),
            )
        assert self.experts_cls is not None
        return make_wna16_moe_kernel(
            moe_quant_config=cache_quant_config,
            moe_config=cache_moe_config,
            experts_cls=self.experts_cls,
            backend=self.wna16_backend,
            routing_tables=layer._expert_routing_tables(),
            w13_g_idx=kernel_w13_g_idx,
            w2_g_idx=kernel_w2_g_idx,
            w13_g_idx_sort_indices=kernel_w13_sort,
            w2_g_idx_sort_indices=kernel_w2_sort,
            is_k_full=self.is_k_full,
        )

    def maybe_init_mixed_vmm_lru_hybrid(self, layer: RoutedExperts) -> None:
        """One tier, two access paths: keep the VMM contiguous expert view
        for prefill and mount the dynamic LRU mirror on its GPU prefix for
        decode.

        The golden mirror and the VMM prefix are built from the same ranking
        in the same order, so the mirror is a zero-copy ``[:capacity]`` view
        of every permuted tensor and adds no VRAM.  Consequently ``cold_map``
        points at VMM row indices (the host-backed copy of each expert) and
        the prefill-size path routes through the dynamic ``cache.rot_map``
        (see apply()), which starts as exactly the installed permutation map
        in the extended row space.

        v2 LAYOUT (the hot-eviction fix): the VMM tensors carry
        ``capacity + local_num_experts`` rows — the device-mapped hot prefix
        ``[0, capacity)`` plus a host-mapped copy of the FULL permutation at
        ``[capacity, capacity + N)``.  Every local expert therefore keeps a
        permanent host-backed row (``cold_map[g] >= capacity`` for ALL local
        g, exactly golden's "cold_map = local id for every local expert"),
        so evicting a statically-hot slot writes over a DUPLICATE and the
        victim can always be re-promoted: the rotation is lossless for hot
        experts too (v1 dropped them from rot_map/hot_map with -1 and lost
        their only copy — see docs/hybrid-implementation-notes.md §5).
        """
        if self._static_hot_cache is not None:
            raise RuntimeError(
                f"Static expert cache already initialized: {self.layer_name}"
            )
        # The VMM initializer leaves no state behind when it declines this
        # layer (no rankings, non-UVA tensors, unsupported backend); in that
        # case the layer keeps plain arm-1 behaviour with no cache at all.
        self._mixed_vmm_state = None
        # Instance lookup on purpose: on the live lane self is an AutoGPTQ
        # MoE method whose shim must also run its name re-alias after VMM.
        self.maybe_init_mixed_vmm_hot_cache(layer)
        state = getattr(self, "_mixed_vmm_state", None)
        if state is None:
            return
        if not state["host_duplicate"]:
            # Unreachable through the gate (this function is only entered
            # with the hybrid flag on, which is the same condition the VMM
            # init used to pick the duplicate layout).  Fail loudly rather
            # than mount an LRU over v1's lossy rows.
            raise RuntimeError(
                "VMM+LRU hybrid requires the v2 duplicate layout; "
                "maybe_init_mixed_vmm_hot_cache declined to enable it"
            )

        original_map = state["original_map"]
        hot_global_ids = list(state["hot_global_ids"])
        hot_local_ids = list(state["hot_local_ids"])
        old_to_new = state["old_to_new"]
        local_num_experts = int(state["local_num_experts"])
        capacity = len(hot_local_ids)
        total_rows = int(state["total_rows"])
        assert total_rows == local_num_experts + capacity
        if capacity <= 0:
            return

        checkpoint_names = (
            ("w13_weight", "w13_weight_packed"),
            ("w2_weight", "w2_weight_packed"),
            ("w13_scale", "w13_weight_scale"),
            ("w2_scale", "w2_weight_scale"),
            ("w13_zp", "w13_weight_zero_point"),
            ("w2_zp", "w2_weight_zero_point"),
            ("w13_g_idx", "w13_weight_g_idx"),
            ("w2_g_idx", "w2_weight_g_idx"),
            ("w13_sort", "w13_g_idx_sort_indices"),
            ("w2_sort", "w2_g_idx_sort_indices"),
        )
        with torch.no_grad():
            mirrors: dict[str, torch.Tensor | None] = {}
            for cache_name, checkpoint_name in checkpoint_names:
                tensor = state["tensors"].get(checkpoint_name)
                # Non-expert-indexed or absent tensors (e.g. zero points on a
                # symmetric checkpoint) stay None exactly like golden's
                # _cache_expert_rows pass-through.
                mirrors[cache_name] = (
                    None
                    if tensor is None
                    or tensor.ndim == 0
                    or tensor.shape[0] != total_rows
                    else tensor[:capacity]
                )
            assert mirrors["w13_weight"] is not None
            assert mirrors["w2_weight"] is not None
            assert mirrors["w13_scale"] is not None
            assert mirrors["w2_scale"] is not None
            if self.is_marlin:
                assert mirrors["w13_g_idx"] is not None
                assert mirrors["w2_g_idx"] is not None
                assert mirrors["w13_sort"] is not None
                assert mirrors["w2_sort"] is not None

            original_map_cpu = [int(v) for v in original_map.detach().cpu().tolist()]
            hot_map = torch.full_like(original_map, -1)
            cold_map = torch.full_like(original_map, -1)
            for global_id, local_id in enumerate(original_map_cpu):
                if local_id >= 0:
                    # v2: EVERY local expert (hot or cold) gets the row
                    # index of its permanent host-backed copy, the exact
                    # analogue of golden's cold_map = local-id-for-all-
                    # locals.  Always >= capacity, never a prefix row:
                    # gather sources stay disjoint from slots (invariant 1)
                    # AND stay valid in the extended row space.
                    cold_map[global_id] = capacity + old_to_new[local_id]
            for slot, global_id in enumerate(hot_global_ids):
                hot_map[global_id] = slot

            cache_kernel = CompressedTensorsWNA16MoEMethod._make_hot_cache_kernel(
                self,
                layer,
                mirrors["w13_scale"],
                mirrors["w2_scale"],
                mirrors["w13_zp"],
                mirrors["w2_zp"],
                mirrors["w13_scale"],
                mirrors["w2_scale"],
                mirrors["w13_zp"],
                mirrors["w2_zp"],
                mirrors["w13_g_idx"],
                mirrors["w2_g_idx"],
                mirrors["w13_sort"],
                mirrors["w2_sort"],
                capacity,
            )

            device = original_map.device
            miss_rows = (
                self._static_hot_cache_max_tokens * self.moe.experts_per_token
            )
            rot_map = state["new_map"].clone()
            self._static_hot_cache = _StaticHotExpertCache(
                w13_weight=mirrors["w13_weight"],
                w2_weight=mirrors["w2_weight"],
                w13_scale=mirrors["w13_scale"],
                w2_scale=mirrors["w2_scale"],
                w13_zp=mirrors["w13_zp"],
                w2_zp=mirrors["w2_zp"],
                w13_g_idx=mirrors["w13_g_idx"],
                w2_g_idx=mirrors["w2_g_idx"],
                w13_sort=mirrors["w13_sort"],
                w2_sort=mirrors["w2_sort"],
                hot_map=hot_map,
                cold_map=cold_map,
                kernel=cache_kernel,
                global_ids=tuple(hot_global_ids),
                stage=None,
                dynamic_lru=True,
                slot_global_ids=torch.tensor(
                    hot_global_ids, dtype=torch.int32, device=device
                ),
                slot_ages=torch.arange(
                    capacity, 0, -1, dtype=torch.int32, device=device
                ),
                clock=torch.tensor([capacity], dtype=torch.int32, device=device),
                miss_local_ids=torch.empty(
                    miss_rows, dtype=torch.int32, device=device
                ),
                miss_slots=torch.empty(
                    miss_rows, dtype=torch.int32, device=device
                ),
                rot_map=rot_map,
                lru_dummy=torch.zeros(1, dtype=torch.int32, device=device),
            )

            # Invariants checked once at install time (spec section 4, v2):
            # 1. gather sources come from cold_map values and destinations
            #    are slots: sources are ALWAYS in the host-backed duplicate
            #    region [capacity, capacity + N) for every local expert
            #    (hot included), destinations are < capacity, so no gather
            #    can ever alias a live prefix row.
            cold_cpu = cold_map.detach().cpu().tolist()
            assert all(
                value == -1 or capacity <= value < capacity + local_num_experts
                for value in cold_cpu
            )
            # 1b. uniformity: no local expert is left without a host row
            #     (the v1 flaw: hot experts carried cold_map = -1).
            assert all(
                cold_cpu[g] >= capacity
                for g, local in enumerate(original_map_cpu)
                if local >= 0
            )
            # 3. the mirror view content equals the ranking order equals the
            #    slot_global_ids initial state: rank i expert sits at row i
            #    of the device prefix AND at row capacity + i of the host
            #    duplicate region.
            assert all(
                old_to_new[local_id] == slot
                for slot, local_id in enumerate(hot_local_ids)
            )
            # 5. the prefill map starts as exactly the installed VMM map
            #    (prefix rows for the hot experts, host rows for the cold
            #    experts), and every row it names exists in the tensor.
            assert rot_map.equal(state["new_map"])
            rot_cpu = rot_map.detach().cpu().tolist()
            assert all(value == -1 or 0 <= value < total_rows for value in rot_cpu)
            # All maps share the installed expert_map's device and dtype.
            assert (
                hot_map.dtype == cold_map.dtype == rot_map.dtype
                == original_map.dtype
            )

        logger.info(
            "[hybrid] VMM+LRU cache: layer=%s capacity=%d tensor_rows=%d "
            "device_prefix=%.2f MiB host_duplicate=%.2f MiB ids=%s",
            self.layer_name,
            capacity,
            total_rows,
            int(state["vmm_gpu_bytes"]) / (1024 * 1024),
            int(state["vmm_dup_bytes"]) / (1024 * 1024),
            hot_global_ids,
        )

    def maybe_init_static_hot_cache(self, layer: RoutedExperts) -> None:
        """Mirror configured hot experts for fully UVA-resident INT4 layers.

        The JSON file named by ``VLLM_WNA16_STATIC_HOT_CACHE_FILE`` maps a
        layer number (as a string) to global expert IDs ordered hottest first.
        Each EP rank filters that order to its local experts and mirrors up to
        ``VLLM_WNA16_STATIC_HOT_CACHE_SIZE`` rows.
        """
        if self._mixed_vmm_enabled and not self._dynamic_lru_enabled:
            self.maybe_init_mixed_vmm_hot_cache(layer)
            return
        if (
            self._mixed_vmm_enabled
            and self._dynamic_lru_enabled
            and getattr(
                self,
                "_vmm_lru_hybrid",
                os.getenv("VLLM_WNA16_VMM_LRU_HYBRID", "0") == "1",
            )
        ):
            # Class-function call because self may be a borrowed AutoGPTQ MoE
            # method (fn-ext auto_gptq shim) whose class carries neither this
            # method nor the __init__ flag; the env fallback mirrors __init__
            # for that case.  With the hybrid flag off, MIXED_VMM=1 + LRU=1
            # keeps falling through to the golden mirror path exactly as
            # today.
            CompressedTensorsWNA16MoEMethod.maybe_init_mixed_vmm_lru_hybrid(
                self, layer
            )
            return
        capacity = int(os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_SIZE", "0"))
        cache_file = os.getenv("VLLM_WNA16_STATIC_HOT_CACHE_FILE")
        if not cache_file:
            # vLLM removes *_FILE variables when it spawns engine workers.  The
            # launcher stages this non-secret input at a deterministic path.
            cache_file = os.path.join(os.getcwd(), "static_hot_cache_rankings.json")
        if capacity <= 0 or (
            not self.is_marlin and self.wna16_backend != WNA16MoEBackend.HUMMING
        ):
            return
        if not os.path.isfile(cache_file):
            raise FileNotFoundError(
                f"Static WNA16 hot-cache rankings not found: {cache_file}"
            )
        # The target model is nested under language_model; the separate MTP
        # draft has its own router distribution and must not reuse these IDs.
        if "language_model.model.layers." not in self.layer_name:
            return
        if self._static_hot_cache is not None:
            raise RuntimeError(
                f"Static expert cache already initialized: {self.layer_name}"
            )

        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", self.layer_name)
        if match is None:
            return
        layer_id = match.group(1)
        with open(cache_file) as source:
            rankings = json.load(source)
        ranked_global_ids = rankings.get(layer_id, [])
        if not ranked_global_ids:
            return

        w13 = layer.w13_weight_packed
        w2 = layer.w2_weight_packed
        if not (
            getattr(w13, "_vllm_is_uva_offloaded", False)
            and getattr(w2, "_vllm_is_uva_offloaded", False)
        ):
            return

        original_map = layer.expert_map
        if original_map is None:
            original_map = torch.arange(
                layer.global_num_experts, dtype=torch.int32, device=w13.device
            )
        original_map_cpu = original_map.detach().cpu().tolist()

        global_ids: list[int] = []
        local_ids_list: list[int] = []
        for raw_id in ranked_global_ids:
            global_id = int(raw_id)
            if global_id < 0 or global_id >= len(original_map_cpu):
                continue
            local_id = int(original_map_cpu[global_id])
            if local_id < 0:
                continue
            global_ids.append(global_id)
            local_ids_list.append(local_id)
            if len(global_ids) == capacity:
                break
        if not global_ids:
            return

        local_ids = torch.tensor(local_ids_list, dtype=torch.long, device=w13.device)
        local_num_experts = w13.shape[0]
        with torch.no_grad():
            hot_w13 = self._cache_expert_rows(w13, local_ids, local_num_experts)
            hot_w2 = self._cache_expert_rows(w2, local_ids, local_num_experts)
            hot_w13_scale = self._cache_expert_rows(
                layer.w13_weight_scale, local_ids, local_num_experts
            )
            hot_w2_scale = self._cache_expert_rows(
                layer.w2_weight_scale, local_ids, local_num_experts
            )
            hot_w13_zp = self._cache_expert_rows(
                getattr(layer, "w13_weight_zero_point", None),
                local_ids,
                local_num_experts,
            )
            hot_w2_zp = self._cache_expert_rows(
                getattr(layer, "w2_weight_zero_point", None),
                local_ids,
                local_num_experts,
            )
            hot_w13_g_idx = self._cache_expert_rows(
                getattr(layer, "w13_weight_g_idx", None),
                local_ids,
                local_num_experts,
            )
            hot_w2_g_idx = self._cache_expert_rows(
                getattr(layer, "w2_weight_g_idx", None),
                local_ids,
                local_num_experts,
            )
            hot_w13_sort = self._cache_expert_rows(
                getattr(layer, "w13_g_idx_sort_indices", None),
                local_ids,
                local_num_experts,
            )
            hot_w2_sort = self._cache_expert_rows(
                getattr(layer, "w2_g_idx_sort_indices", None),
                local_ids,
                local_num_experts,
            )

            assert hot_w13 is not None and hot_w2 is not None
            assert hot_w13_scale is not None and hot_w2_scale is not None
            if self.is_marlin:
                assert hot_w13_g_idx is not None and hot_w2_g_idx is not None
                assert hot_w13_sort is not None and hot_w2_sort is not None

            hot_map = torch.full_like(original_map, -1)
            cold_map = original_map.clone()
            for slot, global_id in enumerate(global_ids):
                hot_map[global_id] = slot
                if not self._dynamic_lru_enabled:
                    cold_map[global_id] = -1

            stage = None
            kernel_w13_scale = hot_w13_scale
            kernel_w2_scale = hot_w2_scale
            kernel_w13_zp = hot_w13_zp
            kernel_w2_zp = hot_w2_zp
            kernel_w13_g_idx = hot_w13_g_idx
            kernel_w2_g_idx = hot_w2_g_idx
            kernel_w13_sort = hot_w13_sort
            kernel_w2_sort = hot_w2_sort
            if self._static_stage_enabled:
                stage = self._get_static_stage_buffers(
                    original_map,
                    local_num_experts,
                    w13,
                    w2,
                    layer.w13_weight_scale,
                    layer.w2_weight_scale,
                    getattr(layer, "w13_weight_zero_point", None),
                    getattr(layer, "w2_weight_zero_point", None),
                    layer.w13_weight_g_idx,
                    layer.w2_weight_g_idx,
                    layer.w13_g_idx_sort_indices,
                    layer.w2_g_idx_sort_indices,
                )
                kernel_w13_scale = stage.w13_scale
                kernel_w2_scale = stage.w2_scale
                kernel_w13_zp = stage.w13_zp
                kernel_w2_zp = stage.w2_zp
                kernel_w13_g_idx = stage.w13_g_idx
                kernel_w2_g_idx = stage.w2_g_idx
                kernel_w13_sort = stage.w13_sort
                kernel_w2_sort = stage.w2_sort

            # Same construction as before, lifted verbatim into the helper
            # shared with the hybrid (mirror_* == hot_*, kernel_* == the
            # staged-or-hot tensors chosen above).  Called as a class
            # function so it also resolves on a borrowed AutoGPTQ self.
            cache_kernel = CompressedTensorsWNA16MoEMethod._make_hot_cache_kernel(
                self,
                layer,
                hot_w13_scale,
                hot_w2_scale,
                hot_w13_zp,
                hot_w2_zp,
                kernel_w13_scale,
                kernel_w2_scale,
                kernel_w13_zp,
                kernel_w2_zp,
                kernel_w13_g_idx,
                kernel_w2_g_idx,
                kernel_w13_sort,
                kernel_w2_sort,
                len(global_ids),
            )

        self._static_hot_cache = _StaticHotExpertCache(
            w13_weight=hot_w13,
            w2_weight=hot_w2,
            w13_scale=hot_w13_scale,
            w2_scale=hot_w2_scale,
            w13_zp=hot_w13_zp,
            w2_zp=hot_w2_zp,
            w13_g_idx=hot_w13_g_idx,
            w2_g_idx=hot_w2_g_idx,
            w13_sort=hot_w13_sort,
            w2_sort=hot_w2_sort,
            hot_map=hot_map,
            cold_map=cold_map,
            kernel=cache_kernel,
            global_ids=tuple(global_ids),
            stage=stage,
            dynamic_lru=self._dynamic_lru_enabled,
            slot_global_ids=(
                torch.tensor(
                    global_ids, dtype=torch.int32, device=original_map.device
                )
                if self._dynamic_lru_enabled
                else None
            ),
            slot_ages=(
                torch.arange(
                    len(global_ids),
                    0,
                    -1,
                    dtype=torch.int32,
                    device=original_map.device,
                )
                if self._dynamic_lru_enabled
                else None
            ),
            clock=(
                torch.tensor(
                    [len(global_ids)],
                    dtype=torch.int32,
                    device=original_map.device,
                )
                if self._dynamic_lru_enabled
                else None
            ),
            miss_local_ids=(
                torch.empty(
                    self._static_hot_cache_max_tokens * self.moe.experts_per_token,
                    dtype=torch.int32,
                    device=original_map.device,
                )
                if self._dynamic_lru_enabled
                else None
            ),
            miss_slots=(
                torch.empty(
                    self._static_hot_cache_max_tokens * self.moe.experts_per_token,
                    dtype=torch.int32,
                    device=original_map.device,
                )
                if self._dynamic_lru_enabled
                else None
            ),
            lru_dummy=(
                torch.zeros(1, dtype=torch.int32, device=original_map.device)
                if self._dynamic_lru_enabled
                else None
            ),
        )
        cache_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (
                hot_w13,
                hot_w2,
                hot_w13_scale,
                hot_w2_scale,
                hot_w13_zp,
                hot_w2_zp,
                hot_w13_g_idx,
                hot_w2_g_idx,
                hot_w13_sort,
                hot_w2_sort,
            )
            if tensor is not None and tensor.device.type != "cpu"
        )
        logger.info(
            "Static WNA16 hot cache: layer=%s experts=%d bytes=%.2f MiB "
            "staging=%s dynamic_lru=%s ids=%s",
            self.layer_name,
            len(global_ids),
            cache_bytes / (1024 * 1024),
            stage is not None,
            self._dynamic_lru_enabled,
            global_ids,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Process weights using the shared oracle infrastructure
        converted = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_backend,
            layer=layer,
            quant_config=self.weight_quant,
            input_dtype=self.input_dtype,
            w13=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            w13_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w13_g_idx=layer.w13_weight_g_idx,
            w2_g_idx=layer.w2_weight_g_idx,
            w13_qzeros=getattr(layer, "w13_weight_zero_point", None),
            w2_qzeros=getattr(layer, "w2_weight_zero_point", None),
        )

        if converted is None:
            if self.wna16_backend == WNA16MoEBackend.HUMMING:
                # device_loading_context restores/re-offloads replacement
                # parameters by their original checkpoint names. Humming's
                # converter calls its packed tensor simply ``weight``; retain
                # ``weight_packed`` until the post-load mixed-VMM initializer
                # so the transformed experts are put back in UVA memory.
                for humming_name, checkpoint_name in (
                    ("w13_weight", "w13_weight_packed"),
                    ("w2_weight", "w2_weight_packed"),
                ):
                    parameter = getattr(layer, humming_name)
                    delattr(layer, humming_name)
                    layer.register_parameter(checkpoint_name, parameter)
            self._setup_kernel(layer)
            return

        (
            w13_qweight,
            w2_qweight,
            w13_scales,
            w2_scales,
            w13_g_idx_processed,
            w2_g_idx_processed,
            w13_g_idx_sort_indices,
            w2_g_idx_sort_indices,
            w13_qzeros,
            w2_qzeros,
            w13_input_global_scale,
            w2_input_global_scale,
            _,  # w13_bias
            _,  # w2_bias
        ) = converted

        # Replace common parameters
        replace_parameter(layer, "w13_weight_packed", w13_qweight)
        replace_parameter(layer, "w2_weight_packed", w2_qweight)
        replace_parameter(layer, "w13_weight_scale", w13_scales)
        replace_parameter(layer, "w2_weight_scale", w2_scales)

        # CPU fused_experts_cpu requires zero points even for symmetric quant
        if not self.symmetric or self.wna16_backend == WNA16MoEBackend.CPU:
            assert w13_qzeros is not None and w2_qzeros is not None
            replace_parameter(layer, "w13_weight_zero_point", w13_qzeros)
            replace_parameter(layer, "w2_weight_zero_point", w2_qzeros)

        # Marlin-specific parameters (not needed for Flashinfer)
        if self.is_marlin:
            if w13_g_idx_processed is not None:
                replace_parameter(layer, "w13_weight_g_idx", w13_g_idx_processed)
            if w2_g_idx_processed is not None:
                replace_parameter(layer, "w2_weight_g_idx", w2_g_idx_processed)
            if w13_g_idx_sort_indices is not None:
                replace_parameter(
                    layer, "w13_g_idx_sort_indices", w13_g_idx_sort_indices
                )
            if w2_g_idx_sort_indices is not None:
                replace_parameter(layer, "w2_g_idx_sort_indices", w2_g_idx_sort_indices)

            # Register input global scales if present
            if w13_input_global_scale is not None:
                layer.register_parameter(
                    "w13_input_global_scale",
                    torch.nn.Parameter(w13_input_global_scale, requires_grad=False),
                )
            if w2_input_global_scale is not None:
                layer.register_parameter(
                    "w2_input_global_scale",
                    torch.nn.Parameter(w2_input_global_scale, requires_grad=False),
                )

            # Marlin workspace — only needed for Marlin-family backends, not emulation.
            if (
                self.experts_cls is not None
                and issubclass(self.experts_cls, FusedMoEExpertsModular)
                and self.wna16_backend != WNA16MoEBackend.EMULATION
            ):
                layer.workspace = marlin_make_workspace_new(
                    layer.w13_weight_g_idx.device,
                    4,
                    existing=getattr(layer, "workspace", None),
                )

        # Alias packed weights to w13_weight/w2_weight for the modular kernel interface
        layer.w13_weight = layer.w13_weight_packed
        layer.w2_weight = layer.w2_weight_packed

        self._setup_kernel(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return make_wna16_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            group_size=self.group_size,
            num_bits=self.num_bits,
            w1_zp=getattr(layer, "w13_weight_zero_point", None),
            w2_zp=getattr(layer, "w2_weight_zero_point", None),
            gemm1_clamp_limit=getattr(layer, "swiglu_limit", None),
            gemm1_alpha=getattr(layer, "swiglu_alpha", None),
            gemm1_beta=getattr(layer, "swiglu_beta", None),
        )

    def apply_monolithic(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert self.is_monolithic
        assert self.moe_kernel is not None
        w13_weight = getattr(layer, "w13_weight", layer.w13_weight_packed)
        w2_weight = getattr(layer, "w2_weight", layer.w2_weight_packed)
        return self.moe_kernel.apply_monolithic(
            x,
            w13_weight,
            w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        # Humming's converted parameters are restored by the offload context
        # under their checkpoint names.  The convenience aliases are not
        # guaranteed to survive later model-level post-load cleanup.
        w13_weight = getattr(layer, "w13_weight", layer.w13_weight_packed)
        w2_weight = getattr(layer, "w2_weight", layer.w2_weight_packed)
        cache = self._static_hot_cache
        # Hybrid: rotation moves experts between the VMM GPU prefix and the
        # host-backed suffix, so the prefill-size path must follow the
        # dynamic rot_map instead of the static permutation map.  Golden
        # caches have no rot_map and keep today's behaviour exactly.
        plain_map = layer.expert_map
        if cache is not None and cache.rot_map is not None:
            plain_map = cache.rot_map
        if cache is None or x.shape[0] > self._static_hot_cache_max_tokens:
            return self.moe_kernel.apply(
                x,
                w13_weight,
                w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=plain_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        if cache.dynamic_lru:
            assert cache.slot_global_ids is not None
            assert cache.slot_ages is not None
            assert cache.clock is not None
            assert cache.miss_local_ids is not None
            assert cache.miss_slots is not None
            global_ids = topk_ids.reshape(-1)
            num_ids = global_ids.numel()
            capacity = len(cache.global_ids)
            if num_ids > cache.miss_local_ids.numel() or num_ids > capacity:
                return self.moe_kernel.apply(
                    x,
                    w13_weight,
                    w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    expert_map=plain_map,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    shared_experts=shared_experts,
                    shared_experts_input=shared_experts_input,
                )
            hybrid_mode = cache.rot_map is not None
            _update_lru_expert_map_kernel[(1,)](
                global_ids,
                cache.cold_map,
                cache.hot_map,
                cache.slot_global_ids,
                cache.slot_ages,
                cache.clock,
                cache.miss_local_ids,
                cache.miss_slots,
                # Golden passes the shared 1-element dummy twice and
                # HYBRID=False, which compiles the rot_map branches out of
                # the kernel entirely (behaviour unchanged).
                cache.rot_map if hybrid_mode else cache.lru_dummy,
                cache.lru_dummy,
                HYBRID=hybrid_mode,
                num_ids=num_ids,
                global_num_experts=cache.hot_map.numel(),
                capacity=capacity,
                id_block=triton.next_power_of_2(num_ids),
                capacity_block=triton.next_power_of_2(capacity),
                num_warps=4,
            )
            for source, output in (
                (w13_weight, cache.w13_weight),
                (w2_weight, cache.w2_weight),
                (layer.w13_weight_scale, cache.w13_scale),
                (layer.w2_weight_scale, cache.w2_scale),
                (
                    getattr(layer, "w13_weight_zero_point", None),
                    cache.w13_zp,
                ),
                (
                    getattr(layer, "w2_weight_zero_point", None),
                    cache.w2_zp,
                ),
                (getattr(layer, "w13_weight_g_idx", None), cache.w13_g_idx),
                (getattr(layer, "w2_weight_g_idx", None), cache.w2_g_idx),
                (
                    getattr(layer, "w13_g_idx_sort_indices", None),
                    cache.w13_sort,
                ),
                (
                    getattr(layer, "w2_g_idx_sort_indices", None),
                    cache.w2_sort,
                ),
            ):
                self._gather_lru_rows(
                    source,
                    output,
                    cache.miss_local_ids,
                    cache.miss_slots,
                    num_ids,
                )
            return cache.kernel.apply(
                x,
                cache.w13_weight,
                cache.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=cache.hot_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        stage = cache.stage
        if stage is not None:
            global_ids = topk_ids.reshape(-1)
            num_ids = global_ids.numel()
            if num_ids > stage.capacity:
                return self.moe_kernel.apply(
                    x,
                    w13_weight,
                    w2_weight,
                    topk_weights=topk_weights,
                    topk_ids=topk_ids,
                    activation=layer.activation,
                    global_num_experts=layer.global_num_experts,
                    expert_map=plain_map,
                    apply_router_weight_on_input=layer.apply_router_weight_on_input,
                    shared_experts=shared_experts,
                    shared_experts_input=shared_experts_input,
                )
            original_map = layer.expert_map
            assert original_map is not None
            stage.expert_map.fill_(-1)
            _prepare_static_stage_map_kernel[(triton.cdiv(num_ids, 128),)](
                global_ids,
                original_map,
                stage.expert_map,
                num_ids=num_ids,
                global_num_experts=original_map.numel(),
                block_size=128,
                num_warps=4,
            )
            for cold, hot, output in (
                (w13_weight, cache.w13_weight, stage.w13_weight),
                (w2_weight, cache.w2_weight, stage.w2_weight),
                (layer.w13_weight_scale, cache.w13_scale, stage.w13_scale),
                (layer.w2_weight_scale, cache.w2_scale, stage.w2_scale),
                (
                    getattr(layer, "w13_weight_zero_point", None),
                    cache.w13_zp,
                    stage.w13_zp,
                ),
                (
                    getattr(layer, "w2_weight_zero_point", None),
                    cache.w2_zp,
                    stage.w2_zp,
                ),
                (layer.w13_weight_g_idx, cache.w13_g_idx, stage.w13_g_idx),
                (layer.w2_weight_g_idx, cache.w2_g_idx, stage.w2_g_idx),
                (
                    layer.w13_g_idx_sort_indices,
                    cache.w13_sort,
                    stage.w13_sort,
                ),
                (layer.w2_g_idx_sort_indices, cache.w2_sort, stage.w2_sort),
            ):
                self._gather_static_stage_rows(
                    cold,
                    hot,
                    output,
                    global_ids,
                    original_map,
                    cache.hot_map,
                )
            return cache.kernel.apply(
                x,
                stage.w13_weight,
                stage.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                activation=layer.activation,
                global_num_experts=layer.global_num_experts,
                expert_map=stage.expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                shared_experts=shared_experts,
                shared_experts_input=shared_experts_input,
            )

        cold_out = self.moe_kernel.apply(
            x,
            w13_weight,
            w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=plain_map if cache.rot_map is not None else cache.cold_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=shared_experts,
            shared_experts_input=shared_experts_input,
        )
        hot_out = cache.kernel.apply(
            x,
            cache.w13_weight,
            cache.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=cache.hot_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts=None,
            shared_experts_input=None,
        )
        return cold_out + hot_out

    @property
    def supports_eplb(self) -> bool:
        return self.wna16_backend == WNA16MoEBackend.TRITON
