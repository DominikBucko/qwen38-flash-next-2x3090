#!/usr/bin/env python3
"""Build a lightweight Qwen3.8 MTP checkpoint with routed experts in INT4.

All non-MTP-expert tensors remain symlinks to the hybrid target checkpoint.
Only the standalone draft model uses this view; the target checkpoint is not
modified.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
from compressed_tensors.compressors.pack_quantized import (
    PackedQuantizationCompressor,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
)
from compressed_tensors.quantization.utils import compute_dynamic_scales_and_zp
from safetensors import safe_open
from safetensors.torch import save_file


RAW_GATE_UP = "mtp.layers.0.mlp.experts.gate_up_proj"
RAW_DOWN = "mtp.layers.0.mlp.experts.down_proj"
OUTPUT_FILE = "mtp-routed-experts-int4.safetensors"


def tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def is_draft_weight(name: str) -> bool:
    """Mirror the draft model's checkpoint-name admission rules."""
    for checkpoint_prefix in ("model.language_model.", "language_model."):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break
    if name.startswith("embed_tokens."):
        return True
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith(("mtp.shared_head.head.", "model.shared_head.head.")):
        return True
    if name.startswith(("shared_head.head.", "model.lm_head.")):
        return True
    if name.startswith("mtp."):
        return True
    return name.startswith(("model.embed_tokens.", "lm_head."))


def compress_weight(
    weight: torch.Tensor,
    args: QuantizationArgs,
    scheme: QuantizationScheme,
) -> dict[str, torch.Tensor]:
    weight = weight.contiguous()
    scale, zero_point = compute_dynamic_scales_and_zp(
        weight,
        args,
        module=None,
    )
    return PackedQuantizationCompressor.compress(
        {
            "weight": weight,
            "weight_scale": scale,
            "weight_zero_point": zero_point,
        },
        scheme,
    )


def add_compressed(
    output: dict[str, torch.Tensor],
    prefix: str,
    weight: torch.Tensor,
    args: QuantizationArgs,
    scheme: QuantizationScheme,
) -> None:
    for suffix, tensor in compress_weight(weight, args, scheme).items():
        output[f"{prefix}.{suffix}"] = tensor.cpu().contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--group-size", type=int, choices=(32, 64, 128), default=128)
    parsed = parser.parse_args()

    source = parsed.source.resolve()
    destination = parsed.destination.resolve()
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")

    torch.set_num_threads(parsed.threads)
    index_path = source / "model.safetensors.index.json"
    config_path = source / "config.json"
    index = json.loads(index_path.read_text())
    config = json.loads(config_path.read_text())
    weight_map: dict[str, str] = index["weight_map"]

    for key in (RAW_GATE_UP, RAW_DOWN):
        if key not in weight_map:
            raise SystemExit(f"missing expected source tensor: {key}")

    quant_args = QuantizationArgs(
        num_bits=4,
        type="int",
        strategy="group",
        group_size=parsed.group_size,
        symmetric=True,
        dynamic=False,
    )
    quant_scheme = QuantizationScheme(
        targets=["RoutedExperts"],
        weights=quant_args,
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-building-", dir=destination.parent)
    )
    output: dict[str, torch.Tensor] = {}
    raw_nbytes = 0

    try:
        gate_up_file = source / weight_map[RAW_GATE_UP]
        with safe_open(gate_up_file, framework="pt", device="cpu") as handle:
            gate_up = handle.get_tensor(RAW_GATE_UP)
        raw_nbytes += tensor_nbytes(gate_up)
        if tuple(gate_up.shape) != (512, 1280, 2560):
            raise ValueError(f"unexpected gate_up shape: {tuple(gate_up.shape)}")
        for expert in range(gate_up.shape[0]):
            add_compressed(
                output,
                f"mtp.layers.0.mlp.experts.{expert}.gate_proj",
                gate_up[expert, :640],
                quant_args,
                quant_scheme,
            )
            add_compressed(
                output,
                f"mtp.layers.0.mlp.experts.{expert}.up_proj",
                gate_up[expert, 640:],
                quant_args,
                quant_scheme,
            )
            if expert % 64 == 63:
                print(f"quantized gate/up experts: {expert + 1}/512", flush=True)
        del gate_up

        down_file = source / weight_map[RAW_DOWN]
        with safe_open(down_file, framework="pt", device="cpu") as handle:
            down = handle.get_tensor(RAW_DOWN)
        raw_nbytes += tensor_nbytes(down)
        if tuple(down.shape) != (512, 2560, 640):
            raise ValueError(f"unexpected down shape: {tuple(down.shape)}")
        for expert in range(down.shape[0]):
            add_compressed(
                output,
                f"mtp.layers.0.mlp.experts.{expert}.down_proj",
                down[expert],
                quant_args,
                quant_scheme,
            )
            if expert % 64 == 63:
                print(f"quantized down experts: {expert + 1}/512", flush=True)
        del down

        save_file(output, temporary / OUTPUT_FILE)
        packed_nbytes = sum(tensor_nbytes(tensor) for tensor in output.values())

        new_weight_map = {
            key: filename
            for key, filename in weight_map.items()
            if key not in {RAW_GATE_UP, RAW_DOWN} and is_draft_weight(key)
        }
        for key in output:
            new_weight_map[key] = OUTPUT_FILE
        new_index = dict(index)
        new_index["weight_map"] = dict(sorted(new_weight_map.items()))
        metadata = dict(new_index.get("metadata", {}))
        if "total_size" in metadata:
            metadata["total_size"] = int(metadata["total_size"]) - raw_nbytes + packed_nbytes
        new_index["metadata"] = metadata

        config["quantization_config"] = {
            "config_groups": {
                "mtp_routed_experts": {
                    "format": "pack-quantized",
                    "input_activations": None,
                    "output_activations": None,
                    "targets": ["RoutedExperts"],
                    "weights": {
                        "actorder": None,
                        "block_structure": None,
                        "dynamic": False,
                        "group_size": parsed.group_size,
                        "num_bits": 4,
                        "observer": "minmax",
                        "observer_kwargs": {},
                        "strategy": "group",
                        "symmetric": True,
                        "type": "int",
                    },
                }
            },
            "format": "pack-quantized",
            "global_compression_ratio": None,
            "ignore": [],
            "kv_cache_scheme": None,
            "quant_method": "compressed-tensors",
            "quantization_status": "compressed",
            "sparsity_config": {},
            "transform_config": {},
        }

        (temporary / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(new_index, indent=2) + "\n"
        )
        required_source_shards = {
            filename
            for filename in new_weight_map.values()
            if filename != OUTPUT_FILE
        }
        raw_expert_shards = {weight_map[RAW_GATE_UP], weight_map[RAW_DOWN]}
        for item in source.iterdir():
            if item.name in {"config.json", "model.safetensors.index.json"}:
                continue
            if item.name == OUTPUT_FILE:
                continue
            if item.suffix == ".safetensors":
                if item.name not in required_source_shards:
                    continue
                if item.name in raw_expert_shards:
                    filtered_keys = [
                        key
                        for key, filename in new_weight_map.items()
                        if filename == item.name
                    ]
                    with safe_open(item, framework="pt", device="cpu") as handle:
                        filtered = {
                            key: handle.get_tensor(key) for key in filtered_keys
                        }
                    save_file(filtered, temporary / item.name)
                    continue
            os.symlink(item, temporary / item.name)

        temporary.rename(destination)
        print(
            json.dumps(
                {
                    "destination": str(destination),
                    "raw_expert_bytes": raw_nbytes,
                    "packed_expert_bytes": packed_nbytes,
                    "compression_ratio": raw_nbytes / packed_nbytes,
                    "packed_tensor_count": len(output),
                    "group_size": parsed.group_size,
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException:
        print(f"incomplete build retained for inspection: {temporary}", flush=True)
        raise


if __name__ == "__main__":
    main()
