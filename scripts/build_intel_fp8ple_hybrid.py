#!/usr/bin/env python3
"""Build a zero-copy Intel AutoRound + RadixArk FP8-PLE hybrid checkpoint.

The script deliberately keeps Intel's native GPTQ/AutoRound packing.  It only
replaces the 102.4 GB BF16 n-gram embedding shard with the already published
RadixArk FP8 PLE shards and omits the bundled BF16 MTP tensors (the runtime
uses a separately quantized draft checkpoint).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
from pathlib import Path


INTEL_REPO = "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound"
INTEL_REVISION = "861536dda5bcb208376fc4cd879b2bf76bece9fe"
PLE_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
PLE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
SMALL_FILES = (
    ".gitattributes",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
)


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as source:
        return json.load(source)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True)
        output.write("\n")
    temporary.replace(path)


def tensor_bytes(path: Path) -> int:
    """Return tensor payload bytes from a safetensors header."""
    with path.open("rb") as source:
        header_length_raw = source.read(8)
        if len(header_length_raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_length = struct.unpack("<Q", header_length_raw)[0]
        header = json.loads(source.read(header_length))
    return sum(
        int(metadata["data_offsets"][1]) - int(metadata["data_offsets"][0])
        for name, metadata in header.items()
        if name != "__metadata__"
    )


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not os.path.samefile(source, destination):
            raise FileExistsError(f"Refusing to overwrite {destination}")
        return
    if copy:
        shutil.copy2(source, destination)
    else:
        os.link(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intel-source", required=True, type=Path)
    parser.add_argument("--ple-source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy large shards instead of hard-linking them.",
    )
    args = parser.parse_args()

    intel_index_path = args.intel_source / "model.safetensors.index.json"
    ple_index_path = args.ple_source / "model.safetensors.index.json"
    intel_index = read_json(intel_index_path)
    ple_index = read_json(ple_index_path)

    # Intel's shard 16 is exclusively the BF16 PLE table.  The extra tensor
    # file is exclusively MTP.  Select by mapped filename rather than tensor
    # name so this remains auditable against the upstream index.
    excluded_intel_files = {
        "model-00016-of-00017.safetensors",
        "model_extra_tensors.safetensors",
    }
    target_map = {
        name: filename
        for name, filename in intel_index["weight_map"].items()
        if filename not in excluded_intel_files
    }
    ple_map = {
        name: filename
        for name, filename in ple_index["weight_map"].items()
        if filename.startswith("model-plefp8-")
    }
    if len(ple_map) != 129:
        raise ValueError(f"Expected 128 FP8 PLE shards plus one scale, got {len(ple_map)}")
    if set(target_map).intersection(ple_map):
        raise ValueError("Intel target and replacement PLE maps unexpectedly overlap")

    output_map = {**target_map, **ple_map}
    required_intel_shards = sorted(set(target_map.values()))
    required_ple_shards = sorted(set(ple_map.values()))
    required_sources = [
        *(args.intel_source / name for name in required_intel_shards),
        *(args.ple_source / name for name in required_ple_shards),
        args.intel_source / "config.json",
    ]
    missing = [str(path) for path in required_sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required inputs:\n" + "\n".join(missing))

    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {args.output}")

    for filename in required_intel_shards:
        link_or_copy(args.intel_source / filename, args.output / filename, args.copy)
    for filename in required_ple_shards:
        link_or_copy(args.ple_source / filename, args.output / filename, args.copy)

    for filename in SMALL_FILES:
        source = args.intel_source / filename
        if not source.is_file():
            source = args.ple_source / filename
        if source.is_file():
            shutil.copy2(source, args.output / filename)

    license_source = args.ple_source / "LICENSE"
    if license_source.is_file():
        shutil.copy2(license_source, args.output / "LICENSE")

    config = read_json(args.intel_source / "config.json")
    text_config = config.setdefault("text_config", {})
    text_config["ple_embedding_dtype"] = "float8_e4m3fn"
    write_json(args.output / "config.json", config)

    all_shards = required_intel_shards + required_ple_shards
    total_size = sum(tensor_bytes(args.output / filename) for filename in all_shards)
    write_json(
        args.output / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": output_map},
    )
    write_json(
        args.output / "hybrid_sources.json",
        {
            "format": "qwen38-flash-next-intel-autoround-fp8-ple-v1",
            "target": {
                "repo": INTEL_REPO,
                "revision": INTEL_REVISION,
                "packing": "native AutoRound GPTQ W4A16; unchanged",
                "omitted_files": sorted(excluded_intel_files),
            },
            "ple": {
                "repo": PLE_REPO,
                "revision": PLE_REVISION,
                "precision": "float8_e4m3fn with one global weight_scale",
                "files": required_ple_shards,
            },
            "mtp": {
                "included": False,
                "runtime": "Supply a separate draft model with vLLM speculative_config",
            },
            "assembly": "hardlink" if not args.copy else "copy",
            "tensor_payload_bytes": total_size,
        },
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "tensors": len(output_map),
                "intel_shards": len(required_intel_shards),
                "ple_shards": len(required_ple_shards),
                "payload_bytes": total_size,
                "payload_gib": round(total_size / 2**30, 3),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
