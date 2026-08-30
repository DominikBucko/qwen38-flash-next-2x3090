#!/usr/bin/env python3
"""Make a self-contained, uploadable Qwen3.8 MTP draft checkpoint.

The development draft uses symlinks to large target shards even though its
index selects only a few tensors from each shard.  This utility materializes
only indexed tensors, preserving their dtype and values, so the upload is a
few GiB instead of appearing to require the full target checkpoint again.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


EXPERT_FILE = "mtp-routed-experts-int4.safetensors"
DENSE_FILE = "mtp-dense.safetensors"
MTP_SOURCE_REPO = "RadixArk/Qwen3.8-Flash-Next-NVFP4"
MTP_SOURCE_REVISION = "7b719225242aacd3dbd3f9407468c2ee9a9d2594"
SMALL_FILES = (
    ".gitattributes",
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)


def payload_bytes(path: Path) -> int:
    with path.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(length))
    return sum(
        meta["data_offsets"][1] - meta["data_offsets"][0]
        for name, meta in header.items()
        if name != "__metadata__"
    )


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if copy:
        shutil.copy2(source, destination)
    else:
        os.link(source.resolve(), destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument(
        "--copy-experts",
        action="store_true",
        help="Copy the packed expert file instead of hard-linking it.",
    )
    args = parser.parse_args()
    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise SystemExit(f"destination already exists: {destination}")

    old_index = json.loads((source / "model.safetensors.index.json").read_text())
    old_map: dict[str, str] = old_index["weight_map"]
    if EXPERT_FILE not in set(old_map.values()):
        raise SystemExit(f"missing indexed {EXPERT_FILE}")

    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-building-", dir=destination.parent)
    )
    try:
        dense = {}
        by_file: dict[str, list[str]] = {}
        for name, filename in old_map.items():
            if filename != EXPERT_FILE:
                by_file.setdefault(filename, []).append(name)
        for filename, names in sorted(by_file.items()):
            with safe_open(source / filename, framework="pt", device="cpu") as handle:
                for name in names:
                    dense[name] = handle.get_tensor(name).contiguous()
        save_file(dense, temporary / DENSE_FILE)
        link_or_copy(
            source / EXPERT_FILE,
            temporary / EXPERT_FILE,
            args.copy_experts,
        )

        new_map = {
            name: EXPERT_FILE if filename == EXPERT_FILE else DENSE_FILE
            for name, filename in old_map.items()
        }
        total_size = payload_bytes(temporary / DENSE_FILE) + payload_bytes(
            temporary / EXPERT_FILE
        )
        (temporary / "model.safetensors.index.json").write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": new_map},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        shutil.copy2(source / "config.json", temporary / "config.json")
        for filename in SMALL_FILES:
            item = source / filename
            if item.is_file():
                shutil.copy2(item, temporary / filename, follow_symlinks=True)
        (temporary / "compact_sources.json").write_text(
            json.dumps(
                {
                    "format": "qwen38-flash-next-mtp-int4-g32-compact-v1",
                    "source": {
                        "repo": MTP_SOURCE_REPO,
                        "revision": MTP_SOURCE_REVISION,
                    },
                    "source_indexed_files": sorted(set(old_map.values())),
                    "dense_tensors": len(dense),
                    "expert_tensors": sum(
                        filename == EXPERT_FILE for filename in old_map.values()
                    ),
                    "tensor_payload_bytes": total_size,
                    "packing": (
                        "MTP routed experts locally RTN-quantized to INT4 symmetric "
                        "group-32 with min/max scales; other MTP tensors unchanged"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        if any(item.is_symlink() for item in temporary.iterdir()):
            raise RuntimeError("compact checkpoint unexpectedly contains a symlink")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(destination)
        print(
            json.dumps(
                {
                    "destination": str(destination),
                    "tensors": len(new_map),
                    "dense_tensors": len(dense),
                    "payload_bytes": total_size,
                    "payload_gib": round(total_size / 2**30, 3),
                    "symlinks": 0,
                },
                indent=2,
            )
        )
    except BaseException:
        print(f"incomplete build retained for inspection: {temporary}")
        raise


if __name__ == "__main__":
    main()
