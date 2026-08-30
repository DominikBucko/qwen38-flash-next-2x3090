#!/usr/bin/env python3
"""Validate the target and compact MTP as one Hugging Face upload tree."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path


def read_header(path: Path) -> dict:
    with path.open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        length = struct.unpack("<Q", raw)[0]
        return json.loads(source.read(length))


def validate_checkpoint(root: Path) -> dict:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weight_map: dict[str, str] = index["weight_map"]
    observed: dict[str, str] = {}
    payload = 0
    errors: list[str] = []
    for filename in sorted(set(weight_map.values())):
        path = root / filename
        if not path.is_file():
            errors.append(f"missing: {path}")
            continue
        if path.is_symlink():
            errors.append(f"symlinked shard: {path}")
        for name, metadata in read_header(path).items():
            if name == "__metadata__":
                continue
            if name in observed:
                errors.append(f"duplicate tensor: {name}")
            observed[name] = filename
            payload += metadata["data_offsets"][1] - metadata["data_offsets"][0]
    if observed != weight_map:
        errors.append(
            f"index/header mismatch: indexed={len(weight_map)}, observed={len(observed)}"
        )
    expected_payload = index.get("metadata", {}).get("total_size")
    if expected_payload != payload:
        errors.append(
            f"payload mismatch: metadata={expected_payload}, observed={payload}"
        )
    return {
        "path": str(root),
        "files": len(set(weight_map.values())),
        "tensors": len(observed),
        "payload_bytes": payload,
        "payload_gib": round(payload / 2**30, 3),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    model = args.model.resolve()
    required = [
        "README.md",
        "LICENSE",
        "config.json",
        "hybrid_sources.json",
        "runtime/README.md",
        "runtime/Dockerfile",
        "runtime/LICENSE",
        "runtime/repro.lock.json",
        "runtime/static_hot_cache_rankings.json",
        "runtime/serve-container.sh",
        "runtime/vllm-overlay/SHA256SUMS.json",
        "runtime/mtp-int4-g32/compact_sources.json",
    ]
    errors = [f"missing required file: {name}" for name in required if not (model / name).is_file()]
    symlinks = [str(path.relative_to(model)) for path in model.rglob("*") if path.is_symlink()]
    if symlinks:
        errors.append(f"upload tree contains {len(symlinks)} symlinks")
    checkpoints = [
        validate_checkpoint(model),
        validate_checkpoint(model / "runtime" / "mtp-int4-g32"),
    ]
    for checkpoint in checkpoints:
        errors.extend(checkpoint["errors"])
    report = {
        "model": str(model),
        "checkpoints": checkpoints,
        "files": sum(path.is_file() for path in model.rglob("*")),
        "symlinks": symlinks,
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
