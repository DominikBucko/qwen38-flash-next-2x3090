#!/usr/bin/env python3
"""Stream-verify that compact MTP tensors are bit-identical to their source."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path


CHUNK = 8 * 1024 * 1024


def header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as source:
        length_raw = source.read(8)
        if len(length_raw) != 8:
            raise ValueError(f"truncated safetensors file: {path}")
        length = struct.unpack("<Q", length_raw)[0]
        return 8 + length, json.loads(source.read(length))


def region_sha256(path: Path, start: int, end: int) -> str:
    digest = hashlib.sha256()
    remaining = end - start
    with path.open("rb") as source:
        source.seek(start)
        while remaining:
            block = source.read(min(CHUNK, remaining))
            if not block:
                raise ValueError(f"truncated tensor payload: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("compact", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    compact = args.compact.resolve()

    source_index = json.loads((source / "model.safetensors.index.json").read_text())
    compact_index = json.loads((compact / "model.safetensors.index.json").read_text())
    source_map: dict[str, str] = source_index["weight_map"]
    compact_map: dict[str, str] = compact_index["weight_map"]
    if source_map.keys() != compact_map.keys():
        raise SystemExit("source and compact tensor-name sets differ")

    file_headers: dict[Path, tuple[int, dict]] = {}

    def cached_header(path: Path) -> tuple[int, dict]:
        if path not in file_headers:
            file_headers[path] = header(path)
        return file_headers[path]

    mismatches: list[str] = []
    verified = 0
    for name in sorted(source_map):
        source_path = source / source_map[name]
        compact_path = compact / compact_map[name]
        source_base, source_header = cached_header(source_path)
        compact_base, compact_header = cached_header(compact_path)
        source_meta = source_header[name]
        compact_meta = compact_header[name]
        if source_meta["dtype"] != compact_meta["dtype"] or source_meta["shape"] != compact_meta["shape"]:
            mismatches.append(f"metadata differs: {name}")
            continue
        source_start, source_end = source_meta["data_offsets"]
        compact_start, compact_end = compact_meta["data_offsets"]
        if source_end - source_start != compact_end - compact_start:
            mismatches.append(f"payload size differs: {name}")
            continue
        source_digest = region_sha256(
            source_path, source_base + source_start, source_base + source_end
        )
        compact_digest = region_sha256(
            compact_path, compact_base + compact_start, compact_base + compact_end
        )
        if source_digest != compact_digest:
            mismatches.append(f"payload differs: {name}")
        else:
            verified += 1

    report = {
        "source": str(source),
        "compact": str(compact),
        "tensors": len(source_map),
        "verified_bit_identical": verified,
        "mismatches": mismatches,
    }
    print(json.dumps(report, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
