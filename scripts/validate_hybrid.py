#!/usr/bin/env python3
"""Perform structural and safetensors-header validation of a hybrid repo."""

from __future__ import annotations

import argparse
import json
import struct
from collections import Counter
from pathlib import Path


def header(path: Path) -> dict:
    with path.open("rb") as source:
        length = struct.unpack("<Q", source.read(8))[0]
        return json.loads(source.read(length))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    index = json.loads((args.model / "model.safetensors.index.json").read_text())
    config = json.loads((args.model / "config.json").read_text())
    weight_map = index["weight_map"]
    filenames = sorted(set(weight_map.values()))
    errors: list[str] = []
    observed: dict[str, str] = {}
    payload = 0

    for filename in filenames:
        path = args.model / filename
        if not path.is_file():
            errors.append(f"missing file: {filename}")
            continue
        if path.is_symlink():
            errors.append(f"upload artifact contains symlink: {filename}")
        file_header = header(path)
        for name, metadata in file_header.items():
            if name == "__metadata__":
                continue
            if name in observed:
                errors.append(f"duplicate tensor: {name}")
            observed[name] = filename
            payload += metadata["data_offsets"][1] - metadata["data_offsets"][0]

    for name, filename in weight_map.items():
        actual = observed.get(name)
        if actual != filename:
            errors.append(f"index mismatch: {name}: expected={filename}, actual={actual}")
    unindexed = sorted(set(observed).difference(weight_map))
    if unindexed:
        errors.append(f"{len(unindexed)} unindexed tensors; first={unindexed[:3]}")
    if payload != index.get("metadata", {}).get("total_size"):
        errors.append(
            "metadata.total_size mismatch: "
            f"index={index.get('metadata', {}).get('total_size')} observed={payload}"
        )

    ple = [name for name in observed if "ple_embedding.ngram_embedding" in name]
    mtp = [name for name in observed if ".mtp." in name or name.startswith("mtp.")]
    quantization = config.get("quantization_config", {})
    report = {
        "files": len(filenames),
        "tensors": len(observed),
        "payload_gib": round(payload / 2**30, 3),
        "files_by_prefix": dict(Counter(name.split("-")[0] for name in filenames)),
        "ple_tensors": len(ple),
        "mtp_tensors": len(mtp),
        "ple_dtype": config.get("text_config", {}).get("ple_embedding_dtype"),
        "quant_method": quantization.get("quant_method"),
        "bits": quantization.get("bits"),
        "group_size": quantization.get("group_size"),
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)
    if len(ple) != 129 or mtp or report["ple_dtype"] != "float8_e4m3fn":
        raise SystemExit("Hybrid semantic checks failed")


if __name__ == "__main__":
    main()
