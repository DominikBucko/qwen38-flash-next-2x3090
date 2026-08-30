#!/usr/bin/env python3
"""Install and verify a file overlay into an existing vLLM Python package."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("overlay", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--destination",
        type=Path,
        help="Override the discovered vLLM package root (useful for validation).",
    )
    args = parser.parse_args()
    overlay = args.overlay.resolve()
    manifest_path = (args.manifest or overlay / "SHA256SUMS.json").resolve()
    expected = json.loads(manifest_path.read_text())
    if args.destination:
        destination = args.destination.resolve()
        destination.mkdir(parents=True, exist_ok=True)
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            raise SystemExit("vLLM package not found")
        destination = Path(next(iter(spec.submodule_search_locations))).resolve()

    observed = {
        str(path.relative_to(overlay)): sha256(path)
        for path in overlay.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if observed != expected:
        raise SystemExit("overlay manifest does not match packaged files")
    for relative, digest in sorted(expected.items()):
        source = overlay / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if sha256(target) != digest:
            raise SystemExit(f"post-install checksum mismatch: {relative}")
    print(f"installed and verified {len(expected)} files into {destination}")


if __name__ == "__main__":
    main()
