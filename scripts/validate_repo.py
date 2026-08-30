#!/usr/bin/env python3
"""Fast, GPU-free checks for the publishable source repository."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True
from render_hillclimb import render


ROOT = Path(__file__).resolve().parent.parent
OVERLAY = ROOT / "runtime" / "vllm-overlay"
MANIFEST = OVERLAY / "SHA256SUMS.json"
MAX_GIT_FILE = 50 * 1024 * 1024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    expected: dict[str, str] = json.loads(MANIFEST.read_text())
    observed = {
        str(path.relative_to(OVERLAY)): sha256(path)
        for path in OVERLAY.rglob("*")
        if path.is_file() and path != MANIFEST
    }
    if observed != expected:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        changed = sorted(
            name for name in expected.keys() & observed.keys()
            if expected[name] != observed[name]
        )
        errors.append(
            f"overlay manifest mismatch: missing={missing}, extra={extra}, changed={changed}"
        )

    lock = json.loads((ROOT / "repro.lock.json").read_text())
    image = lock["runtime"]["container_image"]
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    if f"FROM {image}" not in dockerfile:
        errors.append("Dockerfile base image differs from repro.lock.json")
    humming = lock["runtime"]["humming_kernels_version"]
    if f"humming-kernels=={humming}" not in dockerfile:
        errors.append("Dockerfile Humming version differs from repro.lock.json")

    hillclimb_data = json.loads((ROOT / "benchmarks" / "hillclimb.json").read_text())
    expected_chart = render(hillclimb_data)
    chart_path = ROOT / "docs" / "images" / "hillclimb.svg"
    if not chart_path.is_file() or chart_path.read_text() != expected_chart:
        errors.append("hillclimb.svg is stale; run scripts/render_hillclimb.py")

    target = lock["checkpoint"]["target"]
    ple = lock["checkpoint"]["ple"]
    hybrid_builder = (ROOT / "scripts" / "build_intel_fp8ple_hybrid.py").read_text()
    mtp_builder = (ROOT / "scripts" / "compact_mtp_checkpoint.py").read_text()
    for value, source_name, source_text in (
        (target["repo"], "target repo", hybrid_builder),
        (target["revision"], "target revision", hybrid_builder),
        (ple["repo"], "PLE repo", hybrid_builder),
        (ple["revision"], "PLE revision", hybrid_builder),
        (ple["repo"], "MTP source repo", mtp_builder),
        (ple["revision"], "MTP source revision", mtp_builder),
    ):
        if value not in source_text:
            errors.append(f"{source_name} differs from repro.lock.json")

    forbidden_suffixes = {".safetensors", ".gguf", ".bin", ".pt", ".pth"}
    secret_pattern = re.compile(
        r"(?:hf_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghp_[A-Za-z0-9]{20,})"
    )
    for path in ROOT.rglob("*"):
        if ".git" in path.parts or not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if path.is_symlink():
            errors.append(f"symlink in source repository: {relative}")
        if path.stat().st_size > MAX_GIT_FILE:
            errors.append(f"file exceeds 50 MiB source limit: {relative}")
        if path.suffix.lower() in forbidden_suffixes:
            errors.append(f"model payload must not be committed to GitHub: {relative}")
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            errors.append(f"Python cache file present: {relative}")
        if path.stat().st_size < 5 * 1024 * 1024:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                continue
            if secret_pattern.search(text):
                errors.append(f"possible credential in {relative}")

    required_executables = [
        "scripts/serve-container.sh",
        "scripts/docker_serve.sh",
        "scripts/preflight.sh",
        "scripts/download_sources.sh",
        "scripts/assemble_hf_repo.sh",
        "scripts/assemble_with_docker.sh",
        "scripts/upload_hf.sh",
        "scripts/render_hillclimb.py",
    ]
    for relative in required_executables:
        path = ROOT / relative
        if not os.access(path, os.X_OK):
            errors.append(f"script is not executable: {relative}")

    if errors:
        print("repository validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        raise SystemExit(1)
    total_files = sum(
        path.is_file() and ".git" not in path.parts for path in ROOT.rglob("*")
    )
    print(
        f"repository validation passed: {len(observed)} overlay files, "
        f"{total_files} total files"
    )


if __name__ == "__main__":
    main()
