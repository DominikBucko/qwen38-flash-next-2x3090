#!/usr/bin/env python3
"""Fast, GPU-free checks for the publishable source repository."""

from __future__ import annotations

import gzip
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
PUBLIC_BENCHMARKS = ROOT / "benchmarks" / "2026-09-05"
PUBLIC_REPORTS = (
    "baseline-hot86/short-code.json.gz",
    "baseline-hot86/long-code.json.gz",
    "candidate-hot84-p2p/short-chat.json.gz",
    "candidate-hot84-p2p/long-chat.json.gz",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_report(
    relative: str,
    case: dict[str, object] | None,
    errors: list[str],
) -> str | None:
    path = PUBLIC_BENCHMARKS / relative
    if not path.is_file():
        errors.append(f"missing public benchmark report: benchmarks/2026-09-05/{relative}")
        return None
    try:
        raw_json = gzip.decompress(path.read_bytes())
        report = json.loads(raw_json)
    except (EOFError, OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid public benchmark report {relative}: {error}")
        return None
    if not isinstance(report, dict):
        errors.append(f"public benchmark report is not an object: {relative}")
        return None

    raw_hash = hashlib.sha256(raw_json).hexdigest()
    if case is not None and case.get("raw_json_sha256") != raw_hash:
        errors.append(f"public benchmark SHA differs from summary.json: {relative}")

    protocol = report.get("protocol")
    runs = report.get("runs")
    if not isinstance(protocol, dict) or not isinstance(runs, list):
        errors.append(f"public benchmark report lacks protocol or runs: {relative}")
        return raw_hash
    input_tokens = protocol.get("input_tokens")
    output_tokens = protocol.get("output_tokens")
    protocol_counts_valid = all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (input_tokens, output_tokens)
    )
    if not protocol_counts_valid:
        errors.append(f"public benchmark protocol counts are invalid: {relative}")
    measured = [
        run
        for run in runs
        if isinstance(run, dict) and run.get("phase") == "measured"
    ]
    if protocol.get("measured_runs") != 3 or len(measured) != 3:
        errors.append(f"public benchmark must contain exactly 3 measured runs: {relative}")
    indexes = [run.get("index") for run in measured]
    if any(
        isinstance(index, bool) or not isinstance(index, int) for index in indexes
    ) or set(indexes) != {0, 1, 2}:
        errors.append(f"public benchmark measured run indexes differ: {relative}")
    for run in measured:
        usage = run.get("usage")
        counts_valid = (
            run.get("status") == "ok"
            and run.get("requested_input_tokens") == input_tokens
            and run.get("requested_output_tokens") == output_tokens
            and isinstance(usage, dict)
            and usage.get("prompt_tokens") == input_tokens
            and usage.get("completion_tokens") == output_tokens
            and protocol_counts_valid
            and usage.get("total_tokens") == input_tokens + output_tokens
        )
        if not counts_valid:
            errors.append(
                f"public benchmark measured run {run.get('index')} is invalid: {relative}"
            )
    report_summary = report.get("summary")
    if (
        not isinstance(report_summary, dict)
        or report_summary.get("all_measurements_valid") is not True
        or report_summary.get("valid_measured_runs") != 3
    ):
        errors.append(f"public benchmark validity summary differs: {relative}")
    if case is not None and (
        case.get("input_tokens") != input_tokens
        or case.get("output_tokens") != output_tokens
        or case.get("measured_runs") != 3
    ):
        errors.append(f"public benchmark counts differ from summary.json: {relative}")
    return raw_hash


def validate_public_benchmarks(errors: list[str]) -> None:
    summary_path = PUBLIC_BENCHMARKS / "summary.json"
    if not summary_path.is_file():
        errors.append("missing public benchmark summary: benchmarks/2026-09-05/summary.json")
        return
    try:
        summary = json.loads(summary_path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        errors.append(f"invalid public benchmark summary: {error}")
        return

    cases: list[dict[str, object]] = []
    if isinstance(summary, dict) and isinstance(summary.get("suites"), list):
        for suite in summary["suites"]:
            if isinstance(suite, dict) and isinstance(suite.get("cases"), list):
                cases.extend(case for case in suite["cases"] if isinstance(case, dict))
    paths = [
        case.get("raw_report")
        for case in cases
        if isinstance(case.get("raw_report"), str)
    ]
    if len(paths) != len(cases):
        errors.append("public benchmark summary contains a case without a raw report path")
    if len(paths) != len(set(paths)):
        errors.append("public benchmark summary contains duplicate raw report paths")
    case_by_path = {
        path: case
        for case in cases
        if isinstance((path := case.get("raw_report")), str)
    }
    if set(case_by_path) != set(PUBLIC_REPORTS):
        errors.append(
            "public benchmark summary raw reports differ: "
            f"expected={list(PUBLIC_REPORTS)}, observed={sorted(case_by_path)}"
        )

    source_hash = None
    for relative in PUBLIC_REPORTS:
        observed_hash = validate_raw_report(relative, case_by_path.get(relative), errors)
        if relative == "candidate-hot84-p2p/long-chat.json.gz":
            source_hash = observed_hash

    chart_path = ROOT / "docs" / "images" / "long-context-decode.svg"
    if not chart_path.is_file():
        errors.append("missing long-context decode chart: docs/images/long-context-decode.svg")
        return
    chart = chart_path.read_text()
    source_comments = re.findall(r"source-json-sha256: ([0-9a-f]{64})", chart)
    generator_comments = re.findall(r"generator-file-sha256: ([0-9a-f]{64})", chart)
    if len(source_comments) != 1 or source_comments[0] != source_hash:
        errors.append(
            "long-context-decode.svg source provenance differs from "
            "candidate-hot84-p2p/long-chat.json.gz"
        )
    generator_path = ROOT / "scripts" / "render_public_decode.py"
    if len(generator_comments) != 1 or generator_comments[0] != sha256(generator_path):
        errors.append(
            "long-context-decode.svg generator provenance differs from "
            "scripts/render_public_decode.py"
        )


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

    validate_public_benchmarks(errors)

    agent_notes = (ROOT / "AGENTS.md").read_text()
    matched_points = hillclimb_data["matched_decode"]["points"]
    long_points = hillclimb_data["long_decode"]["points"]
    agent_note_pins = {
        "published model revision": lock["published_model"]["revision"],
        "maximum context": str(lock["runtime"]["max_model_len"]),
        "matched baseline": f"{matched_points[0]['tokens_per_second']:.2f}",
        "matched endpoint": f"{matched_points[-1]['tokens_per_second']:.2f}",
        "long-decode endpoint": f"{long_points[-1]['tokens_per_second']:.2f}",
    }
    for label, value in agent_note_pins.items():
        if value not in agent_notes:
            errors.append(f"AGENTS.md is missing current {label}: {value}")

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
