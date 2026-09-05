#!/usr/bin/env python3
"""Collect new synthetic streaming measurements from the released endpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

ROOT = Path(__file__).resolve().parent.parent
SEED_TEXT = (
    "Reproducible serving measurements use a fixed public prompt. "
    "The client repeats these token IDs and truncates them to the requested length. "
    "This synthetic text contains no private workload data.\n"
)
REPO_CHAT_PREFIX = (
    "Write a detailed tutorial on multi-GPU inference: tensor parallelism, CPU "
    "offload, expert LRU caching, KV state, and speculative decoding. Use the "
    "source below where helpful. Include algorithms, numerical examples, "
    "tradeoffs, and tests. Continue with useful detail for the entire response "
    "budget.\n\n<source>\n"
)
REPO_CHAT_SUFFIX = "\n</source>"
COMPATIBILITY_SOURCE = {
    "inspected_source": {
        "repository": "https://github.com/vllm-project/vllm",
        "commit": "4ab6e99d246478a0f0a1f694b0b19d2c649eaf1b",
        "note": "Local Qwen3.8 source inspected for protocol shape; not claimed as deployed.",
    },
    "expected_repo_runtime": {
        "version": "0.1.dev20073+g8e685d198",
        "note": "Expected from repro.lock.json; this client does not query server identity.",
        "verification": (
            "The matching vendor completion/protocol.py was checked for "
            "return_token_ids, cache_salt, and add_special_tokens; the matching "
            "tokenize/protocol.py was checked for chat messages and generation prompts"
        ),
    },
    "verified_interfaces": [
        "/tokenize",
        "/tokenize chat messages with generation prompt",
        "/v1/completions streaming SSE",
        "token-ID prompts",
        "cache_salt",
        "ignore_eos",
        "return_token_ids",
        "stream_options.include_usage",
    ],
}


class BenchmarkError(RuntimeError):
    """The endpoint or stream did not satisfy the measurement protocol."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def endpoint(base_url: str, path: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("--base-url must be an http(s) server URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("--base-url must not contain credentials, query, or fragment")
    return base_url.rstrip("/") + path


def request(url: str, payload: dict[str, Any]) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        with urllib.request.urlopen(request(url, payload), timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise BenchmarkError(f"HTTP {code} from {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(f"request to {url} failed: {type(exc).__name__}") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkError(f"non-JSON response from {url}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"JSON response from {url} is not an object")
    return value


def tokenize(
    base_url: str, model: str, timeout: float, source_text: str = SEED_TEXT
) -> list[int]:
    value = post_json(
        endpoint(base_url, "/tokenize"),
        {
            "model": model,
            "prompt": source_text,
            "add_special_tokens": False,
        },
        timeout,
    )
    tokens, count = value.get("tokens"), value.get("count")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or token < 0 for token in tokens)
        or count != len(tokens)
    ):
        raise BenchmarkError("/tokenize returned invalid tokens or count")
    return tokens


def tokenize_chat(base_url: str, model: str, timeout: float, content: str) -> list[int]:
    value = post_json(
        endpoint(base_url, "/tokenize"),
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "add_generation_prompt": True,
            "add_special_tokens": False,
        },
        timeout,
    )
    tokens, count = value.get("tokens"), value.get("count")
    if (
        not isinstance(tokens, list)
        or not tokens
        or any(not isinstance(token, int) or token < 0 for token in tokens)
        or count != len(tokens)
    ):
        raise BenchmarkError("/tokenize chat request returned invalid tokens or count")
    return tokens


def build_prompt(seed_tokens: list[int], length: int) -> list[int]:
    repeats = (length + len(seed_tokens) - 1) // len(seed_tokens)
    return (seed_tokens * repeats)[:length]


def common_wrappers(full: list[int], empty: list[int]) -> tuple[list[int], list[int]]:
    prefix_length = 0
    while (
        prefix_length < min(len(full), len(empty))
        and full[prefix_length] == empty[prefix_length]
    ):
        prefix_length += 1
    suffix_length = 0
    remaining = min(len(full), len(empty)) - prefix_length
    while (
        suffix_length < remaining
        and full[-1 - suffix_length] == empty[-1 - suffix_length]
    ):
        suffix_length += 1
    if prefix_length == 0 or suffix_length == 0:
        raise BenchmarkError(
            "chat tokenizations did not expose stable prefix and suffix wrappers"
        )
    return full[:prefix_length], full[len(full) - suffix_length :]


def build_wrapped_prompt(
    prefix: list[int], source: list[int], suffix: list[int], length: int
) -> tuple[list[int], int, int]:
    minimum = len(prefix) + len(suffix) + 1
    if length < minimum:
        raise BenchmarkError(
            f"repo-chat needs at least {minimum} input tokens; requested {length}"
        )
    if not source:
        raise BenchmarkError("repo-chat source token interior is empty")
    source_budget = length - len(prefix) - len(suffix)
    repeats = (source_budget + len(source) - 1) // len(source)
    selected = (source * repeats)[:source_budget]
    prompt = prefix + selected + suffix
    if len(prompt) != length:
        raise AssertionError("repo-chat prompt construction lost exact token count")
    return prompt, source_budget // len(source), source_budget % len(source)


def repo_chat_prompt(
    base_url: str, model: str, timeout: float, source_text: str, length: int
) -> tuple[list[int], dict[str, Any]]:
    full = tokenize_chat(
        base_url, model, timeout, REPO_CHAT_PREFIX + source_text + REPO_CHAT_SUFFIX
    )
    empty = tokenize_chat(base_url, model, timeout, REPO_CHAT_PREFIX + REPO_CHAT_SUFFIX)
    prefix, suffix = common_wrappers(full, empty)
    interior = full[len(prefix) : len(full) - len(suffix)]
    prompt, cycles, remainder = build_wrapped_prompt(prefix, interior, suffix, length)

    def compact(values: list[int]) -> bytes:
        return json.dumps(values, separators=(",", ":")).encode()

    return prompt, {
        "chat_instruction": REPO_CHAT_PREFIX,
        "chat_source_suffix": REPO_CHAT_SUFFIX,
        "chat_full_token_count": len(full),
        "chat_empty_source_token_count": len(empty),
        "fixed_prefix_token_ids": prefix,
        "fixed_prefix_token_ids_sha256": hashlib.sha256(compact(prefix)).hexdigest(),
        "fixed_suffix_token_ids": suffix,
        "fixed_suffix_token_ids_sha256": hashlib.sha256(compact(suffix)).hexdigest(),
        "source_token_ids": interior,
        "source_token_ids_sha256": hashlib.sha256(compact(interior)).hexdigest(),
        "source_token_count": len(interior),
        "source_complete_cycles_in_prompt": cycles,
        "source_prefix_tokens_after_complete_cycles": remainder,
        "includes_complete_source_corpus": cycles >= 1,
        "minimum_input_tokens": len(prefix) + len(suffix) + 1,
        "wrapper_derivation": (
            "longest non-overlapping common prefix and suffix of server-tokenized "
            "full-source and empty-source chats"
        ),
    }


def prompt_source(style: str, root: Path = ROOT) -> tuple[str, dict[str, Any]]:
    if style == "repeated-seed":
        return SEED_TEXT, {
            "seed_text": SEED_TEXT,
            "seed_text_sha256": hashlib.sha256(SEED_TEXT.encode()).hexdigest(),
        }
    if style not in {"repo-code", "repo-chat"}:
        raise BenchmarkError(f"unknown prompt style: {style}")
    overlay = root / "runtime/vllm-overlay"
    paths = sorted(
        overlay.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
    )
    if not paths:
        raise BenchmarkError("repo-code prompt found no overlay Python sources")
    parts: list[str] = []
    files: list[dict[str, str]] = []
    for path in paths:
        relative = path.relative_to(root).as_posix()
        raw = path.read_bytes()
        parts.append(f"# SOURCE: {relative}\n{raw.decode('utf-8')}\n")
        files.append({"path": relative, "sha256": hashlib.sha256(raw).hexdigest()})
    source = "".join(parts)
    return source, {
        "seed_text_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "seed_text_omitted_from_report": True,
        "source_files": files,
        "corpus_format": "# SOURCE: <repository-relative-path>\\n<UTF-8 contents>\\n",
    }


def event_record(
    value: dict[str, Any], elapsed: float
) -> tuple[dict[str, Any], list[int], str]:
    if value.get("error") is not None:
        raise BenchmarkError("server returned an error object in the SSE stream")
    choices = value.get("choices", [])
    if not isinstance(choices, list):
        raise BenchmarkError("SSE event choices is not a list")
    output_token_ids: list[int] = []
    character_count = 0
    finish_reasons: list[str] = []
    for choice in choices:
        if not isinstance(choice, dict):
            raise BenchmarkError("SSE choice is not an object")
        text, token_ids = choice.get("text", ""), choice.get("token_ids")
        if not isinstance(text, str):
            raise BenchmarkError("SSE choice text is not a string")
        if token_ids is None and text:
            raise BenchmarkError("server omitted requested token_ids for output text")
        if token_ids is not None:
            if not isinstance(token_ids, list) or any(
                not isinstance(token, int) or token < 0 for token in token_ids
            ):
                raise BenchmarkError("SSE token_ids is invalid")
            output_token_ids.extend(token_ids)
        character_count += len(text)
        if choice.get("finish_reason") is not None:
            finish_reasons.append(str(choice["finish_reason"]))
    output_text = "".join(choice.get("text", "") for choice in choices)
    return (
        {
            "elapsed_seconds": elapsed,
            "choice_count": len(choices),
            "output_token_ids_count": len(output_token_ids),
            "output_text_characters": character_count,
            "finish_reasons": finish_reasons,
            "has_usage": value.get("usage") is not None,
        },
        output_token_ids,
        output_text,
    )


def read_sse(stream: BinaryIO, started: float) -> dict[str, Any]:
    chunks: list[dict[str, Any]] = []
    data_lines: list[str] = []
    usage: dict[str, Any] | None = None
    finish_reasons: list[str] = []
    first_output = last_output = None
    first_chunk_tokens = 0
    output_token_ids: list[int] = []
    output_parts: list[str] = []
    done = False

    def consume() -> None:
        nonlocal usage, first_output, last_output, first_chunk_tokens
        nonlocal done
        if not data_lines:
            return
        data = "\n".join(data_lines)
        data_lines.clear()
        elapsed = time.perf_counter() - started
        if data == "[DONE]":
            done = True
            return
        try:
            value = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BenchmarkError("invalid JSON in SSE data event") from exc
        if not isinstance(value, dict):
            raise BenchmarkError("SSE JSON event is not an object")
        record, event_token_ids, output_text = event_record(value, elapsed)
        record["sequence"] = len(chunks)
        chunks.append(record)
        output_parts.append(output_text)
        finish_reasons.extend(record["finish_reasons"])
        if event_token_ids:
            if first_output is None:
                first_output, first_chunk_tokens = elapsed, len(event_token_ids)
            last_output = elapsed
            output_token_ids.extend(event_token_ids)
        event_usage = value.get("usage")
        if event_usage is not None:
            if usage is not None or not isinstance(event_usage, dict):
                raise BenchmarkError("stream returned duplicate or invalid usage")
            usage = event_usage

    try:
        for raw_line in stream:
            try:
                line = raw_line.decode().rstrip("\r\n")
            except UnicodeDecodeError as exc:
                raise BenchmarkError("SSE stream is not UTF-8") from exc
            if not line:
                consume()
                if done:
                    break
            elif line.startswith("data:"):
                data_lines.append(line[5:].lstrip(" "))
        consume()
    except (TimeoutError, OSError) as exc:
        raise BenchmarkError(f"SSE stream failed: {type(exc).__name__}") from exc
    if not done:
        raise BenchmarkError("SSE stream ended without [DONE]")
    return {
        "chunks": chunks,
        "usage": usage,
        "finish_reasons": finish_reasons,
        "first_output_seconds": first_output,
        "last_output_seconds": last_output,
        "first_chunk_tokens": first_chunk_tokens,
        "output_token_ids": output_token_ids,
        "output_text": "".join(output_parts),
    }


def measure_run(
    base_url: str,
    model: str,
    prompt: list[int],
    output_tokens: int,
    timeout: float,
    phase: str,
    index: int,
    capture_path: Path | None = None,
) -> dict[str, Any]:
    salt = secrets.token_urlsafe(32)
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": output_tokens,
        "temperature": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
        "return_token_ids": True,
        "cache_salt": salt,
    }
    http_request = request(endpoint(base_url, "/v1/completions"), payload)
    http_request.add_header("Accept", "text/event-stream")
    started_at, started = utc_now(), time.perf_counter()
    try:
        response = urllib.request.urlopen(http_request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        code = exc.code
        exc.close()
        raise BenchmarkError(f"HTTP {code} from /v1/completions") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(
            f"completion request failed: {type(exc).__name__}"
        ) from exc
    with response:
        stream = read_sse(response, started)
    duration = time.perf_counter() - started
    response_finished_at = utc_now()
    output_bytes = stream.pop("output_text").encode("utf-8", errors="surrogatepass")
    output_token_ids = stream.pop("output_token_ids")
    token_ids_bytes = json.dumps(output_token_ids, separators=(",", ":")).encode()
    if capture_path is not None:
        write_atomic_bytes(capture_path, output_bytes)
    errors: list[str] = []
    usage = stream["usage"]
    if not isinstance(usage, dict):
        errors.append("missing final usage")
        usage = {}
    expected = {
        "prompt_tokens": len(prompt),
        "completion_tokens": output_tokens,
        "total_tokens": len(prompt) + output_tokens,
    }
    for key, expected_value in expected.items():
        if usage.get(key) != expected_value:
            errors.append(f"usage {key}={usage.get(key)!r}, expected {expected_value}")
    if len(output_token_ids) != output_tokens:
        errors.append("streamed token_ids count does not equal requested output")
    if stream["finish_reasons"] != ["length"]:
        errors.append(
            f"finish reasons were {stream['finish_reasons']!r}, expected ['length']"
        )
    first, last = stream["first_output_seconds"], stream["last_output_seconds"]
    remaining = output_tokens - stream["first_chunk_tokens"]
    tpot = (
        (last - first) / remaining
        if first is not None and last is not None and remaining > 0 and last > first
        else None
    )
    return {
        "phase": phase,
        "index": index,
        "status": "ok" if not errors else "invalid",
        "errors": errors,
        "started_at": started_at,
        "finished_at": response_finished_at,
        "cache_salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "output_text_sha256": hashlib.sha256(output_bytes).hexdigest(),
        "output_token_ids_sha256": hashlib.sha256(token_ids_bytes).hexdigest(),
        "captured_output_file": capture_path.name if capture_path else None,
        "requested_input_tokens": len(prompt),
        "requested_output_tokens": output_tokens,
        "usage": usage,
        "ttft_seconds": first,
        "request_duration_seconds": duration,
        "end_to_end_output_tokens_per_second": output_tokens / duration,
        "api_observed_tpot_seconds": tpot,
        "api_observed_decode_estimate_tokens_per_second": 1 / tpot if tpot else None,
        "sse_chunks": stream["chunks"],
    }


def git_metadata() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {
            "commit": run("rev-parse", "HEAD"),
            "working_tree_dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "working_tree_dirty": None}


def write_atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_atomic(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    write_atomic_bytes(path, encoded)


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen3.8-Flash-Next")
    parser.add_argument("--input-tokens", type=positive_int, default=128)
    parser.add_argument("--output-tokens", type=positive_int, default=256)
    parser.add_argument("--runs", type=positive_int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("benchmark-result.json"))
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--hardware-note")
    parser.add_argument("--server-settings-note")
    parser.add_argument(
        "--prompt-style",
        choices=("repeated-seed", "repo-code", "repo-chat"),
        default="repeated-seed",
    )
    parser.add_argument(
        "--capture-output",
        action="store_true",
        help="save synthetic model text beside the JSON, one file per run",
    )
    args = parser.parse_args(argv)
    if args.warmup < 0:
        parser.error("--warmup must be at least 0")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    endpoint(args.base_url, "")
    return args


def new_report(args: argparse.Namespace) -> dict[str, Any]:
    lock = json.loads((ROOT / "repro.lock.json").read_text())
    return {
        "schema_version": 1,
        "client_version": "1.0",
        "measurement_scope": "new synthetic HTTP streaming measurements",
        "created_at": utc_now(),
        "repository": git_metadata(),
        "repro_lock": lock,
        "client_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interface_compatibility": COMPATIBILITY_SOURCE,
        "endpoint": {"base_url": args.base_url, "model": args.model},
        "declared_environment": {
            "hardware_note": args.hardware_note,
            "server_settings_note": args.server_settings_note,
        },
        "protocol": {
            "input_tokens": args.input_tokens,
            "output_tokens": args.output_tokens,
            "measured_runs": args.runs,
            "warmup_runs": args.warmup,
            "temperature": 0,
            "ignore_eos": True,
            "streaming": True,
            "usage_required": True,
            "prefix_cache_isolation": (
                "unique random cache_salt for every request; model weights, CUDA "
                "graphs, expert caches, allocator state, and host page cache remain warm"
            ),
            "ttft_definition": "request start to first SSE event containing output token IDs",
            "decode_estimate_definition": "tokens after the first output chunk / time until the final output chunk",
            "decode_estimate_caveat": "API-observed; SSE/network buffering and MTP batches prevent kernel-level TPOT interpretation",
            "prefill_caveat": "TTFT includes queueing, scheduling, prefill, and first-token work; input_tokens/TTFT is not pure prefill throughput",
            "output_hash_encoding": (
                "text: UTF-8 with surrogatepass; token IDs: compact UTF-8 JSON "
                "ordered list"
            ),
        },
        "prompt_recipe": {
            "style": args.prompt_style,
            "add_special_tokens": False,
            "data_scope": (
                "public synthetic seed or checked-in runtime overlay Python only; "
                "no imported private data"
            ),
            "method": (
                "repo-chat derives fixed chat-template wrappers from full and empty "
                "server tokenizations, then repeats/truncates only source interior; "
                "other styles repeat/truncate all server-tokenized source IDs"
            ),
            "generation_caveat": (
                "raw greedy completion with ignore_eos=true; length forcing can expose "
                "degenerate repetition and is not a model-quality score"
            ),
        },
        "runs": [],
    }


def main(argv: list[str] | None = None) -> int:
    args, exit_code = parse_args(argv), 0
    report = new_report(args)
    try:
        source_text, source_metadata = prompt_source(args.prompt_style)
        report["prompt_recipe"].update(source_metadata)
        if args.prompt_style == "repo-chat":
            prompt, chat_metadata = repo_chat_prompt(
                args.base_url,
                args.model,
                args.timeout,
                source_text,
                args.input_tokens,
            )
            report["prompt_recipe"].update(chat_metadata)
        else:
            seed_tokens = tokenize(args.base_url, args.model, args.timeout, source_text)
            prompt = build_prompt(seed_tokens, args.input_tokens)
            report["prompt_recipe"].update(
                source_token_ids=seed_tokens, source_token_count=len(seed_tokens)
            )
        for phase, count in (("warmup", args.warmup), ("measured", args.runs)):
            for index in range(count):
                capture_path = (
                    args.output.parent
                    / f"{args.output.stem}.{phase}-{index}.output.txt"
                    if args.capture_output
                    else None
                )
                result = measure_run(
                    args.base_url,
                    args.model,
                    prompt,
                    args.output_tokens,
                    args.timeout,
                    phase,
                    index,
                    capture_path,
                )
                report["runs"].append(result)
                report["last_checkpoint_at"] = utc_now()
                write_atomic(args.output, report)
                exit_code |= result["status"] != "ok"
                if phase == "warmup" and result["status"] != "ok":
                    raise BenchmarkError("warmup did not satisfy the protocol")
    except BenchmarkError as exc:
        report["fatal_error"], exit_code = str(exc), 1
    report["finished_at"] = utc_now()
    measured = [
        run
        for run in report["runs"]
        if run["phase"] == "measured" and run["status"] == "ok"
    ]
    rates = [
        run["api_observed_decode_estimate_tokens_per_second"]
        for run in measured
        if run["api_observed_decode_estimate_tokens_per_second"] is not None
    ]
    tpots = [
        run["api_observed_tpot_seconds"]
        for run in measured
        if run["api_observed_tpot_seconds"] is not None
    ]
    report["summary"] = {
        "valid_measured_runs": len(measured),
        "requested_measured_runs": args.runs,
        "all_measurements_valid": len(measured) == args.runs,
        "mean_request_duration_seconds": (
            sum(run["request_duration_seconds"] for run in measured) / len(measured)
            if measured
            else None
        ),
        "arithmetic_mean_api_observed_decode_estimate_tokens_per_second": (
            sum(rates) / len(rates) if rates else None
        ),
        "reciprocal_mean_api_observed_tpot_tokens_per_second": (
            len(tpots) / sum(tpots) if tpots else None
        ),
    }
    write_atomic(args.output, report)
    print(f"wrote {args.output} ({len(measured)}/{args.runs} valid measured runs)")
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
