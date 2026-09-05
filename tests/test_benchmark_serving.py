#!/usr/bin/env python3
"""GPU-free tests for the public streaming benchmark client."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import threading
import time
import unittest
from hashlib import sha256
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "benchmark_serving", ROOT / "scripts/benchmark_serving.py"
)
assert SPEC and SPEC.loader
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


class FixtureHandler(BaseHTTPRequestHandler):
    completion_mode = "success"
    requests: list[tuple[str, dict]] = []

    def log_message(self, _format: str, *_args: object) -> None:
        pass

    def _request_json(self) -> dict:
        size = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(size))

    def _json(self, status: int, value: dict) -> None:
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        payload = self._request_json()
        type(self).requests.append((self.path, payload))
        if self.path == "/tokenize":
            if "messages" in payload:
                content = payload["messages"][0]["content"]
                source_empty = (
                    content == CLIENT.REPO_CHAT_PREFIX + CLIENT.REPO_CHAT_SUFFIX
                )
                tokens = [100, 101, 900, 901]
                if not source_empty:
                    tokens = [100, 101, 11, 12, 13, 900, 901]
                self._json(
                    200,
                    {"count": len(tokens), "max_model_len": 1024, "tokens": tokens},
                )
            else:
                self._json(
                    200,
                    {
                        "count": 3,
                        "max_model_len": 1024,
                        "tokens": [11, 12, 13],
                    },
                )
            return
        if self.path != "/v1/completions":
            self._json(404, {"error": "not found"})
            return
        if type(self).completion_mode == "http_error":
            self._json(500, {"error": "fixture failure"})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        if type(self).completion_mode == "broken_json":
            self.wfile.write(b"data: {broken}\n\ndata: [DONE]\n\n")
            return
        requested = payload["max_tokens"]
        invalid = type(self).completion_mode == "invalid_usage"
        events = [
            {
                "id": "cmpl-test",
                "choices": [
                    {"index": 0, "text": "a", "token_ids": [21], "finish_reason": None}
                ],
            },
            {
                "id": "cmpl-test",
                "choices": [
                    {
                        "index": 0,
                        "text": "bc",
                        "token_ids": list(range(22, 22 + requested - 1)),
                        "finish_reason": "stop" if invalid else "length",
                    }
                ],
            },
            {
                "id": "cmpl-test",
                "choices": [],
                "usage": {
                    "prompt_tokens": len(payload["prompt"]) - (1 if invalid else 0),
                    "completion_tokens": requested - (1 if invalid else 0),
                    "total_tokens": len(payload["prompt"])
                    + requested
                    - (2 if invalid else 0),
                },
            },
        ]
        for event in events:
            self.wfile.write(f"data: {json.dumps(event)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


class BenchmarkServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self) -> None:
        FixtureHandler.completion_mode = "success"
        FixtureHandler.requests = []

    def test_multievent_stream_and_exact_prompt_recipe(self) -> None:
        seed = CLIENT.tokenize(self.base_url, "fixture", 2)
        prompt = CLIENT.build_prompt(seed, 5)
        self.assertEqual(prompt, [11, 12, 13, 11, 12])
        result = CLIENT.measure_run(
            self.base_url, "fixture", prompt, 3, 2, "measured", 0
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["usage"]["prompt_tokens"], 5)
        self.assertEqual(result["usage"]["completion_tokens"], 3)
        self.assertEqual(len(result["sse_chunks"]), 3)
        saved_chunks = json.dumps(result["sse_chunks"])
        self.assertNotIn('"text"', saved_chunks)
        self.assertNotIn('"a"', saved_chunks)
        self.assertNotIn('"bc"', saved_chunks)
        self.assertEqual(result["output_text_sha256"], sha256(b"abc").hexdigest())
        token_bytes = json.dumps([21, 22, 23], separators=(",", ":")).encode()
        self.assertEqual(
            result["output_token_ids_sha256"], sha256(token_bytes).hexdigest()
        )
        self.assertIsNone(result["captured_output_file"])
        completion = FixtureHandler.requests[-1][1]
        self.assertEqual(completion["prompt"], prompt)
        self.assertEqual(completion["temperature"], 0)
        self.assertTrue(completion["ignore_eos"])
        self.assertFalse(completion["add_special_tokens"])
        self.assertTrue(completion["return_token_ids"])
        self.assertEqual(completion["stream_options"], {"include_usage": True})
        self.assertTrue(completion["cache_salt"])

    def test_invalid_usage_and_truncation_are_detected(self) -> None:
        FixtureHandler.completion_mode = "invalid_usage"
        result = CLIENT.measure_run(
            self.base_url, "fixture", [11, 12, 13], 3, 2, "measured", 0
        )
        self.assertEqual(result["status"], "invalid")
        self.assertTrue(any("prompt_tokens" in error for error in result["errors"]))
        self.assertTrue(any("completion_tokens" in error for error in result["errors"]))
        self.assertTrue(any("finish reasons" in error for error in result["errors"]))

    def test_http_error_has_status_without_response_body(self) -> None:
        FixtureHandler.completion_mode = "http_error"
        with self.assertRaisesRegex(CLIENT.BenchmarkError, "HTTP 500"):
            CLIENT.measure_run(self.base_url, "fixture", [11, 12], 2, 2, "measured", 0)

    def test_broken_stream_is_rejected(self) -> None:
        FixtureHandler.completion_mode = "broken_json"
        with self.assertRaisesRegex(CLIENT.BenchmarkError, "invalid JSON"):
            CLIENT.measure_run(self.base_url, "fixture", [11, 12], 2, 2, "measured", 0)

    def test_atomic_json_write_replaces_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "result.json"
            target.write_text("old")
            CLIENT.write_atomic(target, {"status": "ok"})
            self.assertEqual(json.loads(target.read_text()), {"status": "ok"})
            self.assertEqual(list(target.parent.glob(f".{target.name}.*")), [])

    def test_optional_output_capture_uses_only_filename_in_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "run.output.txt"
            result = CLIENT.measure_run(
                self.base_url, "fixture", [11, 12], 3, 2, "measured", 0, capture
            )
            self.assertEqual(capture.read_text(), "abc")
            self.assertEqual(result["captured_output_file"], capture.name)
            self.assertNotIn(str(capture.parent), json.dumps(result))

    def test_capture_io_is_excluded_from_request_duration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "run.output.txt"
            original = CLIENT.write_atomic_bytes

            def slow_capture(path: Path, value: bytes) -> None:
                time.sleep(0.08)
                original(path, value)

            started = time.perf_counter()
            with mock.patch.object(
                CLIENT, "write_atomic_bytes", side_effect=slow_capture
            ):
                result = CLIENT.measure_run(
                    self.base_url, "fixture", [11, 12], 3, 2, "measured", 0, capture
                )
            wall_time = time.perf_counter() - started
            self.assertGreater(wall_time - result["request_duration_seconds"], 0.06)

    def test_repo_code_corpus_is_sorted_and_python_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "runtime/vllm-overlay"
            (overlay / "nested").mkdir(parents=True)
            (overlay / "z.py").write_text("z = 1\n")
            (overlay / "nested/a.py").write_text("a = 1\n")
            (overlay / "private.txt").write_text("excluded")
            source, metadata = CLIENT.prompt_source("repo-code", root)
            self.assertLess(source.index("nested/a.py"), source.index("z.py"))
            self.assertNotIn("excluded", source)
            self.assertEqual(
                [item["path"] for item in metadata["source_files"]],
                ["runtime/vllm-overlay/nested/a.py", "runtime/vllm-overlay/z.py"],
            )
            source_again, metadata_again = CLIENT.prompt_source("repo-code", root)
            self.assertEqual((source, metadata), (source_again, metadata_again))

    def test_default_prompt_style_is_unchanged(self) -> None:
        args = CLIENT.parse_args([])
        source, metadata = CLIENT.prompt_source(args.prompt_style)
        self.assertEqual(args.prompt_style, "repeated-seed")
        self.assertEqual(source, CLIENT.SEED_TEXT)
        self.assertEqual(metadata["seed_text"], CLIENT.SEED_TEXT)

    def test_repo_chat_preserves_server_wrappers_and_exact_budget(self) -> None:
        prompt, recipe = CLIENT.repo_chat_prompt(
            self.base_url, "fixture", 2, "public source", 9
        )
        self.assertEqual(prompt, [100, 101, 11, 12, 13, 11, 12, 900, 901])
        self.assertEqual(len(prompt), 9)
        self.assertEqual(recipe["fixed_prefix_token_ids"], [100, 101])
        self.assertEqual(recipe["fixed_suffix_token_ids"], [900, 901])
        self.assertEqual(recipe["source_token_ids"], [11, 12, 13])
        self.assertEqual(recipe["source_complete_cycles_in_prompt"], 1)
        self.assertEqual(recipe["source_prefix_tokens_after_complete_cycles"], 2)
        self.assertTrue(recipe["includes_complete_source_corpus"])
        self.assertEqual(len(recipe["fixed_prefix_token_ids_sha256"]), 64)
        self.assertEqual(len(recipe["fixed_suffix_token_ids_sha256"]), 64)
        self.assertEqual(len(recipe["source_token_ids_sha256"]), 64)
        chat_requests = [
            payload
            for path, payload in FixtureHandler.requests
            if path == "/tokenize" and "messages" in payload
        ]
        self.assertEqual(len(chat_requests), 2)
        self.assertTrue(all(item["add_generation_prompt"] for item in chat_requests))
        self.assertTrue(all(not item["add_special_tokens"] for item in chat_requests))
        self.assertGreaterEqual(len(CLIENT.REPO_CHAT_PREFIX.split()), 20)

    def test_repo_chat_rejects_budget_too_small_for_wrappers(self) -> None:
        with self.assertRaisesRegex(CLIENT.BenchmarkError, "at least 5"):
            CLIENT.repo_chat_prompt(self.base_url, "fixture", 2, "public source", 4)

    def test_repo_chat_style_uses_repo_corpus_but_distinct_recipe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            overlay = root / "runtime/vllm-overlay"
            overlay.mkdir(parents=True)
            (overlay / "source.py").write_text("answer = 42\n")
            code_source, code_metadata = CLIENT.prompt_source("repo-code", root)
            chat_source, chat_metadata = CLIENT.prompt_source("repo-chat", root)
            self.assertEqual(chat_source, code_source)
            self.assertEqual(chat_metadata, code_metadata)
        args = CLIENT.parse_args(["--prompt-style", "repo-chat"])
        report = CLIENT.new_report(args)
        self.assertEqual(report["prompt_recipe"]["style"], "repo-chat")
        self.assertIn("chat-template wrappers", report["prompt_recipe"]["method"])

    def test_report_is_checkpointed_after_each_run(self) -> None:
        completed = {
            "phase": "measured",
            "index": 0,
            "status": "ok",
            "request_duration_seconds": 1.0,
            "api_observed_tpot_seconds": 0.5,
            "api_observed_decode_estimate_tokens_per_second": 2.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            with mock.patch.object(
                CLIENT, "measure_run", side_effect=[completed, KeyboardInterrupt]
            ):
                with self.assertRaises(KeyboardInterrupt):
                    CLIENT.main(
                        [
                            "--base-url",
                            self.base_url,
                            "--model",
                            "fixture",
                            "--runs",
                            "2",
                            "--warmup",
                            "0",
                            "--output",
                            str(output),
                        ]
                    )
            saved = json.loads(output.read_text())
            self.assertEqual(saved["runs"], [completed])
            self.assertEqual(len(saved["client_script_sha256"]), 64)
            self.assertEqual(saved["prompt_recipe"]["style"], "repeated-seed")
            inspected = saved["interface_compatibility"]["inspected_source"]
            self.assertIn("not claimed as deployed", inspected["note"])

    def test_summary_uses_reciprocal_mean_tpot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            status = CLIENT.main(
                [
                    "--base-url",
                    self.base_url,
                    "--model",
                    "fixture",
                    "--output-tokens",
                    "3",
                    "--runs",
                    "2",
                    "--warmup",
                    "0",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(status, 0)
            saved = json.loads(output.read_text())
            tpots = [run["api_observed_tpot_seconds"] for run in saved["runs"]]
            expected = len(tpots) / sum(tpots)
            self.assertAlmostEqual(
                saved["summary"]["reciprocal_mean_api_observed_tpot_tokens_per_second"],
                expected,
            )
            rates = [
                run["api_observed_decode_estimate_tokens_per_second"]
                for run in saved["runs"]
            ]
            self.assertAlmostEqual(
                saved["summary"][
                    "arithmetic_mean_api_observed_decode_estimate_tokens_per_second"
                ],
                sum(rates) / len(rates),
            )


if __name__ == "__main__":
    unittest.main()
