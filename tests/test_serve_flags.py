#!/usr/bin/env python3
"""GPU-free tests for the container serving launchers."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


class ServeFlagsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = Path(self.temporary.name)

        self.model = self.fixture / "model"
        self.mtp_model = self.model / "runtime/mtp-int4-g32"
        self.mtp_model.mkdir(parents=True)
        (self.model / "model.safetensors.index.json").write_text("{}\n")
        (self.mtp_model / "model.safetensors.index.json").write_text("{}\n")

        self.rankings = self.fixture / "static_hot_cache_rankings.json"
        self.rankings.write_text("[]\n")
        self.profile = self.fixture / "2x3090-128gb.env"
        self.profile.write_text((ROOT / "configs/2x3090-128gb.env").read_text())

        launcher_text = (ROOT / "scripts/serve-container.sh").read_text()
        launcher_text = launcher_text.replace(
            "profile=/opt/qwen38/configs/2x3090-128gb.env",
            f"profile={self.profile}",
        ).replace(
            "rankings=/workspace/static_hot_cache_rankings.json",
            f"rankings={self.rankings}",
        )
        self.launcher = self.fixture / "serve-container.sh"
        self.launcher.write_text(launcher_text)
        self.launcher.chmod(0o755)

        self.bin_dir = self.fixture / "bin"
        self.bin_dir.mkdir()
        self.capture = self.fixture / "capture.json"
        fake_vllm = self.bin_dir / "vllm"
        fake_vllm.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE_PATH'], 'w') as target:\n"
            "    json.dump({'argv': sys.argv[1:], 'env': {\n"
            "        'DISABLE_CUSTOM_ALL_REDUCE': os.environ.get('DISABLE_CUSTOM_ALL_REDUCE'),\n"
            "        'PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),\n"
            "        'VLLM_PLE_CPU_OFFLOAD': os.environ.get('VLLM_PLE_CPU_OFFLOAD'),\n"
            "        'ENABLE_VISION': os.environ.get('ENABLE_VISION'),\n"
            "        'VISION_MAX_IMAGES': os.environ.get('VISION_MAX_IMAGES'),\n"
            "        'VISION_MAX_PIXELS': os.environ.get('VISION_MAX_PIXELS'),\n"
            "        'VLLM_WNA16_STATIC_HOT_CACHE_SIZE': os.environ.get('VLLM_WNA16_STATIC_HOT_CACHE_SIZE'),\n"
            "        'VLLM_WNA16_STATIC_HOT_CACHE_FILE': os.environ.get('VLLM_WNA16_STATIC_HOT_CACHE_FILE'),\n"
            "    }}, target)\n"
        )
        fake_vllm.chmod(0o755)
        fake_docker = self.bin_dir / "docker"
        fake_docker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os, sys\n"
            "with open(os.environ['CAPTURE_PATH'], 'w') as target:\n"
            "    json.dump({'argv': sys.argv[1:]}, target)\n"
        )
        fake_docker.chmod(0o755)

    def run_launcher(
        self, value: str | None = None, allocator: str | None = None,
        hot_cache: str | None = None,
        settings: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PATH"] = f"{self.bin_dir}{os.pathsep}{env['PATH']}"
        env["CAPTURE_PATH"] = str(self.capture)
        env.pop("DISABLE_CUSTOM_ALL_REDUCE", None)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("VLLM_WNA16_STATIC_HOT_CACHE_SIZE", None)
        for name in ("ENABLE_VISION", "VISION_MAX_IMAGES", "VISION_MAX_PIXELS",
                     "QWEN38_ASYNC_SCHEDULING", "JIT_CACHE_DIR"):
            env.pop(name, None)
        if value is not None:
            env["DISABLE_CUSTOM_ALL_REDUCE"] = value
        if allocator is not None:
            env["PYTORCH_CUDA_ALLOC_CONF"] = allocator
        if hot_cache is not None:
            env["VLLM_WNA16_STATIC_HOT_CACHE_SIZE"] = hot_cache
        env.update(settings or {})
        return subprocess.run(
            [str(self.launcher), str(self.model)],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def captured(self) -> dict[str, object]:
        return json.loads(self.capture.read_text())

    def run_docker_launcher(
        self, settings: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env.update(
            {
                "PATH": f"{self.bin_dir}{os.pathsep}{env['PATH']}",
                "CAPTURE_PATH": str(self.capture),
                "MODEL_DIR": str(self.model),
            }
        )
        env.pop("DISABLE_CUSTOM_ALL_REDUCE", None)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        for name in ("ENABLE_VISION", "VISION_MAX_IMAGES", "VISION_MAX_PIXELS"):
            env.pop(name, None)
        env.update(settings or {})
        return subprocess.run(
            [str(ROOT / "scripts/docker_serve.sh")],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_default_disables_custom_all_reduce(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = self.captured()
        self.assertEqual(captured["argv"].count("--disable-custom-all-reduce"), 1)
        self.assertEqual(captured["env"]["DISABLE_CUSTOM_ALL_REDUCE"], "1")
        self.assertEqual(
            captured["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True"
        )

    def test_zero_allows_custom_all_reduce(self) -> None:
        allocator = "max_split_size_mb:512, expandable_segments : False"
        result = self.run_launcher("0", allocator)
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = self.captured()
        self.assertNotIn("--disable-custom-all-reduce", captured["argv"])
        self.assertEqual(captured["env"]["DISABLE_CUSTOM_ALL_REDUCE"], "0")
        self.assertEqual(captured["env"]["PYTORCH_CUDA_ALLOC_CONF"], allocator)

    def test_text_only_remains_the_default(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertIn("--language-model-only", argv)
        self.assertNotIn("--limit-mm-per-prompt", argv)

    def test_vision_enables_bounded_images_without_changing_precision(self) -> None:
        result = self.run_launcher(hot_cache="80", settings={"ENABLE_VISION": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertNotIn("--language-model-only", argv)
        self.assertEqual(json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]),
                         {"image": 1, "video": 0})
        self.assertEqual(json.loads(argv[argv.index("--mm-processor-kwargs") + 1]),
                         {"min_pixels": 65536, "max_pixels": 1048576})
        self.assertEqual(argv[argv.index("--dtype") + 1], "bfloat16")
        self.assertEqual(argv[argv.index("--max-model-len") + 1], "262144")
        self.assertEqual(json.loads(argv[argv.index("--speculative-config") + 1])[
            "num_speculative_tokens"], 3)
        self.assertEqual(argv[argv.index("--mm-encoder-tp-mode") + 1], "weights")

    def test_vision_limits_can_be_overridden(self) -> None:
        result = self.run_launcher(settings={"ENABLE_VISION": "1",
            "VISION_MAX_IMAGES": "2", "VISION_MAX_PIXELS": "262144"})
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertEqual(json.loads(argv[argv.index("--limit-mm-per-prompt") + 1]),
                         {"image": 2, "video": 0})
        self.assertEqual(json.loads(argv[argv.index("--mm-processor-kwargs") + 1])[
            "max_pixels"], 262144)

    def test_invalid_vision_settings_fail_before_exec(self) -> None:
        for settings in ({"ENABLE_VISION": "yes"},
                         {"ENABLE_VISION": "1", "VISION_MAX_IMAGES": "0"},
                         {"ENABLE_VISION": "1", "VISION_MAX_IMAGES": "x"},
                         {"ENABLE_VISION": "1", "VISION_MAX_PIXELS": "32768"},
                         {"ENABLE_VISION": "1", "VISION_MAX_PIXELS": "16777217"},
                         {"ENABLE_VISION": "1", "VISION_MAX_PIXELS": "1.5"},
                         {"ENABLE_VISION": "1", "VISION_MAX_PIXELS": "-1"}):
            with self.subTest(settings=settings):
                result = self.run_launcher(settings=settings)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(self.capture.exists())

    def test_docker_launcher_forwards_vision_settings(self) -> None:
        settings = {"ENABLE_VISION": "1", "VISION_MAX_IMAGES": "1",
                    "VISION_MAX_PIXELS": "1048576"}
        result = self.run_docker_launcher(settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        for name, value in settings.items():
            self.assertIn(f"{name}={value}", self.captured()["argv"])

    def test_default_uses_hot84_without_reducing_context(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = self.captured()
        self.assertEqual(captured["env"]["VLLM_WNA16_STATIC_HOT_CACHE_SIZE"], "84")
        argv = captured["argv"]
        for flag, value in (("--max-model-len", "262144"),
                            ("--kv-cache-memory-bytes", "4429185024"),
                            ("--max-num-batched-tokens", "4096")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        example = (ROOT / ".env.example").read_text()
        self.assertIn("\nVLLM_WNA16_STATIC_HOT_CACHE_SIZE=84\n", example)

    def fast_profile(self) -> dict[str, str]:
        values = {}
        for line in (ROOT / "configs/fast-256k.env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                name, _, value = line.partition("=")
                values[name] = value
        return values

    def test_async_scheduling_is_off_by_default(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertIn("--no-async-scheduling", argv)
        self.assertNotIn("--async-scheduling", argv)

    def test_async_scheduling_is_opt_in(self) -> None:
        result = self.run_launcher(settings={"QWEN38_ASYNC_SCHEDULING": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertIn("--async-scheduling", argv)
        self.assertNotIn("--no-async-scheduling", argv)

    def test_invalid_async_scheduling_fails_before_exec(self) -> None:
        result = self.run_launcher(settings={"QWEN38_ASYNC_SCHEDULING": "yes"})
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.capture.exists())

    def test_fast_profile_keeps_full_context_and_enables_fast_paths(self) -> None:
        profile = self.fast_profile()
        self.assertEqual(profile["QWEN38_STREAM_STAGE"], "1")
        self.assertEqual(profile["QWEN38_PLE_PREFAULT"], "1")
        self.assertEqual(profile["VLLM_MTP_DRAFT_VOCAB_RANGES"],
                         "[[0,65536],[248044,248320]]")
        settings = {k: v for k, v in profile.items() if k != "JIT_CACHE_DIR"}
        result = self.run_launcher(settings=settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = self.captured()
        argv = captured["argv"]
        self.assertIn("--async-scheduling", argv)
        self.assertNotIn("--disable-custom-all-reduce", argv)
        self.assertEqual(captured["env"]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:False")
        self.assertEqual(captured["env"]["VLLM_WNA16_STATIC_HOT_CACHE_SIZE"], "84")
        for flag, value in (("--max-model-len", "262144"),
                            ("--kv-cache-memory-bytes", "4429185024"),
                            ("--max-num-batched-tokens", "4096"),
                            ("--kv-cache-dtype", "auto")):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertEqual(json.loads(argv[argv.index("--speculative-config") + 1])[
            "num_speculative_tokens"], 3)

    def test_docker_launcher_forwards_fast_profile_and_jit_cache(self) -> None:
        profile = self.fast_profile()
        profile["JIT_CACHE_DIR"] = str(self.fixture / "jit-cache")
        result = self.run_docker_launcher(profile)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        for name, value in profile.items():
            if name != "JIT_CACHE_DIR":
                self.assertIn(f"{name}={value}", argv)
        cache = (self.fixture / "jit-cache").resolve()
        self.assertIn(f"{cache}/humming:/root/.humming", argv)
        self.assertIn(f"{cache}/triton:/root/.triton", argv)
        self.assertTrue((cache / "humming").is_dir())

    def test_docker_launcher_without_jit_cache_adds_no_mounts(self) -> None:
        result = self.run_docker_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertFalse(any(":/root/.humming" in item for item in argv))

    def test_hot88_remains_an_explicit_override(self) -> None:
        result = self.run_launcher(hot_cache="88")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.captured()["env"]["VLLM_WNA16_STATIC_HOT_CACHE_SIZE"], "88")

    def test_custom_all_reduce_with_expandable_segments_fails_before_exec(self) -> None:
        result = self.run_launcher(
            "0", "max_split_size_mb:512, expandable_segments : True "
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("incompatible with expandable_segments:True", result.stderr)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_invalid_value_fails_before_exec(self) -> None:
        result = self.run_launcher("yes")
        self.assertEqual(result.returncode, 2)
        self.assertIn("DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1", result.stderr)
        self.assertFalse(self.capture.exists())

    def test_original_serving_flags_and_environment_are_retained(self) -> None:
        result = self.run_launcher("0", "expandable_segments:False")
        self.assertEqual(result.returncode, 0, result.stderr)
        captured = self.captured()
        argv = captured["argv"]
        self.assertEqual(argv[:2], ["serve", str(self.model)])
        for flag, value in (
            ("--tensor-parallel-size", "2"),
            ("--all2all-backend", "allgather_reducescatter"),
            ("--moe-backend", "humming"),
            ("--max-model-len", "262144"),
        ):
            self.assertEqual(argv[argv.index(flag) + 1], value)
        self.assertIn("--enable-expert-parallel", argv)
        self.assertIn("--no-async-scheduling", argv)
        speculative = argv[argv.index("--speculative-config") + 1]
        self.assertEqual(json.loads(speculative)["model"], str(self.mtp_model))
        self.assertEqual(captured["env"]["VLLM_PLE_CPU_OFFLOAD"], "1")
        self.assertEqual(
            captured["env"]["VLLM_WNA16_STATIC_HOT_CACHE_FILE"],
            str(self.rankings),
        )

    def test_docker_launcher_forwards_custom_ar_and_allocator_settings(self) -> None:
        result = self.run_docker_launcher(
            {
                "DISABLE_CUSTOM_ALL_REDUCE": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:False",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertIn("DISABLE_CUSTOM_ALL_REDUCE=0", argv)
        self.assertIn("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False", argv)

    def test_docker_launcher_preserves_container_defaults_when_unset(self) -> None:
        result = self.run_docker_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        self.assertFalse(
            any(value.startswith("DISABLE_CUSTOM_ALL_REDUCE=") for value in argv)
        )
        self.assertFalse(
            any(value.startswith("PYTORCH_CUDA_ALLOC_CONF=") for value in argv)
        )
        self.assertFalse(any(value.startswith("ENABLE_VISION=") for value in argv))

    def test_docker_launcher_forwards_two_client_capacity_settings(self) -> None:
        settings = {
            "MAX_NUM_SEQS": "2",
            "MAX_MODEL_LEN": "131072",
            "MAX_NUM_BATCHED_TOKENS": "2048",
            "KV_CACHE_MEMORY_BYTES": "4697620480",
            "VLLM_WNA16_STATIC_HOT_CACHE_SIZE": "80",
            "MTP_DEPTH": "3",
        }
        result = self.run_docker_launcher(settings)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = self.captured()["argv"]
        for name, value in settings.items():
            self.assertIn(f"{name}={value}", argv)

    def test_compose_safety_defaults_match_effective_launcher_defaults(self) -> None:
        result = self.run_launcher()
        self.assertEqual(result.returncode, 0, result.stderr)
        launcher_env = self.captured()["env"]
        compose = (ROOT / "docker/compose.yaml").read_text()
        for name in ("DISABLE_CUSTOM_ALL_REDUCE", "PYTORCH_CUDA_ALLOC_CONF",
                     "VLLM_WNA16_STATIC_HOT_CACHE_SIZE", "ENABLE_VISION",
                     "VISION_MAX_IMAGES", "VISION_MAX_PIXELS"):
            match = re.search(
                rf'^\s+{name}:\s+"\$\{{{name}:-([^}}]+)\}}"\s*$',
                compose,
                re.MULTILINE,
            )
            self.assertIsNotNone(match, f"missing Compose default for {name}")
            assert match is not None
            self.assertEqual(match.group(1), launcher_env[name])


if __name__ == "__main__":
    unittest.main()
