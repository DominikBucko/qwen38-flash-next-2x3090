"""Runtime abliteration equals the weight edit W' = W - r (r^T W) (CPU, needs torch; runs in the image)."""

import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/abliteration.py"
DIRECTION = ROOT / "configs/abliteration/orcarouter.json"

try:
    import torch
except ImportError:  # the GPU-free CI job has no torch; the image does
    torch = None


def load_module():
    spec = importlib.util.spec_from_file_location("q38_abliteration", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DirectionFileTest(unittest.TestCase):
    def test_shipped_direction_is_a_unit_vector_with_provenance(self):
        doc = json.loads(DIRECTION.read_text())
        values = doc["refusal_direction"]
        self.assertEqual(doc["hidden_size"], 2560)
        self.assertEqual(len(values), 2560)
        self.assertAlmostEqual(math.sqrt(sum(v * v for v in values)), 1.0, places=5)
        self.assertEqual(doc["source"], "orcarouter/Qwen3.8-Flash-Next-Uncensored")
        self.assertEqual(len(doc["revision"]), 40)
        self.assertTrue(doc["verification"]["untouched_control_identical"])


@unittest.skipIf(torch is None, "torch not installed")
class ProjectionTest(unittest.TestCase):
    def setUp(self):
        self.ab = load_module()
        torch.manual_seed(0)
        self.r = torch.randn(64, dtype=torch.float64)
        self.r /= self.r.norm()

    def edited(self, w):
        return w - torch.outer(self.r, self.r @ w)

    def test_block_output_projection_equals_weight_edit(self):
        w = torch.randn(64, 48, dtype=torch.float64)
        x = torch.randn(7, 48, dtype=torch.float64)
        y = x @ w.T
        self.ab.project_(y, self.r)
        torch.testing.assert_close(y, x @ self.edited(w).T)

    def test_moe_weighted_sum_equals_editing_every_down_projection(self):
        experts = [torch.randn(64, 32, dtype=torch.float64) for _ in range(5)]
        acts = [torch.randn(6, 32, dtype=torch.float64) for _ in experts]
        gates = torch.rand(6, len(experts), dtype=torch.float64)
        out = sum(gates[:, e : e + 1] * (acts[e] @ w.T) for e, w in enumerate(experts))
        ref = sum(gates[:, e : e + 1] * (acts[e] @ self.edited(w).T) for e, w in enumerate(experts))
        self.ab.project_(out, self.r)
        torch.testing.assert_close(out, ref)

    def test_embedding_rows(self):
        table = torch.randn(10, 64, dtype=torch.float64)
        ids = torch.tensor([3, 3, 9, 0])
        rows = table[ids].clone()
        self.ab.project_(rows, self.r)
        torch.testing.assert_close(rows, (table - torch.outer(table @ self.r, self.r))[ids])
        self.assertLess(float((rows @ self.r).abs().max()), 1e-12)

    def test_non_contiguous_input_and_noop(self):
        y = torch.randn(48, 64, dtype=torch.float64).T.contiguous().T  # non-contiguous view
        ref = y - torch.outer(y @ self.r, self.r)
        self.ab.project_(y, self.r)
        torch.testing.assert_close(y, ref)
        z = torch.randn(3, 64)
        self.assertIs(self.ab.project_(z, None), z)

    def test_switch_resolution_and_loading(self):
        with mock.patch.dict(os.environ, {"QWEN38_ABLITERATION": ""}):
            self.assertIsNone(self.ab.direction_path())
            self.assertIsNone(self.ab.load_direction(2560))
        with mock.patch.dict(os.environ, {"QWEN38_ABLITERATION": "orcarouter"}):
            self.assertEqual(self.ab.direction_path(), "/opt/qwen38/abliteration/orcarouter.json")
        with mock.patch.dict(os.environ, {"QWEN38_ABLITERATION": str(DIRECTION)}):
            r = self.ab.load_direction(2560)
            self.assertEqual(tuple(r.shape), (2560,))
            self.assertAlmostEqual(float(r.norm()), 1.0, places=6)
            with self.assertRaises(ValueError):
                self.ab.load_direction(4096)
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "zero.json"
            bad.write_text(json.dumps({"refusal_direction": [0.0] * 8}))
            with mock.patch.dict(os.environ, {"QWEN38_ABLITERATION": str(bad)}):
                with self.assertRaises(ValueError):
                    self.ab.load_direction(8)


if __name__ == "__main__":
    unittest.main()
