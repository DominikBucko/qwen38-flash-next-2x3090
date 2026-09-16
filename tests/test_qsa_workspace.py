"""GPU-free regression for QSA row coverage and score-buffer lifetime.

This executes the packaged selector with fake tensors; GPU numerical checks
remain separate. Weak references make overlapping score allocations visible.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

QSA = Path(__file__).resolve().parents[1] / "runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/ops/qsa.py"


class Tensor:
    def __init__(self, shape, *, base=None, start=0):
        self.shape = tuple(shape)
        self.device = "cuda:0"
        self.base = base
        self.start = start

    def __getitem__(self, key):
        start, stop, step = key.indices(self.shape[0])
        return Tensor((len(range(start, stop, step)), *self.shape[1:]),
                      base=self, start=self.start + start)

    def stride(self, dimension):
        return math.prod(self.shape[dimension + 1:])


class WorkspaceTests(unittest.TestCase):
    def exercise(self, rows):
        alive_scores = weakref.WeakSet()
        peak = 0
        scored = []
        expanded = []

        def score(q, cache, table, mapping, positions, lengths, ratio):
            nonlocal peak
            columns = table.shape[1] * cache.shape[1]
            logits = Tensor((q.shape[0], columns))
            alive_scores.add(logits)
            peak = max(peak, sum(math.prod(item.shape) * 4 for item in alive_scores))
            scored.append((q.start, q.shape[0], columns, ratio))
            return logits, Tensor((q.shape[0],))

        def topk(logits, visible, blocks, workspace, k, columns):
            self.assertEqual(logits.shape, (blocks.shape[0], columns))
            self.assertEqual(k, 512)

        def expand(blocks, positions, lengths, mapping, ratio, k, out):
            expanded.append((out.start, out.shape[0]))
            self.assertEqual(out.shape[1], 2051)

        tree = ast.parse(QSA.read_text())
        nodes = [node for node in tree.body
                 if (isinstance(node, ast.FunctionDef) and node.name == "qsa_select_paged_tokens")
                 or (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                     and node.targets[0].id in {"_LOGITS_WORKSPACE_BYTES", "_TOPK_WORKSPACE_BYTES"})]
        module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[
            ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
        namespace = {
            "torch": SimpleNamespace(empty=lambda shape, **kw: Tensor(shape),
                                     int32="int32", uint8="uint8",
                                     ops=SimpleNamespace(_C=SimpleNamespace(persistent_topk=topk))),
            "current_platform": SimpleNamespace(has_device_capability=lambda version: False),
            "qsa_mqa_paged": score,
            "expand_qsa_block_indices_cuda": expand,
            "_QSA_TOPK_MODE": "0",
        }
        exec(compile(ast.fix_missing_locations(module), str(QSA), "exec"), namespace)
        output = namespace["qsa_select_paged_tokens"](
            Tensor((rows, 2, 128)), Tensor((1024, 128, 1, 128)), Tensor((2, 512)),
            Tensor((rows,)), Tensor((rows,)), Tensor((2,)), 2048, 4)
        self.assertEqual(output.shape, (rows, 2051))
        self.assertEqual(len(alive_scores), 0)
        return peak, scored, expanded

    def test_long_prefill_keeps_only_one_64mib_score_chunk_live(self):
        peak, scored, expanded = self.exercise(4096)
        self.assertEqual(peak, 64 * 1024**2)
        self.assertEqual(scored, [(start, 256, 65536, 4) for start in range(0, 4096, 256)])
        self.assertEqual(expanded, [(start, 256) for start in range(0, 4096, 256)])

    def test_partial_final_chunk_covers_every_row(self):
        peak, scored, expanded = self.exercise(257)
        self.assertEqual(peak, 64 * 1024**2)
        self.assertEqual(scored, [(0, 256, 65536, 4), (256, 1, 65536, 4)])
        self.assertEqual(expanded, [(0, 256), (256, 1)])

    def test_decode_does_not_split_or_change_columns(self):
        peak, scored, expanded = self.exercise(8)
        self.assertEqual(peak, 8 * 65536 * 4)
        self.assertEqual(scored, [(0, 8, 65536, 4)])

    def test_empty_input_does_not_allocate_scores(self):
        self.assertEqual(self.exercise(0), (0, [], []))


if __name__ == "__main__":
    unittest.main()
