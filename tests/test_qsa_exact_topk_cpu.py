#!/usr/bin/env python3
"""CPU correctness test for the opt-in exact QSA block selector."""

from __future__ import annotations

import ast
from pathlib import Path

import torch


QSA = (
    Path(__file__).parent.parent
    / "runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
)


def load_helpers() -> dict[str, object]:
    tree = ast.parse(QSA.read_text())
    wanted = {
        "_QSA_COLS_CACHE",
        "_qsa_cols",
        "_qsa_mask_invisible_",
        "_qsa_exact_topk",
    }
    nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if isinstance(target, ast.Name) and target.id in wanted:
                nodes.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in wanted:
                nodes.append(node)
    namespace: dict[str, object] = {"torch": torch}
    exec(compile(ast.Module(nodes, type_ignores=[]), str(QSA), "exec"), namespace)
    return namespace


def main() -> None:
    namespace = load_helpers()
    exact_topk = namespace["_qsa_exact_topk"]
    torch.manual_seed(0)
    rows, columns, k = 6, 1000, 512
    logits = torch.randn(rows, columns)
    visible = torch.tensor([1000, 600, 128, 8, 3, 0], dtype=torch.int32)
    reference = [
        set(
            torch.topk(
                logits[row, : int(visible[row])],
                min(k, int(visible[row])),
            ).indices.tolist()
        )
        if int(visible[row])
        else set()
        for row in range(rows)
    ]

    blocks = torch.empty(rows, k, dtype=torch.int32)
    exact_topk(logits.clone(), visible, blocks, k, columns)
    for row in range(rows):
        width = min(k, int(visible[row]))
        got = set(blocks[row, :width].tolist())
        assert got == reference[row], (row, got, reference[row])
        assert all(index < int(visible[row]) for index in got)

    again = torch.empty_like(blocks)
    exact_topk(logits.clone(), visible, again, k, columns)
    assert torch.equal(blocks, again)
    print("exact QSA top-k: visible sets match torch.topk and repeat deterministically")


if __name__ == "__main__":
    main()
