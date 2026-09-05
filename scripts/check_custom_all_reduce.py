#!/usr/bin/env python3
"""Two-rank vLLM custom-all-reduce CUDA graph smoke test.

Run only after the serving workload has stopped:

  VLLM_SKIP_P2P_CHECK=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
    torchrun --standalone --nproc-per-node=2 check_custom_all_reduce.py

Requires the pinned vLLM runtime. This validates small eager and graph sums,
not model fit, model quality, or a throughput improvement.
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_SKIP_P2P_CHECK", "1")

import torch
import torch.distributed as dist
from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce


def assert_value(tensor: torch.Tensor, expected: float, label: str) -> None:
    torch.cuda.synchronize()
    got = tensor.float().cpu()
    if not torch.equal(got, torch.full_like(got, expected)):
        raise RuntimeError(f"{label}: expected {expected}, got {got.tolist()}")


def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    if not torch.cuda.can_device_access_peer(rank, 1 - rank):
        raise RuntimeError(f"CUDA peer access missing for rank {rank}")
    dist.init_process_group("gloo")
    communicator = CustomAllreduce(
        dist.group.WORLD,
        torch.device("cuda", rank),
        max_size=64 * 1024,
        max_all_gather_size=64 * 1024,
        max_mnnvl_all_gather_size=64 * 1024,
        max_reduce_scatter_size=64 * 1024,
        max_mnnvl_reduce_scatter_size=64 * 1024,
    )
    if communicator.disabled:
        raise RuntimeError("vLLM custom all-reduce disabled itself")

    eager_input = torch.full((1024,), rank + 1, dtype=torch.bfloat16, device="cuda")
    eager_output = communicator.custom_all_reduce(eager_input)
    if eager_output is None:
        raise RuntimeError("eager custom all-reduce was not selected")
    assert_value(eager_output, 3.0, "eager")

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    dist.barrier()
    with communicator.capture():
        # Match vLLM's normal path: graph tensors are allocated during the
        # custom-AR capture window, then exported to peer ranks on exit.
        with torch.cuda.stream(capture_stream):
            static_input = torch.full(
                (1024,), rank + 2, dtype=torch.bfloat16, device="cuda"
            )
            with torch.cuda.graph(graph, stream=capture_stream):
                graph_output = communicator.custom_all_reduce(static_input)
    torch.cuda.current_stream().wait_stream(capture_stream)
    if graph_output is None:
        raise RuntimeError("captured custom all-reduce was not selected")

    graph.replay()
    assert_value(graph_output, 5.0, "graph replay")
    dist.barrier()
    if rank == 0:
        print("PASS: eager and CUDA-graph vLLM custom all-reduce")
    communicator.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
