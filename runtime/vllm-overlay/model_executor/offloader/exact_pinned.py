"""Opt-in experiment: avoid power-of-two padding in large UVA weight backing."""
from functools import lru_cache
import os
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='qwen38_exact_pinned_v1',
                sources=[str(Path(__file__).with_suffix('.cpp'))],
                with_cuda=True, extra_cflags=['-O2'], verbose=False)


def pin_cpu_tensor(source):
    # Keep ordinary staging tensors and strided/empty tensors on the existing
    # allocator. This experiment is restricted to large immutable weights.
    if (os.environ.get('VLLM_EXACT_PINNED_WEIGHTS') != '1'
            or source.numel() * source.element_size() < 64 * 1024**2
            or not source.is_contiguous()):
        return source.pin_memory()
    return extension().allocate_copy(source)


def report_exact_pinned_memory(device):
    if device.type != 'cuda' or os.environ.get('VLLM_EXACT_PINNED_WEIGHTS') != '1':
        return
    from vllm.logger import init_logger
    stats = torch.cuda.memory.host_memory_stats()
    init_logger(__name__).info(
        'Exact pinned weights: live_bytes=%d native_active_bytes=%d native_owned_bytes=%d',
        extension().live_bytes(), stats.get('active_bytes.current', 0),
        stats.get('allocated_bytes.current', 0))
