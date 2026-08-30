# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Base class for PLE layers that support cross-process CPU offload.

The CPU process publishes into page-locked shared host memory. The model's
main CUDA stream waits on a mapped host flag, copies the result, and releases
the host buffer. Keeping all H2D CUDA calls on the model thread avoids Driver
lock inversion with lazy Triton/CUDA module loading.

Typical decode-step timeline
----------------------------
Offload process (CPU)                  GPU main stream (forward)
-----------------------------          ------------------------------
wait consumed==1; consumed=0           ... other GPU ops ...
forward_impl(); ready=1                 WaitValue32(ready==1) <- in Graph
                                       H2D(host -> GPU)       <- in Graph
                                       ready=0; consumed=1    <- in Graph
                                       ... model consumes output ...
"""

import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, cast

import torch
from cuda.bindings import driver as cuda_driver
from cuda.bindings.driver import CUstreamWaitValue_flags
from torch import nn

import vllm.envs as envs
from vllm.utils.torch_utils import direct_register_custom_op

# Module-level flag set to True inside the offload subprocess.
# Because the offload process and GPU worker processes are separate OS
# processes (spawned via multiprocessing), each has its own memory space.
# A plain module-level bool is sufficient -- no thread-local storage needed.
_offload_worker_flag = False


def is_offload_process() -> bool:
    """Return True inside the dedicated PLE CPU-offload subprocess."""
    return _offload_worker_flag


def mark_as_offload_worker() -> None:
    """Mark the current process as the dedicated PLE offload worker."""
    global _offload_worker_flag
    _offload_worker_flag = True


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class CpuGpuSemaphore:
    """Cross-process semaphore backed by a one-element int32 CUDA tensor.

    The flag is stored in a regular GPU tensor that is shared through
    PyTorch's CUDA IPC mechanism. Both processes obtain device pointers that
    map to the same physical GPU memory, so a stream-memory wait in the GPU
    worker observes the write issued by the offload process.
    """

    RESET_VALUE = 0
    DONE_VALUE = 1

    def __init__(self, device: torch.device) -> None:
        self._flag_tensor = torch.zeros(1, dtype=torch.int32, device=device)

    @classmethod
    def from_ipc_tensor(cls, flag_tensor: torch.Tensor) -> "CpuGpuSemaphore":
        """Construct a semaphore from a CUDA tensor received through IPC."""
        semaphore = cls.__new__(cls)
        semaphore._flag_tensor = flag_tensor
        return semaphore

    @property
    def flag_tensor(self) -> torch.Tensor:
        """Return the CUDA tensor used to share the semaphore through IPC."""
        return self._flag_tensor

    def reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.RESET_VALUE,
                0,
            ),
            "CpuGpuSemaphore.reset",
        )

    def signal(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WriteValue32(flag=1)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.DONE_VALUE,
                0,
            ),
            "CpuGpuSemaphore.signal",
        )

    def wait_reset(self, stream: torch.cuda.Stream | None = None) -> None:
        """Enqueue ``WaitValue32(flag==0)`` on ``stream``."""
        if stream is None:
            stream = torch.cuda.current_stream()
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self._flag_tensor.data_ptr()),
                self.RESET_VALUE,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "CpuGpuSemaphore.wait_reset",
        )


# ---------------------------------------------------------------------------
# Custom op: vllm::ple_offload_host_copy
# ---------------------------------------------------------------------------
# Wrap the mapped-host wait, H2D copy, and release writes as one opaque graph
# node. hidden_states creates a Dynamo dependency edge that prevents the node
# from moving before preceding model work.
# ---------------------------------------------------------------------------


def _ple_offload_host_copy_impl(
    hidden_states: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    cpu_output_buffer: torch.Tensor,
    ready_device_ptr: int,
    consumed_device_ptr: int,
) -> None:
    """Copy one published host result on the current model stream."""
    stream = torch.cuda.current_stream()
    cuda_stream = cuda_driver.CUstream(stream.cuda_stream)
    _cuda_check(
        cuda_driver.cuStreamWaitValue32(
            cuda_stream,
            cuda_driver.CUdeviceptr(ready_device_ptr),
            1,
            CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
        ),
        "cuStreamWaitValue32(PLE host ready)",
    )
    num_bytes = (
        hidden_states.shape[0]
        * cpu_output_buffer.shape[1]
        * cpu_output_buffer.element_size()
    )
    _cuda_check(
        cuda_driver.cuMemcpyHtoDAsync(
            cuda_driver.CUdeviceptr(gpu_output_buffer.data_ptr()),
            cpu_output_buffer.data_ptr(),
            num_bytes,
            cuda_stream,
        ),
        "cuMemcpyHtoDAsync(PLE host output)",
    )
    _cuda_check(
        cuda_driver.cuStreamWriteValue32(
            cuda_stream,
            cuda_driver.CUdeviceptr(ready_device_ptr),
            0,
            0,
        ),
        "cuStreamWriteValue32(PLE ready reset)",
    )
    _cuda_check(
        cuda_driver.cuStreamWriteValue32(
            cuda_stream,
            cuda_driver.CUdeviceptr(consumed_device_ptr),
            1,
            0,
        ),
        "cuStreamWriteValue32(PLE consumed)",
    )


def _ple_offload_host_copy_fake(
    hidden_states: torch.Tensor,
    gpu_output_buffer: torch.Tensor,
    cpu_output_buffer: torch.Tensor,
    ready_device_ptr: int,
    consumed_device_ptr: int,
) -> None:
    """Represent the side-effect-only transfer during Dynamo tracing."""
    pass


direct_register_custom_op(
    op_name="ple_offload_host_copy",
    op_func=_ple_offload_host_copy_impl,
    mutates_args=["gpu_output_buffer"],
    fake_impl=_ple_offload_host_copy_fake,
)


class PleOffloadLayer(nn.Module, ABC):
    """Base class for embedding-like PLE layers that can run on CPU.

    Subclasses implement :meth:`forward_impl`, which contains the regular
    on-device computation. In a GPU worker with PLE offload enabled, subclass
    constructors are skipped so their large weights are never allocated. The
    CPU process constructs a complete copy and owns those weights instead.
    """

    _is_cpu_offloaded = False
    _gpu_output_buffer: torch.Tensor
    _cpu_output_buffer: torch.Tensor
    _ready_device_ptr: int
    _consumed_device_ptr: int

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Skip subclass initialization in cross-process GPU-worker mode."""
        super().__init_subclass__(**kwargs)
        original_init_obj = cls.__dict__.get("__init__")
        if original_init_obj is None or not callable(original_init_obj):
            return
        original_init = cast(Callable[..., None], original_init_obj)

        @functools.wraps(original_init)
        def guarded_init(
            self: "PleOffloadLayer", *args: object, **kwargs: object
        ) -> None:
            if envs.VLLM_PLE_CPU_OFFLOAD and not is_offload_process():
                nn.Module.__init__(self)
                return
            original_init(self, *args, **kwargs)

        cls.__init__ = guarded_init  # type: ignore[method-assign, assignment]

    @classmethod
    def get_target_device(cls) -> torch.device:
        """Return CPU for the offload process and the active GPU otherwise."""
        if envs.VLLM_PLE_CPU_OFFLOAD:
            return torch.device("cpu")
        return torch.device("cuda", torch.accelerator.current_device_index())

    @abstractmethod
    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Execute the actual embedding computation on the owning device."""
        raise NotImplementedError

    def get_offload_output_dtype(self, default_dtype: torch.dtype) -> torch.dtype:
        """Return the dtype used by cross-process output buffers."""
        return default_dtype

    def setup_cross_process_offload(
        self,
        gpu_output_buffer: torch.Tensor,
        cpu_output_buffer: torch.Tensor,
        ready_device_ptr: int,
        consumed_device_ptr: int,
    ) -> None:
        """Configure the GPU placeholder with stable host/device resources."""
        self._is_cpu_offloaded = True
        self._gpu_output_buffer = gpu_output_buffer
        self._cpu_output_buffer = cpu_output_buffer
        self._ready_device_ptr = ready_device_ptr
        self._consumed_device_ptr = consumed_device_ptr

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        """Wait for an offloaded result or delegate to ``forward_impl``."""
        if self._is_cpu_offloaded:
            torch.ops.vllm.ple_offload_host_copy(
                hidden_states,
                self._gpu_output_buffer,
                self._cpu_output_buffer,
                self._ready_device_ptr,
                self._consumed_device_ptr,
            )
            return self._gpu_output_buffer[: input_ids.shape[0]]
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(
        self,
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        """Compatibility hook; the main-stream host copy releases the buffer."""
