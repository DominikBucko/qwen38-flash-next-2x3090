# SPDX-License-Identifier: Apache-2.0
"""Optional refusal-direction removal ("abliteration") applied at runtime.

``QWEN38_ABLITERATION`` names a JSON file with a ``refusal_direction`` list, or a safetensors file with a
``refusal_direction`` tensor, of length ``hidden_size``. A bare name such as ``orcarouter`` resolves to
``/opt/qwen38/abliteration/<name>.json``.
With it set, every residual-stream write loses its component along the unit direction r:

    x <- x - r (r . x)

applied to the attention and linear-attention block outputs, the MoE block output (routed plus shared experts),
the PLE value projection and the text token embeddings. For one direction this equals the weight edit
``W' = W - r (r^T W)`` on ``self_attn.o_proj``, ``linear_attn.out_proj``, every expert's and the shared expert's
``down_proj``, ``ple.value_proj`` and the rows of ``embed_tokens`` (Arditi et al., 2024): the routed and shared
expert outputs are summed with scalar weights before the residual write, so projecting their sum is the same as
editing each down projection. The INT4 experts and every other weight stay untouched. Unset, nothing is loaded and
no work is added.
"""

import json
import math
import os

import torch

ENV = "QWEN38_ABLITERATION"
BUILTIN_DIR = "/opt/qwen38/abliteration"


def direction_path() -> str | None:
    value = os.environ.get(ENV, "").strip()
    if value in ("", "0"):
        return None
    if os.sep not in value and not value.endswith((".json", ".safetensors")):
        value = os.path.join(BUILTIN_DIR, f"{value}.json")
    return value


def load_direction(hidden_size: int) -> torch.Tensor | None:
    """The unit refusal direction as a float32 CPU tensor, or None when the switch is off."""
    path = direction_path()
    if path is None:
        return None
    if path.endswith(".json"):
        with open(path) as f:
            r = torch.tensor(json.load(f)["refusal_direction"], dtype=torch.float32)
    else:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as f:
            r = f.get_tensor("refusal_direction")
    r = r.to(torch.float32).flatten()
    if r.numel() != hidden_size:
        raise ValueError(f"{ENV}: {path} holds {r.numel()} values; the model's hidden size is {hidden_size}")
    norm = float(r.norm())
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"{ENV}: {path} holds a zero or non-finite direction")
    return r / norm


def register(module: torch.nn.Module, hidden_size: int, dtype: torch.dtype) -> None:
    """Attach the direction as a non-persistent buffer ``_abliteration_r`` (None when the switch is off).

    Call inside the model constructor so the buffer lands on the model's device before CUDA-graph capture. The
    CPU-only PLE worker process never runs these forwards and gets no buffer.
    """
    from vllm.model_executor.layers.ple_offload_layer import is_offload_process

    r = None if is_offload_process() else load_direction(hidden_size)
    if r is None:
        module._abliteration_r = None
        return
    buf = torch.empty(hidden_size, dtype=dtype)
    buf.copy_(r)
    module.register_buffer("_abliteration_r", buf, persistent=False)


def project_(x: torch.Tensor, r: torch.Tensor | None) -> torch.Tensor:
    """In place: remove the component along r from x's last dimension. No-op when r is None."""
    if r is None:
        return x
    width = x.shape[-1]
    if x.is_contiguous():
        flat = x.view(-1, width)
        flat.addr_(flat @ r, r, alpha=-1.0)  # rank-1 update, no full-size temporary
    else:
        x.sub_(torch.outer(x.reshape(-1, width) @ r, r).view(x.shape))
    return x
