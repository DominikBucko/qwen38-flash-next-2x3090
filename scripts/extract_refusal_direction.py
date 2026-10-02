#!/usr/bin/env python3
"""Recover the refusal direction of an abliterated Qwen3.8-Flash-Next release for QWEN38_ABLITERATION.

An abliterated release edits every residual-writing matrix as W' = W - r (r^T W) for one unit direction r. This
script reads four edited matrices and one untouched control tensor from the release (HTTP range reads, ~83 MB),
compares them with the same tensors of the local base checkpoint, recovers r as the top left singular vector of
W - W', checks that the edit is exactly that one projection, and writes the JSON that
``models/qwen3_8_flash_next/nvidia/abliteration.py`` loads.

Needs torch, safetensors and huggingface_hub, and a Hugging Face token whose account has accepted the release's
access terms if it is gated. The shipped ``configs/abliteration/orcarouter.json`` was produced with:

    python3 scripts/extract_refusal_direction.py --model-dir /models/qwen38-flash-next \\
        --repo orcarouter/Qwen3.8-Flash-Next-Uncensored --out configs/abliteration/orcarouter.json
"""

import argparse
import json
import struct
import sys

import torch
from huggingface_hub import HfApi, HfFileSystem, hf_hub_download
from safetensors import safe_open

P = "model.language_model."
EDITED = [P + "layers.3.self_attn.o_proj.weight", P + "layers.0.linear_attn.out_proj.weight",
          P + "layers.1.ple.value_proj.weight", P + "layers.3.mlp.shared_expert.down_proj.weight"]
CONTROL = P + "layers.3.self_attn.k_proj.weight"
EMBED, EMBED_ROWS = P + "embed_tokens.weight", 256


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", required=True, help="local base checkpoint (albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE)")
    parser.add_argument("--repo", default="orcarouter/Qwen3.8-Flash-Next-Uncensored")
    parser.add_argument("--revision", default=None, help="release commit (default: current main)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    revision = args.revision or HfApi().model_info(args.repo).sha
    fs = HfFileSystem()
    their_index = json.load(open(hf_hub_download(args.repo, "model.safetensors.index.json", revision=revision)))["weight_map"]
    our_index = json.load(open(f"{args.model_dir}/model.safetensors.index.json"))["weight_map"]

    def theirs(name: str, rows: int | None = None) -> torch.Tensor:
        with fs.open(f"{args.repo}@{revision}/{their_index[name]}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            meta = json.loads(f.read(n))[name]
            if meta["dtype"] != "BF16":
                raise SystemExit(f"{name}: expected BF16, release has {meta['dtype']}")
            start, end = meta["data_offsets"]
            shape = list(meta["shape"])
            if rows is not None:
                end, shape[0] = start + rows * shape[1] * 2, rows
            f.seek(8 + n + start)
            raw = f.read(end - start)
        return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).reshape(shape).clone()

    def ours(name: str, rows: int | None = None) -> torch.Tensor:
        with safe_open(f"{args.model_dir}/{our_index[name]}", framework="pt") as f:
            t = f.get_slice(name)
            return (t[:rows] if rows is not None else t[:]).clone()

    # The release must start from the same base weights: an untouched tensor is byte-identical.
    control_identical = bool(torch.equal(ours(CONTROL), theirs(CONTROL)))
    pairs = {name: (ours(name), theirs(name)) for name in EDITED}
    deltas = [w.float() - w2.float() for w, w2 in pairs.values()]
    r = torch.linalg.svd(torch.cat(deltas, dim=1), full_matrices=False)[0][:, 0]
    r = r / r.norm()

    rank1, abs_cos, exact = {}, {}, {}
    for (name, (w, w2)), d in zip(pairs.items(), deltas):
        s = torch.linalg.svdvals(d)
        rank1[name] = float(s[1] / s[0])
        abs_cos[name] = float(abs(torch.linalg.svd(d, full_matrices=False)[0][:, 0] @ r))
        wf = w.float()
        exact[name] = float(((wf - torch.outer(r, r @ wf)).to(torch.bfloat16) == w2).float().mean())
    e, e2 = ours(EMBED, EMBED_ROWS), theirs(EMBED, EMBED_ROWS)
    ef = e.float()
    exact[EMBED] = float(((ef - torch.outer(ef @ r, r)).to(torch.bfloat16) == e2).float().mean())

    short = lambda d: {k.removeprefix(P): round(v, 6) for k, v in d.items()}  # noqa: E731
    print(f"control identical: {control_identical}")
    print(f"sigma2/sigma1: {short(rank1)}")
    print(f"abs cos with the joint direction: {short(abs_cos)}")
    print(f"bf16-exact reproduction: {short(exact)}")
    ok = (control_identical and max(rank1.values()) < 0.05 and min(abs_cos.values()) > 0.9999
          and min(exact.values()) > 0.8)
    if not ok:
        print("the release is not a single-direction edit of this base checkpoint; nothing written", file=sys.stderr)
        return 1

    doc = {
        "source": args.repo,
        "revision": revision,
        "edit": "W' = W - r (r^T W) on every residual-writing matrix (Arditi et al., 2024), one direction",
        "recovered_from": "top left singular vector of W - W' over " + ", ".join(n.removeprefix(P) for n in EDITED),
        "verification": {
            "untouched_control_identical": control_identical,
            "per_tensor_abs_cos": {k.removeprefix(P): v for k, v in abs_cos.items()},
            "reproduction_bf16_exact_fraction": {k.removeprefix(P): round(v, 4) for k, v in exact.items()},
        },
        "hidden_size": r.numel(),
        "refusal_direction": [float(f"{x:.9g}") for x in r.tolist()],
    }
    with open(args.out, "w") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
