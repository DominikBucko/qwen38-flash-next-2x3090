---
library_name: transformers
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model:
  - Qwen/Qwen3.8-Flash-Next
  - Intel/Qwen3.8-Flash-Next-W4A16-AutoRound
  - RadixArk/Qwen3.8-Flash-Next-NVFP4
pipeline_tag: text-generation
tags:
  - qwen3.8
  - autoround
  - int4
  - fp8
  - vllm
  - speculative-decoding
  - 256k-context
---

# Qwen3.8-Flash-Next Intel W4A16 + FP8 PLE + MTP3

This hybrid serves the native 262,144-token context on two 24 GiB Ampere GPUs
with 128 GiB of host RAM. It keeps Intel's AutoRound target tensors exactly as
published, replaces only the 102.4 GB BF16 n-gram/PLE table with RadixArk's FP8
table, and adds a compact INT4 group-32 MTP draft under `runtime/mtp-int4-g32`.

No target tensor was requantized or repacked during assembly.

## Composition

| Component | Format | Pinned source |
|---|---|---|
| Target routed experts and eligible linear weights | AutoRound W4A16, INT4 symmetric group-128 | `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@861536dda5bcb208376fc4cd879b2bf76bece9fe` |
| Sensitive target layers | BF16, unchanged | Intel checkpoint above |
| 51.2B-parameter n-gram/PLE table | FP8 E4M3FN plus published scale | `RadixArk/Qwen3.8-Flash-Next-NVFP4@7b719225242aacd3dbd3f9407468c2ee9a9d2594` |
| Optional MTP draft | Routed experts INT4 symmetric group-32; other tensors unchanged | `runtime/mtp-int4-g32` |

The target contains 222,716 indexed tensors in 25 safetensors files with
124,750,778,874 bytes (116.183 GiB) of tensor payload. The compact MTP draft
contains 4,639 tensors in two files with 4,139,535,872 bytes (3.855 GiB) of
payload. `hybrid_sources.json`, `runtime/mtp-int4-g32/compact_sources.json`, and
`runtime/repro.lock.json` are machine-readable provenance records.

## Runtime

This is not a stock Transformers checkpoint. Use the matching GitHub runtime
release and the digest-pinned vLLM image recorded in `runtime/repro.lock.json`.
The default profile uses BF16 KV, TP2+EP2, UVA expert offload, an 88-expert GPU
hot cache, prefix caching, and MTP3. See `runtime/README.md`.

GitHub runtime: https://github.com/DominikBucko/qwen38-flash-next-2x3090 at
release `v0.1.0`.

## Measured performance

On 2× RTX 3090 with 128 GB of system memory:

- 262,016-token prompt: 1,275.6 prompt token/s;
- 128-token decode after that prompt: 54.5 token/s;
- warmed 128-input/4,096-output greedy probes: 127.1–134.0 output token/s;
- MTP acceptance on the warm probes: 86.3–90.8%.

These are single-request measurements. Agent quality evidence is single-run and
provisional; private benchmark fixtures and traces are not included.

## Limitations and license

- CUDA/SM86-specific patched vLLM runtime.
- Optimized for one full-context request rather than high concurrency.
- PLE table is host-resident and should remain out of swap.
- MTP is speculative: target verification preserves target token decisions,
  while the draft affects acceptance and speed.
- Review the Qwen Community License included in this repository and all upstream
  model cards before redistribution or commercial use.
