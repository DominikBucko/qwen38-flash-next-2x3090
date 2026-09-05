---
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
  - rtx-3090
  - dual-gpu
  - cpu-offload
  - local-llm
  - w4a16
---

# Qwen3.8-Flash-Next on dual RTX 3090: W4A16 + FP8 PLE + MTP3

Run **Qwen 3.8 Flash Next locally on two RTX 3090 24 GB GPUs and 128 GB RAM**.
Start with the **[GitHub quickstart and pinned runtime](https://github.com/DominikBucko/qwen38-flash-next-2x3090#run-it)**;
the weights require its custom vLLM overlay.

[Hardware requirements and experimental 4090 guidance](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/hardware.md)
· [Performance tuning and public benchmark client](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/performance.md)
· [Share a hardware result](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=hardware-report.yml)

This hybrid serves one native 262,144-token context across both cards, with
NVMe-backed swap for loading headroom. It keeps Intel's AutoRound target
tensors exactly as published, replaces only the 102.4 GB BF16 n-gram/PLE table
with RadixArk's FP8 table, and adds a compact INT4 group-32 MTP draft under
`runtime/mtp-int4-g32`.

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

The original measurements used runtime release
[`v0.1.0`](https://github.com/DominikBucko/qwen38-flash-next-2x3090/releases/tag/v0.1.0).
The [current setup and measurement guide](https://github.com/DominikBucko/qwen38-flash-next-2x3090)
adds reproducible probes and P2P/allocator diagnostics while retaining the
checkpoint tensor revision `ef554143369a706525336f6b42a09094835dc077`.

Configure at least 32 GiB of fast NVMe swap before loading the checkpoint;
48–64 GiB is safer. The released hot cache is intentionally close to the 24 GiB
VRAM limit. If the first prompt raises a CUDA OOM, lower
`VLLM_WNA16_STATIC_HOT_CACHE_SIZE` from 88 to 86, then 84. Each removed slot
saves roughly 116 MiB per GPU, with a decode-speed tradeoff. The
[memory guide](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/memory.md)
documents host OOMs, KV-cache tuning, prefill transients, and two-client
capacity.

## Measured performance

On 2× RTX 3090 with 128 GB of system memory:

### September 5 verified candidate

- 258,048 input + 4,096 output, three measured `repo-chat` runs with no
  explicit warmup: **75.636 API-observed output tok/s** by reciprocal mean TPOT
  (74.031–76.707), with TTFT from 211.059 to 215.128 seconds;
- 128 input + 4,096 output, one warmup and three measured `repo-chat` runs:
  **77.2845 API-observed output tok/s** (74.746–79.707).

This native candidate used an 84-expert hot cache, CUDA P2P in both directions,
custom all-reduce enabled, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. It ran the pinned vendor
vLLM plus the public overlay in a clean native environment with existing
dependencies, not a fresh Docker build. The released Docker defaults remain an
88-expert cache, custom all-reduce disabled, and expandable segments enabled.
All 27 model weight files matched the published SHA-256 manifest for canonical
tensor revision `ef554143369a706525336f6b42a09094835dc077`.

Four recoverable allocator warnings appeared during the first long prefill;
all three measured streams completed with exact usage counts. Generated-token
counts include reasoning and control tokens, and the forced 4,096-token capture
can end during reasoning. These probes measure serving performance, not answer
quality. See the [benchmark bundle](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-05/README.md),
[machine-readable summary](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-05/summary.json),
and [long-context chart](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/images/long-context-decode.svg).

### Historical release measurements

- 262,016-token prompt: 1,275.6 prompt token/s;
- 128-output boundary probe after that prompt: 54.5 token/s;
- warmed 128-input/4,096-output greedy probes: 127.1–134.0 output token/s;
- MTP acceptance on the warm probes: 86.3–90.8%.

The figures above are historical single-request measurements. The 128-output boundary
probe is too short to characterize sustained long-context generation. The
[public benchmark protocol](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/performance.md)
uses 258,048 input + 4,096 output for that question and keeps new workload
results separate. Agent quality evidence is single-run and provisional;
private benchmark fixtures and traces are not included.

## Limitations and license

- Pinned CUDA runtime validated on SM86/RTX 3090; other architectures are unvalidated.
- Dual RTX 4090 is not yet validated; no 4090 throughput claim is made.
- Optimized for one full-context request rather than high concurrency.
- PLE table is host-resident and should remain out of swap.
- MTP is speculative: target verification preserves target token decisions,
  while the draft affects acceptance and speed.
- Review the Qwen Community License included in this repository and all upstream
  model cards before redistribution or commercial use.
