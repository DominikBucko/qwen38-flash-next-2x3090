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
  - 128k-context
  - rtx-3090
  - single-gpu
  - dual-gpu
  - 24gb-vram
  - 64gb-ram
  - consumer-gpu
  - cpu-offload
  - moe
  - int8-kv-cache
  - local-llm
  - w4a16
---

# Qwen3.8-Flash-Next on one or two RTX 3090s with 64 GB or 128 GB RAM: W4A16 + FP8 PLE + MTP3

Run **Qwen 3.8 Flash Next locally** on one or two 24 GB GPUs. The weights on this page need one of two custom
vLLM runtimes; pick the row for your hardware:

| Hardware | Runtime (GitHub) | Prefill, 131K prompt | Decode | Context |
|---|---|---:|---:|---:|
| **1× RTX 3090 24 GB + 64 GB RAM** | **[qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090)** | **up to 2,106 tok/s** | **43–51 tok/s** | 135,168 |
| **2× RTX 3090 24 GB + 64 GB RAM** | **[qwen38-flash-next-2x3090, 64 GB profile](https://github.com/DominikBucko/qwen38-flash-next-2x3090#new-64-gb-ram-profile)** | **3,401–3,413 tok/s** | **84–89 tok/s** | 135,168 |
| 2× RTX 3090 24 GB + 128 GB RAM | [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090#run-it) | 3,029 tok/s | 109.8 tok/s | up to 262,144 |

## New: two RTX 3090s with 64 GB of RAM

The [64 GB profile](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/configs/2x3090-64gb.env)
of the two-GPU runtime (v0.5.0) runs **both cards with 64 GB of system RAM**. The 128 GB profiles keep a pinned
copy of every expert and the PLE table in RAM. Here each GPU owns its 88 most-used experts per layer outright,
and the other experts of its half live once in RAM (38 GiB for both GPUs). During decode a CPU thread pool per
GPU computes them while the GPU takes a share over PCIe. Prefill streams them to the GPUs in 8,192-token chunks.
The larger chunks, not the RAM layout, are why it prefills faster than the 128 GB profile, which uses 4,096
([details](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-30/README.md#why-prefill-is-faster-than-with-the-128-gb-agent-profile)).
The FP8 PLE table is read in place from these files on NVMe.

| Two RTX 3090s, machine limited to 64 GB RAM, 3 runs | First token | Prefill | Decode |
|---|---:|---:|---:|
| 131,099 + 512 tokens | 38.4–38.5 s | **3,401–3,413 tok/s** | **84.2–88.8 tok/s** |
| 32,799 + 512 | 10.2–10.3 s | 3,200–3,211 tok/s | 77.0–83.9 tok/s |
| 8,218 + 1,024 | 2.6 s | 3,114–3,146 tok/s | 78.2–79.7 tok/s |

With the server restricted to 12 CPU cores on 2 CCDs, decode ran at 71–81 tok/s. Setup: clone the GitHub
repository, then `cat configs/2x3090-64gb.env >> .env && make serve`. No CUDA P2P and no swap are needed. See
the [64 GB benchmark report](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-30/README.md).

## New: a single RTX 3090 with 64 GB of RAM

**[github.com/DominikBucko/qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090)**
serves this checkpoint on **one RTX 3090 (24 GB) and 64 GB of system RAM**, with a 128K context. The GPU keeps
the attention and dense weights, the 32 most-used experts of every layer and an INT8 KV cache (141,504 tokens in
3.05 GB); **the CPU computes the other experts straight from RAM** during decode, while prefill streams them
through the GPU. The FP8 PLE table is read in place from these files on NVMe. The weights are unchanged.

| One RTX 3090, machine limited to 64 GB RAM (2 runs) | First token | Prefill | Decode |
|---|---:|---:|---:|
| 131,099 + 512 tokens | 62.2 / 90.3 s | **2,106** / 1,451 tok/s | **47.4** / 43.3 tok/s |
| 32,799 + 512 | 15.3 / 14.8 s | 2,141 / 2,219 tok/s | 50.3 / 49.5 tok/s |
| 8,218 + 1,024 | 5.2 / 5.7 s | 1,570 / 1,452 tok/s | 49.1 / 48.5 tok/s |

Some prefill requests take a slower path (~1,450 instead of ~2,100 tok/s); see the known issue in the
benchmark report. Decode depends mainly on RAM bandwidth: with the server restricted to 16 cores of the benchmark CPU it ran at
39–47 tok/s, with 8 cores at 34–36 tok/s. Quick start:

```bash
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 --local-dir /models/qwen38-flash-next
git clone https://github.com/DominikBucko/qwen38-flash-next-3090.git && cd qwen38-flash-next-3090
cp .env.example .env    # set MODEL_DIR
make build-image && make serve
```

Or skip the build with the published image:
`IMAGE=ghcr.io/dominikbucko/qwen38-flash-next-3090@sha256:7f176605b59c462af21b1b62fd854f9f3cca96e3d11a295581297483d45490a3 make serve`.

See the [benchmark report](https://github.com/DominikBucko/qwen38-flash-next-3090/blob/main/benchmarks/2026-09-30/README.md),
[how it works](https://github.com/DominikBucko/qwen38-flash-next-3090/blob/main/docs/how-it-works.md) and the
[FAQ](https://github.com/DominikBucko/qwen38-flash-next-3090/blob/main/docs/faq.md).

## Two RTX 3090s: 3,029 tok/s prefill · 109.8 tok/s decode

**2× RTX 3090 + 128 GB RAM · v0.4.0 runtime (September 29)**

With the new **agent 128K profile**, one request with 131,072 input tokens
reaches the first token in **43.3 seconds** (**3,029 input tok/s**), then
generates 2,048 tokens at **109.8 tok/s**. The **fast 256K profile** now reads
the full 256K window (260,096 input tokens) in **90.8 seconds** (**2,865 input
tok/s**), 8% faster than the September 25 release.

| Profile | Input + output tokens | First token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| Agent 128K, best of 2 | 131,072 + 2,048 | **43.3 s** | **3,029** | **109.8** |
| Fast 256K, 3 runs | 131,072 + 2,048 | 45.0–46.0 s | 2,852–2,916 | 98.9–101.9 |
| Fast 256K, 3 runs | 260,096 + 2,048 | **90.8 s** | **2,864–2,865** | 94.0–98.3 |

This is a **runtime release**. Clone the GitHub repository and use
`configs/fast-256k.env` or `configs/agent-128k.env`, or pull the prebuilt
`ghcr.io/dominikbucko/qwen38-flash-next-2x3090:v0.4.0` image (pin the digest
from the release notes). The model weights on this page are unchanged.

v0.4.0 fixes a KV-cache leak that kept one recurrent-state block per prefill
chunk alive in each Mamba layer group. Long requests used to fill the cache and
restart several times, and now they run without preemptions. The agent profile
limits requests to 135,168 tokens and uses the freed cache memory to keep 16 more
experts per GPU on the cards (hot100). Decode then fetches fewer experts from
system memory. Target weights, BF16 KV, FP8 PLE, ten-expert routing and the
approximate QSA budget are unchanged.

Decode depends on free RAM: the runtime keeps about 60 GB of expert weights and
the 51.2 GB PLE table in system memory. Input tok/s means input tokens / time to
first token. See the
[fix report](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-28/README.md),
the [agent profile report](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-29/README.md)
and the [September 25 results](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/benchmarks/2026-09-25/README.md).

## Setup on two GPUs

[Hardware requirements, 4090 guidance and community 5090 report](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/hardware.md)
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
For one GPU, use [qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090)
(INT8 KV cache, CPU cold experts, 135,168-token context); the rest of this section
describes the two-GPU runtime.
The current GitHub default uses BF16 KV, TP2+EP2, UVA expert offload, an
84-expert GPU hot cache, prefix caching, and MTP3. The original bundled
`runtime/README.md` describes the older hot88 release; use the current GitHub
quickstart for the prefill-memory fixes and hot84 default.

The original measurements used runtime release
[`v0.1.0`](https://github.com/DominikBucko/qwen38-flash-next-2x3090/releases/tag/v0.1.0).
The [current setup and measurement guide](https://github.com/DominikBucko/qwen38-flash-next-2x3090)
adds reproducible probes and P2P/allocator diagnostics while retaining the
checkpoint tensor revision `ef554143369a706525336f6b42a09094835dc077`.

Configure at least 32 GiB of fast NVMe swap before loading the checkpoint;
48–64 GiB is safer. If the first prompt raises a CUDA OOM, first check that an
old `.env` is not still selecting hot88. Start with
`VLLM_WNA16_STATIC_HOT_CACHE_SIZE=84`, then try 80 if needed. Each removed slot
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
dependencies, not a fresh Docker build. Current GitHub defaults use an
84-expert cache, custom all-reduce disabled, and expandable segments enabled.
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

- Maintainer validation uses SM86/RTX 3090. The hardware guide separately records
  a community dual-5090 report; it is not a maintainer benchmark.
- Dual RTX 4090 is not yet validated; no 4090 throughput claim is made.
- Optimized for one full-context request rather than high concurrency.
- PLE lives in host memory but can be paged to swap. Sustained paging can hurt
  performance; check residency and swap activity during serving.
- MTP is speculative: target verification preserves target token decisions,
  while the draft affects acceptance and speed.
- Review the Qwen Community License included in this repository and all upstream
  model cards before redistribution or commercial use.
