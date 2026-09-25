# Fast 256K runtime — September 25

**2,752 tok/s prefill · 104.5 tok/s decode** — one request with 131,072 input
tokens and 2,048 output tokens, on the release image built from this repository.
It reached the first token in 47.6 seconds.

**Full 256K window:** 260,096 input + 2,048 output tokens reach the first token
in 98.0 seconds (**2,654 input tok/s**) and then decode at up to **103.1 tok/s**,
on the same release image.

Hardware: two RTX 3090 24 GB cards and 128 GB system memory. This is a runtime
release: use `configs/fast-256k.env` (see [Run it](../../README.md#run-it)).
The checkpoint tensors are unchanged.

## Release image, 131,072 + 2,048

| Run | First token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|
| 1 | 48.7 s | 2,693 | 102.0 |
| 2 | **47.6 s** | **2,752** | **104.5** |
| 3 | 47.6 s | 2,757 | 94.2 |

Reciprocal-mean decode over the three runs: 100.0 tok/s. The raw report,
[fast-131k.json.gz](fast-131k.json.gz), has exact token counts, prompt and
output hashes and per-run timings.

## Release image, 260,096 + 2,048

| Run | First token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|
| 1 | 98.0 s | 2,653 | 92.6 |
| 2 | 98.0 s | 2,654 | 102.5 |
| 3 | 98.0 s | 2,654 | **103.1** |

Reciprocal-mean decode: 99.2 tok/s. Raw report: [fast-260k.json.gz](fast-260k.json.gz).

Earlier runs on the development image with the same runtime files:

| Host condition | First token | Input tok/s | Decode tok/s | MTP acceptance |
|---|---:|---:|---:|---:|
| Quiet host | 98.0–98.1 s | 2,651–2,654 | 100.6–102.8 | 56–64% |
| Another job filling RAM | 98.0–98.7 s | 2,636–2,654 | 91.1–95.5 | 51–54% |

Same-night baseline, the September 18 candidate: 147.0 s to first token
(1,770 input tok/s) and 82.6 tok/s decode on the same prompt.

Decode depends on host memory. The runtime keeps about 60 GB of expert weights
and the 51.2 GB PLE table in RAM. When another job filled the page cache, the
serving processes were pushed into swap and decode fell to 91–95 tok/s.

## What changed

- **Streamed prefill staging.** Large prefill chunks copy each layer's cold
  experts to the GPU by DMA one or two layers ahead, then run the original
  expert GEMM from VRAM. The copy buffers reuse hot-cache pages, so no expert
  cache slots are lost.
- **No host stalls between steps.** Two synchronizing copies in input
  preparation were removed, and async scheduling is now enabled. Host time per
  decode step fell from 23.7 to 2.6 ms.
- **P2P all-reduce** replaces NCCL for decode collectives.
- **Skinny GEMM kernel** for decode-sized dense projections and the LM head.
- **Smaller draft vocabulary** for MTP proposals. The target still verifies
  every token, so output is unchanged.
- **PLE table kept in RAM** after load, and **persistent kernel caches**.

Target INT4 weights, BF16 KV, FP8 PLE, ten-expert routing, the INT4 MTP draft
and the approximate QSA budget are unchanged. No expert pruning.

## Agent check

Six fresh trajectories of three AgentBench v0.2 tasks (DBG-06, LLM-01, OPS-02),
xhigh reasoning, 16,384 tokens per response: all six scored 100 and ended with
a normal stop. Four were strict passes; the other two missed only the pre-edit
discovery gate. This is a selected-task check, not the 15-task suite.

## Limits

- Single requests; no confidence interval. Decode varies with MTP acceptance
  (51–64% observed on this workload).
- Vision and two-client profiles were not re-validated on this runtime before
  publication.
- Input tok/s is input tokens divided by time to first token, including
  scheduling, not kernel-only prefill.
