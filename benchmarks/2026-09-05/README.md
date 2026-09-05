# Public serving probes: 2026-09-05

These measurements exercise one Qwen3.8-Flash-Next request on two RTX 3090
24 GB GPUs with 128 GB of eight-channel DDR4-3200. They add repeatable,
synthetic 4,096-token generation probes to the shorter historical results in
[`../serving-summary.json`](../serving-summary.json).

Every case completed three measured requests with exact input, completion, and
total-token usage. The short cases also had one warmup request. A unique
`cache_salt` prevented prefix-block reuse between requests; model weights, CUDA
graphs, expert caches, allocator state, and host pages remained warm.

## Results

The rate is the reciprocal of mean API-observed time per output token after the
first streamed chunk. It includes SSE and network effects, and streamed chunks
may contain multiple tokens because MTP is enabled. It is not kernel-level
TPOT. TTFT includes scheduling, prefill, and first-token work, so input tokens
divided by TTFT is not isolated prefill throughput.

| Suite and case | Prompt | Input + output | Warmup / measured | Decode estimate | Per-run range | TTFT range |
|---|---|---:|---:|---:|---:|---:|
| `baseline-hot86/short-code` | Raw public code prefix | 128 + 4,096 | 1 / 3 | 102.561 tok/s | 83.205–130.585 tok/s | 0.773–0.779 s |
| `baseline-hot86/long-code` | Raw public code corpus | 258,048 + 4,096 | 0 / 3 | 78.824 tok/s | 78.793–78.856 tok/s | 215.504–221.446 s |
| `candidate-hot84-p2p/short-chat` | Server-templated tutorial request | 128 + 4,096 | 1 / 3 | 77.285 tok/s | 74.746–79.707 tok/s | 0.897–0.903 s |
| `candidate-hot84-p2p/long-chat` | Server-templated tutorial plus public code | 258,048 + 4,096 | 0 / 3 | 75.636 tok/s | 74.031–76.707 tok/s | 211.059–215.128 s |

The long request reaches the 262,144-token limit exactly. Its 4,096-token
generation window is a better sustained-output probe than the historical
262,016-input/128-output capacity check. It still represents one synthetic
request, not concurrent serving or a production workload.

The raw `repo-code` prompt concatenates 27 path-sorted public overlay files.
The short form contains only the first 128 source tokens; the long form contains
one 243,328-token corpus plus a 14,720-token repeated prefix. It has no task
instruction. Short-code output included severe repetition in one run and a long
whitespace run in another, so the high and variable short-code rates are
diagnostic only.

The `repo-chat` prompt asks for a detailed code tutorial and gets its template
and generation wrappers from the server tokenizer. Its fixed wrappers consume
117 tokens. The short form therefore contains only 11 source tokens. The long
form contains one complete 243,329-token source interior plus a 14,602-token
repeated prefix; wrapper plus source accounting reconciles to 258,048 tokens.

## What the generated text establishes

All twelve measured requests produced exactly 4,096 completion tokens. These
counts include every generated token, including reasoning and control tokens.
`ignore_eos=true` forced the requested length and can expose repetition after a
natural stopping point.

The three long raw-code captures had no large repeated region or encoding
corruption. Two eventually formed locally coherent technical explanations and
one remained a plausible code continuation, but none was a response to a task.

All three short-chat and all three long-chat captures were coherent planning or
reasoning text with no gross repetition or encoding corruption. They remained
in reasoning for the full 4,096 tokens and ended mid-thought rather than
delivering a completed tutorial. The captures are useful evidence that the
measured tokens were not a single degenerate loop. They do not establish answer
quality, factual correctness, or task completion.

## Profiles are not a controlled all-reduce A/B

Both suites used TP2, EP2, BF16 KV, MTP depth 3, approximate QSA, a 262,144-token
limit, one sequence, a 4,096-token scheduler budget, and the same model and
public overlay. CUDA peer access was available in both directions. The GPUs had
a PHB PCIe 4.0 x16 topology and no NVLink.

The profiles differ in several material ways:

| Setting | Baseline | Candidate |
|---|---:|---:|
| Hot experts per layer | 86 | 84 |
| vLLM custom all-reduce | Disabled | Enabled |
| PyTorch expandable segments | Enabled/default experiment setting | `False` |
| Prompt recipe | Raw `repo-code` | Server-templated `repo-chat` |

The candidate is evidence that the full model and near-full-context request can
run with peer access and custom all-reduce under that combined profile. The
numbers do not isolate a custom-all-reduce speedup, and the short and long rates
must not be presented as a controlled performance comparison with the baseline.
Different prompts also change expert routing and MTP acceptance.

## MTP counters

Counter snapshots cover every request between the before and after snapshots.
That means the short-case totals include one warmup plus three measured
requests; the long-case totals cover the three measured requests.

| Case | Draft steps | Draft tokens | Accepted draft tokens | Acceptance | Accepted per draft step |
|---|---:|---:|---:|---:|---:|
| `short-code` | 4,902 | 14,706 | 11,485 | 78.097% | 2.343 |
| `long-code` | 3,576 | 10,724 | 8,709 | 81.210% | 2.435 |
| `short-chat` | 6,306 | 18,918 | 10,081 | 53.288% | 1.599 |
| `long-chat` | 4,546 | 13,633 | 7,739 | 56.767% | 1.702 |

The acceptance differences reinforce why rates from different prompt recipes
cannot isolate a runtime mechanism.

## Thermals, power, swap, and errors

These are maxima among telemetry samples timestamped inside measured runs.
Swap-in is the sum of each run's bracketing `/proc/vmstat` delta in pages; a
bracket can include up to one sampling interval outside the request. No measured
request increased `pswpout`.

| Case | Peak CPU | Peak GPU °C, 0 / 1 | Peak GPU W, 0 / 1 | `pswpin` pages | `pswpout` pages | Run errors |
|---|---:|---:|---:|---:|---:|---:|
| `short-code` | 76.25 °C | 72 / 78 | 299.06 / 299.60 | 57,518 | 0 | 0 |
| `long-code` | 79.25 °C | 73 / 81 | 295.68 / 297.62 | 1,184,968 | 0 | 0 |
| `short-chat` | 75.50 °C | 72 / 79 | 298.52 / 299.37 | 128,391 | 0 | 0 |
| `long-chat` | 77.00 °C | 73 / 81 | 298.59 / 299.08 | 1,234,869 | 0 | 0 |

The GPUs were capped at 300 W each. Across each complete suite, the sampled
peaks were 79.25 °C CPU, 73/81 °C GPUs, and 299.14/299.60 W for the baseline;
the candidate reached 77.00 °C, 73/81 °C, and 298.59/299.37 W.

All 12 primary measured requests completed with exact counts and zero recorded
client errors. Allocator pressure was still visible: the baseline server log
contains 2,450 `memory mapping failed with OOM` or `memory allocation failed
with OOM` warning lines, while the candidate log contains 4. These warning-line
counts are not failed requests. Lowering the hot cache from 86 to 84 and
disabling expandable segments reduced retries, but this run does not separate
their effects.

Three earlier full-server startup attempts failed before the published suites:

1. Hot cache 88, custom all-reduce enabled, expandable segments enabled: KV
   allocation OOM on GPU 0.
2. Hot cache 86 with the same collective/allocator settings: the first native
   graph compile could not locate NVRTC builtins; the experiment-local library
   path was corrected.
3. The corrected hot-86 launch: CUDA graph IPC failed in
   `custom_all_reduce.cuh:164`.

A separate two-rank collective smoke test failed with expandable segments
enabled and passed with `expandable_segments:False`. That smoke test establishes
the allocator requirement for this host; it does not measure model speed.
The exact-sum eager and CUDA-graph checks are recorded in
[`collective-smoke.json`](collective-smoke.json).

## Runtime and checkpoint provenance

The probes used the clean vendor vLLM `0.1.dev20073+g8e685d198` plus the public
27-file overlay and the dependency versions recorded in each suite's
`environment.json`. They ran natively through an isolated `PYTHONPATH`, with a
scoped NVRTC library path, unlimited memlock for the launcher, localhost port
18088, and temporary 64 GiB swap in addition to the host's existing 8 GiB. They
did not run from a fresh build of the released Docker image. Treat this as a
documented runtime deviation when comparing with container runs.

The checkpoint was verified after the benchmark against the SHA-256 entries in
the public manifest at canonical tensor revision
`ef554143369a706525336f6b42a09094835dc077`. All 27 weight files matched: 25
target files and two compact MTP draft files, totaling 128,921,364,661 file
bytes. The matching config, indexes, tokenizer files, and source records identify
the Intel AutoRound target revision and RadixArk FP8 PLE source. See
[`model-integrity.json`](model-integrity.json) for every expected and observed
hash.

The local packaged `runtime/repro.lock.json` fills in the published tensor
revision that was null in the pre-upload copy; its manifest line consequently
differs. Those two metadata files do not change any weight bytes. The other
23 audited configuration, tokenizer, index, and source-metadata files matched
the canonical revision exactly.

The post-run restoration proof records that experiment servers were stopped,
the temporary 64 GiB swap was removed, and only the original 8 GiB swap file
remained. GPU memory returned to 277/15 MiB at 45/43 °C. The original 300 W GPU
caps, 280 W CPU cap, `performance` CPU profile, fan step 3, and monitored
services remained in place. See [`restored-state.json`](restored-state.json).

## Receipts

[`summary.json`](summary.json) links the compressed raw client reports, run-level
timing, MTP deltas, telemetry summaries, environment records, model-integrity
proof, collective smoke result, and restored-host-state record checked into this
directory.

The complete public receipts, including raw reports, SSE chunk timing, captured
synthetic outputs, metrics snapshots, telemetry, benchmark client, analyzer,
collective smoke script, and server logs with account-specific paths removed,
are attached to the
[`v0.2.0` release](https://github.com/DominikBucko/qwen38-flash-next-2x3090/releases/tag/v0.2.0)
as
[`qwen38-public-benchmarks-2026-09-05.tar.gz`](https://github.com/DominikBucko/qwen38-flash-next-2x3090/releases/download/v0.2.0/qwen38-public-benchmarks-2026-09-05.tar.gz).
The deterministic archive is 1,609,705 bytes with SHA-256
`84bac9353fc277f93c00e5790d2b31367c42d2d9e582de6af09cc1d16ec755e1`.
