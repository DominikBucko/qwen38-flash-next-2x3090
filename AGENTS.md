# Field notes for agents

Read this before changing the checkpoint, serving flags, scheduler, or benchmark
scripts. The README is for users. This file records the engineering logic that
is easy to lose when looking only at the final configuration.

## Ground truth

The validated target is one Qwen3.8-Flash-Next request spread across two RTX
3090 24 GB cards, backed by 128 GB of system memory. The two GPUs do not each
hold an independent 256K context. Tensor parallelism splits one model instance
and one 262,144-token sequence across both cards.

The published checkpoint is:

```text
albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE
canonical tensor revision: ef554143369a706525336f6b42a09094835dc077
```

`repro.lock.json` is authoritative for upstream revisions, the container digest,
tensor counts, and runtime versions. Do not replace a full revision or digest
with `main`, `latest`, or a floating image tag.

When a verified result changes, update the machine-readable benchmark JSON, the
generated graph, this file, and any public claim in the same commit. Do not let
an old explanation survive after its underlying flag or workload changes.

Keep these benchmark shapes separate:

| Claim | Request shape | Result |
|---|---:|---:|
| Peak prefill | 65,536 input tokens | 1,402 prompt tok/s |
| Balanced full-context prefill | 262,016 input + 128 output | 1,275.583 prompt tok/s |
| Decode after full context | 262,016 input + 128 output | 54.485 output tok/s |
| Matched short-decode endpoint | 128 input + 256 output, 10 requests | 80.08 output tok/s |
| Peak warmed long decode | 128 input + 4,096 output, one request | 135.21 output tok/s |

The 1,402 prefill and 135.21 decode figures are not from the same request. The
80.08 and 135.21 decode figures are not directly comparable either. The longer
decode amortizes one-time request overhead and gives MTP much more time to pay
for itself. Preserve this distinction in every chart, report, and regression
test.

## The model is large for a reason

Calling the checkpoint “INT4” is incomplete. The target backbone uses Intel's
AutoRound W4A16 packing, but Qwen3.8-Flash-Next also has a 51.2B-parameter PLE
n-gram table. That table was 102.4 GB in BF16. The release uses the published
RadixArk FP8 E4M3FN table and scale, which cuts it roughly in half without
inventing a new local quantization scheme.

The release payload is therefore expected to remain large:

- target checkpoint: 116.183 GiB, 222,716 tensors in 25 safetensors files;
- compact draft: 3.855 GiB, 4,639 tensors in two safetensors files;
- target backbone: Intel AutoRound W4A16, symmetric INT4 group-128;
- target layers excluded by Intel's quantization policy: original BF16;
- PLE table: published FP8 E4M3FN plus its global scale;
- MTP routed experts: local symmetric INT4 group-32;
- remaining MTP tensors: copied in their source dtype;
- KV cache: BF16.

Do not “fix” the size by quantizing everything to four bits. That would change
the quality contract. In particular, do not requantize Intel's target tensors,
discard the PLE scale, quantize the sensitive BF16 tensors, or change KV dtype
without a quality evaluation that can detect the loss.

## Why vLLM became the target

The early GGUF route was useful for proving that the model could prefill at a
reasonable rate, but it was the wrong place to optimize this architecture. The
hard parts are not a conventional dense INT4 backbone. They are PLE lookup,
hybrid recurrent state, sparse QSA, hundreds of routed experts, and MTP. The
vLLM Qwen3.8 implementation exposed the scheduler, cache lifecycle, MoE backend,
and speculative path needed to work on all of those together.

The final runtime is not stock upstream vLLM. It starts from the digest-pinned
day-zero Qwen3.8 image and installs the exact overlay under
`runtime/vllm-overlay`. `runtime/vllm-overlay/SHA256SUMS.json` prevents a silent
partial overlay. If the base image changes, assume every copied file needs a
three-way review; do not copy the old overlay onto a new vLLM revision and call
it compatible.

## Checkpoint assembly decisions

### Keep Intel packing intact

`scripts/build_intel_fp8ple_hybrid.py` reads the upstream indexes and builds a
new weight map. It does not decode and repack Intel's GPTQ/AutoRound weights.
Unchanged shards are hard-linked by default. The builder:

1. omits Intel shard 16, which contains only the BF16 PLE table;
2. omits Intel's bundled BF16 MTP tensor file;
3. inserts 128 RadixArk FP8 PLE shards plus the published scale tensor;
4. changes `ple_embedding_dtype` to `float8_e4m3fn`;
5. writes `hybrid_sources.json` and a new auditable safetensors index.

Do not infer a broken download from gaps in shard numbering. The index is the
source of truth. `scripts/validate_hybrid.py` checks every indexed tensor against
the actual safetensors headers, rejects duplicates and unindexed tensors, and
verifies that the target has PLE tensors but no bundled MTP tensors.

### Make MTP separate and small

The target does not need its bundled BF16 MTP copy. A standalone draft makes it
possible to compress draft experts aggressively while leaving target decisions
unchanged.

`scripts/build_mtp_int4.py` quantizes only the 512 routed draft experts. It
splits the source gate/up tensor into separate gate and up projections and packs
gate, up, and down weights as symmetric INT4 group-32. Group-32 costs more scale
metadata than group-128 but is a reasonable quality/performance choice for the
small verifier draft.

The development draft uses symlinks to source tensors. That is convenient
locally and wrong for a public model. `scripts/compact_mtp_checkpoint.py`
materializes only tensors selected by the draft index into:

```text
runtime/mtp-int4-g32/mtp-dense.safetensors
runtime/mtp-int4-g32/mtp-routed-experts-int4.safetensors
```

The compact checkpoint has no symlinks. `scripts/validate_compact_mtp.py`
stream-hashes every materialized tensor against the development draft. Preserve
that check if the packing or file layout changes.

### Why FP8 PLE instead of disk or INT4

PLE is an embedding-style lookup over a huge table. Keeping it on disk would put
latency behind page faults and storage locality. It might look acceptable after
the page cache is warm and collapse on an unseen access pattern. The validated
path keeps it resident in system memory.

FP8 halves the original BF16 table and uses an already published scale. INT4
would save more memory, but it would introduce another unvalidated quality
change in a component that directly injects token-history features. Treat an
INT4 PLE experiment as a new checkpoint, not a runtime optimization.

## Memory placement during serving

The final layout is intentionally asymmetric:

- dense layers, attention work, shared experts, and hot routed experts run on
  the GPUs;
- the complete routed-expert pool remains addressable in pinned system memory;
- an 88-expert-per-layer GPU cache is the full-context default;
- the dynamic LRU changes which experts occupy those slots as the sequence
  evolves;
- the FP8 PLE table belongs to a dedicated CPU process;
- the BF16 KV allocation is 4,429,185,024 bytes, about 4.13 GiB per GPU;
- the compact MTP draft runs beside the target and proposes three tokens.

Cold experts are not dead weights. A token that routes to one still uses it. The
runtime reads it from the host-backed pool or replaces an LRU slot. Offload saves
VRAM; it does not prune the model.

The PLE path is different from copying the whole table to CUDA. The CPU worker
owns the embedding weights, performs the lookup, and publishes the small result
through page-locked shared host buffers. The GPU stream waits on mapped flags,
copies the result, and releases the buffer. The relevant files are:

- `model_executor/layers/ple_offload_layer.py`;
- `v1/ple_offload/worker.py`;
- `v1/ple_offload/connector.py`;
- `models/qwen3_8_flash_next/nvidia/ple_layer.py`.

This is why system-memory latency and bandwidth affect decode even though the
main kernels run on CUDA.

## The serving profile is a coupled system

The checked-in default is `configs/2x3090-128gb.env` plus
`scripts/serve-container.sh`. Important values are:

```text
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=1
MAX_NUM_BATCHED_TOKENS=4096
KV_CACHE_MEMORY_BYTES=4429185024
CPU_OFFLOAD_GB=30
VLLM_WNA16_STATIC_HOT_CACHE_SIZE=88
VLLM_WNA16_STATIC_HOT_CACHE_MAX_TOKENS=16
VLLM_PREFIX_CACHE_RETENTION_INTERVAL=1600
VLLM_PLE_OFFLOAD_READY_TIMEOUT=1200
MTP_DEPTH=3
VLLM_QSA_EXACT_TOPK=0
```

`CPU_OFFLOAD_GB=30` is not the total host-memory footprint. The PLE worker owns
its large table separately, and the expert tier has additional storage and
metadata. Likewise, the 16-token hot-cache threshold keeps the LRU fast path on
small decode-shaped batches instead of making a long prefill churn expert slots.

Important launch choices:

- TP2 and EP2 split one request across both cards.
- `allgather_reducescatter` is the selected MoE all-to-all backend.
- Humming runs the target W4A16 MoE path.
- Marlin-compatible packing runs the compact INT4 draft experts.
- UVA supplies the host-backed expert tier.
- Chunked prefill uses a 4,096-token scheduling budget.
- Prefix caching uses `--mamba-cache-mode align`.
- CUDA graphs are limited to `FULL_DECODE_ONLY`.
- Async scheduling is disabled.
- Custom all-reduce is disabled because CUDA peer access was unavailable on the
  validated two-card topology.

Do not tune one of these as if it were independent. Raising the hot-cache size
takes VRAM away from KV and transient prefill buffers. Raising the batched-token
budget changes prefill speed and transient memory. Changing speculative depth
changes draft memory, scheduler lookahead, recurrent-state rollback, and
acceptance. A configuration that wins at 128 tokens may fail at 262K.

## How the hillclimb happened

The matched series used 10 requests per configuration, each with 128 input and
256 output tokens. Its metric is reciprocal mean time per output token. These
are the steps agents should use when reasoning about the final design.

### 1. BF16 target with MTP2: 32.83 tok/s

This established that the model, recurrent state, PLE path, and speculative
plumbing worked. It was not a viable bandwidth profile. Too much target weight
traffic remained in BF16.

### 2. Intel W4A16 group-128: 34.71 tok/s, +1.88

The first quantized run was only a small win because weight format was not the
only bottleneck. Expert placement and dispatch still pulled too much data from
system memory. The important decision was to use Intel's checkpoint as packed,
not to produce another local target quant.

### 3. Keep more experts on CUDA: 41.10 tok/s, +6.39

This was the first clear sign that expert misses dominated decode. More resident
experts saved host reads and transfer synchronization. From this point onward,
expert-cache policy mattered more than another small dense-kernel tweak.

### 4. Pinned H2D plus chunked host cache: 43.68 tok/s, +2.58

Pinned storage removed avoidable staging, and chunk caching reduced the amount
of work repeated for a miss. This was useful but not sufficient: a miss still
sat on the critical token path.

### 5. Static hot-96: 49.81 tok/s, +6.13

A per-layer ranking kept the common experts resident. The layout was stable
enough for CUDA graph capture, so this improved both expert locality and launch
behavior. `configs/static_hot_cache_rankings.json` is an input, not a universal
truth; it reflects measured traffic and can be workload-dependent.

### 6. Mixed VMM hot-128: 57.37 tok/s, +7.56

The mixed allocation created one logical expert tensor with a CUDA-resident hot
prefix and host-backed cold suffix. Kernels kept ordinary contiguous pointer
semantics while physical pages lived in different memory tiers. This removed
extra dispatch paths and allowed Marlin, Triton, and Humming comparisons on the
same placement scheme.

The best short-context capacity was not the full-context default. The released
profile uses 88 slots to leave enough VRAM for the 256K KV state and other
buffers.

### 7. Fused QSA: 59.97 tok/s, +2.60

Sparse attention had too much launch and selection overhead at decode batch
sizes. The fused QSA implementation reduced intermediate materialization and
repeated block selection. Its approximate persistent selector is fast because
it does not compute a full exact score ranking every step.

The exact selector remains available with `VLLM_QSA_EXACT_TOPK=1`. It is a
quality experiment, not the default.

### 8. Humming target plus Marlin draft: 65.46 tok/s, +5.49

One backend did not win every shape. Humming was the better target MoE path;
Marlin was the right fit for the packed INT4 MTP draft. Preserve the split when
testing backend changes. “Use backend X everywhere” is not a useful hypothesis.

### 9. Dynamic LRU-100: 78.73 tok/s, +13.27

This was the largest single decode gain. Static popularity is a good initial
state, but expert demand changes within a sequence. The LRU uses the static list
to seed GPU slots, then replaces cold residents with experts actually requested
by current tokens. This attacked the remaining host-miss rate directly.

### 10. Dynamic LRU-104: 80.08 tok/s, +1.35

Four more slots still helped, but returns were flattening and the extra VRAM was
not free. The matched workload finished 2.44 times faster than the BF16/MTP2
baseline.

## Why MTP pushed decode above 100 tok/s

On the warmed 128-input/4,096-output workload:

| Mode | Decode |
|---|---:|
| Target only | 61.32 tok/s |
| Fixed MTP3 | 119.94 tok/s |
| Variable-K scheduler path with MTP3 | 135.21 tok/s |

MTP proposes up to three draft tokens and the target verifies a block. High
acceptance amortizes target invocations over several emitted tokens. Two final
repeat probes measured:

| Decode | Acceptance | Mean accepted length |
|---:|---:|---:|
| 133.976 tok/s | 86.328% | 3.590 |
| 127.102 tok/s | 90.758% | 3.723 |

The accepted-length metric can exceed draft depth three because it includes the
target/verifier token emitted with accepted draft tokens.

The “adaptive MTP3” label needs care. `VLLM_FORCE_DYNAMIC_SPEC_SCHEDULING=1`
selects vLLM's variable-K scheduling machinery, but the checked-in lookup keeps
K fixed at three for the supported batch size. The gain came from that scheduler
path and its padding/placeholder behavior, not from allowing arbitrary draft
depth.

Qwen3.8's hybrid state made speculative decoding more than a command-line flag.
The overlay includes rollback-safe PLE n-gram context, QSA compressor state,
Mamba/GDN state handling, hyperconnection streams, and scheduler lookahead at
chunk boundaries. If speculative output becomes incorrect, inspect state
commit/rollback and placeholder alignment before blaming the draft weights.

MTP changes speed and acceptance, not the target's verification rule. Do not
describe draft INT4 as if the target itself had been requantized to group-32.

## Full-context behavior

The boundary test uses 262,016 input tokens and 128 output tokens, exactly
262,144 total. It measured:

```text
TTFT: 205.409 s
prefill: 1,275.583 prompt tok/s
decode after prompt: 54.485 output tok/s
```

The lower decode rate is real. At 256K, more KV and sparse-attention state sit on
the critical path. Do not use the 135 tok/s short-prompt number to predict decode
after a full prompt.

A static expert profile reached 1,629 prompt tok/s at the same context boundary,
but it was not the best combined prefill/decode configuration. The released
1,275.6 profile is intentionally balanced. If an experiment reports only a
prefill record, test post-prefill decode before adopting it.

`MAX_NUM_SEQS=1` is deliberate. The allocation targets one real 256K sequence,
not serving concurrency. Increasing concurrency without a new memory plan is a
different product target.

## Prefix caching was an architectural fix

Ordinary KV prefix caching was not enough for this model. Reusing an attention
block while losing or misaligning the hybrid recurrent state gives either a
cache miss or wrong continuation state. The patched runtime aligns cached
boundaries with Mamba blocks and carries QSA/PLE state through the request
lifecycle.

Use `--mamba-cache-mode align`; the model explicitly rejects `all`. When testing
cache changes:

1. send the exact same prefix twice;
2. confirm the second request reports a real cache hit;
3. compare generated tokens against a no-cache run;
4. test a prefix ending off the normal block boundary;
5. include a tool-style multi-turn request, not only a repeated synthetic
   string;
6. measure both latency saved and host/GPU state retained.

Agent-harness token accounting is not an isolated prefill benchmark. Tool turns,
prefix hits, repeated context, non-streaming time, and generation all mix
together. Use isolated requests for pp/tg claims and the harness only for
end-to-end task behavior.

## QSA precision decision

Approximate QSA is the default because it is faster and the available quality
evidence did not justify the exact cost. The cache-fixed one-trajectory A/B was:

| QSA mode | Strict passes | Hidden-functional passes | Mean score | Workflow passes |
|---|---:|---:|---:|---:|
| Exact | 9/15 | 10/15 | 96.953 | 15/15 |
| Approximate | 7/15 | 11/15 | 97.398 | 14/15 |

This is mixed, noisy evidence. Exact improved strict workflow completion, while
approximate had one more hidden-functional pass and a 0.445 higher mean. One
trajectory per mode is not enough for a quality claim. Run at least three fresh
trajectories per configuration before changing the default.

`tests/test_qsa_exact_topk_cpu.py` verifies the exact selector's visible sets
against `torch.topk` and checks repeatability. That proves a local selection
property, not model-level quality.

## AgentBench notes

The published main run used xhigh reasoning, 32 steps, a 16,384-token response
cap, and one trajectory for each of 15 tasks. It produced 9 strict passes, 11
hidden-functional passes, a 97.601 mean score, 3,841,609 prompt tokens, 196,495
completion tokens, and 313 tool calls. All 15 trajectories ended with a normal
stop.

Long reasoning can still waste budget by revisiting the same hypothesis or
reissuing equivalent tool calls. Do not “solve” that only by shrinking the token
cap; a smaller cap hides the behavior and can cut off legitimate work. When
evaluating a mitigation, retain tool-call and reasoning traces privately and
look for:

- repeated plans with no new evidence;
- the same command or query issued with cosmetic changes;
- failure to update the working hypothesis after a tool result;
- long conclusions that do not change the patch or answer;
- repeated recovery from a tool error without changing the approach.

Harness engineering must not leak into model claims. Keep response parsing,
reasoning preservation, prefix reuse, stop conditions, and endpoint timing
separate in the report. Never publish private tasks, hidden assertions, oracle
code, raw traces, patches, or final workspaces.

## How to run a useful performance experiment

1. Start from the pinned image, checkpoint revision, and released environment.
2. Change one mechanism at a time.
3. Record every environment variable and serving flag.
4. Warm the model before measuring decode.
5. Use the 128-input/256-output matched workload for hillclimb comparisons.
6. Use 128-input/4,096-output only for long-decode and MTP studies.
7. Record MTP acceptance and accepted length with decode throughput.
8. Test 262,016 input + 128 output before claiming full-context compatibility.
9. Record TTFT, pp, tg, request completion, and any stream error separately.
10. Re-run enough times to distinguish a real gain from cache warmth and
    sequence-dependent expert routing.
11. Keep the old result. A hillclimb without the losing configurations is not
    useful evidence.

When a change improves short decode, ask whether it consumed VRAM needed by the
full-context cache. When it improves prefill, ask whether post-prefill decode
regressed. When it improves MTP throughput, inspect acceptance. When it changes
QSA, cache state, PLE precision, or target weights, run a quality evaluation.

## Startup and validation sequence

For the published checkpoint:

```bash
export MODEL_DIR=/models/qwen38-flash-next
make build-image
make preflight
make serve
```

Before publishing source changes:

```bash
make validate
python scripts/check_release_ready.py
bash -n scripts/*.sh
```

CI can check source syntax, overlay hashes, lock consistency, accidental model
blobs, symlinks, and common token formats. It cannot prove CUDA correctness,
the 256K allocation, PLE worker synchronization, prefix-cache correctness, MTP
acceptance, or throughput. Those require the target hardware.

Before publishing a rebuilt checkpoint, run all three validators:

```text
scripts/validate_hybrid.py
scripts/validate_compact_mtp.py
scripts/validate_upload.py
```

`validate_upload.py` must report zero symlinks and exact agreement between each
index and the safetensors headers.

## Failure modes worth checking first

### Server appears hung during startup

The CPU PLE worker has to load roughly half the checkpoint and register shared
buffers. The ready timeout is 1,200 seconds for a reason. Check worker progress,
resident memory, and actual disk reads before killing it. Do not confuse a long
weight load with a 256K prefill.

### Host starts swapping

Stop. Decode depends on resident PLE and expert pages. Swap turns predictable
memory access into storage latency and invalidates performance results. Disk is
not a supported overflow tier for this profile.

### Decode is much slower than expected

Check, in this order:

1. the INT4 target was loaded with its native AutoRound metadata;
2. Humming is active for target MoE;
3. the hot-cache ranking file was found inside the container;
4. dynamic LRU and mixed VMM are enabled;
5. MTP loaded the compact draft rather than the target or BF16 bundled draft;
6. speculative acceptance is nonzero and close to the recorded range;
7. the request is not a 256K post-prefill decode being compared with 128 input;
8. exact QSA was not enabled accidentally;
9. the host is not swapping;
10. CUDA graph capture did not fall back for decode.

### Full context OOMs while short tests pass

Verify the explicit KV byte allocation and `MAX_NUM_SEQS=1`. Then inspect hot
cache capacity, batched-token budget, and transient allocations. Do not reduce
`MAX_MODEL_LEN` and still call the result a full-context profile.

### Prefix caching reports hits but output changes

Suspect hybrid state. Inspect Mamba-aligned splitting, QSA compressor state,
PLE n-gram context, copy-on-write block retention, and speculative rollback.
A KV block hit alone is not sufficient evidence of a correct cache hit.

### MTP works without chunked prefill and fails with it

Inspect scheduler lookahead and placeholder rows at chunk boundaries. Multi-step
MTP reads farther ahead than Eagle-style one-token drafting. The patched
scheduler reserves `num_spec_tokens` of prefill lookahead and drops padding
rather than shortening an invalid speculative tail.

### A shard number appears missing

Read `model.safetensors.index.json`. Validate the mapped filenames and headers.
Do not manufacture or download an unreferenced file merely to make numbering
contiguous.

### Hub upload appears to require another full copy

The assembly uses hard links for unchanged target files; keep source directories
until upload finishes. The public draft must be compacted because Hub uploads
cannot rely on local symlinks. Xet deduplicated the published 121 GB tree against
upstream chunks, so the first release transferred only 2.36 GB over the network.

## Ideas that are still worth testing

These are experiments, not promised wins:

1. Relearn hot-expert rankings from a broader workload, then compare static
   initialization plus LRU against a cold LRU.
2. Sweep cache capacity around the released 88-slot full-context point. Measure
   VRAM headroom, miss rate, short decode, and post-256K decode together.
3. Test MTP depth and scheduling with acceptance-aware reporting. More draft
   tokens can lose when verification or rollback cost grows.
4. Repeat exact-versus-approximate QSA with at least three independent
   AgentBench trajectories per mode.
5. Measure prefix-cache hit rate and latency on realistic multi-turn agent
   transcripts rather than repeated synthetic prompts.
6. Revisit collectives only on a topology where CUDA peer access is actually
   available. Do not assume an x16 link implies CUDA P2P.
7. Profile host expert misses and PLE lookup separately. Both touch system
   memory, but they need different fixes.
8. Evaluate any lower-precision KV or PLE variant as a new quality/performance
   point, with the released BF16-KV/FP8-PLE model as the control.

## Files to inspect before editing a subsystem

| Subsystem | Start here |
|---|---|
| Immutable versions and sizes | `repro.lock.json` |
| Final launch flags | `scripts/serve-container.sh` |
| Full-context defaults | `configs/2x3090-128gb.env` |
| Runtime overview | `docs/architecture.md` |
| Benchmark protocol | `docs/benchmarks.md` |
| Target + FP8 PLE assembly | `scripts/build_intel_fp8ple_hybrid.py` |
| MTP quantization | `scripts/build_mtp_int4.py` |
| Self-contained draft | `scripts/compact_mtp_checkpoint.py` |
| PLE process and IPC | `runtime/vllm-overlay/v1/ple_offload/` |
| Expert hot cache and LRU | `runtime/vllm-overlay/model_executor/layers/quantization/auto_gptq.py` and `compressed_tensors_moe_wna16.py` |
| QSA selector and kernels | `runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/ops/qsa.py` |
| Hybrid QSA cache | `runtime/vllm-overlay/models/qwen3_8_flash_next/common/qsa_cache.py` |
| MTP model integration | `runtime/vllm-overlay/models/qwen3_8_flash_next/nvidia/mtp.py` |
| Scheduler and prefix alignment | `runtime/vllm-overlay/v1/core/sched/scheduler.py` |
| Reproducible build | `docs/reproduce.md` |

The central lesson is that this result did not come from one fast kernel. It
came from choosing a trustworthy target quant, reducing the one table that
dominated system memory, placing experts according to live demand, making the
hybrid cache lifecycle correct, and then giving MTP a scheduler path where high
acceptance could translate into fewer target steps. Preserve that system view.
