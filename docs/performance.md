# Qwen3.8 Flash Next performance tuning on two 24 GB GPUs

## Start with the released profile

Use [`configs/2x3090-128gb.env`](../configs/2x3090-128gb.env) and
[`scripts/serve-container.sh`](../scripts/serve-container.sh) unchanged for the
first successful run. That combination is the tested 2× RTX 3090, 128 GB RAM,
single-request profile. It selects TP2/EP2, an 88-expert GPU cache, approximate
QSA, MTP depth 3, chunked prefill, BF16 KV, and a 262,144-token limit.

There is no globally “fastest” setting. Prefill, short decode, decode after a
full prompt, concurrency, and memory headroom stress different parts of the
system. Choose the request shape that represents the workload before changing
a flag.

## September 5 verified candidate profile

The native candidate used the following measured overrides while preserving the
262,144-token limit and single-request shape:

```bash
VLLM_WNA16_STATIC_HOT_CACHE_SIZE=84
DISABLE_CUSTOM_ALL_REDUCE=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

CUDA P2P worked in both directions. The environment was clean pinned vendor
vLLM plus the public overlay using existing native dependencies, not a fresh
Docker build. All 27 model weight files matched the published SHA-256 manifest
for canonical tensor revision `ef554143369a706525336f6b42a09094835dc077`.
These candidate overrides do not replace the released 88/1/True defaults.

| Public recipe | Exact shape and run policy | API-observed output speed | TTFT |
|---|---|---:|---:|
| `repo-chat` long | 258,048 input + 4,096 output; 3 measured, no explicit warmup | **75.636 tok/s aggregate** (74.031–76.707) | 211.059–215.128 s |
| `repo-chat` short | 128 input + 4,096 output; 1 warmup + 3 measured | **77.2845 tok/s aggregate** (74.746–79.707) | 0.897–0.903 s |

The long-run rates were 74.030749, 76.707158, and 76.224624 tok/s, producing a
75.6360734 tok/s reciprocal-mean aggregate; their TTFT values were 215.128,
211.059, and 211.228 seconds. Four recoverable
allocator warnings appeared during the first long prefill, but every measured
stream completed with exact API usage counts. See the
[September 5 bundle](../benchmarks/2026-09-05/README.md),
[summary JSON](../benchmarks/2026-09-05/summary.json), and
[long-context chart](images/long-context-decode.svg).

A separate raw `repo-code` diagnostic used hot-cache 86, custom all-reduce
disabled, and expandable segments enabled. Its 258,048 + 4,096 three-run
aggregate was 78.8238 tok/s. Because both the prompt recipe and runtime profile
differ, it is not a controlled custom-all-reduce comparison and should not be
used as a promotional result.

## Read every published number with its request shape

The following table records historical release and hillclimb measurements:

| Measurement | Exact shape | Result | Scope |
|---|---:|---:|---|
| Peak prefill | 65,536 input tokens | 1,402 prompt tok/s | Selected prefill record |
| Balanced full-context prefill | 262,016 input + 128 output | 1,275.583 prompt tok/s | Released profile |
| Full-context boundary probe | 262,016 input + 128 output | 54.485 output tok/s | Short output; capacity evidence |
| Matched short decode | 10 requests, each 128 input + 256 output | 80.08 output tok/s | Reciprocal mean TPOT |
| Peak warmed long decode | One request, 128 input + 4,096 output | 135.21 output tok/s | Hillclimb endpoint |
| Warm long-decode repeats | One request, 128 input + 4,096 output | 127.10 and 133.98 output tok/s | Released probes |

The 1,402 prefill and 135.21 decode results are from different requests. The
80.08 and 135.21 decode results are also not directly comparable: the longer
generation amortizes request overhead and gives MTP more time to help. The
54.485 result is a short boundary probe, not sustained long-context generation.

Trace the values to the checked-in data:

- [`benchmarks/2026-09-05`](../benchmarks/2026-09-05/README.md) records the new
  token-exact public probes and candidate runtime facts;
- [`benchmarks/serving-summary.json`](../benchmarks/serving-summary.json) records
  the released full-context request and repeated warm probes;
- [`benchmarks/hillclimb.json`](../benchmarks/hillclimb.json) records the matched
  short-decode series, long-decode study, and selected prefill results;
- [`benchmarks/agentbench-summary.json`](../benchmarks/agentbench-summary.json)
  records sanitized task-level quality aggregates, not isolated throughput.

## Bring up and smoke-test the endpoint

The supported launch path is:

```bash
# On first setup, copy .env.example to .env.
# Set MODEL_DIR in .env to the absolute checkpoint directory.
make preflight
make serve
```

In another shell, this text-only request verifies the OpenAI-compatible route:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-Flash-Next","messages":[{"role":"user","content":"Reply with the word ready."}],"temperature":0,"max_tokens":32}'
```

This is a smoke test, not a benchmark. Shell wall time cannot separate token
accounting, queue time, TTFT, prefill, and decode.

## Collect reproducible measurements on your machine

The standard-library [benchmark client](../scripts/benchmark_serving.py) collects
**new synthetic measurements** from the pinned runtime. The `repo-chat` recipe
asks for a technical tutorial using public repository code. It obtains the chat
template from the server, preserves the instruction and generation wrappers,
and repeats/truncates only the source interior to the exact input length. It
requests an exact output length and validates server usage and streamed counts. Every
request uses a unique cache salt to prevent prefix-block reuse. The expert LRU,
PLE pages, CUDA graphs, and other caches still warm across requests; keep run
order and warmup policy identical when comparing settings. The JSON retains the prompt recipe, per-chunk timing, protocol,
runtime lock, and any errors.

For warmed short-prompt decode:

```bash
python3 scripts/benchmark_serving.py \
  --prompt-style repo-chat \
  --input-tokens 128 --output-tokens 4096 --warmup 1 --runs 3 \
  --output work/short-decode.json
```

For sustained generation near the full context limit:

```bash
python3 scripts/benchmark_serving.py \
  --prompt-style repo-chat \
  --input-tokens 258048 --output-tokens 4096 --warmup 0 --runs 3 \
  --timeout 1800 --output work/long-context.json
```

258,048 input + 4,096 output reaches exactly 262,144 total tokens. This gives a
longer generation window than the historical 262,016-input/128-output boundary
probe. That short probe established capacity; it does not characterize sustained
long-context output speed.

Use `--base-url http://127.0.0.1:8000` (without `/v1`) and adjust the port if
needed. Add `--hardware-note` and `--server-settings-note` to record the host
and any overrides; avoid credentials and private paths. The client requires the
pinned vLLM extensions and stops on missing usage, a broken stream, or a token
count mismatch.

`repo-chat` and `repo-code` use sorted public runtime-overlay Python files and
record their hashes. At 128 input tokens, only a small source prefix fits; the
JSON records whether the complete corpus fits at longer lengths. `repo-code`
is an unwrapped raw continuation diagnostic: short probes showed degenerate
repetition, making their fastest rates unsuitable for promotion. The default
`repeated-seed` diagnostic uses a short periodic text that can favor cache
locality. Name the recipe whenever sharing a result.

Add `--capture-output` and inspect the generated text before interpreting a
rate. All recipes force length with `ignore_eos=true`; even chat formatting
does not guarantee useful output or turn this into a quality evaluation.
Counts include **all generated completion tokens**, including reasoning and
control tokens when emitted. The model's chat-template defaults are retained;
a 4,096-token cap can end during reasoning before a finished answer.

The reported decode estimate divides tokens **after the first output chunk**
by time until the final output chunk. SSE buffering and MTP batches make this
an API-observed metric. Across runs, use the reciprocal mean TPOT reported in
the summary rather than averaging rates for a historical TPOT comparison.
TTFT includes scheduling and first-token work; dividing
input tokens by TTFT is not an isolated prefill benchmark. These public synthetic
prompts do not reconstruct the original historical tests or measure model
quality. Review generated text and test real workloads before adopting a tuning
change. See [CONTRIBUTING.md](../CONTRIBUTING.md) to share a result.

## Pick a tuning goal

### Preserve the released 262K capability

Keep `MAX_NUM_SEQS=1`, `MAX_MODEL_LEN=262144`, and the explicit KV allocation.
Any change must still complete 262,016 input plus 128 output without a stream
error. A short prompt succeeding does not verify full-context compatibility.

The 88-slot expert cache is smaller than some short-context hillclimb settings
because the full-context KV state and transient buffers need VRAM too. A larger
cache may improve short decode and then fail at 262K.

### Reduce VRAM pressure

Follow the released recovery order: lower
`VLLM_WNA16_STATIC_HOT_CACHE_SIZE` from 88 to 86, then 84; only then consider
lowering `KV_CACHE_MEMORY_BYTES` from 4,429,185,024 to 4,294,967,296. Confirm the
startup `GPU KV cache size:` line still covers the target length. Lowering
`MAX_MODEL_LEN` alone does not release the explicit KV pool.

If the first prompt OOMs, lowering `MAX_NUM_BATCHED_TOKENS` from 4,096 to 2,048
can reduce prefill temporaries, with lower prefill throughput. See
[Memory sizing and OOM recovery](memory.md) before combining changes.

### Improve decode for your prompt mix

Expert locality and MTP acceptance are workload-dependent. Measure hot-cache
miss behavior, output throughput, MTP acceptance percentage, and mean accepted
length together. The checked-in ranking seeds the dynamic LRU from observed
traffic; it is not a universal expert-popularity ordering.

Change one mechanism per run. More resident experts consume VRAM. A different
MTP depth changes draft memory, lookahead, rollback work, and acceptance. Treat
an unlisted combination as an experiment until it passes the same memory and
quality checks as the release.

### Improve prefill

Measure prompt throughput and TTFT, then measure decode after that same prompt.
The 1,629 prompt tok/s static-cache record was not selected as the default
because its decode behavior was worse. `MAX_NUM_BATCHED_TOKENS` also trades
prefill efficiency against transient VRAM.

### Evaluate exact QSA

Approximate QSA is the released performance default. The provided
[`configs/exact-qsa.env`](../configs/exact-qsa.env) enables a slower precision
experiment with `VLLM_QSA_EXACT_TOPK=1`. The one-trajectory AgentBench A/B was
mixed and is not enough to claim a quality improvement. Run at least three
fresh trajectories per mode before drawing a model-quality conclusion.

## Reproducible comparison protocol

For each result:

1. Pin the checkpoint revision and container digest from
   [`repro.lock.json`](../repro.lock.json).
2. Record the GPUs, driver, CPU, RAM configuration, NUMA/topology details, and
   whether the host swapped during generation.
3. Record every environment override and final vLLM serving flag.
4. Record the explicit warmup policy. Keep it identical for comparisons; if
   there was no explicit warmup, say so.
5. Compare identical input/output token counts, sampling settings, request
   count, concurrency, and cache state.
6. For the matched hillclimb shape, run 10 requests of 128 input and 256 output
   tokens and compute reciprocal mean time per output token.
7. Use 128 input plus 4,096 output only for long-decode or MTP studies; report
   acceptance and accepted length.
8. Test 262,016 input plus 128 output before claiming full-context support.
9. Record TTFT, prompt tok/s, output tok/s, completion status, and stream errors
   separately.
10. Repeat enough runs to expose cache warmth and sequence-dependent expert
    routing; retain losing configurations as well as winners.

If QSA, PLE precision, KV precision, target weights, prefix caching, or hybrid
state handling changes, throughput is insufficient evidence. Add a quality or
correctness evaluation appropriate to that mechanism.

## Diagnose unexpectedly slow decode

Check that the native AutoRound metadata loaded, Humming is active for target
MoE, the ranking file exists inside the container, dynamic LRU and mixed VMM are
enabled, and the compact MTP draft loaded. Then inspect MTP acceptance, exact
QSA, active host swapping, and CUDA graph fallback. Finally confirm that a
256K post-prefill result is not being compared with a 128-token prompt.

## CUDA P2P and custom all-reduce

The launcher defaults to `DISABLE_CUSTOM_ALL_REDUCE=1`, preserving the original
configuration. This disables vLLM's custom collective; it does **not** disable
CUDA peer access or require NCCL to use host staging. The driver must already
provide peer access before a custom-collective experiment can work.

The pinned custom path also needs an IPC-compatible allocator. On the September
5 test host, peer access worked in both directions, but custom all-reduce with
the default `expandable_segments:True` failed after CUDA graph capture at
`custom_all_reduce.cuh:164` with `invalid argument`. The failure matches the
allocator/legacy IPC incompatibility described in
[vLLM PR #43923](https://github.com/vllm-project/vllm/pull/43923).

The measured September 5 candidate profile used these overrides plus an
84-expert hot cache:

```bash
VLLM_WNA16_STATIC_HOT_CACHE_SIZE=84
DISABLE_CUSTOM_ALL_REDUCE=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

The Docker launcher forwards both from `.env`. Disabling expandable segments
can change fragmentation and VRAM headroom; a small collective test does not
prove that the full model fits. Validate startup, exact output counts, and the
full-context workload before adopting these settings. Compare the same public
workload, warmup, and cache settings. A peer-access capability check alone is
not a performance result. On the September 5 host, this full model profile
completed three exact-count 258,048-input/4,096-output chat runs at a 75.636
tok/s aggregate. Four recoverable allocator warnings appeared during the first
long prefill, with no stream failures.

With serving stopped, test small eager and CUDA graph sums in the built image:

```bash
docker run --rm --gpus all --ipc host --entrypoint torchrun \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e VLLM_SKIP_P2P_CHECK=1 \
  qwen38-flash-next-2x3090:locked \
  --standalone --nproc-per-node=2 /opt/qwen38/scripts/check_custom_all_reduce.py
```

The [test](../scripts/check_custom_all_reduce.py) checks CUDA peer capability
on each rank and exact eager/graph sums. On the September 5 native test host,
`expandable_segments:True` reproduced the graph IPC error and `False` passed.
This is a collective correctness check, not a model-quality or speed benchmark.
The launcher now rejects the known incompatible combination before loading
weights, with instructions for correcting it.

Do not infer a custom-all-reduce speedup from the separate 78.8238 tok/s raw
`repo-code` diagnostic: it used a different prompt recipe, hot-cache size, and
allocator profile. A speed claim needs a same-prompt, same-profile matched run.

## Related documentation

- [README and published results](../README.md)
- [Benchmark evidence](benchmarks.md)
- [Memory sizing and OOM recovery](memory.md)
- [Hardware requirements and compatibility](hardware.md)
