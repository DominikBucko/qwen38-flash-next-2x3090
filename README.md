# Qwen3.8-Flash-Next on 2× RTX 3090

<h2 align="center">1,402 tok/s prefill · 135.2 tok/s decode</h2>
<p align="center"><strong>262,144-token context · 2× RTX 3090 (24 GB) · 128 GB system memory</strong></p>
<p align="center"><a href="https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE"><strong>Download the checkpoint</strong></a></p>

Qwen3.8-Flash-Next, with its high sparsity and low active param count is a great candidate for CPU offloading under right setup. This build keeps the
full expert set and an FP8 Ngram table in system memory, caches active (LRU) experts on
the GPUs, and uses a small MTP3 drafter to recover decode speed.

The [checkpoint](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) combines Intel's AutoRound W4A16 target with FP8 PLE/ngram
tensors, to fit into 128GB memory. The serving code is a pinned vLLM build plus the
patches in this repo.

## Results

| Workload | Shape | Speed |
|---|---:|---:|
| Prefill | 65,536 input tokens | **1,402 prompt tok/s** |
| Full-context prefill | 262,016 input + 128 output | **1,275.6 prompt tok/s** |
| Warm decode | 128 input + 4,096 output | **135.2 output tok/s** |
| Warm decode, repeated | 128 input + 4,096 output | **127.1–134.0 output tok/s** |
| Decode after a 262K prompt | 262,016 input + 128 output | **54.5 output tok/s** |
| Longest tested sequence | Input + output | **262,144 tokens** |

These are single-request measurements. Prefill and decode use different test
shapes; 1,402 and 135.2 tok/s did not come from the same request. The chart marks
the switch from a 256-token decode test to a 4,096-token test with a dotted line.

![Qwen3.8-Flash-Next performance hillclimb](docs/images/hillclimb.svg)

## Speed hillclimb

In the blue series, every point uses 128 input tokens and
256 output tokens. Decode rose from 32.83 to 80.08 tok/s on that fixed workload.

1. **BF16 + MTP2 — 32.83 tok/s.** The original vLLM baseline.

2. **Intel W4A16 — 34.71 tok/s (+1.88).** INT4 cut backbone weight traffic.
   Layers that Intel left in BF16 stayed in BF16.

3. **Larger GPU expert set — 41.10 tok/s (+6.39).** More routed experts stayed
   in VRAM, so fewer token steps had to fetch an expert from system memory.

4. **Pinned copies and a host cache — 43.68 tok/s (+2.58).** Pinned memory made
   expert transfers cheaper. Chunk caching reduced the cost of a miss.

5. **Static hot-96 cache — 49.81 tok/s (+6.13).** The 96 busiest experts stayed
   on the GPUs. The stable layout also made CUDA graph capture practical.

6. **Mixed VMM hot-128 — 57.37 tok/s (+7.56).** The hot set grew to 128 experts
   while the complete expert pool remained addressable in system memory.

7. **Fused QSA — 59.97 tok/s (+2.60).** Fusing sparse-attention selection cut
   launch overhead and repeated block-selection work.

8. **Humming + Marlin — 65.46 tok/s (+5.49).** Humming handled the target MoE
   path; Marlin handled the quantized MTP draft.

9. **Dynamic LRU-100 — 78.73 tok/s (+13.27).** A runtime LRU beat the fixed hot
   list because it followed the experts used by the current sequence.

10. **Dynamic LRU-104 — 80.08 tok/s (+1.35).** Four more resident experts gave
    a small final gain. The matched test ended at **2.44×** its baseline speed.

### Why the graph continues past 80 tok/s

The gold points use a longer output, so they are not direct continuations of the
blue comparison. On the warmed 128-input/4,096-output test:

- Target only: 61.32 tok/s
- Fixed MTP3: 119.94 tok/s
- Adaptive MTP3: 135.21 tok/s

The longer run amortizes one-time request overhead. MTP accounts for most of the
remaining gain: the draft proposes several tokens and the target checks them
together. The target still verifies every accepted token.

### What 256K costs

With 262,016 input tokens already in the cache, decode falls to 54.5 tok/s. That
is the price of the larger KV state and longer attention path. The balanced
profile prefills the same prompt at 1,275.6 tok/s. A static expert cache reached
1,629 prompt tok/s, but its decode behavior was worse, so it is not the default.

## Checkpoint layout

| Part | Storage format |
|---|---|
| Target backbone | Intel AutoRound W4A16, symmetric INT4 group-128 |
| Sensitive target layers | Original BF16 |
| 51.2B-parameter PLE table | FP8 E4M3FN with the published scale |
| MTP routed experts | Symmetric INT4 group-32 |
| Other MTP tensors | Original source precision |
| KV cache | BF16 |

The target payload is 116.183 GiB. The compact MTP draft adds 3.855 GiB. Most of
the surprising size comes from the PLE table and the tensors that remain BF16,
not from an unquantized backbone.

## Run it

You need Linux, Docker, NVIDIA Container Toolkit, two 24 GB RTX 3090 cards, and
128 GB of system memory. Configure at least 32 GiB of swap on fast NVMe before
the first launch; 48–64 GiB is safer. The loader can exceed physical RAM even
though steady serving fits much more comfortably.

Download the pinned checkpoint revision:

```bash
hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir /models/qwen38-flash-next
```

Build and start the server:

```bash
git clone https://github.com/DominikBucko/qwen38-flash-next-2x3090.git
cd qwen38-flash-next-2x3090

cp .env.example .env
# Set MODEL_DIR in .env.

make build-image
make preflight
make serve
```

The OpenAI-compatible endpoint is `http://127.0.0.1:8000/v1`.

### If it runs out of memory

The checked-in profile is close to the VRAM limit. Check `free -h`,
`swapon --show`, and `nvidia-smi` first. Then make one change at a time:

1. Lower `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` from `88` to `86`, then `84` if
   needed. Each removed slot saves roughly 116 MiB per GPU across the 48
   layers, at the cost of more expert traffic from system memory.
2. Lower `KV_CACHE_MEMORY_BYTES` from `4429185024` to `4294967296`. Keep this
   only if the startup log still reports at least 262,144 KV-cache tokens.
3. Lower `MAX_NUM_BATCHED_TOKENS` from `4096` to `2048`. This reduces peak
   prefill temporaries but also reduces prefill throughput.

Changing `MAX_MODEL_LEN` alone does not release the explicitly reserved KV
allocation. For the full explanation, including two-client sizing, see
[`docs/memory.md`](docs/memory.md).

## Rebuild the checkpoint

The build scripts pin every upstream commit. They copy Intel's target tensors
without another quantization or packing pass.

```bash
./scripts/download_sources.sh /models/qwen38-sources
make build-image
./scripts/assemble_with_docker.sh /models/qwen38-sources upload
```

The assembled HF tree is written to `/models/qwen38-sources/upload`. See
[`docs/reproduce.md`](docs/reproduce.md) for source revisions, validation, and
upload commands.

## Data and caveats

The small JSON summaries are public:

- [`benchmarks/serving-summary.json`](benchmarks/serving-summary.json)
- [`benchmarks/hillclimb.json`](benchmarks/hillclimb.json)
- [`benchmarks/agentbench-summary.json`](benchmarks/agentbench-summary.json)

Private AgentBench fixtures, hidden tests, workspaces, and reasoning traces are
not included. The default uses approximate QSA. Exact QSA is available as a
slower comparison profile.

## License

The runtime code is Apache-2.0. The model keeps the upstream Qwen and third-party
terms listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
