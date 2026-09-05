# Qwen3.8-Flash-Next on dual RTX 3090 (2×24 GB)

Run **Qwen 3.8 Flash Next locally on two RTX 3090 GPUs and 128 GB RAM** with
CPU offloading, a pinned vLLM runtime, and a downloadable W4A16 + FP8 PLE
checkpoint. The tested profile serves one **262,144-token (256K) context**
across both cards and exposes an OpenAI-compatible API.

**75.6 API-observed output tok/s near full context: 258,048 input + 4,096 output,
across three exact-count runs.** Historical comparison points include 135.2
output tok/s on one warmed 128 + 4,096 run and 1,402 prompt tok/s at 65K input.
These are different workloads and runtime profiles, not a promise of the same
speed at every context length. [Measured shapes and caveats](#results).

[Quickstart](#run-it) ·
[Download weights](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE) ·
[3090 / 4090 hardware guide](docs/hardware.md) ·
[Performance tuning](docs/performance.md) ·
[OOM help](docs/memory.md) ·
[Share your results](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=hardware-report.yml)

## Is this the setup for you?

| Question | Answer |
|---|---|
| Which model? | The open-weight **Qwen3.8-Flash-Next** checkpoint. Use the exact model name when downloading. |
| Validated GPUs | **2× NVIDIA RTX 3090, 24 GB each**; one model split across both GPUs. |
| RAM and swap | **128 GB system RAM**, plus at least **32 GiB NVMe-backed swap** for loading; 48–64 GiB recommended. |
| Storage | About **120 GiB of model tensor payload**, plus space for the container, download cache, and swap. |
| Software | **Linux x86_64**, Docker, NVIDIA Container Toolkit, and the Hugging Face `hf` CLI. |
| Dual RTX 4090? | **Not yet validated.** See the [2×4090 compatibility and testing guide](docs/hardware.md#can-i-run-qwen38-flash-next-on-dual-rtx-4090). No 4090 speed claim is published here. |
| NVLink required? | **No.** Historical results used a topology without usable P2P; the September 5 PHB host had bidirectional CUDA P2P and the candidate enabled custom all-reduce. See [GPU interconnects](docs/hardware.md). |
| Vision or many simultaneous users? | The default is **text-only, one active request**. [Hardware FAQ](docs/hardware.md) · [two-client memory sizing](docs/memory.md#two-concurrent-requests). |

The model does not fit entirely in 48 GB VRAM. This runtime keeps the complete
expert pool and FP8 n-gram/PLE table in system memory, caches active experts on
the GPUs with an LRU policy, and uses a compact MTP3 speculative draft. The
checkpoint preserves Intel's AutoRound target packing and sensitive BF16 layers.
See the [architecture](docs/architecture.md) and [immutable version lock](repro.lock.json).

<a id="run-it"></a>
## Quickstart: run Qwen3.8 Flash on two RTX 3090s

Check the [hardware requirements](docs/hardware.md) and configure swap before
loading. Installation guides: [Docker](https://docs.docker.com/engine/install/),
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
and [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli).
The default profile is close to the VRAM limit; keep the
[OOM recovery guide](docs/memory.md) handy for machines with less free memory.

Clone the runtime and download the pinned checkpoint into a directory you own:

```bash
git clone https://github.com/DominikBucko/qwen38-flash-next-2x3090.git
cd qwen38-flash-next-2x3090
make preflight

QWEN_MODEL_DIR="$HOME/models/qwen38-flash-next"
mkdir -p "$QWEN_MODEL_DIR"

hf download albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE \
  --revision ef554143369a706525336f6b42a09094835dc077 \
  --local-dir "$QWEN_MODEL_DIR"

cp .env.example .env
```

Start from the same terminal so `QWEN_MODEL_DIR` still points to the download:

```bash
make MODEL_DIR="$QWEN_MODEL_DIR" serve
```

For future launches, set `MODEL_DIR` in `.env` to the expanded absolute download
path; then `make serve` works without the command-line override.

`make serve` builds the digest-pinned image and installs this repository's vLLM
overlay before launching. Stock vLLM alone does not include these patches.
Initial model loading can take several minutes; wait until the API is ready.

### Send your first request

In a second terminal, check the model and send a streaming chat request:

```bash
curl -fsS http://127.0.0.1:8000/v1/models

curl -fsS -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": "Write a short Python function that checks whether a number is prime."}],
    "max_tokens": 512,
    "stream": true
  }'
```

For an OpenAI-compatible client, use base URL `http://127.0.0.1:8000/v1` and
model `Qwen3.8-Flash-Next`. Match the port to `PORT` in `.env` if you change it.
The Docker launcher binds to localhost by default.

### If it runs out of memory

Check `free -h`, `swapon --show`, and `nvidia-smi`. For a **CUDA OOM**, edit
`.env` and make one change at a time:

1. Lower `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` from `88` to `86`, then `84` if
   needed. Each slot saves roughly 116 MiB per GPU across the 48 layers, with
   more expert traffic from system memory.
2. Lower `KV_CACHE_MEMORY_BYTES` from `4429185024` to `4294967296`. Keep this
   only if the startup log still reports at least 262,144 KV-cache tokens.
3. Lower `MAX_NUM_BATCHED_TOKENS` from `4096` to `2048` to reduce peak prefill
   temporary memory, at the cost of prefill throughput.

Restart the server after editing. A host OOM while loading needs RAM/swap
headroom instead. Lowering `MAX_MODEL_LEN` alone does not release the explicitly
reserved KV allocation. See [memory sizing and OOM recovery](docs/memory.md).

## Results

### September 5 verified public probes

| Workload | Shape and run policy | API-observed output speed | TTFT |
|---|---|---:|---:|
| Near-full-context chat | 258,048 input + 4,096 output; 3 measured, no explicit warmup | **75.6 tok/s aggregate** (74.0–76.7) | 211.1–215.1 s |
| Short chat | 128 input + 4,096 output; 1 warmup + 3 measured | **77.3 tok/s aggregate** (74.7–79.7) | 0.897–0.903 s |

The aggregate is the reciprocal mean time per output token, rather than the
arithmetic mean of per-run rates. Both probes used the public `repo-chat`
recipe and exact API token counts on 2× RTX 3090. The native candidate used an
84-expert hot cache, bidirectional CUDA P2P, custom all-reduce enabled, and
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. It was a clean pinned
vendor vLLM plus the public overlay using existing native dependencies, not a
fresh Docker build. The released Docker defaults remain 88 experts, custom
all-reduce disabled, and expandable segments enabled.

All 27 weight files matched the published SHA-256 manifest for canonical tensor
revision `ef554143369a706525336f6b42a09094835dc077`. Four recoverable allocator
warnings appeared during the first candidate long prefill; all three streams
still completed with exact usage counts. Completion counts include reasoning
and control tokens. Because the client forces 4,096 tokens, captured chat can
end during reasoning; these probes measure serving behavior, not answer quality.

[September 5 benchmark bundle](benchmarks/2026-09-05/README.md) ·
[Machine-readable summary](benchmarks/2026-09-05/summary.json)

![Three exact-count near-full-context decode runs on the September 5 candidate profile](docs/images/long-context-decode.svg)

### Historical release and hillclimb results

| Workload | Shape | Speed |
|---|---:|---:|
| Prefill | 65,536 input tokens | **1,402 prompt tok/s** |
| Full-context prefill | 262,016 input + 128 output | **1,275.6 prompt tok/s** |
| Warm decode | 128 input + 4,096 output | **135.2 output tok/s** |
| Warm decode, repeated | 128 input + 4,096 output | **127.1–134.0 output tok/s** |
| Full-context boundary probe (short output) | 262,016 input + 128 output | **54.5 output tok/s** |
| Longest tested sequence | Input + output | **262,144 tokens** |

These are **historical single-request measurements on dual RTX 3090s**. Prefill and decode
use different test shapes; 1,402 and 135.2 tok/s did not come from the same
request. The 135.2 tok/s result is a warmed long-generation probe, not expected
speed at every context length. CPU, memory, prompt mix, and MTP acceptance also
matter. The 128-output boundary probe verifies that the context limit is reachable;
it is too short to characterize sustained long-context generation. This is not
a matched comparison against other repositories or GPUs.

Historical sources: [serving measurements](benchmarks/serving-summary.json),
[hillclimb data](benchmarks/hillclimb.json), and
[benchmark methodology and limitations](docs/benchmarks.md).
You can now [collect your own token-exact streaming measurements](docs/performance.md#collect-reproducible-measurements-on-your-machine) with the public benchmark client.

![Qwen3.8 Flash Next dual RTX 3090 benchmark: matched short decode and separate warm long-decode results](docs/images/hillclimb.svg)

## Speed hillclimb

On the matched 128-input/256-output workload, decode improved from **32.83 to
80.08 tok/s (2.44×)**. Expert placement, pinned host memory, dynamic LRU,
fused QSA, and the Humming/Marlin backend split all contributed. The chart's
separate gold series uses 4,096 output tokens and measures the MTP scheduler path.
The 80.08 endpoint used 104 hot-cache slots; the released 256K profile uses 88
to leave room for the longer context.

Read the [step-by-step optimization history](docs/hillclimb.md) or the
[guide to getting better performance on two 24 GB GPUs](docs/performance.md).

## Checkpoint layout

| Part | Storage format |
|---|---|
| Target backbone | Intel AutoRound W4A16, symmetric INT4 group-128 |
| Sensitive target layers | Original BF16 |
| 51.2B-parameter PLE table | FP8 E4M3FN with the published scale |
| MTP routed experts | Symmetric INT4 group-32 |
| Other MTP tensors | Original source precision |
| KV cache | BF16 |

The target payload is 116.183 GiB; the compact MTP draft adds 3.855 GiB. The PLE
table and preserved BF16 tensors explain why this is larger than a conventional
all-INT4 checkpoint. Model weights are hosted on
[Hugging Face](https://huggingface.co/albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE).

## Rebuild the checkpoint

The build scripts pin upstream commits and copy Intel's target tensors without
another quantization or packing pass:

```bash
./scripts/download_sources.sh /models/qwen38-sources
make build-image
./scripts/assemble_with_docker.sh /models/qwen38-sources upload
```

The assembled tree is written to `/models/qwen38-sources/upload`. See
[reproduction instructions](docs/reproduce.md) for source revisions and validation.

## Questions, hardware reports, and contributions

If this setup helps, **star the repository** so you can find it again. To follow
updates, use GitHub's **Watch → Custom → Releases**.

- **Got it running?** [Share your hardware and results](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=hardware-report.yml), including 3090 reproductions and experimental 4090 runs. Failed attempts are useful too.
- **Need setup help?** [Ask in Discussions](https://github.com/DominikBucko/qwen38-flash-next-2x3090/discussions). For a crash, [file a bug report](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=bug-report.yml).
- **Improved performance?** Include the before/after request shapes, settings, and measurements. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Data and caveats

[September 5 probes](benchmarks/2026-09-05/README.md),
[serving](benchmarks/serving-summary.json), [hillclimb](benchmarks/hillclimb.json),
and [AgentBench](benchmarks/agentbench-summary.json) summaries are public.
Private AgentBench fixtures, hidden tests, workspaces, and reasoning traces are
not included. The default uses approximate QSA; exact QSA is a slower comparison
profile. The available quality evidence is limited and does not establish
quality equivalence to the original model. See [benchmark evidence](docs/benchmarks.md).

## License

Runtime code: Apache-2.0. Model weights retain the upstream Qwen and third-party
terms listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
