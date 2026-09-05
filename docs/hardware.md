# Qwen3.8 Flash Next on dual RTX 3090 and RTX 4090: hardware guide

## Tested hardware contract

The released serving profile was validated on one Linux x86_64 host with:

| Resource | Validated configuration |
|---|---|
| GPUs | 2× NVIDIA GeForce RTX 3090, 24 GB each |
| System memory | 128 GB |
| Swap | At least 32 GiB on fast NVMe; 48–64 GiB recommended |
| Runtime | Docker with NVIDIA Container Toolkit |
| Workload | One model instance, one request, up to 262,144 total tokens |

This is the compatibility target recorded in
[`repro.lock.json`](../repro.lock.json). Hardware outside that target can be a
useful experiment, but this repository does not claim that it will start, fit,
or match the published performance.

### September 5 benchmark host

The new public probes record more detail than the original hardware summary:

| Component | Observed configuration |
|---|---|
| CPU | AMD Threadripper PRO 5975WX, 32 cores / 64 threads |
| RAM | 8×16 GB DDR4-3200, eight populated memory channels |
| GPUs | 2× RTX 3090; 300 W power limit per card |
| PCIe | Gen 4 ×16 on both cards under load; PHB topology, no NVLink |
| Kernel / driver | Linux 7.0.0-30-generic / NVIDIA 595.84 |
| CUDA P2P | Read/write topology checks and PyTorch peer capability passed in both directions |
| Runtime boundary | Isolated clean vendor vLLM + the 27-file public overlay, using existing native dependencies; not a fresh Docker build |

These details describe this host, not a minimum CPU specification or a promise
that every stock 3090 driver provides P2P. The public Docker quickstart remains
the supported reproduction path. The native experiment records its environment
and overrides separately from the original release.

The measured candidate used an 84-expert hot cache, custom all-reduce enabled,
and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False`. It completed three
exact-count `repo-chat` requests of 258,048 input + 4,096 output at a 75.636
tok/s reciprocal-mean aggregate. Four recoverable allocator warnings appeared
during the first long prefill, with no stream failures. The released Docker
defaults remain an 88-expert cache, custom all-reduce disabled, and expandable
segments enabled. See the [September 5 benchmark bundle](../benchmarks/2026-09-05/README.md)
and [machine-readable summary](../benchmarks/2026-09-05/summary.json).

All 27 model weight files on this host matched the published SHA-256 manifest
for canonical tensor revision `ef554143369a706525336f6b42a09094835dc077`.

## Why this fits when the standard vLLM checkpoint does not

The [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
lists 172.78 GiB for the stock FP8 checkpoint and 335.28 GiB for BF16. Its
PLE-only CPU offload and larger-GPU deployments use different checkpoint
formats and memory placement. Those estimates do not describe this runtime.

This is a custom checkpoint and runtime combination. The assembled target is
116.183 GiB of Intel AutoRound W4A16 weights and FP8 PLE, plus a separate
3.855 GiB compact MTP draft. The checked-in overlay keeps PLE and the complete
routed-expert pool in system memory while caching hot experts on the GPUs.
The original BF16 or FP8 checkpoint and stock vLLM flags are different memory
configurations; they do not reproduce this setup. See the exact
[checkpoint formats and runtime pins](../repro.lock.json).

## Why 48 GB of VRAM is not one 48 GB GPU

The two 3090s provide 48 GB in aggregate, but CUDA still exposes two separate
24 GB address spaces. Tensor parallelism splits one model instance across both
cards; it does not pool them into a general-purpose 48 GB allocation. The
released profile also creates a BF16 KV allocation of about 4.13 GiB on each
GPU and keeps 88 hot experts per layer on GPU.

An allocation that must fit on one rank is still limited by that rank's free
VRAM. Both cards therefore need adequate headroom. One card cannot substitute
for the validated pair merely because system RAM can hold offloaded tensors.
See [Memory sizing and OOM recovery](memory.md) for the separate VRAM, RAM, and
swap budgets.

## Is NVLink required?

No. The topology used for the historical release results did not provide usable
CUDA peer access, yet it produced those results. The default launcher passes
`--disable-custom-all-reduce` and uses the selected TP2/EP2 collective path.
The September 5 PHB host also had no NVLink, but CUDA P2P worked in both
directions; the native candidate therefore enabled custom all-reduce with
expandable segments disabled and completed the full model probes. Neither
profile required NVLink.

Do not use a slot label or advertised PCIe width as proof that CUDA peer access
works. Motherboard routing, IOMMU configuration, and the GPU/driver topology can
change the usable path. The published numbers describe the tested topology,
not every two-slot system.

PCIe still matters. Cold routed experts remain in host-backed memory, and data
must cross the host/GPU boundary when the hot cache misses. Poor link placement
or contention can reduce decode throughput even though NVLink is unnecessary.

### Check CUDA peer access on your own host

Host diagnostics:

```bash
nvidia-smi topo -p2p r
nvidia-smi topo -p2p w
```

After building the image, check what CUDA exposes to PyTorch:

```bash
docker run --rm --gpus all --entrypoint python \
  qwen38-flash-next-2x3090:locked -c \
  'import torch; print(torch.cuda.can_device_access_peer(0, 1), torch.cuda.can_device_access_peer(1, 0))'
```

Both directions must work before trying `DISABLE_CUSTOM_ALL_REDUCE=0` in `.env`.
The pinned custom path additionally needs an IPC-compatible allocator; its
CUDA graph registration fails with the default expandable-segment allocator.
That flag controls vLLM's custom collective, not driver-level peer access.
See the [allocator settings and test protocol](performance.md#cuda-p2p-and-custom-all-reduce).

## Why system memory bandwidth matters

This runtime intentionally uses the host as an active memory tier:

- a dedicated CPU process owns the large FP8 PLE/ngram table;
- the complete routed-expert pool remains addressable in pinned host memory;
- only the current hot expert set is resident on the GPUs.

PLE lookups and cold-expert accesses are on the token path. CPU memory latency,
memory-channel population, NUMA placement, and sustained host bandwidth can
therefore affect decode. Fast NVMe swap helps the model survive peak loading,
but active paging during generation is a performance failure. Watch `vmstat 1`;
persistent swap-in or swap-out means the run is no longer comparable with the
published measurements.

## Can I run Qwen3.8 Flash Next on dual RTX 4090?

Two RTX 4090 cards have not been validated by this project. Their 24 GB per-card
capacity does not by itself prove compatibility with the pinned kernels,
container, collectives, topology, or full-context memory budget. No 4090
throughput number or support promise should be inferred from the 3090 results.

If testing 4090s, start with the unchanged released profile, retain the pinned
checkpoint and image, run the full preflight, and treat the result as new
hardware evidence. A successful short prompt is not enough to claim 262K
support; the boundary test is 262,016 input tokens plus 128 output tokens.

## Host and container responsibilities

The host supplies Linux x86_64, the NVIDIA driver, Docker, the NVIDIA Container
Toolkit, physical RAM, swap, and access to both GPUs. `make preflight` checks
the operating system, architecture, Docker daemon, visible GPU count, host
memory, swap, and the PLE IPC-related ptrace setting.

The image supplies the pinned user-space runtime. Its
[`Dockerfile`](../docker/Dockerfile) starts from a digest-pinned Qwen3.8 vLLM
image, installs `humming-kernels==0.1.12`, applies the checksummed vLLM overlay,
and copies the released configuration and hot-cache rankings. Installing a
different host CUDA toolkit does not replace those container components.

The checkpoint stays outside the image and is mounted read-only at `/model`.
The host NVIDIA driver must be compatible with the container's CUDA user-space
stack, but this repository does not declare an unverified minimum driver
version. Keep the pinned image boundary intact when reproducing results.

## Text-only serving is the default

The launcher passes `--language-model-only`. The published configuration and
benchmarks cover text requests through the OpenAI-compatible endpoint. They do
not validate image, audio, or video inputs, even if upstream model or vLLM code
contains multimodal interfaces.

## Setup FAQ

### What should I run before the first launch?

Set `MODEL_DIR` to the downloaded checkpoint, configure NVMe-backed swap, then
run:

```bash
make preflight
make serve
```

The checkpoint must include its compact MTP model under
`runtime/mtp-int4-g32`. The complete setup is in the [README](../README.md),
and immutable source/runtime pins are in the
[reproduction guide](reproduce.md).

### Can I use one 3090?

The checked-in launcher requires two visible CUDA GPUs and hard-codes tensor
parallel size 2. A single-GPU configuration is unsupported and was not tested.

### Can I use 64 GB of system RAM?

No supported 64 GB profile is provided. The tested minimum is 128 GB plus swap.
Swap is load-time headroom, not a substitute for enough resident memory during
serving.

### Why does startup look stuck?

The CPU PLE worker loads a large part of the checkpoint and registers shared
buffers. The released readiness timeout is 1,200 seconds. Check logs, resident
memory, and disk activity before assuming a deadlock.

### What should I change after an OOM?

First identify the memory pool. A kernel OOM while loading points to host RAM or
swap; a `torch.OutOfMemoryError` points to VRAM. For VRAM, lower the hot cache
from 88 to 86, then 84. If necessary, reduce the explicit KV allocation to
4,294,967,296 bytes and confirm the startup log still reports the context
capacity you need. The exact order and tradeoffs are in
[Memory sizing and OOM recovery](memory.md).

## Share a reproduction or a new GPU result

Use the [hardware report form](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=hardware-report.yml)
to share your setup, exact workload, settings, measurements, and failures.
Independent 3090 runs and exploratory 4090 results help improve this guide.

## Related documentation

- [README and quick start](../README.md)
- [Memory sizing and OOM recovery](memory.md)
- [Benchmark evidence](benchmarks.md)
- [Performance tuning and measurement](performance.md)
