# 64 GB RAM profile and prefill profile — September 30

**3,410 tok/s prefill and 84 tok/s decode on a 131K-token prompt, with 64 GB of system RAM**: the new
[`configs/2x3090-64gb.env`](../../configs/2x3090-64gb.env) profile on two RTX 3090s (issue #27). Prefill is
faster than the 128 GB agent profile (3,029 tok/s) only because this profile has the VRAM for 8,192-token prefill
chunks ([why](#why-prefill-is-faster-than-with-the-128-gb-agent-profile)); decode is about a quarter slower (~110
tok/s there).

The same 8K chunks speed up the 128 GB machines: the new [prefill profile](#prefill-profile-for-128-gb-machines)
reaches 4,191 tok/s on the README's 131K benchmark (31.3 s to first token), with each decode step 10–14% slower.

The profile does not keep a pinned copy of every expert in RAM. Each GPU owns its 88 most-used experts per layer
outright; the other 168 of its half live once in a RAM arena (38 GiB for both GPUs). During decode a CPU thread
pool per GPU computes them while the GPU computes a share over PCIe; during prefill each GPU streams them over
PCIe, once per chunk. The FP8 PLE table is read in place from the checkpoint on NVMe. See
[docs/memory.md](../../docs/memory.md#64-gb-ram-the-2x3090-64gb-profile).

## Setup

| | |
|---|---|
| GPUs | 2× RTX 3090 24 GB, PCIe 4.0 ×16 each |
| CPU | AMD Threadripper PRO 5975WX (32 cores, 4 CCDs, Zen 3) |
| RAM | 8×16 GB DDR4-3200; **64 GiB usable**: the RAM above 64 GiB was locked by a memory balloon, started after both GPUs had populated their expert arenas; server container limit 56 GiB |
| NVMe | Samsung PM981a (PCIe 3.0 ×4) |
| Software | Linux 7.0, NVIDIA driver 595.84, image built with `docker/Dockerfile` from this tree, started with `scripts/docker_serve.sh` |
| Profile | `configs/2x3090-64gb.env`: hot 88 per GPU, BF16 KV 2.5 GB (135,168 tokens), 8,192-token prefill chunks, NCCL all-reduce (no P2P needed) |

Protocol: after a fresh server start with warm kernel caches, four requests run one at a time in this order:
4,096 + 256, 32,768 + 512, 131,072 + 512 and 8,192 + 1,024 tokens (input + output), code-review prompts built from
public runtime sources, greedy, a unique `cache_salt` each, outputs forced to full length (the client of the
single-GPU runtime, `scripts/bench.py` there). Prefill = prompt tokens / time to first token; decode = (output
tokens − 1) / time after the first token. MTP3: 1.8–2.5 of 3 draft tokens accepted per step. No request was
preempted in any run. A smoke test after each run (a reasoning answer, a tool call, code with thinking off)
passed every time.

## Results

32 cores, GPU share 0.3 (the profile default) and 0.2:

| Request (input + output) | First token | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| 131,099 + 512 | 38.4–38.5 s | **3,401–3,413** | **84.2–88.8** |
| 32,799 + 512 | 10.2–10.3 s | 3,200–3,211 | 77.0–83.9 |
| 8,218 + 1,024 | 2.6 s | 3,114–3,146 | 78.2–79.7 |
| 4,127 + 256, first request after the start | 3.6–3.7 s | 1,126–1,145 | 70.9–72.1 |

(three runs: share 0.3 once, share 0.2 twice.) Peaks: GPU memory 23,711–23,713 MiB per card of 24,576, GPUs
70–71 °C, container at its 56 GiB limit (46 GiB of process memory, 38 GiB of it expert arenas; the rest page
cache), CPU Tctl 83–85 °C.

The published image `ghcr.io/dominikbucko/qwen38-flash-next-2x3090:v0.5.0`
(`sha256:75adfc64ba1576b8a677f455d54891c152444e4bf43fbc37cb8f1885a8db3387`, built by CI from the release tag; all
57 installed overlay files match the tag's manifest) ran the same protocol once: 1,128 / 69.7 (4K),
3,232 / 81.0 (32K), 3,396 / 86.0 (131K) and 3,174 / 80.4 (8K) prefill / decode tok/s, prefill within 1% of the runs
in the table. The smoke test passed.

### A 12-core desktop CPU

The container restricted to 12 cores on 2 CCDs (`CPUSET=0-5,8-13,32-37,40-45`), the core layout of a Ryzen 9
9900X. The automatic plan gave each GPU one CCD with a 4-thread pool:

| GPU share | 4K + 256 | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|
| 0.3 (default) | 1,132 / 71.2 | 3,237 / 79.1 | 3,413 / 80.7 | 3,180 / 74.8 |
| 0.2 | 1,157 / 66.8 | 3,223 / 73.2 | 3,410 / 75.4 | 3,213 / 69.3 |

(prefill / decode tok/s). Prefill does not depend on the CPU. With fewer cores the GPU share matters more: 0.3 was
~7% faster than 0.2 here, while on 32 cores the two were equal and 0.4 was slower (67.5 / 83.1 / 79.2 / 74.9 tok/s
decode). These are proxies: the same Zen 3 cores and DDR4 memory. A Zen 5 desktop with DDR5-6400 reads memory
faster per core.

### Not measured here

- **PCIe 4.0 ×8 links** (typical for two GPUs on a desktop board): the benchmark host runs both cards at ×16.
  ×8 halves the bandwidth each GPU streams its cold experts with (~20 GB per 8K chunk and GPU: ~1.5 s at ×8,
  still below the ~2.4 s of GPU work per chunk) and the bandwidth of the GPU share; try
  `QWEN38_GPU_SHARE_FRAC=0.2` there.
- **A real 64 GB machine**: the balloon leaves exactly 64 GiB; a machine with 64 GB installed has ~62.5 GiB of
  MemTotal. `MEMORY_LIMIT=auto` gives the container 56 GiB either way.
- **Quality**: the weights, KV precision (BF16) and expert math are the same as in the 128 GB profiles; the CPU
  and GPU-share expert kernels were checked against FP32 references in the single-GPU runtime. No model-level
  quality run was made for this profile.

## Why prefill is faster than with the 128 GB agent profile

Both profiles stream all of a GPU's cold experts over PCIe once per prefill chunk and run each expert's GEMM once,
whatever the chunk's size: ~20 GB per GPU and chunk here, ~21 GB in the agent profile. A 131K prompt takes 16
chunks of 8,192 tokens or 32 of 4,096. This profile keeps 88 experts per layer on each GPU. The agent profile's
expert cache holds 100 (about 1.4 GiB more per card), and it uses 4,096-token chunks. Changing only the chunk size
(the 64 GB rows and the agent run with 8,192-token chunks on the published v0.5.0 image; the agent range is the
five regression runs below):

| Run | 4K + 256 (first request) | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|
| 64 GB profile, 8,192-token chunks (default) | 1,128 / 69.7 | 3,232 / 81.0 | 3,396 / 86.0 | 3,174 / 80.4 |
| 64 GB profile, `MAX_NUM_BATCHED_TOKENS=4096` | 1,135 / 73.9 | 2,732 / 80.5 | 2,754 / 85.9 | 2,551 / 77.8 |
| Agent 128K profile, 4,096-token chunks (default) | 977–1,025 | 2,531–2,735 | 2,907–2,964 | 2,499–2,530 |
| Agent 128K profile, `MAX_NUM_BATCHED_TOKENS=8192` | out of VRAM | | | |
| Same, with 88 cached experts: the [prefill profile](#prefill-profile-for-128-gb-machines), 2 runs | 1,136–1,170 | 3,196–3,276 | 3,884–3,912 | 3,246–3,251 |

(prefill / decode tok/s; prefill only for the agent rows.) With 4,096-token chunks this profile needs 47.6 s
instead of 38.6 s for the 131K prompt and prefills at or slightly below the agent profile; decode does not change.
The agent profile with 8,192-token chunks ran out of VRAM on its first request (80 MiB requested, 34 MiB free):
its VRAM goes to the larger expert cache, which its decode uses. With 88 cached experts per layer it fits, and it
then prefills faster than this profile. A likely reason, not measured: the 128 GB path copies experts that are
already in the GPU kernel's layout, while this profile converts them on the GPU.

## Prefill profile for 128 GB machines

[`configs/agent-128k-prefill.env`](../../configs/agent-128k-prefill.env) is the agent 128K profile with
`MAX_NUM_BATCHED_TOKENS=8192` and `VLLM_WNA16_STATIC_HOT_CACHE_SIZE=88`: 12 fewer cached experts per layer, about
1.36 GiB per card, pay for the larger chunks. Every run used the published v0.5.0 image and all 128 GB of RAM (no
balloon, no memory limit, and the temporary swap file the 128 GB loader needs).

With the benchmark of this report (prefill tok/s / decode tok/s / ms per verify step):

| Run | 4K + 256 (first request) | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|
| Prefill profile, run 1 | 1,170 / 88.7 / 36.6 | 3,196 / 106.5 / 27.6 | 3,884 / 108.8 / 29.4 | 3,251 / 109.2 / 29.0 |
| Prefill profile, run 2 | 1,136 / 88.1 / 37.9 | 3,276 / 110.7 / 27.8 | 3,912 / 108.7 / 29.0 | 3,246 / 105.8 / 29.2 |
| Agent 128K profile, the 5 regression runs below | 977–1,025 / 87.8–96.3 / 32.7–38.2 | 2,531–2,735 / 115.5–123.7 / 24.4–25.6 | 2,907–2,964 / 119.0–126.5 / 26.2–26.6 | 2,499–2,530 / 122.4–130.2 / 24.8–26.1 |

Prefill rose 20–33% on the 8K–131K prompts: the 131K prompt reached its first token in 33.5–33.7 s instead of
44.2–45.1 s. A verify step cost 10–14% more (the first request 4–8% more), because more experts come from system
memory each step. Draft acceptance stayed in the same range (1.9–2.3 of 3 tokens).

With the README's headline benchmark (`scripts/benchmark_serving.py --prompt-style repo-chat`, 131,072 + 2,048 and
8,192 + 2,048, two runs each, no warmup, the prompt built from this tree), both profiles on the same afternoon:

| Profile | 131K + 2K: first token | Input tok/s | Decode tok/s | 8K + 2K: first token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|---:|---:|---:|
| Prefill profile | 35.98 / **31.27 s** | 3,643 / **4,191** | 91.9 / 101.8 | 2.13 / 2.13 s | 3,851 / 3,844 | 98.9 / 101.1 |
| Agent 128K profile | 47.56 / 43.05 s | 2,756 / 3,045 | 104.8 / **111.5** | 2.90 / 2.90 s | 2,823 / 2,822 | 103.2 / 106.2 |

(run 1 / run 2.) Run 1 of each profile is the first request after the start and was slower in both. The prompt
is built from the overlay sources, which changed in v0.5.0, so it differs from the one behind the September 29
numbers (43.3 s, 3,029 and 109.8 tok/s): compare the two profiles here with each other.

GPU memory peaked at 23,799–24,033 MiB per card of 24,576 (agent profile: 23,987–23,991), CPU Tctl at 84 °C and
the GPUs at 76 °C. No request was preempted, and the smoke test passed after every run. On a card that also drives
a display, lower the cache further (each slot removed saves about 116 MiB per GPU; not tested). The weights, KV
precision and expert math are unchanged. As with any hot-set size, kernel tile choices can depend on the local
expert count, which changes floating-point reduction order; no separate quality run was made.

## Regression check of the 128 GB profiles

The overlay adds 9 files and changes 11. Every new path is behind a switch the 128 GB profiles do not set. To
check that they behave the same, the [agent 128K profile](../../configs/agent-128k.env) ran with the same client
and 128 GB (no balloon, no memory limit) on this tree's image, the published v0.4.0 image and a local build of the
v0.4.0 tree (prefill tok/s / decode tok/s / ms per verify step):

| Image | 4K + 256 (first request) | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|
| new image, run 1 | 977 / 90.2 / 35.6 | 2,735 / 116.3 / 24.4 | 2,964 / 119.0 / 26.6 | 2,512 / 122.4 / 25.7 |
| new image, run 2 | 1,012 / 87.8 / 38.2 | 2,531 / 120.8 / 25.5 | 2,907 / 119.1 / 26.3 | 2,530 / 122.5 / 26.1 |
| v0.4.0 GHCR image, run 1 | 993 / 96.3 / 35.3 | 2,649 / 123.7 / 24.7 | 2,943 / 122.0 / 26.5 | 2,530 / 130.2 / 25.2 |
| v0.4.0 GHCR image, run 2 | 1,023 / 95.3 / 34.3 | 2,728 / 115.5 / 25.6 | 2,956 / 126.5 / 26.2 | 2,521 / 123.4 / 26.1 |
| v0.4.0 tree, built locally | 1,025 / 95.3 / 32.7 | 2,722 / 123.7 / 24.6 | 2,928 / 123.1 / 26.2 | 2,499 / 129.5 / 24.8 |

Prefill is unchanged. The cost of a verify step is the same after the first request: 24.4–25.6, 26.2–26.6 and
24.8–26.1 ms at 32K, 131K and 8K for both. The decode rate follows how many draft tokens greedy decoding
accepts, which varies between runs (for example 116.3 tok/s at 1.84 accepted tokens vs 123.7 at 2.05). The first
request after the start was slower per step with the new image in both runs (35.6 and 38.2 vs 32.7–35.3 ms). It
overlaps the background PLE prefault of the 128 GB profiles; with two runs each this is not separable from
startup noise. The new image's startups took 607 s (the first, including the first-time compile of the changed
kernels) and 456 s; v0.4.0's took 457–459 s.

## Files

- [`summary.json`](summary.json): every run above.
- [`raw/`](raw/): the benchmark client output of each run; `raw/headline-*.json.gz` are the
  `scripts/benchmark_serving.py` reports of the headline benchmark.
