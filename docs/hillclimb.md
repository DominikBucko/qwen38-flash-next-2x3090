# Qwen3.8 Flash Next: the dual RTX 3090 performance hillclimb

[Back to the README](../README.md) · [Performance tuning](performance.md) · [Benchmark evidence](benchmarks.md)

![Measured speed hillclimb](images/hillclimb.svg)

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

### The original 256K boundary probe

The 262,016-input/128-output probe measured 54.5 output tok/s and established
that the 262,144-token boundary is reachable. Its output is too short to
characterize sustained long-context generation or isolate the cost of the
larger KV state. The balanced profile prefills that prompt at 1,275.6 tok/s.
A static expert cache reached
1,629 prompt tok/s, but its decode behavior was worse, so it is not the default.
