# Mamba state-block fix — September 28

**+7.5–8.2% prefill at 260K and +6% at 131K**, with no KV preemptions. This is a
one-file runtime fix for the [fast 256K profile](../../configs/fast-256k.env).
Decode, 8K prefill and output quality are unchanged.

Hardware: two RTX 3090 24 GB cards and 128 GB system memory. Same profile as the
[September 25 release](../2026-09-25/README.md): hot84, BF16 KV, 4,429,185,024
KV bytes per GPU, 262,144-token context, MTP depth 3, async scheduling on.

## Same-night A/B

Both images were measured the same night, back to back, with the frozen client:
two measured runs per shape after one 4,096+256 warmup, with a unique cache salt
per request.

| Shape | Release image | With the fix | Change |
|---|---:|---:|---:|
| 260,096 + 2,048, input tok/s | 2,630 / 2,642 | **2,828 / 2,859** | +7.5% / +8.2% |
| 260,096 + 2,048, first token | 98.9 / 98.4 s | **92.0 / 91.0 s** | −6.9 / −7.4 s |
| 131,072 + 2,048, input tok/s | 2,747 / 2,748 | **2,912 / 2,906** | +6.0% / +5.8% |
| 131,072 + 2,048, first token | 47.7 / 47.7 s | **45.0 / 45.1 s** | −2.7 / −2.6 s |
| 8,192 + 2,048, input tok/s | 2,683 / 2,685 | 2,684 / 2,688 | none |
| KV preemptions (260K / 131K / 8K) | 10 / 2 / 0 | **0 / 0 / 0** | |

Decode did not change beyond run-to-run noise. The cost of a verify cycle moved
by −0.24, +0.31 and +0.73 ms at 260K, 131K and 8K. Accepted tokens per cycle
moved in both directions, because greedy output is not bit-reproducible on this
runtime and acceptance follows the generated text. Mean decode over the three
shapes was 98.6 tok/s with the fix and 99.1 tok/s without it.

A 256K retrieval check on the fixed image passed 5/5: 255,645-token prompts,
needle depths from 5% to 95%, thinking off. Each took 90.0 s.

Raw reports: [fix-260k](fix-260k.json.gz), [fix-131k](fix-131k.json.gz),
[fix-8k](fix-8k.json.gz), [base-260k](base-260k.json.gz),
[base-131k](base-131k.json.gz) and [base-8k](base-8k.json.gz).
[summary.json](summary.json) has the per-run numbers.

## What was wrong

With `--mamba-cache-mode align`, each Mamba layer group keeps the recurrent
state of the running step in one cache block. When the next step starts, the
previous state block is no longer needed and is freed.

The base image frees KV blocks only up to the tokens that have finished
processing, so that an in-flight step never loses a block it still reads. With
async scheduling one step is always in flight. The Mamba manager, however,
remembers only one pending state block per request. It skips the free while the
previous step is in flight, then records the next pending block over the old
entry. The generic sweep never reaches the orphaned block either, because it
stops at the first empty entry.

As a result, each 4,096-token prefill chunk left one block allocated in each of
the four Mamba groups until the request finished. During a 131K prefill the
Mamba groups held 17 blocks each at 57K tokens instead of 6. On 260K requests
the pool filled about five times. Each time, the scheduler preempted the
request, which then resumed from its own prefix cache.

Synchronous scheduling (the default profile) is not affected.

## The fix

`runtime/vllm-overlay/v1/core/single_type_kv_cache_manager.py` frees every
state block below the latest completed step, rather than only the one it
remembers. It uses the same safety condition as before, now applied to every
pending block. A per-request index means each entry is visited once. A CPU
emulation drove the real block pool and Mamba manager with one step always in
flight, and checked that no block was freed while a step still read it. Peak
live blocks per Mamba group fell from 35 to 6 at 131K, and from 67 to 6 at 260K.

Freed state blocks stay in the prefix cache until they are evicted, so prefix
reuse is unchanged.

## Sizing note

Under async scheduling a prefill step holds 6 state blocks per Mamba group:
three running states (the completed step, the in-flight step and the new step)
plus three speculative blocks. vLLM's static capacity estimate assumes 5. The
released KV size has 195 blocks against the 189 that a 262,144-token request
really needs, so it is unaffected. A pool sized to exactly 1.00x reported
concurrency needs four more blocks (about 90 MB per GPU) to avoid preemptions
near the end of a maximum-length prefill.
