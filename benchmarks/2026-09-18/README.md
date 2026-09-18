# Long-context progress — September 18

**Full 256K window: 139.8 seconds to first token, then 86.2 tok/s decode.**
The request contains 260,096 input tokens and 2,048 output tokens. It completes
in 163.6 seconds on two RTX 3090 24 GB cards with 128 GB system memory.

These are experimental runtime measurements, not a runtime release. The default
launcher and published checkpoint are unchanged. The experimental images are
not published; their digests in [summary.json](summary.json) identify the tested
builds for the audit record, not images that users can pull.

## Long prompts: about 35% less waiting

![Time to first token across long-context screens](../../docs/images/prefill-long-context.svg)

| Screen | 131,072 input: first token | 260,096 input: first token | Full-context decode |
|---|---:|---:|---:|
| Before tiered prefill | 104.0 s | 214.5 s | 74.9 tok/s |
| Tiered prefill + kernel warmup | 74.0 s | 146.4 s | 81.1 tok/s |
| Latest, with transfer overlap | 75.9 s | **139.8 s** | **86.2 tok/s** |

Each point has 2,048 output tokens. At full context, the latest screen saves
74.7 seconds before generation starts: 34.8% less wait than the first row.
Its input/TTFT rate is 1,860 tok/s, versus 1,213 in the earlier screen.
At 128K input, the latest screen gives 1,727 input tok/s and 89.1 decode tok/s.

These are observations across development builds, not a repeated controlled
comparison. The prompts match, but preceding requests, host paging and cache
warmth differ. The latest build is not the fastest prefill-only build at every
length. Lines join measured points; they do not predict untested lengths.

## What changed

- **Use one GPU/host expert view for large prefill.** Hot experts stay on the
  GPU; the cold suffix remains in immutable host storage. Large prefills use
  one contiguous logical view instead of the small-decode LRU path. In an
  earlier 128K screen, this cut time to first token from 100.7 to 72.9 seconds
  (27.6%). That comparison also had different preceding warmup requests.
- **Warm the shapes that occur at the context boundary.** Startup covers the
  actual ten-expert routing shapes. A three-query QSA case uses the existing
  four-row padded specialization. In the recorded boundary checks, the final
  three-token pause fell from 4.29 to 0.12 seconds. This moves work into
  startup; it is not free, and total request time did not improve in that pair.
- **Overlap part of a cold-expert transfer with compute.** The latest candidate
  overlaps down-projection weight staging with gate/up compute for a bounded
  small-query shape. It completed the full-context and replay screens above.
  A matched overlap-OFF run was not completed, so the 86.2 tok/s result is not
  an isolated gain attributable to overlap.

The target remains Intel W4A16 group-128, with sensitive layers and KV in BF16,
FP8 PLE, MTP3 using the same INT4 group-32 draft, all ten routed experts per
token, and the same approximate QSA budget. No expert pruning or new target
quantization was used to obtain these results.

## Fresh agent: prefill only the new part

![Fresh-agent prefill and prefix-cache reuse](../../docs/images/prefill-agent-cache.svg)

One completed fresh DBG-06 task on the **tiered + warmup candidate**, before
the transfer-overlap change, made 13 requests as context grew from 9,255 to
30,084 tokens:

- **1,477.7 new-token/s prefill:** 58,191 newly computed tokens / 39.379 seconds.
- **75.0% prefix reuse:** 174,400 of 232,591 submitted prompt tokens were cached.
- **79.4 tok/s decode**, with xhigh reasoning and a 16,384-token response cap.
- The task passed its strict checks. This is one smoke test, not a new suite
  score or evidence of quality parity.

The latest overlap candidate separately reached **85.3 tok/s** on the warmed
six-request pass of frozen agent-history replay. Replay measures latency; it
does not execute new tool-use trajectories. Its fresh-agent run was interrupted
before completion and has no final score or throughput claim.

The old 127–135 tok/s warm-decode peaks used a short, highly repetitive workload.
They are not a baseline for agent throughput. The original hillclimb remains
in the main README with its original shapes.

## Conditions and limits

The long requests use a fixed, non-tiled source-code corpus through the
`repo-chat` client. A unique cache salt prevents prefix-block reuse for each
request. Model weights, expert caches, CUDA graphs and host pages remain warm.
Each long-context row is one valid stream with exact usage counts and a length
finish. Completion counts include reasoning and control tokens.

TTFT includes scheduling, prefill and first-token work. Input/TTFT is therefore
an end-to-end input rate, not a kernel-only prefill measurement. Decode uses
tokens after the first output chunk divided by the remaining output time;
SSE buffering and MTP batches affect this API-observed rate. Fresh-agent prefill
uses observed completed-request prefill time and excludes cache-hit tokens.
Its rate is not the same metric as long-request input/TTFT.

Startup and PLE prefault are outside request timing. Latest startup took about
550 seconds. The preceding candidate's PLE prefault alone took about 77 seconds
and did not make the table fully resident. These are not cold-boot timings or
a claim that swap is absent. See the [memory guide](../../docs/memory.md).

The overlap integration passed 140 changing-route cases on each GPU in
batch-invariant diagnostic mode. That is local correctness evidence, not
whole-model quality parity. Normal-mode greedy variation existed in controls
and candidates. Repeated matched runs, overlap ON/OFF and fresh-agent quality
checks remain necessary before these changes become defaults.

## Data and chart reproduction

[summary.json](summary.json) contains the plotted points, per-request agent
counts, profile, source hashes and limitations. No private fixtures, hidden
tests, generated text or reasoning traces are included.

From the repository root, regenerate the SVGs with the Python standard library:

```bash
python3 scripts/render_prefill_progress.py
```
