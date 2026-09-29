# Agent 128K profile — September 29

**~110 tok/s decode and ~3,000 tok/s prefill on a 131,072-token prompt**, with
the [agent 128K profile](../../configs/agent-128k.env). On the 7-prompt decode
set, a verify cycle costs 9% less than with the fast 256K profile.

The profile halves the context window. The KV memory this frees holds 16 more
hot experts per GPU (hot100 instead of hot84), so fewer experts come from system
memory. Everything else matches the fast 256K profile. Weights and precision are
unchanged.

Hardware: two RTX 3090 24 GB cards and 128 GB system memory. The image was built
with `docker/Dockerfile` from this tree and launched with `scripts/docker_serve.sh`.
GPU memory in use was 23,949 and 23,945 MiB after the runs.

## Single request, 131,072 + 2,048

| Profile | First token | Input tok/s | Decode tok/s |
|---|---:|---:|---:|
| Agent 128K, run 1 | 44.1 s | 2,974 | 108.5 |
| Agent 128K, run 2 | **43.3 s** | **3,029** | **109.8** |
| Fast 256K with the [state-block fix](../2026-09-28/README.md), 3 runs | 45.0–46.0 s | 2,852–2,916 | 98.9–101.9 |

At 8,192 + 2,048 the agent profile decoded at 110.3 and 106.3 tok/s. There
were no preemptions.

## Decode cost, 7 prompts

The 7-prompt set is four 8K, two 32K and one 131K request, each producing 2,048
greedy tokens. Totals come from the server counters.

| Profile | ms per verify cycle | Tokens per cycle | Decode tok/s |
|---|---:|---:|---:|
| Fast 256K (hot84), two runs | 30.23–30.34 | 3.167–3.178 | 104.4–105.1 |
| **Agent 128K (hot100)** | **27.41** | 3.181 | **116.1** |

Acceptance is unchanged. The saving is expert traffic: 16 more experts per layer
stay on each GPU instead of being copied from system memory during verify.

## Prefix reuse

The repository's cache checklist was run on a 32,794-token prompt with 256
generated tokens:

| Request | Cached prompt tokens | Time |
|---|---:|---:|
| First | 0 | 13.6 s |
| Same prefix again | 28,800 | 4.8 s |
| Next turn (prefix + reply + new question) | 30,400 | 3.8 s |
| No-cache control | 0 | 13.1 s |

The cached rerun matched the no-cache control for the first 32 greedy tokens, as
closely as two no-cache runs match each other on this runtime (33 tokens).

## Limits

- A request can use up to 135,168 tokens in total: for example, 131,072 prompt
  tokens and 4,096 generated tokens.
- The KV pool has 110 blocks, sized for one full-length request plus the extra
  recurrent-state block per Mamba group that async prefill holds. Without the
  September 28 state-block fix, this profile is preempted about ten times on a
  131K prompt.
- The hot-set size changes where expert weights live, not the math. Kernel tile
  choices can depend on the local expert count. That changes floating-point
  reduction order, as run-to-run nondeterminism already does. No separate
  quality evaluation was run.
- GPU memory is nearly full (23.9 of 24 GiB in use per card). On a card that
  also drives a display, use `VLLM_WNA16_STATIC_HOT_CACHE_SIZE=96`, which frees
  about 460 MiB.
- Host RAM conditions apply as for the fast 256K profile.

Raw reports: [agent-131k](agent-131k.json.gz) and [agent-8k](agent-8k.json.gz).
[summary.json](summary.json) has the per-run numbers.
