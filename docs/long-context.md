# Extended context beyond 262,144 (YaRN)

The checkpoint's trained window is 262,144 tokens and every profile in this
repo serves exactly that. Long-running agent sessions outgrow it, so this is
the recipe for a larger window using YaRN RoPE rescaling — training-free,
opt-in, and inert until you set `HF_OVERRIDES_JSON`.

## What YaRN changes and what it does not

YaRN rescales the rotary position frequencies by interpolating their
wavelengths (with a ramp between `beta_slow` and `beta_fast` that keeps high
frequencies mostly untouched). It changes **position handling only**: weights,
the QSA sparse-attention budget, the PLE n-gram table, MTP and the KV layout
are untouched, so all other guidance in `docs/memory.md` still applies.

At a factor of ~1.7x with the defaults below, YaRN without fine-tuning is
generally near-lossless for retrieval-style tasks, but it is a quality change
like any other: run your own long-context evaluation at the target window
before treating the larger window as production behavior. The repo's harness
in `benchmarks/` is the right place to pin that evidence.

## Wiring

`scripts/serve-container.sh` passes `--hf-overrides` when `HF_OVERRIDES_JSON`
is set, and `scripts/docker_serve.sh` forwards it (plus
`VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, vLLM's opt-in flag for serving beyond the
checkpoint's `max_position_embeddings`). Without the env var, the launcher
behaves exactly as before.

The config block is `configs/yarn-455k.env`. Two details in it are load-bearing:

1. **`--hf-overrides` replaces the `rope_parameters` object wholesale.** Every
   field the checkpoint relies on inside it — `mrope_section`,
   `partial_rotary_factor`, `rope_theta` — must be repeated verbatim from
   `config.json`; omitting one silently resets that field to a library default
   and quietly changes model behavior.
2. **The factor should be derived, not chosen.** 455,000 / 262,144 =
   1.735687255859375 exactly; picking a round factor and a round
   `MAX_MODEL_LEN` that don't correspond leaves either dead window or an
   over-stretched position range.

## Paying for the window

KV bytes grow linearly with the window, so a 1.74x window needs 1.74x the KV
pool: about 7.7 GB across two ranks at 455,000, versus the stock BF16 budget
of 4.43 GB that covers 262,144 exactly. On a 2x RTX 3090 box the options are
a quantized KV cache (`KV_CACHE_DTYPE=fp8_per_token_head`, shipped since
v0.5.0, halves the pool) or more total VRAM with a raised
`KV_CACHE_MEMORY_BYTES`. Everything else — batch chunking, hot cache, PLE —
is unchanged by the wider window.

## Results from a community rig

Measured on **4x RTX 5060 Ti 16 GB (sm_120), TP4 + expert parallel, 128 GB
host RAM** — a different shape than the reference 2x3090 rig, reported as-is:

| quantity | value |
| --- | --- |
| served window | 455,000 (YaRN factor 1.735687255859375) |
| KV cache | per-tensor `fp8` (e4m3), pool sized for the full window |
| decode (reciprocal throughput, MTP off) | 59.7 tok/s |
| prefill (mean over 5 runs, repo repro harness shape) | 2,501.7 tok/s |
| time to first token (3K prompt) | 1.23 s |

This rig has 16 GB cards, not 24 GB, and reaches 455K only because fp8 KV
halves the pool; the byte math above is the transferable part. The lane has
served daily agent traffic at this window since the configuration was
adopted.
