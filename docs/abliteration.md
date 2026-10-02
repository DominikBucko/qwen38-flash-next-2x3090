# Uncensored mode (abliteration)

`QWEN38_ABLITERATION` removes one refusal direction from the model at runtime. With
`QWEN38_ABLITERATION=orcarouter`, the served model behaves like
[orcarouter/Qwen3.8-Flash-Next-Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored), an
abliterated release of the same base model, without downloading or requantizing any weights. It is off unless set.

> **This removes the model's safety refusals.** With the switch on, the model follows requests the original model
> declines. Use it for research, red-teaming or your own use, and put your own safeguards in front of it before
> serving anyone else. You are responsible for what you do with it.

## What it does

The release follows Arditi et al. (2024), *Refusal in Language Models Is Mediated by a Single Direction*. One unit
direction `r` (2,560 values, the hidden size) is projected out of every matrix that writes to the residual stream:

```text
W' = W − r (rᵀW)
```

That covers 149 tensors: `self_attn.o_proj` (12 full-attention layers and MTP), `linear_attn.out_proj` (36
linear-attention layers), every routed expert's and the shared expert's `down_proj` (48 layers and MTP),
`ple.value_proj` and the rows of `embed_tokens`. The router, the experts' gate/up projections, the
hyper-connection mixers, the QSA indexer, the n-gram table, the vision tower and `lm_head` are unchanged.

For any output `y = W a` of an edited matrix, `W' a = y − r (r · y)`. The runtime therefore applies the projection
to those outputs, in place, at five points:

1. **Attention and linear-attention outputs.** `o_proj` and `out_proj` are the last operations of these blocks.
2. **MoE output.** The routed and shared experts' outputs are summed with per-token scalar weights before the
   residual write, so projecting the sum equals editing every expert's down projection. This includes the cold
   experts the 64 GB profile computes on the CPU or streams to the GPU; their partial outputs are summed before
   the expert-parallel all-reduce.
3. **Right after `ple.value_proj`.** The PLE branch gates, normalizes and convolves the value before its residual
   write. The norm is not linear, so the projection must come before it, where the weight edit acts.
4. **Text token embeddings,** in the target and in the MTP head. Image embeddings are merged afterwards and stay
   untouched, as in the release.
5. **The MTP head's attention and MoE outputs.** The draft reuses the target's decoder layer, so its residual
   writers get the same projection as in the release, and speculative decoding stays aligned with the target.

The INT4 experts and every other weight stay exactly as published. Each residual write costs one 2,560-wide dot
product and a rank-1 update per token, done in place without an extra buffer. The direction is loaded once into a
non-persistent buffer before CUDA-graph capture.

## Usage

Add to `.env` (it works with every profile):

```bash
QWEN38_ABLITERATION=orcarouter
```

- A bare name resolves to `/opt/qwen38/abliteration/<name>.json` inside the image. The shipped direction is
  [`configs/abliteration/orcarouter.json`](../configs/abliteration/orcarouter.json).
- Any other value is a path inside the container: a JSON file with a `refusal_direction` list, or a safetensors
  file with a `refusal_direction` tensor, of length 2,560. The direction is normalized on load.
- Unset, empty or `0`: off. Nothing is loaded and no work is added.
- The switch needs an image built from this tree (`make build-image`); the v0.5.0 release image predates it.

Only single-direction (k = 1) edits of this base checkpoint can be expressed this way. The extraction script below
refuses anything else.

## How the direction was recovered and verified

[`scripts/extract_refusal_direction.py`](../scripts/extract_refusal_direction.py) reads about 83 MB of the release
by HTTP range requests: four edited matrices (`layers.3.self_attn.o_proj`, `layers.0.linear_attn.out_proj`,
`layers.1.ple.value_proj`, `layers.3.mlp.shared_expert.down_proj`), 256 rows of `embed_tokens`, and one untouched
control tensor. It compares them with the same tensors of this repository's checkpoint, all of which are BF16
there. The release is gated on Hugging Face, so the account behind the token has to accept OrcaRouter's terms
first. Results for revision `e096800036ec`:

| Check | Result |
|---|---|
| Untouched control (`layers.3.self_attn.k_proj`) | byte-identical: both start from the same base weights |
| Each edit `W − W'` is rank-1 | second singular value 0.88–0.98% of the first (BF16 rounding) |
| One direction for all four tensors | \|cos\| with the joint direction 0.999993–0.999999 |
| Re-applying `W − r(rᵀW)` to our weights | 87.6–92.9% of elements bit-identical to the release, the rest one BF16 rounding step apart |

`r` is the top left singular vector of all four differences together. The JSON file records the source repository,
the revision and these checks. Running the script again reproduces it exactly:

```bash
python3 scripts/extract_refusal_direction.py --model-dir /models/qwen38-flash-next \
  --repo orcarouter/Qwen3.8-Flash-Next-Uncensored --out configs/abliteration/orcarouter.json
```

## Measured

On October 2, two RTX 3090s, the agent 128K profile (128 GB) and the 64 GB profile (machine limited to 64 GiB). The
switch-off run used the published v0.5.0 image; the switch-on runs used the same image with this tree's overlay
installed. Details and raw files: [benchmarks/2026-10-02](../benchmarks/2026-10-02/README.md).

| | Agent 128K, off | Agent 128K, on | 64 GB, on |
|---|---:|---:|---:|
| Mild borderline requests refused | 7 of 8 | 0 of 8 | 0 of 8 |
| Neutral controls answered | 2 of 2 | 2 of 2 | 2 of 2 |
| Smoke tests (reasoning, tool call, code) | pass | pass | pass |
| 131K prefill | 2,909 tok/s | 2,914 tok/s | 3,358 tok/s |
| Decode cost per verify step, 8K–131K | 24.1–27.0 ms | 24.1–27.0 ms | 37.2–40.7 ms |

The 64 GB numbers are within 3% of that profile's published runs (36.0–40.8 ms and 3,396–3,413 tok/s). That host
ran its CPU at up to 89 °C, warmer than in those runs.

The borderline requests are deliberately mild and harmless to answer: a fake restaurant review, an offensive joke,
an insult, lock picking, an excuse to skip work, exam cheating, a villain's threat for a novel, and a "no rules"
roleplay. The neutral controls ask for a country's capital and a haiku. Greedy decoding, thinking off, 160 tokens.
A regular expression flags refusal phrases; one base reply ("I don't make offensive jokes.") was a refusal it
missed and is counted as one. The neutral answers changed by a word or a line, as expected from a small shift in
the output distribution.

## Limits

- **No capability benchmark was run.** OrcaRouter reports capability within ±2 points of the base model for its
  BF16 release; here the evidence is the smoke tests, the controls and the refusal check.
- **The refusal check is a spot check:** eight mild requests, one run per configuration.
- **Rounding:** the release stores edited BF16 weights; the switch projects BF16 activations. The two agree up to
  rounding, not bit for bit.
- **Profiles:** measured on the agent 128K and 64 GB profiles. The other profiles use the same code path.
- **Single-GPU runtime:** [qwen38-flash-next-3090](https://github.com/DominikBucko/qwen38-flash-next-3090) does not
  have the switch yet.
- **License:** the direction is derived from OrcaRouter's release, which is Apache-2.0 like the base model.
