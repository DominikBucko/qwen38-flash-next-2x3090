# Runtime snapshot

This directory is copied from the tagged GitHub release that assembled the
checkpoint. It records the exact vLLM base image, overlay hashes, launch profile,
and compact MTP draft.

Recommended use is to clone the matching GitHub tag, build its Dockerfile, and
mount this model directory read-only. The important defaults are:

```text
BF16 KV; max context 262144; TP2 + EP2; hot cache 88; UVA expert offload;
prefix caching with Mamba-aligned boundaries; MTP depth 3; approximate QSA.
```

Set `VLLM_QSA_EXACT_TOPK=1` only to reproduce the slower exact-QSA experiment.
The target and draft checkpoint provenance live in `hybrid_sources.json` and
`mtp-int4-g32/compact_sources.json` respectively.

The host needs 128 GiB RAM plus at least 32 GiB of fast NVMe swap for loading;
48–64 GiB is recommended. If a 24 GiB GPU runs out of memory on the first
prompt, reduce `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` from 88 to 86, then 84 before
changing the KV allocation. Each removed hot-cache slot saves roughly 116 MiB
per GPU but can reduce decode speed. See the GitHub
[memory guide](https://github.com/DominikBucko/qwen38-flash-next-2x3090/blob/main/docs/memory.md)
for the full OOM decision guide.
