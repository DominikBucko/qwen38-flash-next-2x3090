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
