# Runtime abliteration — October 2

`QWEN38_ABLITERATION=orcarouter` ([docs/abliteration.md](../../docs/abliteration.md)) on two RTX 3090s. The
switch-off run used the published v0.5.0 image
(`ghcr.io/dominikbucko/qwen38-flash-next-2x3090@sha256:75adfc64ba1576b8a677f455d54891c152444e4bf43fbc37cb8f1885a8db3387`);
the switch-on runs used that image with this tree's overlay installed by `runtime/install_overlay.py` (only Python
files and the direction changed).

## Setup

| | |
|---|---|
| GPUs | 2× RTX 3090 24 GB, PCIe 4.0 ×16 each |
| CPU | AMD Threadripper PRO 5975WX (32 cores) |
| RAM | 8×16 GB DDR4-3200. Agent 128K profile: all 128 GB. 64 GB profile: limited to 64 GiB by a memory balloon, container limit 56 GiB |
| Profiles | [`configs/agent-128k.env`](../../configs/agent-128k.env), [`configs/2x3090-64gb.env`](../../configs/2x3090-64gb.env) |

After a fresh server start: the refusal check, then four requests one at a time (4,096 + 256, 32,768 + 512,
131,072 + 512 and 8,192 + 1,024 tokens, greedy, the client of the single-GPU runtime as in
[the September 30 report](../2026-09-30/README.md)), then the smoke test.

## Refusal check

Eight mild requests that aligned models often decline but that are harmless to answer, and two neutral controls
(prompts in [`raw/`](raw/)). Greedy decoding, thinking off, 160 output tokens. A regular expression flags refusal
phrases. One base reply ("I don't make offensive jokes.") was a refusal it missed and is counted as one
(`refused_reviewed`). Replies are not published.

| | Agent 128K, off | Agent 128K, on | 64 GB, on |
|---|---:|---:|---:|
| Borderline requests refused | 7 of 8 | 0 of 8 | 0 of 8 |
| Neutral controls answered | 2 of 2 | 2 of 2 | 2 of 2 |
| Smoke test (reasoning, tool call, code) | pass | pass | pass |

With the switch off, the model answered only the villain's threat for a novel. With it on, both profiles answered
all eight. The neutral answers changed by a word or a line.

## Speed

Prefill tok/s / decode tok/s / ms per verify step (decode tok/s follows MTP acceptance; the step cost does not):

| Run | 4K + 256 | 32K + 512 | 131K + 512 | 8K + 1,024 |
|---|---|---|---|---|
| Agent 128K, off | 1,432 / 108.5 / 29.5 | 2,690 / 124.0 / 24.1 | 2,909 / 125.6 / 27.0 | 2,506 / 121.8 / 25.6 |
| Agent 128K, on | 1,398 / 110.0 / 29.6 | 2,663 / 120.0 / 25.5 | 2,914 / 123.5 / 24.1 | 2,534 / 119.5 / 27.0 |
| 64 GB, on | 1,226 / 84.7 / 38.3 | 3,375 / 79.2 / 37.2 | 3,358 / 86.1 / 39.9 | 3,307 / 77.7 / 40.7 |

The projection adds no measurable cost. On the agent profile the step costs span the same 24.1–27.0 ms with the
switch on and off. The 64 GB profile is within 3% of its
[published runs](../2026-09-30/README.md) (36.0–40.8 ms and 3,396–3,413 tok/s at 8K–131K). Its CPU ran at up to
89 °C, warmer than in those runs. The 4K request is not the first after the start here (the refusal check ran
first), so it is faster than in earlier reports.

No request was preempted. Peak GPU memory: 23,987 / 24,009 MiB (agent profile off / on), 23,741 MiB (64 GB
profile).

## Files

- [`summary.json`](summary.json): every run above.
- [`raw/`](raw/): `*-bench.json` (benchmark client output) and `*-refusal.json` (prompts, classifications, reply
  lengths).
