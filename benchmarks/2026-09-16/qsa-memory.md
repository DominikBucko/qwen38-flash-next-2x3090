# QSA memory checks

The selector now uses 64 MiB score chunks and releases each chunk before
allocating the next. Previously, two 128 MiB score tensors could be live at
once. Visible keys, top-k, weights, and numeric precision are unchanged.

The GPU component check reduced peak extra allocation from 266.02 to 73.51 MiB
at 1,024 rows. All seven tested shapes selected the same token sets. Index order
can vary even between unchanged baseline runs. The component kernel was about
3% slower; the full-context capacity checks had similar time to first token.
See [component and capacity results](qsa-memory.json).

## Longer decode comparison

All profiles used 2× RTX 3090, 128 GB eight-channel DDR4-3200, MTP3, BF16 KV,
approximate QSA, a 4,096-token scheduler budget, and a 262,144-token limit.
Custom all-reduce was disabled and expandable segments were enabled.

Each fresh server ran a 128+512 smoke test and a 128+4,096 warmup, then three
128+4,096 requests and one 258,048+4,096 request. The public `repo-chat` input
token hashes match across profiles. Every request used a fresh prefix-cache
salt; expert and PLE caches remained warm.

| Profile | Short decode, three-run aggregate | Near-full-context decode, one run | Inference allocation retries |
|---|---:|---:|---:|
| hot88 | 78.92 tok/s | 77.50 tok/s | 2 |
| hot86 | 76.06 tok/s | 77.67 tok/s | 2 |
| hot84 | 75.81 tok/s | 77.59 tok/s | 0 |

All 18 streams, including smoke and warmup, passed exact token-count checks.
There were no preemptions or failed requests. Each profile also had two
recoverable allocation retries during loading. Hot84 passed an earlier cold
full-context check with no inference retries, too.

The metric is reciprocal mean API-observed time per token after the first
output chunk. MTP acceptance and generated continuations varied, so the small
differences are not precise cache-size penalties. These chat probes are not the
historical 130+ tok/s workload and do not replace the README's historical results.

PLE was partly swapped out. Decode-only samples read about 86–154 MiB from
swap per 4,096 output tokens; its latency cost was not isolated. Do not describe
these runs as fully RAM-resident PLE. Forced-length outputs include reasoning
tokens and do not establish completed-task quality.

These were native pinned-runtime tests with both the separate PLE capture fix
and this QSA fix installed, not fresh Docker deployments. The prompt corpus
came from the September 5 checkout, before the overlay gained two more files.
See [settings, per-run measurements, hashes, and limits](long-decode.json).
