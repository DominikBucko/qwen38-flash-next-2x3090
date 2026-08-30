# Runtime architecture

## Checkpoint composition

| Component | Representation | Placement while serving |
|---|---|---|
| Target routed experts and eligible linear weights | Intel AutoRound W4A16 | 88 hot experts per layer on GPU; all experts accessible via UVA host memory |
| Sensitive target layers | Intel BF16, unchanged | GPU and configured vLLM offload path |
| 51.2B-parameter n-gram/PLE table | RadixArk FP8 E4M3FN + published scale | Host RAM, fetched through PLE offload workers |
| KV cache | BF16 | Approximately 4.13 GiB per GPU for the validated one-sequence profile |
| MTP draft routed experts | Local RTN INT4 symmetric group-32 | Draft path used for three-token speculation |
| Remaining MTP tensors | Unchanged source dtype | Draft path |

The target checkpoint is 116.183 GiB of tensor payload. The compact MTP draft is
3.855 GiB. The large figure is expected: W4A16 compresses eligible target
weights, while the n-gram table alone still contains 51.2B parameters and is
stored in FP8 rather than four bits.

## Serving flow

1. vLLM runs tensor parallelism across both GPUs and expert parallelism across
   routed experts.
2. Dense work and the static expert hot set run on the 3090s.
3. Missed experts remain available through the host-resident UVA pool and the
   system-memory path.
4. MTP proposes up to three tokens; the target model verifies them, preserving
   target sampling semantics while raising decode throughput when acceptance is
   high.
5. Prefix caching uses Mamba-aligned boundaries from the patched scheduler.
6. QSA uses the faster approximate persistent top-k selector by default. Exact
   score-ranked selection is available through an environment switch.

## Why these flags are fixed

- `--max-num-seqs 1` and the explicit KV allocation prioritize a real 256K
  request over concurrency.
- `--max-num-batched-tokens 4096` balances chunked prefill with transient VRAM.
- `--disable-custom-all-reduce` reflects the lack of usable CUDA P2P between the
  two consumer cards on the validated topology.
- `--no-async-scheduling` avoids nondeterministic interaction between PLE CPU
  token mirroring and speculative placeholders in this runtime snapshot.
- BF16 KV remains the quality-conservative choice and already fits.
