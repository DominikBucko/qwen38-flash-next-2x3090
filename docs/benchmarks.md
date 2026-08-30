# Benchmark evidence

## Isolated serving measurements

The sanitized machine-readable results are in
[`benchmarks/serving-summary.json`](../benchmarks/serving-summary.json).

The full-context request used 262,016 input tokens and 128 output tokens, exactly
reaching the model's 262,144-token limit. Prefill was 1,275.6 token/s, TTFT was
205.4 seconds, and decode after that prompt was 54.5 token/s. Two warmed
128-input/4,096-output greedy probes with the compact MTP draft produced 134.0
and 127.1 output token/s.

These are single-request probes. Prompt accounting from an agent harness is not
a substitute for isolated prefill or decode timing because repeated context,
prefix-cache hits, tool turns, and non-streaming response time are mixed.

## LBX AgentBench v0.2

[`benchmarks/agentbench-summary.json`](../benchmarks/agentbench-summary.json)
contains sanitized aggregate results only. The main Intel hybrid run used xhigh
reasoning, 32 steps, a 16,384-token per-response ceiling, and one trajectory per
task. It achieved 9/15 strict passes, 11/15 full private-suite passes, and a
97.601 mean score.

The exact-versus-approximate QSA A/B used a cache-fixed harness and one
trajectory per mode. Exact QSA improved strict workflow completion but had one
fewer hidden-functional pass and a 0.445 lower mean score, so approximate remains
the default. The evidence is directional, not statistically conclusive; the
benchmark protocol calls for three fresh trajectories per configuration before
making a strong quality claim.

Private fixtures, hidden assertions, oracle code, raw reasoning traces, final
workspaces, and task patches are deliberately excluded. Publishing them would
contaminate future evaluations. The retained internal archive should remain
private and is identified only by hashes/configuration metadata in release
notes.

## Reproduction record

Record the driver, container digest, server flags, context length, request
shape, and warmup state with every published run. Use the same request shape
when you compare two runtime changes.
