# Contributing hardware reports, fixes, and benchmarks

The most useful contributions are independent reproductions, clear failure
reports, and measured improvements. You do not need to write CUDA code to help.

## Share a run on your hardware

Use the [hardware report form](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=hardware-report.yml).
Successful dual RTX 3090 reproductions, experiments on dual RTX 4090, and failed
attempts all help establish what works. A 4090 report starts as community evidence;
it does not automatically become a validated profile.

Include the GPU models, CPU, RAM capacity and memory configuration, driver,
PCIe topology, runtime commit, checkpoint revision, environment overrides,
prompt/output token counts, warmup state, and measurements. Keep prefill, TTFT,
and output speed separate. A completed short prompt is useful but does not
prove full-context support.

Use public synthetic prompts and redact tokens, private prompts, user paths,
and machine identifiers from logs. See the [performance guide](docs/performance.md)
for the comparison protocol and the [hardware guide](docs/hardware.md) for
compatibility boundaries.

## Ask a question or report a bug

Use [Discussions](https://github.com/DominikBucko/qwen38-flash-next-2x3090/discussions)
for setup questions and tuning ideas. For crashes or incorrect behavior, use the
[bug report form](https://github.com/DominikBucko/qwen38-flash-next-2x3090/issues/new?template=bug-report.yml).
Check the [memory guide](docs/memory.md) first for OOMs, and link an existing issue
when your symptoms match. Include failed configurations as well as the fix.

## Submit a pull request

For a documentation fix, edit the relevant page and check its commands and links.
For runtime, checkpoint, scheduler, or benchmark changes, read [AGENTS.md](AGENTS.md)
first; it explains the coupled memory, performance, and correctness constraints.
Keep immutable pins and the runtime overlay manifest consistent.

Describe the problem, the resulting behavior, and the evidence. For performance
changes, compare identical workloads and record full-context behavior as well as
short decode. For precision or state-handling changes, include a relevant quality
or correctness check. GPU-free CI alone cannot verify CUDA correctness or speed.

Before submitting source changes:

```bash
make validate
python3 scripts/check_release_ready.py
python3 -B -m unittest discover -s tests -p test_benchmark_serving.py
python3 -B -m unittest discover -s tests -p test_serve_flags.py
for script in scripts/*.sh; do bash -n "$script" || exit; done
```

Use a focused `fix/` or `feature/` branch, or the associated issue tracker slug.
Keep discussion constructive, credit prior work, and distinguish measured results
from proposed experiments.
