# Reproduction and release guide

## 1. Hardware and host prerequisites

The validated system has two RTX 3090 24 GB cards and 128 GB of system memory.
The launch profile assumes two visible CUDA GPUs. It reserves approximately
4.13 GiB of KV cache on each GPU for one 262,144-token sequence.

Required software:

- Linux x86_64 with a working NVIDIA Container Toolkit;
- Docker with Compose v2 (Compose is optional if using `make serve`);
- Python 3.10+ for assembly scripts;
- `hf` from `huggingface_hub` with `hf_xet`;
- enough local storage for both upstream sources, the 120 GiB release tree, and
  a temporary 4 GiB MTP build.

Run `make preflight` before the first image build. It is read-only. The runtime
container receives `SYS_PTRACE` for cross-process PLE CUDA IPC. On hosts where
that is insufficient, `kernel.yama.ptrace_scope=0` may be required temporarily;
do not make that security relaxation persistent without reviewing it.

## 2. Immutable inputs

[`repro.lock.json`](../repro.lock.json) is authoritative. It pins:

- Intel's AutoRound checkpoint by full Hub commit;
- RadixArk's PLE/MTP source by full Hub commit;
- the vLLM day-0 container by OCI digest;
- Humming kernels by package version;
- target/MTP tensor counts and payload byte counts;
- the validated hardware and serving profile.

Never replace these with branch names such as `main` in a tagged release.

## 3. Build the Hugging Face upload tree

Install the current Hugging Face CLI, authenticate, and download the inputs:

```bash
python3 -m pip install --upgrade 'huggingface_hub[hf_xet]'
hf auth login
./scripts/download_sources.sh /models/qwen38-sources
```

The recommended builder is the same digest-pinned container used for serving,
which already provides the matching PyTorch, safetensors, and
compressed-tensors stack:

```bash
make build-image
./scripts/assemble_with_docker.sh /models/qwen38-sources upload
```

This creates `/models/qwen38-sources/upload` and keeps every large path under a
single container mount, so hard links remain available. Advanced users can run
`assemble_hf_repo.sh` directly in a compatible host Python environment.

The target builder does not requantize or repack Intel tensors. It omits the
Intel BF16 PLE-only shard and bundled BF16 MTP file, inserts the published FP8
PLE shards, and writes a new auditable index. The separate MTP builder quantizes
only routed draft experts to symmetric INT4 group-32; target verification still
determines emitted tokens.

Set `KEEP_WORK=1` if you want to retain the large symlinked MTP intermediate for
debugging. Otherwise the script deletes only the uniquely named temporary
directory it created beneath the supplied work directory.

## 4. Upload and cross-pin releases

```bash
./scripts/upload_hf.sh OWNER/MODEL /models/qwen38-sources/upload
```

The script validates first, uses `hf upload`/Xet so interrupted uploads can be
resumed, and prints the resulting Hub commit. Before tagging GitHub:

Keep the source directories in place until the upload finishes because the
assembled target uses hard links to unchanged upstream shards.

1. put the Hugging Face repo and full commit in `repro.lock.json`;
2. put the GitHub repository and tag/commit in the Hugging Face model card;
3. rerun `make validate`;
4. commit, tag `v1.0.0`, and create a GitHub release;
5. verify a fresh machine can download by the pinned Hub revision and start the
   endpoint without relying on an untracked file.

Do not upload model weights to GitHub or GitHub Releases. The Hub is designed
for resumable, content-addressed model uploads and keeps the model card next to
the tensors.

## 5. Serve

```bash
export MODEL_DIR=/models/qwen38-upload
make build-image
make preflight
make serve
```

Equivalent Compose launch:

```bash
export MODEL_DIR=/models/qwen38-upload
docker compose -f docker/compose.yaml up --build
```

The approximate QSA selector is default. To opt into exact selection:

```bash
export VLLM_QSA_EXACT_TOPK=1
make serve
```

The exact setting is a precision experiment, not the recommended performance
profile. A single AgentBench A/B did not improve hidden-functional passes or
mean score.

## 6. What CI can and cannot prove

GitHub CI checks Python syntax, overlay checksums, lock/Docker consistency,
accidental model blobs, symlinks, cache files, and common token formats. It
cannot validate CUDA kernels, 256K allocation, or throughput. Those checks need
the target hardware and the procedure in [`benchmarks.md`](benchmarks.md).
