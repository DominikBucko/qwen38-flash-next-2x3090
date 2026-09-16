# Memory sizing and OOM recovery

This model uses three different memory pools. Treat them separately:

- GPU VRAM holds dense weights, the MTP draft, hot routed experts, recurrent
  state, CUDA graphs, and the BF16 KV cache.
- System RAM holds the FP8 PLE table and the host-backed routed-expert tier.
- Swap provides temporary headroom while checkpoints are loaded and repacked.

## Host memory and swap

128 GiB of RAM is the minimum tested host size, not the complete virtual-memory
requirement. Configure at least 32 GiB of swap on fast NVMe; 48–64 GiB is the
recommended starting point. Check it before launching:

```bash
free -h
swapon --show
```

On a typical ext4/xfs Linux host with no existing swap, a dedicated 64 GiB
swapfile can be created like this:

```bash
sudo fallocate -l 64G /swapfile-qwen38
sudo chmod 600 /swapfile-qwen38
sudo mkswap /swapfile-qwen38
sudo swapon /swapfile-qwen38
```

Add `/swapfile-qwen38 none swap sw 0 0` to `/etc/fstab` only after confirming
the file works. Do not run these commands over an existing file. Btrfs, ZFS,
encrypted-root, and network-backed filesystems can require different swapfile
setup; follow the filesystem's own documentation.

The profile passes `--max-parallel-loading-workers 1`, but the pinned runtime
ignores that option. The target loader and PLE worker overlap, so a machine can
run out of host memory before the server is ready. Slow progress through the
25 checkpoint shards is normal when the host is under memory pressure.

Configured or allocated swap is not automatically a serving failure. Continuous
swap traffic is. Watch `vmstat 1` while generating: persistent nonzero `si` or
`so` means the working set is reaching storage and throughput will suffer.

## GPU memory

The default 256K profile uses hot84. Hot88 is an optional, tighter profile.
The main adjustable users of VRAM are:

| Setting | Released value | Lower-memory value | Tradeoff |
|---|---:|---:|---|
| `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` | 84 | 80 | Saves about 116 MiB per removed slot on each GPU; more expert misses can reduce decode speed. |
| `KV_CACHE_MEMORY_BYTES` | 4,429,185,024 | 4,294,967,296 | Saves 128 MiB per GPU; verify that reported KV capacity remains at least 262,144 tokens. |
| `MAX_NUM_BATCHED_TOKENS` | 4,096 | 2,048 | Reduces prefill temporary tensors; lowers prefill throughput. |

Update and rebuild before tuning around a QSA score-allocation OOM. The patched
selector limits score chunks to 64 MiB and releases each chunk before allocating
the next. The previous implementation could have two 128 MiB score tensors live
at once. This changes temporary storage, not weight precision, attention top-k,
or the number of visible keys. It does not remove the memory needed by other
prefill operations, so hot88 can still be too tight on some hosts.

Hot84 frees about
464 MiB of expert storage per GPU compared with hot88, without reducing context
or model precision. Existing `.env` files must be updated explicitly; rebuilding
does not override a value of `88` supplied by the user.
Experts that are not cached on the GPU are still available from RAM. With the
patched runtime, hot84 passed a 262,016-input + 128-output request as the first
request after startup, then two shorter requests, with no inference-time
allocation retries. Hot88 passed too, but still logged four retries. Both had
two recoverable retries during weight loading; that is a separate phase.
These are native-runtime checks on one host, not a guarantee for every driver
or display setup. The [validation record](../benchmarks/2026-09-16/qsa-memory.json)
contains the settings and limits.

The fresh Docker repeat also passed the same cold-256K-first sequence on hot84,
with zero inference allocation retries and the full 276,313-token KV pool.
No hot-cache or capacity overrides were supplied to that launch. See the
[Docker record](../benchmarks/2026-09-16/docker-validation.json).

The longer 4,096-output-token comparison measured 78.92 / 76.06 / 75.81 tok/s
at short context for hot88 / hot86 / hot84. Near-full-context decode was
77.50 / 77.67 / 77.59 tok/s. Hot84 had no inference allocation retries; hot88
and hot86 had two each. These are three short runs and one long run per profile,
not a precise cache-size penalty. See the [test notes](../benchmarks/2026-09-16/qsa-memory.md).

Apply the smallest change that starts reliably. Do not lower weight precision,
KV precision, MTP precision, or PLE precision as an OOM workaround; those alter
the model's quality contract.

Use the failure location to choose the pool:

- If the process is killed while loading shards and the kernel log contains an
  out-of-memory kill, add host swap. Changing `MAX_PARALLEL_LOADING_WORKERS`
  will not help with this pinned runtime.
- If vLLM reports `torch.OutOfMemoryError` on a GPU during startup or the first
  prompt, reduce the hot cache first, then the explicit KV allocation.
- If the server runs but `vmstat 1` shows sustained swap-in/swap-out during
  decode, the host-resident PLE or expert working set is paging. More swap will
  prevent a crash but will not restore speed; stop other memory-heavy processes
  or add RAM.

An OOM during the first real prompt can still be a VRAM-headroom failure. A
4,096-token prefill materializes an 80 MiB BF16 HyperConnection gate. If the
server starts with only a few dozen MiB free, it can pass CUDA-graph capture and
then fail on that allocation. Lower the hot cache first, or reduce the explicit
KV allocation if the resulting capacity still covers the desired context.

`MAX_MODEL_LEN` is not the shared allocation. It is the maximum length allowed
for each request. `KV_CACHE_MEMORY_BYTES` creates the pool shared by active
requests, and vLLM assigns blocks from that pool according to actual sequence
length. Reducing `MAX_MODEL_LEN` while leaving the KV byte allocation fixed does
not return that reserved VRAM.

## Two concurrent requests

Rebuild the container after updating the runtime overlay. Older images can hang
halfway through CUDA graph capture when `MAX_NUM_SEQS=2`. The PLE dummy-output
flag must be re-armed before each capture warmup, not just once before all graph
shapes. The updated overlay fixes this handshake; restricting graph sizes is
not required.

Set `MAX_NUM_SEQS=2` to admit two requests. The KV pool is shared rather than
split into two fixed halves, but this hybrid model also reserves recurrent state
per sequence. At the same KV byte setting, increasing `MAX_NUM_SEQS` therefore
slightly reduces the reported token capacity.

For two equal 128K total context windows, use `MAX_MODEL_LEN=131072` as a
per-request admission limit and verify the startup log reports at least 262,144
KV-cache tokens. Context length includes prompt and generated tokens. Leaving
`MAX_MODEL_LEN=262144` also permits two shorter requests, but it allows one
client to consume almost the entire pool and force the other to wait or be
preempted.

The capacity profile used for our two-client test traded hot experts for a
larger shared KV pool:

```dotenv
MAX_NUM_SEQS=2
MAX_MODEL_LEN=131072
MAX_NUM_BATCHED_TOKENS=2048
KV_CACHE_MEMORY_BYTES=4697620480
VLLM_WNA16_STATIC_HOT_CACHE_SIZE=80
MTP_DEPTH=3
```

Put these overrides in the repository's `.env` file when using `make serve`,
or export them before calling `scripts/docker_serve.sh`. Keep the usual
`MODEL_DIR` setting. `KV_CACHE_MEMORY_BYTES` is per GPU in this TP2 setup; it is
not a separate pool per client.

This is a capacity-oriented profile, not the published single-stream speed
profile. It admits two 128K windows and reported 263,416 KV-cache tokens in our
test; aggregate decode still depends strongly on prompt mix and MTP acceptance.

The September 16 integration check completed two distinct requests of 129,024
input + 2,048 output tokens each, with MTP3 and the default graph sizes. Both
requests were resident together; peak KV use was 97.1%, with zero preemptions.
Small sequential and concurrent JSON/secret-isolation checks also passed.
This was one native-runtime test, not a fresh Docker deployment or a broad
quality evaluation. Rebuild and check capacity on your own host.

A fresh Docker build also passed the same full-context pair: 97.1% peak KV use,
zero preemptions, and no inference allocation retries. Overlapping decode was
81.9 tok/s aggregate; the short 1,024-input + 2,048-output pair reached 93.0
tok/s aggregate. Those are single capacity checks, not repeat-averaged speed
claims. See the [Docker validation record](../benchmarks/2026-09-16/docker-validation.json).

A 4 GiB pool (`4294967296`) reported only 240,510 tokens with two sequences in
our pinned runtime. It cannot keep two full 131,072-token windows resident at
once. More admission slots do not create more KV capacity.

Do not infer capacity from the configuration alone. The startup line beginning
`GPU KV cache size:` is authoritative for that launch.
