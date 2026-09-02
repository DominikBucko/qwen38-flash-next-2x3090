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

The profile loads tensor-parallel ranks with
`--max-parallel-loading-workers 1`, but the target loader and PLE worker still
overlap. A machine can therefore run out of host memory before the server is
ready. Slow progress through the 25 checkpoint shards is normal when the host
is under memory pressure.

Configured or allocated swap is not automatically a serving failure. Continuous
swap traffic is. Watch `vmstat 1` while generating: persistent nonzero `si` or
`so` means the working set is reaching storage and throughput will suffer.

## GPU memory

The released 256K profile is deliberately tight. Its main adjustable users of
VRAM are:

| Setting | Released value | Lower-memory value | Tradeoff |
|---|---:|---:|---|
| `VLLM_WNA16_STATIC_HOT_CACHE_SIZE` | 88 | 86, then 84 | Saves about 116 MiB per removed slot on each GPU; more expert misses reduce decode speed. |
| `KV_CACHE_MEMORY_BYTES` | 4,429,185,024 | 4,294,967,296 | Saves 128 MiB per GPU; verify that reported KV capacity remains at least 262,144 tokens. |
| `MAX_NUM_BATCHED_TOKENS` | 4,096 | 2,048 | Reduces prefill temporary tensors; lowers prefill throughput. |

Apply the smallest change that starts reliably. Do not lower weight precision,
KV precision, MTP precision, or PLE precision as an OOM workaround; those alter
the model's quality contract.

Use the failure location to choose the pool:

- If the process is killed while loading shards and the kernel log contains an
  out-of-memory kill, add host swap and keep
  `MAX_PARALLEL_LOADING_WORKERS=1`.
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
KV_CACHE_MEMORY_BYTES=4697620480
VLLM_WNA16_STATIC_HOT_CACHE_SIZE=80
```

This is a capacity-oriented profile, not the published single-stream speed
profile. It admits two 128K windows and reported 263,416 KV-cache tokens in our
test; aggregate decode still depends strongly on prompt mix and MTP acceptance.

Do not infer capacity from the configuration alone. The startup line beginning
`GPU KV cache size:` is authoritative for that launch.
