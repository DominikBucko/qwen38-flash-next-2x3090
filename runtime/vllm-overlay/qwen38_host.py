# SPDX-License-Identifier: Apache-2.0
"""Host plan for hot-only serving (QWEN38_HOT_ONLY): which CPUs run the cold-expert pool of each GPU rank, and how
much RAM the expert arenas get.

Standard library only. Imported by the GPU workers (cpu_experts.py, stream_v2.py, expert_store.py) and the PLE
worker; the container entrypoint runs this file directly to print the plan before the server starts.

CPU pool (QWEN38_CPU_EXPERTS_CPUS=auto): one hardware thread on each physical core, except the first quarter of
the cores of every L3 domain (CCD/CCX on AMD, the whole die on most Intel desktops), which stay with the serving
processes. At most QWEN38_CPU_EXPERTS_MAX_THREADS (24) threads, spread evenly over the domains. Decode is bound by
memory bandwidth, and on AMD each CCD has its own link to memory, so the pool spans every CCD. With several GPU
ranks (one worker process each), every rank gets its own pool: whole L3 domains in turn when there are at least as
many domains as ranks, otherwise the pool cores in turn. An explicit list may give one pool per rank, separated by
";" (for example 2-7,10-15;18-23,26-31).

Serving processes (QWEN38_MAIN_CPUS=auto): every allowed CPU except the SMT siblings of the pools' cores. They may
use the pools' cores too; a pool sleeps during long prefill steps, when the serving processes need the cores.

Expert arena (QWEN38_EXPERT_ARENA_SLOTS=auto, per rank): the memory budget minus QWEN38_SERVE_OVERHEAD_GIB (10 GiB
for the processes, CUDA, pinned buffers and page cache), split evenly between the ranks, in 2,535,424-byte expert
slots. The budget is the container's memory limit, or without one the installed RAM (MemTotal rounded up to a
multiple of 8 GiB) minus QWEN38_HOST_RESERVE_GIB (8 GiB): 56 GiB on a 64 GB machine. Cold experts that do not fit
stay on NVMe (the "tail"), behind the page cache.
"""
from __future__ import annotations

import os
import sys

GIB = 1 << 30
SLOT_BYTES = 2_535_424          # expert_store.SLOT
_SYS = "/sys"
_PROC = "/proc"
_CGROUP = "/sys/fs/cgroup"


def parse_cpus(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def format_cpus(cpus) -> str:
    cpus = sorted(set(cpus))
    runs: list[str] = []
    i = 0
    while i < len(cpus):
        j = i
        while j + 1 < len(cpus) and cpus[j + 1] == cpus[j] + 1:
            j += 1
        runs.append(str(cpus[i]) if i == j else f"{cpus[i]}-{cpus[j]}")
        i = j + 1
    return ",".join(runs)


def _read(path: str) -> str | None:
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _allowed() -> list[int]:
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return list(range(os.cpu_count() or 1))


def topology(allowed: list[int], sys_root: str = _SYS) -> list[dict]:
    """Physical cores among `allowed`: [{"threads": [cpu, ...], "domain": key}], ordered by first CPU id."""
    cores: dict[tuple, dict] = {}
    for cpu in allowed:
        base = f"{sys_root}/devices/system/cpu/cpu{cpu}"
        pkg = _read(f"{base}/topology/physical_package_id") or "0"
        siblings = _read(f"{base}/topology/thread_siblings_list") or str(cpu)
        domain = None
        for index in range(10):
            level = _read(f"{base}/cache/index{index}/level")
            if level is None:
                break
            if level == "3":
                domain = _read(f"{base}/cache/index{index}/shared_cpu_list")
                break
        core = cores.setdefault((pkg, siblings), {"threads": [], "domain": (pkg, domain or pkg)})
        core["threads"].append(cpu)
    return sorted(cores.values(), key=lambda c: c["threads"][0])


def _domain_pools(allowed: list[int], sys_root: str) -> tuple[list[list[int]], list[dict]]:
    """Pool candidates per L3 domain (the first ceil(n / 4) cores of each domain stay with the serving processes)."""
    cores = topology(allowed, sys_root)
    domains: dict = {}
    for core in cores:
        domains.setdefault(core["domain"], []).append(core)
    per_domain = []
    for members in domains.values():
        keep = -(-len(members) // 4)
        per_domain.append([c["threads"][0] for c in members[keep:]])
    return per_domain, cores


def _spread(per_domain: list[list[int]], cap: int) -> list[int]:
    """Round-robin over the domains, so a thread cap still spans all of them."""
    pool: list[int] = []
    for rank in range(max((len(p) for p in per_domain), default=0)):
        for members in per_domain:
            if rank < len(members) and len(pool) < cap:
                pool.append(members[rank])
    return pool


def auto_pools(ranks: int = 1, allowed: list[int] | None = None, sys_root: str = _SYS,
               max_threads: int | None = None) -> list[list[int]]:
    """One CPU pool per GPU rank."""
    allowed = _allowed() if allowed is None else sorted(allowed)
    if max_threads is None:
        max_threads = int(os.environ.get("QWEN38_CPU_EXPERTS_MAX_THREADS", "24"))
    per_domain, cores = _domain_pools(allowed, sys_root)
    cap = max(1, -(-max_threads // ranks))
    if ranks > 1 and len([d for d in per_domain if d]) >= ranks:
        domains = [d for d in per_domain if d]
        pools = [_spread(domains[r::ranks], cap) for r in range(ranks)]
    else:
        flat = _spread(per_domain, max_threads)
        pools = [flat[r::ranks][:cap] for r in range(ranks)]
    if not all(pools):                          # tiny hosts: every rank shares what there is
        firsts = [c["threads"][0] for c in cores]
        fallback = firsts[1:] or firsts[:1]
        pools = [p or fallback for p in pools]
    return [sorted(p) for p in pools]


def auto_pool(allowed: list[int] | None = None, sys_root: str = _SYS, max_threads: int | None = None) -> list[int]:
    return auto_pools(1, allowed, sys_root, max_threads)[0]


def all_pools(ranks: int = 1, sys_root: str = _SYS) -> list[list[int]]:
    spec = os.environ.get("QWEN38_CPU_EXPERTS_CPUS", "auto").strip()
    if spec and spec != "auto":
        parts = [p for p in spec.split(";") if p.strip()]
        if len(parts) == ranks:
            return [parse_cpus(p) for p in parts]
        cpus = parse_cpus(",".join(parts))
        if ranks == 1:
            return [cpus]
        size = -(-len(cpus) // ranks)           # one list for several ranks: consecutive shares
        return [cpus[r * size:(r + 1) * size] or cpus for r in range(ranks)]
    return auto_pools(ranks, sys_root=sys_root)


def pool_cpus(sys_root: str = _SYS, rank: int = 0, ranks: int = 1) -> list[int]:
    return all_pools(ranks, sys_root)[rank]


def main_cpus(pool: list[int] | None = None, sys_root: str = _SYS, ranks: int = 1) -> list[int]:
    spec = os.environ.get("QWEN38_MAIN_CPUS", "auto").strip()
    allowed = _allowed()
    if spec == "all":
        return allowed
    if spec and spec != "auto":
        return parse_cpus(spec)
    busy_pool = set(pool) if pool is not None else {c for p in all_pools(ranks, sys_root) for c in p}
    busy = set()
    for core in topology(allowed, sys_root):
        if core["threads"][0] in busy_pool:
            busy.update(core["threads"][1:])    # the pool cores' SMT siblings stay idle
    return [cpu for cpu in allowed if cpu not in busy]


def mem_total_bytes(proc_root: str = _PROC) -> int:
    text = _read(f"{proc_root}/meminfo") or ""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemTotal not found in /proc/meminfo")


def cgroup_limit_bytes(cgroup_root: str = _CGROUP) -> int | None:
    for path in (f"{cgroup_root}/memory.max", f"{cgroup_root}/memory/memory.limit_in_bytes"):
        value = _read(path)
        if value and value != "max":
            try:
                limit = int(value)
            except ValueError:
                continue
            if limit < 1 << 60:                 # cgroup v1 reports "no limit" as a huge number
                return limit
    return None


def installed_bytes(mem_total: int) -> int:
    """Installed RAM estimated from MemTotal (the kernel keeps 1-2 GiB): rounded up to a multiple of 8 GiB."""
    step = 8 * GIB
    return -(-mem_total // step) * step


def memory_budget_bytes(proc_root: str = _PROC, cgroup_root: str = _CGROUP) -> tuple[int, str]:
    total = mem_total_bytes(proc_root)
    limit = cgroup_limit_bytes(cgroup_root)
    if limit is not None and limit < total:
        return limit, "container memory limit"
    reserve = float(os.environ.get("QWEN38_HOST_RESERVE_GIB", "8")) * GIB
    return int(min(total, installed_bytes(total) - reserve)), \
        f"{installed_bytes(total) / GIB:g} GiB installed - {reserve / GIB:g} GiB for the OS"


def arena_slots(total_cold: int, layers: int = 48, proc_root: str = _PROC, cgroup_root: str = _CGROUP,
                ranks: int = 1) -> int:
    """Cold experts one rank keeps in its RAM arena (the rest stay on NVMe), at least one per layer."""
    spec = os.environ.get("QWEN38_EXPERT_ARENA_SLOTS", "auto").strip()
    if spec and spec != "auto":
        want = int(spec)
    else:
        budget, _ = memory_budget_bytes(proc_root, cgroup_root)
        overhead = float(os.environ.get("QWEN38_SERVE_OVERHEAD_GIB", "10")) * GIB
        want = int((budget - overhead) // ranks // SLOT_BYTES)
    return max(min(layers, total_cold), min(total_cold, want))


def cold_per_rank(hot: int, ranks: int, experts: int = 512, layers: int = 48) -> int:
    return layers * (experts // ranks - hot)


def describe(hot: int = 32, ranks: int = 1) -> str:
    pools = all_pools(ranks)
    main = main_cpus(ranks=ranks)
    try:
        budget, source = memory_budget_bytes()
    except (OSError, RuntimeError):
        budget, source = 0, "unknown"
    total = cold_per_rank(hot, ranks)
    slots = arena_slots(total, ranks=ranks)
    tail = total - slots
    lines = []
    for r, pool in enumerate(pools):
        label = "CPU expert pool" if ranks == 1 else f"CPU expert pool, GPU rank {r}"
        lines.append(f"{label}: {len(pool)} threads on CPUs {format_cpus(pool)}")
    per = "" if ranks == 1 else " per GPU rank"
    lines += [
        f"Serving processes: CPUs {format_cpus(main)}",
        f"Memory budget: {budget / GIB:.1f} GiB ({source})",
        f"Expert arena{per}: {slots:,} of {total:,} cold experts in RAM ({slots * SLOT_BYTES / GIB:.1f} GiB), "
        f"{tail:,} on NVMe ({tail * SLOT_BYTES / GIB:.1f} GiB)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", action="store_true", help="print the resolved settings as KEY=VALUE lines")
    ap.add_argument("--ranks", type=int, default=1, help="GPU ranks (tensor/expert parallel size)")
    args = ap.parse_args()
    hot = int(os.environ.get("QWEN38_HOT_ONLY", "32"))
    if args.env:
        pools = all_pools(args.ranks)
        print(f"QWEN38_CPU_EXPERTS_CPUS={';'.join(format_cpus(p) for p in pools)}")
        print(f"QWEN38_MAIN_CPUS={format_cpus(main_cpus(ranks=args.ranks))}")
        print(f"QWEN38_EXPERT_ARENA_SLOTS={arena_slots(cold_per_rank(hot, args.ranks), ranks=args.ranks)}")
    else:
        print(describe(hot, args.ranks))
