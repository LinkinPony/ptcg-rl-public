"""Read-only opportunity and GPU resource probes for league workers."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ptcg_rl.evaluation.continuous_league.models import (
    ResourceConfig,
    ResourceSnapshot,
)


def probe_resources(
    config: ResourceConfig,
    *,
    gpu_index: int | None = None,
) -> ResourceSnapshot:
    """Collect affinity, load, memory, and one selected visible GPU."""
    affinity_count = len(os.sched_getaffinity(0))
    load_1m = os.getloadavg()[0]
    memory_available = _memory_available_bytes()
    gpu = _best_gpu(gpu_index=gpu_index)
    cuda_available = gpu is not None
    quiet = (
        load_1m / affinity_count <= config.max_load_per_cpu
        and memory_available >= config.minimum_available_memory_bytes
    )
    if gpu is not None:
        quiet = (
            quiet
            and gpu[1] <= config.maximum_gpu_utilization_percent
            and gpu[2] >= config.minimum_gpu_free_memory_bytes
        )
    return ResourceSnapshot(
        cpu_affinity_count=affinity_count,
        load_1m=load_1m,
        memory_available_bytes=memory_available,
        cuda_available=cuda_available,
        gpu_index=None if gpu is None else gpu[0],
        gpu_utilization_percent=None if gpu is None else gpu[1],
        gpu_memory_free_bytes=None if gpu is None else gpu[2],
        quiet=quiet,
    )


def _memory_available_bytes() -> int:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("/proc/meminfo does not report MemAvailable")


def _best_gpu(*, gpu_index: int | None = None) -> tuple[int, int, int] | None:
    command = (
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.free",
        "--format=csv,noheader,nounits",
    )
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    candidates: list[tuple[int, int, int]] = []
    for line in result.stdout.splitlines():
        try:
            index, utilization, free_mib = (
                int(value.strip()) for value in line.split(",")
            )
        except (TypeError, ValueError):
            continue
        candidates.append((index, utilization, free_mib * 1024**2))
    if not candidates:
        return None
    if gpu_index is not None:
        return next((item for item in candidates if item[0] == gpu_index), None)
    return min(candidates, key=lambda item: (item[1], -item[2], item[0]))


__all__ = ["probe_resources"]
