"""Linux process-memory observation and safe-boundary heap trimming."""

from __future__ import annotations

import ctypes
import gc
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_CGROUP_V2_MEMORY_CURRENT = Path("/sys/fs/cgroup/memory.current")
_CGROUP_V2_MEMORY_MAX = Path("/sys/fs/cgroup/memory.max")


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """Current RSS and the effective host/cgroup memory budget."""

    process_rss_bytes: int
    system_available_bytes: int
    cgroup_memory_current_bytes: int | None = None
    cgroup_memory_limit_bytes: int | None = None


@dataclass(frozen=True, slots=True)
class HostMemoryTrimResult:
    """One best-effort allocator trim observed at a quiescent boundary."""

    before: HostMemorySnapshot
    after: HostMemorySnapshot
    collected_objects: int
    allocator_trimmed: bool

    @property
    def released_bytes(self) -> int:
        """Return the contemporaneous RSS decrease."""
        return max(0, self.before.process_rss_bytes - self.after.process_rss_bytes)


def host_memory_snapshot() -> HostMemorySnapshot:
    """Read RSS and the smaller of host and cgroup-v2 headroom."""
    page_size = int(os.sysconf("SC_PAGE_SIZE"))
    with open("/proc/self/statm", encoding="utf-8") as handle:
        fields = handle.readline().split()
    if len(fields) < 2:
        raise RuntimeError("/proc/self/statm omitted the resident page count")
    resident_bytes = int(fields[1]) * page_size

    available_bytes: int | None = None
    with open("/proc/meminfo", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("MemAvailable:"):
                continue
            parts = line.split()
            if len(parts) < 2:
                break
            available_bytes = int(parts[1]) * 1024
            break
    if available_bytes is None:
        raise RuntimeError("/proc/meminfo omitted MemAvailable")
    cgroup_current, cgroup_limit = _cgroup_memory_usage()
    if cgroup_current is not None and cgroup_limit is not None:
        available_bytes = min(
            available_bytes,
            max(0, cgroup_limit - cgroup_current),
        )
    return HostMemorySnapshot(
        process_rss_bytes=resident_bytes,
        system_available_bytes=available_bytes,
        cgroup_memory_current_bytes=cgroup_current,
        cgroup_memory_limit_bytes=cgroup_limit,
    )


def _cgroup_memory_usage() -> tuple[int | None, int | None]:
    """Return cgroup-v2 current/limit bytes when a finite limit exists."""
    try:
        current_text = _CGROUP_V2_MEMORY_CURRENT.read_text(encoding="utf-8").strip()
        limit_text = _CGROUP_V2_MEMORY_MAX.read_text(encoding="utf-8").strip()
    except OSError:
        return None, None
    if limit_text == "max":
        return None, None
    try:
        current = int(current_text)
        limit = int(limit_text)
    except ValueError as error:
        raise RuntimeError("cgroup-v2 memory counters are not integers") from error
    if current < 0 or limit <= 0:
        raise RuntimeError("cgroup-v2 memory counters are outside their domain")
    return current, limit


def trim_process_heap() -> HostMemoryTrimResult:
    """Collect Python garbage and return free glibc heap pages to the OS."""
    before = host_memory_snapshot()
    collected_objects = gc.collect()
    malloc_trim = _malloc_trim()
    allocator_trimmed = bool(malloc_trim(0)) if malloc_trim is not None else False
    after = host_memory_snapshot()
    return HostMemoryTrimResult(
        before=before,
        after=after,
        collected_objects=collected_objects,
        allocator_trimmed=allocator_trimmed,
    )


@lru_cache(maxsize=1)
def _malloc_trim() -> Any | None:
    """Resolve glibc malloc_trim when the process libc provides it."""
    library = ctypes.CDLL(None)
    function = getattr(library, "malloc_trim", None)
    if function is None:
        return None
    function.argtypes = [ctypes.c_size_t]
    function.restype = ctypes.c_int
    return function


__all__ = [
    "HostMemorySnapshot",
    "HostMemoryTrimResult",
    "host_memory_snapshot",
    "trim_process_heap",
]
