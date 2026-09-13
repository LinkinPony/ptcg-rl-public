"""Pressure-aware CUDA cache reclamation for shared collection GPUs."""

from __future__ import annotations

import gc
import logging
import math
from dataclasses import dataclass

import torch

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CudaMemorySnapshot:
    """One allocator and device-memory observation."""

    allocated_bytes: int
    reserved_bytes: int
    device_free_bytes: int
    device_total_bytes: int

    @property
    def reclaimable_bytes(self) -> int:
        """Return inactive allocator bytes that ``empty_cache`` may release."""
        return max(self.reserved_bytes - self.allocated_bytes, 0)


@dataclass(frozen=True)
class CudaCacheTrimResult:
    """Observed effect of one pressure-triggered allocator trim."""

    before: CudaMemorySnapshot
    after: CudaMemorySnapshot
    collected_objects: int
    cublas_workspaces_cleared: bool = False

    @property
    def allocator_released_bytes(self) -> int:
        """Return bytes released from this process's caching allocator."""
        return max(self.before.reserved_bytes - self.after.reserved_bytes, 0)

    @property
    def device_released_bytes(self) -> int:
        """Return the contemporaneous increase in device-wide free memory."""
        return max(self.after.device_free_bytes - self.before.device_free_bytes, 0)


def cuda_memory_snapshot(device: torch.device) -> CudaMemorySnapshot:
    """Read allocator-local and device-wide CUDA memory without mutating it."""
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    return CudaMemorySnapshot(
        allocated_bytes=torch.cuda.memory_allocated(device),
        reserved_bytes=torch.cuda.memory_reserved(device),
        device_free_bytes=free_bytes,
        device_total_bytes=total_bytes,
    )


def should_trim_cuda_cache(
    snapshot: CudaMemorySnapshot,
    *,
    device_free_floor_fraction: float,
    minimum_reclaimable_device_fraction: float,
) -> bool:
    """Return whether device pressure justifies discarding allocator locality."""
    if snapshot.device_total_bytes <= 0:
        raise ValueError("CUDA device capacity must be positive")
    if not 0.0 < device_free_floor_fraction < 1.0:
        raise ValueError("CUDA device free floor fraction must be between zero and one")
    if not 0.0 < minimum_reclaimable_device_fraction < 1.0:
        raise ValueError(
            "CUDA minimum reclaimable fraction must be between zero and one"
        )
    free_floor_bytes = math.ceil(
        snapshot.device_total_bytes * device_free_floor_fraction
    )
    minimum_reclaimable_bytes = math.ceil(
        snapshot.device_total_bytes * minimum_reclaimable_device_fraction
    )
    return (
        snapshot.device_free_bytes < free_floor_bytes
        and snapshot.reclaimable_bytes >= minimum_reclaimable_bytes
    )


def _release_inactive_cuda_cache(
    device: torch.device,
    *,
    before: CudaMemorySnapshot,
    collected_objects: int,
    clear_cublas_workspaces: bool = False,
) -> CudaCacheTrimResult:
    """Release allocator-owned inactive pages after a synchronized snapshot."""
    cublas_workspaces_cleared = False
    if clear_cublas_workspaces:
        cublas_workspaces_cleared = _clear_cublas_workspaces_best_effort()
    torch.cuda.empty_cache()
    after = cuda_memory_snapshot(device)
    if after.device_total_bytes != before.device_total_bytes:
        raise RuntimeError("CUDA device capacity changed while trimming cache")
    return CudaCacheTrimResult(
        before=before,
        after=after,
        collected_objects=collected_objects,
        cublas_workspaces_cleared=cublas_workspaces_cleared,
    )


def _clear_cublas_workspaces_best_effort() -> bool:
    """Clear process-local cuBLAS workspaces without making it a hard gate."""
    clear = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)  # noqa: SLF001
    if not callable(clear):
        return False
    try:
        clear()
    except Exception as exc:  # pragma: no cover - defensive private-API boundary.
        _LOGGER.warning("CUDA cuBLAS workspace cleanup failed: %s", exc)
        return False
    return True


def trim_cuda_cache(
    device: torch.device,
    *,
    clear_cublas_workspaces: bool = False,
) -> CudaCacheTrimResult:
    """Unconditionally release inactive allocator pages at a safe boundary."""
    collected_objects = gc.collect()
    torch.cuda.synchronize(device)
    before = cuda_memory_snapshot(device)
    return _release_inactive_cuda_cache(
        device,
        before=before,
        collected_objects=collected_objects,
        clear_cublas_workspaces=clear_cublas_workspaces,
    )


def trim_cuda_cache_if_needed(
    device: torch.device,
    *,
    device_free_floor_fraction: float,
    minimum_reclaimable_device_fraction: float,
) -> CudaCacheTrimResult | None:
    """Release inactive blocks only when a shared device is under pressure.

    The second pressure check follows Python collection and CUDA synchronization.
    Another MPS client may have relieved device pressure while this process reached
    the safe boundary, in which case retaining the warm cache is preferable.
    """
    observed = cuda_memory_snapshot(device)
    if not should_trim_cuda_cache(
        observed,
        device_free_floor_fraction=device_free_floor_fraction,
        minimum_reclaimable_device_fraction=minimum_reclaimable_device_fraction,
    ):
        return None
    collected_objects = gc.collect()
    torch.cuda.synchronize(device)
    before = cuda_memory_snapshot(device)
    if not should_trim_cuda_cache(
        before,
        device_free_floor_fraction=device_free_floor_fraction,
        minimum_reclaimable_device_fraction=minimum_reclaimable_device_fraction,
    ):
        return None
    return _release_inactive_cuda_cache(
        device,
        before=before,
        collected_objects=collected_objects,
    )


__all__ = [
    "CudaCacheTrimResult",
    "CudaMemorySnapshot",
    "cuda_memory_snapshot",
    "should_trim_cuda_cache",
    "trim_cuda_cache",
    "trim_cuda_cache_if_needed",
]
