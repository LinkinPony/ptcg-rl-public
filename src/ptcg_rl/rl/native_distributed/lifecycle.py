"""Process-level lifecycle signals for native collection workers."""

from __future__ import annotations

NATIVE_WORKER_RECYCLE_EXIT_CODE = 75


class NativeWorkerProcessRecycleError(RuntimeError):
    """Request a clean worker process image after quiescent teardown."""


__all__ = [
    "NATIVE_WORKER_RECYCLE_EXIT_CODE",
    "NativeWorkerProcessRecycleError",
]
