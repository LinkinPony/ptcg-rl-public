"""CUDA allocator defaults for variable-shape RL workloads."""

from __future__ import annotations

import os
from collections.abc import MutableMapping

CUDA_ALLOCATOR_ENV = "PYTORCH_ALLOC_CONF"
DEFAULT_CUDA_ALLOCATOR_CONFIG = (
    "expandable_segments:True,garbage_collection_threshold:0.8"
)


def configure_cuda_allocator(
    environment: MutableMapping[str, str] | None = None,
) -> str:
    """Install the fragmentation-safe default unless the caller overrides it."""
    target = os.environ if environment is None else environment
    return target.setdefault(
        CUDA_ALLOCATOR_ENV,
        DEFAULT_CUDA_ALLOCATOR_CONFIG,
    )


__all__ = [
    "CUDA_ALLOCATOR_ENV",
    "DEFAULT_CUDA_ALLOCATOR_CONFIG",
    "configure_cuda_allocator",
]
