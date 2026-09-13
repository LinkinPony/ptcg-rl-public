"""Deterministic schedule partitioning shared by evaluation backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

_ItemT = TypeVar("_ItemT")


def partition_replica_items(
    items: Sequence[_ItemT],
    *,
    workers: int,
) -> tuple[tuple[_ItemT, ...], ...]:
    """Stripe immutable schedule rows evenly and exactly once."""
    if workers <= 0 or workers > len(items):
        raise ValueError("replica workers must cover at least one item each")
    return tuple(tuple(items[index::workers]) for index in range(workers))


def partition_replica_chunks(
    items: Sequence[_ItemT],
    *,
    chunk_size: int,
    wave_size: int | None = None,
) -> tuple[tuple[_ItemT, ...], ...]:
    """Build balanced, bounded work units in optional full replica waves.

    Striding by the number of chunks spreads the immutable schedule across all
    work units.  This keeps each unit close to a full native arena while
    avoiding a slow deck or seat stratum being concentrated in the final unit.
    Rounding the chunk count to full waves prevents a final cohort from leaving
    most expensive model replicas idle.
    """
    if chunk_size <= 0:
        raise ValueError("replica chunk size must be positive")
    if wave_size is not None and wave_size <= 0:
        raise ValueError("replica wave size must be positive")
    if not items:
        return ()
    chunk_count = (len(items) + chunk_size - 1) // chunk_size
    if wave_size is not None:
        chunk_count = min(
            len(items),
            ((chunk_count + wave_size - 1) // wave_size) * wave_size,
        )
    return partition_replica_items(items, workers=chunk_count)


__all__ = ["partition_replica_chunks", "partition_replica_items"]
