"""Fixed-memory stratified reservoirs for replay audit sampling."""

from __future__ import annotations

import hashlib
import heapq
import math
import struct
from collections.abc import Callable, Sequence
from typing import Generic, Protocol, TypeVar

_ItemT = TypeVar("_ItemT")
_STRATUM_DOMAIN = b"ptcg-rl/candidate-regret/stratum/v1\x00"
_ROOT_DOMAIN = b"ptcg-rl/candidate-regret/root/v1\x00"


class _Digest(Protocol):
    def update(self, value: bytes) -> None:
        """Add bytes to an incremental digest."""


class BoundedStratifiedReservoir(Generic[_ItemT]):
    """Keep low-hash roots inside a bounded low-hash set of strata."""

    def __init__(
        self,
        *,
        max_strata: int,
        roots_per_stratum: int,
        seed: str,
    ) -> None:
        if max_strata <= 0 or roots_per_stratum <= 0:
            raise ValueError("reservoir bounds must be positive")
        self._max_strata = max_strata
        self._roots_per_stratum = roots_per_stratum
        self._seed = seed.encode("utf-8")
        self._stratum_ranks: dict[str, int] = {}
        self._roots: dict[str, list[tuple[int, str, str, _ItemT]]] = {}
        self._item_keys: dict[str, set[str]] = {}

    def add(
        self,
        item: _ItemT,
        *,
        stratum_key: str,
        root_key: str,
        item_key: str,
    ) -> None:
        """Offer one item without allowing strata or rows to grow unbounded."""
        if stratum_key not in self._stratum_ranks:
            rank = self._rank(_STRATUM_DOMAIN, stratum_key)
            if len(self._stratum_ranks) >= self._max_strata:
                worst_key = max(
                    self._stratum_ranks,
                    key=lambda key: (self._stratum_ranks[key], key),
                )
                if (rank, stratum_key) >= (
                    self._stratum_ranks[worst_key],
                    worst_key,
                ):
                    return
                del self._stratum_ranks[worst_key]
                del self._roots[worst_key]
                del self._item_keys[worst_key]
            self._stratum_ranks[stratum_key] = rank
            self._roots[stratum_key] = []
            self._item_keys[stratum_key] = set()
        keys = self._item_keys[stratum_key]
        if item_key in keys:
            return
        heap = self._roots[stratum_key]
        root_rank = self._rank(_ROOT_DOMAIN, root_key)
        entry = (-root_rank, item_key, root_key, item)
        if len(heap) < self._roots_per_stratum:
            heapq.heappush(heap, entry)
            keys.add(item_key)
            return
        worst_rank = -heap[0][0]
        worst_key = heap[0][1]
        if (root_rank, item_key) < (worst_rank, worst_key):
            removed = heapq.heapreplace(heap, entry)
            keys.remove(removed[1])
            keys.add(item_key)

    def retained(self, *, max_roots: int) -> tuple[_ItemT, ...]:
        """Return a deterministic globally bounded union."""
        entries = [entry for heap in self._roots.values() for entry in heap]
        return tuple(
            entry[3]
            for entry in sorted(
                entries,
                key=lambda entry: (
                    self._rank(_ROOT_DOMAIN, entry[2]),
                    entry[1],
                ),
            )[:max_roots]
        )

    def _rank(self, domain: bytes, value: str) -> int:
        digest = hashlib.sha256(domain)
        _update_framed(digest, self._seed)
        _update_framed(digest, value.encode("utf-8"))
        return int.from_bytes(digest.digest()[:16], "big")


class DistinctSketch:
    """Fixed-memory linear-counting estimate for scan composition telemetry."""

    def __init__(self, buckets: int = 65_536) -> None:
        if buckets <= 0:
            raise ValueError("distinct sketch bucket count must be positive")
        self._bits = bytearray(buckets)
        self._occupied = 0

    def add(self, fingerprint: str) -> None:
        """Set one deterministic bucket without retaining the identity."""
        index = int(fingerprint[:16], 16) % len(self._bits)
        if self._bits[index] == 0:
            self._bits[index] = 1
            self._occupied += 1

    @property
    def estimate(self) -> int:
        """Return the standard fixed-memory linear-counting estimate."""
        empty = len(self._bits) - self._occupied
        if empty <= 0:
            return len(self._bits) * 10
        return int(round(-len(self._bits) * math.log(empty / len(self._bits))))


def balanced_union(
    first: Sequence[_ItemT],
    second: Sequence[_ItemT],
    *,
    max_items: int,
    identity: Callable[[_ItemT], str],
) -> tuple[_ItemT, ...]:
    """Interleave two stratified samples before global truncation."""
    output: list[_ItemT] = []
    seen: set[str] = set()
    for index in range(max(len(first), len(second))):
        for items in (first, second):
            if index >= len(items):
                continue
            item = items[index]
            key = identity(item)
            if key in seen:
                continue
            seen.add(key)
            output.append(item)
            if len(output) >= max_items:
                return tuple(output)
    return tuple(output)


def _update_framed(digest: _Digest, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


__all__ = ["BoundedStratifiedReservoir", "DistinctSketch", "balanced_union"]
