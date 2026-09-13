"""Deterministic low-discrepancy schedules for curriculum assignments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Generic, TypeVar

_T = TypeVar("_T")
_PHASE_DENOMINATOR = 1 << 64


@dataclass(frozen=True)
class WeightedCycleItem(Generic[_T]):
    """One uniquely keyed value and its positive target mass."""

    key: str
    value: _T
    weight: float


class DeterministicWeightedCycle(Generic[_T]):
    """Serve exact cycle quotas in a deterministic low-discrepancy order.

    Largest-remainder apportionment fixes the long-run marginal distribution
    within every cycle. Each cell's quota is then spread across the cycle using
    a keyed phase, avoiding both long homogeneous cohorts and random clumping.
    """

    def __init__(
        self,
        items: tuple[WeightedCycleItem[_T], ...],
        *,
        cycle_size: int,
        seed: int,
    ) -> None:
        """Validate the target population and precompute exact quotas."""
        if not items:
            raise ValueError("weighted cycle requires at least one item")
        if cycle_size <= 0:
            raise ValueError("weighted cycle size must be positive")
        keys = tuple(item.key for item in items)
        if any(not key for key in keys) or len(keys) != len(set(keys)):
            raise ValueError("weighted cycle item keys must be non-empty and unique")
        if any(not math.isfinite(item.weight) or item.weight <= 0.0 for item in items):
            raise ValueError("weighted cycle weights must be finite and positive")
        self._items = items
        self._cycle_size = int(cycle_size)
        self._seed = int(seed)
        self._quotas = _largest_remainder_quotas(
            items,
            cycle_size=self._cycle_size,
            seed=self._seed,
        )
        missing = tuple(
            item.key
            for item, quota in zip(items, self._quotas, strict=True)
            if quota <= 0
        )
        if missing:
            raise ValueError(
                "weighted cycle is too short to cover every positive cell; "
                f"increase cycle_size (missing={len(missing)})"
            )
        self._fingerprint = _schedule_fingerprint(
            items,
            cycle_size=self._cycle_size,
            seed=self._seed,
        )
        self._cached_cycle_index: int | None = None
        self._cached_order: tuple[int, ...] = ()

    @property
    def fingerprint(self) -> str:
        """Return the immutable identity of keys, masses, size, and seed."""
        return self._fingerprint

    @property
    def cycle_size(self) -> int:
        """Return the number of assignments in one exact-quota cycle."""
        return self._cycle_size

    @property
    def quotas(self) -> dict[str, int]:
        """Return exact per-cycle quotas keyed by assignment identity."""
        return {
            item.key: quota
            for item, quota in zip(self._items, self._quotas, strict=True)
        }

    def value_at(self, index: int) -> _T:
        """Return one deterministic value at a non-negative global index."""
        if index < 0:
            raise ValueError("weighted cycle index must be non-negative")
        cycle_index, offset = divmod(index, self._cycle_size)
        if cycle_index != self._cached_cycle_index:
            self._cached_order = self._build_cycle_order(cycle_index)
            self._cached_cycle_index = cycle_index
        return self._items[self._cached_order[offset]].value

    def _build_cycle_order(self, cycle_index: int) -> tuple[int, ...]:
        """Spread each quota through one cycle using deterministic phases."""
        scheduled: list[tuple[float, int, int]] = []
        for item_index, (item, quota) in enumerate(
            zip(self._items, self._quotas, strict=True)
        ):
            phase_bits = _keyed_uint64(
                self._seed,
                self._fingerprint,
                str(cycle_index),
                item.key,
            )
            phase = phase_bits / _PHASE_DENOMINATOR
            tie_break = _keyed_uint64(
                self._seed,
                "tie",
                self._fingerprint,
                str(cycle_index),
                item.key,
            )
            scheduled.extend(
                ((occurrence + phase) / quota, tie_break, item_index)
                for occurrence in range(quota)
            )
        scheduled.sort()
        if len(scheduled) != self._cycle_size:
            raise RuntimeError("weighted cycle quota construction lost assignments")
        return tuple(item_index for _position, _tie, item_index in scheduled)


def automatic_cycle_size(items: tuple[WeightedCycleItem[_T], ...]) -> int:
    """Choose a power-of-two cycle large enough to cover every target cell."""
    if not items:
        raise ValueError("automatic cycle size requires at least one item")
    total = sum(item.weight for item in items)
    minimum_probability = min(item.weight / total for item in items)
    required = max(len(items) * 4, math.ceil(1.0 / minimum_probability))
    return 1 << max(0, required - 1).bit_length()


def _largest_remainder_quotas(
    items: tuple[WeightedCycleItem[_T], ...],
    *,
    cycle_size: int,
    seed: int,
) -> tuple[int, ...]:
    """Apportion an integer cycle with deterministic remainder tie breaks."""
    total = sum(item.weight for item in items)
    scaled = tuple(item.weight * cycle_size / total for item in items)
    quotas = [math.floor(value) for value in scaled]
    remaining = cycle_size - sum(quotas)
    remainder_order = sorted(
        range(len(items)),
        key=lambda index: (
            -(scaled[index] - quotas[index]),
            _keyed_uint64(seed, "remainder", items[index].key),
        ),
    )
    for index in remainder_order[:remaining]:
        quotas[index] += 1
    return tuple(quotas)


def _schedule_fingerprint(
    items: tuple[WeightedCycleItem[_T], ...],
    *,
    cycle_size: int,
    seed: int,
) -> str:
    """Hash the normalized schedule without serializing assignment payloads."""
    total = sum(item.weight for item in items)
    digest = hashlib.sha256()
    digest.update(b"ptcg-rl/curriculum-weighted-cycle/v1\0")
    digest.update(str(cycle_size).encode("ascii"))
    digest.update(b"\0")
    digest.update(str(seed).encode("ascii"))
    for item in items:
        digest.update(b"\0")
        digest.update(item.key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(format(item.weight / total, ".17g").encode("ascii"))
    return digest.hexdigest()


def _keyed_uint64(seed: int, *parts: str) -> int:
    """Return a stable unsigned phase/tie value for arbitrary string parts."""
    digest = hashlib.sha256()
    digest.update(str(seed).encode("ascii"))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big", signed=False)
