"""Deterministic artifact-coherent native arena planning."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from ptcg_rl.rl.stateless_collection import StatelessAssignedGame

NativeRouteKind = Literal[
    "current",
    "past_self",
    "historical",
    "scripted",
]

_ENGINE_SEED_DOMAIN = b"ptcg-rl/native-engine-seed/v1\x00"
_SCRIPTED_SEED_DOMAIN = b"ptcg-rl/native-scripted-seed/v1\x00"
_GENERATOR_SEED_DOMAIN = b"ptcg-rl/native-route-generator/v1\x00"
_SHA256_CHARS = frozenset("0123456789abcdef")


@dataclass(frozen=True, order=True, slots=True)
class NativeRouteKey:
    """One immutable policy runtime that may share inference batches."""

    kind: NativeRouteKind
    artifact_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256)


@dataclass(frozen=True, order=True, slots=True)
class NativeArenaKey:
    """Complete runtime identity that is safe to share in one engine arena."""

    kind: NativeRouteKind
    artifact_sha256: str
    policy_fingerprint: str
    input_contract_fingerprint: str = ""
    exact_registry_fingerprint: str = ""
    runtime_id: str = ""

    def __post_init__(self) -> None:
        _require_sha256(self.artifact_sha256)
        _require_sha256(self.policy_fingerprint)
        for value in (
            self.input_contract_fingerprint,
            self.exact_registry_fingerprint,
        ):
            if value:
                _require_sha256(value)
        if self.kind == "scripted" and not self.runtime_id:
            raise ValueError("scripted native arena requires a runtime ID")


@dataclass(frozen=True, slots=True)
class NativeArenaCohort:
    """One lease-ordered artifact arena with a private pending queue."""

    route_key: NativeArenaKey
    assignment_indices: tuple[int, ...]
    capacity: int
    shard_index: int = 0

    def __post_init__(self) -> None:
        if (
            not self.assignment_indices
            or len(set(self.assignment_indices)) != len(self.assignment_indices)
            or any(index < 0 for index in self.assignment_indices)
        ):
            raise ValueError("native arena assignment indices are invalid")
        if not 1 <= self.capacity <= len(self.assignment_indices):
            raise ValueError("native arena live capacity is invalid")
        if self.shard_index < 0:
            raise ValueError("native arena index must be non-negative")


def plan_native_artifact_arenas(
    assignments: Sequence[StatelessAssignedGame],
    *,
    total_capacity: int,
    group_keys: Sequence[NativeArenaKey],
) -> tuple[NativeArenaCohort, ...]:
    """Create exactly one live arena per immutable opponent runtime.

    ``total_capacity`` is a window-wide live-slot budget, not a per-artifact
    shard size.  Extra assignments remain in their artifact arena's pending
    queue and refill terminal slots in lease order.
    """
    if not assignments:
        raise ValueError("native arena planning requires assignments")
    if total_capacity <= 0:
        raise ValueError("native arena capacity must be positive")
    assignment_count = len(assignments)
    keyed_groups = _stable_assignment_groups(
        assignment_count,
        group_keys=group_keys,
    )
    groups = tuple(indices for _key, indices in keyed_groups)
    capacity_budget = min(total_capacity, assignment_count)
    if capacity_budget < len(groups):
        raise ValueError(
            "native arena capacity cannot give every artifact one live slot"
        )
    capacities = [1] * len(groups)
    remaining = capacity_budget - len(groups)
    while remaining:
        eligible = tuple(
            index
            for index, capacity in enumerate(capacities)
            if capacity < len(groups[index])
        )
        if not eligible:
            break
        selected = max(
            eligible,
            key=lambda index: (
                len(groups[index]) / capacities[index],
                -index,
            ),
        )
        capacities[selected] += 1
        remaining -= 1
    return tuple(
        NativeArenaCohort(
            route_key=key,
            assignment_indices=indices,
            capacity=capacity,
            shard_index=index,
        )
        for index, ((key, indices), capacity) in enumerate(
            zip(keyed_groups, capacities, strict=True)
        )
    )


def plan_native_artifact_banks(
    assignments: Sequence[StatelessAssignedGame],
    *,
    bank_count: int,
    group_keys: Sequence[NativeArenaKey],
) -> tuple[tuple[int, ...], ...]:
    """Partition leased assignments into artifact-coherent execution banks.

    The curriculum remains the sole owner of opponent sampling. This planner
    only reorders immutable assignment indices after leasing, so candidate
    deck, seat, and lane marginals cannot change. Past-self and historical
    routes are treated as frozen checkpoint artifacts, and no bank receives
    two frozen artifacts. When there are more banks than selected artifacts,
    one artifact may occupy multiple banks so every bank can remain useful.

    Non-frozen assignments fill the currently smallest bank in original lease
    order. Consequently the result is as balanced as unit-sized non-frozen
    work permits without mixing distinct frozen artifacts in one bank.
    """
    if not assignments:
        raise ValueError("native bank planning requires assignments")
    if bank_count <= 0:
        raise ValueError("native bank count must be positive")
    if bank_count > len(assignments):
        raise ValueError("native bank count cannot exceed assignments")
    if len(group_keys) != len(assignments):
        raise ValueError("native bank group keys differ from assignments")

    frozen_by_artifact: dict[str, list[int]] = {}
    non_frozen: list[int] = []
    for index, key in enumerate(group_keys):
        if key.kind in ("past_self", "historical"):
            frozen_by_artifact.setdefault(key.artifact_sha256, []).append(index)
        else:
            non_frozen.append(index)
    if len(frozen_by_artifact) > bank_count:
        raise ValueError(
            "frozen artifact count exceeds native execution banks"
        )

    banks: list[list[int]] = [[] for _index in range(bank_count)]
    ordered_frozen = sorted(
        frozen_by_artifact.items(),
        key=lambda item: (-len(item[1]), item[0]),
    )
    artifact_banks = {
        artifact: [bank_index]
        for bank_index, (artifact, _indices) in enumerate(ordered_frozen)
    }
    for bank_index in range(len(ordered_frozen), bank_count):
        if not ordered_frozen:
            break
        artifact = max(
            artifact_banks,
            key=lambda item: (
                len(frozen_by_artifact[item]) / len(artifact_banks[item]),
                item,
            ),
        )
        artifact_banks[artifact].append(bank_index)
    for artifact, indices in ordered_frozen:
        destinations = artifact_banks[artifact]
        for assignment_index in indices:
            bank_index = min(
                destinations,
                key=lambda index: (len(banks[index]), index),
            )
            banks[bank_index].append(assignment_index)
    for assignment_index in non_frozen:
        bank_index = min(
            range(bank_count),
            key=lambda index: (len(banks[index]), index),
        )
        banks[bank_index].append(assignment_index)

    planned = tuple(tuple(sorted(bank)) for bank in banks)
    if any(not bank for bank in planned):
        raise ValueError("native artifact affinity left an execution bank empty")
    flattened = tuple(index for bank in planned for index in bank)
    if len(flattened) != len(assignments) or set(flattened) != set(
        range(len(assignments))
    ):
        raise RuntimeError("native bank plan does not cover assignments exactly")
    return planned


def _stable_assignment_groups(
    assignment_count: int,
    *,
    group_keys: Sequence[NativeArenaKey],
) -> tuple[tuple[NativeArenaKey, tuple[int, ...]], ...]:
    """Group lease indices by hashable key without sorting identities."""
    if len(group_keys) != assignment_count:
        raise ValueError("native arena group keys differ from assignments")
    grouped: dict[NativeArenaKey, list[int]] = {}
    for index, key in enumerate(group_keys):
        grouped.setdefault(key, []).append(index)
    return tuple((key, tuple(indices)) for key, indices in grouped.items())


@dataclass(frozen=True, slots=True)
class NativeFrozenWavePlan:
    """One deterministic release decision for a single inference wave."""

    released: frozenset[NativeRouteKey]
    parked_rows: int
    threshold_releases: int
    deadline_releases: int
    forced_releases: int


class NativeFrozenBatchScheduler:
    """Deterministically batch frozen-opponent rows across inference waves.

    Frozen past-self and historical rows may wait in their live engine slots
    (their encoder state is frozen at the pending decision) until either the
    route accumulates ``min_rows`` ready rows across all arenas or the route
    has waited ``max_wait_waves`` waves. The decision is a pure function of
    the wave sequence, so the sampling stream stays deterministic for one
    fixed assignment lease; no thread timing enters the schedule.
    """

    def __init__(self, *, min_rows: int, max_wait_waves: int) -> None:
        if min_rows < 1:
            raise ValueError("frozen batch min rows must be positive")
        if max_wait_waves < 1:
            raise ValueError("frozen batch max wait waves must be positive")
        self.min_rows = int(min_rows)
        self.max_wait_waves = int(max_wait_waves)
        self._waited_waves: dict[NativeRouteKey, int] = {}

    def plan_wave(
        self,
        ready_rows: Mapping[NativeRouteKey, int],
        *,
        draining: bool,
        other_work_rows: int,
    ) -> NativeFrozenWavePlan:
        """Return the frozen routes that must serve during this wave."""
        if any(rows <= 0 for rows in ready_rows.values()):
            raise ValueError("frozen route ready rows must be positive")
        if other_work_rows < 0:
            raise ValueError("frozen scheduler work rows must be non-negative")
        released: set[NativeRouteKey] = set()
        threshold_releases = 0
        deadline_releases = 0
        forced_releases = 0
        waited = {
            route: self._waited_waves.get(route, 0) + 1 for route in ready_rows
        }
        for route, rows in ready_rows.items():
            # Draining stops admitting new games, but it must not turn every
            # frozen route into a per-wave microbatch. The regular threshold
            # and deadline still bound batching latency. Once no other rows can
            # advance, the forced release below guarantees forward progress.
            if rows >= self.min_rows:
                released.add(route)
                threshold_releases += 1
            elif waited[route] >= self.max_wait_waves:
                released.add(route)
                deadline_releases += 1
        if ready_rows and not released and other_work_rows == 0:
            # Every live row is waiting on a frozen route; release the
            # largest queue so the wave always advances at least one arena.
            forced = max(ready_rows, key=lambda route: (ready_rows[route], route))
            released.add(forced)
            forced_releases += 1
        self._waited_waves = {
            route: waves
            for route, waves in waited.items()
            if route not in released
        }
        parked_rows = sum(
            rows for route, rows in ready_rows.items() if route not in released
        )
        return NativeFrozenWavePlan(
            released=frozenset(released),
            parked_rows=parked_rows,
            threshold_releases=threshold_releases,
            deadline_releases=deadline_releases,
            forced_releases=forced_releases,
        )


def native_assignment_engine_seeds(
    base_seed: int,
    assignments: Sequence[StatelessAssignedGame],
) -> npt.NDArray[np.uint32]:
    """Derive engine seeds from immutable assignment identities, not slots."""
    return cast(
        npt.NDArray[np.uint32],
        np.asarray(
            [
                _stable_uint64(
                    _ENGINE_SEED_DOMAIN,
                    base_seed,
                    assignment.balance.assignment_id,
                    assignment.curriculum.assignment_id,
                )
                & int(np.iinfo(np.uint32).max)
                for assignment in assignments
            ],
            dtype=np.uint32,
        ),
    )


def native_assignment_scripted_seeds(
    base_seed: int,
    assignments: Sequence[StatelessAssignedGame],
) -> npt.NDArray[np.uint32]:
    """Derive one stable scripted RNG stream per immutable assignment."""
    return cast(
        npt.NDArray[np.uint32],
        np.asarray(
            [
                _stable_uint64(
                    _SCRIPTED_SEED_DOMAIN,
                    base_seed,
                    assignment.balance.assignment_id,
                    assignment.curriculum.assignment_id,
                )
                & int(np.iinfo(np.uint32).max)
                for assignment in assignments
            ],
            dtype=np.uint32,
        ),
    )


def native_route_generator_seed(
    base_seed: int,
    route: NativeRouteKey,
) -> int:
    """Return a stable independent torch RNG seed for one policy runtime."""
    return _stable_uint64(
        _GENERATOR_SEED_DOMAIN,
        base_seed,
        route.kind,
        route.artifact_sha256,
    )


def _stable_uint64(domain: bytes, base_seed: int, *parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(domain)
    digest.update(str(int(base_seed)).encode("ascii"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="little", signed=False)


def _require_sha256(value: str) -> None:
    if len(value) != 64 or not set(value) <= _SHA256_CHARS:
        raise ValueError("native route identity must be lowercase SHA-256")


__all__ = [
    "NativeArenaCohort",
    "NativeArenaKey",
    "NativeFrozenBatchScheduler",
    "NativeFrozenWavePlan",
    "NativeRouteKey",
    "NativeRouteKind",
    "native_assignment_engine_seeds",
    "native_assignment_scripted_seeds",
    "native_route_generator_seed",
    "plan_native_artifact_banks",
    "plan_native_artifact_arenas",
]
