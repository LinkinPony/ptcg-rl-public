"""Capacity-aware V3 exposure-wave geometry selection."""

from __future__ import annotations

import math
from collections.abc import Sequence
from fractions import Fraction
from typing import Protocol


class ExposureCapacityTier(Protocol):
    """Structural capacity fields shared by config and wire tiers."""

    @property
    def concurrent_games(self) -> int:
        """Return the number of assignments in the tier."""
        ...

    @property
    def native_arena_capacity(self) -> int:
        """Return simultaneously live native arena lanes."""
        ...

    @property
    def estimated_trainable_decisions(self) -> int:
        """Return the scheduler credit for one complete tier."""
        ...


def select_v3_exposure_games(
    inventories: Sequence[Sequence[ExposureCapacityTier]],
    *,
    target_trainable_decisions: int,
    required_artifact_count: int,
    topology_worker_count: int,
) -> int:
    """Choose a common, full-arena single-wave exposure geometry.

    The first V3 wave uses this tier on every worker whenever frozen artifact
    coverage is required.  A tier is budget-feasible when complete coverage
    plus first-wave filler stays within one tier credit of the target.  Within
    that bound, GPU arena fill is preferred over minimizing assigned games;
    ties prefer fewer engine waves and then the smaller shard.
    """
    if (
        not inventories
        or any(not inventory for inventory in inventories)
        or target_trainable_decisions <= 0
        or required_artifact_count <= 0
        or topology_worker_count <= 0
    ):
        raise ValueError("native V3 exposure scheduling inputs are invalid")
    common_games = {
        tier.concurrent_games for tier in inventories[0] if tier.concurrent_games > 0
    }
    for inventory in inventories[1:]:
        common_games &= {
            tier.concurrent_games for tier in inventory if tier.concurrent_games > 0
        }
    if not common_games:
        raise ValueError("native V3 workers share no exposure tier geometry")

    maximum_arenas = tuple(
        max(tier.native_arena_capacity for tier in inventory)
        for inventory in inventories
    )
    required_shards = max(required_artifact_count, topology_worker_count)
    candidates: list[tuple[bool, Fraction, int, int, int]] = []
    for concurrent_games in sorted(common_games):
        selected = tuple(
            max(
                (
                    tier
                    for tier in inventory
                    if tier.concurrent_games == concurrent_games
                ),
                key=lambda tier: (
                    tier.native_arena_capacity,
                    -tier.estimated_trainable_decisions,
                ),
            )
            for inventory in inventories
        )
        minimum_arena_fill = min(
            Fraction(tier.native_arena_capacity, maximum_arena)
            for tier, maximum_arena in zip(
                selected,
                maximum_arenas,
                strict=True,
            )
        )
        maximum_engine_waves = max(
            math.ceil(tier.concurrent_games / tier.native_arena_capacity)
            for tier in selected
        )
        maximum_credit = max(tier.estimated_trainable_decisions for tier in selected)
        projected_credit = required_shards * maximum_credit
        budget_feasible = (
            projected_credit <= target_trainable_decisions + maximum_credit
        )
        candidates.append(
            (
                budget_feasible,
                minimum_arena_fill,
                maximum_engine_waves,
                projected_credit,
                concurrent_games,
            )
        )

    feasible = tuple(candidate for candidate in candidates if candidate[0])
    if feasible:
        single_wave = tuple(candidate for candidate in feasible if candidate[2] == 1)
        preferred = single_wave or feasible
        return max(
            preferred,
            key=lambda candidate: (
                candidate[1],
                -candidate[2],
                -candidate[4],
            ),
        )[4]
    return min(
        candidates,
        key=lambda candidate: (
            candidate[3],
            -candidate[1],
            candidate[2],
            candidate[4],
        ),
    )[4]
