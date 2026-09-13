"""Root-perspective aggregation for one-action engine consequences."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from ptcg_rl.rl.amortized_policy_iteration.contracts import ConsequenceKind

Wdl = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class WorldConsequenceTarget:
    """One protected candidate-by-world target before particle aggregation."""

    root_id: str
    candidate_id: str
    particle_id: str
    kind: ConsequenceKind
    root_player: int
    leaf_player: int | None = None
    engine_result: int | None = None
    successor_actor_wdl: Wdl | None = None
    sampling_weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.root_id or not self.candidate_id or not self.particle_id:
            raise ValueError("consequence identities must be non-empty")
        if self.root_player not in (0, 1):
            raise ValueError("root_player must be zero or one")
        if not math.isfinite(self.sampling_weight) or self.sampling_weight <= 0.0:
            raise ValueError("sampling_weight must be finite and positive")
        if self.kind == "terminal":
            if self.engine_result not in (0, 1, 2):
                raise ValueError("terminal consequence requires winner/draw result")
            if self.successor_actor_wdl is not None:
                raise ValueError("terminal consequence cannot carry a bootstrap")
            return
        if self.kind == "infrastructure_error":
            if self.engine_result is not None or self.successor_actor_wdl is not None:
                raise ValueError("infrastructure errors cannot carry target values")
            return
        if self.engine_result is not None:
            raise ValueError("nonterminal consequence cannot carry engine_result")
        if self.leaf_player not in (0, 1):
            raise ValueError("nonterminal consequence requires leaf_player")
        if self.successor_actor_wdl is None:
            raise ValueError("nonterminal consequence requires successor W/D/L")
        validate_wdl(self.successor_actor_wdl)
        if self.kind == "same_seat" and self.leaf_player != self.root_player:
            raise ValueError("same-seat successor has another leaf player")
        if self.kind == "handoff" and self.leaf_player == self.root_player:
            raise ValueError("handoff successor did not change actor")


@dataclass(frozen=True, slots=True)
class AggregatedActionTarget:
    """Student-visible W/D/L target with all particle identity removed."""

    root_id: str
    candidate_id: str
    root_wdl: Wdl
    valid_worlds: int
    omitted_worlds: int
    total_sampling_weight: float

    def __post_init__(self) -> None:
        if self.valid_worlds <= 0 or self.omitted_worlds < 0:
            raise ValueError("aggregated world counts are invalid")
        if not math.isfinite(self.total_sampling_weight) or (
            self.total_sampling_weight <= 0.0
        ):
            raise ValueError("aggregate sampling weight must be positive")
        validate_wdl(self.root_wdl)


def aggregate_world_consequences(
    rows: Iterable[WorldConsequenceTarget],
) -> tuple[AggregatedActionTarget, ...]:
    """Aggregate valid worlds by candidate; omit infrastructure failures."""
    grouped: dict[tuple[str, str], list[WorldConsequenceTarget]] = defaultdict(list)
    for row in rows:
        grouped[(row.root_id, row.candidate_id)].append(row)
    if not grouped:
        raise ValueError("counterfactual target rows must not be empty")

    targets = []
    for (root_id, candidate_id), group in sorted(grouped.items()):
        particle_ids = tuple(row.particle_id for row in group)
        if len(set(particle_ids)) != len(particle_ids):
            raise ValueError("candidate consequences repeat a particle identity")
        valid = [row for row in group if row.kind != "infrastructure_error"]
        if not valid:
            continue
        weighted = [0.0, 0.0, 0.0]
        total_weight = 0.0
        for row in valid:
            root_wdl = consequence_root_wdl(row)
            for outcome, probability in enumerate(root_wdl):
                weighted[outcome] += row.sampling_weight * probability
            total_weight += row.sampling_weight
        aggregate = tuple(value / total_weight for value in weighted)
        targets.append(
            AggregatedActionTarget(
                root_id=root_id,
                candidate_id=candidate_id,
                root_wdl=(aggregate[0], aggregate[1], aggregate[2]),
                valid_worlds=len(valid),
                omitted_worlds=len(group) - len(valid),
                total_sampling_weight=total_weight,
            )
        )
    return tuple(targets)


def consequence_root_wdl(row: WorldConsequenceTarget) -> Wdl:
    """Route an exact terminal or actual next-actor value to root perspective."""
    if row.kind == "infrastructure_error":
        raise ValueError("infrastructure errors emit no target row")
    if row.kind == "terminal":
        result = row.engine_result
        if result == 2:
            return (0.0, 1.0, 0.0)
        if result == row.root_player:
            return (0.0, 0.0, 1.0)
        return (1.0, 0.0, 0.0)
    successor = row.successor_actor_wdl
    if successor is None:
        raise ValueError("nonterminal consequence is missing successor W/D/L")
    if row.kind == "same_seat":
        return successor
    if row.kind == "handoff":
        return reverse_wdl_perspective(successor)
    raise ValueError(f"unsupported consequence kind: {row.kind}")


def reverse_wdl_perspective(distribution: Wdl) -> Wdl:
    """Swap loss/win while preserving draw for the other player."""
    validate_wdl(distribution)
    return (distribution[2], distribution[1], distribution[0])


def validate_wdl(distribution: Wdl, *, atol: float = 1.0e-6) -> None:
    """Validate one finite loss/draw/win probability triple."""
    if len(distribution) != 3:
        raise ValueError("W/D/L distribution must contain three probabilities")
    if any(not math.isfinite(value) or value < 0.0 for value in distribution):
        raise ValueError("W/D/L probabilities must be finite and non-negative")
    if not math.isclose(sum(distribution), 1.0, rel_tol=0.0, abs_tol=atol):
        raise ValueError("W/D/L probabilities must sum to one")


__all__ = [
    "AggregatedActionTarget",
    "Wdl",
    "WorldConsequenceTarget",
    "aggregate_world_consequences",
    "consequence_root_wdl",
    "reverse_wdl_perspective",
    "validate_wdl",
]
