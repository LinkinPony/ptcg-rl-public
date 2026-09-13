"""Fixed role-budget targets for behavior-v4 historical opponent sampling."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.rl.opponent_pool._identity import (
    Sha256,
    canonical_fingerprint,
    normalize_sha256,
)
from ptcg_rl.rl.opponent_pool.adaptive import (
    AdaptiveCellScore,
    AdaptiveTargetWeight,
)
from ptcg_rl.rl.opponent_pool.models import CandidateSeat, QuotaStratum

RoleBudgetName = Literal[
    "protected",
    "recent",
    "counter",
    "frontier",
    "recovery",
]

ROLE_ORDER: tuple[RoleBudgetName, ...] = (
    "protected",
    "recent",
    "counter",
    "frontier",
    "recovery",
)

# These are policy semantics, not a search space. Protected and recent are
# explicit safety budgets; the residual keeps hard-PFSP, frontier-PFSP, and
# forgotten/uncertain/stale recovery as separate reasons for sampling.
ROLE_TARGET_SHARES: dict[RoleBudgetName, float] = {
    "protected": 0.25,
    "recent": 0.15,
    "counter": 0.30,
    "frontier": 0.15,
    "recovery": 0.15,
}

_ROLE_BY_STRATUM: dict[QuotaStratum, RoleBudgetName] = {
    "protected": "protected",
    "recent": "recent",
    "counter_frontier": "counter",
    "age_diverse": "frontier",
    "probe_reentry": "recovery",
}


class RoleBudgetAllocationSnapshot(BaseModel):
    """Immutable targets attached to one pending role-budget quota plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    window_sequence: int = Field(ge=0)
    candidate_target_shares: dict[Sha256, float]
    target_weights: tuple[AdaptiveTargetWeight, ...]
    role_target_mass: dict[RoleBudgetName, float]
    evidence_cells: int = Field(ge=0)
    low_evidence_cells: int = Field(ge=0)

    @field_validator("candidate_target_shares")
    @classmethod
    def valid_candidate_simplex(
        cls,
        value: dict[str, float],
    ) -> dict[str, float]:
        """Validate candidate identities and their declared target simplex."""
        normalized = {normalize_sha256(key): item for key, item in value.items()}
        if len(normalized) != len(value) or any(
            not math.isfinite(item) or item <= 0.0 for item in normalized.values()
        ):
            raise ValueError("role-budget candidate targets must be positive")
        if not math.isclose(sum(normalized.values()), 1.0, abs_tol=1e-9):
            raise ValueError("role-budget candidate targets must sum to one")
        return normalized

    @field_validator("role_target_mass")
    @classmethod
    def fixed_role_budget(
        cls,
        value: dict[RoleBudgetName, float],
    ) -> dict[RoleBudgetName, float]:
        """Reject checkpoints that silently changed the fixed allocation policy."""
        if set(value) != set(ROLE_ORDER) or any(
            not math.isclose(value[name], ROLE_TARGET_SHARES[name], abs_tol=1e-12)
            for name in ROLE_ORDER
        ):
            raise ValueError("role-budget target mass differs from fixed policy")
        return value

    @model_validator(mode="after")
    def targets_are_canonical(self) -> Self:
        """Keep route targets unique, canonical, and normalized per seat."""
        keys = tuple(item.matchup_key for item in self.target_weights)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("role-budget target weights must be canonical")
        totals: defaultdict[tuple[str, int], float] = defaultdict(float)
        for item in self.target_weights:
            totals[(item.candidate_deck_digest, item.candidate_seat)] += item.share
        if not totals or any(
            not math.isclose(total, 1.0, abs_tol=1e-9) for total in totals.values()
        ):
            raise ValueError("role-budget route targets must sum to one per seat")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the fixed-policy target snapshot identity."""
        return canonical_fingerprint(
            "role-budget-opponent-allocation-snapshot-v1",
            self.model_dump(mode="json"),
        )


def role_for_stratum(stratum: QuotaStratum) -> RoleBudgetName:
    """Return the explicit allocation reason represented by a V4 stratum."""
    try:
        return _ROLE_BY_STRATUM[stratum]
    except KeyError as exc:
        raise ValueError(f"stratum {stratum!r} has no role-budget meaning") from exc


def role_priority(score: AdaptiveCellScore, role: RoleBudgetName) -> float:
    """Return a positive within-artifact route priority for one explicit role."""
    if role in {"protected", "recent"}:
        # Floors are enforced at the artifact level. Uniform route coverage
        # inside those artifacts avoids turning a safety budget into PFSP again.
        value = 1.0
    elif role == "counter":
        value = score.components["counter"]
    elif role == "frontier":
        value = score.components["frontier"]
    else:
        value = max(
            score.components["rehearsal"],
            math.sqrt(score.components["probe"] * score.components["staleness"]),
        )
    return score.base_weight * max(value, 1e-9)


def hierarchical_role_targets(
    scores: Sequence[AdaptiveCellScore],
    *,
    artifact_roles: Mapping[str, RoleBudgetName],
) -> tuple[AdaptiveTargetWeight, ...]:
    """Allocate role, then artifact, then route without route-count bias."""
    if not scores:
        raise ValueError("role-budget allocation requires active matchup scores")
    unknown_artifacts = {item.identity.artifact_id for item in scores} - set(
        artifact_roles
    )
    if unknown_artifacts:
        raise ValueError("role-budget scores contain unclassified artifacts")

    grouped: defaultdict[
        tuple[str, CandidateSeat],
        list[AdaptiveCellScore],
    ] = defaultdict(list)
    for score in scores:
        grouped[
            (
                score.identity.candidate_deck_digest,
                score.identity.candidate_seat,
            )
        ].append(score)

    result: list[AdaptiveTargetWeight] = []
    for candidate_cell, cell_scores in sorted(grouped.items()):
        by_role: defaultdict[
            RoleBudgetName,
            defaultdict[str, list[AdaptiveCellScore]],
        ] = defaultdict(lambda: defaultdict(list))
        for score in cell_scores:
            role = artifact_roles[score.identity.artifact_id]
            by_role[role][score.identity.artifact_id].append(score)
        if set(by_role) != set(ROLE_ORDER):
            raise ValueError("active role-budget revision lacks a required role")

        for role in ROLE_ORDER:
            artifacts = by_role[role]
            artifact_share = ROLE_TARGET_SHARES[role] / float(len(artifacts))
            for _artifact_id, route_scores in sorted(artifacts.items()):
                priorities = {
                    item.identity.matchup_key: role_priority(item, role)
                    for item in route_scores
                }
                total = sum(priorities.values())
                result.extend(
                    AdaptiveTargetWeight(
                        candidate_deck_digest=candidate_cell[0],
                        route_id=key[1],
                        candidate_seat=candidate_cell[1],
                        share=artifact_share * priority / total,
                    )
                    for key, priority in sorted(priorities.items())
                )
    return tuple(sorted(result, key=lambda item: item.matchup_key))


__all__ = [
    "ROLE_ORDER",
    "ROLE_TARGET_SHARES",
    "RoleBudgetAllocationSnapshot",
    "RoleBudgetName",
    "hierarchical_role_targets",
    "role_for_stratum",
    "role_priority",
]
