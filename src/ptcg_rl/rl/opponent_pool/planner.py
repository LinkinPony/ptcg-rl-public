"""Pure capacity-aware planning that emits executable assignments directly."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ptcg_rl.rl.opponent_pool._identity import Sha256, canonical_fingerprint
from ptcg_rl.rl.opponent_pool.models import (
    STRATUM_ORDER,
    CandidateSeat,
    OpponentArtifact,
    OpponentRoute,
    PoolEntry,
    QuotaStratum,
)
from ptcg_rl.rl.opponent_pool.state import LeagueState

_MAX_SELECTION_PRODUCTS = 100_000
_MatchupKey = tuple[str, str, int]
_Key = TypeVar("_Key", bound=str | tuple[str, str, int])


class ArtifactCapacity(BaseModel):
    """Resident artifact budget for one planning window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    maximum_artifacts: int = Field(gt=0)
    maximum_wire_bf16_artifacts: int | None = Field(default=None, ge=0)
    maximum_legacy_resident_artifacts: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def subcapacities_fit_total(self) -> Self:
        """Bound optional runtime limits by total resident capacity."""
        for limit in (
            self.maximum_wire_bf16_artifacts,
            self.maximum_legacy_resident_artifacts,
        ):
            if limit is not None and limit > self.maximum_artifacts:
                raise ValueError("runtime capacity cannot exceed total capacity")
        return self


class StratumPolicy(BaseModel):
    """Artifact slots and target game share for one scheduling stratum."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stratum: QuotaStratum
    artifact_slots: int = Field(gt=0)
    target_fraction: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)


class PlannerPolicy(BaseModel):
    """Canonical scheduling policy for all active strata."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strata: tuple[StratumPolicy, ...]

    @model_validator(mode="after")
    def strata_are_complete_and_canonical(self) -> Self:
        """Require canonical unique strata and a complete game-share simplex."""
        if not self.strata:
            raise ValueError("planner policy needs at least one stratum")
        order = {value: index for index, value in enumerate(STRATUM_ORDER)}
        names = tuple(item.stratum for item in self.strata)
        if names != tuple(sorted(names, key=order.__getitem__)):
            raise ValueError("stratum policies must follow canonical order")
        if len(set(names)) != len(names):
            raise ValueError("stratum policies must be unique")
        if abs(sum(item.target_fraction for item in self.strata) - 1.0) > 1e-9:
            raise ValueError("stratum target fractions must sum to one")
        return self


class OpponentAssignment(BaseModel):
    """One executable game assignment within an immutable window plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    assignment_index: int = Field(ge=0)
    stratum: QuotaStratum
    artifact_id: Sha256
    route_id: Sha256
    candidate_deck_digest: Sha256
    candidate_seat: CandidateSeat

    @property
    def matchup_key(self) -> tuple[str, str, int]:
        """Return the normalized committed-stat key."""
        return (
            self.candidate_deck_digest,
            self.route_id,
            self.candidate_seat,
        )


class WindowPlan(BaseModel):
    """Assignments bound to exactly one revision and committed base state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: Sha256
    revision_fingerprint: Sha256
    base_state_fingerprint: Sha256
    window_sequence: int = Field(ge=0)
    assignments: tuple[OpponentAssignment, ...]
    selected_artifact_ids: tuple[Sha256, ...]
    required_exposure_artifact_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def content_is_canonical(self) -> Self:
        """Validate assignment indexing, selected sets, and content identity."""
        if not self.assignments:
            raise ValueError("window plan needs at least one assignment")
        indices = tuple(item.assignment_index for item in self.assignments)
        if indices != tuple(range(len(self.assignments))):
            raise ValueError("assignment indices must be contiguous")
        assigned = tuple(sorted({item.artifact_id for item in self.assignments}))
        if self.selected_artifact_ids != assigned:
            raise ValueError("selected artifacts must exactly match assignments")
        required = self.required_exposure_artifact_ids
        if required != tuple(sorted(set(required))):
            raise ValueError("required exposure artifacts must be sorted and unique")
        if not set(required).issubset(assigned):
            raise ValueError("required exposure artifacts must be selected")
        artifact_strata: dict[str, QuotaStratum] = {}
        for assignment in self.assignments:
            previous = artifact_strata.setdefault(
                assignment.artifact_id,
                assignment.stratum,
            )
            if previous != assignment.stratum:
                raise ValueError("one artifact cannot span plan strata")
        if self.plan_id != window_plan_fingerprint(self):
            raise ValueError("plan_id does not match plan content")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the validated plan identity."""
        return self.plan_id


class OpponentQuotaCell(BaseModel):
    """One aggregate matchup row expanded only when a shard is leased."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_index: int = Field(ge=0)
    stratum: QuotaStratum
    artifact_id: Sha256
    route_id: Sha256
    candidate_deck_digest: Sha256
    candidate_seat: CandidateSeat
    game_count: int = Field(gt=0)

    @property
    def matchup_key(self) -> tuple[str, str, int]:
        """Return the normalized committed-stat key."""
        return (
            self.candidate_deck_digest,
            self.route_id,
            self.candidate_seat,
        )


class QuotaWindowPlan(BaseModel):
    """Compact immutable window plan whose size depends on matchup cells."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: Sha256
    revision_fingerprint: Sha256
    base_state_fingerprint: Sha256
    window_sequence: int = Field(ge=0)
    cells: tuple[OpponentQuotaCell, ...]
    selected_artifact_ids: tuple[Sha256, ...]
    required_exposure_artifact_ids: tuple[Sha256, ...]

    @model_validator(mode="after")
    def content_is_canonical(self) -> Self:
        """Require canonical aggregate rows and a content-derived identity."""
        if not self.cells:
            raise ValueError("quota window plan needs at least one cell")
        if tuple(item.cell_index for item in self.cells) != tuple(
            range(len(self.cells))
        ):
            raise ValueError("quota cell indices must be contiguous")
        keys = tuple(
            (
                item.stratum,
                item.artifact_id,
                item.route_id,
                item.candidate_deck_digest,
                item.candidate_seat,
            )
            for item in self.cells
        )
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("quota cells must be canonical and unique")
        assigned = tuple(sorted({item.artifact_id for item in self.cells}))
        if self.selected_artifact_ids != assigned:
            raise ValueError("quota selected artifacts differ from its cells")
        required = self.required_exposure_artifact_ids
        if required != tuple(sorted(set(required))) or not set(required) <= set(
            assigned
        ):
            raise ValueError("quota required artifact exposure is invalid")
        if self.plan_id != quota_window_plan_fingerprint(self):
            raise ValueError("quota plan ID does not match its content")
        return self

    @property
    def fingerprint(self) -> str:
        """Return the immutable aggregate plan identity."""
        return self.plan_id


def quota_window_plan_fingerprint(plan: QuotaWindowPlan) -> str:
    """Fingerprint one compact plan without any per-game JSON."""
    return canonical_fingerprint(
        "quota-window-plan-v1",
        plan.model_dump(mode="json", exclude={"plan_id"}),
    )


def window_plan_fingerprint(plan: WindowPlan) -> str:
    """Fingerprint a complete executable window plan."""
    return canonical_fingerprint(
        "window-plan",
        plan.model_dump(mode="json", exclude={"plan_id"}),
    )


@dataclass(frozen=True)
class _Candidate:
    artifact: OpponentArtifact
    entry: PoolEntry
    learning_exposures: int
    exposure_debt: float
    windows_since_exposure: int
    priority: float


def _validated_priorities(
    supplied: Mapping[_Key, float] | None,
    known: set[_Key],
    kind: str,
) -> dict[_Key, float]:
    """Validate optional ephemeral priorities without persisting them."""
    priorities = dict(supplied or {})
    unknown = set(priorities) - known
    if unknown:
        raise ValueError(f"{kind} priorities contain unknown identities")
    if any(not math.isfinite(value) or value <= 0.0 for value in priorities.values()):
        raise ValueError(f"{kind} priorities must be positive and finite")
    return priorities


def _normalized_matchup_debt(
    key: _MatchupKey,
    *,
    historical_exposures: Mapping[_MatchupKey, int],
    planned_exposures: Mapping[_MatchupKey, int],
    priorities: Mapping[_MatchupKey, float],
) -> float:
    """Return total matchup exposure normalized by its target priority."""
    return (
        historical_exposures.get(key, 0) + planned_exposures.get(key, 0)
    ) / priorities.get(key, 1.0)


def _allocate_counts(
    total: int,
    weights: Mapping[_Key, float],
    minima: Mapping[_Key, int],
) -> dict[_Key, int]:
    """Allocate integer counts by deterministic weighted fair queuing."""
    counts = {key: minima.get(key, 0) for key in weights}
    if sum(counts.values()) > total:
        raise ValueError("allocation minima exceed the available games")
    remaining = total - sum(counts.values())
    for _ in range(remaining):
        chosen = min(
            weights,
            key=lambda key: (counts[key] / weights[key], key),
        )
        counts[chosen] += 1
    return counts


def _candidate_rank_key(
    candidate: _Candidate,
) -> tuple[float, float, float, str]:
    """Rank one candidate deterministically within its exclusive stratum."""
    return (
        -candidate.exposure_debt,
        -float(candidate.windows_since_exposure),
        -candidate.priority,
        candidate.artifact.artifact_id,
    )


def _runtime_capacity_allows(
    candidates: tuple[_Candidate, ...],
    capacity: ArtifactCapacity,
) -> bool:
    """Return whether a selected cohort fits all runtime budgets."""
    wire_count = sum(item.artifact.runtime_kind == "wire_bf16" for item in candidates)
    legacy_count = len(candidates) - wire_count
    return not (
        capacity.maximum_wire_bf16_artifacts is not None
        and wire_count > capacity.maximum_wire_bf16_artifacts
        or capacity.maximum_legacy_resident_artifacts is not None
        and legacy_count > capacity.maximum_legacy_resident_artifacts
    )


def _select_artifacts(
    state: LeagueState,
    capacity: ArtifactCapacity,
    policy: PlannerPolicy,
    priorities: Mapping[str, float],
    mandatory_artifact_ids: frozenset[str] = frozenset(),
) -> dict[QuotaStratum, tuple[_Candidate, ...]]:
    """Choose exclusive cohorts using soft sampling debt and runtime capacity."""
    entries_by_stratum = {
        stratum: tuple(
            entry for entry in state.revision.entries if entry.stratum == stratum
        )
        for stratum in STRATUM_ORDER
    }
    active_strata = {entry.stratum for entry in state.revision.entries}
    policy_strata = {item.stratum for item in policy.strata}
    if active_strata != policy_strata:
        raise ValueError("planner policy must exactly cover active strata")
    if sum(item.artifact_slots for item in policy.strata) > (
        capacity.maximum_artifacts
    ):
        raise ValueError("stratum slots exceed total artifact capacity")

    artifact_by_id = {
        artifact.artifact_id: artifact for artifact in state.revision.artifacts
    }
    if not mandatory_artifact_ids.issubset(artifact_by_id):
        raise ValueError("mandatory artifacts are absent from the active revision")
    choices: list[tuple[tuple[_Candidate, ...], ...]] = []
    ranks: list[dict[str, int]] = []
    product_size = 1
    for stratum_policy in policy.strata:
        entries = entries_by_stratum[stratum_policy.stratum]
        if len(entries) < stratum_policy.artifact_slots:
            raise ValueError("a stratum has fewer artifacts than required slots")
        exposure_counts = [
            state.artifact_learning_exposures(entry.artifact_id) for entry in entries
        ]
        mean_exposures = sum(exposure_counts) / len(exposure_counts)
        candidates: list[_Candidate] = []
        for entry, exposure_count in zip(entries, exposure_counts, strict=True):
            last_window = state.artifact_last_learning_window(entry.artifact_id)
            if last_window is None:
                gap = max(0, state.generation - entry.admission_generation)
            else:
                gap = state.next_window_sequence - last_window
            candidates.append(
                _Candidate(
                    artifact=artifact_by_id[entry.artifact_id],
                    entry=entry,
                    learning_exposures=exposure_count,
                    exposure_debt=mean_exposures - exposure_count,
                    windows_since_exposure=gap,
                    priority=priorities.get(entry.artifact_id, 1.0),
                )
            )
        ranked = sorted(
            candidates,
            key=_candidate_rank_key,
        )
        ranks.append(
            {item.artifact.artifact_id: index for index, item in enumerate(ranked)}
        )
        combinations = tuple(
            combination
            for combination in itertools.combinations(
                candidates,
                stratum_policy.artifact_slots,
            )
            if {
                artifact_id
                for artifact_id in mandatory_artifact_ids
                if entries_by_stratum[stratum_policy.stratum]
                and artifact_id
                in {
                    entry.artifact_id
                    for entry in entries_by_stratum[stratum_policy.stratum]
                }
            }.issubset({item.artifact.artifact_id for item in combination})
        )
        if not combinations:
            raise ValueError("no artifact combination includes mandatory artifacts")
        product_size *= len(combinations)
        if product_size > _MAX_SELECTION_PRODUCTS:
            raise ValueError("artifact selection search exceeds its safety bound")
        choices.append(combinations)

    best: tuple[int, tuple[str, ...], tuple[tuple[_Candidate, ...], ...]] | None = None
    for cohort_product in itertools.product(*choices):
        flattened = tuple(item for cohort in cohort_product for item in cohort)
        if not _runtime_capacity_allows(flattened, capacity):
            continue
        score = sum(
            rank[item.artifact.artifact_id]
            for rank, cohort in zip(ranks, cohort_product, strict=True)
            for item in cohort
        )
        identities = tuple(sorted(item.artifact.artifact_id for item in flattened))
        candidate_result = (score, identities, cohort_product)
        if best is None or candidate_result[:2] < best[:2]:
            best = candidate_result
    if best is None:
        raise ValueError("no artifact selection fits runtime capacity")
    return {
        item.stratum: tuple(
            sorted(cohort, key=lambda candidate: candidate.artifact.artifact_id)
        )
        for item, cohort in zip(policy.strata, best[2], strict=True)
    }


def plan_window(
    state: LeagueState,
    *,
    capacity: ArtifactCapacity,
    policy: PlannerPolicy,
    total_games: int,
    artifact_priorities: Mapping[str, float] | None = None,
    matchup_priorities: Mapping[tuple[str, str, int], float] | None = None,
) -> WindowPlan:
    """Create a deterministic executable plan from committed normalized state."""
    known_artifacts = {artifact.artifact_id for artifact in state.revision.artifacts}
    known_matchups = {matchup.key for matchup in state.revision.active_matchups}
    artifact_priority = _validated_priorities(
        artifact_priorities,
        known_artifacts,
        "artifact",
    )
    matchup_priority = _validated_priorities(
        matchup_priorities,
        known_matchups,
        "matchup",
    )
    selected = _select_artifacts(
        state,
        capacity,
        policy,
        artifact_priority,
    )
    selected_count = sum(len(items) for items in selected.values())
    if total_games < selected_count:
        raise ValueError("total games must expose every selected artifact")

    stratum_weights = {item.stratum: item.target_fraction for item in policy.strata}
    stratum_minima = {
        item.stratum: len(selected[item.stratum]) for item in policy.strata
    }
    stratum_games = _allocate_counts(
        total_games,
        stratum_weights,
        stratum_minima,
    )
    route_artifacts = {
        route.route_id: route.artifact_id for route in state.revision.routes
    }
    cells_by_artifact: dict[str, list[tuple[str, str, int]]] = {
        artifact_id: [] for artifact_id in known_artifacts
    }
    for matchup in state.revision.active_matchups:
        cells_by_artifact[route_artifacts[matchup.route_id]].append(matchup.key)
    stats = {stat.matchup.key: stat for stat in state.matchup_stats}
    assignments: list[OpponentAssignment] = []
    for stratum in STRATUM_ORDER:
        cohort = selected.get(stratum)
        if cohort is None:
            continue
        artifact_weights = {item.artifact.artifact_id: 1.0 for item in cohort}
        artifact_games = _allocate_counts(
            stratum_games[stratum],
            artifact_weights,
            dict.fromkeys(artifact_weights, 1),
        )
        for candidate in cohort:
            artifact_id = candidate.artifact.artifact_id
            cells = cells_by_artifact[artifact_id]
            planned = dict.fromkeys(cells, 0)
            for _ in range(artifact_games[artifact_id]):
                cell = min(
                    cells,
                    key=lambda key: (
                        (
                            (stats[key].learning_exposures if key in stats else 0)
                            + planned[key]
                        )
                        / matchup_priority.get(key, 1.0),
                        (
                            stats[key].last_learning_exposure_window
                            if key in stats
                            and stats[key].last_learning_exposure_window is not None
                            else -1
                        ),
                        key,
                    ),
                )
                planned[cell] += 1
                assignments.append(
                    OpponentAssignment(
                        assignment_index=len(assignments),
                        stratum=stratum,
                        artifact_id=artifact_id,
                        route_id=cell[1],
                        candidate_deck_digest=cell[0],
                        candidate_seat=cast(CandidateSeat, cell[2]),
                    )
                )

    selected_ids = tuple(
        sorted(
            item.artifact.artifact_id for cohort in selected.values() for item in cohort
        )
    )
    required_ids: tuple[Sha256, ...] = ()
    assignments_tuple = tuple(assignments)
    fingerprint_content = {
        "revision_fingerprint": state.revision.fingerprint,
        "base_state_fingerprint": state.fingerprint,
        "window_sequence": state.next_window_sequence,
        "assignments": [item.model_dump(mode="json") for item in assignments_tuple],
        "selected_artifact_ids": selected_ids,
        "required_exposure_artifact_ids": required_ids,
    }
    return WindowPlan(
        plan_id=canonical_fingerprint("window-plan", fingerprint_content),
        revision_fingerprint=state.revision.fingerprint,
        base_state_fingerprint=state.fingerprint,
        window_sequence=state.next_window_sequence,
        assignments=assignments_tuple,
        selected_artifact_ids=selected_ids,
        required_exposure_artifact_ids=required_ids,
    )


def plan_window_for_candidate_cells(
    state: LeagueState,
    *,
    capacity: ArtifactCapacity,
    policy: PlannerPolicy,
    candidate_cells: Sequence[tuple[str, CandidateSeat]],
    stratum_decision_coverage: Mapping[QuotaStratum, int],
    mandatory_artifact_ids: frozenset[str] = frozenset(),
    artifact_priorities: Mapping[str, float] | None = None,
    matchup_priorities: Mapping[tuple[str, str, int], float] | None = None,
) -> WindowPlan:
    """Plan opponents without changing scheduler-owned candidate deck/seat cells."""
    cells = tuple(candidate_cells)
    if not cells:
        raise ValueError("fixed-cell opponent plan requires candidate cells")
    known_artifacts = {artifact.artifact_id for artifact in state.revision.artifacts}
    known_matchups = {matchup.key for matchup in state.revision.active_matchups}
    active_candidate_cells = {
        (matchup.candidate_deck_digest, matchup.candidate_seat)
        for matchup in state.revision.active_matchups
    }
    if any(cell not in active_candidate_cells for cell in cells):
        raise ValueError("fixed candidate cell is absent from the active revision")
    artifact_priority = _validated_priorities(
        artifact_priorities,
        known_artifacts,
        "artifact",
    )
    matchup_priority = _validated_priorities(
        matchup_priorities,
        known_matchups,
        "matchup",
    )
    selected = _select_artifacts(
        state,
        capacity,
        policy,
        artifact_priority,
        mandatory_artifact_ids,
    )
    selected_count = sum(len(items) for items in selected.values())
    if len(cells) < selected_count:
        raise ValueError("candidate cells cannot expose every selected artifact")
    policy_by_stratum = {item.stratum: item for item in policy.strata}
    expected_strata = set(policy_by_stratum)
    if set(stratum_decision_coverage) != expected_strata:
        raise ValueError("stratum decision coverage differs from effective policy")
    if any(value < 0 for value in stratum_decision_coverage.values()):
        raise ValueError("stratum decision coverage cannot be negative")

    routes_by_artifact: dict[str, list[str]] = {
        artifact_id: [] for artifact_id in known_artifacts
    }
    for route in state.revision.routes:
        routes_by_artifact[route.artifact_id].append(route.route_id)
    historical_matchups = {
        stat.matchup.key: stat.learning_exposures for stat in state.matchup_stats
    }
    global_exposures = sum(stat.learning_exposures for stat in state.matchup_stats)
    global_decisions = sum(stat.trainable_decisions for stat in state.matchup_stats)
    projected_credit = (
        1.0
        if global_exposures == 0
        else max(1.0, global_decisions / float(global_exposures))
    )
    total_coverage = float(sum(stratum_decision_coverage.values()))
    planned_strata = dict.fromkeys(expected_strata, 0)
    planned_credits = dict.fromkeys(expected_strata, 0.0)
    planned_artifacts = dict.fromkeys(known_artifacts, 0)
    planned_matchups = dict.fromkeys(known_matchups, 0)
    assignments: list[OpponentAssignment] = []
    order = {value: index for index, value in enumerate(STRATUM_ORDER)}

    for position, (candidate_deck, candidate_seat) in enumerate(cells):
        remaining = len(cells) - position
        needed = {
            stratum: max(0, len(cohort) - planned_strata[stratum])
            for stratum, cohort in selected.items()
        }
        available_strata = tuple(selected)
        if sum(needed.values()) >= remaining:
            available_strata = tuple(
                stratum for stratum in available_strata if needed[stratum] > 0
            )
        projected_total = total_coverage + projected_credit * float(position + 1)
        stratum = max(
            available_strata,
            key=lambda item: (
                policy_by_stratum[item].target_fraction * projected_total
                - (float(stratum_decision_coverage[item]) + planned_credits[item]),
                -order[item],
            ),
        )
        cohort = selected[stratum]
        unplanned = tuple(
            candidate
            for candidate in cohort
            if planned_artifacts[candidate.artifact.artifact_id] == 0
        )
        artifact_candidate = min(
            unplanned or cohort,
            key=lambda item: (
                state.artifact_trainable_decisions(item.artifact.artifact_id)
                + planned_artifacts[item.artifact.artifact_id] * projected_credit,
                item.artifact.artifact_id,
            ),
        )
        artifact_id = artifact_candidate.artifact.artifact_id
        matchup_keys = tuple(
            (candidate_deck, route_id, int(candidate_seat))
            for route_id in routes_by_artifact[artifact_id]
        )
        route_key = min(
            matchup_keys,
            key=lambda key: (
                _normalized_matchup_debt(
                    key,
                    historical_exposures=historical_matchups,
                    planned_exposures=planned_matchups,
                    priorities=matchup_priority,
                ),
                key,
            ),
        )
        assignments.append(
            OpponentAssignment(
                assignment_index=position,
                stratum=stratum,
                artifact_id=artifact_id,
                route_id=route_key[1],
                candidate_deck_digest=candidate_deck,
                candidate_seat=candidate_seat,
            )
        )
        planned_strata[stratum] += 1
        planned_credits[stratum] += projected_credit
        planned_artifacts[artifact_id] += 1
        planned_matchups[route_key] += 1

    selected_ids = tuple(sorted({item.artifact_id for item in assignments}))
    expected_selected = {
        candidate.artifact.artifact_id
        for cohort in selected.values()
        for candidate in cohort
    }
    if set(selected_ids) != expected_selected:
        raise RuntimeError("fixed-cell plan failed to expose a selected artifact")
    required_ids: tuple[Sha256, ...] = ()
    fingerprint_content = {
        "revision_fingerprint": state.revision.fingerprint,
        "base_state_fingerprint": state.fingerprint,
        "window_sequence": state.next_window_sequence,
        "assignments": [item.model_dump(mode="json") for item in assignments],
        "selected_artifact_ids": selected_ids,
        "required_exposure_artifact_ids": required_ids,
    }
    return WindowPlan(
        plan_id=canonical_fingerprint("window-plan", fingerprint_content),
        revision_fingerprint=state.revision.fingerprint,
        base_state_fingerprint=state.fingerprint,
        window_sequence=state.next_window_sequence,
        assignments=tuple(assignments),
        selected_artifact_ids=selected_ids,
        required_exposure_artifact_ids=required_ids,
    )


def plan_quota_window_for_candidate_counts(
    state: LeagueState,
    *,
    capacity: ArtifactCapacity,
    policy: PlannerPolicy,
    candidate_cell_counts: Mapping[tuple[str, CandidateSeat], int],
    stratum_decision_coverage: Mapping[QuotaStratum, int],
    mandatory_artifact_ids: frozenset[str] = frozenset(),
    artifact_priorities: Mapping[str, float] | None = None,
    matchup_priorities: Mapping[tuple[str, str, int], float] | None = None,
) -> QuotaWindowPlan:
    """Plan aggregate matchup quotas without constructing per-game objects."""
    requested = {
        (deck, seat): int(count)
        for (deck, seat), count in candidate_cell_counts.items()
    }
    if not requested or any(count <= 0 for count in requested.values()):
        raise ValueError("quota candidate cells must have positive counts")
    known_artifacts = {item.artifact_id for item in state.revision.artifacts}
    known_matchups = {item.key for item in state.revision.active_matchups}
    active_candidate_cells = {
        (item[0], cast(CandidateSeat, item[2])) for item in known_matchups
    }
    if not set(requested) <= active_candidate_cells:
        raise ValueError("quota candidate cell is absent from the active revision")
    artifact_priority = _validated_priorities(
        artifact_priorities,
        known_artifacts,
        "artifact",
    )
    matchup_priority = _validated_priorities(
        matchup_priorities,
        known_matchups,
        "matchup",
    )
    selected = _select_artifacts(
        state,
        capacity,
        policy,
        artifact_priority,
        mandatory_artifact_ids,
    )
    selected_count = sum(len(items) for items in selected.values())
    total_games = sum(requested.values())
    if total_games < selected_count:
        raise ValueError("quota cells cannot expose every selected artifact")
    policy_by_stratum = {item.stratum: item for item in policy.strata}
    if set(stratum_decision_coverage) != set(policy_by_stratum):
        raise ValueError("quota stratum coverage differs from effective policy")
    if any(value < 0 for value in stratum_decision_coverage.values()):
        raise ValueError("quota stratum coverage cannot be negative")

    global_exposures = sum(item.learning_exposures for item in state.matchup_stats)
    global_decisions = sum(item.trainable_decisions for item in state.matchup_stats)
    projected_credit = (
        1.0
        if global_exposures == 0
        else max(1.0, global_decisions / float(global_exposures))
    )
    planned_strata: Counter[QuotaStratum] = Counter()
    stratum_order = {value: index for index, value in enumerate(STRATUM_ORDER)}
    for position in range(total_games):
        remaining = total_games - position
        missing = {
            stratum: max(0, len(cohort) - planned_strata[stratum])
            for stratum, cohort in selected.items()
        }
        available = tuple(selected)
        if sum(missing.values()) >= remaining:
            available = tuple(item for item in available if missing[item] > 0)
        projected_total = float(sum(stratum_decision_coverage.values())) + (
            projected_credit * float(position + 1)
        )
        chosen = max(
            available,
            key=lambda item: (
                policy_by_stratum[item].target_fraction * projected_total
                - (
                    float(stratum_decision_coverage[item])
                    + projected_credit * float(planned_strata[item])
                ),
                -stratum_order[item],
            ),
        )
        planned_strata[chosen] += 1

    artifact_counts: Counter[str] = Counter()
    artifact_strata: dict[str, QuotaStratum] = {}
    selected_artifact_ids = {
        candidate.artifact.artifact_id
        for cohort in selected.values()
        for candidate in cohort
    }
    artifact_base_decisions = {
        artifact_id: state.artifact_trainable_decisions(artifact_id)
        for artifact_id in selected_artifact_ids
    }
    for stratum, cohort in selected.items():
        for candidate in cohort:
            artifact_strata[candidate.artifact.artifact_id] = stratum
        for _index in range(planned_strata[stratum]):
            unexposed = tuple(
                item
                for item in cohort
                if artifact_counts[item.artifact.artifact_id] == 0
            )
            selected_candidate = min(
                unexposed or cohort,
                key=lambda item: (
                    artifact_base_decisions[item.artifact.artifact_id]
                    + projected_credit
                    * float(artifact_counts[item.artifact.artifact_id]),
                    item.artifact.artifact_id,
                ),
            )
            artifact_counts[selected_candidate.artifact.artifact_id] += 1

    route_artifact = {
        route.route_id: route.artifact_id for route in state.revision.routes
    }
    routes_by_cell: dict[tuple[str, str, CandidateSeat], list[str]] = {}
    for matchup in state.revision.active_matchups:
        key = (
            route_artifact[matchup.route_id],
            matchup.candidate_deck_digest,
            matchup.candidate_seat,
        )
        routes_by_cell.setdefault(key, []).append(matchup.route_id)

    remaining_cells = dict(requested)
    remaining_artifacts = dict(artifact_counts)
    issued_artifacts: Counter[str] = Counter()
    issued_cells: Counter[tuple[str, CandidateSeat]] = Counter()
    issued_matchups: Counter[tuple[str, str, int]] = Counter()
    historical_matchups = {
        item.matchup.key: item.learning_exposures for item in state.matchup_stats
    }
    quotas: Counter[tuple[QuotaStratum, str, str, str, CandidateSeat]] = Counter()
    for _index in range(total_games):
        artifact_id = min(
            (item for item, count in remaining_artifacts.items() if count > 0),
            key=lambda item: (
                issued_artifacts[item] / artifact_counts[item],
                item,
            ),
        )
        candidates = tuple(
            cell
            for cell, count in remaining_cells.items()
            if count > 0 and routes_by_cell.get((artifact_id, cell[0], cell[1]))
        )
        if not candidates:
            raise ValueError("quota artifact cannot serve remaining candidate cells")
        candidate_cell = min(
            candidates,
            key=lambda cell: (
                issued_cells[cell] / requested[cell],
                cell,
            ),
        )
        route_id = min(
            routes_by_cell[(artifact_id, candidate_cell[0], candidate_cell[1])],
            key=lambda candidate_route: (
                _normalized_matchup_debt(
                    (
                        candidate_cell[0],
                        candidate_route,
                        candidate_cell[1],
                    ),
                    historical_exposures=historical_matchups,
                    planned_exposures=issued_matchups,
                    priorities=matchup_priority,
                ),
                candidate_route,
            ),
        )
        matchup_key = (candidate_cell[0], route_id, int(candidate_cell[1]))
        quotas[
            (
                artifact_strata[artifact_id],
                artifact_id,
                route_id,
                candidate_cell[0],
                candidate_cell[1],
            )
        ] += 1
        remaining_artifacts[artifact_id] -= 1
        remaining_cells[candidate_cell] -= 1
        issued_artifacts[artifact_id] += 1
        issued_cells[candidate_cell] += 1
        issued_matchups[matchup_key] += 1

    if any(remaining_cells.values()) or any(remaining_artifacts.values()):
        raise RuntimeError("quota planner failed to conserve its requested games")
    ordered = sorted(quotas.items())
    cells = tuple(
        OpponentQuotaCell(
            cell_index=index,
            stratum=key[0],
            artifact_id=key[1],
            route_id=key[2],
            candidate_deck_digest=key[3],
            candidate_seat=key[4],
            game_count=count,
        )
        for index, (key, count) in enumerate(ordered)
    )
    selected_ids = tuple(sorted(artifact_counts))
    content = {
        "revision_fingerprint": state.revision.fingerprint,
        "base_state_fingerprint": state.fingerprint,
        "window_sequence": state.next_window_sequence,
        "cells": [item.model_dump(mode="json") for item in cells],
        "selected_artifact_ids": selected_ids,
        "required_exposure_artifact_ids": selected_ids,
    }
    return QuotaWindowPlan(
        plan_id=canonical_fingerprint("quota-window-plan-v1", content),
        revision_fingerprint=state.revision.fingerprint,
        base_state_fingerprint=state.fingerprint,
        window_sequence=state.next_window_sequence,
        cells=cells,
        selected_artifact_ids=selected_ids,
        required_exposure_artifact_ids=selected_ids,
    )


def plan_adaptive_quota_window_for_candidate_counts(
    state: LeagueState,
    *,
    capacity: ArtifactCapacity,
    candidate_cell_counts: Mapping[tuple[str, CandidateSeat], int],
    matchup_targets: Mapping[_MatchupKey, float],
    matchup_decision_mass: Mapping[_MatchupKey, float],
    expected_decisions_per_game: Mapping[_MatchupKey, float],
    minimum_artifact_games: int,
    matchup_game_batch_size: int = 1,
    matchup_last_learning_windows: Mapping[_MatchupKey, int | None] | None = None,
    matchup_coverage_windows: int | None = None,
) -> QuotaWindowPlan:
    """Plan one joint decision-mass target without fixed stratum quotas."""
    requested = {
        (deck, seat): int(count)
        for (deck, seat), count in candidate_cell_counts.items()
    }
    if not requested or any(count <= 0 for count in requested.values()):
        raise ValueError("adaptive quota candidate cells must be positive")
    if minimum_artifact_games <= 0:
        raise ValueError("adaptive artifact coverage must be positive")
    if matchup_game_batch_size <= 0:
        raise ValueError("adaptive matchup game batch size must be positive")
    artifacts = {item.artifact_id: item for item in state.revision.artifacts}
    entries = {item.artifact_id: item for item in state.revision.entries}
    routes = {item.route_id: item for item in state.revision.routes}
    active_matchups = {item.key for item in state.revision.active_matchups}
    active_candidate_cells = {
        (item[0], cast(CandidateSeat, item[2])) for item in active_matchups
    }
    if not set(requested) <= active_candidate_cells:
        raise ValueError("adaptive candidate cell is absent from the revision")
    if set(matchup_targets) != active_matchups:
        raise ValueError("adaptive targets must exactly cover active matchups")
    if set(matchup_decision_mass) != active_matchups:
        raise ValueError("adaptive decision mass must cover active matchups")
    if set(expected_decisions_per_game) != active_matchups:
        raise ValueError("adaptive decision credits must cover active matchups")
    if any(
        not math.isfinite(value) or value <= 0.0 for value in matchup_targets.values()
    ):
        raise ValueError("adaptive matchup targets must be positive and finite")
    if any(
        not math.isfinite(value) or value < 0.0
        for value in matchup_decision_mass.values()
    ):
        raise ValueError("adaptive matchup decision mass is invalid")
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in expected_decisions_per_game.values()
    ):
        raise ValueError("adaptive expected decision credits are invalid")
    sparse_execution = matchup_game_batch_size > 1
    if sparse_execution:
        if matchup_coverage_windows is None or matchup_coverage_windows <= 0:
            raise ValueError("sparse allocation requires a positive coverage window")
        if (
            matchup_last_learning_windows is None
            or set(matchup_last_learning_windows) != active_matchups
            or any(
                value is not None and (value < 0 or value > state.next_window_sequence)
                for value in matchup_last_learning_windows.values()
            )
        ):
            raise ValueError("sparse allocation learning clocks are invalid")
    elif (
        matchup_last_learning_windows is not None
        or matchup_coverage_windows is not None
    ):
        raise ValueError("dense allocation cannot declare sparse coverage state")
    selected_artifacts = tuple(sorted(artifacts))
    if len(selected_artifacts) > capacity.maximum_artifacts:
        raise ValueError("adaptive artifact set exceeds planning capacity")
    selected_candidates = tuple(
        _Candidate(
            artifact=artifact,
            entry=entries[artifact_id],
            learning_exposures=state.artifact_learning_exposures(artifact_id),
            exposure_debt=0.0,
            windows_since_exposure=0,
            priority=1.0,
        )
        for artifact_id, artifact in sorted(artifacts.items())
    )
    if not _runtime_capacity_allows(selected_candidates, capacity):
        raise ValueError("adaptive artifact set exceeds runtime subcapacity")
    total_games = sum(requested.values())
    if minimum_artifact_games * len(selected_artifacts) > total_games:
        raise ValueError("adaptive artifact coverage exceeds available games")

    if sparse_execution:
        return _plan_sparse_adaptive_quota_window(
            state,
            requested=requested,
            entries=entries,
            routes=routes,
            selected_artifacts=selected_artifacts,
            matchup_targets=matchup_targets,
            matchup_decision_mass=matchup_decision_mass,
            expected_decisions_per_game=expected_decisions_per_game,
            minimum_artifact_games=minimum_artifact_games,
            matchup_game_batch_size=matchup_game_batch_size,
            matchup_last_learning_windows=cast(
                Mapping[_MatchupKey, int | None],
                matchup_last_learning_windows,
            ),
            matchup_coverage_windows=cast(int, matchup_coverage_windows),
        )

    route_artifact = {route_id: route.artifact_id for route_id, route in routes.items()}
    routes_by_cell: defaultdict[tuple[str, CandidateSeat], list[str]] = defaultdict(
        list
    )
    routes_by_artifact_cell: defaultdict[tuple[str, str, CandidateSeat], list[str]] = (
        defaultdict(list)
    )
    for matchup in state.revision.active_matchups:
        candidate_cell = (
            matchup.candidate_deck_digest,
            matchup.candidate_seat,
        )
        routes_by_cell[candidate_cell].append(matchup.route_id)
        routes_by_artifact_cell[
            (
                route_artifact[matchup.route_id],
                matchup.candidate_deck_digest,
                matchup.candidate_seat,
            )
        ].append(matchup.route_id)

    remaining = dict(requested)
    issued_cells: Counter[tuple[str, CandidateSeat]] = Counter()
    planned_decisions: defaultdict[_MatchupKey, float] = defaultdict(float)
    quotas: Counter[tuple[QuotaStratum, str, str, str, CandidateSeat]] = Counter()

    def debt(key: _MatchupKey) -> float:
        return (
            matchup_decision_mass.get(key, 0.0) + planned_decisions[key]
        ) / matchup_targets[key]

    def issue(
        candidate_cell: tuple[str, CandidateSeat],
        route_id: str,
    ) -> None:
        artifact_id = route_artifact[route_id]
        key = (candidate_cell[0], route_id, int(candidate_cell[1]))
        quotas[
            (
                entries[artifact_id].stratum,
                artifact_id,
                route_id,
                candidate_cell[0],
                candidate_cell[1],
            )
        ] += 1
        remaining[candidate_cell] -= 1
        issued_cells[candidate_cell] += 1
        planned_decisions[key] += expected_decisions_per_game[key]

    for artifact_id in selected_artifacts:
        for _index in range(minimum_artifact_games):
            choices: list[tuple[float, float, tuple[str, CandidateSeat], str]] = []
            for candidate_cell, count in remaining.items():
                if count <= 0:
                    continue
                candidate_routes = routes_by_artifact_cell.get(
                    (artifact_id, candidate_cell[0], candidate_cell[1]),
                    (),
                )
                if not candidate_routes:
                    continue
                route_id = min(
                    candidate_routes,
                    key=lambda value: (
                        debt((candidate_cell[0], value, int(candidate_cell[1]))),
                        value,
                    ),
                )
                choices.append(
                    (
                        issued_cells[candidate_cell] / requested[candidate_cell],
                        debt((candidate_cell[0], route_id, int(candidate_cell[1]))),
                        candidate_cell,
                        route_id,
                    )
                )
            if not choices:
                raise ValueError("adaptive artifact cannot serve a candidate cell")
            _share, _debt, candidate_cell, route_id = min(choices)
            issue(candidate_cell, route_id)

    for candidate_cell in sorted(remaining):
        route_heap = [
            (
                debt((candidate_cell[0], route_id, int(candidate_cell[1]))),
                route_id,
            )
            for route_id in routes_by_cell[candidate_cell]
        ]
        heapq.heapify(route_heap)
        for _index in range(remaining[candidate_cell]):
            _route_debt, route_id = heapq.heappop(route_heap)
            issue(candidate_cell, route_id)
            heapq.heappush(
                route_heap,
                (
                    debt((candidate_cell[0], route_id, int(candidate_cell[1]))),
                    route_id,
                ),
            )

    if any(remaining.values()):
        raise RuntimeError("adaptive quota planner lost candidate games")
    ordered = sorted(quotas.items())
    cells = tuple(
        OpponentQuotaCell(
            cell_index=index,
            stratum=key[0],
            artifact_id=key[1],
            route_id=key[2],
            candidate_deck_digest=key[3],
            candidate_seat=key[4],
            game_count=count,
        )
        for index, (key, count) in enumerate(ordered)
    )
    content = {
        "revision_fingerprint": state.revision.fingerprint,
        "base_state_fingerprint": state.fingerprint,
        "window_sequence": state.next_window_sequence,
        "cells": [item.model_dump(mode="json") for item in cells],
        "selected_artifact_ids": selected_artifacts,
        "required_exposure_artifact_ids": selected_artifacts,
    }
    return QuotaWindowPlan(
        plan_id=canonical_fingerprint("quota-window-plan-v1", content),
        revision_fingerprint=state.revision.fingerprint,
        base_state_fingerprint=state.fingerprint,
        window_sequence=state.next_window_sequence,
        cells=cells,
        selected_artifact_ids=selected_artifacts,
        required_exposure_artifact_ids=selected_artifacts,
    )


def _plan_sparse_adaptive_quota_window(
    state: LeagueState,
    *,
    requested: Mapping[tuple[str, CandidateSeat], int],
    entries: Mapping[str, PoolEntry],
    routes: Mapping[str, OpponentRoute],
    selected_artifacts: tuple[str, ...],
    matchup_targets: Mapping[_MatchupKey, float],
    matchup_decision_mass: Mapping[_MatchupKey, float],
    expected_decisions_per_game: Mapping[_MatchupKey, float],
    minimum_artifact_games: int,
    matchup_game_batch_size: int,
    matchup_last_learning_windows: Mapping[_MatchupKey, int | None],
    matchup_coverage_windows: int,
) -> QuotaWindowPlan:
    """Plan sparse exact cells while preserving hierarchical decision budgets."""
    route_artifact = {route_id: route.artifact_id for route_id, route in routes.items()}
    routes_by_artifact_cell: defaultdict[tuple[str, str, CandidateSeat], list[str]] = (
        defaultdict(list)
    )
    artifacts_by_stratum: defaultdict[QuotaStratum, list[str]] = defaultdict(list)
    for artifact_id, entry in entries.items():
        artifacts_by_stratum[entry.stratum].append(artifact_id)
    for matchup in state.revision.active_matchups:
        candidate_cell = (
            matchup.candidate_deck_digest,
            matchup.candidate_seat,
        )
        routes_by_artifact_cell[
            (
                route_artifact[matchup.route_id],
                candidate_cell[0],
                candidate_cell[1],
            )
        ].append(matchup.route_id)
    for route_ids in routes_by_artifact_cell.values():
        route_ids.sort()
    for artifact_ids in artifacts_by_stratum.values():
        artifact_ids.sort()

    artifact_targets: defaultdict[tuple[str, CandidateSeat, str], float] = defaultdict(
        float
    )
    stratum_targets: defaultdict[tuple[str, CandidateSeat, QuotaStratum], float] = (
        defaultdict(float)
    )
    artifact_base_mass: defaultdict[tuple[str, CandidateSeat, str], float] = (
        defaultdict(float)
    )
    stratum_base_mass: defaultdict[tuple[str, CandidateSeat, QuotaStratum], float] = (
        defaultdict(float)
    )
    for key, target in matchup_targets.items():
        candidate_cell = (key[0], cast(CandidateSeat, key[2]))
        if candidate_cell not in requested:
            continue
        artifact_id = route_artifact[key[1]]
        stratum = entries[artifact_id].stratum
        artifact_key = (*candidate_cell, artifact_id)
        stratum_key = (*candidate_cell, stratum)
        artifact_targets[artifact_key] += target
        stratum_targets[stratum_key] += target
        artifact_base_mass[artifact_key] += matchup_decision_mass[key]
        stratum_base_mass[stratum_key] += matchup_decision_mass[key]
    if any(
        artifact_targets[(*candidate_cell, artifact_id)] <= 0.0
        for candidate_cell in requested
        for artifact_id in selected_artifacts
    ):
        raise ValueError("sparse allocation lacks an artifact target")

    chunks = {
        candidate_cell: _sparse_game_chunks(count, matchup_game_batch_size)
        for candidate_cell, count in requested.items()
    }
    chunk_cursors = dict.fromkeys(chunks, 0)
    if sum(len(values) for values in chunks.values()) < len(selected_artifacts):
        raise ValueError("sparse allocation has too few batches for artifact coverage")

    planned_decisions: defaultdict[_MatchupKey, float] = defaultdict(float)
    planned_route_games: Counter[_MatchupKey] = Counter()
    planned_artifact_decisions: defaultdict[tuple[str, CandidateSeat, str], float] = (
        defaultdict(float)
    )
    planned_stratum_decisions: defaultdict[
        tuple[str, CandidateSeat, QuotaStratum], float
    ] = defaultdict(float)
    issued_cell_games: Counter[tuple[str, CandidateSeat]] = Counter()
    issued_artifact_games: Counter[str] = Counter()
    quotas: Counter[tuple[QuotaStratum, str, str, str, CandidateSeat]] = Counter()

    def peek_chunk(candidate_cell: tuple[str, CandidateSeat]) -> int | None:
        cursor = chunk_cursors[candidate_cell]
        values = chunks[candidate_cell]
        return None if cursor >= len(values) else values[cursor]

    def route_key(
        candidate_cell: tuple[str, CandidateSeat],
        route_id: str,
        game_count: int,
    ) -> tuple[float, float, float, str]:
        key = (candidate_cell[0], route_id, int(candidate_cell[1]))
        credit = expected_decisions_per_game[key] * game_count
        projected = (
            matchup_decision_mass[key] + planned_decisions[key] + credit
        ) / matchup_targets[key]
        last_learning = (
            state.next_window_sequence
            if planned_route_games[key] > 0
            else matchup_last_learning_windows[key]
        )
        last_rank = -1.0 if last_learning is None else float(last_learning)
        overdue = last_learning is None or (
            state.next_window_sequence - last_learning >= matchup_coverage_windows
        )
        if overdue:
            return (0.0, last_rank, projected, route_id)
        return (1.0, projected, last_rank, route_id)

    def choose_route(
        candidate_cell: tuple[str, CandidateSeat],
        artifact_id: str,
        game_count: int,
    ) -> str:
        candidates = routes_by_artifact_cell.get(
            (artifact_id, candidate_cell[0], candidate_cell[1]),
            (),
        )
        if not candidates:
            raise ValueError("sparse artifact cannot serve a candidate cell")
        return min(
            candidates,
            key=lambda route_id: route_key(
                candidate_cell,
                route_id,
                game_count,
            ),
        )

    def artifact_choice(
        candidate_cell: tuple[str, CandidateSeat],
        artifact_id: str,
        game_count: int,
    ) -> tuple[float, str, str]:
        route_id = choose_route(candidate_cell, artifact_id, game_count)
        key = (candidate_cell[0], route_id, int(candidate_cell[1]))
        credit = expected_decisions_per_game[key] * game_count
        artifact_key = (*candidate_cell, artifact_id)
        projected = (
            artifact_base_mass[artifact_key]
            + planned_artifact_decisions[artifact_key]
            + credit
        ) / artifact_targets[artifact_key]
        return (projected, artifact_id, route_id)

    def choose_hierarchical_route(
        candidate_cell: tuple[str, CandidateSeat],
        game_count: int,
    ) -> str:
        stratum_choices: list[tuple[float, str, str, str]] = []
        for stratum, artifact_ids in artifacts_by_stratum.items():
            _artifact_debt, artifact_id, route_id = min(
                artifact_choice(candidate_cell, artifact_id, game_count)
                for artifact_id in artifact_ids
            )
            key = (candidate_cell[0], route_id, int(candidate_cell[1]))
            credit = expected_decisions_per_game[key] * game_count
            stratum_key = (*candidate_cell, stratum)
            projected = (
                stratum_base_mass[stratum_key]
                + planned_stratum_decisions[stratum_key]
                + credit
            ) / stratum_targets[stratum_key]
            stratum_choices.append((projected, stratum, artifact_id, route_id))
        return min(stratum_choices)[3]

    def issue(
        candidate_cell: tuple[str, CandidateSeat],
        route_id: str,
        game_count: int,
    ) -> None:
        available = peek_chunk(candidate_cell)
        if available != game_count:
            raise RuntimeError("sparse allocation consumed the wrong game batch")
        artifact_id = route_artifact[route_id]
        stratum = entries[artifact_id].stratum
        key = (candidate_cell[0], route_id, int(candidate_cell[1]))
        credit = expected_decisions_per_game[key] * game_count
        quotas[
            (
                stratum,
                artifact_id,
                route_id,
                candidate_cell[0],
                candidate_cell[1],
            )
        ] += game_count
        planned_decisions[key] += credit
        planned_route_games[key] += game_count
        planned_artifact_decisions[(*candidate_cell, artifact_id)] += credit
        planned_stratum_decisions[(*candidate_cell, stratum)] += credit
        issued_cell_games[candidate_cell] += game_count
        issued_artifact_games[artifact_id] += game_count
        chunk_cursors[candidate_cell] += 1

    for artifact_id in selected_artifacts:
        while issued_artifact_games[artifact_id] < minimum_artifact_games:
            choices: list[tuple[float, float, float, str, int, str]] = []
            for candidate_cell in sorted(requested):
                game_count = peek_chunk(candidate_cell)
                if game_count is None:
                    continue
                artifact_debt, _artifact_id, route_id = artifact_choice(
                    candidate_cell,
                    artifact_id,
                    game_count,
                )
                key = (candidate_cell[0], route_id, int(candidate_cell[1]))
                credit = expected_decisions_per_game[key] * game_count
                stratum = entries[artifact_id].stratum
                stratum_key = (*candidate_cell, stratum)
                stratum_debt = (
                    stratum_base_mass[stratum_key]
                    + planned_stratum_decisions[stratum_key]
                    + credit
                ) / stratum_targets[stratum_key]
                choices.append(
                    (
                        issued_cell_games[candidate_cell] / requested[candidate_cell],
                        artifact_debt,
                        stratum_debt,
                        candidate_cell[0],
                        int(candidate_cell[1]),
                        route_id,
                    )
                )
            if not choices:
                raise ValueError("sparse artifact coverage exhausted game batches")
            selected = min(choices)
            candidate_cell = (selected[3], cast(CandidateSeat, selected[4]))
            game_count = peek_chunk(candidate_cell)
            if game_count is None:
                raise RuntimeError("sparse artifact coverage lost its game batch")
            issue(candidate_cell, selected[5], game_count)

    for candidate_cell in sorted(requested):
        while (game_count := peek_chunk(candidate_cell)) is not None:
            issue(
                candidate_cell,
                choose_hierarchical_route(candidate_cell, game_count),
                game_count,
            )

    if any(
        chunk_cursors[candidate_cell] != len(values)
        for candidate_cell, values in chunks.items()
    ):
        raise RuntimeError("sparse quota planner lost candidate game batches")
    ordered = sorted(quotas.items())
    cells = tuple(
        OpponentQuotaCell(
            cell_index=index,
            stratum=key[0],
            artifact_id=key[1],
            route_id=key[2],
            candidate_deck_digest=key[3],
            candidate_seat=key[4],
            game_count=count,
        )
        for index, (key, count) in enumerate(ordered)
    )
    if sum(item.game_count for item in cells) != sum(requested.values()):
        raise RuntimeError("sparse quota planner failed to conserve games")
    content = {
        "revision_fingerprint": state.revision.fingerprint,
        "base_state_fingerprint": state.fingerprint,
        "window_sequence": state.next_window_sequence,
        "cells": [item.model_dump(mode="json") for item in cells],
        "selected_artifact_ids": selected_artifacts,
        "required_exposure_artifact_ids": selected_artifacts,
    }
    return QuotaWindowPlan(
        plan_id=canonical_fingerprint("quota-window-plan-v1", content),
        revision_fingerprint=state.revision.fingerprint,
        base_state_fingerprint=state.fingerprint,
        window_sequence=state.next_window_sequence,
        cells=cells,
        selected_artifact_ids=selected_artifacts,
        required_exposure_artifact_ids=selected_artifacts,
    )


def _sparse_game_chunks(total_games: int, batch_size: int) -> tuple[int, ...]:
    """Partition one candidate cell without creating a short tail matchup."""
    if total_games <= 0 or batch_size <= 0:
        raise ValueError("sparse game chunk inputs must be positive")
    if total_games < batch_size:
        return (total_games,)
    full_batches, remainder = divmod(total_games, batch_size)
    return (
        batch_size + remainder,
        *(batch_size for _index in range(full_batches - 1)),
    )
