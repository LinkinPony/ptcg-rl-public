"""Detached schema-10 evidence from native counterfactual macro search."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from ptcg_rl.engine.compact_consequence import SemanticEndpoint
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.search_evidence import (
    SEARCH_EVIDENCE_EXACT_INDEX,
    SEARCH_EVIDENCE_FEATURE_SIZE,
    SearchCandidateEvidence,
    SearchEvidence,
)
from ptcg_rl.rl.planner_evidence import ScenarioSupportMode

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class MacroTeacherCandidateEvidence:
    """One complete action and its exact macro-boundary consequences."""

    action: tuple[int, ...]
    aggregate_features: tuple[float, ...]
    exact_effect_features: tuple[float, ...]
    robust_score: float
    endpoint_probabilities: tuple[float, float, float]
    rules_exact: bool

    def __post_init__(self) -> None:
        """Validate the fixed-width public candidate row."""
        action = tuple(int(value) for value in self.action)
        aggregate = tuple(float(value) for value in self.aggregate_features)
        effects = tuple(float(value) for value in self.exact_effect_features)
        endpoints = tuple(float(value) for value in self.endpoint_probabilities)
        if len(set(action)) != len(action) or any(value < 0 for value in action):
            raise ValueError("macro teacher action indices are invalid")
        if len(aggregate) != SEARCH_EVIDENCE_FEATURE_SIZE:
            raise ValueError("macro teacher aggregate feature width is invalid")
        if len(effects) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError("macro teacher effect feature width is invalid")
        if len(endpoints) != 3:
            raise ValueError("macro teacher endpoint distribution is invalid")
        numeric = (*aggregate, *effects, self.robust_score, *endpoints)
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("macro teacher candidate values must be finite")
        if any(value < 0.0 or value > 1.0 for value in endpoints) or not math.isclose(
            sum(endpoints), 1.0, rel_tol=0.0, abs_tol=1.0e-5
        ):
            raise ValueError("macro teacher endpoint probabilities must sum to one")
        expected_exact = 1.0 if self.rules_exact else 0.0
        if aggregate[SEARCH_EVIDENCE_EXACT_INDEX] != expected_exact:
            raise ValueError("macro teacher exactness differs from search features")
        SearchCandidateEvidence(action=action, features=aggregate)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "aggregate_features", aggregate)
        object.__setattr__(self, "exact_effect_features", effects)
        object.__setattr__(self, "endpoint_probabilities", endpoints)


@dataclass(frozen=True, slots=True)
class MacroTeacherEvidence:
    """Complete native candidate grid persisted only after successful resolution."""

    candidates: tuple[MacroTeacherCandidateEvidence, ...]
    target_candidate_index: int
    legal_action_count: int
    scenario_count: int
    support_exhaustive: bool
    scenario_grid_complete: bool
    scenario_support_mode: ScenarioSupportMode
    leaf_bootstrapped: bool
    policy_version: int
    constructor_fingerprint: str
    scorer_fingerprint: str
    controller_fingerprint: str
    adapter_fingerprint: str
    producer_fingerprint: str

    def __post_init__(self) -> None:
        """Reject partial grids, duplicate support, and ambiguous identities."""
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("macro teacher evidence requires candidates")
        if not self.scenario_grid_complete:
            raise ValueError("partial native macro grids must remain absent")
        if self.target_candidate_index < 0 or self.target_candidate_index >= len(
            candidates
        ):
            raise ValueError("macro teacher target candidate is out of range")
        actions = tuple(candidate.action for candidate in candidates)
        if len(set(actions)) != len(actions):
            raise ValueError("macro teacher candidate actions must be unique")
        if self.legal_action_count < len(candidates):
            raise ValueError("macro teacher legal action count is below support")
        if self.support_exhaustive != (self.legal_action_count == len(candidates)):
            raise ValueError("macro teacher exhaustive flag differs from support")
        if self.scenario_count <= 0:
            raise ValueError("macro teacher evidence requires scenarios")
        if self.policy_version < 0:
            raise ValueError("macro teacher policy version must be non-negative")
        if self.scenario_support_mode is ScenarioSupportMode.UNSPECIFIED:
            raise ValueError("macro teacher scenario support must be explicit")
        for value in (
            self.constructor_fingerprint,
            self.scorer_fingerprint,
            self.controller_fingerprint,
            self.adapter_fingerprint,
            self.producer_fingerprint,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("macro teacher identities must be SHA-256")
        object.__setattr__(self, "candidates", candidates)

    @property
    def target_action(self) -> tuple[int, ...]:
        """Return the robust teacher action."""
        return self.candidates[self.target_candidate_index].action

    def compatible_search_evidence(self) -> SearchEvidence:
        """Return schema-8-width rows for the existing candidate reranker."""
        global_exact = self.support_exhaustive and all(
            candidate.rules_exact for candidate in self.candidates
        )
        search_candidates = tuple(
            SearchCandidateEvidence(
                action=candidate.action,
                features=tuple(
                    1.0
                    if global_exact and index == SEARCH_EVIDENCE_EXACT_INDEX
                    else 0.0
                    if index == SEARCH_EVIDENCE_EXACT_INDEX
                    else value
                    for index, value in enumerate(candidate.aggregate_features)
                ),
            )
            for candidate in self.candidates
        )
        return SearchEvidence(
            candidates=search_candidates,
            legal_action_count=self.legal_action_count,
            world_count=self.scenario_count,
            exhaustive=self.support_exhaustive,
            exact=global_exact,
        )


@dataclass(frozen=True, slots=True)
class MacroTeacherRequest:
    """Actor-owned retained planner batch resolved after behavior submission."""

    planner_batch: Any
    row_index: int

    def __post_init__(self) -> None:
        rows = tuple(getattr(self.planner_batch, "rows", ()))
        if self.row_index < 0 or self.row_index >= len(rows):
            raise ValueError("macro teacher request row is out of range")


def build_native_macro_teacher_target(
    *,
    actions: tuple[tuple[int, ...], ...],
    scored: Any,
    metadata: Any,
) -> Any:
    """Convert one complete native planner grid into auxiliary teacher evidence."""
    from ptcg_rl.rl.engine_teacher import EngineTeacherTarget

    candidate_count = len(actions)
    if candidate_count <= 0:
        raise ValueError("native macro teacher requires candidates")
    outcomes = tuple(scored.outcome_batch.outcomes)
    if len(outcomes) != candidate_count:
        raise ValueError("native macro outcomes differ from candidate support")
    scenario_count = len(outcomes[0].cells)
    if scenario_count <= 0 or any(
        len(outcome.cells) != scenario_count for outcome in outcomes
    ):
        raise ValueError("native macro teacher requires a complete scenario grid")
    weights = np.asarray(
        [float(cell.weight) for cell in outcomes[0].cells],
        dtype=np.float64,
    )
    if not bool(np.isfinite(weights).all()) or bool((weights <= 0.0).any()):
        raise ValueError("native macro scenario weights are invalid")
    for outcome in outcomes[1:]:
        candidate_weights = np.asarray(
            [float(cell.weight) for cell in outcome.cells],
            dtype=np.float64,
        )
        if not np.array_equal(candidate_weights, weights):
            raise ValueError("native macro candidates use different scenario weights")
    weights /= float(weights.sum())
    cells = scored.outcome_batch.scoring_cells
    exact_effects = np.asarray(cells.exact_effects, dtype=np.float64).reshape(
        candidate_count,
        scenario_count,
        -1,
    )
    endpoints = np.asarray(cells.endpoints, dtype=np.int64).reshape(
        candidate_count,
        scenario_count,
    )
    if exact_effects.shape[2] != DYNAMIC_EFFECT_FEATURE_SIZE:
        raise ValueError("native macro exact-effect width is invalid")
    aggregate_features = tuple(scored.search_evidence.feature_rows)
    robust_scores = tuple(
        float(candidate.robust_score) for candidate in scored.aggregation.candidates
    )
    rules_exact = tuple(bool(value) for value in scored.search_evidence.rules_exact)
    if not (
        len(aggregate_features)
        == len(robust_scores)
        == len(rules_exact)
        == candidate_count
    ):
        raise ValueError("native macro teacher candidate columns are misaligned")
    candidates: list[MacroTeacherCandidateEvidence] = []
    endpoint_values = (
        SemanticEndpoint.TERMINAL,
        SemanticEndpoint.TURN_HANDOFF,
        SemanticEndpoint.SAME_SEAT_MAIN,
    )
    for index, action in enumerate(actions):
        endpoint_values_for_candidate = tuple(
            float(weights[endpoints[index] == int(endpoint)].sum())
            for endpoint in endpoint_values
        )
        endpoint_probabilities = (
            endpoint_values_for_candidate[0],
            endpoint_values_for_candidate[1],
            endpoint_values_for_candidate[2],
        )
        candidates.append(
            MacroTeacherCandidateEvidence(
                action=action,
                aggregate_features=aggregate_features[index],
                exact_effect_features=tuple(
                    float(value)
                    for value in np.average(
                        exact_effects[index],
                        axis=0,
                        weights=weights,
                    )
                ),
                robust_score=robust_scores[index],
                endpoint_probabilities=endpoint_probabilities,
                rules_exact=rules_exact[index],
            )
        )
    target_index = max(
        range(candidate_count),
        key=lambda index: (robust_scores[index], -index),
    )
    ordered_scores = sorted(robust_scores, reverse=True)
    confidence = (
        1.0
        if len(ordered_scores) == 1
        else 1.0 / (1.0 + math.exp(-4.0 * (ordered_scores[0] - ordered_scores[1])))
    )
    evidence = MacroTeacherEvidence(
        candidates=tuple(candidates),
        target_candidate_index=target_index,
        legal_action_count=int(metadata.legal_action_count),
        scenario_count=scenario_count,
        support_exhaustive=bool(metadata.support_exhaustive),
        scenario_grid_complete=bool(metadata.scenario_grid_complete),
        scenario_support_mode=ScenarioSupportMode(metadata.scenario_support_mode),
        leaf_bootstrapped=bool(metadata.leaf_bootstrapped),
        policy_version=int(metadata.policy_version),
        constructor_fingerprint=str(metadata.constructor_fingerprint),
        scorer_fingerprint=str(metadata.scorer_fingerprint),
        controller_fingerprint=str(metadata.controller_fingerprint),
        adapter_fingerprint=str(metadata.model_fingerprint),
        producer_fingerprint=str(metadata.planner_fingerprint),
    )
    search_evidence = evidence.compatible_search_evidence()
    return EngineTeacherTarget(
        action=evidence.target_action,
        confidence=confidence,
        weight=1.0,
        search_evidence=search_evidence,
        macro_evidence=evidence,
    )


__all__ = [
    "MacroTeacherCandidateEvidence",
    "MacroTeacherEvidence",
    "MacroTeacherRequest",
    "build_native_macro_teacher_target",
]
