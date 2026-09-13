"""Compact two-level CSR arrays for schema-9 planner behavior evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields

import numpy as np

from ptcg_rl.agent.search.planner_fallback import PlannerFallbackReason
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.rl.planner_evidence import (
    PLANNER_SOURCE_COUNT,
    PlannerBehaviorBranch,
    PlannerBehaviorEvidence,
    PlannerCandidateEvidence,
    ScenarioSupportMode,
)

_FINGERPRINT_FIELDS = (
    "root_information_fingerprints",
    "scenario_support_fingerprints",
    "model_fingerprints",
    "constructor_fingerprints",
    "scorer_fingerprints",
    "controller_fingerprints",
    "planner_fingerprints",
    "evidence_fingerprints",
)


@dataclass(frozen=True, slots=True)
class PlannerEvidenceArrayBlock:
    """Decision-to-candidate-to-option CSR plus fixed-width metadata."""

    decision_candidate_offsets: np.ndarray
    candidate_action_offsets: np.ndarray
    candidate_action_indices: np.ndarray
    candidate_features: np.ndarray
    candidate_base_logprobs: np.ndarray
    candidate_proposal_logprobs: np.ndarray
    candidate_score_priors: np.ndarray
    candidate_target_probabilities: np.ndarray
    candidate_behavior_probabilities: np.ndarray
    candidate_robust_scores: np.ndarray
    candidate_source_bits: np.ndarray
    candidate_rules_exact: np.ndarray
    branches: np.ndarray
    fallback_reasons: np.ndarray
    selected_candidate_indices: np.ndarray
    root_information_fingerprints: np.ndarray
    scenario_support_fingerprints: np.ndarray
    model_fingerprints: np.ndarray
    constructor_fingerprints: np.ndarray
    scorer_fingerprints: np.ndarray
    controller_fingerprints: np.ndarray
    planner_fingerprints: np.ndarray
    evidence_fingerprints: np.ndarray
    policy_versions: np.ndarray
    proposal_versions: np.ndarray
    constructor_versions: np.ndarray
    planner_versions: np.ndarray
    scenario_support_modes: np.ndarray
    legal_action_counts: np.ndarray
    scenario_counts: np.ndarray
    support_exhaustive: np.ndarray
    scenario_grid_complete: np.ndarray
    leaf_bootstrapped: np.ndarray
    support_censored: np.ndarray
    configured_source_quotas: np.ndarray
    used_source_quotas: np.ndarray
    engine_transition_limits: np.ndarray
    engine_transitions_used: np.ndarray
    prefix_node_limits: np.ndarray
    prefix_nodes_used: np.ndarray
    wall_clock_limits_ms: np.ndarray
    wall_clock_used_ms: np.ndarray
    planner_temperatures: np.ndarray

    @property
    def decision_count(self) -> int:
        """Return the number of behavior rows."""
        return int(self.branches.shape[0])

    def evidence_at(self, index: int) -> PlannerBehaviorEvidence:
        """Reconstruct one immutable schema-9 record."""
        if not 0 <= index < self.decision_count:
            raise IndexError("planner evidence row is out of range")
        candidate_start = int(self.decision_candidate_offsets[index])
        candidate_stop = int(self.decision_candidate_offsets[index + 1])
        candidates: list[PlannerCandidateEvidence] = []
        for candidate_index in range(candidate_start, candidate_stop):
            action_start = int(self.candidate_action_offsets[candidate_index])
            action_stop = int(self.candidate_action_offsets[candidate_index + 1])
            candidates.append(
                PlannerCandidateEvidence(
                    action=tuple(
                        int(value)
                        for value in self.candidate_action_indices[
                            action_start:action_stop
                        ]
                    ),
                    aggregate_features=tuple(
                        float(value)
                        for value in self.candidate_features[candidate_index]
                    ),
                    base_logprob=float(self.candidate_base_logprobs[candidate_index]),
                    proposal_logprob=float(
                        self.candidate_proposal_logprobs[candidate_index]
                    ),
                    score_prior=float(self.candidate_score_priors[candidate_index]),
                    target_probability=float(
                        self.candidate_target_probabilities[candidate_index]
                    ),
                    behavior_probability=float(
                        self.candidate_behavior_probabilities[candidate_index]
                    ),
                    robust_score=float(self.candidate_robust_scores[candidate_index]),
                    source_bits=int(self.candidate_source_bits[candidate_index]),
                    rules_exact=bool(self.candidate_rules_exact[candidate_index]),
                )
            )
        evidence = PlannerBehaviorEvidence(
            branch=PlannerBehaviorBranch(int(self.branches[index])),
            fallback_reason=PlannerFallbackReason(int(self.fallback_reasons[index])),
            candidates=tuple(candidates),
            selected_candidate_index=int(self.selected_candidate_indices[index]),
            root_information_fingerprint=_fingerprint_at(
                self.root_information_fingerprints, index
            ),
            scenario_support_fingerprint=_fingerprint_at(
                self.scenario_support_fingerprints, index
            ),
            model_fingerprint=_fingerprint_at(self.model_fingerprints, index),
            constructor_fingerprint=_fingerprint_at(
                self.constructor_fingerprints, index
            ),
            scorer_fingerprint=_fingerprint_at(self.scorer_fingerprints, index),
            controller_fingerprint=_fingerprint_at(self.controller_fingerprints, index),
            planner_fingerprint=_fingerprint_at(self.planner_fingerprints, index),
            policy_version=int(self.policy_versions[index]),
            proposal_version=int(self.proposal_versions[index]),
            constructor_version=int(self.constructor_versions[index]),
            planner_version=int(self.planner_versions[index]),
            scenario_support_mode=ScenarioSupportMode(
                int(self.scenario_support_modes[index])
            ),
            legal_action_count=int(self.legal_action_counts[index]),
            scenario_count=int(self.scenario_counts[index]),
            support_exhaustive=bool(self.support_exhaustive[index]),
            scenario_grid_complete=bool(self.scenario_grid_complete[index]),
            leaf_bootstrapped=bool(self.leaf_bootstrapped[index]),
            support_censored=bool(self.support_censored[index]),
            configured_source_quotas=tuple(
                int(value) for value in self.configured_source_quotas[index]
            ),
            used_source_quotas=tuple(
                int(value) for value in self.used_source_quotas[index]
            ),
            engine_transition_limit=int(self.engine_transition_limits[index]),
            engine_transitions_used=int(self.engine_transitions_used[index]),
            prefix_node_limit=int(self.prefix_node_limits[index]),
            prefix_nodes_used=int(self.prefix_nodes_used[index]),
            wall_clock_limit_ms=int(self.wall_clock_limits_ms[index]),
            wall_clock_used_ms=int(self.wall_clock_used_ms[index]),
            planner_temperature=float(self.planner_temperatures[index]),
        )
        expected = _fingerprint_at(self.evidence_fingerprints, index)
        if evidence.evidence_fingerprint != expected:
            raise ValueError("planner evidence fingerprint does not match its arrays")
        return evidence


def build_planner_evidence_array_block(
    evidence: Sequence[PlannerBehaviorEvidence | None],
) -> PlannerEvidenceArrayBlock | None:
    """Flatten complete schema-9 behavior records without object-tree replay."""
    presence = tuple(item is not None for item in evidence)
    if not any(presence):
        return None
    if not all(presence):
        raise ValueError("schema-9 blocks require a behavior branch for every row")
    records = tuple(item for item in evidence if item is not None)
    decision_offsets = [0]
    action_offsets = [0]
    action_indices: list[int] = []
    candidates: list[PlannerCandidateEvidence] = []
    for record in records:
        for candidate in record.candidates:
            action_indices.extend(candidate.action)
            action_offsets.append(len(action_indices))
            candidates.append(candidate)
        decision_offsets.append(len(candidates))
    candidate_count = len(candidates)
    int32_max = np.iinfo(np.int32).max
    if (
        len(records) > int32_max
        or candidate_count > int32_max
        or len(action_indices) > int32_max
    ):
        raise ValueError("planner evidence CSR exceeds int32 wire capacity")
    feature_rows = np.asarray(
        [candidate.aggregate_features for candidate in candidates],
        dtype=np.float32,
    ).reshape(candidate_count, SEARCH_EVIDENCE_FEATURE_SIZE)
    block = PlannerEvidenceArrayBlock(
        decision_candidate_offsets=np.asarray(decision_offsets, dtype=np.int32),
        candidate_action_offsets=np.asarray(action_offsets, dtype=np.int32),
        candidate_action_indices=np.asarray(action_indices, dtype=np.int32),
        candidate_features=feature_rows,
        candidate_base_logprobs=_candidate_float_array(candidates, "base_logprob"),
        candidate_proposal_logprobs=_candidate_float_array(
            candidates, "proposal_logprob"
        ),
        candidate_score_priors=_candidate_float_array(candidates, "score_prior"),
        candidate_target_probabilities=_candidate_float_array(
            candidates, "target_probability"
        ),
        candidate_behavior_probabilities=_candidate_float_array(
            candidates, "behavior_probability"
        ),
        candidate_robust_scores=_candidate_float_array(candidates, "robust_score"),
        candidate_source_bits=np.asarray(
            [candidate.source_bits for candidate in candidates], dtype=np.uint16
        ),
        candidate_rules_exact=np.asarray(
            [candidate.rules_exact for candidate in candidates], dtype=np.bool_
        ),
        branches=np.asarray([int(record.branch) for record in records], dtype=np.uint8),
        fallback_reasons=np.asarray(
            [int(record.fallback_reason) for record in records], dtype=np.uint8
        ),
        selected_candidate_indices=np.asarray(
            [record.selected_candidate_index for record in records], dtype=np.int32
        ),
        root_information_fingerprints=_fingerprint_array(
            [record.root_information_fingerprint for record in records]
        ),
        scenario_support_fingerprints=_fingerprint_array(
            [record.scenario_support_fingerprint for record in records]
        ),
        model_fingerprints=_fingerprint_array(
            [record.model_fingerprint for record in records]
        ),
        constructor_fingerprints=_fingerprint_array(
            [record.constructor_fingerprint for record in records]
        ),
        scorer_fingerprints=_fingerprint_array(
            [record.scorer_fingerprint for record in records]
        ),
        controller_fingerprints=_fingerprint_array(
            [record.controller_fingerprint for record in records]
        ),
        planner_fingerprints=_fingerprint_array(
            [record.planner_fingerprint for record in records]
        ),
        evidence_fingerprints=_fingerprint_array(
            [record.evidence_fingerprint for record in records]
        ),
        policy_versions=_record_int_array(records, "policy_version"),
        proposal_versions=_record_int_array(records, "proposal_version"),
        constructor_versions=_record_int_array(records, "constructor_version"),
        planner_versions=_record_int_array(records, "planner_version"),
        scenario_support_modes=np.asarray(
            [int(record.scenario_support_mode) for record in records], dtype=np.uint8
        ),
        legal_action_counts=np.asarray(
            [record.legal_action_count for record in records], dtype=np.int64
        ),
        scenario_counts=_record_int_array(records, "scenario_count"),
        support_exhaustive=_record_bool_array(records, "support_exhaustive"),
        scenario_grid_complete=_record_bool_array(records, "scenario_grid_complete"),
        leaf_bootstrapped=_record_bool_array(records, "leaf_bootstrapped"),
        support_censored=_record_bool_array(records, "support_censored"),
        configured_source_quotas=_quota_array(
            [record.configured_source_quotas for record in records]
        ),
        used_source_quotas=_quota_array(
            [record.used_source_quotas for record in records]
        ),
        engine_transition_limits=_record_int_array(records, "engine_transition_limit"),
        engine_transitions_used=_record_int_array(records, "engine_transitions_used"),
        prefix_node_limits=_record_int_array(records, "prefix_node_limit"),
        prefix_nodes_used=_record_int_array(records, "prefix_nodes_used"),
        wall_clock_limits_ms=_record_int_array(records, "wall_clock_limit_ms"),
        wall_clock_used_ms=_record_int_array(records, "wall_clock_used_ms"),
        planner_temperatures=np.asarray(
            [record.planner_temperature for record in records], dtype=np.float32
        ),
    )
    validate_planner_evidence_array_block(block, decision_count=len(records))
    return block


def validate_planner_evidence_array_block(
    block: PlannerEvidenceArrayBlock,
    *,
    decision_count: int,
) -> tuple[PlannerBehaviorEvidence, ...]:
    """Reject corrupt CSR layout, dtypes, identities, and branch semantics."""
    _require_array(
        block.decision_candidate_offsets,
        "decision_candidate_offsets",
        np.int32,
        (decision_count + 1,),
    )
    _require_offsets(block.decision_candidate_offsets, "planner decision")
    candidate_count = int(block.decision_candidate_offsets[-1])
    _require_array(
        block.candidate_action_offsets,
        "candidate_action_offsets",
        np.int32,
        (candidate_count + 1,),
    )
    _require_offsets(block.candidate_action_offsets, "planner candidate action")
    _require_array(
        block.candidate_action_indices,
        "candidate_action_indices",
        np.int32,
        (int(block.candidate_action_offsets[-1]),),
    )
    _require_array(
        block.candidate_features,
        "candidate_features",
        np.float32,
        (candidate_count, SEARCH_EVIDENCE_FEATURE_SIZE),
    )
    candidate_float_names = (
        "candidate_base_logprobs",
        "candidate_proposal_logprobs",
        "candidate_score_priors",
        "candidate_target_probabilities",
        "candidate_behavior_probabilities",
        "candidate_robust_scores",
    )
    for name in candidate_float_names:
        _require_array(getattr(block, name), name, np.float32, (candidate_count,))
    if not all(
        bool(np.isfinite(getattr(block, name)).all())
        for name in ("candidate_features", *candidate_float_names)
    ):
        raise ValueError("planner candidate numeric evidence must be finite")
    _require_array(
        block.candidate_source_bits,
        "candidate_source_bits",
        np.uint16,
        (candidate_count,),
    )
    _require_array(
        block.candidate_rules_exact,
        "candidate_rules_exact",
        np.bool_,
        (candidate_count,),
    )
    _require_array(block.branches, "branches", np.uint8, (decision_count,))
    _require_array(
        block.fallback_reasons,
        "fallback_reasons",
        np.uint8,
        (decision_count,),
    )
    _require_array(
        block.selected_candidate_indices,
        "selected_candidate_indices",
        np.int32,
        (decision_count,),
    )
    for name in _FINGERPRINT_FIELDS:
        _require_array(getattr(block, name), name, np.uint8, (decision_count, 32))
    for name in (
        "policy_versions",
        "proposal_versions",
        "constructor_versions",
        "planner_versions",
        "scenario_counts",
        "engine_transition_limits",
        "engine_transitions_used",
        "prefix_node_limits",
        "prefix_nodes_used",
        "wall_clock_limits_ms",
        "wall_clock_used_ms",
    ):
        _require_array(getattr(block, name), name, np.int32, (decision_count,))
    _require_array(
        block.scenario_support_modes,
        "scenario_support_modes",
        np.uint8,
        (decision_count,),
    )
    _require_array(
        block.legal_action_counts,
        "legal_action_counts",
        np.int64,
        (decision_count,),
    )
    for name in (
        "support_exhaustive",
        "scenario_grid_complete",
        "leaf_bootstrapped",
        "support_censored",
    ):
        _require_array(getattr(block, name), name, np.bool_, (decision_count,))
    for name in ("configured_source_quotas", "used_source_quotas"):
        _require_array(
            getattr(block, name),
            name,
            np.uint16,
            (decision_count, PLANNER_SOURCE_COUNT),
        )
    _require_array(
        block.planner_temperatures,
        "planner_temperatures",
        np.float32,
        (decision_count,),
    )
    if not bool(np.isfinite(block.planner_temperatures).all()) or bool(
        np.any(block.planner_temperatures <= 0)
    ):
        raise ValueError("planner temperatures must be finite and positive")
    return tuple(block.evidence_at(index) for index in range(decision_count))


def planner_array_field_names() -> tuple[str, ...]:
    """Return the stable transport field order."""
    return tuple(field.name for field in fields(PlannerEvidenceArrayBlock))


def _candidate_float_array(
    candidates: Sequence[PlannerCandidateEvidence],
    name: str,
) -> np.ndarray:
    return np.asarray(
        [getattr(candidate, name) for candidate in candidates], dtype=np.float32
    )


def _record_int_array(
    records: Sequence[PlannerBehaviorEvidence],
    name: str,
) -> np.ndarray:
    return np.asarray([getattr(record, name) for record in records], dtype=np.int32)


def _record_bool_array(
    records: Sequence[PlannerBehaviorEvidence],
    name: str,
) -> np.ndarray:
    return np.asarray([getattr(record, name) for record in records], dtype=np.bool_)


def _quota_array(rows: Sequence[tuple[int, ...]]) -> np.ndarray:
    if any(any(value > np.iinfo(np.uint16).max for value in row) for row in rows):
        raise ValueError("planner source quota exceeds uint16 wire capacity")
    return np.asarray(rows, dtype=np.uint16).reshape(-1, PLANNER_SOURCE_COUNT)


def _fingerprint_array(values: Sequence[str]) -> np.ndarray:
    return np.asarray([tuple(bytes.fromhex(value)) for value in values], dtype=np.uint8)


def _fingerprint_at(values: np.ndarray, index: int) -> str:
    return bytes(values[index]).hex()


def _require_offsets(values: np.ndarray, label: str) -> None:
    if int(values[0]) != 0 or bool(np.any(values[1:] < values[:-1])):
        raise ValueError(f"{label} offsets must start at zero and be monotonic")


def _require_array(
    value: np.ndarray,
    name: str,
    dtype: type[np.generic],
    shape: tuple[int, ...],
) -> None:
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} has the wrong NumPy dtype")
    if value.shape != shape:
        raise ValueError(f"{name} has the wrong shape")


__all__ = [
    "PlannerEvidenceArrayBlock",
    "build_planner_evidence_array_block",
    "planner_array_field_names",
    "validate_planner_evidence_array_block",
]
