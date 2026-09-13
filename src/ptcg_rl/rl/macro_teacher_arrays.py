"""Compact sparse arrays for schema-10 native macro-teacher evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.search_evidence import SEARCH_EVIDENCE_FEATURE_SIZE
from ptcg_rl.rl.engine_teacher import EngineTeacherTarget
from ptcg_rl.rl.macro_teacher import (
    MacroTeacherCandidateEvidence,
    MacroTeacherEvidence,
)
from ptcg_rl.rl.planner_evidence import ScenarioSupportMode

_FINGERPRINT_COUNT = 5


@dataclass(frozen=True, eq=False)
class MacroTeacherArrayBlock:
    """Decision-to-candidate CSR with complete fixed-width consequence rows."""

    decision_candidate_offsets: np.ndarray
    candidate_action_offsets: np.ndarray
    candidate_action_indices: np.ndarray
    candidate_aggregate_features: np.ndarray
    candidate_effect_features: np.ndarray
    candidate_robust_scores: np.ndarray
    candidate_endpoint_probabilities: np.ndarray
    candidate_rules_exact: np.ndarray
    target_candidate_indices: np.ndarray
    legal_action_counts: np.ndarray
    scenario_counts: np.ndarray
    support_exhaustive: np.ndarray
    scenario_grid_complete: np.ndarray
    scenario_support_modes: np.ndarray
    leaf_bootstrapped: np.ndarray
    policy_versions: np.ndarray
    fingerprints: np.ndarray
    masks: np.ndarray

    @property
    def decision_count(self) -> int:
        """Return the dense decision row count."""
        return int(self.masks.shape[0])

    @property
    def candidate_count(self) -> int:
        """Return the flattened candidate count."""
        return int(self.candidate_robust_scores.shape[0])

    def evidence_at(self, decision_index: int) -> MacroTeacherEvidence | None:
        """Reconstruct one sparse teacher evidence row."""
        if decision_index < 0 or decision_index >= self.decision_count:
            raise IndexError("macro teacher decision index is out of range")
        if not bool(self.masks[decision_index]):
            return None
        start = int(self.decision_candidate_offsets[decision_index])
        stop = int(self.decision_candidate_offsets[decision_index + 1])
        candidates: list[MacroTeacherCandidateEvidence] = []
        for candidate_index in range(start, stop):
            action_start = int(self.candidate_action_offsets[candidate_index])
            action_stop = int(self.candidate_action_offsets[candidate_index + 1])
            endpoint_values = self.candidate_endpoint_probabilities[candidate_index]
            candidates.append(
                MacroTeacherCandidateEvidence(
                    action=tuple(
                        int(value)
                        for value in self.candidate_action_indices[
                            action_start:action_stop
                        ]
                    ),
                    aggregate_features=tuple(
                        float(value)
                        for value in self.candidate_aggregate_features[candidate_index]
                    ),
                    exact_effect_features=tuple(
                        float(value)
                        for value in self.candidate_effect_features[candidate_index]
                    ),
                    robust_score=float(self.candidate_robust_scores[candidate_index]),
                    endpoint_probabilities=(
                        float(endpoint_values[0]),
                        float(endpoint_values[1]),
                        float(endpoint_values[2]),
                    ),
                    rules_exact=bool(self.candidate_rules_exact[candidate_index]),
                )
            )
        fingerprint_values = tuple(
            bytes(self.fingerprints[decision_index, index]).hex()
            for index in range(_FINGERPRINT_COUNT)
        )
        return MacroTeacherEvidence(
            candidates=tuple(candidates),
            target_candidate_index=int(self.target_candidate_indices[decision_index]),
            legal_action_count=int(self.legal_action_counts[decision_index]),
            scenario_count=int(self.scenario_counts[decision_index]),
            support_exhaustive=bool(self.support_exhaustive[decision_index]),
            scenario_grid_complete=bool(self.scenario_grid_complete[decision_index]),
            scenario_support_mode=ScenarioSupportMode(
                int(self.scenario_support_modes[decision_index])
            ),
            leaf_bootstrapped=bool(self.leaf_bootstrapped[decision_index]),
            policy_version=int(self.policy_versions[decision_index]),
            constructor_fingerprint=fingerprint_values[0],
            scorer_fingerprint=fingerprint_values[1],
            controller_fingerprint=fingerprint_values[2],
            adapter_fingerprint=fingerprint_values[3],
            producer_fingerprint=fingerprint_values[4],
        )


def build_macro_teacher_array_block(
    targets: list[EngineTeacherTarget | None] | tuple[EngineTeacherTarget | None, ...],
) -> MacroTeacherArrayBlock | None:
    """Flatten sparse target-attached macro evidence into schema-10 arrays."""
    evidence_rows = tuple(
        None if target is None else target.macro_evidence for target in targets
    )
    if not any(evidence is not None for evidence in evidence_rows):
        return None
    decision_offsets = [0]
    action_offsets = [0]
    action_indices: list[int] = []
    aggregate_features: list[tuple[float, ...]] = []
    effect_features: list[tuple[float, ...]] = []
    robust_scores: list[float] = []
    endpoint_probabilities: list[tuple[float, float, float]] = []
    rules_exact: list[bool] = []
    target_indices: list[int] = []
    legal_counts: list[int] = []
    scenario_counts: list[int] = []
    exhaustive: list[bool] = []
    complete: list[bool] = []
    support_modes: list[int] = []
    bootstrapped: list[bool] = []
    policy_versions: list[int] = []
    fingerprints: list[tuple[bytes, ...]] = []
    masks: list[bool] = []
    for evidence in evidence_rows:
        masks.append(evidence is not None)
        if evidence is None:
            decision_offsets.append(len(aggregate_features))
            target_indices.append(-1)
            legal_counts.append(0)
            scenario_counts.append(0)
            exhaustive.append(False)
            complete.append(False)
            support_modes.append(int(ScenarioSupportMode.UNSPECIFIED))
            bootstrapped.append(False)
            policy_versions.append(-1)
            fingerprints.append((b"\x00" * 32,) * _FINGERPRINT_COUNT)
            continue
        for candidate in evidence.candidates:
            action_indices.extend(candidate.action)
            action_offsets.append(len(action_indices))
            aggregate_features.append(candidate.aggregate_features)
            effect_features.append(candidate.exact_effect_features)
            robust_scores.append(candidate.robust_score)
            endpoint_probabilities.append(candidate.endpoint_probabilities)
            rules_exact.append(candidate.rules_exact)
        decision_offsets.append(len(aggregate_features))
        target_indices.append(evidence.target_candidate_index)
        legal_counts.append(evidence.legal_action_count)
        scenario_counts.append(evidence.scenario_count)
        exhaustive.append(evidence.support_exhaustive)
        complete.append(evidence.scenario_grid_complete)
        support_modes.append(int(evidence.scenario_support_mode))
        bootstrapped.append(evidence.leaf_bootstrapped)
        policy_versions.append(evidence.policy_version)
        fingerprints.append(
            tuple(
                bytes.fromhex(value)
                for value in (
                    evidence.constructor_fingerprint,
                    evidence.scorer_fingerprint,
                    evidence.controller_fingerprint,
                    evidence.adapter_fingerprint,
                    evidence.producer_fingerprint,
                )
            )
        )
    block = MacroTeacherArrayBlock(
        decision_candidate_offsets=np.asarray(decision_offsets, dtype=np.int32),
        candidate_action_offsets=np.asarray(action_offsets, dtype=np.int32),
        candidate_action_indices=np.asarray(action_indices, dtype=np.int32),
        candidate_aggregate_features=np.asarray(
            aggregate_features, dtype=np.float32
        ).reshape((-1, SEARCH_EVIDENCE_FEATURE_SIZE)),
        candidate_effect_features=np.asarray(effect_features, dtype=np.float32).reshape(
            (-1, DYNAMIC_EFFECT_FEATURE_SIZE)
        ),
        candidate_robust_scores=np.asarray(robust_scores, dtype=np.float32),
        candidate_endpoint_probabilities=np.asarray(
            endpoint_probabilities, dtype=np.float32
        ).reshape((-1, 3)),
        candidate_rules_exact=np.asarray(rules_exact, dtype=np.bool_),
        target_candidate_indices=np.asarray(target_indices, dtype=np.int16),
        legal_action_counts=np.asarray(legal_counts, dtype=np.int32),
        scenario_counts=np.asarray(scenario_counts, dtype=np.uint16),
        support_exhaustive=np.asarray(exhaustive, dtype=np.bool_),
        scenario_grid_complete=np.asarray(complete, dtype=np.bool_),
        scenario_support_modes=np.asarray(support_modes, dtype=np.uint8),
        leaf_bootstrapped=np.asarray(bootstrapped, dtype=np.bool_),
        policy_versions=np.asarray(policy_versions, dtype=np.int32),
        fingerprints=np.asarray(
            [[tuple(raw) for raw in row] for row in fingerprints],
            dtype=np.uint8,
        ),
        masks=np.asarray(masks, dtype=np.bool_),
    )
    validate_macro_teacher_array_block(block)
    return block


def validate_macro_teacher_array_block(block: MacroTeacherArrayBlock) -> None:
    """Reject incomplete sparse rows and partial native grids."""
    decisions = block.decision_count
    candidates = block.candidate_count
    expected = (
        (block.decision_candidate_offsets, np.int32, (decisions + 1,)),
        (block.candidate_action_offsets, np.int32, (candidates + 1,)),
        (
            block.candidate_action_indices,
            np.int32,
            (int(block.candidate_action_offsets[-1]),),
        ),
        (
            block.candidate_aggregate_features,
            np.float32,
            (candidates, SEARCH_EVIDENCE_FEATURE_SIZE),
        ),
        (
            block.candidate_effect_features,
            np.float32,
            (candidates, DYNAMIC_EFFECT_FEATURE_SIZE),
        ),
        (block.candidate_robust_scores, np.float32, (candidates,)),
        (block.candidate_endpoint_probabilities, np.float32, (candidates, 3)),
        (block.candidate_rules_exact, np.bool_, (candidates,)),
        (block.target_candidate_indices, np.int16, (decisions,)),
        (block.legal_action_counts, np.int32, (decisions,)),
        (block.scenario_counts, np.uint16, (decisions,)),
        (block.support_exhaustive, np.bool_, (decisions,)),
        (block.scenario_grid_complete, np.bool_, (decisions,)),
        (block.scenario_support_modes, np.uint8, (decisions,)),
        (block.leaf_bootstrapped, np.bool_, (decisions,)),
        (block.policy_versions, np.int32, (decisions,)),
        (block.fingerprints, np.uint8, (decisions, _FINGERPRINT_COUNT, 32)),
        (block.masks, np.bool_, (decisions,)),
    )
    for values, dtype, shape in expected:
        if values.dtype != dtype or values.shape != shape:
            raise ValueError("macro teacher array field is misaligned")
    for offsets in (
        block.decision_candidate_offsets,
        block.candidate_action_offsets,
    ):
        if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
            raise ValueError("macro teacher CSR offsets are invalid")
    if int(block.decision_candidate_offsets[-1]) != candidates:
        raise ValueError("macro teacher candidate offsets are misaligned")
    floating = (
        block.candidate_aggregate_features,
        block.candidate_effect_features,
        block.candidate_robust_scores,
        block.candidate_endpoint_probabilities,
    )
    if any(not bool(np.isfinite(values).all()) for values in floating):
        raise ValueError("macro teacher arrays must be finite")
    for decision_index in range(decisions):
        start = int(block.decision_candidate_offsets[decision_index])
        stop = int(block.decision_candidate_offsets[decision_index + 1])
        if not bool(block.masks[decision_index]):
            if (
                start != stop
                or int(block.target_candidate_indices[decision_index]) != -1
                or int(block.policy_versions[decision_index]) != -1
            ):
                raise ValueError("absent macro teacher row carries candidates")
            continue
        if stop <= start or not bool(block.scenario_grid_complete[decision_index]):
            raise ValueError("resolved macro teacher row requires a complete grid")
        target = int(block.target_candidate_indices[decision_index])
        if target < 0 or target >= stop - start:
            raise ValueError("macro teacher target candidate is out of range")
        block.evidence_at(decision_index)


def macro_teacher_array_field_names() -> tuple[str, ...]:
    """Return ndarray field names in stable transport order."""
    return tuple(MacroTeacherArrayBlock.__dataclass_fields__)


__all__ = [
    "MacroTeacherArrayBlock",
    "build_macro_teacher_array_block",
    "macro_teacher_array_field_names",
    "validate_macro_teacher_array_block",
]
