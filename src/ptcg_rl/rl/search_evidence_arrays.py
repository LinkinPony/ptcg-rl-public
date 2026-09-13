"""Compact ragged arrays for complete-action search evidence."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ptcg_rl.engine.search_evidence import (
    SEARCH_EVIDENCE_FEATURE_SIZE,
    SearchCandidateEvidence,
    SearchEvidence,
)


@dataclass(frozen=True)
class SearchEvidenceArrayBlock:
    """Two-level CSR representation: decision to candidate to option index."""

    decision_candidate_offsets: np.ndarray
    candidate_action_offsets: np.ndarray
    candidate_action_indices: np.ndarray
    candidate_features: np.ndarray
    legal_action_counts: np.ndarray
    world_counts: np.ndarray
    exhaustive: np.ndarray
    exact: np.ndarray
    masks: np.ndarray

    def evidence_at(self, index: int) -> SearchEvidence | None:
        """Reconstruct one immutable public evidence record."""
        if not bool(self.masks[index]):
            return None
        candidate_start = int(self.decision_candidate_offsets[index])
        candidate_stop = int(self.decision_candidate_offsets[index + 1])
        candidates = []
        for candidate_index in range(candidate_start, candidate_stop):
            action_start = int(self.candidate_action_offsets[candidate_index])
            action_stop = int(self.candidate_action_offsets[candidate_index + 1])
            candidates.append(
                SearchCandidateEvidence(
                    action=tuple(
                        int(value)
                        for value in self.candidate_action_indices[
                            action_start:action_stop
                        ]
                    ),
                    features=tuple(
                        float(value)
                        for value in self.candidate_features[candidate_index]
                    ),
                )
            )
        return SearchEvidence(
            candidates=tuple(candidates),
            legal_action_count=int(self.legal_action_counts[index]),
            world_count=int(self.world_counts[index]),
            exhaustive=bool(self.exhaustive[index]),
            exact=bool(self.exact[index]),
        )


def build_search_evidence_array_block(
    evidence: Sequence[SearchEvidence | None],
) -> SearchEvidenceArrayBlock | None:
    """Flatten sparse decision evidence without materializing engine state."""
    if not any(item is not None for item in evidence):
        return None
    decision_offsets = [0]
    action_offsets = [0]
    action_indices: list[int] = []
    feature_rows: list[tuple[float, ...]] = []
    legal_action_counts: list[int] = []
    world_counts: list[int] = []
    exhaustive: list[bool] = []
    exact: list[bool] = []
    masks: list[bool] = []
    for item in evidence:
        masks.append(item is not None)
        legal_action_counts.append(0 if item is None else item.legal_action_count)
        world_counts.append(0 if item is None else item.world_count)
        exhaustive.append(False if item is None else item.exhaustive)
        exact.append(False if item is None else item.exact)
        if item is not None:
            for candidate in item.candidates:
                action_indices.extend(candidate.action)
                action_offsets.append(len(action_indices))
                feature_rows.append(candidate.features)
        decision_offsets.append(len(feature_rows))
    block = SearchEvidenceArrayBlock(
        decision_candidate_offsets=np.asarray(decision_offsets, dtype=np.int32),
        candidate_action_offsets=np.asarray(action_offsets, dtype=np.int32),
        candidate_action_indices=np.asarray(action_indices, dtype=np.int32),
        candidate_features=np.asarray(feature_rows, dtype=np.float32).reshape(
            -1,
            SEARCH_EVIDENCE_FEATURE_SIZE,
        ),
        legal_action_counts=np.asarray(legal_action_counts, dtype=np.int64),
        world_counts=np.asarray(world_counts, dtype=np.int32),
        exhaustive=np.asarray(exhaustive, dtype=np.bool_),
        exact=np.asarray(exact, dtype=np.bool_),
        masks=np.asarray(masks, dtype=np.bool_),
    )
    validate_search_evidence_array_block(block, decision_count=len(evidence))
    return block


def validate_search_evidence_array_block(
    block: SearchEvidenceArrayBlock,
    *,
    decision_count: int,
) -> None:
    """Reject corrupt offsets, metadata, and non-finite feature rows."""
    decision_offsets = block.decision_candidate_offsets
    action_offsets = block.candidate_action_offsets
    action_indices = block.candidate_action_indices
    features = block.candidate_features
    if decision_offsets.dtype != np.int32:
        raise TypeError("search decision offsets must use int32")
    if decision_offsets.shape != (decision_count + 1,):
        raise ValueError("search decision offsets must align with decisions")
    if int(decision_offsets[0]) != 0 or bool(
        np.any(decision_offsets[1:] < decision_offsets[:-1])
    ):
        raise ValueError("search decision offsets must start at zero and be monotonic")
    candidate_count = int(decision_offsets[-1])
    if action_offsets.dtype != np.int32:
        raise TypeError("search candidate action offsets must use int32")
    if action_offsets.shape != (candidate_count + 1,):
        raise ValueError("search candidate action offsets must align with candidates")
    if int(action_offsets[0]) != 0 or bool(
        np.any(action_offsets[1:] < action_offsets[:-1])
    ):
        raise ValueError("search action offsets must start at zero and be monotonic")
    if action_indices.dtype != np.int32 or action_indices.shape != (
        int(action_offsets[-1]),
    ):
        raise ValueError("search action indices must be int32 and align with offsets")
    if features.dtype != np.float32 or features.shape != (
        candidate_count,
        SEARCH_EVIDENCE_FEATURE_SIZE,
    ):
        raise ValueError("search candidate features have an invalid shape or dtype")
    if not bool(np.isfinite(features).all()):
        raise ValueError("search candidate features must be finite")
    expected_shape = (decision_count,)
    if block.legal_action_counts.dtype != np.int64:
        raise TypeError("search legal action counts must use int64")
    if block.world_counts.dtype != np.int32:
        raise TypeError("search world counts must use int32")
    for name, values in (
        ("legal_action_counts", block.legal_action_counts),
        ("world_counts", block.world_counts),
    ):
        if values.shape != expected_shape:
            raise ValueError(f"search {name} must align with decisions")
    for name, values in (
        ("exhaustive", block.exhaustive),
        ("exact", block.exact),
        ("masks", block.masks),
    ):
        if values.dtype != np.bool_ or values.shape != expected_shape:
            raise ValueError(f"search {name} must be bool and align with decisions")
    if not bool(block.masks.any()):
        raise ValueError("search evidence blocks require at least one valid row")
    for index in range(decision_count):
        start = int(decision_offsets[index])
        stop = int(decision_offsets[index + 1])
        if not bool(block.masks[index]):
            if stop != start:
                raise ValueError("masked search rows cannot carry candidates")
            if (
                block.legal_action_counts[index] != 0
                or block.world_counts[index] != 0
                or block.exhaustive[index]
                or block.exact[index]
            ):
                raise ValueError("masked search rows must have zero metadata")
            continue
        if stop <= start:
            raise ValueError("valid search rows require at least one candidate")
        # Reconstruction applies the authoritative SearchEvidence invariants.
        block.evidence_at(index)


__all__ = [
    "SearchEvidenceArrayBlock",
    "build_search_evidence_array_block",
    "validate_search_evidence_array_block",
]
