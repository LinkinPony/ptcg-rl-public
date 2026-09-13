"""Splice request-bound equivalence-probe cells into a planner baseline grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.agent.search.reusable_evidence import ReusableCompactEvidenceEnvelope
from ptcg_rl.engine.consequence_request_identity import PreparedRequestIdentity

Int32Array = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class ReusedPlannerSeedGrid:
    """Gather map proving which baseline engine cells need no re-execution."""

    retained_candidate_to_cached_candidate: Int32Array
    retained_cell_to_cached_cell: Int32Array
    reused_candidate_count: int
    reused_transition_count: int
    request_fingerprint: str

    def __post_init__(self) -> None:
        """Require immutable gather maps and exact accounting."""
        for name, values in (
            (
                "retained_candidate_to_cached_candidate",
                self.retained_candidate_to_cached_candidate,
            ),
            ("retained_cell_to_cached_cell", self.retained_cell_to_cached_cell),
        ):
            if values.dtype != np.dtype(np.int32):
                raise TypeError(f"{name} must use int32")
            if values.flags.writeable:
                raise ValueError(f"{name} must be read-only")
        if self.retained_candidate_to_cached_candidate.ndim != 1:
            raise ValueError("candidate reuse map must be one-dimensional")
        if self.retained_cell_to_cached_cell.ndim != 2:
            raise ValueError("cell reuse map must be candidate-by-scenario")
        if (
            self.retained_cell_to_cached_cell.shape[0]
            != (self.retained_candidate_to_cached_candidate.shape[0])
        ):
            raise ValueError("candidate and cell reuse maps differ")
        if self.reused_candidate_count != int(
            np.count_nonzero(self.retained_candidate_to_cached_candidate >= 0)
        ):
            raise ValueError("reused candidate accounting differs from gather map")
        if self.reused_transition_count != int(
            np.count_nonzero(self.retained_cell_to_cached_cell >= 0)
        ):
            raise ValueError("reused transition accounting differs from gather map")


def splice_reusable_probe_evidence(
    *,
    envelope: ReusableCompactEvidenceEnvelope,
    expected_request_identity: PreparedRequestIdentity,
    retained_actions: tuple[tuple[int, ...], ...],
) -> ReusedPlannerSeedGrid:
    """Map cached full-grid rows into one request without rerunning native work."""
    if envelope.request_identity != expected_request_identity:
        raise ValueError("reusable probe evidence belongs to another request")
    if envelope.request_fingerprint != expected_request_identity.contract_fingerprint:
        raise ValueError("reusable probe request fingerprint mismatch")
    if len(set(retained_actions)) != len(retained_actions):
        raise ValueError("retained planner actions must be unique")
    scenario_count = envelope.batch.scenario_count
    candidate_map = np.full(len(retained_actions), -1, dtype=np.int32)
    cell_map = np.full((len(retained_actions), scenario_count), -1, dtype=np.int32)
    for retained_index, action in enumerate(retained_actions):
        cached_index = envelope.candidate_index(action)
        if cached_index is None or not bool(
            envelope.candidate_rankable_mask[cached_index]
        ):
            continue
        candidate_map[retained_index] = cached_index
        cell_start = cached_index * scenario_count
        cell_map[retained_index] = np.arange(
            cell_start,
            cell_start + scenario_count,
            dtype=np.int32,
        )
    candidate_map.setflags(write=False)
    cell_map.setflags(write=False)
    return ReusedPlannerSeedGrid(
        retained_candidate_to_cached_candidate=candidate_map,
        retained_cell_to_cached_cell=cell_map,
        reused_candidate_count=int(np.count_nonzero(candidate_map >= 0)),
        reused_transition_count=int(np.count_nonzero(cell_map >= 0)),
        request_fingerprint=envelope.request_fingerprint,
    )


__all__ = ["ReusedPlannerSeedGrid", "splice_reusable_probe_evidence"]
