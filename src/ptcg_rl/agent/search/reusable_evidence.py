"""Ephemeral request-bound compact evidence reusable within one planner call."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.compact_consequence import (
    CELL_ENDPOINT_COLUMN,
    CELL_RULES_EXACT_MASK_COLUMN,
    CELL_VALID_MASK_COLUMN,
    CompactConsequenceBatch,
    SemanticEndpoint,
)
from ptcg_rl.engine.consequence_identity import candidate_action_fingerprint
from ptcg_rl.engine.consequence_request_identity import PreparedRequestIdentity

BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ReusableCompactEvidenceEnvelope:
    """Validated native result retained only for request-local reuse.

    The envelope contains public complete actions, compact root-observable
    buffers, and opaque validation identities.  It deliberately carries no
    hidden-world material, engine state, or replay serialization method.
    """

    batch: CompactConsequenceBatch
    request_identity: PreparedRequestIdentity
    candidate_actions: tuple[tuple[int, ...], ...]
    cell_comparable_mask: BoolArray
    candidate_rankable_mask: BoolArray

    def __post_init__(self) -> None:
        """Bind the cached buffers to the exact native request and candidates."""
        _validate_request_binding(self.batch, self.request_identity)
        fingerprints = tuple(
            candidate_action_fingerprint(action) for action in self.candidate_actions
        )
        if fingerprints != self.request_identity.candidate_fingerprints:
            raise ValueError("reusable evidence actions differ from request identity")
        if len(set(self.candidate_actions)) != len(self.candidate_actions):
            raise ValueError("reusable evidence actions must be unique")
        _require_readonly_bool_array(
            self.cell_comparable_mask,
            name="cell_comparable_mask",
            shape=(self.batch.cell_count,),
        )
        _require_readonly_bool_array(
            self.candidate_rankable_mask,
            name="candidate_rankable_mask",
            shape=(self.batch.candidate_count,),
        )
        derived_comparable = _derived_comparable_mask(self.batch)
        if not bool(np.array_equal(self.cell_comparable_mask, derived_comparable)):
            raise ValueError("reusable evidence comparable mask differs from batch")
        derived_rankable = derived_comparable.reshape(
            self.batch.candidate_count,
            self.batch.scenario_count,
        ).all(axis=1)
        if not bool(np.array_equal(self.candidate_rankable_mask, derived_rankable)):
            raise ValueError("reusable evidence rankable mask differs from batch")

    @property
    def request_fingerprint(self) -> str:
        """Return the full producer contract fingerprint for cache identity."""
        return self.request_identity.contract_fingerprint

    @property
    def root_player(self) -> int:
        """Return the immutable root player of the cached native request."""
        return self.request_identity.root_player

    def candidate_index(self, action: tuple[int, ...]) -> int | None:
        """Return a cached candidate row, or ``None`` when execution is needed."""
        fingerprint = candidate_action_fingerprint(action)
        try:
            return self.request_identity.candidate_fingerprints.index(fingerprint)
        except ValueError:
            return None


def reusable_compact_evidence(
    *,
    batch: CompactConsequenceBatch,
    request_identity: PreparedRequestIdentity,
    candidate_actions: tuple[tuple[int, ...], ...],
) -> ReusableCompactEvidenceEnvelope:
    """Create a request-local reuse handle with masks derived from the batch."""
    comparable = _derived_comparable_mask(batch)
    rankable = np.asarray(
        comparable.reshape(batch.candidate_count, batch.scenario_count).all(axis=1),
        dtype=np.bool_,
    )
    comparable.setflags(write=False)
    rankable.setflags(write=False)
    return ReusableCompactEvidenceEnvelope(
        batch=batch,
        request_identity=request_identity,
        candidate_actions=candidate_actions,
        cell_comparable_mask=comparable,
        candidate_rankable_mask=rankable,
    )


def _validate_request_binding(
    batch: CompactConsequenceBatch,
    identity: PreparedRequestIdentity,
) -> None:
    contract = batch.contract
    support = contract.scenario_support
    if contract.root_state_fingerprint != identity.root_state_fingerprint:
        raise ValueError("reusable evidence root state differs from request")
    if contract.observation_schema_fingerprint != (
        identity.root_observation_schema_fingerprint
    ):
        raise ValueError("reusable evidence observation schema differs from request")
    if contract.candidate_fingerprints != identity.candidate_fingerprints:
        raise ValueError("reusable evidence candidates differ from request")
    if contract.legal_action_count != identity.legal_action_count:
        raise ValueError("reusable evidence legal-action count differs from request")
    if support.support_fingerprint != identity.scenario_support_fingerprint:
        raise ValueError("reusable evidence scenario support differs from request")
    if support.mode is not identity.support_mode:
        raise ValueError("reusable evidence support mode differs from request")
    scenario_fingerprints = tuple(
        scenario.scenario_fingerprint for scenario in support.scenarios
    )
    if scenario_fingerprints != identity.scenario_fingerprints:
        raise ValueError("reusable evidence scenarios differ from request")
    handle_pairs = tuple(
        (scenario.belief_world_handle, scenario.chance_support_handle)
        for scenario in support.scenarios
    )
    if handle_pairs != identity.scenario_handle_pairs:
        raise ValueError("reusable evidence handles differ from request")


def _derived_comparable_mask(batch: CompactConsequenceBatch) -> BoolArray:
    metadata = batch.cell_metadata
    endpoints = metadata[:, CELL_ENDPOINT_COLUMN]
    return np.asarray(
        (metadata[:, CELL_VALID_MASK_COLUMN] == 1)
        & (metadata[:, CELL_RULES_EXACT_MASK_COLUMN] == 1)
        & np.isin(
            endpoints,
            (
                int(SemanticEndpoint.TERMINAL),
                int(SemanticEndpoint.SAME_SEAT_MAIN),
                int(SemanticEndpoint.TURN_HANDOFF),
            ),
        ),
        dtype=np.bool_,
    )


def _require_readonly_bool_array(
    values: np.ndarray,
    *,
    name: str,
    shape: tuple[int, ...],
) -> None:
    if not isinstance(values, np.ndarray) or values.dtype != np.dtype(np.bool_):
        raise TypeError(f"{name} must be a bool NumPy array")
    if values.shape != shape:
        raise ValueError(f"{name} has the wrong shape")
    if values.flags.writeable:
        raise ValueError(f"{name} must be read-only")


__all__ = [
    "ReusableCompactEvidenceEnvelope",
    "reusable_compact_evidence",
]
