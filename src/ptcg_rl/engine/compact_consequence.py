"""Contiguous, engine-grounded consequence buffers.

Native code emits only root-observable engine-state bytes and transition
metadata.  Python derives the exact 33-value effect and later combines the
observation with prompt context and belief to build the model-facing
``RootInformationStateTensorBatch``.  Native execution cannot construct that
information-state tensor by itself.

The layout is candidate-major: ``candidate * scenario_count + scenario``.
Row access returns NumPy slices over the original buffers, not per-cell copies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.consequence_schema import (
    COMPACT_CONSEQUENCE_SCHEMA_VERSION,
    ROOT_OBSERVATION_ENCODING,
    CompactConsequenceMetadata,
    ConsequenceExactnessFlags,
    RootCandidateScenarioIdentity,
    ScenarioHandle,
    ScenarioSupport,
    ScenarioSupportMode,
    SemanticEndpoint,
    root_candidate_scenario_fingerprint,
    scenario_fingerprint,
    scenario_support_fingerprint,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE

COMPACT_CONSEQUENCE_METADATA_WIDTH = 6
CELL_ENDPOINT_COLUMN = 0
CELL_TRANSITION_STEPS_COLUMN = 1
CELL_ERROR_CODE_COLUMN = 2
CELL_VALID_MASK_COLUMN = 3
CELL_RULES_EXACT_MASK_COLUMN = 4
CELL_ENGINE_RESULT_COLUMN = 5

UInt8Array = npt.NDArray[np.uint8]
Int32Array = npt.NDArray[np.int32]
Float32Array = npt.NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class RootObservableEngineStateView:
    """Zero-copy view of one native root-observable numeric-JSON payload."""

    payload: UInt8Array
    encoding: str = ROOT_OBSERVATION_ENCODING

    def as_memoryview(self) -> memoryview:
        """Return a zero-copy byte-oriented view for a downstream tensorizer."""
        return self.payload.data


@dataclass(frozen=True, slots=True)
class CompactConsequenceRowView:
    """Zero-copy numeric views and validated identity for one grid cell."""

    identity: RootCandidateScenarioIdentity
    root_observable_state: RootObservableEngineStateView
    exact_effect: Float32Array
    metadata: Int32Array

    @property
    def endpoint(self) -> SemanticEndpoint:
        """Return the semantic endpoint encoded by native metadata."""
        return SemanticEndpoint(int(self.metadata[CELL_ENDPOINT_COLUMN]))

    @property
    def transition_steps(self) -> int:
        """Return the number of primitive engine transitions consumed."""
        return int(self.metadata[CELL_TRANSITION_STEPS_COLUMN])

    @property
    def error_code(self) -> int:
        """Return zero on success or the native adapter error code."""
        return int(self.metadata[CELL_ERROR_CODE_COLUMN])

    @property
    def valid(self) -> bool:
        """Return whether the cell reached a consumable semantic endpoint."""
        return bool(self.metadata[CELL_VALID_MASK_COLUMN])

    @property
    def rules_exact(self) -> bool:
        """Return whether bundled-engine execution produced the transition."""
        return bool(self.metadata[CELL_RULES_EXACT_MASK_COLUMN])

    @property
    def engine_result(self) -> int:
        """Return winner seat 0/1, draw 2, or -1 for a nonterminal cell."""
        return int(self.metadata[CELL_ENGINE_RESULT_COLUMN])


@dataclass(frozen=True, slots=True)
class CompactConsequenceBatch:
    """Contiguous candidate-major consequence buffers from one root request."""

    contract: CompactConsequenceMetadata
    root_observation_offsets: Int32Array
    root_observation_bytes: UInt8Array
    exact_effects: Float32Array
    cell_metadata: Int32Array

    def __post_init__(self) -> None:
        """Reject corrupt or semantically inconsistent native/Python buffers."""
        validate_compact_consequence_batch(self)

    @property
    def candidate_count(self) -> int:
        """Return the retained root candidate count."""
        return len(self.contract.candidate_fingerprints)

    @property
    def scenario_count(self) -> int:
        """Return the common paired-scenario count."""
        return len(self.contract.scenario_support.scenarios)

    @property
    def cell_count(self) -> int:
        """Return the complete candidate-by-scenario grid size."""
        return self.candidate_count * self.scenario_count

    @property
    def scenario_weights(self) -> npt.NDArray[np.float64]:
        """Materialize normalized support weights for aggregation."""
        return np.fromiter(
            (item.weight for item in self.contract.scenario_support.scenarios),
            dtype=np.float64,
            count=self.scenario_count,
        )

    def identity_at(
        self,
        candidate_index: int,
        scenario_index: int,
    ) -> RootCandidateScenarioIdentity:
        """Build the validated immutable identity for one grid position."""
        self._cell_index(candidate_index, scenario_index)
        support = self.contract.scenario_support
        candidate_fingerprint = self.contract.candidate_fingerprints[
            candidate_index
        ]
        scenario_identity = support.scenarios[scenario_index].scenario_fingerprint
        identity_fingerprint = root_candidate_scenario_fingerprint(
            root_state_fingerprint=self.contract.root_state_fingerprint,
            candidate_fingerprint=candidate_fingerprint,
            scenario_fingerprint=scenario_identity,
            scenario_support_fingerprint=support.support_fingerprint,
        )
        return RootCandidateScenarioIdentity(
            candidate_index=candidate_index,
            scenario_index=scenario_index,
            root_state_fingerprint=self.contract.root_state_fingerprint,
            candidate_fingerprint=candidate_fingerprint,
            scenario_fingerprint=scenario_identity,
            scenario_support_fingerprint=support.support_fingerprint,
            identity_fingerprint=identity_fingerprint,
        )

    def row_at(
        self,
        candidate_index: int,
        scenario_index: int,
    ) -> CompactConsequenceRowView:
        """Return zero-copy observation, effect, and metadata row views."""
        cell_index = self._cell_index(candidate_index, scenario_index)
        start = int(self.root_observation_offsets[cell_index])
        stop = int(self.root_observation_offsets[cell_index + 1])
        return CompactConsequenceRowView(
            identity=self.identity_at(candidate_index, scenario_index),
            root_observable_state=RootObservableEngineStateView(
                self.root_observation_bytes[start:stop]
            ),
            exact_effect=self.exact_effects[cell_index],
            metadata=self.cell_metadata[cell_index],
        )

    def _cell_index(self, candidate_index: int, scenario_index: int) -> int:
        if not 0 <= candidate_index < self.candidate_count:
            raise IndexError("candidate index is outside the compact consequence grid")
        if not 0 <= scenario_index < self.scenario_count:
            raise IndexError("scenario index is outside the compact consequence grid")
        return candidate_index * self.scenario_count + scenario_index


def validate_compact_consequence_batch(batch: CompactConsequenceBatch) -> None:
    """Validate buffer layout, grid completeness, and exact rules metadata."""
    cell_count = batch.cell_count
    _require_array(
        batch.root_observation_offsets,
        name="root_observation_offsets",
        dtype=np.dtype(np.int32),
        shape=(cell_count + 1,),
    )
    _require_array(
        batch.root_observation_bytes,
        name="root_observation_bytes",
        dtype=np.dtype(np.uint8),
        shape=(int(batch.root_observation_offsets[-1]),),
    )
    _require_array(
        batch.exact_effects,
        name="exact_effects",
        dtype=np.dtype(np.float32),
        shape=(cell_count, DYNAMIC_EFFECT_FEATURE_SIZE),
    )
    _require_array(
        batch.cell_metadata,
        name="cell_metadata",
        dtype=np.dtype(np.int32),
        shape=(cell_count, COMPACT_CONSEQUENCE_METADATA_WIDTH),
    )

    offsets = batch.root_observation_offsets
    if int(offsets[0]) != 0 or bool(np.any(offsets[1:] < offsets[:-1])):
        raise ValueError("root observation offsets must start at zero and be monotonic")
    if not bool(np.isfinite(batch.exact_effects).all()):
        raise ValueError("exact effect rows must be finite")

    metadata = batch.cell_metadata
    endpoint_values = metadata[:, CELL_ENDPOINT_COLUMN]
    valid_endpoint_values = np.asarray(
        [int(endpoint) for endpoint in SemanticEndpoint],
        dtype=np.int32,
    )
    if not bool(np.isin(endpoint_values, valid_endpoint_values).all()):
        raise ValueError("cell metadata contains an unknown semantic endpoint")
    if bool(np.any(metadata[:, CELL_TRANSITION_STEPS_COLUMN] < 0)):
        raise ValueError("transition step counts must be non-negative")
    for column, name in (
        (CELL_VALID_MASK_COLUMN, "valid mask"),
        (CELL_RULES_EXACT_MASK_COLUMN, "rules-exact mask"),
    ):
        if not bool(np.isin(metadata[:, column], (0, 1)).all()):
            raise ValueError(f"{name} values must be zero or one")

    valid_mask = metadata[:, CELL_VALID_MASK_COLUMN].astype(np.bool_, copy=False)
    rules_exact_mask = metadata[:, CELL_RULES_EXACT_MASK_COLUMN].astype(
        np.bool_, copy=False
    )
    error_codes = metadata[:, CELL_ERROR_CODE_COLUMN]
    if bool(np.any(error_codes < 0)):
        raise ValueError("cell error codes must be non-negative")
    resolved_endpoint_mask = endpoint_values != int(SemanticEndpoint.INVALID)
    if bool(np.any(rules_exact_mask & ~valid_mask)):
        raise ValueError("rules-exact cells must also be valid")
    if bool(np.any(valid_mask != (error_codes == 0))):
        raise ValueError("valid mask must agree with zero versus nonzero error code")
    if bool(np.any(valid_mask != resolved_endpoint_mask)):
        raise ValueError("valid mask must agree with resolved versus invalid endpoint")

    engine_results = metadata[:, CELL_ENGINE_RESULT_COLUMN]
    if bool(np.any((engine_results < -1) | (engine_results > 2))):
        raise ValueError("engine results must be -1, winner seat 0/1, or draw 2")
    terminal_mask = endpoint_values == int(SemanticEndpoint.TERMINAL)
    if bool(np.any(terminal_mask & (engine_results < 0))):
        raise ValueError("terminal cells must carry an engine result")
    if bool(np.any(~terminal_mask & (engine_results != -1))):
        raise ValueError("nonterminal and invalid cells cannot carry an engine result")

    observation_lengths = offsets[1:] - offsets[:-1]
    if bool(np.any(valid_mask != (observation_lengths > 0))):
        raise ValueError("only valid cells may carry root-observable state bytes")
    if bool(np.any(batch.exact_effects[~valid_mask] != 0.0)):
        raise ValueError("invalid cells must carry zeroed exact effect rows")

    grid_complete = bool(valid_mask.all())
    if batch.contract.scenario_grid_complete != grid_complete:
        raise ValueError("scenario_grid_complete must match the full valid cell grid")


def _require_array(
    values: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> None:
    if not isinstance(values, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if values.dtype != dtype:
        raise TypeError(f"{name} must use {dtype.name}")
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if values.flags.writeable:
        raise ValueError(f"{name} must be read-only")


__all__ = [
    "CELL_ENGINE_RESULT_COLUMN",
    "CELL_ENDPOINT_COLUMN",
    "CELL_ERROR_CODE_COLUMN",
    "CELL_RULES_EXACT_MASK_COLUMN",
    "CELL_TRANSITION_STEPS_COLUMN",
    "CELL_VALID_MASK_COLUMN",
    "COMPACT_CONSEQUENCE_METADATA_WIDTH",
    "COMPACT_CONSEQUENCE_SCHEMA_VERSION",
    "ROOT_OBSERVATION_ENCODING",
    "CompactConsequenceBatch",
    "CompactConsequenceMetadata",
    "CompactConsequenceRowView",
    "ConsequenceExactnessFlags",
    "RootCandidateScenarioIdentity",
    "RootObservableEngineStateView",
    "ScenarioHandle",
    "ScenarioSupport",
    "ScenarioSupportMode",
    "SemanticEndpoint",
    "root_candidate_scenario_fingerprint",
    "scenario_fingerprint",
    "scenario_support_fingerprint",
    "validate_compact_consequence_batch",
]
