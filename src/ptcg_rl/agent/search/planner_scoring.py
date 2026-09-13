"""One fingerprinted leaf-scoring contract shared by training and serving."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from typing import Any, Generic, Literal, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.root_information import (
    RootInformationLeafBatch,
    RootInformationStateTensorBatch,
    RootInformationTensorizer,
)
from ptcg_rl.engine.compact_consequence import (
    CELL_ENDPOINT_COLUMN,
    CELL_ENGINE_RESULT_COLUMN,
    CELL_RULES_EXACT_MASK_COLUMN,
    CompactConsequenceBatch,
    SemanticEndpoint,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.runtime.planner_telemetry import (
    PlannerRequestTelemetry,
    PlannerStage,
    PlannerStageEvent,
)

_SCORER_DOMAIN = b"ptcg-rl/shared-leaf-scorer/v1\x00"
TensorInputs = TypeVar("TensorInputs")
Float32Array = npt.NDArray[np.float32]
Int32Array = npt.NDArray[np.int32]
BoolArray = npt.NDArray[np.bool_]


class PlannerScoringConfig(BaseModel):
    """Fixed leaf, robust aggregation, and planner-target semantics."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: Literal[1] = 1
    engine_tiebreak_weight: float = Field(ge=0.0)
    engine_tiebreak_feature_weights: tuple[float, ...]
    score_clip: float = Field(gt=0.0, le=1.0)
    risk_std_weight: float = Field(ge=0.0)
    planner_score_weight: float = Field(ge=0.0)
    planner_score_scale: float = Field(gt=0.0)
    planner_temperature: float = Field(gt=0.0)

    @field_validator(
        "engine_tiebreak_weight",
        "score_clip",
        "risk_std_weight",
        "planner_score_weight",
        "planner_score_scale",
        "planner_temperature",
    )
    @classmethod
    def finite_float(cls, value: float) -> float:
        """Reject non-replayable NaN and infinite scorer parameters."""
        if not math.isfinite(value):
            raise ValueError("planner scoring parameters must be finite")
        return value

    @field_validator("engine_tiebreak_feature_weights")
    @classmethod
    def valid_engine_tiebreak_feature_weights(
        cls,
        values: tuple[float, ...],
    ) -> tuple[float, ...]:
        """Bind the generic exact-effect utility to the stable feature width."""
        if len(values) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError("engine tiebreak feature weights have the wrong width")
        weights = np.asarray(values, dtype=np.float32)
        if not bool(np.isfinite(weights).all()):
            raise ValueError("engine tiebreak feature weights must be finite")
        return tuple(float(value) for value in weights)

    @property
    def scorer_fingerprint(self) -> str:
        """Return the train/serve identity of all decision-score semantics."""
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(_SCORER_DOMAIN + payload).hexdigest()


@dataclass(frozen=True, slots=True)
class LeafScoringBatch:
    """Compact candidate-major inputs aligned with a deduplicated leaf batch."""

    endpoints: Int32Array
    engine_results: Int32Array
    exact_effects: Float32Array
    rules_exact_mask: BoolArray
    root_player: int

    def __post_init__(self) -> None:
        """Validate immutable aligned scoring columns."""
        cell_count = int(self.endpoints.shape[0])
        _require_array(
            self.endpoints,
            name="endpoints",
            dtype=np.dtype(np.int32),
            shape=(cell_count,),
        )
        _require_array(
            self.engine_results,
            name="engine_results",
            dtype=np.dtype(np.int32),
            shape=(cell_count,),
        )
        _require_array(
            self.exact_effects,
            name="exact_effects",
            dtype=np.dtype(np.float32),
            shape=(cell_count, DYNAMIC_EFFECT_FEATURE_SIZE),
        )
        _require_array(
            self.rules_exact_mask,
            name="rules_exact_mask",
            dtype=np.dtype(np.bool_),
            shape=(cell_count,),
        )
        if not bool(np.isfinite(self.exact_effects).all()):
            raise ValueError("exact_effects must be finite")
        if isinstance(self.root_player, bool) or self.root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")


class RootInformationValueProvider(Protocol[TensorInputs]):
    """Model surface used by the shared semantic leaf scorer."""

    def values(
        self,
        batch: RootInformationStateTensorBatch[TensorInputs],
    ) -> npt.ArrayLike:
        """Return root-perspective values aligned with unique leaves."""


def leaf_scoring_batch_from_compact(
    batch: CompactConsequenceBatch,
    *,
    root_player: int,
) -> LeafScoringBatch:
    """Project compact engine columns into the sole semantic scoring input."""
    if isinstance(root_player, bool) or root_player not in (0, 1):
        raise ValueError("root_player must be 0 or 1")
    endpoints = batch.cell_metadata[:, CELL_ENDPOINT_COLUMN]
    engine_results = batch.cell_metadata[:, CELL_ENGINE_RESULT_COLUMN]
    rules_exact = batch.cell_metadata[:, CELL_RULES_EXACT_MASK_COLUMN].astype(
        np.bool_,
        copy=True,
    )
    rules_exact.setflags(write=False)
    return LeafScoringBatch(
        endpoints=endpoints,
        engine_results=engine_results,
        exact_effects=batch.exact_effects,
        rules_exact_mask=rules_exact,
        root_player=root_player,
    )


@dataclass(frozen=True, slots=True)
class SharedLeafScoreResult:
    """Immutable per-cell scores and unique inferred value rows."""

    cell_scores: Float32Array
    unique_leaf_values: Float32Array
    scorer_fingerprint: str
    leaf_bootstrapped: bool


class SharedRootInformationLeafScorer(Generic[TensorInputs]):
    """Execute identical semantic leaf scoring in collection and learning."""

    def __init__(
        self,
        *,
        config: PlannerScoringConfig,
        tensorizer: RootInformationTensorizer[TensorInputs],
        value_provider: RootInformationValueProvider[TensorInputs],
        telemetry: PlannerRequestTelemetry | None = None,
    ) -> None:
        self.config = config
        self._tensorizer = tensorizer
        self._value_provider = value_provider
        self._telemetry = telemetry
        self._leaf_lookup_rows = 0
        self._unique_leaf_rows = 0
        self._consequence_cells = 0

    @property
    def reuse_stats(self) -> tuple[int, int, int]:
        """Return leaf lookups, unique model rows, and consequence cells."""
        return (
            self._leaf_lookup_rows,
            self._unique_leaf_rows,
            self._consequence_cells,
        )

    @property
    def scorer_fingerprint(self) -> str:
        """Return the immutable train/serve scorer identity."""
        return self.config.scorer_fingerprint

    def score(
        self,
        *,
        leaves: RootInformationLeafBatch,
        cells: LeafScoringBatch,
    ) -> SharedLeafScoreResult:
        """Score terminal facts and unique nonterminal root-information leaves."""
        cell_count = int(cells.endpoints.shape[0])
        if leaves.cell_to_leaf.shape != (cell_count,):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "leaf gather map differs from the consequence grid",
            )
        if not bool(cells.rules_exact_mask.all()):
            raise PlannerEvidenceError(
                PlannerFallbackReason.RULES_INEXACT,
                "leaf scoring received inexact engine evidence",
            )
        terminal = cells.endpoints == int(SemanticEndpoint.TERMINAL)
        same_seat = cells.endpoints == int(SemanticEndpoint.SAME_SEAT_MAIN)
        handoff = cells.endpoints == int(SemanticEndpoint.TURN_HANDOFF)
        if not bool((terminal | same_seat | handoff).all()):
            raise PlannerEvidenceError(
                PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
                "leaf scoring received a non-comparable endpoint",
            )
        if bool(np.any(leaves.cell_to_leaf[terminal] != -1)):
            raise ValueError("terminal cells must not map to value leaves")
        nonterminal = ~terminal
        self._consequence_cells += cell_count
        self._leaf_lookup_rows += int(np.count_nonzero(nonterminal))
        if bool(np.any(leaves.cell_to_leaf[nonterminal] < 0)):
            raise PlannerEvidenceError(
                PlannerFallbackReason.LEAF_VALUE_UNAVAILABLE,
                "nonterminal cell is missing a root-information leaf",
            )
        terminal_results = cells.engine_results[terminal]
        if terminal_results.size and bool(
            np.any(~np.isin(terminal_results, (0, 1, 2)))
        ):
            raise ValueError("terminal cells require an engine winner/draw result")
        if bool(np.any(cells.engine_results[nonterminal] != -1)):
            raise ValueError("nonterminal cells must use engine result -1")
        for cell_index in np.flatnonzero(nonterminal):
            index = int(cell_index)
            leaf_index = int(leaves.cell_to_leaf[index])
            if int(leaves.leaves[leaf_index].endpoint) != int(cells.endpoints[index]):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "value leaf endpoint differs from its consequence cell",
                )

        unique_values = self._unique_values(leaves)
        aggregation_started = time.perf_counter()
        scores = np.empty(cell_count, dtype=np.float32)
        scores[terminal] = np.fromiter(
            (
                engine_result_to_root_value(int(result), cells.root_player)
                for result in terminal_results
            ),
            dtype=np.float32,
            count=int(terminal_results.size),
        )
        if bool(nonterminal.any()):
            gathered = unique_values[leaves.cell_to_leaf[nonterminal]]
            tiebreak_weights = np.asarray(
                self.config.engine_tiebreak_feature_weights,
                dtype=np.float32,
            )
            engine_tiebreaks = cells.exact_effects[nonterminal] @ tiebreak_weights
            nonterminal_scores = gathered + (
                self.config.engine_tiebreak_weight
                * engine_tiebreaks
            )
            scores[nonterminal] = np.clip(
                nonterminal_scores,
                -self.config.score_clip,
                self.config.score_clip,
            )
        scores.setflags(write=False)
        if self._telemetry is not None:
            self._telemetry.record(
                PlannerStageEvent(
                    stage=PlannerStage.AGGREGATION,
                    seconds=time.perf_counter() - aggregation_started,
                    rows=cell_count,
                )
            )
        return SharedLeafScoreResult(
            cell_scores=scores,
            unique_leaf_values=unique_values,
            scorer_fingerprint=self.scorer_fingerprint,
            leaf_bootstrapped=bool(nonterminal.any()),
        )

    def _unique_values(self, leaves: RootInformationLeafBatch) -> Float32Array:
        if not leaves.leaves:
            values = np.empty(0, dtype=np.float32)
            values.setflags(write=False)
            return values
        tensorize_started = time.perf_counter()
        tensor_batch = leaves.tensorize(self._tensorizer)
        leaf_count = len(leaves.leaves)
        self._unique_leaf_rows += leaf_count
        if self._telemetry is not None:
            self._telemetry.record(
                PlannerStageEvent(
                    stage=PlannerStage.TENSORIZATION,
                    seconds=time.perf_counter() - tensorize_started,
                    rows=leaf_count,
                )
            )
        values = np.asarray(
            self._value_provider.values(tensor_batch),
            dtype=np.float32,
        )
        if values.shape != (len(leaves.leaves),):
            raise PlannerEvidenceError(
                PlannerFallbackReason.LEAF_VALUE_UNAVAILABLE,
                "value provider returned the wrong number of unique leaves",
            )
        if not bool(np.isfinite(values).all()):
            raise PlannerEvidenceError(
                PlannerFallbackReason.LEAF_VALUE_UNAVAILABLE,
                "value provider returned a non-finite root value",
            )
        values = np.clip(values, -1.0, 1.0)
        values.setflags(write=False)
        return values


def engine_result_to_root_value(engine_result: int, root_player: int) -> float:
    """Convert the engine's winner index to root-perspective W/D/L once."""
    if isinstance(root_player, bool) or root_player not in (0, 1):
        raise ValueError("root_player must be 0 or 1")
    if isinstance(engine_result, bool):
        raise ValueError("engine_result must be winner 0/1 or draw 2")
    if engine_result == 2:
        return 0.0
    if engine_result not in (0, 1):
        raise ValueError("engine_result must be winner 0/1 or draw 2")
    return 1.0 if engine_result == root_player else -1.0


def _require_array(
    array: np.ndarray,
    *,
    name: str,
    dtype: np.dtype[Any],
    shape: tuple[int, ...],
) -> None:
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a NumPy array")
    if array.dtype != dtype:
        raise TypeError(f"{name} has the wrong dtype")
    if array.shape != shape:
        raise ValueError(f"{name} has the wrong shape")
    if array.flags.writeable:
        raise ValueError(f"{name} must be read-only")


__all__ = [
    "LeafScoringBatch",
    "PlannerScoringConfig",
    "RootInformationValueProvider",
    "SharedLeafScoreResult",
    "SharedRootInformationLeafScorer",
    "engine_result_to_root_value",
    "leaf_scoring_batch_from_compact",
]
