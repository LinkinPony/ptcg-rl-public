"""Public contracts for bounded v5 hierarchical option evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.search.hierarchical_contract import OptionOutcome
from ptcg_rl.agent.search.planner_scoring import LeafScoringBatch
from ptcg_rl.agent.search.root_information import RootInformationLeafBatch
from ptcg_rl.agent.search.root_information_context import (
    PublicBeliefFeatureProducer,
)
from ptcg_rl.context import GameContextSnapshot
from ptcg_rl.engine.consequence_identity import MaterializedScenario
from ptcg_rl.engine.native_planning_session_request import NativePlanningSessionCaps


class HierarchicalSearchConfig(BaseModel):
    """Fixed whole-tree limits selected before a formal runtime starts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: int = 1
    max_continuation_depth: int
    max_chance_depth: int
    max_chance_nodes: int
    max_state_slots: int
    max_continue_rows_per_call: int
    max_engine_steps_per_call: int
    max_forced_steps_per_transition: int
    max_observation_bytes_per_call: int

    @field_validator(
        "architecture_version",
        "max_continuation_depth",
        "max_chance_nodes",
        "max_state_slots",
        "max_continue_rows_per_call",
        "max_engine_steps_per_call",
        "max_observation_bytes_per_call",
    )
    @classmethod
    def positive_limit(cls, value: int) -> int:
        """Require positive capacities."""
        if value <= 0:
            raise ValueError("hierarchical search capacities must be positive")
        return value

    @field_validator("max_chance_depth", "max_forced_steps_per_transition")
    @classmethod
    def nonnegative_limit(cls, value: int) -> int:
        """Allow zero chance/forced depth while rejecting negative values."""
        if value < 0:
            raise ValueError("hierarchical search depths must be non-negative")
        return value

    @model_validator(mode="after")
    def chance_identity_fits_uint64(self) -> Self:
        """Chance path handles use one marker bit plus the path bits."""
        if self.max_chance_depth > 63:
            raise ValueError("max_chance_depth cannot exceed 63")
        return self

    @property
    def native_caps(self) -> NativePlanningSessionCaps:
        """Return the v5 per-call cap object bound into request identity."""
        return NativePlanningSessionCaps(
            max_engine_steps=self.max_engine_steps_per_call,
            max_forced_steps=self.max_forced_steps_per_transition,
            max_observation_bytes=self.max_observation_bytes_per_call,
        )


@dataclass(frozen=True, slots=True)
class ContinuationPrompt:
    """One unique root-observable strategic information history."""

    information_history_fingerprint: str
    observation: Mapping[str, Any]
    root_action: tuple[int, ...]
    continuation_depth: int


class ContinuationActionProvider(Protocol):
    """Model-leased controller queried once per unique public history."""

    def select_actions(
        self,
        prompts: tuple[ContinuationPrompt, ...],
    ) -> Sequence[Sequence[int]]:
        """Return one complete legal selection per unique prompt."""


@dataclass(frozen=True, slots=True)
class HierarchicalSearchRequest:
    """Request-local inputs that never cross trajectory or IPC boundaries."""

    state_token: bytes | str
    root_observation: Mapping[str, Any]
    scenarios: tuple[MaterializedScenario, ...]
    candidate_actions: tuple[tuple[int, ...], ...]
    legal_action_count: int
    root_player: int
    context_snapshot: GameContextSnapshot
    belief_feature_producer: PublicBeliefFeatureProducer | None
    belief_summary_width: int
    producer_context: bytes
    belief_summary: tuple[float, ...]
    producer_contract_fingerprint: bytes
    support_exhaustive: bool

    @classmethod
    def from_sequences(
        cls,
        state_token: bytes | str,
        *,
        root_observation: Mapping[str, Any],
        scenarios: Sequence[MaterializedScenario],
        candidate_actions: Sequence[Sequence[int]],
        legal_action_count: int,
        root_player: int,
        context_snapshot: GameContextSnapshot,
        belief_feature_producer: PublicBeliefFeatureProducer | None,
        belief_summary_width: int,
        producer_context: bytes,
        belief_summary: Sequence[float],
        producer_contract_fingerprint: bytes,
        support_exhaustive: bool,
    ) -> Self:
        """Freeze ragged caller inputs before native work begins."""
        return cls(
            state_token=state_token,
            root_observation=root_observation,
            scenarios=tuple(scenarios),
            candidate_actions=tuple(tuple(action) for action in candidate_actions),
            legal_action_count=legal_action_count,
            root_player=root_player,
            context_snapshot=context_snapshot,
            belief_feature_producer=belief_feature_producer,
            belief_summary_width=int(belief_summary_width),
            producer_context=bytes(producer_context),
            belief_summary=tuple(float(value) for value in belief_summary),
            producer_contract_fingerprint=bytes(producer_contract_fingerprint),
            support_exhaustive=bool(support_exhaustive),
        )


@dataclass(frozen=True, slots=True)
class HierarchicalOutcomeBatch:
    """Complete candidate-major outcomes ready for shared leaf scoring."""

    actions: tuple[tuple[int, ...], ...]
    outcomes: tuple[OptionOutcome, ...]
    leaves: RootInformationLeafBatch
    scoring_cells: LeafScoringBatch
    path_steps: npt.NDArray[np.int32]
    chance_depth: int
    continuation_nodes: int
    producer_contract_fingerprint: str

    def __post_init__(self) -> None:
        """Require all candidate/scenario columns to remain aligned."""
        if not self.actions or len(self.actions) != len(self.outcomes):
            raise ValueError("hierarchical actions and outcomes must align")
        scenario_count = len(self.outcomes[0].cells)
        cell_count = len(self.actions) * scenario_count
        if self.path_steps.dtype != np.dtype(np.int32):
            raise TypeError("path_steps must use int32")
        if self.path_steps.shape != (cell_count,) or self.path_steps.flags.writeable:
            raise ValueError("path_steps must be immutable and cell-aligned")
        if self.scoring_cells.endpoints.shape != (cell_count,):
            raise ValueError("scoring cells differ from hierarchical outcomes")
        if self.leaves.cell_to_leaf.shape != (cell_count,):
            raise ValueError("leaf gather differs from hierarchical outcomes")
        if self.chance_depth < 0 or self.continuation_nodes < 0:
            raise ValueError("hierarchical work accounting must be non-negative")
        if (
            len(self.producer_contract_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.producer_contract_fingerprint
            )
        ):
            raise ValueError("producer contract fingerprint must be SHA-256 hex")


__all__ = [
    "ContinuationActionProvider",
    "ContinuationPrompt",
    "HierarchicalOutcomeBatch",
    "HierarchicalSearchConfig",
    "HierarchicalSearchRequest",
]
