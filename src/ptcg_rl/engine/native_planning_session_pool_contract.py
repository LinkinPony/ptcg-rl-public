"""Typed geometry and exact ABI accounting for pinned planning sessions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.engine.native_planning_session import NativePlanningSessionBatchResult
from ptcg_rl.engine.native_planning_session_payload import NativePlanningSessionHandle
from ptcg_rl.engine.native_planning_session_request import NativePlanningSessionCaps
from ptcg_rl.engine.session import HiddenInformation

_SESSION_HEADER_BYTES = 8 * 4 + 2 * 32
_SESSION_METADATA_ROW_BYTES = 14 * 4


class NativePlanningSessionPoolConfig(BaseModel):
    """Resolved persistent-lane geometry selected by integrated profiling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_count: int
    max_inflight_jobs: int
    max_transitions_per_call: int
    call_guard_seconds: float

    @field_validator("lane_count", "max_inflight_jobs", "max_transitions_per_call")
    @classmethod
    def positive_capacity(cls, value: int) -> int:
        """Require every service capacity to be positive."""
        if value <= 0:
            raise ValueError("planning-session pool capacities must be positive")
        return value

    @field_validator("call_guard_seconds")
    @classmethod
    def finite_call_guard(cls, value: float) -> float:
        """Require a finite non-negative return guard."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("call_guard_seconds must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def enough_job_slots(self) -> Self:
        """Keep one executor job available for each pinned lane."""
        if self.max_inflight_jobs < self.lane_count:
            raise ValueError("max_inflight_jobs cannot be smaller than lane_count")
        return self


@dataclass(frozen=True)
class NativePlanningSessionOpenCall:
    """Frozen inputs for one candidate-major root grid."""

    state_token: bytes | str
    hidden_worlds: tuple[HiddenInformation, ...]
    candidate_actions: tuple[tuple[int, ...], ...]
    producer_contract_fingerprint: bytes
    root_player: int
    manual_coin: bool
    max_state_slots: int
    caps: NativePlanningSessionCaps

    @classmethod
    def from_sequences(
        cls,
        state_token: bytes | str,
        *,
        hidden_worlds: Sequence[HiddenInformation],
        candidate_actions: Sequence[Sequence[int]],
        producer_contract_fingerprint: bytes,
        root_player: int,
        manual_coin: bool = False,
        max_state_slots: int,
        caps: NativePlanningSessionCaps,
    ) -> Self:
        """Freeze caller-owned ragged arrays before executor submission."""
        return cls(
            state_token=state_token,
            hidden_worlds=tuple(hidden_worlds),
            candidate_actions=tuple(tuple(action) for action in candidate_actions),
            producer_contract_fingerprint=bytes(producer_contract_fingerprint),
            root_player=root_player,
            manual_coin=manual_coin,
            max_state_slots=max_state_slots,
            caps=caps,
        )

    @property
    def transitions(self) -> int:
        """Return root candidate-by-world rows."""
        return len(self.hidden_worlds) * len(self.candidate_actions)


@dataclass(frozen=True)
class NativePlanningSessionContinueCall:
    """Frozen aligned handle/action continuation batch."""

    handles: tuple[NativePlanningSessionHandle, ...]
    actions: tuple[tuple[int, ...], ...]
    caps: NativePlanningSessionCaps

    @classmethod
    def from_sequences(
        cls,
        handles: Sequence[NativePlanningSessionHandle],
        actions: Sequence[Sequence[int]],
        *,
        caps: NativePlanningSessionCaps,
    ) -> Self:
        """Freeze one request-aligned continuation batch."""
        return cls(
            handles=tuple(handles),
            actions=tuple(tuple(action) for action in actions),
            caps=caps,
        )

    @property
    def transitions(self) -> int:
        """Return request-aligned transition rows."""
        return len(self.handles)


@dataclass(frozen=True)
class NativePlanningSessionPoolResult:
    """Native result plus bounded service scheduling latency."""

    batch: NativePlanningSessionBatchResult
    queue_wait_seconds: float


@dataclass(frozen=True)
class NativePlanningSessionPoolStats:
    """Thread-safe lifetime service counters."""

    lanes: int
    submitted: int
    completed: int
    failed: int
    saturated: int
    deadline_rejected: int
    active_calls: int
    active_sessions: int
    peak_active_calls: int
    lanes_replaced: int
    queue_wait_seconds: float


def maximum_open_host_bytes(call: NativePlanningSessionOpenCall) -> int:
    """Return ABI input plus maximum v5 OPEN response storage."""
    state_bytes = (
        len(call.state_token)
        if isinstance(call.state_token, bytes)
        else len(call.state_token.encode("ascii"))
    )
    hidden_values = sum(
        len(zone)
        for world in call.hidden_worlds
        for zone in (
            world.your_deck,
            world.your_prize,
            world.opponent_deck,
            world.opponent_prize,
            world.opponent_hand,
            world.opponent_active,
        )
    )
    hidden_counts = 6 * len(call.hidden_worlds)
    candidate_values = sum(len(action) for action in call.candidate_actions)
    candidate_counts = len(call.candidate_actions)
    input_bytes = (
        state_bytes
        + len(call.producer_contract_fingerprint)
        + 4 * (hidden_values + hidden_counts + candidate_values + candidate_counts)
    )
    return input_bytes + _maximum_response_bytes(
        rows=call.transitions,
        observation_bytes=call.caps.max_observation_bytes,
    )


def maximum_continue_host_bytes(
    call: NativePlanningSessionContinueCall,
    *,
    producer_contract_fingerprint_bytes: int = 32,
) -> int:
    """Return ABI input plus maximum v5 CONTINUE response storage."""
    action_values = sum(len(action) for action in call.actions)
    input_bytes = producer_contract_fingerprint_bytes + 4 * (
        len(call.handles) + len(call.actions) + action_values
    )
    return input_bytes + _maximum_response_bytes(
        rows=call.transitions,
        observation_bytes=call.caps.max_observation_bytes,
    )


def _maximum_response_bytes(*, rows: int, observation_bytes: int) -> int:
    return (
        _SESSION_HEADER_BYTES
        + rows * _SESSION_METADATA_ROW_BYTES
        + observation_bytes
    )


__all__ = [
    "NativePlanningSessionContinueCall",
    "NativePlanningSessionOpenCall",
    "NativePlanningSessionPoolConfig",
    "NativePlanningSessionPoolResult",
    "NativePlanningSessionPoolStats",
    "maximum_continue_host_bytes",
    "maximum_open_host_bytes",
]
