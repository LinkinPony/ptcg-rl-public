"""Typed geometry and wire accounting for persistent native lane pools."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.engine.native_consequence import NativeConsequenceBatchResult
from ptcg_rl.engine.session import HiddenInformation


class NativeConsequencePoolConfig(BaseModel):
    """Resolved persistent-lane geometry selected by integrated profiling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    lane_count: int
    max_inflight_jobs: int
    max_transitions_per_call: int
    call_guard_seconds: float

    @field_validator("lane_count", "max_inflight_jobs", "max_transitions_per_call")
    @classmethod
    def positive_capacity(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("native pool capacities must be positive")
        return value

    @field_validator("call_guard_seconds")
    @classmethod
    def finite_call_guard(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("call_guard_seconds must be finite and non-negative")
        return value

    @model_validator(mode="after")
    def enough_job_slots(self) -> Self:
        if self.max_inflight_jobs < self.lane_count:
            raise ValueError("max_inflight_jobs cannot be smaller than lane_count")
        return self


@dataclass(frozen=True)
class NativeConsequenceCall:
    """Typed arguments for one bounded candidates-by-scenarios native call."""

    state_token: bytes | str
    hidden_worlds: tuple[HiddenInformation, ...]
    candidate_actions: tuple[tuple[int, ...], ...]
    producer_contract_fingerprint: bytes
    root_player: int
    manual_coin: bool
    max_cells: int
    max_engine_steps: int
    max_forced_steps: int
    max_observation_bytes: int
    stochastic_seed: int = 0

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
        stochastic_seed: int = 0,
        max_cells: int,
        max_engine_steps: int,
        max_forced_steps: int,
        max_observation_bytes: int,
    ) -> Self:
        """Freeze caller-owned ragged inputs before asynchronous execution."""
        return cls(
            state_token=state_token,
            hidden_worlds=tuple(hidden_worlds),
            candidate_actions=tuple(tuple(action) for action in candidate_actions),
            producer_contract_fingerprint=bytes(producer_contract_fingerprint),
            root_player=root_player,
            manual_coin=manual_coin,
            max_cells=max_cells,
            max_engine_steps=max_engine_steps,
            max_forced_steps=max_forced_steps,
            max_observation_bytes=max_observation_bytes,
            stochastic_seed=stochastic_seed,
        )

    @property
    def transitions(self) -> int:
        return len(self.hidden_worlds) * len(self.candidate_actions)


@dataclass(frozen=True)
class NativeConsequencePoolResult:
    """Native result plus pool scheduling latency."""

    batch: NativeConsequenceBatchResult
    queue_wait_seconds: float


@dataclass(frozen=True)
class NativeConsequencePoolStats:
    """Thread-safe lifetime service counters."""

    lanes: int
    submitted: int
    completed: int
    failed: int
    saturated: int
    deadline_rejected: int
    active: int
    peak_active: int
    queue_wait_seconds: float


def maximum_call_host_bytes(call: NativeConsequenceCall) -> int:
    """Return exact bounded input plus maximum v6 output storage bytes."""
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
    header_and_fingerprints = 6 * 4 + 2 * 32
    metadata_bytes = call.transitions * 14 * 4
    output_bytes = header_and_fingerprints + metadata_bytes + call.max_observation_bytes
    return input_bytes + output_bytes


__all__ = [
    "NativeConsequenceCall",
    "NativeConsequencePoolConfig",
    "NativeConsequencePoolResult",
    "NativeConsequencePoolStats",
    "maximum_call_host_bytes",
]
