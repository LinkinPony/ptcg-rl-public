"""Canonical model and input identity for recurrent runtime state leases."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Self

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.context import (
    PUBLIC_EVENT_DECISION_CLOCK_FINGERPRINT,
    PUBLIC_EVENT_SCHEMA_FINGERPRINT,
    PublicEventBatch,
)
from ptcg_rl.model import AgentNetworkConfig, RecurrentPolicyState
from ptcg_rl.model.network import SampleDecodeTrace
from ptcg_rl.rl.model_compatibility import (
    RECURRENT_SEQUENCE_TRAJECTORY_SCHEMA_VERSION,
    DistributedModelCompatibility,
)

_POLICY_ARTIFACT_IDENTITY_DOMAIN = b"ptcg-rl/policy-artifact-identity/v1\x00"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class PolicyArtifactIdentity(BaseModel):
    """Immutable identity of weights and every recurrent input contract.

    ``policy_version`` and route aliases are deliberately absent: they locate a
    publication but do not identify its content.  Recurrent state may be reused
    only when this complete identity is unchanged.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    model_fingerprint: str
    compatibility: DistributedModelCompatibility
    decision_clock_fingerprint: str = ""

    @field_validator("model_fingerprint")
    @classmethod
    def valid_model_fingerprint(cls, value: str) -> str:
        """Require the canonical full model-state digest."""
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("model fingerprint must be lowercase SHA256")
        return value

    @field_validator("decision_clock_fingerprint")
    @classmethod
    def valid_decision_clock_fingerprint(cls, value: str) -> str:
        """Accept stateless absence or one canonical decision clock."""
        if value not in ("", PUBLIC_EVENT_DECISION_CLOCK_FINGERPRINT):
            raise ValueError("public event decision clock fingerprint mismatch")
        return value

    @model_validator(mode="after")
    def coherent_recurrent_contract(self) -> Self:
        """Keep event, wire-schema, and decision-clock identities atomic."""
        event_fingerprint = self.compatibility.public_event_schema_fingerprint
        recurrent = bool(event_fingerprint)
        if recurrent:
            if event_fingerprint != PUBLIC_EVENT_SCHEMA_FINGERPRINT:
                raise ValueError("recurrent artifact event schema mismatch")
            if (
                self.compatibility.trajectory_schema_version
                != RECURRENT_SEQUENCE_TRAJECTORY_SCHEMA_VERSION
            ):
                raise ValueError("recurrent artifact requires sequence schema 12")
            if (
                self.decision_clock_fingerprint
                != PUBLIC_EVENT_DECISION_CLOCK_FINGERPRINT
            ):
                raise ValueError("recurrent artifact decision clock is missing")
        elif self.decision_clock_fingerprint:
            raise ValueError("stateless artifact cannot claim a decision clock")
        return self

    @classmethod
    def from_model_config(
        cls,
        config: AgentNetworkConfig,
        *,
        model_fingerprint: str,
    ) -> Self:
        """Build the complete identity for a verified model-state snapshot."""
        compatibility = DistributedModelCompatibility.from_model_config(config)
        return cls(
            model_fingerprint=model_fingerprint,
            compatibility=compatibility,
            decision_clock_fingerprint=(
                ""
                if config.recurrent is None
                else PUBLIC_EVENT_DECISION_CLOCK_FINGERPRINT
            ),
        )

    @property
    def fingerprint(self) -> str:
        """Return a canonical digest of all authoritative identity fields."""
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(_POLICY_ARTIFACT_IDENTITY_DOMAIN + encoded).hexdigest()


def validate_policy_artifact_identity(
    expected: PolicyArtifactIdentity,
    actual: PolicyArtifactIdentity | Mapping[str, Any] | None,
) -> PolicyArtifactIdentity:
    """Return an exact identity match or reject a missing/mixed artifact."""
    if actual is None:
        raise ValueError("policy artifact identity is missing")
    parsed = (
        actual
        if isinstance(actual, PolicyArtifactIdentity)
        else PolicyArtifactIdentity.model_validate(actual)
    )
    if parsed != expected:
        raise ValueError("policy artifact identity mismatch")
    return parsed


class RecurrentSequenceIdentity(BaseModel):
    """Stable public owner of one actor-side game-seat recurrent sequence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    game_id: str
    seat: int
    exact_deck_signature: str

    @field_validator("game_id", "exact_deck_signature")
    @classmethod
    def valid_non_empty_identity(cls, value: str) -> str:
        """Reject empty routing identities at the RPC trust boundary."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("recurrent sequence identity fields must be non-empty")
        return cleaned

    @field_validator("seat")
    @classmethod
    def valid_seat(cls, value: int) -> int:
        """Restrict recurrent state to one of the two battle seats."""
        if value not in (0, 1):
            raise ValueError("recurrent sequence seat must be zero or one")
        return value


@dataclass(frozen=True)
class RecurrentInferenceBatch:
    """Complete recurrent input aligned with one inference decision batch."""

    public_events: PublicEventBatch
    previous_state: RecurrentPolicyState
    event_generations: tuple[int, ...]
    sequences: tuple[RecurrentSequenceIdentity, ...]
    expected_artifact: PolicyArtifactIdentity | None

    def __post_init__(self) -> None:
        """Reject row, device, generation, and first-bind inconsistencies."""
        batch_size = self.public_events.batch_size
        if self.previous_state.batch_size != batch_size:
            raise ValueError("recurrent events and old state batch sizes differ")
        if len(self.event_generations) != batch_size:
            raise ValueError("recurrent event generations are misaligned")
        if len(self.sequences) != batch_size:
            raise ValueError("recurrent sequence identities are misaligned")
        if len(set(self.sequences)) != batch_size:
            raise ValueError("recurrent inference batch contains duplicate sequences")
        if any(generation < 0 for generation in self.event_generations):
            raise ValueError("recurrent event generations must be non-negative")
        if self.previous_state.hidden.device != self.public_events.event_types.device:
            raise ValueError("recurrent events and old state use different devices")
        if self.expected_artifact is None and not _is_zero_recurrent_state(
            self.previous_state
        ):
            raise ValueError("an unbound recurrent request requires exact zero state")
        # These are GameContext stale-token generations, not recurrent
        # timesteps. A reset may legitimately make the first value non-zero;
        # canonical zero state plus an unbound server namespace proves reset.

    @property
    def batch_size(self) -> int:
        """Return recurrent decision rows in this batch."""
        return len(self.sequences)


@dataclass(frozen=True)
class RecurrentDecodeResult:
    """One complete-action trace plus an uncommitted recurrent proposal."""

    trace: SampleDecodeTrace
    proposed_state: RecurrentPolicyState
    event_generations: tuple[int, ...]
    sequences: tuple[RecurrentSequenceIdentity, ...]
    served_artifact: PolicyArtifactIdentity

    def __post_init__(self) -> None:
        """Require all returned evidence to describe the same decision rows."""
        batch_size = len(self.trace.actions)
        if self.proposed_state.batch_size != batch_size:
            raise ValueError("recurrent proposed state batch is misaligned")
        if len(self.event_generations) != batch_size:
            raise ValueError("recurrent response generations are misaligned")
        if len(self.sequences) != batch_size:
            raise ValueError("recurrent response sequences are misaligned")


def _is_zero_recurrent_state(state: RecurrentPolicyState) -> bool:
    """Return whether a first-bind state is the canonical explicit reset."""
    return bool(
        torch.count_nonzero(state.hidden).item() == 0
        and torch.count_nonzero(state.cell).item() == 0
    )


__all__ = [
    "PolicyArtifactIdentity",
    "RecurrentDecodeResult",
    "RecurrentInferenceBatch",
    "RecurrentSequenceIdentity",
    "validate_policy_artifact_identity",
]
