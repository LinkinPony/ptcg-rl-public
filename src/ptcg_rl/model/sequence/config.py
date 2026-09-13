"""Validated identity for the generalist temporal policy."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.context.public_events import PUBLIC_EVENT_SCHEMA_FINGERPRINT
from ptcg_rl.engine.prospective_facts import ProspectiveEngineFactConfig

GENERALIST_SEQUENCE_ARCHITECTURE = "generalist_sequence_v1"
GENERALIST_SEQUENCE_V2_ARCHITECTURE = "generalist_sequence_v2"
GENERALIST_SEQUENCE_V3_ARCHITECTURE = "generalist_sequence_v3"
GENERALIST_SEQUENCE_SCHEMA_VERSION = 1
GENERALIST_SEQUENCE_TOKEN_ORDER = ("event", "state", "action")
GENERALIST_SEQUENCE_OVERFLOW = "complete_block_sliding_window_global_position_v1"


def _contract_fingerprint() -> str:
    descriptor = {
        "schema_version": GENERALIST_SEQUENCE_SCHEMA_VERSION,
        "token_order": GENERALIST_SEQUENCE_TOKEN_ORDER,
        "decision_clock": "accepted_non_forced_policy_decision",
        "policy_read_position": "state",
        "action_visibility": "future_blocks_only",
        "overflow": GENERALIST_SEQUENCE_OVERFLOW,
        "raw_sequence_is_truth": True,
        "kv_is_cache": True,
    }
    payload = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(
        b"ptcg-rl/generalist-sequence-contract/v1\x00" + payload
    ).hexdigest()


GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT = _contract_fingerprint()


class GeneralistSequenceConfig(BaseModel):
    """Immutable causal Transformer and sequence-contract configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    architecture_version: Literal[1] = 1
    contract_fingerprint: str = GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT
    public_event_schema_fingerprint: str = PUBLIC_EVENT_SCHEMA_FINGERPRINT
    d_model: int = Field(default=512, gt=0)
    num_layers: int = Field(default=6, ge=2)
    attention_heads: int = Field(default=8, gt=0)
    feedforward_dim: int = Field(default=2048, gt=0)
    max_context_blocks: int = Field(default=64, ge=2)
    learner_target_blocks: int = Field(default=16, ge=1)
    dropout: float = 0.0
    serial_hash_buckets: int = Field(default=4096, gt=0)
    attack_hash_buckets: int = Field(default=2048, gt=0)
    engine_facts: ProspectiveEngineFactConfig = Field(
        default_factory=ProspectiveEngineFactConfig
    )

    @field_validator("contract_fingerprint")
    @classmethod
    def valid_contract_fingerprint(cls, value: str) -> str:
        """Bind weights and KV caches to the exact sequence semantics."""
        if value != GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT:
            raise ValueError("generalist sequence contract fingerprint mismatch")
        return value

    @field_validator("public_event_schema_fingerprint")
    @classmethod
    def valid_public_event_fingerprint(cls, value: str) -> str:
        """Reject an event producer with different privacy semantics."""
        if value != PUBLIC_EVENT_SCHEMA_FINGERPRINT:
            raise ValueError("generalist sequence public-event schema mismatch")
        return value

    @field_validator("dropout")
    @classmethod
    def deterministic_dropout(cls, value: float) -> float:
        """Keep behavior-policy and current-weight reconstruction deterministic."""
        if value != 0.0:
            raise ValueError("generalist sequence dropout must remain zero")
        return value

    @model_validator(mode="after")
    def coherent_dimensions(self) -> Self:
        """Validate attention and overlap geometry."""
        if self.d_model % self.attention_heads:
            raise ValueError("temporal width must be divisible by attention heads")
        if self.feedforward_dim < self.d_model:
            raise ValueError("temporal feedforward width cannot bottleneck d_model")
        if self.learner_target_blocks > self.max_context_blocks:
            raise ValueError(
                "learner target blocks cannot exceed the runtime context window"
            )
        return self


__all__ = [
    "GENERALIST_SEQUENCE_ARCHITECTURE",
    "GENERALIST_SEQUENCE_V2_ARCHITECTURE",
    "GENERALIST_SEQUENCE_V3_ARCHITECTURE",
    "GENERALIST_SEQUENCE_CONTRACT_FINGERPRINT",
    "GENERALIST_SEQUENCE_OVERFLOW",
    "GENERALIST_SEQUENCE_SCHEMA_VERSION",
    "GENERALIST_SEQUENCE_TOKEN_ORDER",
    "GeneralistSequenceConfig",
]
