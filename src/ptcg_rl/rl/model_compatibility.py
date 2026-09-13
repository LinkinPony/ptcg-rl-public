"""Portable model and trajectory compatibility without transport imports."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.context import PUBLIC_EVENT_SCHEMA_FINGERPRINT
from ptcg_rl.model import POLICY_INPUT_SCHEMA_FINGERPRINT, AgentNetworkConfig

LATEST_TRAJECTORY_SCHEMA_VERSION = 10
PUBLIC_EVENT_TRAJECTORY_SCHEMA_VERSION = 11
RECURRENT_SEQUENCE_TRAJECTORY_SCHEMA_VERSION = 12
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class DistributedModelCompatibility(BaseModel):
    """Portable contract shared by learner, rollout, and weight clients."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    trajectory_schema_version: int = LATEST_TRAJECTORY_SCHEMA_VERSION
    deck_conditioning_architecture_version: int = 0
    resolved_registry_sha256: str = ""
    model_config_sha256: str
    policy_input_schema_fingerprint: str
    public_event_schema_fingerprint: str = ""

    @field_validator(
        "trajectory_schema_version",
        "deck_conditioning_architecture_version",
    )
    @classmethod
    def valid_non_negative_version(cls, value: int) -> int:
        """Reject negative wire or model architecture versions."""
        if value < 0:
            raise ValueError("distributed compatibility versions cannot be negative")
        return value

    @field_validator("resolved_registry_sha256")
    @classmethod
    def valid_optional_registry_sha256(cls, value: str) -> str:
        """Accept an empty legacy registry or one canonical SHA256 digest."""
        if value and _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("resolved registry fingerprint must be lowercase SHA256")
        return value

    @field_validator("model_config_sha256")
    @classmethod
    def valid_model_config_sha256(cls, value: str) -> str:
        """Require a canonical model configuration fingerprint."""
        if _SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError("model config fingerprint must be lowercase SHA256")
        return value

    @field_validator("policy_input_schema_fingerprint")
    @classmethod
    def valid_policy_input_schema_fingerprint(cls, value: str) -> str:
        """Require the exact train/serve public-input contract."""
        if value != POLICY_INPUT_SCHEMA_FINGERPRINT:
            raise ValueError("policy input schema fingerprint mismatch")
        return value

    @field_validator("public_event_schema_fingerprint")
    @classmethod
    def valid_public_event_schema_fingerprint(cls, value: str) -> str:
        """Accept stateless absence or the exact recurrent event contract."""
        if value not in ("", PUBLIC_EVENT_SCHEMA_FINGERPRINT):
            raise ValueError("public event schema fingerprint mismatch")
        return value

    @classmethod
    def from_model_config(cls, config: AgentNetworkConfig) -> Self:
        """Build a path-independent compatibility contract for one model."""
        conditioning = config.deck_conditioning
        enabled = conditioning is not None and conditioning.enabled
        recurrent = config.recurrent
        registry_sha256 = (
            ""
            if not enabled or conditioning is None
            else conditioning.resolved_registry_sha256
        )
        if enabled and not registry_sha256:
            raise ValueError(
                "distributed deck conditioning requires a resolved registry"
            )
        return cls(
            trajectory_schema_version=(
                LATEST_TRAJECTORY_SCHEMA_VERSION
                if recurrent is None
                else RECURRENT_SEQUENCE_TRAJECTORY_SCHEMA_VERSION
            ),
            deck_conditioning_architecture_version=(
                0
                if not enabled or conditioning is None
                else conditioning.architecture_version
            ),
            resolved_registry_sha256=registry_sha256 or "",
            model_config_sha256=model_config_fingerprint(config),
            policy_input_schema_fingerprint=POLICY_INPUT_SCHEMA_FINGERPRINT,
            public_event_schema_fingerprint=(
                "" if recurrent is None else recurrent.public_event_schema_fingerprint
            ),
        )

    @classmethod
    def for_public_event_trajectory(cls, config: AgentNetworkConfig) -> Self:
        """Build the explicit schema-11 contract used by recurrent actors."""
        base = cls.from_model_config(config)
        return cls.model_validate(
            {
                **base.model_dump(mode="python"),
                "trajectory_schema_version": PUBLIC_EVENT_TRAJECTORY_SCHEMA_VERSION,
                "public_event_schema_fingerprint": PUBLIC_EVENT_SCHEMA_FINGERPRINT,
            }
        )


def validate_distributed_compatibility(
    expected: DistributedModelCompatibility,
    actual: DistributedModelCompatibility | Mapping[str, Any] | None,
) -> DistributedModelCompatibility:
    """Return a validated exact match or reject the remote host contract."""
    if actual is None:
        raise ValueError("distributed compatibility manifest is missing")
    parsed = (
        actual
        if isinstance(actual, DistributedModelCompatibility)
        else DistributedModelCompatibility.model_validate(actual)
    )
    mismatches = [
        field_name
        for field_name in DistributedModelCompatibility.model_fields
        if getattr(expected, field_name) != getattr(parsed, field_name)
    ]
    if mismatches:
        raise ValueError("distributed compatibility mismatch: " + ", ".join(mismatches))
    return parsed


def model_config_fingerprint(config: BaseModel) -> str:
    """Return the stable, path-independent identity of a model config."""
    payload = config.model_dump(mode="json")
    if payload.get("sequence") is None:
        # Preserve identities of checkpoints written before the optional
        # generalist-sequence discriminator existed.
        payload.pop("sequence", None)
    if payload.get("architecture") != "generalist_sequence_v3":
        # Preserve identities of checkpoints written before family-private
        # routing fields existed on the shared Pydantic model.
        payload.pop("family_routes", None)
        payload.pop("resolved_family_registry_sha256", None)
    card_encoder = payload.get("card_encoder")
    if isinstance(card_encoder, dict):
        # Host-local table paths and mmap choices do not alter module shapes.
        card_encoder.pop("feature_table_path", None)
        card_encoder.pop("feature_table_mmap_mode", None)
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "LATEST_TRAJECTORY_SCHEMA_VERSION",
    "PUBLIC_EVENT_TRAJECTORY_SCHEMA_VERSION",
    "RECURRENT_SEQUENCE_TRAJECTORY_SCHEMA_VERSION",
    "DistributedModelCompatibility",
    "model_config_fingerprint",
    "validate_distributed_compatibility",
]
