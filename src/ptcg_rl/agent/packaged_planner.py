"""Immutable packaged configuration and construction for schema-9 planning."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.belief.sampling import BeliefSamplerConfig
from ptcg_rl.context import OpponentBeliefFeatureConfig
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.planner_runtime_factory import (
    PlannerBehaviorRuntime,
    create_planner_behavior_runtime,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeConfig

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class _CheckpointPlannerSurface(Protocol):
    """Checkpoint policy fields required by the packaged planner builder."""

    @property
    def planner_model(self) -> Any:
        """Return the strictly loaded model."""

    @property
    def planner_device(self) -> Any:
        """Return the model device."""

    @property
    def checkpoint_sha256(self) -> str:
        """Return the immutable checkpoint fingerprint."""

    @property
    def policy_version(self) -> int:
        """Return the checkpoint publication version."""

    def prewarm(self) -> None:
        """Prewarm the underlying model when requested by ActTime."""


class PackagedPlannerConfig(BaseModel):
    """Self-contained planner semantics shipped beside ``main.py``.

    Paths are resolved relative to this config file before the model is
    returned. Composite identities bind the exact source checkpoint, migrated
    model, static planner, and mutable serving lease.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    planner_enabled_by_default: bool
    source_checkpoint_sha256: str
    checkpoint_sha256: str
    model_fingerprint: str
    policy_version: int = Field(ge=0)
    proposal_version: int = Field(ge=0)
    expected_planner_fingerprint: str
    expected_runtime_fingerprint: str
    native_library_sha256: str
    native_abi_fingerprint: str
    native_schema_fingerprint: str
    belief_prior_sha256: str
    belief_runtime_fingerprint: str
    native_library_path: Path
    planner: ResolvedPlannerRuntimeConfig
    belief_sampler: BeliefSamplerConfig
    belief_producer: OpponentBeliefFeatureConfig
    stochastic_seed: int

    @field_validator(
        "checkpoint_sha256",
        "source_checkpoint_sha256",
        "model_fingerprint",
        "expected_planner_fingerprint",
        "expected_runtime_fingerprint",
        "native_library_sha256",
        "native_abi_fingerprint",
        "native_schema_fingerprint",
        "belief_prior_sha256",
        "belief_runtime_fingerprint",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require canonical immutable content identities."""
        if _SHA256.fullmatch(value) is None or value == "0" * 64:
            raise ValueError("packaged planner identities must be SHA-256")
        return value

    @model_validator(mode="after")
    def valid_derived_identities(self) -> PackagedPlannerConfig:
        """Reject copied or stale composite identities before file loading."""
        resolved = self.planner.resolve_for_lease(
            model_fingerprint=self.model_fingerprint,
            policy_version=self.policy_version,
            proposal_version=self.proposal_version,
        )
        if resolved.static.planner_fingerprint != self.expected_planner_fingerprint:
            raise ValueError("packaged planner fingerprint differs from config")
        if resolved.runtime_fingerprint != self.expected_runtime_fingerprint:
            raise ValueError("packaged runtime fingerprint differs from config")
        engine = self.planner.engine
        if (
            engine.library_fingerprint != self.native_library_sha256
            or engine.native_abi_fingerprint != self.native_abi_fingerprint
            or engine.native_schema_fingerprint != self.native_schema_fingerprint
        ):
            raise ValueError("packaged native identity differs from planner")
        sampler_prior = self.belief_sampler.prior_deck_signature_summary_sha256
        producer_prior = self.belief_producer.deck_signature_summary_sha256
        if sampler_prior != self.belief_prior_sha256 or (
            producer_prior != self.belief_prior_sha256
        ):
            raise ValueError("packaged belief prior identity differs from config")
        if (
            self.planner.scenario.belief_sampler_fingerprint
            != self.belief_runtime_fingerprint
        ):
            raise ValueError("packaged belief runtime identity differs from planner")
        if not self.belief_producer.enabled:
            raise ValueError("packaged planner requires its public belief producer")
        return self

    @classmethod
    def from_file(cls, path: Path) -> PackagedPlannerConfig:
        """Read one small JSON spec and resolve only its packaged asset paths."""
        resolved_path = path.resolve()
        with resolved_path.open("r", encoding="utf-8") as source:
            raw = json.load(source)
        if not isinstance(raw, dict):
            raise ValueError("packaged planner config must be a JSON object")
        base = resolved_path.parent
        payload = dict(raw)
        payload["native_library_path"] = _asset_path(
            base,
            payload.get("native_library_path"),
            label="native library",
        )
        for key, path_key in (
            ("belief_sampler", "prior_deck_signature_summary_path"),
            ("belief_producer", "deck_signature_summary_path"),
        ):
            nested = payload.get(key)
            if not isinstance(nested, dict):
                raise ValueError(f"packaged planner {key} must be an object")
            copied = dict(nested)
            if copied.get(path_key) is not None:
                copied[path_key] = _asset_path(
                    base,
                    copied[path_key],
                    label=f"{key} prior",
                )
            payload[key] = copied
        return cls.model_validate(payload)


def build_packaged_planner_policy(
    checkpoint_policy: _CheckpointPlannerSurface,
    config: PackagedPlannerConfig,
) -> tuple[Any, PlannerBehaviorRuntime]:
    """Wrap one strictly loaded CPU checkpoint in the shared v5 service."""
    if checkpoint_policy.checkpoint_sha256 != config.checkpoint_sha256:
        raise ValueError("packaged planner loaded another checkpoint")
    if checkpoint_policy.policy_version != config.policy_version:
        raise ValueError("packaged planner checkpoint version differs from config")
    native_sha256 = _file_sha256(config.native_library_path)
    if native_sha256 != config.native_library_sha256:
        raise ValueError("packaged native library differs from planner identity")
    model_fingerprint = canonical_model_state_fingerprint(
        checkpoint_policy.planner_model
    )
    if model_fingerprint != config.model_fingerprint:
        raise ValueError("packaged migrated model differs from planner identity")

    runtime = create_planner_behavior_runtime(
        runtime_config=config.planner,
        sampler_config=config.belief_sampler,
        belief_config=config.belief_producer,
        stochastic_seed=config.stochastic_seed,
        native_library_path=config.native_library_path,
    )
    try:
        from ptcg_rl.agent.planner_select_policy import PlannerSelectPolicy

        policy = PlannerSelectPolicy(
            model=checkpoint_policy.planner_model,
            device=checkpoint_policy.planner_device,
            runtime_config=config.planner,
            service=runtime.service,
            policy_version=config.policy_version,
            proposal_version=config.proposal_version,
            verified_model_fingerprint=config.model_fingerprint,
            base_policy=checkpoint_policy,
            owned_runtime=runtime,
            temperature=0.0,
        )
    except Exception:
        runtime.close()
        raise
    return policy, runtime


def _asset_path(base: Path, value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"packaged planner {label} path must be non-empty")
    path = Path(value)
    resolved = path if path.is_absolute() else base / path
    if not resolved.is_file():
        raise FileNotFoundError(f"packaged planner {label} not found: {resolved}")
    return resolved.resolve()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "PackagedPlannerConfig",
    "build_packaged_planner_policy",
]
