"""Controlled warm-start transitions between exact-deck expert registries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, cast

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor

from ptcg_rl.model import (
    DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION,
    DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION,
    AgentNetworkConfig,
    AgentPolicyValueNet,
)

_SHA256_HEX_LENGTH = 64
_PRIVATE_STATE_PREFIXES = (
    "state_encoder.private_adapters.",
    "state_encoder.private_lora.",
    "policy_head.private_lora.",
    "private_policy_adapters.",
    "private_root_value_heads.",
    "private_prefix_value_heads.",
    "state_encoder.private_strategy_stacks.",
    "policy_head.private_strategies.",
    "dense_private_root_value_heads.",
    "dense_private_prefix_value_heads.",
    "exact_capsules.",
)
ExpertInitializationMode = Literal["zero", "clone"]
ExpertTransitionAction = Literal["continued", "remapped", "zero", "cloned"]
TransitionSchedulerMode = Literal["restart", "continue"]
TransitionOptimizerMode = Literal["transplant", "fresh"]
TransitionTopologyChange = Literal["none", "action_value_wdl_v1"]
TransitionKind = Literal[
    "registry_v2",
    "dense_v3_registry",
    "compositional_v4_registry",
    "dense_v3_action_value_wdl_v1",
    "lora_v2_to_dense_v3",
]


class ExpertInitialization(BaseModel):
    """Explicit initialization for one target-only expert lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_expert_id: str
    mode: ExpertInitializationMode
    source_expert_id: str | None = None

    @field_validator("target_expert_id", "source_expert_id")
    @classmethod
    def valid_expert_id(cls, value: str | None) -> str | None:
        """Validate full SHA256 expert lineage identifiers."""
        return None if value is None else _validate_sha256(value, name="expert ID")

    @model_validator(mode="after")
    def consistent_mode(self) -> Self:
        """Require a donor only for clone initialization."""
        if self.mode == "clone" and self.source_expert_id is None:
            raise ValueError("clone initialization requires source_expert_id")
        if self.mode == "zero" and self.source_expert_id is not None:
            raise ValueError("zero initialization cannot declare source_expert_id")
        return self


class DeckRegistryTransitionConfig(BaseModel):
    """Immutable declaration for one new-run registry transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_checkpoint_sha256: str
    source_registry_sha256: str
    optimizer_state_path: Path
    optimizer_state_sha256: str
    optimizer_state_size_bytes: int
    scheduler_mode: TransitionSchedulerMode
    optimizer_mode: TransitionOptimizerMode = "transplant"
    topology_change: TransitionTopologyChange = "none"
    initializations: tuple[ExpertInitialization, ...] = ()
    retired_expert_ids: tuple[str, ...] = ()

    @field_validator(
        "source_checkpoint_sha256",
        "source_registry_sha256",
        "optimizer_state_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Validate source artifact and registry bindings."""
        return _validate_sha256(value, name="transition source")

    @field_validator("optimizer_state_path")
    @classmethod
    def immutable_optimizer_state_path(cls, value: Path) -> Path:
        """Reject moving sidecar aliases."""
        if any("latest" in part.lower() for part in value.parts):
            raise ValueError("transition optimizer state path cannot contain latest")
        return value

    @field_validator("optimizer_state_size_bytes")
    @classmethod
    def positive_optimizer_state_size(cls, value: int) -> int:
        """Require an immutable optimizer sidecar byte size."""
        if value <= 0:
            raise ValueError("optimizer_state_size_bytes must be positive")
        return value

    @field_validator("retired_expert_ids")
    @classmethod
    def valid_retired_experts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Normalize a unique deterministic retired-expert declaration."""
        normalized = tuple(
            _validate_sha256(item, name="retired expert") for item in value
        )
        if len(normalized) != len(set(normalized)):
            raise ValueError("retired expert IDs must be unique")
        return tuple(sorted(normalized))

    @model_validator(mode="after")
    def unique_target_initializations(self) -> Self:
        """Require one unambiguous initializer per declared target expert."""
        targets = tuple(item.target_expert_id for item in self.initializations)
        if len(targets) != len(set(targets)):
            raise ValueError("target expert initializations must be unique")
        if self.optimizer_mode == "fresh" and self.scheduler_mode != "restart":
            raise ValueError("a fresh transition optimizer requires scheduler restart")
        if self.topology_change != "none" and self.optimizer_mode != "fresh":
            raise ValueError("a topology transition requires a fresh optimizer")
        return self


@dataclass(frozen=True)
class ExpertTransition:
    """Derived action for one target expert."""

    target_expert_id: str
    action: ExpertTransitionAction
    source_expert_id: str | None
    source_deck_digests: tuple[str, ...]
    target_deck_digests: tuple[str, ...]

    @property
    def preserve_optimizer_state(self) -> bool:
        """Return whether this exact expert behavior contract is unchanged."""
        return self.action == "continued"


@dataclass(frozen=True)
class DeckRegistryTransitionPlan:
    """Validated source-to-target expert and route transition."""

    source_checkpoint_sha256: str
    source_registry_sha256: str
    target_registry_sha256: str
    source_model_config_sha256: str
    target_model_config_sha256: str
    optimizer_state_sha256: str
    optimizer_state_size_bytes: int
    scheduler_mode: TransitionSchedulerMode
    optimizer_mode: TransitionOptimizerMode
    topology_change: TransitionTopologyChange
    source_expert_ids: tuple[str, ...]
    experts: tuple[ExpertTransition, ...]
    retired_expert_ids: tuple[str, ...]
    manifest_sha256: str
    transition_kind: TransitionKind = "registry_v2"

    @property
    def by_target_expert(self) -> dict[str, ExpertTransition]:
        """Return target transition actions keyed by stable lineage."""
        return {item.target_expert_id: item for item in self.experts}

    def summary(self) -> dict[str, Any]:
        """Return a portable transition summary for training artifacts."""
        return {
            "transition_kind": self.transition_kind,
            "source_checkpoint_sha256": self.source_checkpoint_sha256,
            "source_registry_sha256": self.source_registry_sha256,
            "target_registry_sha256": self.target_registry_sha256,
            "source_model_config_sha256": self.source_model_config_sha256,
            "target_model_config_sha256": self.target_model_config_sha256,
            "optimizer_state_sha256": self.optimizer_state_sha256,
            "optimizer_state_size_bytes": self.optimizer_state_size_bytes,
            "scheduler_mode": self.scheduler_mode,
            "optimizer_mode": self.optimizer_mode,
            "topology_change": self.topology_change,
            "source_expert_ids": list(self.source_expert_ids),
            "manifest_sha256": self.manifest_sha256,
            "retired_expert_ids": list(self.retired_expert_ids),
            "experts": [
                {
                    "target_expert_id": item.target_expert_id,
                    "source_expert_id": item.source_expert_id,
                    "action": item.action,
                    "source_deck_digests": list(item.source_deck_digests),
                    "target_deck_digests": list(item.target_deck_digests),
                    "optimizer_state_preserved": item.preserve_optimizer_state,
                }
                for item in self.experts
            ],
        }


def build_deck_registry_transition_plan(
    source_config: AgentNetworkConfig,
    target_config: AgentNetworkConfig,
    declaration: DeckRegistryTransitionConfig,
) -> DeckRegistryTransitionPlan:
    """Validate and derive one explicit expert-registry or architecture transition."""
    source_conditioning = source_config.deck_conditioning
    target_conditioning = target_config.deck_conditioning
    source_is_dense_v3 = (
        source_conditioning is not None
        and source_conditioning.enabled
        and source_conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
    )
    target_is_dense_v3 = (
        target_conditioning is not None
        and target_conditioning.enabled
        and target_conditioning.architecture_version
        == DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
    )
    source_is_compositional_v4 = (
        source_conditioning is not None
        and source_conditioning.enabled
        and source_conditioning.architecture_version
        == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
    )
    target_is_compositional_v4 = (
        target_conditioning is not None
        and target_conditioning.enabled
        and target_conditioning.architecture_version
        == DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
    )
    if declaration.topology_change != "none" and not (
        source_is_dense_v3 and target_is_dense_v3
    ):
        raise ValueError("action-value topology transition requires dense v3")
    if source_is_compositional_v4 or target_is_compositional_v4:
        source = _routed_compositional_v4_conditioning(
            source_config,
            label="source",
        )
        target = _routed_compositional_v4_conditioning(
            target_config,
            label="target",
        )
        transition_kind: TransitionKind = "compositional_v4_registry"
    elif source_is_dense_v3:
        source = _routed_dense_v3_conditioning(source_config, label="source")
        target = _routed_dense_v3_conditioning(target_config, label="target")
        transition_kind = (
            "dense_v3_action_value_wdl_v1"
            if declaration.topology_change == "action_value_wdl_v1"
            else "dense_v3_registry"
        )
    elif target_is_dense_v3:
        source = _routed_v2_conditioning(source_config, label="source")
        target = _routed_dense_v3_conditioning(target_config, label="target")
        transition_kind = "lora_v2_to_dense_v3"
    else:
        source = _routed_v2_conditioning(source_config, label="source")
        target = _routed_v2_conditioning(target_config, label="target")
        transition_kind = "registry_v2"
    if source.resolved_registry_sha256 != declaration.source_registry_sha256:
        raise ValueError("transition source registry fingerprint does not match")
    if source.resolved_registry_sha256 is None:
        raise ValueError("transition source registry is unresolved")
    if target.resolved_registry_sha256 is None:
        raise ValueError("transition target registry is unresolved")
    if transition_kind in {
        "registry_v2",
        "dense_v3_registry",
        "compositional_v4_registry",
    }:
        if _structural_model_payload(source_config) != _structural_model_payload(
            target_config
        ):
            raise ValueError(
                "registry transition cannot change the routed model structure"
            )
    elif transition_kind == "dense_v3_action_value_wdl_v1":
        _validate_action_value_topology_change(source_config, target_config)
        if _structural_model_payload_without_action_value(
            source_config
        ) != _structural_model_payload_without_action_value(target_config):
            raise ValueError(
                "action-value transition cannot change another model structure"
            )
    elif _model_payload_without_conditioning(
        source_config
    ) != _model_payload_without_conditioning(target_config):
        raise ValueError(
            "dense-private transition cannot change the shared model structure"
        )

    source_routes = _expert_deck_digests(source)
    target_routes = _expert_deck_digests(target)
    source_experts = set(source_routes)
    target_experts = set(target_routes)
    if not target_experts:
        raise ValueError("registry transition target must retain at least one expert")
    initializers = {item.target_expert_id: item for item in declaration.initializations}
    target_only = target_experts - source_experts
    if set(initializers) != target_only:
        raise ValueError(
            "target-only experts require exactly one explicit initialization"
        )
    source_only = source_experts - target_experts
    if set(declaration.retired_expert_ids) != source_only:
        raise ValueError("source-only experts require an exact retired declaration")
    if transition_kind in {
        "lora_v2_to_dense_v3",
        "dense_v3_action_value_wdl_v1",
    }:
        if source_experts & target_experts:
            raise ValueError("topology transition requires new expert lineage IDs")
        for target_expert, initializer in initializers.items():
            donor = initializer.source_expert_id
            if initializer.mode != "clone" or donor is None:
                raise ValueError(
                    "topology-transition experts require explicit clone initializers"
                )
            if source_routes.get(donor) != target_routes[target_expert]:
                raise ValueError(
                    "topology clone donor must represent the same exact deck"
                )

    transitions: list[ExpertTransition] = []
    for expert_id in sorted(target_experts):
        source_digests = source_routes.get(expert_id, ())
        target_digests = target_routes[expert_id]
        if expert_id in source_experts:
            action: ExpertTransitionAction = (
                "continued" if source_digests == target_digests else "remapped"
            )
            source_expert_id: str | None = expert_id
        else:
            initializer = initializers[expert_id]
            if initializer.mode == "clone":
                donor = cast(str, initializer.source_expert_id)
                if donor not in source_experts:
                    raise ValueError(
                        "clone donor is not present in the source registry"
                    )
                if donor == expert_id:
                    raise ValueError("clone donor must differ from its target expert")
                action = "cloned"
                source_expert_id = donor
                source_digests = source_routes[donor]
            else:
                action = "zero"
                source_expert_id = None
        transitions.append(
            ExpertTransition(
                target_expert_id=expert_id,
                action=action,
                source_expert_id=source_expert_id,
                source_deck_digests=source_digests,
                target_deck_digests=target_digests,
            )
        )

    source_model_config_sha256 = _fingerprint(source_config.model_dump(mode="json"))
    target_model_config_sha256 = _fingerprint(target_config.model_dump(mode="json"))
    schema_by_transition_kind = {
        "registry_v2": "deck-registry-transition-v2",
        "dense_v3_registry": "dense-private-registry-transition-v1",
        "compositional_v4_registry": (
            "compositional-exact-capsule-registry-transition-v1"
        ),
        "dense_v3_action_value_wdl_v1": (
            "dense-private-action-value-wdl-transition-v1"
        ),
        "lora_v2_to_dense_v3": "dense-private-strategy-transition-v1",
    }
    payload = {
        "schema": schema_by_transition_kind[transition_kind],
        "source_checkpoint_sha256": declaration.source_checkpoint_sha256,
        "source_registry_sha256": source.resolved_registry_sha256,
        "target_registry_sha256": target.resolved_registry_sha256,
        "source_model_config_sha256": source_model_config_sha256,
        "target_model_config_sha256": target_model_config_sha256,
        "optimizer_state_sha256": declaration.optimizer_state_sha256,
        "optimizer_state_size_bytes": declaration.optimizer_state_size_bytes,
        "scheduler_mode": declaration.scheduler_mode,
        "optimizer_mode": declaration.optimizer_mode,
        "topology_change": declaration.topology_change,
        "source_expert_ids": sorted(source_experts),
        "retired_expert_ids": list(declaration.retired_expert_ids),
        "experts": [item.__dict__ for item in transitions],
    }
    return DeckRegistryTransitionPlan(
        source_checkpoint_sha256=declaration.source_checkpoint_sha256,
        source_registry_sha256=source.resolved_registry_sha256,
        target_registry_sha256=target.resolved_registry_sha256,
        source_model_config_sha256=source_model_config_sha256,
        target_model_config_sha256=target_model_config_sha256,
        optimizer_state_sha256=declaration.optimizer_state_sha256,
        optimizer_state_size_bytes=declaration.optimizer_state_size_bytes,
        scheduler_mode=declaration.scheduler_mode,
        optimizer_mode=declaration.optimizer_mode,
        topology_change=declaration.topology_change,
        source_expert_ids=tuple(sorted(source_experts)),
        experts=tuple(transitions),
        retired_expert_ids=declaration.retired_expert_ids,
        manifest_sha256=_fingerprint(payload),
        transition_kind=transition_kind,
    )


def migrate_deck_registry_weights(
    model: AgentPolicyValueNet,
    source_state_dict: Mapping[str, Any],
    plan: DeckRegistryTransitionPlan,
    *,
    source_config: AgentNetworkConfig | None = None,
) -> dict[str, Any]:
    """Strictly copy shared/continued/cloned weights into a target model."""
    if plan.transition_kind == "lora_v2_to_dense_v3":
        if source_config is None:
            raise ValueError("dense-private migration requires the source model config")
        from ptcg_rl.rl.dense_private_transition import (
            migrate_lora_v2_to_dense_private,
        )

        return migrate_lora_v2_to_dense_private(
            model,
            source_state_dict,
            source_config=source_config,
            expert_sources={
                item.target_expert_id: cast(str, item.source_expert_id)
                for item in plan.experts
            },
        )
    target_state = model.state_dict()
    migrated = dict(target_state)
    transitions = plan.by_target_expert
    copied_tensors = 0
    copied_numel = 0
    initialized_tensors = 0
    initialized_numel = 0
    consumed_source_keys: set[str] = set()
    parameter_actions: dict[str, dict[str, str | None]] = {}

    _validate_source_state_inventory(
        source_state_dict,
        target_state=target_state,
        plan=plan,
    )

    for target_key, target_value in target_state.items():
        expert_id = private_state_expert_id(target_key)
        source_key = target_key
        if expert_id is not None:
            transition = transitions.get(expert_id)
            if transition is None:
                raise ValueError(
                    f"target private state has no transition: {target_key}"
                )
            if transition.source_expert_id is None:
                initialized_tensors += 1
                initialized_numel += int(target_value.numel())
                parameter_actions[target_key] = {
                    "action": "initialized",
                    "source_key": None,
                }
                continue
            source_key = _replace_expert_id(
                target_key,
                target_expert_id=expert_id,
                source_expert_id=transition.source_expert_id,
            )
        source_value = source_state_dict.get(source_key)
        if source_value is None and _is_fresh_topology_parameter(
            target_key,
            plan=plan,
        ):
            initialized_tensors += 1
            initialized_numel += int(target_value.numel())
            parameter_actions[target_key] = {
                "action": "fresh_topology_initialization",
                "source_key": None,
            }
            continue
        if not isinstance(source_value, Tensor):
            raise ValueError(f"transition source tensor is missing: {source_key}")
        _validate_weight_tensor(source_value, target_value, name=target_key)
        migrated[target_key] = source_value.detach().clone()
        consumed_source_keys.add(source_key)
        copied_tensors += 1
        copied_numel += int(target_value.numel())
        parameter_actions[target_key] = {
            "action": "copied",
            "source_key": source_key,
        }

    for raw_key in source_state_dict:
        key = str(raw_key)
        if key in consumed_source_keys:
            continue
        if private_state_expert_id(key) is None:
            raise ValueError(f"transition source has unexpected shared state: {key}")

    model.load_state_dict(migrated, strict=True)
    return {
        "copied_tensors": copied_tensors,
        "copied_numel": copied_numel,
        "initialized_tensors": initialized_tensors,
        "initialized_numel": initialized_numel,
        "parameter_actions_sha256": _fingerprint(parameter_actions),
    }


def _validate_source_state_inventory(
    source_state: Mapping[str, Any],
    *,
    target_state: Mapping[str, Any],
    plan: DeckRegistryTransitionPlan,
) -> None:
    """Require exact shared keys and complete private schemas per source expert."""
    source_shared = {
        str(key) for key in source_state if private_state_expert_id(str(key)) is None
    }
    target_shared = {
        str(key) for key in target_state if private_state_expert_id(str(key)) is None
    }
    fresh_target_shared = {
        key for key in target_shared if _is_fresh_topology_parameter(key, plan=plan)
    }
    if source_shared != target_shared - fresh_target_shared:
        raise ValueError(
            "transition source shared-state inventory mismatch: "
            f"missing={sorted(target_shared - fresh_target_shared - source_shared)}, "
            f"extra={sorted(source_shared - target_shared)}"
        )

    target_schemas = _private_state_schemas(target_state)
    if not target_schemas:
        raise ValueError("transition target has no private parameter schema")
    expected_schema = next(iter(target_schemas.values()))
    if any(schema != expected_schema for schema in target_schemas.values()):
        raise ValueError("transition target experts have inconsistent state schemas")
    source_schemas = _private_state_schemas(source_state)
    expected_source_experts = set(plan.source_expert_ids)
    if set(source_schemas) != expected_source_experts:
        raise ValueError(
            "transition source private expert inventory mismatch: "
            f"expected={sorted(expected_source_experts)}, "
            f"actual={sorted(source_schemas)}"
        )
    for expert_id, schema in source_schemas.items():
        if schema != expected_schema:
            raise ValueError(
                f"transition source private state schema mismatch: {expert_id}"
            )


def _private_state_schemas(
    state_dict: Mapping[str, Any],
) -> dict[str, frozenset[str]]:
    grouped: dict[str, set[str]] = {}
    for raw_key in state_dict:
        key = str(raw_key)
        expert_id = private_state_expert_id(key)
        if expert_id is None:
            continue
        normalized = key.replace(f".deck_{expert_id}.", ".deck_<expert>.")
        grouped.setdefault(expert_id, set()).add(normalized)
    return {expert_id: frozenset(schema) for expert_id, schema in grouped.items()}


def private_state_expert_id(name: str) -> str | None:
    """Extract the stable expert ID from one private parameter/state name."""
    if not name.startswith(_PRIVATE_STATE_PREFIXES):
        return None
    matches = [
        chunk.removeprefix("deck_")
        for chunk in name.split(".")
        if chunk.startswith("deck_") and len(chunk) == 69
    ]
    if len(matches) != 1:
        raise ValueError(f"private state key has no unique expert ID: {name}")
    return _validate_sha256(matches[0], name="private state expert")


def _routed_v2_conditioning(config: AgentNetworkConfig, *, label: str) -> Any:
    conditioning = config.deck_conditioning
    if (
        conditioning is None
        or not conditioning.enabled
        or conditioning.architecture_version
        != DECK_CONDITIONING_LORA_ARCHITECTURE_VERSION
        or conditioning.lora is None
        or conditioning.lora.export_mode != "routed"
        or conditioning.deck_context_mode != "encoded"
    ):
        raise ValueError(f"transition {label} must be routed architecture v2")
    return conditioning


def _routed_dense_v3_conditioning(
    config: AgentNetworkConfig,
    *,
    label: str,
) -> Any:
    conditioning = config.deck_conditioning
    if (
        conditioning is None
        or not conditioning.enabled
        or conditioning.architecture_version
        != DECK_CONDITIONING_DENSE_PRIVATE_ARCHITECTURE_VERSION
        or conditioning.lora is not None
        or conditioning.dense_private is None
        or conditioning.dense_private.export_mode != "routed"
        or conditioning.deck_context_mode != "encoded"
    ):
        raise ValueError(f"transition {label} must be routed dense architecture v3")
    return conditioning


def _routed_compositional_v4_conditioning(
    config: AgentNetworkConfig,
    *,
    label: str,
) -> Any:
    conditioning = config.deck_conditioning
    if (
        conditioning is None
        or not conditioning.enabled
        or conditioning.architecture_version
        != DECK_CONDITIONING_COMPOSITIONAL_ARCHITECTURE_VERSION
        or conditioning.lora is not None
        or conditioning.dense_private is not None
        or conditioning.compositional is None
        or conditioning.compositional.export_mode != "routed"
        or conditioning.deck_context_mode != "encoded"
    ):
        raise ValueError(
            f"transition {label} must be routed compositional architecture v4"
        )
    return conditioning


def _expert_deck_digests(conditioning: Any) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    for route in conditioning.expert_routes:
        grouped.setdefault(route.expert_id, []).append(route.deck_digest)
    return {expert_id: tuple(sorted(digests)) for expert_id, digests in grouped.items()}


def _structural_model_payload(config: AgentNetworkConfig) -> dict[str, Any]:
    payload = config.model_dump(mode="json")
    conditioning = cast(dict[str, Any], payload["deck_conditioning"])
    conditioning.pop("expert_routes", None)
    conditioning.pop("resolved_registry_sha256", None)
    return payload


def _structural_model_payload_without_action_value(
    config: AgentNetworkConfig,
) -> dict[str, Any]:
    """Return routed structure while excluding the declared new Q head."""
    payload = _structural_model_payload(config)
    payload.pop("action_value", None)
    return payload


def _validate_action_value_topology_change(
    source: AgentNetworkConfig,
    target: AgentNetworkConfig,
) -> None:
    """Require the one supported disabled-to-categorical-Q migration."""
    if source.action_value.enabled:
        raise ValueError("action-value topology source must not already contain Q")
    if not target.action_value.enabled or target.action_value.architecture_version != 1:
        raise ValueError("action-value topology target must enable W/D/L Q version 1")


def _is_fresh_topology_parameter(
    name: str,
    *,
    plan: DeckRegistryTransitionPlan,
) -> bool:
    """Return whether a tensor is explicitly fresh in this topology change."""
    return plan.transition_kind == "dense_v3_action_value_wdl_v1" and name.startswith(
        "action_value_head."
    )


def _model_payload_without_conditioning(config: AgentNetworkConfig) -> dict[str, Any]:
    """Return shared architecture fields for a v2-to-v3 transition."""
    payload = config.model_dump(mode="json")
    payload.pop("deck_conditioning", None)
    return payload


def _replace_expert_id(
    name: str,
    *,
    target_expert_id: str,
    source_expert_id: str,
) -> str:
    target = f".deck_{target_expert_id}."
    if name.count(target) != 1:
        raise ValueError(f"target expert key is malformed: {name}")
    return name.replace(target, f".deck_{source_expert_id}.")


def _validate_weight_tensor(source: Tensor, target: Tensor, *, name: str) -> None:
    if source.shape != target.shape or source.dtype != target.dtype:
        raise ValueError(f"transition tensor shape or dtype mismatch: {name}")
    if source.is_floating_point() and not torch.isfinite(source).all():
        raise ValueError(f"transition tensor is non-finite: {name}")


def _validate_sha256(value: str, *, name: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return normalized


def _fingerprint(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "DeckRegistryTransitionConfig",
    "DeckRegistryTransitionPlan",
    "ExpertInitialization",
    "ExpertTransition",
    "build_deck_registry_transition_plan",
    "migrate_deck_registry_weights",
    "private_state_expert_id",
]
