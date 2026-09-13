"""Fail-closed simple-stateless v1 to v2 checkpoint-pair transition."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Self, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn

from ptcg_rl.decks.identity import deck_module_key
from ptcg_rl.decks.registry import DeckExpertRoute
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessModelConfig,
    SimpleStatelessPolicyValueNet,
)
from ptcg_rl.rl import stateless_topology_transition_ops as transition_ops
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.stateless_checkpoint import LoadedStatelessCheckpointPair

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class StatelessTopologyRouteClone(BaseModel):
    """One same-deck clone into a fresh v2 expert lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    source_expert_id: str
    target_expert_id: str
    mode: Literal["clone"] = "clone"

    @field_validator("deck_digest", "source_expert_id", "target_expert_id")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require immutable exact-deck and expert identities."""
        return _fingerprint(value)

    @model_validator(mode="after")
    def fresh_lineage(self) -> Self:
        """A topology change must never retain the old parameter lineage."""
        if self.source_expert_id == self.target_expert_id:
            raise ValueError("v2 topology clone requires a fresh target expert lineage")
        return self

    @property
    def source_module_key(self) -> str:
        """Return the physical v1 source module name."""
        return deck_module_key(self.source_expert_id)

    @property
    def target_module_key(self) -> str:
        """Return the physical v2 target module name."""
        return deck_module_key(self.target_expert_id)


class StatelessTopologyTransitionDeclaration(BaseModel):
    """Predeclared immutable pair, configurations, and lineage conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "simple-stateless-v1-to-v2-declaration-v1",
        "simple-stateless-v2-to-generalist-sequence-v1-declaration-v1",
    ] = "simple-stateless-v1-to-v2-declaration-v1"
    source_architecture: Literal[
        "simple_stateless_v1",
        "simple_stateless_v2",
    ] = "simple_stateless_v1"
    target_architecture: Literal[
        "simple_stateless_v2",
        "generalist_sequence_v1",
    ] = "simple_stateless_v2"
    source_registry_sha256: str
    target_registry_sha256: str
    source_pair_manifest_sha256: str
    source_policy_sha256: str
    source_learner_state_sha256: str
    source_model_config_fingerprint: str
    source_model_state_fingerprint: str
    target_model_config_fingerprint: str
    route_clones: tuple[StatelessTopologyRouteClone, ...]

    @field_validator(
        "source_registry_sha256",
        "target_registry_sha256",
        "source_pair_manifest_sha256",
        "source_policy_sha256",
        "source_learner_state_sha256",
        "source_model_config_fingerprint",
        "source_model_state_fingerprint",
        "target_model_config_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical artifact and configuration fingerprints."""
        return _fingerprint(value)

    @model_validator(mode="after")
    def complete_unique_clones(self) -> Self:
        """Keep the route declaration deterministic and collision-free."""
        expected_format = {
            (
                "simple_stateless_v1",
                "simple_stateless_v2",
            ): "simple-stateless-v1-to-v2-declaration-v1",
            (
                "simple_stateless_v2",
                "generalist_sequence_v1",
            ): "simple-stateless-v2-to-generalist-sequence-v1-declaration-v1",
        }.get((self.source_architecture, self.target_architecture))
        if expected_format is None or self.format != expected_format:
            raise ValueError("unsupported stateless topology architecture transition")
        if not self.route_clones:
            raise ValueError("v2 topology transition requires exact route clones")
        digests = tuple(item.deck_digest for item in self.route_clones)
        source_experts = tuple(item.source_expert_id for item in self.route_clones)
        target_experts = tuple(item.target_expert_id for item in self.route_clones)
        if digests != tuple(sorted(digests)):
            raise ValueError("v2 topology route clones must be sorted by deck_digest")
        for name, values in (
            ("deck digests", digests),
            ("source expert IDs", source_experts),
            ("target expert IDs", target_experts),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"v2 topology {name} must be unique")
        if set(source_experts) & set(target_experts):
            raise ValueError("v2 expert lineages must be fresh across the whole roster")
        return self


class StatelessTopologyTransitionPlan(BaseModel):
    """Validated v1-to-v2 topology and same-deck lineage mapping."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_registry_sha256: str
    target_registry_sha256: str
    route_clones: tuple[StatelessTopologyRouteClone, ...]

    @field_validator("source_registry_sha256", "target_registry_sha256")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical registry fingerprints."""
        return _fingerprint(value)

    @property
    def module_key_map(self) -> dict[str, str]:
        """Map stable v1 state-dict path segments to fresh v2 segments."""
        return {
            clone.source_module_key: clone.target_module_key
            for clone in self.route_clones
        }

    def target_tensor_name(self, source_name: str) -> str:
        """Map one source state name through the declared route transition."""
        return transition_ops.rebind_name(source_name, self.module_key_map)

    @property
    def summary(self) -> dict[str, object]:
        """Return the path-free route lifecycle evidence."""
        return {
            "source_registry_sha256": self.source_registry_sha256,
            "target_registry_sha256": self.target_registry_sha256,
            "routes": [
                {
                    "deck_digest": item.deck_digest,
                    "source_expert_id": item.source_expert_id,
                    "target_expert_id": item.target_expert_id,
                    "mode": item.mode,
                }
                for item in self.route_clones
            ],
        }


class StatelessTopologyModelAudit(BaseModel):
    """State inventory and output-inert initialization evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_state_fingerprint: str
    target_state_fingerprint: str
    state_key_mapping_fingerprint: str
    initialized_state_fingerprint: str
    copied_tensors: int = Field(ge=0)
    remapped_tensors: int = Field(ge=0)
    initialized_tensors: int = Field(ge=1)
    initialized_tensor_names: tuple[str, ...]
    inert_output_tensor_names: tuple[str, ...]
    output_inert_validated: Literal[True] = True

    @field_validator(
        "source_state_fingerprint",
        "target_state_fingerprint",
        "state_key_mapping_fingerprint",
        "initialized_state_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical state and inventory fingerprints."""
        return _fingerprint(value)


class StatelessTopologyOptimizerAudit(BaseModel):
    """Stable-name AdamW transplant evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    parameter_mapping_fingerprint: str
    copied_parameters: int = Field(ge=0)
    remapped_parameters: int = Field(ge=0)
    preserved_without_state: int = Field(ge=0)
    reset_parameters: int = Field(ge=1)

    @field_validator("parameter_mapping_fingerprint")
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require a canonical name-mapping identity."""
        return _fingerprint(value)


class StatelessTopologyTransitionAuditReport(BaseModel):
    """Complete immutable audit record for one v1-to-v2 pair conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "simple-stateless-v1-to-v2-audit-v1",
        "simple-stateless-v2-to-generalist-sequence-v1-audit-v1",
    ] = "simple-stateless-v1-to-v2-audit-v1"
    declaration: StatelessTopologyTransitionDeclaration
    plan: StatelessTopologyTransitionPlan
    model: StatelessTopologyModelAudit
    optimizer: StatelessTopologyOptimizerAudit


@dataclass(frozen=True)
class StatelessTopologyTransitionResult:
    """Converted model state plus its immutable audit record."""

    state_dict: dict[str, Tensor]
    report: StatelessTopologyTransitionAuditReport


def stateless_topology_config_fingerprint(config: BaseModel) -> str:
    """Return the canonical model-config identity used by the declaration."""
    return model_config_fingerprint(config)


def build_stateless_topology_transition_plan(
    source: BaseModel,
    target: BaseModel,
    declaration: StatelessTopologyTransitionDeclaration,
) -> StatelessTopologyTransitionPlan:
    """Validate an exact-roster v1-to-v2 conversion with fresh lineages."""
    source_architecture, source_registry, source_routes = _routed_config(source)
    target_architecture, target_registry, target_routes = _routed_config(target)
    if (
        source_architecture != declaration.source_architecture
        or target_architecture != declaration.target_architecture
    ):
        raise ValueError("stateless topology transition architecture mismatch")
    if source_registry != declaration.source_registry_sha256:
        raise ValueError("stateless topology source registry is not authorized")
    if target_registry != declaration.target_registry_sha256:
        raise ValueError("stateless topology target registry is not authorized")
    if (
        stateless_topology_config_fingerprint(source)
        != declaration.source_model_config_fingerprint
    ):
        raise ValueError("stateless topology source model config is not authorized")
    if (
        stateless_topology_config_fingerprint(target)
        != declaration.target_model_config_fingerprint
    ):
        raise ValueError("stateless topology target model config is not authorized")

    source_by_deck = {route.deck_digest: route for route in source_routes}
    target_by_deck = {route.deck_digest: route for route in target_routes}
    clones_by_deck = {clone.deck_digest: clone for clone in declaration.route_clones}
    if set(source_by_deck) != set(target_by_deck):
        raise ValueError("stateless topology transition cannot also edit the roster")
    if set(source_by_deck) != set(clones_by_deck):
        raise ValueError("stateless topology route clones must cover the exact roster")
    for digest, clone in clones_by_deck.items():
        source_route = source_by_deck[digest]
        target_route = target_by_deck[digest]
        if source_route.expert_id != clone.source_expert_id:
            raise ValueError("stateless topology clone source lineage mismatch")
        if target_route.expert_id != clone.target_expert_id:
            raise ValueError("stateless topology clone target lineage mismatch")
    return StatelessTopologyTransitionPlan(
        source_registry_sha256=source_registry,
        target_registry_sha256=target_registry,
        route_clones=declaration.route_clones,
    )


def migrate_stateless_topology_model(
    *,
    target: nn.Module,
    source_state: Mapping[str, Tensor],
    plan: StatelessTopologyTransitionPlan,
    inert_output_tensor_names: Sequence[str],
) -> StatelessTopologyModelAudit:
    """Copy all v1 tensors and retain only audited inert v2 initialization."""
    evidence = transition_ops.migrate_topology_model(
        target=target,
        source_state=source_state,
        module_key_map=plan.module_key_map,
        inert_output_tensor_names=inert_output_tensor_names,
    )
    return StatelessTopologyModelAudit(
        source_state_fingerprint=evidence.source_state_fingerprint,
        target_state_fingerprint=evidence.target_state_fingerprint,
        state_key_mapping_fingerprint=evidence.state_key_mapping_fingerprint,
        initialized_state_fingerprint=evidence.initialized_state_fingerprint,
        copied_tensors=evidence.copied_tensors,
        remapped_tensors=evidence.remapped_tensors,
        initialized_tensors=len(evidence.initialized_tensor_names),
        initialized_tensor_names=evidence.initialized_tensor_names,
        inert_output_tensor_names=evidence.inert_output_tensor_names,
    )


def _transplant_stateless_topology_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: nn.Module,
    source_parameter_names: Sequence[Sequence[str]],
    source_optimizer_state: Mapping[str, Any],
    plan: StatelessTopologyTransitionPlan,
) -> StatelessTopologyOptimizerAudit:
    """Transplant AdamW moments by stable/remapped names and reset v2 state."""
    evidence = transition_ops.transplant_topology_optimizer(
        optimizer=optimizer,
        target_model=target_model,
        source_parameter_names=source_parameter_names,
        source_optimizer_state=source_optimizer_state,
        module_key_map=plan.module_key_map,
    )
    return StatelessTopologyOptimizerAudit(
        parameter_mapping_fingerprint=evidence.parameter_mapping_fingerprint,
        copied_parameters=evidence.copied_parameters,
        remapped_parameters=evidence.remapped_parameters,
        preserved_without_state=evidence.preserved_without_state,
        reset_parameters=evidence.reset_parameters,
    )


def stateless_topology_source_parameter_names(
    source_model: nn.Module,
    source_optimizer_state: Mapping[str, Any],
) -> tuple[tuple[str, ...], ...]:
    """Recover the current stateless one-group optimizer's stable name order."""
    source_groups = source_optimizer_state.get("param_groups")
    if not isinstance(source_groups, list) or len(source_groups) != 1:
        raise ValueError(
            "stateless topology source optimizer must have exactly one group"
        )
    parameter_ids = source_groups[0].get("params")
    names = tuple(name for name, _ in source_model.named_parameters())
    if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
        raise ValueError("stateless topology source optimizer inventory is malformed")
    return (names,)


def transplant_stateless_topology_optimizer(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: nn.Module,
    source_model_config: SimpleStatelessModelConfig,
    source_optimizer_state: Mapping[str, Any],
    plan: StatelessTopologyTransitionPlan,
) -> StatelessTopologyOptimizerAudit:
    """Reconstruct v1 stable names and transplant its current AdamW state."""
    with torch.random.fork_rng(devices=[]):
        source_model = SimpleStatelessPolicyValueNet(
            source_model_config,
            load_static_features=False,
            initialize=False,
        )
    source_names = stateless_topology_source_parameter_names(
        source_model,
        source_optimizer_state,
    )
    return _transplant_stateless_topology_optimizer_state(
        optimizer=optimizer,
        target_model=target_model,
        source_parameter_names=source_names,
        source_optimizer_state=source_optimizer_state,
        plan=plan,
    )


def migrate_stateless_topology_from_pair(
    *,
    source: LoadedStatelessCheckpointPair,
    source_model: nn.Module,
    target_model: nn.Module,
    target_optimizer: torch.optim.Optimizer,
    target_config: BaseModel,
    declaration: StatelessTopologyTransitionDeclaration,
    inert_output_tensor_names: Sequence[str],
) -> StatelessTopologyTransitionResult:
    """Convert one exactly bound v1 policy/learner pair into an inert v2 state."""
    validate_stateless_topology_source_pair(source, declaration)
    source_model.load_state_dict(source.model_state, strict=True)
    plan = build_stateless_topology_transition_plan(
        source.model_config_value,
        target_config,
        declaration,
    )
    model_audit = migrate_stateless_topology_model(
        target=target_model,
        source_state=source.model_state,
        plan=plan,
        inert_output_tensor_names=inert_output_tensor_names,
    )
    if model_audit.source_state_fingerprint != declaration.source_model_state_fingerprint:
        raise ValueError("stateless topology source model state is not authorized")
    optimizer_audit = _transplant_stateless_topology_optimizer_state(
        optimizer=target_optimizer,
        target_model=target_model,
        source_parameter_names=stateless_topology_source_parameter_names(
            source_model,
            source.optimizer_state,
        ),
        source_optimizer_state=source.optimizer_state,
        plan=plan,
    )
    report = StatelessTopologyTransitionAuditReport(
        format=(
            "simple-stateless-v2-to-generalist-sequence-v1-audit-v1"
            if declaration.target_architecture == "generalist_sequence_v1"
            else "simple-stateless-v1-to-v2-audit-v1"
        ),
        declaration=declaration,
        plan=plan,
        model=model_audit,
        optimizer=optimizer_audit,
    )
    converted = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in target_model.state_dict().items()
    }
    if canonical_model_state_fingerprint(converted) != model_audit.target_state_fingerprint:
        raise RuntimeError("stateless topology target state changed after audit")
    return StatelessTopologyTransitionResult(state_dict=converted, report=report)


def validate_stateless_topology_source_pair(
    source: LoadedStatelessCheckpointPair,
    declaration: StatelessTopologyTransitionDeclaration,
) -> None:
    expected = {
        "pair manifest": (
            source.pair.pair_manifest_sha256,
            declaration.source_pair_manifest_sha256,
        ),
        "policy": (source.pair.policy_sha256, declaration.source_policy_sha256),
        "learner state": (
            source.pair.learner_state_sha256,
            declaration.source_learner_state_sha256,
        ),
        "model state": (
            source.pair.policy_model_fingerprint,
            declaration.source_model_state_fingerprint,
        ),
    }
    for label, (actual, declared) in expected.items():
        if actual != declared:
            raise ValueError(f"stateless topology source {label} is not authorized")


def _routed_config(
    config: BaseModel,
) -> tuple[str, str, tuple[DeckExpertRoute, ...]]:
    architecture = getattr(config, "architecture", None)
    registry = getattr(config, "resolved_registry_sha256", None)
    routes = getattr(config, "exact_routes", None)
    if not isinstance(architecture, str):
        raise ValueError("stateless topology config has no architecture")
    if not isinstance(registry, str) or _SHA256_PATTERN.fullmatch(registry) is None:
        raise ValueError("stateless topology config has no routed registry")
    if not isinstance(routes, tuple) or not all(
        isinstance(route, DeckExpertRoute) for route in routes
    ):
        raise ValueError("stateless topology config has invalid exact routes")
    return architecture, registry, cast(tuple[DeckExpertRoute, ...], routes)


def _fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("stateless topology identities must be lowercase SHA-256")
    return normalized


__all__ = [
    "StatelessTopologyModelAudit",
    "StatelessTopologyOptimizerAudit",
    "StatelessTopologyRouteClone",
    "StatelessTopologyTransitionAuditReport",
    "StatelessTopologyTransitionDeclaration",
    "StatelessTopologyTransitionPlan",
    "StatelessTopologyTransitionResult",
    "build_stateless_topology_transition_plan",
    "migrate_stateless_topology_from_pair",
    "migrate_stateless_topology_model",
    "stateless_topology_config_fingerprint",
    "stateless_topology_source_parameter_names",
    "transplant_stateless_topology_optimizer",
    "validate_stateless_topology_source_pair",
]
