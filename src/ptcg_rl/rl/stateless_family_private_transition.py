"""Fail-closed generalist-sequence v2 to family-private v3 migration."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Self

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn

from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.model.sequence.config import (
    GENERALIST_SEQUENCE_V2_ARCHITECTURE,
    GENERALIST_SEQUENCE_V3_ARCHITECTURE,
)
from ptcg_rl.model.simple_stateless.config import SimpleStatelessModelConfig
from ptcg_rl.model.simple_stateless.family_private import (
    FAMILY_PRIVATE_CLONED_LAYERS,
    FAMILY_PRIVATE_SHARED_LAYERS,
)
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.stateless_checkpoint import LoadedStatelessCheckpointPair
from ptcg_rl.rl.stateless_topology_transition import StatelessTopologyRouteClone

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SOURCE_UPPER_PREFIX = "backbone.trunk.layers."
_TARGET_GENERIC_PREFIX = "backbone.family_private.generic_upper."
_TARGET_FAMILY_PREFIX = "backbone.family_private.tails."


class StatelessFamilyPrivateRouteInitialization(BaseModel):
    """One target-only exact leaf created during the v2-to-v3 migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_deck_digest: str
    target_expert_id: str
    mode: Literal["zero"] = "zero"

    @field_validator("target_deck_digest", "target_expert_id")
    @classmethod
    def valid_identity(cls, value: str) -> str:
        """Require immutable exact-deck and lineage identities."""
        return _fingerprint(value)


class StatelessFamilyPrivateTransitionDeclaration(BaseModel):
    """Bind one immutable RL pair to one exact v2-to-v3 conversion."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "generalist-sequence-v2-to-family-private-v3-declaration-v1"
    ] = "generalist-sequence-v2-to-family-private-v3-declaration-v1"
    source_architecture: Literal["generalist_sequence_v2"] = (
        "generalist_sequence_v2"
    )
    target_architecture: Literal["generalist_sequence_v3"] = (
        "generalist_sequence_v3"
    )
    source_registry_sha256: str
    target_registry_sha256: str
    target_family_registry_sha256: str
    source_pair_manifest_sha256: str
    source_policy_sha256: str
    source_learner_state_sha256: str
    source_model_config_fingerprint: str
    source_model_state_fingerprint: str
    target_model_config_fingerprint: str
    route_clones: tuple[StatelessTopologyRouteClone, ...]
    route_initializations: tuple[
        StatelessFamilyPrivateRouteInitialization, ...
    ] = ()

    @field_validator(
        "source_registry_sha256",
        "target_registry_sha256",
        "target_family_registry_sha256",
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
        """Require a fresh exact lineage clone for every roster member."""
        if not self.route_clones:
            raise ValueError("family-private transition requires route clones")
        digests = tuple(clone.deck_digest for clone in self.route_clones)
        sources = tuple(clone.source_expert_id for clone in self.route_clones)
        targets = tuple(clone.target_expert_id for clone in self.route_clones)
        initialized_digests = tuple(
            item.target_deck_digest for item in self.route_initializations
        )
        initialized_targets = tuple(
            item.target_expert_id for item in self.route_initializations
        )
        if digests != tuple(sorted(digests)):
            raise ValueError("family-private route clones must be deck-sorted")
        if initialized_digests != tuple(sorted(initialized_digests)):
            raise ValueError(
                "family-private route initializations must be deck-sorted"
            )
        for label, values in (
            ("deck digests", digests),
            ("source experts", sources),
            ("target experts", targets),
            ("initialized deck digests", initialized_digests),
            ("initialized target experts", initialized_targets),
        ):
            if len(values) != len(set(values)):
                raise ValueError(f"family-private {label} must be unique")
        if set(digests) & set(initialized_digests):
            raise ValueError(
                "family-private decks cannot be both cloned and initialized"
            )
        all_targets = set(targets) | set(initialized_targets)
        if len(all_targets) != len(targets) + len(initialized_targets):
            raise ValueError("family-private target expert lineages must be unique")
        if set(sources) & all_targets:
            raise ValueError("family-private exact lineages must all be fresh")
        return self


class StatelessFamilyPrivateTransitionAudit(BaseModel):
    """Complete tensor inventory for one v2-to-v3 policy migration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "generalist-sequence-v2-to-family-private-v3-audit-v1"
    ] = "generalist-sequence-v2-to-family-private-v3-audit-v1"
    declaration_fingerprint: str
    source_state_fingerprint: str
    target_state_fingerprint: str
    state_mapping_fingerprint: str
    initialized_state_fingerprint: str
    source_tensors: int = Field(gt=0)
    inherited_target_tensors: int = Field(gt=0)
    cloned_upper_target_tensors: int = Field(gt=0)
    initialized_exact_target_tensors: int = Field(ge=0)
    initialized_appended_target_tensors: int = Field(gt=0)
    initialized_tensors: int = Field(gt=0)
    initialized_tensor_names: tuple[str, ...]
    inert_output_tensor_names: tuple[str, ...]
    output_inert_validated: Literal[True] = True

    @field_validator(
        "declaration_fingerprint",
        "source_state_fingerprint",
        "target_state_fingerprint",
        "state_mapping_fingerprint",
        "initialized_state_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical migration evidence identities."""
        return _fingerprint(value)


@dataclass(frozen=True)
class StatelessFamilyPrivateTransitionResult:
    """Migrated target state and its immutable audit."""

    state_dict: dict[str, Tensor]
    audit: StatelessFamilyPrivateTransitionAudit


def family_private_transition_fingerprint(
    declaration: StatelessFamilyPrivateTransitionDeclaration,
) -> str:
    """Hash one complete declarative migration authorization."""
    return fingerprint_payload(declaration.model_dump(mode="json"))


def migrate_family_private_from_pair(
    *,
    source: LoadedStatelessCheckpointPair,
    target_model: nn.Module,
    target_config: SimpleStatelessModelConfig,
    declaration: StatelessFamilyPrivateTransitionDeclaration,
    inert_output_tensor_names: Sequence[str],
) -> StatelessFamilyPrivateTransitionResult:
    """Clone old upper blocks per family and retain inert appended blocks."""
    _validate_source_pair(source, declaration)
    source_config = source.model_config_value
    if not isinstance(source_config, SimpleStatelessModelConfig):
        raise ValueError("family-private source is not a simple-stateless model")
    return migrate_family_private_model(
        source_config=source_config,
        source_state=source.model_state,
        target_model=target_model,
        target_config=target_config,
        declaration=declaration,
        inert_output_tensor_names=inert_output_tensor_names,
    )


def migrate_family_private_model(
    *,
    source_config: SimpleStatelessModelConfig,
    source_state: Mapping[str, Tensor],
    target_model: nn.Module,
    target_config: SimpleStatelessModelConfig,
    declaration: StatelessFamilyPrivateTransitionDeclaration,
    inert_output_tensor_names: Sequence[str],
) -> StatelessFamilyPrivateTransitionResult:
    """Migrate a validated policy state without carrying optimizer state."""
    exact_key_map, family_keys, initialized_exact_keys = _validate_configs(
        source_config,
        target_config,
        declaration,
    )
    audit = _migrate_model(
        target_model,
        source_state=source_state,
        exact_key_map=exact_key_map,
        family_keys=family_keys,
        initialized_exact_keys=initialized_exact_keys,
        inert_output_tensor_names=inert_output_tensor_names,
        declaration=declaration,
    )
    converted = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in target_model.state_dict().items()
    }
    if canonical_model_state_fingerprint(converted) != audit.target_state_fingerprint:
        raise RuntimeError("family-private target state changed after migration audit")
    return StatelessFamilyPrivateTransitionResult(
        state_dict=converted,
        audit=audit,
    )


def _validate_configs(
    source: SimpleStatelessModelConfig,
    target: SimpleStatelessModelConfig,
    declaration: StatelessFamilyPrivateTransitionDeclaration,
) -> tuple[dict[str, str], tuple[str, ...], tuple[str, ...]]:
    if (
        source.architecture != GENERALIST_SEQUENCE_V2_ARCHITECTURE
        or target.architecture != GENERALIST_SEQUENCE_V3_ARCHITECTURE
    ):
        raise ValueError("family-private transition architecture mismatch")
    if source.resolved_registry_sha256 != declaration.source_registry_sha256:
        raise ValueError("family-private source registry is not authorized")
    if target.resolved_registry_sha256 != declaration.target_registry_sha256:
        raise ValueError("family-private target registry is not authorized")
    if (
        target.resolved_family_registry_sha256
        != declaration.target_family_registry_sha256
    ):
        raise ValueError("family-private target family registry is not authorized")
    if model_config_fingerprint(source) != declaration.source_model_config_fingerprint:
        raise ValueError("family-private source model config is not authorized")
    if model_config_fingerprint(target) != declaration.target_model_config_fingerprint:
        raise ValueError("family-private target model config is not authorized")
    source_by_deck = {route.deck_digest: route for route in source.exact_routes}
    target_by_deck = {route.deck_digest: route for route in target.exact_routes}
    clones = {clone.deck_digest: clone for clone in declaration.route_clones}
    initializations = {
        item.target_deck_digest: item
        for item in declaration.route_initializations
    }
    family_decks = {route.deck_digest for route in target.family_routes}
    if set(source_by_deck) != set(clones):
        raise ValueError(
            "family-private route clones must cover the complete source roster"
        )
    if set(target_by_deck) != set(clones) | set(initializations):
        raise ValueError(
            "family-private target roster differs from declared clones and additions"
        )
    if set(target_by_deck) != family_decks:
        raise ValueError("family-private families must cover the target roster")
    exact_key_map: dict[str, str] = {}
    for digest, clone in clones.items():
        source_route = source_by_deck[digest]
        target_route = target_by_deck[digest]
        if source_route.expert_id != clone.source_expert_id:
            raise ValueError("family-private source expert lineage mismatch")
        if target_route.expert_id != clone.target_expert_id:
            raise ValueError("family-private target expert lineage mismatch")
        exact_key_map[source_route.module_key] = target_route.module_key
    initialized_exact_keys: list[str] = []
    for digest, initialization in initializations.items():
        target_route = target_by_deck[digest]
        if target_route.expert_id != initialization.target_expert_id:
            raise ValueError(
                "family-private initialized target expert lineage mismatch"
            )
        initialized_exact_keys.append(target_route.module_key)
    family_keys = tuple(
        sorted({route.module_key for route in target.family_routes})
    )
    if not family_keys:
        raise ValueError("family-private target declares no physical family")
    if target.export_mode != "routed":
        raise ValueError("family-private migration requires the routed BC topology")
    return exact_key_map, family_keys, tuple(sorted(initialized_exact_keys))


def _migrate_model(
    target: nn.Module,
    *,
    source_state: Mapping[str, Tensor],
    exact_key_map: Mapping[str, str],
    family_keys: tuple[str, ...],
    initialized_exact_keys: tuple[str, ...],
    inert_output_tensor_names: Sequence[str],
    declaration: StatelessFamilyPrivateTransitionDeclaration,
) -> StatelessFamilyPrivateTransitionAudit:
    source = _tensor_state(source_state, label="source")
    if (
        canonical_model_state_fingerprint(source)
        != declaration.source_model_state_fingerprint
    ):
        raise ValueError("family-private source model state is not authorized")
    target_state = _tensor_state(target.state_dict(), label="target")
    migrated = dict(target_state)
    target_to_source: dict[str, str] = {}
    cloned_upper_names: set[str] = set()

    def assign(target_name: str, source_name: str, *, cloned_upper: bool) -> None:
        if target_name not in target_state:
            raise ValueError(
                f"family-private target tensor is missing: {target_name}"
            )
        if target_name in target_to_source:
            raise ValueError("family-private state mapping collided")
        source_value = source[source_name]
        target_value = target_state[target_name]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"family-private tensor shape changed: {source_name}->{target_name}"
            )
        if source_value.dtype != target_value.dtype:
            raise ValueError(
                f"family-private tensor dtype changed: {source_name}->{target_name}"
            )
        migrated[target_name] = source_value.detach().to(
            device=target_value.device
        ).clone()
        target_to_source[target_name] = source_name
        if cloned_upper:
            cloned_upper_names.add(target_name)

    for source_name in source:
        upper = _source_upper_tensor(source_name)
        if upper is None:
            assign(
                _rebind_exact_key(source_name, exact_key_map),
                source_name,
                cloned_upper=False,
            )
            continue
        upper_index, suffix = upper
        target_index = upper_index - FAMILY_PRIVATE_SHARED_LAYERS
        if not 0 <= target_index < FAMILY_PRIVATE_CLONED_LAYERS:
            raise ValueError("family-private source upper layer is out of range")
        assign(
            f"{_TARGET_GENERIC_PREFIX}{target_index}.{suffix}",
            source_name,
            cloned_upper=True,
        )
        for family_key in family_keys:
            assign(
                f"{_TARGET_FAMILY_PREFIX}{family_key}.cloned_layers."
                f"{target_index}.{suffix}",
                source_name,
                cloned_upper=True,
            )

    initialized_names = tuple(sorted(set(target_state) - set(target_to_source)))
    expected_appended_prefixes = tuple(
        f"{_TARGET_FAMILY_PREFIX}{family_key}.appended_layers."
        for family_key in family_keys
    )
    appended_names = tuple(
        name
        for name in initialized_names
        if name.startswith(expected_appended_prefixes)
    )
    exact_names = tuple(
        name
        for name in initialized_names
        if _target_exact_route_key(name) in initialized_exact_keys
    )
    if (
        not appended_names
        or len(appended_names) + len(exact_names) != len(initialized_names)
    ):
        raise ValueError(
            "family-private target-only state is not confined to appended blocks "
            "and declared exact additions"
        )
    for family_key in family_keys:
        prefix = f"{_TARGET_FAMILY_PREFIX}{family_key}.appended_layers."
        if not any(name.startswith(prefix) for name in appended_names):
            raise ValueError("family-private family has no appended block state")
    for exact_key in initialized_exact_keys:
        if not any(_target_exact_route_key(name) == exact_key for name in exact_names):
            raise ValueError("family-private exact addition has no private state")
    family_inert_names = tuple(inert_output_tensor_names)
    if family_inert_names != tuple(sorted(set(family_inert_names))) or not (
        family_inert_names
    ):
        raise ValueError(
            "family-private inert outputs must be nonempty, sorted, and unique"
        )
    if not set(family_inert_names).issubset(appended_names):
        raise ValueError("family-private inert outputs are not appended state")
    exact_inert_names = _initialized_exact_zero_tensor_names(
        exact_names,
        initialized_exact_keys=initialized_exact_keys,
    )
    inert_names = tuple(sorted((*family_inert_names, *exact_inert_names)))
    if not set(inert_names).issubset(initialized_names):
        raise ValueError("family-private inert outputs are not target-only state")
    _validate_finite_state(
        {name: migrated[name] for name in initialized_names},
        label="initialized appended",
    )
    for name in inert_names:
        if torch.count_nonzero(migrated[name]).item() != 0:
            raise ValueError(f"family-private appended output is not inert: {name}")
    target.load_state_dict(migrated, strict=True)
    loaded = target.state_dict()
    for target_name, source_name in target_to_source.items():
        if not torch.equal(
            loaded[target_name].detach().cpu(),
            source[source_name].detach().cpu(),
        ):
            raise RuntimeError(
                f"family-private migration failed exact copy: "
                f"{source_name}->{target_name}"
            )
    initialized = {name: loaded[name] for name in initialized_names}
    return StatelessFamilyPrivateTransitionAudit(
        declaration_fingerprint=family_private_transition_fingerprint(declaration),
        source_state_fingerprint=canonical_model_state_fingerprint(source),
        target_state_fingerprint=canonical_model_state_fingerprint(loaded),
        state_mapping_fingerprint=fingerprint_payload(
            {"target_to_source": sorted(target_to_source.items())}
        ),
        initialized_state_fingerprint=canonical_model_state_fingerprint(initialized),
        source_tensors=len(source),
        inherited_target_tensors=len(target_to_source),
        cloned_upper_target_tensors=len(cloned_upper_names),
        initialized_exact_target_tensors=len(exact_names),
        initialized_appended_target_tensors=len(appended_names),
        initialized_tensors=len(initialized_names),
        initialized_tensor_names=initialized_names,
        inert_output_tensor_names=inert_names,
    )


def _source_upper_tensor(name: str) -> tuple[int, str] | None:
    if not name.startswith(_SOURCE_UPPER_PREFIX):
        return None
    suffix = name.removeprefix(_SOURCE_UPPER_PREFIX)
    raw_index, separator, remainder = suffix.partition(".")
    if not separator or not raw_index.isdigit() or not remainder:
        raise ValueError("family-private source trunk tensor name is malformed")
    layer_index = int(raw_index)
    if layer_index < FAMILY_PRIVATE_SHARED_LAYERS:
        return None
    return layer_index, remainder


def _rebind_exact_key(name: str, key_map: Mapping[str, str]) -> str:
    segments = name.split(".")
    matches = [index for index, segment in enumerate(segments) if segment in key_map]
    if len(matches) > 1:
        raise ValueError("family-private state name contains multiple exact routes")
    if matches:
        index = matches[0]
        segments[index] = key_map[segments[index]]
    return ".".join(segments)


def _target_exact_route_key(name: str) -> str | None:
    """Return the exact module key for a route-private target tensor."""
    direct_prefixes = (
        "heads.policy_residuals.",
        "heads.option_residuals.",
        "heads.value_residuals.",
        "backbone.v2_adapters.prompts.",
    )
    for prefix in direct_prefixes:
        if name.startswith(prefix):
            suffix = name.removeprefix(prefix)
            key, separator, _remainder = suffix.partition(".")
            if not separator or not key:
                raise ValueError("family-private exact target tensor is malformed")
            return key
    stage_prefix = "backbone.v2_adapters.stages."
    capsule_separator = ".exact_capsules."
    if name.startswith(stage_prefix) and capsule_separator in name:
        suffix = name.split(capsule_separator, maxsplit=1)[1]
        key, separator, _remainder = suffix.partition(".")
        if not separator or not key:
            raise ValueError("family-private exact capsule tensor is malformed")
        return key
    return None


def _initialized_exact_zero_tensor_names(
    names: Sequence[str],
    *,
    initialized_exact_keys: Sequence[str],
) -> tuple[str, ...]:
    """Enumerate and validate neutral gates for every new exact leaf."""
    keys = set(initialized_exact_keys)
    zero_names: list[str] = []
    covered: set[str] = set()
    for name in names:
        key = _target_exact_route_key(name)
        if key not in keys:
            continue
        suffix = name.split(f".{key}.", maxsplit=1)[1]
        is_zero_gate = (
            name.startswith("backbone.v2_adapters.prompts.")
            or suffix in {
                "up.weight",
                "up.bias",
                "output.weight",
                "output.bias",
                "policy_coefficients",
                "value_coefficients",
                "policy_residual.up.weight",
                "policy_residual.up.bias",
                "value_residual.up.weight",
                "value_residual.up.bias",
            }
        )
        if is_zero_gate:
            zero_names.append(name)
            covered.add(key)
    if covered != keys:
        raise ValueError("family-private exact addition has no neutral output gates")
    return tuple(sorted(zero_names))


def _validate_source_pair(
    source: LoadedStatelessCheckpointPair,
    declaration: StatelessFamilyPrivateTransitionDeclaration,
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
            raise ValueError(f"family-private source {label} is not authorized")


def _tensor_state(
    state: Mapping[str, Tensor],
    *,
    label: str,
) -> dict[str, Tensor]:
    if not state:
        raise ValueError(f"family-private {label} state is empty")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError(
                f"family-private {label} state must contain named tensors"
            )
        result[name] = value
    return result


def _validate_finite_state(state: Mapping[str, Tensor], *, label: str) -> None:
    for name, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise ValueError(f"family-private {label} tensor is non-finite: {name}")


def _fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("family-private identities must be lowercase SHA-256")
    return normalized


__all__ = [
    "StatelessFamilyPrivateRouteInitialization",
    "StatelessFamilyPrivateTransitionAudit",
    "StatelessFamilyPrivateTransitionDeclaration",
    "StatelessFamilyPrivateTransitionResult",
    "family_private_transition_fingerprint",
    "migrate_family_private_from_pair",
    "migrate_family_private_model",
]
