"""Model-agnostic tensor and AdamW operations for a stateless BC overlay."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import Tensor, nn

from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class StatelessBcOverlayModelAudit(BaseModel):
    """Independent source-to-overlay tensor ownership proof."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_model_state_fingerprint: str
    target_model_state_fingerprint: str
    target_deck_digest: str
    target_expert_id: str
    trainable_parameter_names: tuple[str, ...]
    trainable_tensor_count: int = Field(ge=1)
    changed_parameter_names: tuple[str, ...]
    changed_tensor_count: int = Field(ge=1)
    frozen_tensor_count: int = Field(ge=1)

    @field_validator(
        "source_model_state_fingerprint",
        "target_model_state_fingerprint",
        "target_deck_digest",
        "target_expert_id",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require full state and route identities."""
        return _fingerprint(value)

    @model_validator(mode="after")
    def coherent_inventory(self) -> StatelessBcOverlayModelAudit:
        """Keep whitelist and observed movement deterministic."""
        if (
            tuple(sorted(set(self.trainable_parameter_names)))
            != self.trainable_parameter_names
            or len(self.trainable_parameter_names) != self.trainable_tensor_count
        ):
            raise ValueError("BC overlay trainable names must be sorted and unique")
        if (
            tuple(sorted(set(self.changed_parameter_names)))
            != self.changed_parameter_names
            or len(self.changed_parameter_names) != self.changed_tensor_count
        ):
            raise ValueError("BC overlay changed names must be sorted and unique")
        if not set(self.changed_parameter_names).issubset(
            self.trainable_parameter_names
        ):
            raise ValueError("BC overlay changed a non-trainable tensor")
        return self


class StatelessBcOverlayOptimizerAudit(BaseModel):
    """AdamW state retained by name except for the full BC whitelist."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reset_parameter_names: tuple[str, ...]
    reset_parameters: int = Field(ge=1)
    preserved_parameters_with_state: int = Field(ge=0)
    preserved_parameters_without_state: int = Field(ge=0)

    @model_validator(mode="after")
    def coherent_reset_inventory(self) -> StatelessBcOverlayOptimizerAudit:
        """Require exact deterministic reset ownership."""
        if (
            tuple(sorted(set(self.reset_parameter_names))) != self.reset_parameter_names
            or len(self.reset_parameter_names) != self.reset_parameters
        ):
            raise ValueError("BC optimizer reset names must be sorted and unique")
        return self


def audit_stateless_bc_overlay_state(
    source_state: Mapping[str, Tensor],
    target_state: Mapping[str, Tensor],
    *,
    trainable_parameter_names: tuple[str, ...],
    target_deck_digest: str,
    target_expert_id: str,
) -> StatelessBcOverlayModelAudit:
    """Prove bitwise equality outside one sorted parameter whitelist."""
    source = _tensor_state(source_state, label="source")
    target = _tensor_state(target_state, label="target")
    trainable = tuple(trainable_parameter_names)
    if tuple(sorted(set(trainable))) != trainable or not trainable:
        raise ValueError("BC overlay whitelist must be nonempty, sorted, and unique")
    if set(source) != set(target):
        raise ValueError("BC overlay changed the model state inventory")
    if not set(trainable).issubset(source):
        raise ValueError("BC overlay whitelist contains an unknown tensor")
    changed: list[str] = []
    for name in sorted(source):
        before = source[name]
        after = target[name]
        if before.shape != after.shape or before.dtype != after.dtype:
            raise ValueError(f"BC overlay tensor contract changed: {name}")
        if (after.is_floating_point() or after.is_complex()) and not torch.isfinite(
            after
        ).all():
            raise ValueError(f"BC overlay tensor is non-finite: {name}")
        if not torch.equal(before, after):
            changed.append(name)
    changed_names = tuple(changed)
    if not changed_names:
        raise ValueError("BC overlay did not change any trainable tensor")
    if not set(changed_names).issubset(trainable):
        raise ValueError("BC overlay changed a frozen tensor")
    return StatelessBcOverlayModelAudit(
        source_model_state_fingerprint=canonical_model_state_fingerprint(source),
        target_model_state_fingerprint=canonical_model_state_fingerprint(target),
        target_deck_digest=target_deck_digest,
        target_expert_id=target_expert_id,
        trainable_parameter_names=trainable,
        trainable_tensor_count=len(trainable),
        changed_parameter_names=changed_names,
        changed_tensor_count=len(changed_names),
        frozen_tensor_count=len(source) - len(trainable),
    )


def transplant_stateless_bc_overlay_optimizer(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: nn.Module,
    source_optimizer_state: Mapping[str, Any],
    reset_parameter_names: tuple[str, ...],
) -> StatelessBcOverlayOptimizerAudit:
    """Retain same-topology AdamW state and clear every BC-owned parameter."""
    reset_names = tuple(reset_parameter_names)
    if tuple(sorted(set(reset_names))) != reset_names or not reset_names:
        raise ValueError("BC optimizer reset names must be nonempty and sorted")
    named_parameters = dict(target_model.named_parameters())
    if not set(reset_names).issubset(named_parameters):
        raise ValueError("BC optimizer reset references an unknown parameter")
    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    if (
        not isinstance(source_groups, list)
        or len(source_groups) != 1
        or not isinstance(source_states, Mapping)
    ):
        raise ValueError("BC overlay source optimizer must have one valid group")
    source_group = source_groups[0]
    source_ids = source_group.get("params")
    parameter_names = tuple(named_parameters)
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != len(parameter_names)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError("BC overlay source optimizer inventory is malformed")
    if set(source_states) - set(source_ids):
        raise ValueError("BC overlay source optimizer has unreferenced state")

    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    target_states = target_snapshot.get("state")
    if (
        not isinstance(target_groups, list)
        or len(target_groups) != 1
        or not isinstance(target_states, Mapping)
        or target_states
    ):
        raise ValueError("BC overlay target optimizer must be fresh and single-group")
    target_names = _optimizer_parameter_names(target_model, optimizer)
    if target_names != (parameter_names,):
        raise ValueError(
            "BC overlay optimizer must cover every model parameter in order"
        )
    target_ids = target_groups[0].get("params")
    if not isinstance(target_ids, list) or len(target_ids) != len(parameter_names):
        raise RuntimeError("BC overlay target optimizer inventory changed")

    transplanted: dict[int, Any] = {}
    preserved_with_state = 0
    preserved_without_state = 0
    reset = set(reset_names)
    for source_id, target_id, name in zip(
        source_ids,
        target_ids,
        parameter_names,
        strict=True,
    ):
        raw_state = source_states.get(source_id, {})
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"BC optimizer state is malformed: {name}")
        _validate_adam_state(raw_state, named_parameters[name], name=name)
        if name in reset:
            continue
        if raw_state:
            transplanted[int(target_id)] = dict(raw_state)
            preserved_with_state += 1
        else:
            preserved_without_state += 1
    target_group = dict(source_group)
    target_group["params"] = target_ids
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": [target_group],
        }
    )
    if any(optimizer.state.get(named_parameters[name]) for name in reset_names):
        raise RuntimeError("BC optimizer reset parameter retained Adam state")
    return StatelessBcOverlayOptimizerAudit(
        reset_parameter_names=reset_names,
        reset_parameters=len(reset_names),
        preserved_parameters_with_state=preserved_with_state,
        preserved_parameters_without_state=preserved_without_state,
    )


def _optimizer_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[str, ...], ...]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    groups: list[tuple[str, ...]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            parameter_id = id(parameter)
            if parameter_id in seen or parameter_id not in names_by_id:
                raise ValueError("BC overlay optimizer parameter is ambiguous")
            seen.add(parameter_id)
            names.append(names_by_id[parameter_id])
        groups.append(tuple(names))
    return tuple(groups)


def _validate_adam_state(
    state: Mapping[str, Any],
    parameter: nn.Parameter,
    *,
    name: str,
) -> None:
    if not state:
        return
    required = {"step", "exp_avg", "exp_avg_sq"}
    if required - set(state):
        raise ValueError(f"BC optimizer state is incomplete: {name}")
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is not None and (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.shape != parameter.shape
        ):
            raise ValueError(f"BC optimizer moment is incompatible: {name}:{key}")
    step = state["step"]
    if isinstance(step, Tensor) and step.numel() != 1:
        raise ValueError(f"BC optimizer step is malformed: {name}")


def _tensor_state(
    state: Mapping[str, Tensor],
    *,
    label: str,
) -> dict[str, Tensor]:
    if not state:
        raise ValueError(f"BC overlay {label} state is empty")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError(f"BC overlay {label} state must contain named tensors")
        result[name] = value
    return result


def _fingerprint(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256_PATTERN.fullmatch(normalized) is None:
        raise ValueError("BC overlay identity must be lowercase SHA-256")
    return normalized


__all__ = [
    "StatelessBcOverlayModelAudit",
    "StatelessBcOverlayOptimizerAudit",
    "audit_stateless_bc_overlay_state",
    "transplant_stateless_bc_overlay_optimizer",
]
