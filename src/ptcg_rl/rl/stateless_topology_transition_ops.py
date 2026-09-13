"""Model-agnostic state operations for stateless topology transitions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint


@dataclass(frozen=True)
class ModelMigrationEvidence:
    """Raw evidence from one exact state-dict superset migration."""

    source_state_fingerprint: str
    target_state_fingerprint: str
    state_key_mapping_fingerprint: str
    initialized_state_fingerprint: str
    copied_tensors: int
    remapped_tensors: int
    initialized_tensor_names: tuple[str, ...]
    inert_output_tensor_names: tuple[str, ...]


@dataclass(frozen=True)
class OptimizerMigrationEvidence:
    """Raw evidence from one stable-name AdamW state transplant."""

    parameter_mapping_fingerprint: str
    copied_parameters: int
    remapped_parameters: int
    preserved_without_state: int
    reset_parameters: int


def migrate_topology_model(
    *,
    target: nn.Module,
    source_state: Mapping[str, Tensor],
    module_key_map: Mapping[str, str],
    inert_output_tensor_names: Sequence[str],
) -> ModelMigrationEvidence:
    """Copy all source tensors and preserve only audited inert new state."""
    source_tensors = _tensor_state(source_state, label="source")
    target_state = _tensor_state(target.state_dict(), label="target")
    key_mapping = _state_key_mapping(
        source_tensors,
        target_state,
        module_key_map=module_key_map,
    )
    initialized_names = tuple(sorted(set(target_state) - set(key_mapping.values())))
    if not initialized_names:
        raise ValueError("stateless topology target has no new v2 state")
    inert_names = tuple(inert_output_tensor_names)
    if inert_names != tuple(sorted(set(inert_names))) or not inert_names:
        raise ValueError("inert output tensor names must be nonempty, sorted, and unique")
    if not set(inert_names).issubset(initialized_names):
        raise ValueError("inert output tensors must be target-only v2 state")

    migrated = dict(target_state)
    for source_name, target_name in key_mapping.items():
        source_value = source_tensors[source_name]
        target_value = target_state[target_name]
        if source_value.shape != target_value.shape:
            raise ValueError(
                f"stateless topology tensor shape changed: {source_name}->{target_name}"
            )
        if source_value.dtype != target_value.dtype:
            raise ValueError(
                f"stateless topology tensor dtype changed: {source_name}->{target_name}"
            )
        migrated[target_name] = source_value.detach().to(
            device=target_value.device
        ).clone()
    _validate_finite_state(
        {name: migrated[name] for name in initialized_names},
        label="initialized v2",
    )
    for name in inert_names:
        if torch.count_nonzero(migrated[name]).item() != 0:
            raise ValueError(f"new v2 output is not inert: {name}")
    target.load_state_dict(migrated, strict=True)
    loaded_state = target.state_dict()
    for source_name, target_name in key_mapping.items():
        if not torch.equal(
            source_tensors[source_name].detach().cpu(),
            loaded_state[target_name].detach().cpu(),
        ):
            raise RuntimeError(
                f"stateless topology failed exact copy: {source_name}->{target_name}"
            )

    initialized = {name: loaded_state[name] for name in initialized_names}
    return ModelMigrationEvidence(
        source_state_fingerprint=canonical_model_state_fingerprint(source_tensors),
        target_state_fingerprint=canonical_model_state_fingerprint(loaded_state),
        state_key_mapping_fingerprint=fingerprint_payload(
            {"state_key_mapping": sorted(key_mapping.items())}
        ),
        initialized_state_fingerprint=canonical_model_state_fingerprint(initialized),
        copied_tensors=len(key_mapping),
        remapped_tensors=sum(
            source != target for source, target in key_mapping.items()
        ),
        initialized_tensor_names=initialized_names,
        inert_output_tensor_names=inert_names,
    )


def transplant_topology_optimizer(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: nn.Module,
    source_parameter_names: Sequence[Sequence[str]],
    source_optimizer_state: Mapping[str, Any],
    module_key_map: Mapping[str, str],
) -> OptimizerMigrationEvidence:
    """Transplant AdamW moments by stable/remapped names and reset new state."""
    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    if not isinstance(source_groups, list) or not isinstance(source_states, Mapping):
        raise ValueError("stateless topology source optimizer state is malformed")
    if len(source_groups) != len(source_parameter_names):
        raise ValueError("stateless topology source optimizer groups changed")

    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    target_states = target_snapshot.get("state")
    if not isinstance(target_groups, list) or not isinstance(target_states, Mapping):
        raise ValueError("stateless topology target optimizer state is malformed")
    if target_states:
        raise ValueError("stateless topology target optimizer must be fresh")
    target_names = _optimizer_parameter_names(target_model, optimizer)
    if len(target_groups) != len(target_names):
        raise RuntimeError("stateless topology target optimizer groups changed")
    if len(source_groups) != len(target_groups):
        raise ValueError("stateless topology optimizer group topology changed")
    target_parameters = dict(target_model.named_parameters())

    source_by_target: dict[str, Mapping[str, Any]] = {}
    source_name_by_target: dict[str, str] = {}
    referenced_source_ids: set[object] = set()
    for group, names in zip(source_groups, source_parameter_names, strict=True):
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
            raise ValueError("stateless topology source optimizer inventory is malformed")
        for parameter_id, source_name in zip(parameter_ids, names, strict=True):
            target_name = rebind_name(source_name, module_key_map)
            if target_name in source_by_target:
                raise ValueError("stateless topology optimizer name mapping collided")
            if target_name not in target_parameters:
                raise ValueError(
                    f"stateless topology target parameter is missing: {target_name}"
                )
            parameter_state = source_states.get(parameter_id, {})
            if not isinstance(parameter_state, Mapping):
                raise ValueError(
                    f"stateless topology optimizer state is malformed: {source_name}"
                )
            source_by_target[target_name] = cast(Mapping[str, Any], parameter_state)
            source_name_by_target[target_name] = source_name
            referenced_source_ids.add(parameter_id)
    if set(source_states) - referenced_source_ids:
        raise ValueError("stateless topology optimizer has unreferenced state")

    flat_target_names = tuple(name for group in target_names for name in group)
    if set(flat_target_names) != set(target_parameters):
        raise ValueError("stateless topology optimizer must cover every model parameter")
    reset_names = set(flat_target_names) - set(source_by_target)
    if not reset_names:
        raise ValueError("stateless topology optimizer has no new v2 parameters")
    transplanted: dict[int, Any] = {}
    copied = 0
    preserved_without_state = 0
    remapped = 0
    for group, names in zip(target_groups, target_names, strict=True):
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
            raise RuntimeError("stateless topology target optimizer inventory changed")
        for parameter_id, target_name in zip(parameter_ids, names, strict=True):
            if target_name in reset_names:
                continue
            state = source_by_target[target_name]
            _validate_adam_state(
                state,
                target_parameters[target_name],
                name=target_name,
            )
            if state:
                transplanted[int(parameter_id)] = {
                    key: value.detach().clone() if isinstance(value, Tensor) else value
                    for key, value in state.items()
                }
                copied += 1
            else:
                preserved_without_state += 1
            if source_name_by_target[target_name] != target_name:
                remapped += 1
    transplanted_groups: list[dict[str, Any]] = []
    for source_group, target_group in zip(
        source_groups,
        target_groups,
        strict=True,
    ):
        target_ids = target_group.get("params")
        if not isinstance(target_ids, list):
            raise RuntimeError("stateless topology target optimizer group is malformed")
        transplanted_group = dict(source_group)
        transplanted_group["params"] = target_ids
        transplanted_groups.append(transplanted_group)
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": transplanted_groups,
        }
    )
    return OptimizerMigrationEvidence(
        parameter_mapping_fingerprint=fingerprint_payload(
            {
                "parameter_mapping": sorted(
                    (source, target)
                    for target, source in source_name_by_target.items()
                )
            }
        ),
        copied_parameters=copied,
        remapped_parameters=remapped,
        preserved_without_state=preserved_without_state,
        reset_parameters=len(reset_names),
    )


def rebind_name(name: str, module_key_map: Mapping[str, str]) -> str:
    """Replace at most one exact module-key path segment."""
    segments = name.split(".")
    indices = [
        index for index, segment in enumerate(segments) if segment in module_key_map
    ]
    if len(indices) > 1:
        raise ValueError(f"stateless topology state key has multiple routes: {name}")
    if indices:
        index = indices[0]
        segments[index] = module_key_map[segments[index]]
    return ".".join(segments)


def _state_key_mapping(
    source: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    *,
    module_key_map: Mapping[str, str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    used_targets: set[str] = set()
    for source_name in source:
        target_name = rebind_name(source_name, module_key_map)
        if target_name not in target:
            raise ValueError(
                f"stateless topology did not retain v1 tensor: {source_name}"
            )
        if target_name in used_targets:
            raise ValueError("stateless topology state key mapping collided")
        mapping[source_name] = target_name
        used_targets.add(target_name)
    return mapping


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
                raise ValueError("stateless topology optimizer parameter is ambiguous")
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
        raise ValueError(f"stateless topology optimizer state is incomplete: {name}")
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is not None and (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.shape != parameter.shape
        ):
            raise ValueError(
                f"stateless topology optimizer moment is incompatible: {name}:{key}"
            )
    step = state["step"]
    if isinstance(step, Tensor) and step.numel() != 1:
        raise ValueError(f"stateless topology optimizer step is malformed: {name}")


def _tensor_state(
    state: Mapping[str, Tensor],
    *,
    label: str,
) -> dict[str, Tensor]:
    if not state:
        raise ValueError(f"stateless topology {label} state is empty")
    result: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(name, str) or not isinstance(value, Tensor):
            raise ValueError(
                f"stateless topology {label} state must contain named tensors"
            )
        result[name] = value
    return result


def _validate_finite_state(state: Mapping[str, Tensor], *, label: str) -> None:
    for name, value in state.items():
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(
            value
        ).all():
            raise ValueError(f"stateless topology {label} tensor is non-finite: {name}")


__all__ = [
    "ModelMigrationEvidence",
    "OptimizerMigrationEvidence",
    "migrate_topology_model",
    "rebind_name",
    "transplant_topology_optimizer",
]
