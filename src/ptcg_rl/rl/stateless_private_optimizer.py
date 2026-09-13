"""Fail-closed private-bank ownership and AdamW scope projection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn

from ptcg_rl.model.simple_stateless import SimpleStatelessPolicyValueNet

STATELESS_OPTIMIZER_GROUP_ROLE_KEY = "stateless_scope_role"
STATELESS_PRIVATE_GROUP_ROLE = "private"
STATELESS_SHARED_GROUP_ROLE = "shared"


@dataclass(frozen=True)
class StatelessTrainableParameterScope:
    """Resolved model-order parameter ownership for one stateless learner."""

    mode: Literal["full_model", "private_only", "hybrid"]
    parameter_names: tuple[str, ...]
    parameters: tuple[nn.Parameter, ...]
    parameter_elements: int
    frozen_parameter_names: tuple[str, ...]
    frozen_parameter_elements: int
    private_parameter_names: tuple[str, ...]
    private_parameters: tuple[nn.Parameter, ...]
    private_parameter_elements: int
    shared_parameter_names: tuple[str, ...]
    shared_parameters: tuple[nn.Parameter, ...]
    shared_parameter_elements: int
    route_bank_names: tuple[str, ...]
    family_bank_names: tuple[str, ...]

    @property
    def parameter_name_fingerprint(self) -> str:
        """Identify the ordered trainable tensor inventory."""
        payload = json.dumps(
            self.parameter_names,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(
            b"ptcg-rl/stateless-trainable-parameter-names/v1\x00" + payload
        ).hexdigest()


class StatelessPrivateOptimizerTransitionAudit(BaseModel):
    """Serializable evidence for one audited private-optimizer transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["stateless-private-optimizer-transition-v1"] = (
        "stateless-private-optimizer-transition-v1"
    )
    source_scope: Literal["full_model", "private_only"] = "full_model"
    target_scope: Literal["private_only"] = "private_only"
    target_private_learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    parameter_name_fingerprint: str
    source_parameter_tensors: int = Field(gt=0)
    source_parameter_elements: int = Field(gt=0)
    private_parameter_tensors: int = Field(gt=0)
    private_parameter_elements: int = Field(gt=0)
    frozen_parameter_tensors: int = Field(gt=0)
    frozen_parameter_elements: int = Field(gt=0)
    copied_private_states: int = Field(ge=0)
    private_parameters_without_state: int = Field(ge=0)
    discarded_shared_states: int = Field(ge=0)
    route_bank_names: tuple[str, ...]
    family_bank_names: tuple[str, ...]


class StatelessHybridOptimizerTransitionAudit(BaseModel):
    """Serializable evidence for one private-to-hybrid AdamW transition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal["stateless-hybrid-optimizer-transition-v1"] = (
        "stateless-hybrid-optimizer-transition-v1"
    )
    source_scope: Literal["private_only"] = "private_only"
    target_scope: Literal["hybrid"] = "hybrid"
    target_private_learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    target_shared_learning_rate: float = Field(gt=0.0, allow_inf_nan=False)
    parameter_name_fingerprint: str
    private_parameter_tensors: int = Field(gt=0)
    private_parameter_elements: int = Field(gt=0)
    shared_parameter_tensors: int = Field(gt=0)
    shared_parameter_elements: int = Field(gt=0)
    copied_private_states: int = Field(ge=0)
    private_parameters_without_state: int = Field(ge=0)
    fresh_shared_parameters: int = Field(gt=0)
    route_bank_names: tuple[str, ...]
    family_bank_names: tuple[str, ...]


def configure_stateless_trainable_scope(
    model: SimpleStatelessPolicyValueNet,
    mode: Literal["full_model", "private_only", "hybrid"],
) -> StatelessTrainableParameterScope:
    """Freeze the model, then enable exactly the requested physical banks."""
    named_parameters = tuple(model.named_parameters())
    if not named_parameters:
        raise RuntimeError("stateless learner model has no parameters")
    route_banks = tuple(model.route_private_banks())
    family_banks = tuple(model.family_private_banks())
    route_bank_names = _unique_bank_names(route_banks, label="route-private")
    family_bank_names = _unique_bank_names(family_banks, label="family-private")
    if set(route_bank_names) & set(family_bank_names):
        raise RuntimeError("stateless private bank names overlap")

    private_ids = _private_parameter_ids(
        route_banks=route_banks,
        family_banks=family_banks,
    )
    model_parameter_ids = {id(parameter) for _, parameter in named_parameters}
    if private_ids - model_parameter_ids:
        raise RuntimeError("private bank contains a parameter outside the model")
    private = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if id(parameter) in private_ids
    )
    shared = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if id(parameter) not in private_ids
    )
    if len(private) != len(private_ids):
        raise RuntimeError("stateless private parameter ownership is ambiguous")

    if mode == "full_model":
        selected_ids = {id(parameter) for _, parameter in named_parameters}
    elif mode == "private_only":
        selected_ids = private_ids
        if not route_banks or not selected_ids:
            raise RuntimeError(
                "private-only stateless scope requires physical route-private banks"
            )
    elif mode == "hybrid":
        selected_ids = model_parameter_ids
        if not route_banks or not private or not shared:
            raise RuntimeError(
                "hybrid stateless scope requires private banks and shared parameters"
            )
    else:
        raise ValueError(f"unsupported stateless optimizer scope: {mode}")

    selected = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if id(parameter) in selected_ids
    )
    if len(selected) != len(selected_ids):
        raise RuntimeError("stateless private parameter ownership is ambiguous")
    selected_names = tuple(name for name, _ in selected)
    selected_name_set = set(selected_names)
    for name, parameter in named_parameters:
        parameter.requires_grad_(name in selected_name_set)
    enabled_names = tuple(
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    )
    if enabled_names != selected_names:
        raise RuntimeError("stateless trainable parameters differ from their scope")

    frozen = tuple(
        (name, parameter)
        for name, parameter in named_parameters
        if name not in selected_name_set
    )
    return StatelessTrainableParameterScope(
        mode=mode,
        parameter_names=selected_names,
        parameters=tuple(parameter for _, parameter in selected),
        parameter_elements=sum(parameter.numel() for _, parameter in selected),
        frozen_parameter_names=tuple(name for name, _ in frozen),
        frozen_parameter_elements=sum(parameter.numel() for _, parameter in frozen),
        private_parameter_names=tuple(name for name, _ in private),
        private_parameters=tuple(parameter for _, parameter in private),
        private_parameter_elements=sum(parameter.numel() for _, parameter in private),
        shared_parameter_names=tuple(name for name, _ in shared),
        shared_parameters=tuple(parameter for _, parameter in shared),
        shared_parameter_elements=sum(parameter.numel() for _, parameter in shared),
        route_bank_names=route_bank_names,
        family_bank_names=family_bank_names,
    )


def build_stateless_hybrid_optimizer_groups(
    scope: StatelessTrainableParameterScope,
    *,
    private_learning_rate: float,
    shared_learning_rate: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the only supported private/shared AdamW group ordering."""
    if scope.mode != "hybrid":
        raise ValueError("hybrid optimizer groups require a hybrid scope")
    _require_positive_learning_rate(private_learning_rate, label="private")
    _require_positive_learning_rate(shared_learning_rate, label="shared")
    if not scope.private_parameters or not scope.shared_parameters:
        raise RuntimeError("hybrid optimizer parameter groups cannot be empty")
    return (
        {
            "params": scope.private_parameters,
            "lr": private_learning_rate,
            STATELESS_OPTIMIZER_GROUP_ROLE_KEY: STATELESS_PRIVATE_GROUP_ROLE,
        },
        {
            "params": scope.shared_parameters,
            "lr": shared_learning_rate,
            STATELESS_OPTIMIZER_GROUP_ROLE_KEY: STATELESS_SHARED_GROUP_ROLE,
        },
    )


def transplant_full_optimizer_to_private(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: SimpleStatelessPolicyValueNet,
    source_optimizer_state: Mapping[str, Any],
    scope: StatelessTrainableParameterScope,
    private_learning_rate: float,
) -> StatelessPrivateOptimizerTransitionAudit:
    """Project a same-topology full-model AdamW state onto private parameters."""
    if scope.mode != "private_only":
        raise ValueError("optimizer projection target must be private-only")
    if not math.isfinite(private_learning_rate) or private_learning_rate <= 0.0:
        raise ValueError("private optimizer learning rate must be finite and positive")
    all_named_parameters = tuple(target_model.named_parameters())
    all_names = tuple(name for name, _ in all_named_parameters)
    all_parameters = dict(all_named_parameters)
    if not all_names or not scope.parameter_names:
        raise RuntimeError("optimizer projection parameter inventory is empty")

    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    target_states = target_snapshot.get("state")
    if (
        not isinstance(source_groups, list)
        or len(source_groups) != 1
        or not isinstance(source_states, Mapping)
        or not isinstance(target_groups, list)
        or len(target_groups) != 1
        or not isinstance(target_states, Mapping)
        or target_states
    ):
        raise ValueError(
            "private optimizer transition requires one full source group and "
            "one fresh target group"
        )
    source_ids = source_groups[0].get("params")
    target_ids = target_groups[0].get("params")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != len(all_names)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError(
            "private optimizer transition source does not cover the full model"
        )
    if (
        not isinstance(target_ids, list)
        or len(target_ids) != len(scope.parameter_names)
        or len(set(target_ids)) != len(target_ids)
    ):
        raise ValueError("private optimizer transition target inventory is malformed")
    if set(source_states) - set(source_ids):
        raise ValueError("private optimizer source has unreferenced parameter state")
    if _optimizer_parameter_names(target_model, optimizer) != (scope.parameter_names,):
        raise ValueError(
            "private optimizer parameters differ from the validated private scope"
        )

    source_state_by_name: dict[str, Mapping[str, Any]] = {}
    discarded_shared_states = 0
    private_names = set(scope.parameter_names)
    for source_id, name in zip(source_ids, all_names, strict=True):
        raw_state = source_states.get(source_id, {})
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"private optimizer state is malformed: {name}")
        _validate_adam_state(raw_state, all_parameters[name], name=name)
        source_state_by_name[name] = cast(Mapping[str, Any], raw_state)
        if name not in private_names and raw_state:
            discarded_shared_states += 1

    transplanted: dict[int, dict[str, Any]] = {}
    copied_private_states = 0
    private_parameters_without_state = 0
    for target_id, name in zip(
        target_ids,
        scope.parameter_names,
        strict=True,
    ):
        raw_state = source_state_by_name[name]
        if raw_state:
            transplanted[int(target_id)] = {
                key: value.detach().clone() if isinstance(value, Tensor) else value
                for key, value in raw_state.items()
            }
            copied_private_states += 1
        else:
            private_parameters_without_state += 1

    target_group = dict(target_groups[0])
    target_group["params"] = target_ids
    target_group["lr"] = private_learning_rate
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": [target_group],
        }
    )
    if any(
        float(group["lr"]) != private_learning_rate for group in optimizer.param_groups
    ):
        raise RuntimeError("private optimizer learning rate changed during projection")
    if any(
        parameter in optimizer.state
        for name, parameter in all_named_parameters
        if name not in private_names
    ):
        raise RuntimeError("projected optimizer retained shared parameter state")

    return StatelessPrivateOptimizerTransitionAudit(
        source_scope="full_model",
        target_private_learning_rate=private_learning_rate,
        parameter_name_fingerprint=scope.parameter_name_fingerprint,
        source_parameter_tensors=len(all_named_parameters),
        source_parameter_elements=sum(
            parameter.numel() for _, parameter in all_named_parameters
        ),
        private_parameter_tensors=len(scope.parameter_names),
        private_parameter_elements=scope.parameter_elements,
        frozen_parameter_tensors=len(scope.frozen_parameter_names),
        frozen_parameter_elements=scope.frozen_parameter_elements,
        copied_private_states=copied_private_states,
        private_parameters_without_state=private_parameters_without_state,
        discarded_shared_states=discarded_shared_states,
        route_bank_names=scope.route_bank_names,
        family_bank_names=scope.family_bank_names,
    )


def retarget_private_optimizer_learning_rate(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: SimpleStatelessPolicyValueNet,
    source_optimizer_state: Mapping[str, Any],
    scope: StatelessTrainableParameterScope,
    private_learning_rate: float,
) -> StatelessPrivateOptimizerTransitionAudit:
    """Preserve a same-topology private AdamW state while changing its LR."""
    if scope.mode != "private_only":
        raise ValueError("optimizer LR retarget target must be private-only")
    if not math.isfinite(private_learning_rate) or private_learning_rate <= 0.0:
        raise ValueError("private optimizer learning rate must be finite and positive")
    all_named_parameters = tuple(target_model.named_parameters())
    all_parameters = dict(all_named_parameters)
    if not all_named_parameters or not scope.parameter_names:
        raise RuntimeError("optimizer LR retarget parameter inventory is empty")

    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    target_states = target_snapshot.get("state")
    if (
        not isinstance(source_groups, list)
        or len(source_groups) != 1
        or not isinstance(source_states, Mapping)
        or not isinstance(target_groups, list)
        or len(target_groups) != 1
        or not isinstance(target_states, Mapping)
        or target_states
    ):
        raise ValueError(
            "private optimizer LR retarget requires one private source group "
            "and one fresh target group"
        )
    source_ids = source_groups[0].get("params")
    target_ids = target_groups[0].get("params")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != len(scope.parameter_names)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError(
            "private optimizer LR retarget source does not cover the private scope"
        )
    if (
        not isinstance(target_ids, list)
        or len(target_ids) != len(scope.parameter_names)
        or len(set(target_ids)) != len(target_ids)
    ):
        raise ValueError("private optimizer LR retarget target inventory is malformed")
    if set(source_states) - set(source_ids):
        raise ValueError("private optimizer source has unreferenced parameter state")
    if _optimizer_parameter_names(target_model, optimizer) != (scope.parameter_names,):
        raise ValueError(
            "private optimizer parameters differ from the validated private scope"
        )

    transplanted: dict[int, dict[str, Any]] = {}
    copied_private_states = 0
    private_parameters_without_state = 0
    for source_id, target_id, name in zip(
        source_ids,
        target_ids,
        scope.parameter_names,
        strict=True,
    ):
        raw_state = source_states.get(source_id, {})
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"private optimizer state is malformed: {name}")
        _validate_adam_state(raw_state, all_parameters[name], name=name)
        if raw_state:
            transplanted[int(target_id)] = {
                key: value.detach().clone() if isinstance(value, Tensor) else value
                for key, value in raw_state.items()
            }
            copied_private_states += 1
        else:
            private_parameters_without_state += 1

    target_group = dict(source_groups[0])
    target_group["params"] = target_ids
    target_group["lr"] = private_learning_rate
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": [target_group],
        }
    )
    if any(
        float(group["lr"]) != private_learning_rate for group in optimizer.param_groups
    ):
        raise RuntimeError("private optimizer learning rate changed during retarget")

    return StatelessPrivateOptimizerTransitionAudit(
        source_scope="private_only",
        target_private_learning_rate=private_learning_rate,
        parameter_name_fingerprint=scope.parameter_name_fingerprint,
        source_parameter_tensors=len(scope.parameter_names),
        source_parameter_elements=scope.parameter_elements,
        private_parameter_tensors=len(scope.parameter_names),
        private_parameter_elements=scope.parameter_elements,
        frozen_parameter_tensors=len(scope.frozen_parameter_names),
        frozen_parameter_elements=scope.frozen_parameter_elements,
        copied_private_states=copied_private_states,
        private_parameters_without_state=private_parameters_without_state,
        discarded_shared_states=0,
        route_bank_names=scope.route_bank_names,
        family_bank_names=scope.family_bank_names,
    )


def transplant_private_optimizer_to_hybrid(
    *,
    optimizer: torch.optim.Optimizer,
    target_model: SimpleStatelessPolicyValueNet,
    source_optimizer_state: Mapping[str, Any],
    scope: StatelessTrainableParameterScope,
    private_learning_rate: float,
    shared_learning_rate: float,
) -> StatelessHybridOptimizerTransitionAudit:
    """Preserve private AdamW moments and introduce shared parameters fresh."""
    if scope.mode != "hybrid":
        raise ValueError("optimizer expansion target must be hybrid")
    _require_positive_learning_rate(private_learning_rate, label="private")
    _require_positive_learning_rate(shared_learning_rate, label="shared")
    if not scope.private_parameter_names or not scope.shared_parameter_names:
        raise RuntimeError("hybrid optimizer parameter inventory is empty")

    all_parameters = dict(target_model.named_parameters())
    source_groups = source_optimizer_state.get("param_groups")
    source_states = source_optimizer_state.get("state")
    target_snapshot = optimizer.state_dict()
    target_groups = target_snapshot.get("param_groups")
    target_states = target_snapshot.get("state")
    if (
        not isinstance(source_groups, list)
        or len(source_groups) != 1
        or not isinstance(source_states, Mapping)
        or not isinstance(target_groups, list)
        or len(target_groups) != 2
        or not isinstance(target_states, Mapping)
        or target_states
    ):
        raise ValueError(
            "hybrid optimizer transition requires one private source group and "
            "two fresh target groups"
        )
    source_ids = source_groups[0].get("params")
    private_target_ids = target_groups[0].get("params")
    shared_target_ids = target_groups[1].get("params")
    if (
        not isinstance(source_ids, list)
        or len(source_ids) != len(scope.private_parameter_names)
        or len(set(source_ids)) != len(source_ids)
    ):
        raise ValueError(
            "hybrid optimizer transition source does not cover the private scope"
        )
    if (
        not isinstance(private_target_ids, list)
        or len(private_target_ids) != len(scope.private_parameter_names)
        or len(set(private_target_ids)) != len(private_target_ids)
        or not isinstance(shared_target_ids, list)
        or len(shared_target_ids) != len(scope.shared_parameter_names)
        or len(set(shared_target_ids)) != len(shared_target_ids)
        or set(private_target_ids) & set(shared_target_ids)
    ):
        raise ValueError("hybrid optimizer transition target inventory is malformed")
    if set(source_states) - set(source_ids):
        raise ValueError("hybrid optimizer source has unreferenced parameter state")
    expected_target_names = (
        scope.private_parameter_names,
        scope.shared_parameter_names,
    )
    if _optimizer_parameter_names(target_model, optimizer) != expected_target_names:
        raise ValueError(
            "hybrid optimizer parameters differ from the validated scope partitions"
        )
    _require_hybrid_group_roles(target_groups)

    transplanted: dict[int, dict[str, Any]] = {}
    copied_private_states = 0
    private_parameters_without_state = 0
    for source_id, target_id, name in zip(
        source_ids,
        private_target_ids,
        scope.private_parameter_names,
        strict=True,
    ):
        raw_state = source_states.get(source_id, {})
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"hybrid private optimizer state is malformed: {name}")
        _validate_adam_state(raw_state, all_parameters[name], name=name)
        if raw_state:
            transplanted[int(target_id)] = {
                key: value.detach().clone() if isinstance(value, Tensor) else value
                for key, value in raw_state.items()
            }
            copied_private_states += 1
        else:
            private_parameters_without_state += 1

    loaded_groups = [dict(group) for group in target_groups]
    loaded_groups[0]["params"] = private_target_ids
    loaded_groups[0]["lr"] = private_learning_rate
    loaded_groups[1]["params"] = shared_target_ids
    loaded_groups[1]["lr"] = shared_learning_rate
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": loaded_groups,
        }
    )
    _require_hybrid_group_roles(optimizer.param_groups)
    if float(optimizer.param_groups[0]["lr"]) != private_learning_rate:
        raise RuntimeError("hybrid private learning rate changed during transition")
    if float(optimizer.param_groups[1]["lr"]) != shared_learning_rate:
        raise RuntimeError("hybrid shared learning rate changed during transition")
    shared_parameters = set(scope.shared_parameters)
    if any(parameter in optimizer.state for parameter in shared_parameters):
        raise RuntimeError("hybrid optimizer initialized shared Adam state eagerly")

    return StatelessHybridOptimizerTransitionAudit(
        target_private_learning_rate=private_learning_rate,
        target_shared_learning_rate=shared_learning_rate,
        parameter_name_fingerprint=scope.parameter_name_fingerprint,
        private_parameter_tensors=len(scope.private_parameter_names),
        private_parameter_elements=scope.private_parameter_elements,
        shared_parameter_tensors=len(scope.shared_parameter_names),
        shared_parameter_elements=scope.shared_parameter_elements,
        copied_private_states=copied_private_states,
        private_parameters_without_state=private_parameters_without_state,
        fresh_shared_parameters=len(scope.shared_parameter_names),
        route_bank_names=scope.route_bank_names,
        family_bank_names=scope.family_bank_names,
    )


def _require_positive_learning_rate(value: float, *, label: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{label} optimizer learning rate must be finite and positive")


def _require_hybrid_group_roles(groups: list[dict[str, Any]]) -> None:
    roles = tuple(group.get(STATELESS_OPTIMIZER_GROUP_ROLE_KEY) for group in groups)
    if roles != (STATELESS_PRIVATE_GROUP_ROLE, STATELESS_SHARED_GROUP_ROLE):
        raise ValueError("hybrid optimizer group roles or ordering changed")


def _unique_bank_names(
    banks: tuple[tuple[str, nn.ModuleDict], ...],
    *,
    label: str,
) -> tuple[str, ...]:
    names = tuple(name for name, _ in banks)
    if any(not name for name in names) or len(names) != len(set(names)):
        raise RuntimeError(f"{label} bank names are empty or duplicated")
    return names


def _private_parameter_ids(
    *,
    route_banks: tuple[tuple[str, nn.ModuleDict], ...],
    family_banks: tuple[tuple[str, nn.ModuleDict], ...],
) -> set[int]:
    owners: dict[int, str] = {}
    for bank_name, bank in (*route_banks, *family_banks):
        bank_parameters = tuple(bank.parameters())
        if not bank_parameters:
            raise RuntimeError(f"stateless private bank has no parameters: {bank_name}")
        for parameter in bank_parameters:
            parameter_id = id(parameter)
            previous = owners.get(parameter_id)
            if previous is not None:
                raise RuntimeError(
                    "stateless private parameter is shared across banks: "
                    f"{previous}, {bank_name}"
                )
            owners[parameter_id] = bank_name
    return set(owners)


def _optimizer_parameter_names(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[tuple[str, ...], ...]:
    names_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    seen: set[int] = set()
    groups: list[tuple[str, ...]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for raw_parameter in group["params"]:
            parameter = cast(nn.Parameter, raw_parameter)
            parameter_id = id(parameter)
            if parameter_id in seen or parameter_id not in names_by_id:
                raise ValueError("private optimizer parameter ownership is ambiguous")
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
    if {"step", "exp_avg", "exp_avg_sq"} - set(state):
        raise ValueError(f"private AdamW state is incomplete: {name}")
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is not None and (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.shape != parameter.shape
        ):
            raise ValueError(f"private AdamW moment is incompatible: {name}:{key}")
    step = state["step"]
    if isinstance(step, Tensor) and step.numel() != 1:
        raise ValueError(f"private AdamW step is malformed: {name}")


__all__ = [
    "STATELESS_OPTIMIZER_GROUP_ROLE_KEY",
    "STATELESS_PRIVATE_GROUP_ROLE",
    "STATELESS_SHARED_GROUP_ROLE",
    "StatelessHybridOptimizerTransitionAudit",
    "StatelessPrivateOptimizerTransitionAudit",
    "StatelessTrainableParameterScope",
    "build_stateless_hybrid_optimizer_groups",
    "configure_stateless_trainable_scope",
    "retarget_private_optimizer_learning_rate",
    "transplant_full_optimizer_to_private",
    "transplant_private_optimizer_to_hybrid",
]
