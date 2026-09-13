"""Name-bound optimizer-state transplant for deck-registry transitions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.model import AgentPolicyValueNet
from ptcg_rl.rl.deck_transition import (
    DeckRegistryTransitionPlan,
    private_state_expert_id,
)


def transplant_deck_registry_optimizer_state(
    *,
    optimizer: torch.optim.Optimizer,
    model: AgentPolicyValueNet,
    optimizer_parameter_names: Sequence[Sequence[str]],
    state_path: Path,
    source_checkpoint_path: Path,
    source_checkpoint_sha256: str,
    optimizer_state_sha256: str,
    optimizer_state_size_bytes: int,
    plan: DeckRegistryTransitionPlan,
) -> dict[str, Any]:
    """Transplant selected Adam state while keeping target groups and LRs."""
    payload = _load_bound_transition_state(
        state_path=state_path,
        source_checkpoint_path=source_checkpoint_path,
        source_checkpoint_sha256=source_checkpoint_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
        optimizer_state_size_bytes=optimizer_state_size_bytes,
        plan=plan,
    )
    source_optimizer = payload.get("optimizer_state_dict")
    source_names = payload.get("optimizer_parameter_names")
    if not isinstance(source_optimizer, Mapping) or not isinstance(
        source_names, (tuple, list)
    ):
        raise ValueError("transition optimizer sidecar is incomplete")
    source_completed_iterations = _nonnegative_payload_int(
        payload,
        "completed_iterations",
    )
    source_total_optimizer_updates = _nonnegative_payload_int(
        payload,
        "total_optimizer_updates",
    )
    source_planned_scheduler_steps = _positive_payload_int(
        payload,
        "planned_scheduler_steps",
    )
    source_by_name = _optimizer_state_by_name(source_optimizer, source_names)
    source_expert_scheduler_clocks = _source_expert_scheduler_clocks(
        source_optimizer,
        source_names,
        plan=plan,
        total_optimizer_updates=source_total_optimizer_updates,
        planned_scheduler_steps=source_planned_scheduler_steps,
    )
    current_state = optimizer.state_dict()
    current_groups = current_state.get("param_groups")
    if not isinstance(current_groups, list):
        raise ValueError("target optimizer has no parameter groups")
    target_names = tuple(tuple(group) for group in optimizer_parameter_names)
    if len(target_names) != len(current_groups):
        raise ValueError("target optimizer parameter names do not align with groups")
    parameters = dict(model.named_parameters())
    transitions = plan.by_target_expert
    transplanted: dict[int, Any] = {}
    copied_parameters = 0
    preserved_without_state = 0
    reset_parameters = 0
    parameter_actions: dict[str, str] = {}
    for group, names in zip(current_groups, target_names, strict=True):
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
            raise ValueError("target optimizer parameter group is malformed")
        for parameter_id, name in zip(parameter_ids, names, strict=True):
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f"target optimizer parameter is unknown: {name}")
            expert_id = private_state_expert_id(name)
            preserve = (
                expert_id is None or transitions[expert_id].preserve_optimizer_state
            )
            if preserve and name not in source_by_name:
                raise ValueError(
                    f"preserved optimizer parameter is absent from donor map: {name}"
                )
            source_parameter_state = source_by_name.get(name, {})
            if preserve:
                _validate_optimizer_parameter_state(
                    source_parameter_state,
                    parameter,
                    name=name,
                )
                if source_parameter_state:
                    transplanted[int(parameter_id)] = _clone_optimizer_state(
                        source_parameter_state
                    )
                    copied_parameters += 1
                    parameter_actions[name] = "copied"
                else:
                    preserved_without_state += 1
                    parameter_actions[name] = "preserved_without_state"
            else:
                reset_parameters += 1
                parameter_actions[name] = "reset"
    optimizer.load_state_dict(
        {
            "state": transplanted,
            "param_groups": current_groups,
        }
    )
    return {
        "copied_parameters": copied_parameters,
        "preserved_without_state": preserved_without_state,
        "reset_parameters": reset_parameters,
        "parameter_actions_sha256": _fingerprint_actions(parameter_actions),
        "optimizer_state_sha256": optimizer_state_sha256,
        "optimizer_state_size_bytes": optimizer_state_size_bytes,
        "source_completed_iterations": source_completed_iterations,
        "source_total_optimizer_updates": source_total_optimizer_updates,
        "source_planned_scheduler_steps": source_planned_scheduler_steps,
        "source_expert_scheduler_clocks": source_expert_scheduler_clocks,
    }


def validate_deck_registry_source_state(
    *,
    state_path: Path,
    source_checkpoint_path: Path,
    source_checkpoint_sha256: str,
    optimizer_state_sha256: str,
    optimizer_state_size_bytes: int,
    plan: DeckRegistryTransitionPlan,
) -> dict[str, Any]:
    """Validate a source pair while intentionally starting optimizer state fresh."""
    payload = _load_bound_transition_state(
        state_path=state_path,
        source_checkpoint_path=source_checkpoint_path,
        source_checkpoint_sha256=source_checkpoint_sha256,
        optimizer_state_sha256=optimizer_state_sha256,
        optimizer_state_size_bytes=optimizer_state_size_bytes,
        plan=plan,
    )
    return {
        "mode": "fresh",
        "copied_parameters": 0,
        "optimizer_state_sha256": optimizer_state_sha256,
        "optimizer_state_size_bytes": optimizer_state_size_bytes,
        "source_completed_iterations": _nonnegative_payload_int(
            payload,
            "completed_iterations",
        ),
        "source_total_optimizer_updates": _nonnegative_payload_int(
            payload,
            "total_optimizer_updates",
        ),
        "source_planned_scheduler_steps": _positive_payload_int(
            payload,
            "planned_scheduler_steps",
        ),
    }


def _load_bound_transition_state(
    *,
    state_path: Path,
    source_checkpoint_path: Path,
    source_checkpoint_sha256: str,
    optimizer_state_sha256: str,
    optimizer_state_size_bytes: int,
    plan: DeckRegistryTransitionPlan,
) -> Mapping[str, Any]:
    """Load an immutable learner sidecar bound to the declared source policy."""
    if source_checkpoint_sha256 != plan.source_checkpoint_sha256:
        raise ValueError("transition optimizer checkpoint binding disagrees with plan")
    if (
        optimizer_state_sha256 != plan.optimizer_state_sha256
        or optimizer_state_size_bytes != plan.optimizer_state_size_bytes
    ):
        raise ValueError("transition optimizer sidecar binding disagrees with plan")
    if not state_path.is_file():
        raise FileNotFoundError(
            f"transition optimizer sidecar is missing: {state_path}"
        )
    if state_path.stat().st_size != optimizer_state_size_bytes:
        raise ValueError("transition optimizer sidecar size mismatch")
    if file_sha256(state_path) != optimizer_state_sha256:
        raise ValueError("transition optimizer sidecar SHA256 mismatch")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or int(
        payload.get("schema_version", -1)
    ) not in {
        2,
        3,
    }:
        raise ValueError(
            "registry transition requires a bound schema-v2/v3 optimizer sidecar"
        )
    if payload.get("policy_sha256") != source_checkpoint_sha256:
        raise ValueError("transition optimizer sidecar binds a different checkpoint")
    if (
        int(payload.get("policy_size_bytes", -1))
        != source_checkpoint_path.stat().st_size
    ):
        raise ValueError("transition optimizer sidecar checkpoint size mismatch")
    return cast(Mapping[str, Any], payload)


def _optimizer_state_by_name(
    optimizer_state: Mapping[str, Any],
    parameter_names: Sequence[Sequence[str]],
) -> dict[str, Mapping[str, Any]]:
    groups = optimizer_state.get("param_groups")
    states = optimizer_state.get("state")
    if not isinstance(groups, list) or not isinstance(states, Mapping):
        raise ValueError("source optimizer state is malformed")
    if len(groups) != len(parameter_names):
        raise ValueError("source optimizer names do not align with groups")
    result: dict[str, Mapping[str, Any]] = {}
    referenced_parameter_ids: set[Any] = set()
    for group, names in zip(groups, parameter_names, strict=True):
        if not isinstance(group, Mapping):
            raise ValueError("source optimizer parameter group is malformed")
        parameter_ids = group.get("params")
        if not isinstance(parameter_ids, list) or len(parameter_ids) != len(names):
            raise ValueError("source optimizer names do not align with parameters")
        for parameter_id, raw_name in zip(parameter_ids, names, strict=True):
            if parameter_id in referenced_parameter_ids:
                raise ValueError("duplicate source optimizer parameter ID")
            referenced_parameter_ids.add(parameter_id)
            name = str(raw_name)
            if name in result:
                raise ValueError(f"duplicate source optimizer parameter name: {name}")
            state = states.get(parameter_id, {})
            if not isinstance(state, Mapping):
                raise ValueError(f"source optimizer state is malformed: {name}")
            result[name] = cast(Mapping[str, Any], state)
    extra_state_ids = set(states) - referenced_parameter_ids
    if extra_state_ids:
        raise ValueError("source optimizer contains unreferenced parameter states")
    return result


def _source_expert_scheduler_clocks(
    optimizer_state: Mapping[str, Any],
    parameter_names: Sequence[Sequence[str]],
    *,
    plan: DeckRegistryTransitionPlan,
    total_optimizer_updates: int,
    planned_scheduler_steps: int,
) -> dict[str, dict[str, int]]:
    """Read exact per-expert LR clocks required by a continued schedule."""
    if plan.scheduler_mode != "continue":
        return {}
    groups = optimizer_state.get("param_groups")
    if not isinstance(groups, list) or len(groups) != len(parameter_names):
        raise ValueError("source optimizer names do not align with groups")

    clocks: dict[str, dict[str, int]] = {}
    for group, names in zip(groups, parameter_names, strict=True):
        if not isinstance(group, Mapping):
            raise ValueError("source optimizer parameter group is malformed")
        row_expert_ids = tuple(private_state_expert_id(str(name)) for name in names)
        private_expert_ids = {
            expert_id for expert_id in row_expert_ids if expert_id is not None
        }
        declared_expert_id = group.get("expert_id")
        if not private_expert_ids:
            if declared_expert_id is not None:
                raise ValueError(
                    "source optimizer non-private group declares an expert ID"
                )
            continue
        if len(private_expert_ids) != 1 or any(
            expert_id is None for expert_id in row_expert_ids
        ):
            raise ValueError("source optimizer group mixes private expert identities")
        expert_id = next(iter(private_expert_ids))
        if declared_expert_id != expert_id:
            raise ValueError("source optimizer group expert metadata is inconsistent")
        if group.get("name") != f"private.deck_{expert_id}":
            raise ValueError("source optimizer group name does not match its expert")
        if expert_id in clocks:
            raise ValueError("source optimizer contains duplicate expert groups")
        active_updates = _nonnegative_group_int(
            group,
            "scheduler_active_updates",
        )
        horizon_updates = _positive_group_int(
            group,
            "scheduler_horizon_updates",
        )
        if (
            active_updates > horizon_updates
            or active_updates > total_optimizer_updates
            or horizon_updates > planned_scheduler_steps
        ):
            raise ValueError("source optimizer expert clock is inconsistent")
        clocks[expert_id] = {
            "active_updates": active_updates,
            "horizon_updates": horizon_updates,
        }

    expected_experts = set(plan.source_expert_ids)
    if set(clocks) != expected_experts:
        missing = sorted(expected_experts - set(clocks))
        extra = sorted(set(clocks) - expected_experts)
        raise ValueError(
            "source optimizer expert groups do not match the source registry: "
            f"missing={missing}, extra={extra}"
        )
    return {expert_id: clocks[expert_id] for expert_id in sorted(clocks)}


def _validate_optimizer_parameter_state(
    state: Mapping[str, Any],
    parameter: nn.Parameter,
    *,
    name: str,
) -> None:
    if not state:
        return
    required = {"step", "exp_avg", "exp_avg_sq"}
    missing = required - set(state)
    extra = set(state) - required - {"max_exp_avg_sq"}
    if missing or extra:
        raise ValueError(
            f"optimizer Adam state keys are malformed: {name}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not parameter.is_floating_point():
        raise ValueError(f"optimizer target parameter must be floating point: {name}")
    moment_dtypes: set[torch.dtype] = set()
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is None:
            continue
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise ValueError(f"optimizer moment must be floating tensor: {name}:{key}")
        if value.shape != parameter.shape:
            raise ValueError(f"optimizer moment shape mismatch: {name}:{key}")
        if value.dtype not in _compatible_adamw_moment_dtypes(parameter.dtype):
            raise ValueError(
                "optimizer moment dtype is incompatible with target parameter: "
                f"{name}:{key}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"optimizer state is non-finite: {name}:{key}")
        moment_dtypes.add(value.dtype)
    if len(moment_dtypes) != 1:
        raise ValueError(f"optimizer moment dtypes disagree: {name}")
    step = state["step"]
    if isinstance(step, Tensor):
        if step.numel() != 1 or not torch.isfinite(step).all() or step.item() < 0:
            raise ValueError(f"optimizer step is invalid: {name}")
    elif not isinstance(step, (int, float)) or not math.isfinite(step) or step < 0:
        raise ValueError(f"optimizer step is invalid: {name}")


def _compatible_adamw_moment_dtypes(parameter_dtype: torch.dtype) -> set[torch.dtype]:
    """Return supported AdamW moment dtypes for one target parameter dtype."""
    compatible = {parameter_dtype}
    if parameter_dtype in {torch.float16, torch.bfloat16}:
        compatible.add(torch.float32)
    return compatible


def _clone_optimizer_state(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().clone() if isinstance(value, Tensor) else value
        for key, value in state.items()
    }


def _nonnegative_payload_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"transition optimizer sidecar has invalid {key}")
    return value


def _positive_payload_int(payload: Mapping[str, Any], key: str) -> int:
    value = _nonnegative_payload_int(payload, key)
    if value <= 0:
        raise ValueError(f"transition optimizer sidecar has invalid {key}")
    return value


def _nonnegative_group_int(group: Mapping[str, Any], key: str) -> int:
    value = group.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"source optimizer group has invalid {key}")
    return value


def _positive_group_int(group: Mapping[str, Any], key: str) -> int:
    value = _nonnegative_group_int(group, key)
    if value <= 0:
        raise ValueError(f"source optimizer group has invalid {key}")
    return value


def _fingerprint_actions(actions: Mapping[str, str]) -> str:
    import hashlib
    import json

    serialized = json.dumps(
        actions,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


__all__ = [
    "transplant_deck_registry_optimizer_state",
    "validate_deck_registry_source_state",
]
