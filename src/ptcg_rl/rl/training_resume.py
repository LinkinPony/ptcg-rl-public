"""Persistent optimizer and scheduler state for exact RL training resumes."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.rl.checkpoint_pair_io import (
    atomic_write_bytes,
    json_payload,
    publish_torch_file,
)
from ptcg_rl.rl.durable_writer import (
    adopt_frozen_durable_value,
    freeze_durable_value,
)

ResumeMode = Literal["warm_start", "resume", "legacy_resume"]
OptimizerParameterNames = tuple[tuple[str, ...], ...]
_PAIR_BOUND_SCHEMA_VERSION = 3
_BOUND_SCHEMA_VERSION = 2
_LEGACY_SCHEMA_VERSION = 1
_PAIR_MANIFEST_SCHEMA_VERSION = 1
_PAIR_MANIFEST_FORMAT = "exact_policy_learner_pair_v1"
_SHA256_HEX_LENGTH = 64
_OPTIMIZER_GROUP_IDENTITY_FIELDS = ("name", "expert_id")
_OPTIMIZER_GROUP_CLOCK_FIELDS = (
    "scheduler_active_updates",
    "scheduler_horizon_updates",
)


@dataclass(frozen=True, slots=True)
class PreparedTrainingState:
    """CPU-owned exact learner snapshot safe for background serialization."""

    policy_version: int
    completed_iterations: int
    total_optimizer_updates: int
    planned_scheduler_steps: int
    resume_config_sha256: str
    optimizer_parameter_names: OptimizerParameterNames
    optimizer_state_dict: Mapping[str, Any]
    lr_scheduler_state_dict: Mapping[str, Any]
    auxiliary_state: Mapping[str, Any] | None


class TrainingResumeConfig(BaseModel):
    """Control whether a checkpoint is a warm start or an exact resume."""

    model_config = ConfigDict(extra="forbid")

    mode: ResumeMode = "warm_start"
    state_path: Path | None = None
    state_size_bytes: int | None = None
    state_sha256: str | None = None
    completed_iterations: int | None = None
    total_optimizer_updates: int | None = None
    allow_legacy_unbound_state: bool = False
    max_staleness_migration_from: int | None = None
    effective_batch_size_migration_from: int | None = None
    curriculum_assignment_block_size_migration_from: int | None = None
    curriculum_assignment_lanes_migration_from_disabled: bool = False
    learner_shuffle_each_epoch_migration_from: bool | None = None
    frozen_league_migration_from_disabled: bool = False
    engine_teacher_migration_from: Mapping[str, Any] | None = None
    policy_iteration_pending_targets_migration: Literal["empty"] | None = None
    approved_previous_resume_config_sha256: str | None = None

    @field_validator("completed_iterations", "total_optimizer_updates")
    @classmethod
    def valid_optional_counter(cls, value: int | None) -> int | None:
        """Reject negative explicit legacy progress counters."""
        if value is not None and value < 0:
            raise ValueError("resume progress counters must be non-negative")
        return value

    @field_validator("state_size_bytes")
    @classmethod
    def valid_state_size_bytes(cls, value: int | None) -> int | None:
        """Reject invalid optional exact-state byte sizes."""
        if value is not None and value <= 0:
            raise ValueError("resume state_size_bytes must be positive")
        return value

    @field_validator("state_sha256")
    @classmethod
    def valid_state_sha256(cls, value: str | None) -> str | None:
        """Normalize an optional externally audited state fingerprint."""
        if value is None:
            return None
        return _validated_sha256(value, name="resume state")

    @field_validator("max_staleness_migration_from")
    @classmethod
    def valid_staleness_migration_source(cls, value: int | None) -> int | None:
        """Reject negative prior staleness limits."""
        if value is not None and value < 0:
            raise ValueError("prior max staleness must be non-negative")
        return value

    @field_validator("effective_batch_size_migration_from")
    @classmethod
    def valid_batch_size_migration_source(cls, value: int | None) -> int | None:
        """Reject non-positive prior effective learner batch sizes."""
        if value is not None and value <= 0:
            raise ValueError("prior effective batch size must be positive")
        return value

    @field_validator("curriculum_assignment_block_size_migration_from")
    @classmethod
    def valid_assignment_block_migration_source(cls, value: int | None) -> int | None:
        """Reject non-positive prior assignment cohort sizes."""
        if value is not None and value <= 0:
            raise ValueError("prior curriculum assignment block size must be positive")
        return value

    @field_validator("approved_previous_resume_config_sha256")
    @classmethod
    def valid_previous_resume_config_sha256(cls, value: str | None) -> str | None:
        """Require a canonical audited source fingerprint when supplied."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != _SHA256_HEX_LENGTH or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError(
                "approved_previous_resume_config_sha256 must be a SHA-256 digest"
            )
        return normalized

    @model_validator(mode="after")
    def valid_mode_fields(self) -> TrainingResumeConfig:
        """Require explicit progress when converting a legacy checkpoint."""
        if (self.state_size_bytes is None) != (self.state_sha256 is None):
            raise ValueError(
                "resume state_size_bytes and state_sha256 must be provided together"
            )
        if self.state_size_bytes is not None and self.mode != "resume":
            raise ValueError(
                "resume state fingerprint is only valid for exact resume mode"
            )
        if self.allow_legacy_unbound_state and self.mode != "resume":
            raise ValueError(
                "allow_legacy_unbound_state is only valid for exact resume mode"
            )
        if self.max_staleness_migration_from is not None and self.mode != "resume":
            raise ValueError(
                "max_staleness_migration_from is only valid for exact resume mode"
            )
        if (
            self.effective_batch_size_migration_from is not None
            and self.mode != "resume"
        ):
            raise ValueError(
                "effective_batch_size_migration_from is only valid for exact "
                "resume mode"
            )
        if (
            self.curriculum_assignment_block_size_migration_from is not None
            and self.mode != "resume"
        ):
            raise ValueError(
                "curriculum assignment migration is only valid for exact resume mode"
            )
        if (
            self.curriculum_assignment_lanes_migration_from_disabled
            and self.mode != "resume"
        ):
            raise ValueError(
                "curriculum lane migration is only valid for exact resume mode"
            )
        if (
            self.learner_shuffle_each_epoch_migration_from is not None
            and self.mode != "resume"
        ):
            raise ValueError(
                "learner shuffle migration is only valid for exact resume mode"
            )
        if self.frozen_league_migration_from_disabled and self.mode != "resume":
            raise ValueError(
                "frozen_league_migration_from_disabled is only valid for exact "
                "resume mode"
            )
        if self.engine_teacher_migration_from is not None and self.mode != "resume":
            raise ValueError(
                "engine_teacher_migration_from is only valid for exact resume mode"
            )
        if self.engine_teacher_migration_from is not None and not (
            self.engine_teacher_migration_from
        ):
            raise ValueError("engine_teacher_migration_from must not be empty")
        if (
            self.policy_iteration_pending_targets_migration is not None
            and self.mode != "resume"
        ):
            raise ValueError(
                "policy-iteration pending-target migration is only valid for "
                "exact resume mode"
            )
        has_declared_migration = bool(
            self.max_staleness_migration_from is not None
            or self.effective_batch_size_migration_from is not None
            or self.curriculum_assignment_block_size_migration_from is not None
            or self.curriculum_assignment_lanes_migration_from_disabled
            or self.learner_shuffle_each_epoch_migration_from is not None
            or self.frozen_league_migration_from_disabled
            or self.engine_teacher_migration_from is not None
            or self.policy_iteration_pending_targets_migration is not None
        )
        if (
            self.approved_previous_resume_config_sha256 is not None
            and not has_declared_migration
        ):
            raise ValueError(
                "an approved previous resume hash requires an explicit resume "
                "migration declaration"
            )
        if (
            self.max_staleness_migration_from is not None
            or self.effective_batch_size_migration_from is not None
            or self.curriculum_assignment_block_size_migration_from is not None
            or self.curriculum_assignment_lanes_migration_from_disabled
            or self.learner_shuffle_each_epoch_migration_from is not None
            or self.frozen_league_migration_from_disabled
            or self.engine_teacher_migration_from is not None
            or self.policy_iteration_pending_targets_migration is not None
            or self.approved_previous_resume_config_sha256 is not None
        ) and self.allow_legacy_unbound_state:
            raise ValueError("resume config migrations require a bound resume state")
        if self.mode == "legacy_resume":
            if self.completed_iterations is None:
                raise ValueError("legacy_resume requires resume.completed_iterations")
            if self.total_optimizer_updates is None:
                raise ValueError(
                    "legacy_resume requires resume.total_optimizer_updates"
                )
        elif self.completed_iterations is not None:
            raise ValueError(
                "resume.completed_iterations is only valid for legacy_resume"
            )
        elif self.total_optimizer_updates is not None:
            raise ValueError(
                "resume.total_optimizer_updates is only valid for legacy_resume"
            )
        return self


@dataclass(frozen=True)
class TrainingProgress:
    """Cumulative learner progress restored before entering the train loop."""

    completed_iterations: int
    total_optimizer_updates: int
    state_path: Path | None
    ppo_optimizer_updates: int | None = None


def restore_training_progress(
    *,
    config: TrainingResumeConfig,
    checkpoint_path: Path | None,
    policy_version: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    planned_scheduler_steps: int,
    resume_config_sha256: str | None = None,
    approved_previous_resume_config_sha256: str | None = None,
    optimizer_parameter_names: Sequence[Sequence[str]] = (),
    new_optimizer_parameters: Sequence[torch.Tensor] = (),
) -> TrainingProgress:
    """Restore exact state or initialize explicitly requested legacy progress."""
    if config.mode == "warm_start":
        return TrainingProgress(0, 0, None, 0)
    if checkpoint_path is None:
        raise ValueError(f"resume mode {config.mode!r} requires checkpoint_path")
    if config.mode == "legacy_resume":
        completed_iterations = _required_counter(
            config.completed_iterations,
            name="completed_iterations",
        )
        total_optimizer_updates = _required_counter(
            config.total_optimizer_updates,
            name="total_optimizer_updates",
        )
        _set_cosine_scheduler_progress(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            completed_updates=total_optimizer_updates,
        )
        return TrainingProgress(
            completed_iterations=completed_iterations,
            total_optimizer_updates=total_optimizer_updates,
            state_path=None,
            ppo_optimizer_updates=None,
        )

    state_path = config.state_path or inferred_training_state_path(
        checkpoint_path,
        policy_version=policy_version,
    )
    if not state_path.exists():
        raise FileNotFoundError(
            "exact resume state is missing; use warm_start for a new branch or "
            "legacy_resume with explicit progress for an old checkpoint: "
            f"{state_path}"
        )
    if config.state_size_bytes is not None and config.state_sha256 is not None:
        actual_state_fingerprint = _stable_file_fingerprint(state_path)
        expected_state_fingerprint = (
            config.state_size_bytes,
            config.state_sha256,
        )
        if actual_state_fingerprint != expected_state_fingerprint:
            raise ValueError(
                "resume state fingerprint does not match the selected sidecar: "
                f"expected={expected_state_fingerprint}, "
                f"actual={actual_state_fingerprint}"
            )
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"resume state must be a mapping: {state_path}")
    schema_version = _validate_resume_payload(
        payload,
        state_path=state_path,
        checkpoint_path=checkpoint_path,
        policy_version=policy_version,
        planned_scheduler_steps=planned_scheduler_steps,
        resume_config_sha256=resume_config_sha256,
        approved_previous_resume_config_sha256=(approved_previous_resume_config_sha256),
        optimizer_parameter_names=optimizer_parameter_names,
        optimizer=optimizer,
        allow_legacy_unbound_state=config.allow_legacy_unbound_state,
    )
    expected_scheduler_base_lrs = _validated_learning_rates(
        lr_scheduler.base_lrs,
        name="fresh scheduler base learning rates",
        expected_count=len(optimizer.param_groups),
    )
    _load_optimizer_state_dict(
        optimizer,
        payload["optimizer_state_dict"],
        new_parameters=(
            new_optimizer_parameters if schema_version == _LEGACY_SCHEMA_VERSION else ()
        ),
        require_bound_group_identity=schema_version != _LEGACY_SCHEMA_VERSION,
    )
    lr_scheduler.load_state_dict(payload["lr_scheduler_state_dict"])
    _validate_restored_scheduler(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        expected_base_lrs=expected_scheduler_base_lrs,
    )
    return TrainingProgress(
        completed_iterations=int(payload["completed_iterations"]),
        total_optimizer_updates=int(payload["total_optimizer_updates"]),
        state_path=state_path,
        ppo_optimizer_updates=_optional_auxiliary_counter(
            payload,
            name="ppo_optimizer_updates",
        ),
    )


def _optional_auxiliary_counter(
    payload: Mapping[str, Any],
    *,
    name: str,
) -> int | None:
    """Read one optional non-negative exact-resume auxiliary clock."""
    auxiliary = payload.get("auxiliary_state")
    if not isinstance(auxiliary, Mapping) or name not in auxiliary:
        return None
    value = auxiliary[name]
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"resume auxiliary {name} must be a non-negative integer")
    return value


def _load_optimizer_state_dict(
    optimizer: torch.optim.Optimizer,
    state_dict: Any,
    *,
    new_parameters: Sequence[torch.Tensor],
    require_bound_group_identity: bool,
) -> None:
    """Load optimizer state, explicitly inserting newly introduced parameters."""
    if not isinstance(state_dict, dict):
        raise ValueError("optimizer state must be a mapping")
    current_state = optimizer.state_dict()
    saved_groups = state_dict.get("param_groups")
    current_groups = current_state.get("param_groups")
    if not isinstance(saved_groups, list) or not isinstance(current_groups, list):
        raise ValueError("optimizer state must contain parameter groups")
    if len(saved_groups) != len(current_groups):
        raise ValueError("optimizer parameter group count changed across exact resume")
    if require_bound_group_identity:
        _validate_bound_optimizer_groups(saved_groups, current_groups)
    if all(
        isinstance(saved, Mapping)
        and isinstance(current, Mapping)
        and isinstance(saved.get("params"), list)
        and isinstance(current.get("params"), list)
        and len(saved["params"]) == len(current["params"])
        for saved, current in zip(saved_groups, current_groups, strict=True)
    ):
        _validate_optimizer_parameter_states(optimizer, state_dict)
        optimizer.load_state_dict(state_dict)
        return
    if not new_parameters:
        raise ValueError("optimizer parameter layout changed across exact resume")

    new_parameter_ids = {id(parameter) for parameter in new_parameters}
    migrated_state: dict[Any, Any] = {}
    saved_state = state_dict.get("state")
    if not isinstance(saved_state, dict):
        raise ValueError("optimizer state must contain a state mapping")
    migrated_groups: list[dict[str, Any]] = []
    for group, saved_group, current_group in zip(
        optimizer.param_groups,
        saved_groups,
        current_groups,
        strict=True,
    ):
        saved_ids = list(saved_group["params"])
        current_ids = list(current_group["params"])
        old_current_parameters = [
            (parameter, parameter_id)
            for parameter, parameter_id in zip(
                group["params"],
                current_ids,
                strict=True,
            )
            if id(parameter) not in new_parameter_ids
        ]
        if len(saved_ids) != len(old_current_parameters):
            raise ValueError(
                "optimizer parameter layout changed beyond declared new parameters"
            )
        for saved_id, (parameter, current_id) in zip(
            saved_ids,
            old_current_parameters,
            strict=True,
        ):
            if saved_id in saved_state:
                parameter_state = saved_state[saved_id]
                _validate_adamw_parameter_state(parameter, parameter_state)
                migrated_state[current_id] = parameter_state
        migrated_group = dict(saved_group)
        migrated_group["params"] = current_ids
        migrated_groups.append(migrated_group)
    optimizer.load_state_dict(
        {
            "state": migrated_state,
            "param_groups": migrated_groups,
        }
    )


def _validate_bound_optimizer_groups(
    saved_groups: Sequence[Any],
    current_groups: Sequence[Any],
) -> None:
    """Bind persisted group metadata and parameter IDs to the fresh optimizer."""
    for group_index, (saved_group, current_group) in enumerate(
        zip(saved_groups, current_groups, strict=True)
    ):
        if not isinstance(saved_group, Mapping) or not isinstance(
            current_group,
            Mapping,
        ):
            raise ValueError("optimizer parameter group must be a mapping")
        saved_ids = saved_group.get("params")
        current_ids = current_group.get("params")
        if (
            not isinstance(saved_ids, list)
            or not isinstance(current_ids, list)
            or saved_ids != current_ids
        ):
            raise ValueError(
                "optimizer parameter IDs or order changed across exact resume"
            )
        _validate_optimizer_group_metadata(
            saved_group,
            label=f"saved optimizer group {group_index}",
        )
        _validate_optimizer_group_metadata(
            current_group,
            label=f"current optimizer group {group_index}",
        )
        for field in _OPTIMIZER_GROUP_IDENTITY_FIELDS:
            if (field in saved_group) != (field in current_group):
                raise ValueError(
                    f"optimizer group identity field changed across exact resume: {field}"
                )
            if field in saved_group and saved_group[field] != current_group[field]:
                raise ValueError(
                    f"optimizer group identity changed across exact resume: {field}"
                )
        for field in _OPTIMIZER_GROUP_CLOCK_FIELDS:
            if (field in saved_group) != (field in current_group):
                raise ValueError(
                    f"optimizer group scheduler field changed across exact resume: {field}"
                )


def _validate_optimizer_group_metadata(
    group: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Validate reserved optimizer group metadata without rewriting its clock."""
    if "name" in group:
        name = group["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{label} name must be a non-empty string")
    if "expert_id" in group:
        expert_id = group["expert_id"]
        if expert_id is not None:
            if not isinstance(expert_id, str):
                raise ValueError(f"{label} expert ID must be a string or null")
            _validated_sha256(expert_id, name=f"{label} expert ID")
    active_updates: int | None = None
    if "scheduler_active_updates" in group:
        active_updates = _validated_group_counter(
            group["scheduler_active_updates"],
            name=f"{label} scheduler_active_updates",
            positive=False,
        )
    horizon_updates: int | None = None
    if "scheduler_horizon_updates" in group:
        horizon_updates = _validated_group_counter(
            group["scheduler_horizon_updates"],
            name=f"{label} scheduler_horizon_updates",
            positive=True,
        )
    if (
        active_updates is not None
        and horizon_updates is not None
        and active_updates > horizon_updates
    ):
        raise ValueError(
            f"{label} scheduler_active_updates cannot exceed scheduler_horizon_updates"
        )


def _validated_group_counter(value: Any, *, name: str, positive: bool) -> int:
    """Return one real integer optimizer clock after checking its range."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    if value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _validate_restored_scheduler(
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    expected_base_lrs: Sequence[float],
) -> None:
    """Reject a sidecar whose scheduler no longer agrees with group clocks."""
    group_count = len(optimizer.param_groups)
    restored_base_lrs = _validated_learning_rates(
        lr_scheduler.base_lrs,
        name="restored scheduler base learning rates",
        expected_count=group_count,
    )
    for restored, expected in zip(
        restored_base_lrs,
        expected_base_lrs,
        strict=True,
    ):
        _require_matching_learning_rate(
            restored,
            expected,
            message="scheduler base learning rates changed across exact resume",
        )

    last_lrs = _validated_learning_rates(
        lr_scheduler.get_last_lr(),
        name="restored scheduler last learning rates",
        expected_count=group_count,
    )
    group_lrs = _validated_learning_rates(
        tuple(group.get("lr") for group in optimizer.param_groups),
        name="restored optimizer group learning rates",
        expected_count=group_count,
    )
    for group_lr, last_lr in zip(group_lrs, last_lrs, strict=True):
        _require_matching_learning_rate(
            group_lr,
            last_lr,
            message="optimizer learning rate disagrees with scheduler _last_lr",
        )

    if not isinstance(lr_scheduler, torch.optim.lr_scheduler.LambdaLR):
        return
    last_epoch = lr_scheduler.last_epoch
    if not isinstance(last_epoch, int) or isinstance(last_epoch, bool):
        raise ValueError("restored LambdaLR last_epoch must be an integer")
    if len(lr_scheduler.lr_lambdas) != group_count:
        raise ValueError("restored LambdaLR function count does not match groups")
    for group_index, (base_lr, schedule, group_lr) in enumerate(
        zip(
            expected_base_lrs,
            lr_scheduler.lr_lambdas,
            group_lrs,
            strict=True,
        )
    ):
        multiplier = _validated_learning_rate(
            schedule(last_epoch),
            name=f"restored LambdaLR group {group_index} multiplier",
        )
        expected_lr = base_lr * multiplier
        _require_matching_learning_rate(
            group_lr,
            expected_lr,
            message=(
                "optimizer learning rate disagrees with the clock-derived "
                f"LambdaLR value for group {group_index}"
            ),
        )


def _validated_learning_rates(
    values: Any,
    *,
    name: str,
    expected_count: int,
) -> tuple[float, ...]:
    """Return one finite, non-negative scalar learning rate per group."""
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    if len(values) != expected_count:
        raise ValueError(f"{name} count does not match optimizer groups")
    return tuple(
        _validated_learning_rate(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )


def _validated_learning_rate(value: Any, *, name: str) -> float:
    """Coerce one scalar learning rate without accepting bool or non-finite data."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype == torch.bool or value.is_complex():
            raise ValueError(f"{name} must be a real scalar")
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _require_matching_learning_rate(
    actual: float,
    expected: float,
    *,
    message: str,
) -> None:
    """Require persisted floating-point LR values to encode the same schedule."""
    if not math.isclose(actual, expected, rel_tol=1.0e-9, abs_tol=1.0e-12):
        raise ValueError(f"{message}: {actual} != {expected}")


def _validate_optimizer_parameter_states(
    optimizer: torch.optim.Optimizer,
    state_dict: dict[str, Any],
) -> None:
    """Validate saved optimizer tensors against their named current positions."""
    saved_groups = state_dict.get("param_groups")
    saved_state = state_dict.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise ValueError("optimizer state must contain parameter groups and state")
    for group, saved_group in zip(
        optimizer.param_groups,
        saved_groups,
        strict=True,
    ):
        saved_ids = saved_group.get("params")
        if not isinstance(saved_ids, list) or len(saved_ids) != len(group["params"]):
            raise ValueError("optimizer parameter layout changed across exact resume")
        for parameter, saved_id in zip(
            group["params"],
            saved_ids,
            strict=True,
        ):
            if saved_id in saved_state:
                _validate_adamw_parameter_state(parameter, saved_state[saved_id])


def _validate_adamw_parameter_state(
    parameter: torch.Tensor,
    state: Any,
) -> None:
    """Reject malformed AdamW moments before PyTorch casts or installs them."""
    if not isinstance(state, dict):
        raise ValueError("optimizer parameter state must be a mapping")
    required = {"step", "exp_avg", "exp_avg_sq"}
    missing = required - set(state)
    extra = set(state) - required - {"max_exp_avg_sq"}
    if missing or extra:
        raise ValueError(
            "optimizer AdamW state keys are malformed: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not parameter.is_floating_point():
        raise ValueError("optimizer target parameter must be floating point")
    moment_dtypes: set[torch.dtype] = set()
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(key)
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise ValueError(f"optimizer moment must be a floating tensor: {key}")
        if value.shape != parameter.shape:
            raise ValueError(
                "optimizer parameter order changed across exact resume: "
                f"{key} has shape {tuple(value.shape)}, expected "
                f"{tuple(parameter.shape)}"
            )
        if value.dtype not in _compatible_adamw_moment_dtypes(parameter.dtype):
            raise ValueError(
                "optimizer moment dtype is incompatible with target parameter: "
                f"{key} has dtype {value.dtype}, expected {parameter.dtype}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"optimizer moment is non-finite: {key}")
        moment_dtypes.add(value.dtype)
    if len(moment_dtypes) != 1:
        raise ValueError("optimizer moment dtypes disagree")
    _validate_optimizer_step(state["step"])


def _compatible_adamw_moment_dtypes(parameter_dtype: torch.dtype) -> set[torch.dtype]:
    """Return supported AdamW moment dtypes for one target parameter dtype."""
    compatible = {parameter_dtype}
    if parameter_dtype in {torch.float16, torch.bfloat16}:
        compatible.add(torch.float32)
    return compatible


def _validate_optimizer_step(value: Any) -> None:
    """Reject non-scalar, non-finite, or negative AdamW update clocks."""
    if isinstance(value, torch.Tensor):
        if (
            value.numel() != 1
            or value.dtype == torch.bool
            or value.is_complex()
            or not torch.isfinite(value).all()
            or float(value.item()) < 0.0
        ):
            raise ValueError("optimizer step is invalid")
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError("optimizer step is invalid")


def prepare_training_state(
    *,
    policy_version: int,
    completed_iterations: int,
    total_optimizer_updates: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    planned_scheduler_steps: int,
    resume_config_sha256: str,
    optimizer_parameter_names: Sequence[Sequence[str]],
    auxiliary_state: Mapping[str, Any] | None = None,
    take_auxiliary_ownership: bool = False,
) -> PreparedTrainingState:
    """Freeze every mutable learner tensor before asynchronous publication."""
    if policy_version < 0:
        raise ValueError("policy_version must be non-negative")
    if completed_iterations < 0 or total_optimizer_updates < 0:
        raise ValueError("training progress counters must be non-negative")
    if planned_scheduler_steps <= 0:
        raise ValueError("planned_scheduler_steps must be positive")
    validated_config_sha256 = _validated_sha256(
        resume_config_sha256,
        name="resume config",
    )
    validated_parameter_names = _validated_optimizer_parameter_names(
        optimizer_parameter_names,
        optimizer=optimizer,
    )
    frozen_optimizer = freeze_durable_value(optimizer.state_dict())
    frozen_scheduler = freeze_durable_value(lr_scheduler.state_dict())
    frozen_auxiliary = (
        None
        if auxiliary_state is None
        else (
            adopt_frozen_durable_value(auxiliary_state)
            if take_auxiliary_ownership
            else freeze_durable_value(auxiliary_state)
        )
    )
    if not isinstance(frozen_optimizer, Mapping):
        raise TypeError("frozen optimizer state must be a mapping")
    if not isinstance(frozen_scheduler, Mapping):
        raise TypeError("frozen LR scheduler state must be a mapping")
    if frozen_auxiliary is not None and not isinstance(frozen_auxiliary, Mapping):
        raise TypeError("frozen auxiliary state must be a mapping")
    return PreparedTrainingState(
        policy_version=policy_version,
        completed_iterations=completed_iterations,
        total_optimizer_updates=total_optimizer_updates,
        planned_scheduler_steps=planned_scheduler_steps,
        resume_config_sha256=validated_config_sha256,
        optimizer_parameter_names=validated_parameter_names,
        optimizer_state_dict=frozen_optimizer,
        lr_scheduler_state_dict=frozen_scheduler,
        auxiliary_state=frozen_auxiliary,
    )


def prepared_training_state_payload(
    prepared: PreparedTrainingState,
    *,
    policy_size_bytes: int,
    policy_sha256: str,
) -> dict[str, Any]:
    """Bind a frozen learner snapshot to one immutable policy file."""
    if policy_size_bytes <= 0:
        raise ValueError("bound policy size must be positive")
    validated_policy_sha256 = _validated_sha256(
        policy_sha256,
        name="bound policy",
    )
    payload: dict[str, Any] = {
        "schema_version": _PAIR_BOUND_SCHEMA_VERSION,
        "policy_version": prepared.policy_version,
        "completed_iterations": prepared.completed_iterations,
        "total_optimizer_updates": prepared.total_optimizer_updates,
        "planned_scheduler_steps": prepared.planned_scheduler_steps,
        "policy_size_bytes": policy_size_bytes,
        "policy_sha256": validated_policy_sha256,
        "resume_config_sha256": prepared.resume_config_sha256,
        "optimizer_parameter_names": prepared.optimizer_parameter_names,
        "optimizer_state_dict": prepared.optimizer_state_dict,
        "lr_scheduler_state_dict": prepared.lr_scheduler_state_dict,
    }
    if prepared.auxiliary_state is not None:
        payload["auxiliary_state"] = prepared.auxiliary_state
    return payload


def commit_prepared_training_state(
    directory: Path,
    *,
    path: Path,
    prepared: PreparedTrainingState,
    policy_size_bytes: int,
    policy_sha256: str,
    keep_last: int,
    retain_every_versions: int | None,
    prune: bool = True,
) -> None:
    """Publish the legacy resume pointer after its complete pair exists."""
    expected_path = directory / _state_filename(prepared.policy_version)
    if path.resolve() != expected_path.resolve():
        raise ValueError("prepared training state path is not canonical")
    if not expected_path.is_file():
        raise FileNotFoundError(f"prepared training state is missing: {expected_path}")
    _write_latest_pointer(
        directory,
        schema_version=_PAIR_BOUND_SCHEMA_VERSION,
        path=expected_path,
        policy_version=prepared.policy_version,
        completed_iterations=prepared.completed_iterations,
        total_optimizer_updates=prepared.total_optimizer_updates,
        policy_size_bytes=policy_size_bytes,
        policy_sha256=_validated_sha256(policy_sha256, name="bound policy"),
        resume_config_sha256=prepared.resume_config_sha256,
    )
    if prune:
        _prune_training_states(
            directory,
            keep_last=keep_last,
            retain_every_versions=retain_every_versions,
        )


def prune_training_states(
    directory: Path,
    *,
    keep_last: int,
    retain_every_versions: int | None,
) -> None:
    """Apply exact-resume retention after a complete pair commit."""
    _prune_training_states(
        directory,
        keep_last=keep_last,
        retain_every_versions=retain_every_versions,
    )


def training_state_path(directory: Path, *, policy_version: int) -> Path:
    """Return the canonical exact-resume sidecar path for one version."""
    if policy_version < 0:
        raise ValueError("policy_version must be non-negative")
    return Path(directory) / _state_filename(policy_version)


def publish_training_state(
    directory: Path,
    *,
    policy_version: int,
    completed_iterations: int,
    total_optimizer_updates: int,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    planned_scheduler_steps: int,
    policy_path: Path,
    resume_config_sha256: str,
    optimizer_parameter_names: Sequence[Sequence[str]],
    auxiliary_state: Mapping[str, Any] | None = None,
    keep_last: int = 2,
    retain_every_versions: int | None = None,
) -> Path:
    """Atomically publish a compact learner-state sidecar and latest pointer."""
    if keep_last <= 0:
        raise ValueError("keep_last must be positive")
    if retain_every_versions is not None and retain_every_versions <= 0:
        raise ValueError("retain_every_versions must be positive when set")
    policy_size_bytes, policy_sha256 = _stable_file_fingerprint(policy_path)
    validated_config_sha256 = _validated_sha256(
        resume_config_sha256,
        name="resume config",
    )
    validated_parameter_names = _validated_optimizer_parameter_names(
        optimizer_parameter_names,
        optimizer=optimizer,
    )
    directory.mkdir(parents=True, exist_ok=True)
    final_path = directory / _state_filename(policy_version)
    payload: dict[str, Any] = {
        # This compatibility writer predates the authoritative pair manifest.
        # New training publications use ``prepared_training_state_payload``.
        "schema_version": _BOUND_SCHEMA_VERSION,
        "policy_version": policy_version,
        "completed_iterations": completed_iterations,
        "total_optimizer_updates": total_optimizer_updates,
        "planned_scheduler_steps": planned_scheduler_steps,
        "policy_size_bytes": policy_size_bytes,
        "policy_sha256": policy_sha256,
        "resume_config_sha256": validated_config_sha256,
        "optimizer_parameter_names": validated_parameter_names,
        "optimizer_state_dict": optimizer.state_dict(),
        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
    }
    if auxiliary_state is not None:
        payload["auxiliary_state"] = dict(auxiliary_state)
    publish_torch_file(final_path, payload)
    _write_latest_pointer(
        directory,
        schema_version=_BOUND_SCHEMA_VERSION,
        path=final_path,
        policy_version=policy_version,
        completed_iterations=completed_iterations,
        total_optimizer_updates=total_optimizer_updates,
        policy_size_bytes=policy_size_bytes,
        policy_sha256=policy_sha256,
        resume_config_sha256=validated_config_sha256,
    )
    _prune_training_states(
        directory,
        keep_last=keep_last,
        retain_every_versions=retain_every_versions,
    )
    return final_path


def adopt_training_state_pointer(
    directory: Path,
    *,
    state_path: Path,
    policy_version: int,
) -> Path:
    """Rebuild a host-local pointer for an already validated resume sidecar."""
    expected_path = directory / _state_filename(policy_version)
    if state_path.resolve() != expected_path.resolve():
        raise ValueError(
            "adopted training state must use the run's numbered resume path: "
            f"{state_path} != {expected_path}"
        )
    if not expected_path.is_file():
        raise FileNotFoundError(f"adopted training state is missing: {expected_path}")
    payload = torch.load(expected_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"resume state must be a mapping: {expected_path}")
    schema_version = int(payload.get("schema_version", -1))
    if schema_version not in {_BOUND_SCHEMA_VERSION, _PAIR_BOUND_SCHEMA_VERSION}:
        raise ValueError(
            f"only a bound schema-v2/v3 training state can be adopted: {expected_path}"
        )
    if int(payload.get("policy_version", -1)) != policy_version:
        raise ValueError(
            "adopted training state policy version does not match its filename"
        )
    policy_sha256 = _validated_sha256(
        payload.get("policy_sha256"),
        name="bound policy",
    )
    resume_config_sha256 = _validated_sha256(
        payload.get("resume_config_sha256"),
        name="bound resume config",
    )
    directory.mkdir(parents=True, exist_ok=True)
    _write_latest_pointer(
        directory,
        schema_version=schema_version,
        path=expected_path,
        policy_version=policy_version,
        completed_iterations=int(payload["completed_iterations"]),
        total_optimizer_updates=int(payload["total_optimizer_updates"]),
        policy_size_bytes=int(payload["policy_size_bytes"]),
        policy_sha256=policy_sha256,
        resume_config_sha256=resume_config_sha256,
    )
    return expected_path


def inferred_training_state_path(
    checkpoint_path: Path,
    *,
    policy_version: int,
) -> Path:
    """Infer the resume sidecar next to a run's weights directory."""
    weights_dir = checkpoint_path.parent
    if weights_dir.name != "weights":
        raise ValueError(
            "cannot infer resume state from a checkpoint outside a weights "
            f"directory: {checkpoint_path}"
        )
    return weights_dir.parent / "resume" / _state_filename(policy_version)


def _required_counter(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"legacy resume requires {name}")
    return value


def _set_cosine_scheduler_progress(
    *,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    completed_updates: int,
) -> None:
    if not isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
        raise TypeError("legacy resume only supports CosineAnnealingLR")
    scheduler_state = lr_scheduler.state_dict()
    total_steps = int(scheduler_state["T_max"])
    eta_min = float(scheduler_state["eta_min"])
    base_lrs = [float(value) for value in scheduler_state["base_lrs"]]
    bounded_updates = min(completed_updates, total_steps)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * bounded_updates / total_steps))
    current_lrs = [eta_min + (base_lr - eta_min) * multiplier for base_lr in base_lrs]
    scheduler_state["last_epoch"] = completed_updates
    scheduler_state["_step_count"] = completed_updates + 1
    scheduler_state["_last_lr"] = current_lrs
    lr_scheduler.load_state_dict(scheduler_state)
    for parameter_group, learning_rate in zip(
        optimizer.param_groups,
        current_lrs,
        strict=True,
    ):
        parameter_group["lr"] = learning_rate


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    state_path: Path,
    checkpoint_path: Path,
    policy_version: int,
    planned_scheduler_steps: int,
    resume_config_sha256: str | None,
    approved_previous_resume_config_sha256: str | None,
    optimizer_parameter_names: Sequence[Sequence[str]],
    optimizer: torch.optim.Optimizer,
    allow_legacy_unbound_state: bool,
) -> int:
    common_required = {
        "schema_version",
        "policy_version",
        "completed_iterations",
        "total_optimizer_updates",
        "planned_scheduler_steps",
        "optimizer_state_dict",
        "lr_scheduler_state_dict",
    }
    missing = sorted(common_required.difference(payload))
    if missing:
        raise ValueError(f"resume state is missing fields {missing}: {state_path}")
    schema_version = int(payload["schema_version"])
    if schema_version == _LEGACY_SCHEMA_VERSION:
        if not allow_legacy_unbound_state:
            raise ValueError(
                "legacy unbound resume state is disabled; set "
                "resume.allow_legacy_unbound_state=true only for an audited "
                f"historical recovery: {state_path}"
            )
    elif schema_version not in {
        _BOUND_SCHEMA_VERSION,
        _PAIR_BOUND_SCHEMA_VERSION,
    }:
        raise ValueError(f"unsupported resume state schema: {state_path}")
    if int(payload["policy_version"]) != policy_version:
        raise ValueError(
            "resume state policy version does not match checkpoint: "
            f"{payload['policy_version']} != {policy_version}"
        )
    if int(payload["planned_scheduler_steps"]) != planned_scheduler_steps:
        raise ValueError(
            "resume scheduler plan does not match the current config: "
            f"{payload['planned_scheduler_steps']} != {planned_scheduler_steps}"
        )
    if schema_version == _LEGACY_SCHEMA_VERSION:
        return schema_version

    pair_policy_fingerprint: tuple[int, str] | None = None
    if schema_version == _PAIR_BOUND_SCHEMA_VERSION:
        pair_policy_fingerprint = _validate_checkpoint_pair_manifest(
            state_path=state_path,
            checkpoint_path=checkpoint_path,
            policy_version=policy_version,
        )

    bound_required = {
        "policy_size_bytes",
        "policy_sha256",
        "resume_config_sha256",
        "optimizer_parameter_names",
    }
    missing = sorted(bound_required.difference(payload))
    if missing:
        raise ValueError(f"resume state is missing fields {missing}: {state_path}")
    expected_size = int(payload["policy_size_bytes"])
    expected_policy_sha256 = _validated_sha256(
        payload["policy_sha256"],
        name="bound policy",
    )
    actual_size, actual_policy_sha256 = (
        pair_policy_fingerprint
        if pair_policy_fingerprint is not None
        else _stable_file_fingerprint(checkpoint_path)
    )
    if (actual_size, actual_policy_sha256) != (
        expected_size,
        expected_policy_sha256,
    ):
        raise ValueError(
            "resume state policy fingerprint does not match checkpoint: "
            f"expected=({expected_size}, {expected_policy_sha256}), "
            f"actual=({actual_size}, {actual_policy_sha256})"
        )
    if resume_config_sha256 is None:
        raise ValueError("exact resume requires a resume-relevant config fingerprint")
    expected_config_sha256 = _validated_sha256(
        payload["resume_config_sha256"],
        name="bound resume config",
    )
    actual_config_sha256 = _validated_sha256(
        resume_config_sha256,
        name="current resume config",
    )
    if actual_config_sha256 != expected_config_sha256:
        if approved_previous_resume_config_sha256 is None:
            raise ValueError(
                "resume-relevant config fingerprint does not match state: "
                f"{actual_config_sha256} != {expected_config_sha256}"
            )
        approved_previous_sha256 = _validated_sha256(
            approved_previous_resume_config_sha256,
            name="approved previous resume config",
        )
        if expected_config_sha256 != approved_previous_sha256:
            raise ValueError(
                "bound resume config does not match the approved migration "
                f"source: {expected_config_sha256} != {approved_previous_sha256}"
            )
    expected_parameter_names = _validated_optimizer_parameter_names(
        payload["optimizer_parameter_names"],
        optimizer=optimizer,
    )
    actual_parameter_names = _validated_optimizer_parameter_names(
        optimizer_parameter_names,
        optimizer=optimizer,
    )
    if actual_parameter_names != expected_parameter_names:
        raise ValueError(
            "optimizer parameter names or order changed across exact resume"
        )
    return schema_version


def _validate_checkpoint_pair_manifest(
    *,
    state_path: Path,
    checkpoint_path: Path,
    policy_version: int,
) -> tuple[int, str]:
    """Require one committed manifest that fingerprints both resume artifacts."""
    manifest_path = _checkpoint_pair_manifest_path(
        checkpoint_path,
        policy_version=policy_version,
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"exact resume checkpoint-pair manifest is missing: {manifest_path}"
        )
    try:
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"checkpoint-pair manifest is not valid JSON: {manifest_path}"
        ) from exc
    if not isinstance(record, Mapping):
        raise ValueError(
            f"checkpoint-pair manifest must be a JSON object: {manifest_path}"
        )
    if record.get("format") != _PAIR_MANIFEST_FORMAT:
        raise ValueError(f"unsupported checkpoint-pair format: {manifest_path}")
    if _manifest_integer(record.get("schema_version"), name="schema version") != (
        _PAIR_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported checkpoint-pair schema: {manifest_path}")
    if _manifest_integer(record.get("version"), name="policy version") != (
        policy_version
    ):
        raise ValueError(
            "checkpoint-pair manifest policy version does not match checkpoint"
        )
    policy_fingerprint = _validate_pair_artifact_record(
        record.get("policy"),
        label="policy",
        expected_path=checkpoint_path,
    )
    state_record = record.get("training_state")
    if not isinstance(state_record, Mapping):
        raise ValueError("checkpoint-pair training state record must be a mapping")
    _validate_pair_artifact_record(
        state_record,
        label="training state",
        expected_path=state_path,
    )
    bound_policy_sha256 = _validated_sha256(
        state_record.get("policy_sha256"),
        name="checkpoint-pair training-state policy",
    )
    if bound_policy_sha256 != policy_fingerprint[1]:
        raise ValueError(
            "checkpoint-pair training state is bound to a different policy"
        )
    return policy_fingerprint


def _checkpoint_pair_manifest_path(
    checkpoint_path: Path,
    *,
    policy_version: int,
) -> Path:
    """Infer the immutable pair manifest next to weights and resume directories."""
    weights_dir = checkpoint_path.parent
    if weights_dir.name != "weights":
        raise ValueError(
            "cannot infer checkpoint-pair manifest from a checkpoint outside a "
            f"weights directory: {checkpoint_path}"
        )
    return (
        weights_dir.parent
        / "checkpoint_pairs"
        / f"checkpoint_pair_v{policy_version}.json"
    )


def _validate_pair_artifact_record(
    value: Any,
    *,
    label: str,
    expected_path: Path,
) -> tuple[int, str]:
    """Validate a pair record against selected bytes and canonical layout."""
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint-pair {label} record must be a mapping")
    raw_path = value.get("path")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise ValueError(f"checkpoint-pair {label} path must be absolute")
    recorded_path = Path(raw_path)
    if (recorded_path.parent.name, recorded_path.name) != (
        expected_path.parent.name,
        expected_path.name,
    ):
        raise ValueError(
            f"checkpoint-pair {label} layout does not match selected artifact"
        )
    expected_size = _manifest_integer(
        value.get("size_bytes"),
        name=f"{label} size",
    )
    if expected_size <= 0:
        raise ValueError(f"checkpoint-pair {label} size must be positive")
    expected_sha256 = _validated_sha256(
        value.get("sha256"),
        name=f"checkpoint-pair {label}",
    )
    actual_size, actual_sha256 = _stable_file_fingerprint(expected_path)
    if (actual_size, actual_sha256) != (expected_size, expected_sha256):
        raise ValueError(
            f"checkpoint-pair {label} fingerprint does not match selected artifact"
        )
    return expected_size, expected_sha256


def _manifest_integer(value: Any, *, name: str) -> int:
    """Read a strict integer from an immutable pair manifest."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"checkpoint-pair {name} must be an integer")
    return value


def _write_latest_pointer(
    directory: Path,
    *,
    schema_version: int,
    path: Path,
    policy_version: int,
    completed_iterations: int,
    total_optimizer_updates: int,
    policy_size_bytes: int,
    policy_sha256: str,
    resume_config_sha256: str,
) -> None:
    record = {
        "schema_version": schema_version,
        "path": str(path),
        "policy_version": policy_version,
        "completed_iterations": completed_iterations,
        "total_optimizer_updates": total_optimizer_updates,
        "policy_size_bytes": policy_size_bytes,
        "policy_sha256": policy_sha256,
        "resume_config_sha256": resume_config_sha256,
    }
    latest_path = directory / "latest.json"
    atomic_write_bytes(
        latest_path,
        json_payload(record),
        overwrite=True,
    )


def _prune_training_states(
    directory: Path,
    *,
    keep_last: int,
    retain_every_versions: int | None,
) -> None:
    versioned = [
        (version, path)
        for path in directory.glob("training_state_v*.pt")
        if (version := _state_version(path)) is not None
    ]
    ordered = sorted(versioned, reverse=True)
    protected = {version for version, _path in ordered[:keep_last]}
    if retain_every_versions is not None:
        protected.update(
            version
            for version, _path in ordered
            if version % retain_every_versions == 0
        )
    for version, path in ordered:
        if version in protected:
            continue
        path.unlink(missing_ok=True)


def _state_filename(policy_version: int) -> str:
    return f"training_state_v{policy_version}.pt"


def _state_version(path: Path) -> int | None:
    raw_version = path.stem.removeprefix("training_state_v")
    return int(raw_version) if raw_version.isdigit() else None


def _stable_file_fingerprint(path: Path) -> tuple[int, str]:
    """Return a streaming fingerprint and reject files changing during hashing."""
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    after = path.stat()
    before_signature = (before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise RuntimeError(f"checkpoint artifact changed while hashing: {path}")
    return after.st_size, digest.hexdigest()


def _validated_sha256(value: Any, *, name: str) -> str:
    normalized = str(value).strip().lower()
    if len(normalized) != _SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} SHA-256 must be a 64-character hex digest")
    return normalized


def _validated_optimizer_parameter_names(
    values: Sequence[Sequence[str]],
    *,
    optimizer: torch.optim.Optimizer,
) -> OptimizerParameterNames:
    names = _coerce_optimizer_parameter_names(values)
    expected_lengths = tuple(len(group["params"]) for group in optimizer.param_groups)
    if tuple(len(group) for group in names) != expected_lengths:
        raise ValueError("optimizer parameter names do not match parameter groups")
    flattened = [name for group in names for name in group]
    if len(set(flattened)) != len(flattened):
        raise ValueError("optimizer parameter names must be unique")
    return names


def _coerce_optimizer_parameter_names(value: Any) -> OptimizerParameterNames:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("optimizer parameter names must contain parameter groups")
    groups: list[tuple[str, ...]] = []
    for raw_group in value:
        if not isinstance(raw_group, Sequence) or isinstance(
            raw_group,
            (str, bytes),
        ):
            raise ValueError("optimizer parameter names must contain name sequences")
        group = tuple(str(name).strip() for name in raw_group)
        if any(not name for name in group):
            raise ValueError("optimizer parameter names must be non-empty")
        groups.append(group)
    return tuple(groups)
