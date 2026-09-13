"""Validation and I/O helpers for a DCCR-v4 checkpoint-pair transition."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.model import AgentNetworkConfig, build_agent_policy_value_net
from ptcg_rl.rl.checkpoint_pair import PublishedCheckpointPair
from ptcg_rl.rl.training import (
    RLTrainConfig,
    _build_lr_scheduler,
    _build_ppo_optimizer,
    _optimizer_parameter_names,
    _planned_lr_scheduler_steps,
    _resume_relevant_config_sha256,
    resolve_rl_train_output_dir,
)
from ptcg_rl.rl.training_resume import restore_training_progress


def converted_auxiliary_state(
    source: Mapping[str, Any],
    *,
    target_state: Mapping[str, Tensor],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Replace the target model while preserving declared auxiliary state."""
    converted = dict(source)
    raw_policy_iteration = source.get("amortized_policy_iteration")
    if not isinstance(raw_policy_iteration, Mapping):
        raise ValueError("source policy-iteration auxiliary state is missing")
    optimizer_updates = raw_policy_iteration.get("optimizer_updates")
    if (
        not isinstance(optimizer_updates, int)
        or isinstance(optimizer_updates, bool)
        or optimizer_updates < 0
    ):
        raise ValueError("source policy-iteration optimizer clock is invalid")
    pending = raw_policy_iteration.get("pending_targets", ())
    if not isinstance(pending, (tuple, list)):
        raise ValueError("source pending-target deque is invalid")
    converted["amortized_policy_iteration"] = {
        "schema_version": 2,
        "optimizer_updates": optimizer_updates,
        "target_model_state_dict": dict(target_state),
        "pending_targets": tuple(pending),
    }
    converted["ppo_optimizer_updates"] = 0
    other_keys = sorted(
        set(source).difference({"amortized_policy_iteration", "ppo_optimizer_updates"})
    )
    return converted, {
        "ppo_optimizer_updates": "reset_zero_new_branch",
        "pending_targets": f"preserved_exact_count_{len(pending)}",
        "other_auxiliary": (
            "preserved:" + ",".join(other_keys) if other_keys else "none"
        ),
    }


def verify_published_pair(
    config: RLTrainConfig,
    *,
    pair: PublishedCheckpointPair,
    policy_version: int,
) -> None:
    """Strict-load both models and exact-resume the freshly published pair."""
    checkpoint = torch.load(pair.policy.path, map_location="cpu", weights_only=False)
    if checkpoint_model_config(checkpoint) != config.model:
        raise RuntimeError("published transition checkpoint config changed")
    model = build_agent_policy_value_net(config.model).cpu()
    model.load_state_dict(checkpoint_state_dict(checkpoint), strict=True)
    optimizer = _build_ppo_optimizer(
        config,
        model=model,
        device=torch.device("cpu"),
    )
    planned_steps = _planned_lr_scheduler_steps(config)
    scheduler = _build_lr_scheduler(
        optimizer,
        config=config,
        planned_steps=planned_steps,
    )
    progress = restore_training_progress(
        config=config.resume,
        checkpoint_path=pair.policy.path,
        policy_version=policy_version,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        planned_scheduler_steps=planned_steps,
        resume_config_sha256=_resume_relevant_config_sha256(config),
        optimizer_parameter_names=_optimizer_parameter_names(model, optimizer),
    )
    if progress.completed_iterations != 0 or progress.total_optimizer_updates != 0:
        raise RuntimeError("published transition pair did not reset branch clocks")
    if progress.ppo_optimizer_updates != 0:
        raise RuntimeError("published transition pair did not reset its PPO clock")
    sidecar = torch.load(
        pair.training_state_path,
        map_location="cpu",
        weights_only=False,
    )
    auxiliary = sidecar.get("auxiliary_state") if isinstance(sidecar, Mapping) else None
    policy_iteration = (
        auxiliary.get("amortized_policy_iteration")
        if isinstance(auxiliary, Mapping)
        else None
    )
    target_state = (
        policy_iteration.get("target_model_state_dict")
        if isinstance(policy_iteration, Mapping)
        else None
    )
    if not isinstance(target_state, Mapping):
        raise RuntimeError("published transition sidecar has no target model")
    model.load_state_dict(tensor_state_dict(target_state, label="published target"))


def validate_target_profile(
    config: RLTrainConfig,
    *,
    source_policy_path: Path,
    output_dir: Path,
    policy_version: int,
) -> None:
    """Bind exact-resume paths, output, and dense teacher to the new branch."""
    if config.resume.mode != "resume":
        raise ValueError("converted v4 pair must launch through exact resume mode")
    if config.registry_transition is not None:
        raise ValueError(
            "v4 topology conversion is not a same-schema registry transition"
        )
    resolved_output = deck_records.repo_path(
        resolve_rl_train_output_dir(config)
    ).resolve()
    if resolved_output != output_dir:
        raise ValueError(
            "target profile output directory differs from transition output"
        )
    expected_policy = output_dir / "weights" / f"policy_v{policy_version}.pt"
    expected_state = output_dir / "resume" / f"training_state_v{policy_version}.pt"
    if config.checkpoint_path is None or (
        deck_records.repo_path(config.checkpoint_path).resolve() != expected_policy
    ):
        raise ValueError("target profile checkpoint_path is not the converted policy")
    if config.resume.state_path is None or (
        deck_records.repo_path(config.resume.state_path).resolve() != expected_state
    ):
        raise ValueError("target profile resume state is not the converted sidecar")
    if config.anchor_checkpoint_path is None or (
        deck_records.repo_path(config.anchor_checkpoint_path).resolve()
        != source_policy_path
    ):
        raise ValueError("target profile anchor is not the immutable dense source")


def validate_source_sidecar(
    payload: Mapping[str, Any],
    *,
    source_policy_path: Path,
    source_policy_sha256: str,
    policy_version: int,
) -> None:
    """Require the sidecar's exact policy binding and valid source clocks."""
    expected = {
        "policy_version": policy_version,
        "policy_sha256": source_policy_sha256,
        "policy_size_bytes": source_policy_path.stat().st_size,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"source learner-state binding mismatch: {key}")
    for key in (
        "completed_iterations",
        "total_optimizer_updates",
        "planned_scheduler_steps",
    ):
        value = payload.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"source learner-state counter is invalid: {key}")


def checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig:
    """Read the portable model config from a policy checkpoint."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("source checkpoint must be a mapping")
    raw = checkpoint.get("model_config")
    if not isinstance(raw, Mapping):
        raise ValueError("source checkpoint has no portable model_config")
    return AgentNetworkConfig.model_validate(raw)


def checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Tensor]:
    """Read a tensor-only model state from a supported checkpoint envelope."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must be a mapping")
    for key in ("model_state_dict", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return tensor_state_dict(value, label="checkpoint model")
    return tensor_state_dict(checkpoint, label="checkpoint model")


def tensor_state_dict(value: Mapping[Any, Any], *, label: str) -> Mapping[str, Tensor]:
    """Validate a nonempty string-to-tensor state mapping without copying tensors."""
    state: dict[str, Tensor] = {}
    for key, tensor in value.items():
        if not isinstance(key, str) or not isinstance(tensor, Tensor):
            raise ValueError(f"{label} state must contain named tensors only")
        state[key] = tensor
    if not state:
        raise ValueError(f"{label} state is empty")
    return state


def checkpoint_version(path: Path) -> int:
    """Parse the immutable version from a policy filename."""
    raw = path.stem.removeprefix("policy_v")
    if not raw.isdigit():
        raise ValueError("source policy filename must be policy_v<version>.pt")
    return int(raw)


def resolved_file(path: Path, *, label: str) -> Path:
    """Resolve a repository path and require one existing regular file."""
    resolved = deck_records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


def require_file_sha256(path: Path, expected: str, *, label: str) -> None:
    """Reject immutable transition input bytes that differ from preregistration."""
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: {actual} != {expected}")


__all__ = [
    "checkpoint_model_config",
    "checkpoint_state_dict",
    "checkpoint_version",
    "converted_auxiliary_state",
    "require_file_sha256",
    "resolved_file",
    "tensor_state_dict",
    "validate_source_sidecar",
    "validate_target_profile",
    "verify_published_pair",
]
