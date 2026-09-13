"""Hydra-backed rollout and learner throughput profiler."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.vector_battle import DeckPairSampler, VectorBattlePool
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    AgentPolicyValueNet,
    build_agent_policy_value_net,
)
from ptcg_rl.profiling import StageTimer
from ptcg_rl.rl.advantage import GaeConfig
from ptcg_rl.rl.collection import (
    AutocastMode,
    RolloutPolicyKind,
    build_rollout_policy,
)
from ptcg_rl.rl.experience import TensorTrajectoryRecorder
from ptcg_rl.rl.learner import (
    LearnerBatchConfig,
    LearnerIterationResult,
    build_ppo_minibatches,
    run_ppo_iteration,
)
from ptcg_rl.rl.ppo import (
    AnchorStepLogitsCache,
    PpoConfig,
    load_frozen_anchor_model,
)
from ptcg_rl.rl.rollout import (
    RolloutActors,
    RolloutRunSummary,
    RolloutStepper,
    VectorPoolLike,
)

ProfilePoolFactory = Callable[
    [int, DeckPairSampler, StageTimer],
    AbstractContextManager[VectorPoolLike],
]


class RLProfileConfig(BaseModel):
    """Config for single-process RL throughput profiling."""

    model_config = ConfigDict(extra="forbid")

    output_path: Path | None = Path("outputs/rl/profile/summary.json")
    deck_path: Path = Path("data/sample_submission/deck.csv")
    checkpoint_path: Path | None = None
    anchor_checkpoint_path: Path | None = None
    device: str = "auto"
    seed: int = 0
    policy_kind: RolloutPolicyKind = "model"
    autocast: AutocastMode = "bf16"
    num_concurrent_games: int = 64
    total_finished_games: int | None = 8
    total_recorded_decisions: int | None = None
    max_iterations: int = 100_000
    temperature: float = 1.0
    learner_minibatch_size: int = 128
    learner_gradient_accumulation_steps: int = 1
    learner_ppo_epochs: int = 1
    learner_max_staleness: int = 1_000_000
    learner_learning_rate: float = 1.0e-5
    learner_final_learning_rate: float = 1.0e-6
    learner_fused_adamw: bool = True
    learner_shape_bucket_accumulation: bool = False
    learner_route_bucket_minibatches: bool = False
    model: AgentNetworkConfig = Field(default_factory=AgentNetworkConfig)
    gae: GaeConfig = Field(default_factory=GaeConfig)
    ppo: PpoConfig = Field(default_factory=PpoConfig)

    @field_validator(
        "num_concurrent_games",
        "max_iterations",
        "learner_minibatch_size",
        "learner_gradient_accumulation_steps",
        "learner_ppo_epochs",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject invalid profile counters."""
        if value <= 0:
            raise ValueError("profile counters must be positive")
        return value

    @field_validator("total_finished_games", "total_recorded_decisions")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject invalid optional rollout budgets."""
        if value is not None and value <= 0:
            raise ValueError("rollout budgets must be positive when set")
        return value

    @field_validator("learner_max_staleness")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject invalid staleness limits."""
        if value < 0:
            raise ValueError("learner_max_staleness must be non-negative")
        return value

    @field_validator("temperature")
    @classmethod
    def valid_temperature(cls, value: float) -> float:
        """Reject invalid sampling temperatures."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("temperature must be finite and positive")
        return value

    @field_validator("learner_learning_rate", "learner_final_learning_rate")
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject invalid learner learning rates."""
        if value <= 0.0:
            raise ValueError("learner learning rates must be positive")
        return value

    @model_validator(mode="after")
    def valid_rollout_budget(self) -> RLProfileConfig:
        """Require at least one rollout stopping condition."""
        if self.total_finished_games is None and self.total_recorded_decisions is None:
            raise ValueError(
                "total_finished_games or total_recorded_decisions is required"
            )
        return self


def run_rl_profile(
    config: RLProfileConfig,
    *,
    pool_factory: ProfilePoolFactory | None = None,
) -> dict[str, Any]:
    """Run rollout and learner profiling and write a JSON summary."""
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(_cuda_device_index(device))
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(_cuda_device_index(device))
    timer = StageTimer(synchronize=_sync_callback(device))

    rollout_start = time.perf_counter()
    run_summary, trajectories = _profile_rollout(
        config,
        device=device,
        timer=timer,
        pool_factory=pool_factory,
    )
    _synchronize_if_cuda(device)
    rollout_elapsed = time.perf_counter() - rollout_start

    learner_start = time.perf_counter()
    learner_summary = _profile_learner(
        config,
        device=device,
        timer=timer,
        trajectories=trajectories,
    )
    _synchronize_if_cuda(device)
    learner_elapsed = time.perf_counter() - learner_start
    learner_summary["elapsed_seconds"] = learner_elapsed
    learner_summary["samples_per_second"] = _rate(
        int(learner_summary.get("sample_passes", 0)),
        learner_elapsed,
    )

    output = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.model_dump(mode="json"),
        "device": str(device),
        "rollout": {
            "elapsed_seconds": rollout_elapsed,
            "run": _run_summary_dict(run_summary),
            "rates": {
                "recorded_decisions_per_second": _rate(
                    run_summary.recorded_decisions,
                    rollout_elapsed,
                ),
                "policy_actions_per_second": _rate(
                    run_summary.policy_actions,
                    rollout_elapsed,
                ),
                "engine_submissions_per_second": _rate(
                    (
                        run_summary.forced_actions
                        + run_summary.scripted_actions
                        + run_summary.policy_actions
                    ),
                    rollout_elapsed,
                ),
            },
            "completed_trajectories": len(trajectories),
            "completed_decisions": sum(
                trajectory.decision_count for trajectory in trajectories
            ),
        },
        "learner": learner_summary,
        "stage_timings": timer.summary(),
        "measurement_synchronizations": timer.synchronizations,
        "cuda": _cuda_summary(device),
    }
    _write_summary(config.output_path, output)
    return output


def _profile_rollout(
    config: RLProfileConfig,
    *,
    device: torch.device,
    timer: StageTimer,
    pool_factory: ProfilePoolFactory | None,
) -> tuple[RolloutRunSummary, tuple[Any, ...]]:
    deck = tuple(records.read_deck(records.repo_path(config.deck_path)))
    policy = build_rollout_policy(
        policy_kind=config.policy_kind,
        checkpoint_path=config.checkpoint_path,
        model_config=config.model,
        device=device,
        autocast=config.autocast,
    )
    recorder = TensorTrajectoryRecorder()
    factory = pool_factory or _default_pool_factory
    with factory(config.num_concurrent_games, lambda: (deck, deck), timer) as pool:
        stepper = RolloutStepper(
            pool=pool,
            actors=RolloutActors(mode="self_play", candidate_policy=policy),
            recorder=recorder,
            temperature=config.temperature,
            device=device,
            timer=timer,
        )
        _synchronize_if_cuda(device)
        run_summary = stepper.run_until(
            total_finished_games=config.total_finished_games,
            total_recorded_decisions=config.total_recorded_decisions,
            max_iterations=config.max_iterations,
        )
    return (run_summary, recorder.pop_completed())


def _profile_learner(
    config: RLProfileConfig,
    *,
    device: torch.device,
    timer: StageTimer,
    trajectories: tuple[Any, ...],
) -> dict[str, Any]:
    if not trajectories:
        return {
            "status": "skipped",
            "reason": "no_completed_trajectories",
            "updates": 0,
            "sample_passes": 0,
        }
    model = _load_profile_model(config, device=device)
    anchor_model = (
        load_frozen_anchor_model(
            config.anchor_checkpoint_path,
            fallback_config=model.config,
            device=device,
        )
        if config.anchor_checkpoint_path is not None
        else None
    )
    optimizer_kwargs: dict[str, Any] = {}
    if config.learner_fused_adamw and device.type == "cuda":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learner_learning_rate,
        weight_decay=0.0,
        **optimizer_kwargs,
    )
    lr_scheduler_steps = _profile_lr_scheduler_steps(config, trajectories)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lr_scheduler_steps,
        eta_min=config.learner_final_learning_rate,
    )
    update_results: list[LearnerIterationResult] = []
    batch_config = LearnerBatchConfig(
        microbatch_size=config.learner_minibatch_size,
        gradient_accumulation_steps=config.learner_gradient_accumulation_steps,
        max_staleness=config.learner_max_staleness,
        shuffle=True,
        drop_last=False,
        shape_bucket_accumulation=config.learner_shape_bucket_accumulation,
        route_bucket_minibatches=config.learner_route_bucket_minibatches,
        seed=config.seed,
        gae=config.gae,
    )
    batch_result = build_ppo_minibatches(
        trajectories,
        current_policy_version=0,
        config=batch_config,
        device=None,
        timer=timer,
    )
    anchor_cache = AnchorStepLogitsCache() if anchor_model is not None else None
    for _epoch in range(config.learner_ppo_epochs):
        result = run_ppo_iteration(
            model=model,
            optimizer=optimizer,
            trajectories=trajectories,
            current_policy_version=0,
            batch_config=batch_config,
            ppo_config=config.ppo,
            anchor_model=anchor_model,
            anchor_cache=anchor_cache,
            first_update_index=sum(len(item.updates) for item in update_results),
            device=device,
            timer=timer,
            lr_scheduler=lr_scheduler,
            batch_result=batch_result,
        )
        update_results.append(result)
        if result.updates and result.updates[-1].should_stop:
            break
    final_result = update_results[-1]
    sample_passes = sum(
        update.sample_count
        for result in update_results
        for update in result.updates
    )
    return {
        "status": "completed",
        "epochs_requested": config.learner_ppo_epochs,
        "epochs_run": len(update_results),
        "updates": sum(len(result.updates) for result in update_results),
        "sample_passes": sample_passes,
        "autocast": config.ppo.autocast,
        "update_diagnostics": _update_diagnostics(update_results),
        "batch_stats": final_result.batch_result.stats.__dict__,
        "optimizer": {
            "fused": bool(optimizer_kwargs.get("fused", False)),
            "learning_rate": config.learner_learning_rate,
            "final_learning_rate": config.learner_final_learning_rate,
            "current_learning_rate": _current_learning_rate(optimizer),
            "lr_scheduler": {
                "type": "cosine",
                "planned_steps": lr_scheduler_steps,
                "steps": int(lr_scheduler.last_epoch),
            },
        },
        "target_kl_stopped": any(
            bool(result.updates) and result.updates[-1].should_stop
            for result in update_results
        ),
    }


def _update_diagnostics(
    update_results: Sequence[LearnerIterationResult],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    update_index = 0
    for epoch, result in enumerate(update_results):
        for update in result.updates:
            breakdown = update.breakdown
            diagnostics.append(
                {
                    "epoch": epoch,
                    "update": update_index,
                    "approx_kl": breakdown.approx_kl,
                    "anchor_kl": breakdown.anchor_kl,
                    "ratio_mean": breakdown.ratio_mean,
                    "ratio_p95": breakdown.ratio_p95,
                    "clip_fraction": breakdown.clip_fraction,
                    "entropy": breakdown.entropy,
                    "value_mean": breakdown.value_mean,
                    "engine_teacher_loss": (
                        None
                        if breakdown.engine_teacher_loss is None
                        else float(breakdown.engine_teacher_loss.detach().item())
                    ),
                    "engine_teacher_decisions": update.engine_teacher_decisions,
                    "critic_warmup": breakdown.critic_warmup,
                    "should_stop": update.should_stop,
                }
            )
            update_index += 1
    return diagnostics


def _default_pool_factory(
    num_games: int,
    deck_pair_sampler: DeckPairSampler,
    timer: StageTimer,
) -> AbstractContextManager[VectorPoolLike]:
    return VectorBattlePool(
        num_games,
        deck_pair_sampler,
        include_search_input=False,
        timer=timer,
    )


def _load_profile_model(
    config: RLProfileConfig,
    *,
    device: torch.device,
) -> AgentPolicyValueNet:
    checkpoint = _load_checkpoint(config.checkpoint_path)
    model_config = _checkpoint_model_config(checkpoint) or config.model
    model = build_agent_policy_value_net(model_config).to(device)
    if checkpoint is not None:
        incompatible = model.load_state_dict(
            _checkpoint_state_dict(checkpoint),
            strict=False,
        )
        allowed_missing = {
            "opponent_hand_head.weight",
            "opponent_hand_head.bias",
        } | LEGACY_STATE_ENCODER_MISSING_KEYS
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing - allowed_missing or unexpected:
            raise RuntimeError(
                "checkpoint state dict is incompatible with profile model"
            )
    model.train()
    return model


def _profile_lr_scheduler_steps(
    config: RLProfileConfig,
    trajectories: tuple[Any, ...],
) -> int:
    decision_count = sum(int(trajectory.decision_count) for trajectory in trajectories)
    microbatches_per_epoch = math.ceil(
        decision_count / config.learner_minibatch_size
    )
    updates_per_epoch = math.ceil(
        microbatches_per_epoch / config.learner_gradient_accumulation_steps
    )
    return max(1, config.learner_ppo_epochs * updates_per_epoch)


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _load_checkpoint(checkpoint_path: Path | None) -> Any | None:
    if checkpoint_path is None:
        return None
    return torch.load(records.repo_path(checkpoint_path), map_location="cpu")


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_lightning_model_prefix(cast(Mapping[str, Any], value))
        return _strip_lightning_model_prefix(cast(Mapping[str, Any], checkpoint))
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {
            str(key).removeprefix("model."): value for key, value in state_dict.items()
        }
    return state_dict


def _checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig | None:
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        model_config = full_config.get("model")
        if isinstance(model_config, Mapping):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _resolve_device(raw_device: str) -> torch.device:
    normalized = raw_device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "gpu":
        normalized = "cuda"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {raw_device}")
    return device


def _sync_callback(device: torch.device) -> Callable[[], None] | None:
    if device.type != "cuda":
        return None
    return lambda: torch.cuda.synchronize(_cuda_device_index(device))


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(_cuda_device_index(device))


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"not a CUDA device: {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _cuda_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "available": torch.cuda.is_available(),
            "device_name": None,
            "max_memory_allocated_mb": None,
            "max_memory_reserved_mb": None,
        }
    device_index = _cuda_device_index(device)
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(device_index),
        "max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device_index) / 1_000_000.0
        ),
        "max_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(device_index) / 1_000_000.0
        ),
    }


def _run_summary_dict(summary: RolloutRunSummary) -> dict[str, int]:
    return {
        "iterations": summary.iterations,
        "forced_actions": summary.forced_actions,
        "scripted_actions": summary.scripted_actions,
        "policy_actions": summary.policy_actions,
        "recorded_decisions": summary.recorded_decisions,
        "finished_games": summary.finished_games,
        "engine_submissions": (
            summary.forced_actions + summary.scripted_actions + summary.policy_actions
        ),
    }


def _write_summary(path: Path | None, summary: Mapping[str, Any]) -> None:
    if path is None:
        return
    resolved = records.repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rate(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return 0.0
    return float(count) / elapsed_seconds
