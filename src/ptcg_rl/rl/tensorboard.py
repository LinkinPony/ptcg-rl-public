"""Low-overhead TensorBoard scalar writers for RL training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from ptcg_rl.profiling import StageTimer
from ptcg_rl.rl.factual_metrics import aggregate_factual_updates
from ptcg_rl.rl.learner import LearnerIterationResult
from ptcg_rl.rl.macro_credit_metrics import aggregate_macro_credit_updates
from ptcg_rl.rl.planner_metrics import (
    aggregate_planner_updates,
    planner_imitation_metrics_summary,
    planner_update_metrics_summary,
)
from ptcg_rl.rl.ppo import PpoUpdateResult
from ptcg_rl.rl.teacher_metrics import aggregate_engine_teacher_updates

_ENGINE_TEACHER_RUNTIME_SCALARS = (
    ("engine_teacher_requests", "requests"),
    ("engine_teacher_targets", "targets"),
    ("engine_teacher_search_targets", "search_targets"),
    ("engine_teacher_eligible", "eligible"),
    ("engine_teacher_attempted", "attempted"),
    ("engine_teacher_emitted", "emitted"),
    ("engine_teacher_behavior_matches_online", "behavior_matches"),
    ("engine_teacher_worlds", "worlds"),
    ("engine_teacher_nodes", "nodes"),
    ("engine_teacher_step_calls", "step_calls"),
    ("engine_teacher_probability_skips", "probability_skips"),
    ("engine_teacher_attempt_cap_skips", "attempt_cap_skips"),
    ("engine_teacher_step_budget_skips", "step_budget_skips"),
    ("engine_teacher_deadline_expiries", "deadline_expiries"),
    ("engine_teacher_isolated_process", "isolated_processes"),
    ("engine_teacher_worker_starts", "worker_starts"),
    ("engine_teacher_worker_restarts", "worker_restarts"),
    ("engine_teacher_worker_startup_failures", "worker_startup_failures"),
    ("engine_teacher_worker_hard_timeouts", "worker_hard_timeouts"),
    ("engine_teacher_worker_crashes", "worker_crashes"),
    ("engine_teacher_worker_forced_terminations", "worker_forced_terminations"),
    ("attempt_rate", "attempt_rate"),
    ("emit_rate", "emit_rate"),
    ("search_target_rate", "search_target_rate"),
    ("mean_attempt_coverage", "mean_attempt_coverage"),
    ("mean_emitted_confidence", "mean_emitted_confidence"),
    ("seconds", "seconds"),
    ("max_step_seconds", "max_step_seconds"),
)


@dataclass
class TensorboardMetricWriter:
    """Small optional TensorBoard scalar writer used off the hot path."""

    writer: Any | None
    log_dir: Path | None

    @classmethod
    def create(
        cls,
        log_dir: Path,
        *,
        enabled: bool,
        flush_seconds: float,
    ) -> TensorboardMetricWriter:
        """Create a TensorBoard writer when enabled and available."""
        if not enabled:
            return cls(writer=None, log_dir=None)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            return cls(writer=None, log_dir=None)
        log_dir.mkdir(parents=True, exist_ok=True)
        summary_writer = cast(Any, SummaryWriter)
        return cls(
            writer=summary_writer(str(log_dir), flush_secs=flush_seconds),
            log_dir=log_dir,
        )

    @property
    def enabled(self) -> bool:
        """Return whether scalar writes will be persisted."""
        return self.writer is not None

    def add_scalar(
        self,
        tag: str,
        value: Any,
        step: int,
        *,
        walltime: float | None = None,
    ) -> None:
        """Write one finite numeric scalar if TensorBoard is enabled."""
        if self.writer is None:
            return
        numeric = _tensorboard_scalar(value)
        if numeric is None:
            return
        self.writer.add_scalar(tag, numeric, step, walltime=walltime)

    def flush(self) -> None:
        """Flush buffered events without forcing callers to know writer internals."""
        if self.writer is not None:
            self.writer.flush()

    def close(self) -> None:
        """Close the underlying writer if one was created."""
        if self.writer is not None:
            self.writer.close()


def tensorboard_root_dir(output_path: Path) -> Path:
    """Return the TensorBoard root directory for one RL run."""
    return output_path / "tensorboard"


def write_learner_tensorboard(
    writer: TensorboardMetricWriter,
    *,
    step: int,
    iteration: Mapping[str, Any],
    update_results: Sequence[LearnerIterationResult],
    optimizer: torch.optim.Optimizer,
    timer: StageTimer,
) -> None:
    """Write one learner-window TensorBoard sample."""
    if not writer.enabled:
        return
    learner = _as_mapping(iteration.get("learner"))
    batch_stats = _as_mapping(learner.get("batch_stats"))
    total_decisions = _as_float(batch_stats.get("total_decisions"))
    stale_decisions = _as_float(batch_stats.get("stale_decisions"))
    sample_passes = sum(
        result.batch_result.stats.kept_decisions for result in update_results
    )
    updates = _learner_update_count(update_results)

    writer.add_scalar("learner/window_iteration", iteration.get("iteration"), step)
    writer.add_scalar(
        "learner/source_policy_version",
        iteration.get("policy_version"),
        step,
    )
    writer.add_scalar(
        "learner/published_version",
        learner.get("published_version"),
        step,
    )
    writer.add_scalar("learner/updates", updates, step)
    writer.add_scalar("learner/sample_passes", sample_passes, step)
    writer.add_scalar(
        "learner/samples_per_update",
        sample_passes / updates if updates > 0 else None,
        step,
    )
    writer.add_scalar("batch/trajectories", iteration.get("trajectories"), step)
    writer.add_scalar(
        "batch/trajectory_decisions",
        iteration.get("trajectory_decisions"),
        step,
    )
    for key in (
        "total_decisions",
        "kept_decisions",
        "stale_decisions",
        "minibatches",
        "trajectories",
    ):
        writer.add_scalar(f"batch/{key}", batch_stats.get(key), step)
    writer.add_scalar(
        "batch/stale_fraction",
        stale_decisions / total_decisions if total_decisions > 0.0 else 0.0,
        step,
    )

    updates_for_window = _ppo_updates(update_results)
    teacher_metrics = aggregate_engine_teacher_updates(updates_for_window)
    factual_metrics = aggregate_factual_updates(updates_for_window)
    macro_metrics = aggregate_macro_credit_updates(updates_for_window)
    planner_metrics = aggregate_planner_updates(updates_for_window)
    last_update = _last_ppo_update(update_results)
    if last_update is not None:
        for key, value in _ppo_update_summary(last_update).items():
            tag = (
                f"ppo/last_{key}"
                if key
                in {
                    "engine_teacher_decisions",
                    "engine_teacher_loss",
                    "factual_decisions",
                    "factual_effect_loss",
                    "factual_successor_loss",
                    "macro_roots",
                    "macro_conditional_loss",
                    "macro_expected_loss",
                    "planner_decisions",
                    "planner_applicable_decisions",
                    "planner_applicable_fraction",
                    "candidate_rerank_loss",
                    "proposal_distillation_loss",
                    "root_information_value_rows",
                    "root_information_value_loss",
                    "planner_stale_decisions",
                    "planner_inexact_decisions",
                    "planner_incomplete_grid_decisions",
                    "planner_identity_invalid_decisions",
                    "planner_censored_support_decisions",
                    "planner_exhaustive_support_decisions",
                    "planner_clipped_candidates",
                    "planner_candidate_count",
                    "planner_clipped_candidate_fraction",
                }
                else f"ppo/{key}"
            )
            writer.add_scalar(tag, value, step)
    writer.add_scalar(
        "ppo/engine_teacher_decisions",
        teacher_metrics.decisions,
        step,
    )
    writer.add_scalar("ppo/engine_teacher_loss", teacher_metrics.loss, step)
    writer.add_scalar("ppo/factual_decisions", factual_metrics.decisions, step)
    writer.add_scalar("ppo/factual_effect_loss", factual_metrics.effect_loss, step)
    writer.add_scalar(
        "ppo/factual_successor_loss",
        factual_metrics.successor_loss,
        step,
    )
    writer.add_scalar("ppo/macro_roots", macro_metrics.roots, step)
    writer.add_scalar(
        "ppo/macro_conditional_loss",
        macro_metrics.conditional_loss,
        step,
    )
    writer.add_scalar(
        "ppo/macro_expected_loss",
        macro_metrics.expected_loss,
        step,
    )
    for key, value in planner_update_metrics_summary(planner_metrics).items():
        writer.add_scalar(f"ppo/{key}", value, step)
    early_stop = _learner_early_stop_summary(update_results)
    writer.add_scalar("ppo/early_stopped", early_stop.get("early_stopped"), step)
    writer.add_scalar("ppo/early_stop_epoch", early_stop.get("epoch"), step)
    writer.add_scalar(
        "optimizer/learning_rate",
        _current_learning_rate(optimizer),
        step,
    )

    for stage, timing in timer.summary().items():
        timing_data = _as_mapping(timing)
        writer.add_scalar(
            f"learner_timing/{stage}/seconds",
            timing_data.get("seconds"),
            step,
        )
        writer.add_scalar(
            f"learner_timing/{stage}/count",
            timing_data.get("count"),
            step,
        )
        writer.add_scalar(
            f"learner_timing/{stage}/mean_ms",
            timing_data.get("mean_ms"),
            step,
        )
    writer.flush()


def write_runtime_tensorboard(
    writer: TensorboardMetricWriter,
    *,
    summary: Mapping[str, Any],
    inference: Mapping[str, Any] | None,
    actor_queue: Mapping[str, Any] | None,
    step: int,
) -> None:
    """Write one runtime-monitor TensorBoard sample."""
    if not writer.enabled:
        return
    last_sample = _as_mapping(summary.get("last_sample"))
    queues = _as_mapping(last_sample.get("queues"))
    gpu = _as_mapping(last_sample.get("gpu"))
    learner = _as_mapping(last_sample.get("learner"))
    inference_process = _as_mapping(last_sample.get("inference"))
    actors = _as_sequence(last_sample.get("actors"))
    actor_cpu_percent = sum(
        _as_float(_as_mapping(actor).get("cpu_percent")) for actor in actors
    )
    actors_alive = sum(
        1 for actor in actors if _as_mapping(actor).get("exitcode") is None
    )

    writer.add_scalar("runtime/sample_count", summary.get("sample_count"), step)
    writer.add_scalar("runtime/gpu_sample_count", summary.get("gpu_sample_count"), step)
    writer.add_scalar(
        "runtime/max_gpu_memory_used_mb",
        summary.get("max_gpu_memory_used_mb"),
        step,
    )
    for name, value in queues.items():
        writer.add_scalar(f"queue/{name}", value, step)
    for key in (
        "memory_used_mb",
        "memory_free_mb",
        "memory_total_mb",
        "utilization_percent",
        "memory_utilization_percent",
        "power_watts",
        "temperature_celsius",
    ):
        writer.add_scalar(f"gpu/{key}", gpu.get(key), step)
    writer.add_scalar("process/learner_rss_mb", learner.get("rss_mb"), step)
    writer.add_scalar("process/learner_cpu_percent", learner.get("cpu_percent"), step)
    writer.add_scalar("process/inference_rss_mb", inference_process.get("rss_mb"), step)
    writer.add_scalar(
        "process/inference_cpu_percent",
        inference_process.get("cpu_percent"),
        step,
    )
    writer.add_scalar("process/actors_alive", actors_alive, step)
    writer.add_scalar("process/actors_cpu_percent", actor_cpu_percent, step)

    if inference is not None:
        for key in (
            "decisions",
            "decisions_per_second",
            "requests",
            "responses",
            "policy_batches",
            "policy_batch_p50",
            "policy_batch_p95",
            "request_latency_p50_ms",
            "request_latency_p95_ms",
            "current_weight_version",
            "expired_teacher_requests",
            "expired_teacher_decisions",
        ):
            writer.add_scalar(f"inference/{key}", inference.get(key), step)
        server_stage_fraction = _as_mapping(inference.get("server_stage_fraction"))
        for stage, value in server_stage_fraction.items():
            writer.add_scalar(f"inference_stage_fraction/{stage}", value, step)

    if actor_queue is not None:
        for key in (
            "queued_trajectories",
            "queued_trajectory_decisions",
            "queue_full_retries",
            "queue_transfer_seconds",
            "mean_queue_transfer_seconds_per_trajectory",
            "actor_elapsed_seconds",
            "rollout_policy_wait_seconds",
            "rollout_policy_wait_fraction",
        ):
            writer.add_scalar(f"actors/{key}", actor_queue.get(key), step)
        for stage, timing in _as_mapping(
            actor_queue.get("rollout_stage_timings")
        ).items():
            timing_data = _as_mapping(timing)
            writer.add_scalar(
                f"actor_timing/{stage}/seconds",
                timing_data.get("seconds"),
                step,
            )
            writer.add_scalar(
                f"actor_timing/{stage}/mean_ms",
                timing_data.get("mean_ms"),
                step,
            )
        rollout_features = _as_mapping(actor_queue.get("rollout_features"))
        if "live_game_count" in rollout_features:
            for key in (
                "live_game_count",
                "live_game_steps_max",
                "max_live_game_steps_seen",
            ):
                writer.add_scalar(
                    f"actors/{key}",
                    rollout_features.get(key),
                    step,
                )
        if "stale_recurrent_recycle_polls" in rollout_features:
            for key in (
                "stale_recurrent_recycle_polls",
                "stale_recurrent_candidate_sequences_examined",
                "stale_recurrent_games_recycled",
                "stale_recurrent_games_deferred_pending_evidence",
                "stale_recurrent_sequences_released",
                "stale_recurrent_last_learner_version",
                "stale_recurrent_last_served_version",
                "stale_recurrent_oldest_candidate_policy_version",
                "stale_recurrent_max_candidate_version_age",
            ):
                writer.add_scalar(
                    f"actors/{key}",
                    rollout_features.get(key),
                    step,
                )
        teacher = _as_mapping(rollout_features.get("engine_teacher"))
        for source, tag in _ENGINE_TEACHER_RUNTIME_SCALARS:
            writer.add_scalar(f"engine_teacher/{tag}", teacher.get(source), step)

    for phase, phase_summary in _as_mapping(summary.get("gpu_phase_summary")).items():
        phase_data = _as_mapping(phase_summary)
        for key in (
            "gpu_power_occupancy_fraction",
            "mean_power_watts",
            "mean_utilization_percent",
            "inference_decisions_per_second",
        ):
            writer.add_scalar(f"gpu_phase/{phase}/{key}", phase_data.get(key), step)
    writer.flush()


def _learner_update_count(
    update_results: Sequence[LearnerIterationResult],
) -> int:
    return sum(len(result.updates) for result in update_results)


def _ppo_updates(
    update_results: Sequence[LearnerIterationResult],
) -> tuple[PpoUpdateResult, ...]:
    return tuple(update for result in update_results for update in result.updates)


def _last_ppo_update(
    update_results: Sequence[LearnerIterationResult],
) -> PpoUpdateResult | None:
    for result in reversed(update_results):
        if result.updates:
            return result.updates[-1]
    return None


def _learner_early_stop_summary(
    update_results: Sequence[LearnerIterationResult],
) -> dict[str, Any]:
    update_index = 0
    for epoch, result in enumerate(update_results):
        for minibatch, update in enumerate(result.updates):
            if update.should_stop:
                return {
                    "early_stopped": True,
                    "epoch": epoch,
                    "minibatch": minibatch,
                    "iteration_update_index": update_index,
                    "approx_kl": update.breakdown.approx_kl,
                    "approx_kl_k3": update.breakdown.approx_kl_k3,
                    "kl_stop_baseline_k3": update.kl_stop_baseline_k3,
                    "kl_stop_delta_k3": update.kl_stop_delta_k3,
                    "kl_stop_threshold": update.kl_stop_threshold,
                }
            update_index += 1
    return {"early_stopped": False}


def _ppo_update_summary(update: PpoUpdateResult) -> dict[str, Any]:
    breakdown = update.breakdown
    planner_applicable_fraction = (
        update.planner_applicable_decisions / update.planner_decisions
        if update.planner_decisions > 0
        else 0.0
    )
    return {
        "approx_kl": breakdown.approx_kl,
        "approx_kl_k3": breakdown.approx_kl_k3,
        "kl_stop_baseline_k3": update.kl_stop_baseline_k3,
        "kl_stop_delta_k3": update.kl_stop_delta_k3,
        "kl_stop_threshold": update.kl_stop_threshold,
        "anchor_kl": breakdown.anchor_kl,
        "entropy": breakdown.entropy,
        "clip_fraction": breakdown.clip_fraction,
        "ratio_mean": breakdown.ratio_mean,
        "ratio_p95": breakdown.ratio_p95,
        "value_mean": breakdown.value_mean,
        "engine_teacher_loss": (
            None
            if breakdown.engine_teacher_loss is None
            else float(breakdown.engine_teacher_loss.detach().item())
        ),
        "engine_teacher_decisions": update.engine_teacher_decisions,
        "factual_effect_loss": (
            None
            if breakdown.factual_effect_loss is None
            else float(breakdown.factual_effect_loss.detach().item())
        ),
        "factual_successor_loss": (
            None
            if breakdown.factual_successor_loss is None
            else float(breakdown.factual_successor_loss.detach().item())
        ),
        "factual_decisions": update.factual_decisions,
        "macro_conditional_loss": (
            None
            if breakdown.macro_conditional_loss is None
            else float(breakdown.macro_conditional_loss.detach().item())
        ),
        "macro_expected_loss": (
            None
            if breakdown.macro_expected_loss is None
            else float(breakdown.macro_expected_loss.detach().item())
        ),
        "macro_roots": update.macro_roots,
        "candidate_rerank_loss": (
            None
            if breakdown.candidate_rerank_loss is None
            else float(breakdown.candidate_rerank_loss.detach().item())
        ),
        "proposal_distillation_loss": (
            None
            if breakdown.proposal_distillation_loss is None
            else float(breakdown.proposal_distillation_loss.detach().item())
        ),
        "planner_decisions": update.planner_decisions,
        "planner_applicable_decisions": update.planner_applicable_decisions,
        "planner_applicable_fraction": planner_applicable_fraction,
        "root_information_value_loss": (
            None
            if breakdown.root_information_value_loss is None
            else float(breakdown.root_information_value_loss.detach().item())
        ),
        "root_information_value_rows": update.root_information_value_rows,
        **planner_imitation_metrics_summary(breakdown.planner_metrics),
        "grad_norm": update.grad_norm,
        "should_stop": update.should_stop,
        "critic_warmup": breakdown.critic_warmup,
        "transition_distillation": breakdown.transition_distillation,
        "transition_sequence_kl": _optional_loss_scalar(
            breakdown.transition_sequence_kl
        ),
        "transition_prefix_kl": _optional_loss_scalar(breakdown.transition_prefix_kl),
        "transition_count_kl": _optional_loss_scalar(breakdown.transition_count_kl),
        "transition_root_value_loss": _optional_loss_scalar(
            breakdown.transition_root_value_loss
        ),
        "transition_prefix_value_loss": _optional_loss_scalar(
            breakdown.transition_prefix_value_loss
        ),
        "transition_action_wdl_loss": _optional_loss_scalar(
            breakdown.transition_action_wdl_loss
        ),
        "transition_engine_return_loss": _optional_loss_scalar(
            breakdown.transition_engine_return_loss
        ),
    }


def _current_learning_rate(optimizer: torch.optim.Optimizer) -> float | None:
    if not optimizer.param_groups:
        return None
    return float(optimizer.param_groups[0]["lr"])


def _optional_loss_scalar(value: torch.Tensor | None) -> float | None:
    """Convert an optional detached loss to a logging scalar."""
    return None if value is None else float(value.detach().item())


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _tensorboard_scalar(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None
