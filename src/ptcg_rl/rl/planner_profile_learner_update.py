"""PPO configuration and execution for the fixed learner-kernel workload."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_config import IntegratedPlannerProfileConfig
from ptcg_rl.evaluation.planner_profile_workloads import (
    PlannerLearnerWorkloadConfig,
)
from ptcg_rl.model.network import AgentPolicyValueNet
from ptcg_rl.rl.learner import (
    LearnerBatchConfig,
    LearnerBatchResult,
    reshuffle_ppo_minibatches,
    run_ppo_iteration,
)
from ptcg_rl.rl.planner_profile_inference import load_profile_model
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity
from ptcg_rl.rl.ppo import AnchorStepLogitsCache, PpoConfig, PpoUpdateResult

_FIXED_KERNEL_ROWS = 1_024
_FIXED_MICROBATCH_SIZE = 512
_FIXED_ACCUMULATION_STEPS = 2
_FIXED_PPO_EPOCHS = 2


@dataclass(frozen=True, slots=True)
class ProfileKernelUpdateResult:
    """Actual counters returned by PPO for one two-epoch kernel cycle."""

    next_update_index: int
    optimizer_steps: int
    ppo_rows: int
    factual_rows: int
    planner_rows: int
    planner_applicable_rows: int
    root_value_rows: int


def validate_fixed_kernel_contract(workload: PlannerLearnerWorkloadConfig) -> None:
    """Reject geometry or coefficients that no longer define this benchmark."""
    geometry = (
        workload.kernel_rows_per_update,
        workload.microbatch_size,
        workload.gradient_accumulation_steps,
        workload.ppo_epochs,
    )
    if geometry != (
        _FIXED_KERNEL_ROWS,
        _FIXED_MICROBATCH_SIZE,
        _FIXED_ACCUMULATION_STEPS,
        _FIXED_PPO_EPOCHS,
    ):
        raise ValueError("profile learner kernel geometry must remain 1024/512x2/2")
    if workload.engine_teacher_coefficient != 0.0:
        raise ValueError("profile learner kernel requires the retired teacher off")
    active_coefficients = (
        workload.factual_effect_coefficient,
        workload.factual_successor_coefficient,
        workload.root_information_value_coefficient,
        workload.candidate_rerank_coefficient,
        workload.proposal_distillation_coefficient,
        workload.anchor_coefficient,
    )
    if any(value <= 0.0 for value in active_coefficients):
        raise ValueError("profile learner kernel requires every active objective")


def build_profile_kernel_batch_config(
    train_config: Any,
    *,
    workload: PlannerLearnerWorkloadConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
) -> LearnerBatchConfig:
    """Build the production-shaped collation contract for fixed kernel rows."""
    validate_fixed_kernel_contract(workload)
    return LearnerBatchConfig(
        microbatch_size=workload.microbatch_size,
        gradient_accumulation_steps=workload.gradient_accumulation_steps,
        max_decisions=workload.kernel_rows_per_update,
        max_staleness=workload.max_policy_age,
        staleness_scope=train_config.learner.staleness_scope,
        shuffle=True,
        shuffle_each_epoch=train_config.learner.shuffle_each_epoch,
        drop_last=False,
        pin_memory=train_config.learner.pin_memory,
        non_blocking_transfer=train_config.learner.non_blocking_transfer,
        copy_stream=train_config.learner.copy_stream,
        shape_bucket_accumulation=train_config.learner.shape_bucket_accumulation,
        seed=int(train_config.seed),
        require_deck_context=True,
        schema9=resolved.schema9_learner_config(),
        gae=train_config.gae,
    )


def build_profile_kernel_ppo_config(
    training_payload: Mapping[str, Any],
    workload: PlannerLearnerWorkloadConfig,
) -> PpoConfig:
    """Apply all fixed learner-kernel objective coefficients."""
    raw = training_payload.get("ppo")
    if not isinstance(raw, Mapping):
        raise ValueError("profile checkpoint has no PPO config")
    config = PpoConfig.model_validate(raw).model_copy(
        update={
            "engine_teacher_coef": workload.engine_teacher_coefficient,
            "factual_effect_coef": workload.factual_effect_coefficient,
            "factual_successor_coef": workload.factual_successor_coefficient,
            "root_information_value_coef": (
                workload.root_information_value_coefficient
            ),
            "candidate_rerank_coef": workload.candidate_rerank_coefficient,
            "proposal_distillation_coef": (workload.proposal_distillation_coefficient),
            "kl_anchor_coef": workload.anchor_coefficient,
            "planner_imitation_max_policy_age": workload.max_policy_age,
            "planner_target_ratio_clip": workload.planner_target_ratio_clip,
        }
    )
    if config.target_kl_early_stop is not None:
        raise ValueError("fixed learner kernel does not permit early optimizer stops")
    return config


def load_fixed_profile_anchor(
    workload: PlannerLearnerWorkloadConfig,
    *,
    campaign: IntegratedPlannerProfileConfig,
    device: str,
) -> AgentPolicyValueNet:
    """Load the immutable raw v27470 model as the KL anchor."""
    if workload.anchor_checkpoint_sha256 != campaign.expected_checkpoint_sha256:
        raise ValueError("profile learner anchor is not the fixed raw checkpoint")
    actual = file_sha256(workload.anchor_checkpoint_path)
    if actual != workload.anchor_checkpoint_sha256:
        raise ValueError("profile learner anchor fingerprint differs from config")
    anchor = load_profile_model(
        workload.anchor_checkpoint_path,
        device=device,
        expected_model_fingerprint=campaign.expected_model_fingerprint,
    )
    anchor.requires_grad_(False)
    anchor.eval()
    return anchor


def run_profile_kernel_update(
    *,
    model: AgentPolicyValueNet,
    anchor: AgentPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    batch_result: LearnerBatchResult,
    batch_config: LearnerBatchConfig,
    ppo_config: PpoConfig,
    ppo_epochs: int,
    policy_version: int,
    kernel_cycle_index: int,
    first_update_index: int,
) -> ProfileKernelUpdateResult:
    """Run one discarded-model kernel cycle and validate actual PPO counters."""
    anchor_cache = AnchorStepLogitsCache()
    update_index = first_update_index
    updates: list[PpoUpdateResult] = []
    for epoch in range(ppo_epochs):
        epoch_batches = reshuffle_ppo_minibatches(
            batch_result,
            config=batch_config,
            seed=(batch_config.seed + kernel_cycle_index * ppo_epochs + epoch),
            device=None,
        )
        result = run_ppo_iteration(
            model=model,
            optimizer=optimizer,
            trajectories=(),
            current_policy_version=policy_version,
            batch_config=batch_config,
            ppo_config=ppo_config,
            anchor_model=anchor,
            anchor_cache=anchor_cache,
            first_update_index=update_index,
            device="cuda",
            lr_scheduler=scheduler,
            batch_result=epoch_batches,
            anchor_cache_key_offset=epoch * len(batch_result.batches),
            collect_diagnostics=False,
        )
        if len(result.updates) != 1:
            raise RuntimeError("profile learner kernel update geometry changed")
        update = result.updates[0]
        _validate_ppo_update(update, expected_rows=_FIXED_KERNEL_ROWS)
        updates.append(update)
        update_index += len(result.updates)
    return ProfileKernelUpdateResult(
        next_update_index=update_index,
        optimizer_steps=len(updates),
        ppo_rows=sum(item.sample_count for item in updates),
        factual_rows=sum(item.factual_decisions for item in updates),
        planner_rows=sum(item.planner_decisions for item in updates),
        planner_applicable_rows=sum(
            item.planner_applicable_decisions for item in updates
        ),
        root_value_rows=sum(item.root_information_value_rows for item in updates),
    )


def _validate_ppo_update(
    update: PpoUpdateResult,
    *,
    expected_rows: int,
) -> None:
    actual = (
        update.sample_count,
        update.factual_decisions,
        update.planner_decisions,
        update.planner_applicable_decisions,
        update.root_information_value_rows,
    )
    if actual != (expected_rows,) * 5:
        raise RuntimeError(f"profile learner PPO objective counts changed: {actual}")
    if update.microbatch_count != _FIXED_ACCUMULATION_STEPS:
        raise RuntimeError("profile learner PPO accumulation geometry changed")
    if update.engine_teacher_decisions != 0:
        raise RuntimeError("profile learner PPO revived the retired teacher")
    if update.should_stop:
        raise RuntimeError("profile learner PPO stopped before fixed work completed")


__all__ = [
    "ProfileKernelUpdateResult",
    "build_profile_kernel_batch_config",
    "build_profile_kernel_ppo_config",
    "load_fixed_profile_anchor",
    "run_profile_kernel_update",
    "validate_fixed_kernel_contract",
]
