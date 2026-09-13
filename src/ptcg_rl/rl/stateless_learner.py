"""Single-H200 mixed-precision learner for clean stateless deck-macro PPO."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Literal, TypeVar

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor

from ptcg_rl.context.public_event_arrays import collate_public_event_deltas
from ptcg_rl.model.policy import collate_encoded_options
from ptcg_rl.model.sequence.action import collate_accepted_actions
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    build_sparse_belief_targets,
    normalized_sparse_belief_row_losses,
    resolve_simple_exact_routes,
    select_simple_stateless_backbone_rows,
)
from ptcg_rl.model.tensor_validation import require_tensor_condition
from ptcg_rl.rl.ordered_preparation import (
    OrderedPreparationPipeline,
    TimedPreparedItem,
)
from ptcg_rl.rl.policy_inputs import (
    collate_simple_stateless_actor_rows,
    collate_simple_stateless_observation_rows,
)
from ptcg_rl.rl.sequence_array_collation import (
    StatelessArraySequenceContext,
    StatelessArraySequenceTargets,
    collate_stateless_array_sequence_context,
    collate_stateless_array_sequence_targets,
)
from ptcg_rl.rl.sequence_array_replay import (
    ArraySequenceMicrobatchPlan,
    ArraySequenceReplayIndex,
    build_array_sequence_replay_index,
    plan_array_sequence_microbatch,
    schedule_array_sequence_logical_batches,
)
from ptcg_rl.rl.sequence_replay import (
    SequenceReplayIndex,
    build_sequence_replay_index,
    plan_sequence_microbatch,
    schedule_sequence_logical_batches,
)
from ptcg_rl.rl.stateless_array_collation import (
    collate_stateless_array_microbatch,
)
from ptcg_rl.rl.stateless_array_replay import StatelessArrayOptimizerWindow
from ptcg_rl.rl.stateless_macro_weights import normalized_present_deck_shares
from ptcg_rl.rl.stateless_ppo import (
    SimpleStatelessPpoConfig,
    StatelessLearnerPrecision,
    StatelessLogicalBatch,
    StatelessMicrobatch,
    StatelessPpoLoss,
    StatelessPpoLossInputs,
    schedule_stateless_logical_batches,
    schedule_stateless_microbatches,
    stateless_deck_macro_ppo_loss,
    stateless_learning_rate,
)
from ptcg_rl.rl.stateless_private_optimizer import (
    STATELESS_OPTIMIZER_GROUP_ROLE_KEY,
    STATELESS_PRIVATE_GROUP_ROLE,
    STATELESS_SHARED_GROUP_ROLE,
    StatelessTrainableParameterScope,
    build_stateless_hybrid_optimizer_groups,
    configure_stateless_trainable_scope,
)
from ptcg_rl.rl.stateless_replay import (
    StatelessOptimizerWindow,
    StatelessPpoTarget,
)

_ModelT = TypeVar("_ModelT", bound=torch.nn.Module)

_StructureT = TypeVar("_StructureT")


def _structure_to_device(value: _StructureT, device: torch.device) -> _StructureT:
    """Move every tensor inside a frozen dataclass structure to one device."""
    if isinstance(value, Tensor):
        return value.to(device=device)  # type: ignore[return-value]
    if is_dataclass(value) and not isinstance(value, type):
        updates: dict[str, Any] = {}
        for field in fields(value):
            current = getattr(value, field.name)
            moved = _structure_to_device(current, device)
            if moved is not current:
                updates[field.name] = moved
        return replace(value, **updates) if updates else value
    return value


@dataclass(frozen=True)
class _PreparedArraySequenceMicrobatch:
    """Host-side collation output ready for one device evaluation."""

    microbatch: StatelessMicrobatch
    plan: ArraySequenceMicrobatchPlan
    context: StatelessArraySequenceContext
    targets: StatelessArraySequenceTargets


class DeckMacroUpdateReport(BaseModel):
    """Actual decision and objective weight for one present exact deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    deck_digest: str
    decisions: int = Field(gt=0)
    decision_share: float = Field(gt=0.0, le=1.0)
    effective_macro_weight: float = Field(gt=0.0, le=1.0)
    belief_decisions: int = Field(ge=0)
    belief_macro_weight: float = Field(ge=0.0, le=1.0)


class StatelessOptimizerStepReport(BaseModel):
    """Diagnostics and durable cursors for one actual AdamW step."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer_step_index: int = Field(ge=0)
    epoch_index: int = Field(ge=0)
    logical_batch_index: int = Field(ge=0)
    decisions: int = Field(gt=0)
    lr_schedule_decisions_seen: int = Field(gt=0)
    active_decode_tokens: int = Field(gt=0)
    microbatches: int = Field(gt=0)
    homogeneous_microbatches: int = Field(ge=0)
    learning_rate: float = Field(ge=0.0)
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    belief_loss: float
    ratio_mean: float
    approximate_kl: float
    clip_fraction: float = Field(ge=0.0, le=1.0)
    target_kl_exceeded: bool
    gradient_norm: float = Field(ge=0.0)
    host_prepare_task_seconds: float = Field(ge=0.0)
    host_prepare_wait_seconds: float = Field(ge=0.0)


class SimpleStatelessLearnerUpdate(BaseModel):
    """Serializable metrics for one fresh rollout window."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    update_index: int = Field(ge=0)
    decisions: int = Field(gt=0)
    active_decode_tokens: int = Field(gt=0)
    fragments_seen: int = Field(gt=0)
    fragments_retained: int = Field(gt=0)
    fragments_stale: int = Field(ge=0)
    optimizer_steps: int = Field(gt=0)
    optimizer_step_start: int = Field(ge=0)
    optimizer_step_end: int = Field(gt=0)
    fresh_decisions_start: int = Field(ge=0)
    fresh_decisions_end: int = Field(gt=0)
    lr_schedule_decisions_start: int = Field(ge=0)
    lr_schedule_decisions_end: int = Field(gt=0)
    microbatches: int = Field(gt=0)
    homogeneous_microbatches: int = Field(ge=0)
    learning_rate: float = Field(ge=0.0)
    loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    belief_loss: float
    ratio_mean: float
    approximate_kl: float
    clip_fraction: float = Field(ge=0.0, le=1.0)
    target_kl_exceeded: bool
    gradient_norm: float = Field(ge=0.0)
    host_prepare_task_seconds: float = Field(ge=0.0)
    host_prepare_wait_seconds: float = Field(ge=0.0)
    advantage_mean: float
    advantage_std: float = Field(ge=0.0)
    decks: tuple[DeckMacroUpdateReport, ...]
    step_reports: tuple[StatelessOptimizerStepReport, ...]


@dataclass
class _AccumulatedLoss:
    """Detached metric totals retained on the training device."""

    values: Tensor

    @classmethod
    def create(cls, device: torch.device) -> _AccumulatedLoss:
        """Create double-precision totals without a host-device transfer."""
        return cls(torch.zeros(9, dtype=torch.float64, device=device))

    def add(
        self,
        *,
        loss: Tensor,
        policy: Tensor,
        value: Tensor,
        entropy: Tensor,
        belief: Tensor,
        ratio_sum: Tensor,
        approximate_kl_sum: Tensor,
        clipped_tokens: Tensor,
        active_tokens: Tensor,
    ) -> None:
        """Accumulate one microbatch without synchronizing the device."""
        batch_values = torch.stack(
            (
                loss.detach(),
                policy.detach(),
                value.detach(),
                entropy.detach(),
                belief.detach(),
                ratio_sum.detach(),
                approximate_kl_sum.detach(),
                clipped_tokens.detach(),
                active_tokens.detach(),
            )
        ).to(device=self.values.device, dtype=self.values.dtype)
        self.values.add_(batch_values)


@dataclass(frozen=True)
class _RowTensors:
    """Aligned scalar replay targets transferred as one numeric block."""

    old_root_values: Tensor
    root_returns: Tensor
    decision_macro_weights: Tensor
    belief_macro_weights: Tensor
    expected_belief_valid: Tensor


@dataclass(frozen=True)
class _CompletedEpoch:
    """One synchronized optimizer step and its additive diagnostics."""

    optimizer_step_index: int
    epoch_index: int
    logical_batch_index: int
    decisions: int
    lr_schedule_decisions_seen: int
    learning_rate: float
    batches: int
    homogeneous_batches: int
    active_tokens: int
    loss: float
    policy_loss: float
    value_loss: float
    entropy_loss: float
    belief_loss: float
    ratio_sum: float
    approximate_kl_sum: float
    clipped_tokens: float
    target_kl_exceeded: bool
    gradient_norm: float
    host_prepare_task_seconds: float
    host_prepare_wait_seconds: float


@dataclass(frozen=True)
class _OptimizerStepPlan:
    """Execution inputs known before an optimizer window starts."""

    batches: tuple[StatelessMicrobatch, ...]
    optimizer_step_index: int
    epoch_index: int
    logical_batch_index: int
    logical_batch_decisions: int
    lr_schedule_decisions_seen: int
    decision_weights: Mapping[int, float] | None
    belief_weights: Mapping[int, float] | None
    learning_rate: float


def _row_tensors(
    window: StatelessOptimizerWindow,
    targets: tuple[StatelessPpoTarget, ...],
    indices: tuple[int, ...],
    *,
    device: torch.device,
) -> _RowTensors:
    """Pack scalar replay targets on CPU before two bulk device transfers."""
    if len(targets) != len(indices):
        raise RuntimeError("learner replay scalar targets are misaligned")
    values = np.asarray(
        (
            tuple(target.decision.root_value for target in targets),
            tuple(target.return_value for target in targets),
            tuple(window.decision_macro_weights[index] for index in indices),
            tuple(window.belief_macro_weights[index] for index in indices),
        ),
        dtype=np.float32,
    )
    expected_valid = np.asarray(
        tuple(target.belief_target_valid for target in targets),
        dtype=np.bool_,
    )
    device_values = torch.from_numpy(values).to(device=device)
    return _RowTensors(
        old_root_values=device_values[0],
        root_returns=device_values[1],
        decision_macro_weights=device_values[2],
        belief_macro_weights=device_values[3],
        expected_belief_valid=torch.from_numpy(expected_valid).to(device=device),
    )


class SimpleStatelessLearner:
    """Train on fresh windows with independently bounded AdamW steps."""

    def __init__(
        self,
        model: SimpleStatelessPolicyValueNet,
        config: SimpleStatelessPpoConfig,
        *,
        device: torch.device | str = "cuda",
        require_h200: bool = True,
        precision: StatelessLearnerPrecision = "bf16",
        fused_adamw: bool = False,
        compile_shared_backbone: bool = False,
        trainable_scope: Literal["full_model", "private_only", "hybrid"] = (
            "full_model"
        ),
        private_learning_rate: float | None = None,
        shared_initial_learning_rate: float | None = None,
        shared_learning_rate: float | None = None,
        shared_warmup_start_update: int | None = None,
        shared_warmup_updates: int | None = None,
        deck_macro_target_shares: Mapping[str, float] | None = None,
        host_prepare_workers: int = 1,
        host_prepare_prefetch_batches: int = 1,
    ) -> None:
        """Move an FP32 master policy to one H200 and create optimizer state."""
        if host_prepare_workers <= 0:
            raise ValueError("host preparation workers must be positive")
        if host_prepare_prefetch_batches < host_prepare_workers:
            raise ValueError(
                "host preparation prefetch must cover every preparation worker"
            )
        self.device = torch.device(device)
        _validate_training_device(self.device, require_h200=require_h200)
        if precision not in {"fp32", "bf16"}:
            raise ValueError("stateless learner precision must be fp32 or bf16")
        self.model = _move_fp32_master_model(model, self.device)
        _validate_optimizer_scope_learning_rates(
            trainable_scope=trainable_scope,
            private_learning_rate=private_learning_rate,
            shared_initial_learning_rate=shared_initial_learning_rate,
            shared_learning_rate=shared_learning_rate,
            shared_warmup_start_update=shared_warmup_start_update,
            shared_warmup_updates=shared_warmup_updates,
        )
        self.trainable_scope: StatelessTrainableParameterScope = (
            configure_stateless_trainable_scope(self.model, trainable_scope)
        )
        self._fixed_learning_rate = (
            private_learning_rate if trainable_scope == "private_only" else None
        )
        self._private_learning_rate = private_learning_rate
        self._shared_initial_learning_rate = shared_initial_learning_rate
        self._shared_learning_rate = shared_learning_rate
        self._shared_warmup_start_update = shared_warmup_start_update
        self._shared_warmup_updates = shared_warmup_updates
        self._family_ids_by_deck = {
            route.deck_digest: route.family_id
            for route in self.model.config.family_routes
        }
        if compile_shared_backbone:
            if precision != "bf16":
                raise ValueError(
                    "compiled learner shared backbone requires BF16 precision"
                )
            self.model.enable_bfloat16_learner_inductor()
        self.config = config
        self.precision = precision
        self.deck_macro_target_shares = (
            None if deck_macro_target_shares is None else dict(deck_macro_target_shares)
        )
        if self.deck_macro_target_shares is not None:
            normalized_present_deck_shares(
                self.deck_macro_target_shares,
                self.deck_macro_target_shares,
            )
        optimizer_kwargs = _adamw_execution_kwargs(
            self.device,
            fused=fused_adamw,
        )
        if trainable_scope == "hybrid":
            assert private_learning_rate is not None
            assert shared_initial_learning_rate is not None
            self.optimizer = torch.optim.AdamW(
                build_stateless_hybrid_optimizer_groups(
                    self.trainable_scope,
                    private_learning_rate=private_learning_rate,
                    shared_learning_rate=shared_initial_learning_rate,
                ),
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
                **optimizer_kwargs,
            )
        else:
            self.optimizer = torch.optim.AdamW(
                self.trainable_scope.parameters,
                lr=(
                    config.learning_rate
                    if private_learning_rate is None
                    else private_learning_rate
                ),
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay,
                **optimizer_kwargs,
            )
        self.update_index = 0
        self.optimizer_step_index = 0
        self.fresh_decisions_seen = 0
        self.lr_schedule_decisions_seen = 0
        self._host_prepare_workers = host_prepare_workers
        self._host_prepare_prefetch_batches = host_prepare_prefetch_batches
        self._prepare_pool: ThreadPoolExecutor | None = None

    def _prepare_worker(self) -> ThreadPoolExecutor:
        """Return the bounded host-collation pool, creating it lazily."""
        if self._prepare_pool is None:
            self._prepare_pool = ThreadPoolExecutor(
                max_workers=self._host_prepare_workers,
                thread_name_prefix="learner-prepare",
            )
        return self._prepare_pool

    def set_deck_macro_target_shares(
        self,
        target_shares: Mapping[str, float] | None,
    ) -> None:
        """Set the exact deck macro objective for the next optimizer window."""
        normalized = None if target_shares is None else dict(target_shares)
        if normalized is not None:
            normalized_present_deck_shares(normalized, normalized)
        self.deck_macro_target_shares = normalized

    def restore_exact(
        self,
        *,
        optimizer_state: Mapping[str, Any],
        update_index: int,
        optimizer_step_index: int | None = None,
        fresh_decisions_seen: int = 0,
        lr_schedule_decisions_seen: int | None = None,
    ) -> None:
        """Restore a verified sidecar without altering optimizer semantics."""
        if update_index < 0:
            raise ValueError("learner update index must be non-negative")
        if optimizer_step_index is not None and optimizer_step_index < 0:
            raise ValueError("optimizer step index must be non-negative")
        if fresh_decisions_seen < 0:
            raise ValueError("fresh decision cursor must be non-negative")
        if lr_schedule_decisions_seen is not None and lr_schedule_decisions_seen < 0:
            raise ValueError("LR schedule decision cursor must be non-negative")
        self.optimizer.load_state_dict(dict(optimizer_state))
        if self.trainable_scope.mode == "private_only":
            if self._fixed_learning_rate is None or any(
                float(group["lr"]) != self._fixed_learning_rate
                for group in self.optimizer.param_groups
            ):
                raise ValueError(
                    "private-only optimizer state has a different learning rate"
                )
        elif self.trainable_scope.mode == "hybrid":
            _require_hybrid_optimizer_groups(self.optimizer)
            if (
                self._private_learning_rate is None
                or float(self.optimizer.param_groups[0]["lr"])
                != self._private_learning_rate
            ):
                raise ValueError(
                    "hybrid optimizer state has a different private learning rate"
                )
            warmup_start = self._shared_warmup_start_update
            if warmup_start is None or update_index < warmup_start:
                raise ValueError("hybrid optimizer cursor precedes its warmup anchor")
        _require_fp32_optimizer_state(self.optimizer)
        self.update_index = update_index
        self.optimizer_step_index = (
            update_index if optimizer_step_index is None else optimizer_step_index
        )
        self.fresh_decisions_seen = fresh_decisions_seen
        self.lr_schedule_decisions_seen = (
            fresh_decisions_seen
            if lr_schedule_decisions_seen is None
            else lr_schedule_decisions_seen
        )

    def update(
        self,
        window: StatelessOptimizerWindow,
    ) -> SimpleStatelessLearnerUpdate:
        """Run configured logical optimizer steps over one fresh object window."""
        fresh_start = self.fresh_decisions_seen
        schedule_start = self.lr_schedule_decisions_seen
        sequence_replay = (
            build_sequence_replay_index(window)
            if self.model.sequence is not None
            else None
        )
        logical_batches = (
            schedule_sequence_logical_batches(
                window,
                target_decisions=self.config.logical_batch_decisions,
                locality_chunk_decisions=self.config.microbatch_decisions,
            )
            if sequence_replay is not None
            else None
        )
        steps = self._run_window(
            window.deck_digests,
            tuple(target.belief_target_valid for target in window.targets),
            lambda microbatch: self._evaluate_microbatch(
                window,
                microbatch,
                sequence_replay=sequence_replay,
            ),
            logical_batches=logical_batches,
        )
        report = _update_report(
            update_index=self.update_index,
            decisions=len(window.targets),
            fragments_seen=window.fragments_seen,
            fragments_retained=window.fragments_retained,
            fragments_stale=window.fragments_stale,
            advantage_mean=window.advantage_mean,
            advantage_std=window.advantage_std,
            decks=_deck_reports(window),
            fresh_decisions_start=fresh_start,
            fresh_decisions_end=self.fresh_decisions_seen,
            lr_schedule_decisions_start=schedule_start,
            lr_schedule_decisions_end=self.lr_schedule_decisions_seen,
            steps=steps,
        )
        self.update_index += 1
        return report

    def update_array(
        self,
        window: StatelessArrayOptimizerWindow,
    ) -> SimpleStatelessLearnerUpdate:
        """Run logical optimizer steps directly from compact replay columns."""
        sequence_replay = (
            build_array_sequence_replay_index(window)
            if self.model.sequence is not None
            else None
        )
        logical_batches = (
            schedule_array_sequence_logical_batches(
                window,
                sequence_replay,
                target_decisions=self.config.logical_batch_decisions,
                locality_chunk_decisions=self.config.microbatch_decisions,
            )
            if sequence_replay is not None
            else None
        )
        fresh_start = self.fresh_decisions_seen
        schedule_start = self.lr_schedule_decisions_seen
        prepare: (
            Callable[[StatelessMicrobatch], _PreparedArraySequenceMicrobatch] | None
        ) = None
        evaluate_prepared: (
            Callable[[_PreparedArraySequenceMicrobatch], StatelessPpoLossInputs] | None
        ) = None
        if sequence_replay is not None:
            replay_index = sequence_replay
            prepare = lambda microbatch: self._prepare_array_sequence_microbatch(  # noqa: E731
                window,
                microbatch,
                sequence_replay=replay_index,
            )
            evaluate_prepared = self._evaluate_prepared_array_sequence_microbatch
        steps = self._run_window(
            window.route_deck_digests,
            tuple(bool(value) for value in window.belief_target_valid),
            lambda microbatch: self._evaluate_array_microbatch(
                window,
                microbatch,
                sequence_replay=sequence_replay,
            ),
            logical_batches=logical_batches,
            prepare=prepare,
            evaluate_prepared=evaluate_prepared,
        )
        report = _update_report(
            update_index=self.update_index,
            decisions=window.decision_count,
            fragments_seen=window.fragments_seen,
            fragments_retained=window.fragments_retained,
            fragments_stale=window.fragments_stale,
            advantage_mean=window.advantage_mean,
            advantage_std=window.advantage_std,
            decks=_array_deck_reports(window),
            fresh_decisions_start=fresh_start,
            fresh_decisions_end=self.fresh_decisions_seen,
            lr_schedule_decisions_start=schedule_start,
            lr_schedule_decisions_end=self.lr_schedule_decisions_seen,
            steps=steps,
        )
        self.update_index += 1
        return report

    def _run_window(
        self,
        deck_digests: Sequence[str],
        belief_valid: Sequence[bool],
        evaluate: Callable[[StatelessMicrobatch], StatelessPpoLossInputs],
        *,
        logical_batches: Sequence[StatelessLogicalBatch] | None = None,
        prepare: (
            Callable[[StatelessMicrobatch], _PreparedArraySequenceMicrobatch] | None
        ) = None,
        evaluate_prepared: (
            Callable[[_PreparedArraySequenceMicrobatch], StatelessPpoLossInputs] | None
        ) = None,
    ) -> tuple[_CompletedEpoch, ...]:
        """Execute deterministic deck-stratified steps for one fresh window."""
        if len(deck_digests) != len(belief_valid):
            raise RuntimeError("learner window deck and belief rows are misaligned")
        if logical_batches is None:
            logical_batches = schedule_stateless_logical_batches(
                deck_digests,
                target_decisions=self.config.logical_batch_decisions,
            )
        fresh_decisions_end = self.fresh_decisions_seen + len(deck_digests)
        schedule_decisions_end = self.lr_schedule_decisions_seen + len(deck_digests)
        schedule_batch_ends: list[int] = []
        scheduled_decisions = self.lr_schedule_decisions_seen
        for logical_batch in logical_batches:
            scheduled_decisions += len(logical_batch.indices)
            schedule_batch_ends.append(scheduled_decisions)
        plans: list[_OptimizerStepPlan] = []
        local_weights_required = len(logical_batches) > 1
        for epoch_index in range(self.config.ppo_epochs):
            for logical_batch_index, logical_batch in enumerate(logical_batches):
                lr_schedule_cursor = (
                    schedule_batch_ends[logical_batch_index]
                    if epoch_index == 0
                    else schedule_decisions_end
                )
                learning_rate = stateless_learning_rate(
                    self.config,
                    update_index=self.update_index,
                    lr_schedule_decisions_seen=lr_schedule_cursor,
                )
                batches = _logical_microbatches(
                    logical_batch,
                    deck_digests,
                    maximum_decisions=self.config.microbatch_decisions,
                    homogeneous_min_decisions=(self.config.homogeneous_min_decisions),
                    family_ids_by_deck=getattr(
                        self,
                        "_family_ids_by_deck",
                        None,
                    ),
                )
                decision_weights: Mapping[int, float] | None = None
                belief_weights: Mapping[int, float] | None = None
                if local_weights_required:
                    decision_weights, belief_weights = _logical_macro_weights(
                        deck_digests,
                        belief_valid,
                        logical_batch,
                        target_shares=getattr(
                            self,
                            "deck_macro_target_shares",
                            None,
                        ),
                    )
                plans.append(
                    _OptimizerStepPlan(
                        batches=tuple(batches),
                        optimizer_step_index=self.optimizer_step_index + len(plans),
                        epoch_index=epoch_index,
                        logical_batch_index=logical_batch_index,
                        logical_batch_decisions=len(logical_batch.indices),
                        lr_schedule_decisions_seen=lr_schedule_cursor,
                        decision_weights=decision_weights,
                        belief_weights=belief_weights,
                        learning_rate=learning_rate,
                    )
                )
        if (prepare is None) != (evaluate_prepared is None):
            raise RuntimeError(
                "host preparation and prepared evaluation must be configured together"
            )
        preparation_pipeline: (
            OrderedPreparationPipeline[
                StatelessMicrobatch,
                _PreparedArraySequenceMicrobatch,
            ]
            | None
        ) = None
        prepared_batches: (
            Iterator[TimedPreparedItem[_PreparedArraySequenceMicrobatch]] | None
        ) = None
        if prepare is not None:
            preparation_pipeline = OrderedPreparationPipeline(
                executor=self._prepare_worker(),
                prepare=prepare,
                items=(batch for plan in plans for batch in plan.batches),
                capacity=self._host_prepare_prefetch_batches,
            )
            prepared_batches = iter(preparation_pipeline)
        completed: list[_CompletedEpoch] = []
        try:
            for plan in plans:
                completed.append(
                    self._run_optimizer_step(
                        plan.batches,
                        evaluate,
                        prepared_batches=prepared_batches,
                        evaluate_prepared=evaluate_prepared,
                        optimizer_step_index=plan.optimizer_step_index,
                        epoch_index=plan.epoch_index,
                        logical_batch_index=plan.logical_batch_index,
                        logical_batch_decisions=plan.logical_batch_decisions,
                        lr_schedule_decisions_seen=(plan.lr_schedule_decisions_seen),
                        decision_weights=plan.decision_weights,
                        belief_weights=plan.belief_weights,
                        learning_rate=plan.learning_rate,
                    )
                )
                self.optimizer_step_index += 1
        finally:
            if preparation_pipeline is not None:
                preparation_pipeline.close()
        self.fresh_decisions_seen = fresh_decisions_end
        self.lr_schedule_decisions_seen = schedule_decisions_end
        return tuple(completed)

    def _run_optimizer_step(
        self,
        batches: Sequence[StatelessMicrobatch],
        evaluate: Callable[[StatelessMicrobatch], StatelessPpoLossInputs],
        *,
        prepared_batches: (
            Iterator[TimedPreparedItem[_PreparedArraySequenceMicrobatch]] | None
        ),
        evaluate_prepared: (
            Callable[[_PreparedArraySequenceMicrobatch], StatelessPpoLossInputs] | None
        ) = None,
        optimizer_step_index: int,
        epoch_index: int,
        logical_batch_index: int,
        logical_batch_decisions: int,
        lr_schedule_decisions_seen: int,
        decision_weights: Mapping[int, float] | None,
        belief_weights: Mapping[int, float] | None,
        learning_rate: float,
    ) -> _CompletedEpoch:
        """Accumulate bounded microbatches and take one actual AdamW step."""
        effective_learning_rate = self._configure_optimizer_learning_rates(
            learning_rate
        )
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        metrics = _AccumulatedLoss.create(self.device)
        host_prepare_task_seconds = 0.0
        host_prepare_wait_seconds = 0.0
        for microbatch in batches:
            if prepared_batches is not None:
                if evaluate_prepared is None:
                    raise RuntimeError("prepared learner evaluation is missing")
                try:
                    prepared = next(prepared_batches)
                except StopIteration as error:
                    raise RuntimeError(
                        "host preparation ended before optimizer execution"
                    ) from error
                if prepared.value.microbatch != microbatch:
                    raise RuntimeError("host preparation changed microbatch order")
                host_prepare_task_seconds += prepared.task_seconds
                host_prepare_wait_seconds += prepared.wait_seconds
                loss_inputs = evaluate_prepared(prepared.value)
            else:
                loss_inputs = evaluate(microbatch)
            if decision_weights is not None and belief_weights is not None:
                loss_inputs = _replace_macro_weights(
                    loss_inputs,
                    microbatch,
                    decision_weights=decision_weights,
                    belief_weights=belief_weights,
                )
            breakdown = stateless_deck_macro_ppo_loss(
                loss_inputs,
                self.config,
            )
            _require_finite_fp32_loss(breakdown)
            breakdown.loss.backward()  # type: ignore[no-untyped-call]
            metrics.add(
                loss=breakdown.loss,
                policy=breakdown.policy_loss,
                value=breakdown.value_loss,
                entropy=breakdown.entropy_loss,
                belief=breakdown.belief_loss,
                ratio_sum=breakdown.ratio_sum,
                approximate_kl_sum=breakdown.approximate_kl_sum,
                clipped_tokens=breakdown.clipped_token_count,
                active_tokens=breakdown.active_token_count,
            )
        _require_fp32_master_gradients(self.model)
        trainable_scope = getattr(self, "trainable_scope", None)
        gradient_parameters = (
            self.model.parameters()
            if trainable_scope is None
            else trainable_scope.parameters
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            gradient_parameters,
            self.config.maximum_gradient_norm,
        )
        metric_values = torch.cat(
            (
                metrics.values,
                gradient_norm.detach()
                .reshape(1)
                .to(device=self.device, dtype=metrics.values.dtype),
            )
        ).cpu()
        (
            loss_value,
            policy_value,
            value_value,
            entropy_value,
            belief_value,
            ratio_sum_value,
            approximate_kl_sum_value,
            clipped_tokens_value,
            active_tokens_value,
            gradient_norm_value,
        ) = metric_values.tolist()
        if not math.isfinite(gradient_norm_value):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("stateless learner gradient norm is not finite")
        self.optimizer.step()
        _require_fp32_optimizer_state(self.optimizer)
        approximate_kl = approximate_kl_sum_value / float(active_tokens_value)
        return _CompletedEpoch(
            optimizer_step_index=optimizer_step_index,
            epoch_index=epoch_index,
            logical_batch_index=logical_batch_index,
            decisions=logical_batch_decisions,
            lr_schedule_decisions_seen=lr_schedule_decisions_seen,
            learning_rate=effective_learning_rate,
            batches=len(batches),
            homogeneous_batches=sum(int(batch.homogeneous) for batch in batches),
            active_tokens=int(active_tokens_value),
            loss=loss_value,
            policy_loss=policy_value,
            value_loss=value_value,
            entropy_loss=entropy_value,
            belief_loss=belief_value,
            ratio_sum=ratio_sum_value,
            approximate_kl_sum=approximate_kl_sum_value,
            clipped_tokens=clipped_tokens_value,
            target_kl_exceeded=(
                self.config.target_kl is not None
                and approximate_kl > self.config.target_kl
            ),
            gradient_norm=gradient_norm_value,
            host_prepare_task_seconds=host_prepare_task_seconds,
            host_prepare_wait_seconds=host_prepare_wait_seconds,
        )

    def _configure_optimizer_learning_rates(self, scheduled_rate: float) -> float:
        """Apply the configured scope rates and return the primary report rate."""
        trainable_scope = getattr(self, "trainable_scope", None)
        if trainable_scope is None or trainable_scope.mode == "full_model":
            for group in self.optimizer.param_groups:
                group["lr"] = scheduled_rate
            return scheduled_rate
        if trainable_scope.mode == "private_only":
            fixed_rate = getattr(self, "_fixed_learning_rate", None)
            if fixed_rate is None:
                raise RuntimeError("private-only learner lost its fixed learning rate")
            for group in self.optimizer.param_groups:
                group["lr"] = fixed_rate
            return float(fixed_rate)
        if trainable_scope.mode != "hybrid":
            raise RuntimeError("stateless learner has an unknown optimizer scope")

        _require_hybrid_optimizer_groups(self.optimizer)
        private_rate = getattr(self, "_private_learning_rate", None)
        shared_initial_rate = getattr(self, "_shared_initial_learning_rate", None)
        shared_rate = getattr(self, "_shared_learning_rate", None)
        warmup_start = getattr(self, "_shared_warmup_start_update", None)
        warmup_updates = getattr(self, "_shared_warmup_updates", None)
        if (
            private_rate is None
            or shared_initial_rate is None
            or shared_rate is None
            or warmup_start is None
            or warmup_updates is None
        ):
            raise RuntimeError("hybrid learner lost its learning-rate schedule")
        current_shared_rate = stateless_hybrid_shared_learning_rate(
            update_index=self.update_index,
            warmup_start_update=warmup_start,
            warmup_updates=warmup_updates,
            initial_learning_rate=shared_initial_rate,
            target_learning_rate=shared_rate,
        )
        self.optimizer.param_groups[0]["lr"] = private_rate
        self.optimizer.param_groups[1]["lr"] = current_shared_rate
        return float(private_rate)

    def _evaluate_microbatch(
        self,
        window: StatelessOptimizerWindow,
        microbatch: StatelessMicrobatch,
        *,
        sequence_replay: SequenceReplayIndex | None = None,
    ) -> StatelessPpoLossInputs:
        if self.model.sequence is not None:
            if sequence_replay is None:
                raise RuntimeError("sequence evaluator requires a replay index")
            return self._evaluate_sequence_microbatch(
                window,
                microbatch,
                sequence_replay=sequence_replay,
            )
        targets = tuple(window.targets[index] for index in microbatch.indices)
        batch = collate_simple_stateless_actor_rows(
            tuple(target.decision.actor_row for target in targets),
            device=self.device,
            deduplicate_belief=True,
        )
        actions = tuple(target.decision.action for target in targets)
        routes = resolve_simple_exact_routes(
            batch.deck_signatures,
            self.model.config,
            device=self.device,
        )
        with _learner_autocast_context(self.device, self.precision):
            state = self.model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
            )
            option_embeddings = self.model.encode_legal_options(
                state,
                batch.options,
                route_plan=routes,
            )
            evaluation = self.model.heads.teacher_forced(
                state.policy,
                state.opponent_belief,
                option_embeddings,
                batch.options,
                actions,
                route_plan=routes,
                temperature=self.config.behavior_temperature,
            )
            root_values = self.model.heads.root_value(
                state.value,
                state.opponent_belief,
                route_plan=routes,
            )
            belief_logits = self.model.belief_logits(state)
        sparse_belief = build_sparse_belief_targets(
            tuple(target.opponent_deck for target in targets),
            tuple(
                Counter(dict(target.decision.known_opponent_counts))
                for target in targets
            ),
            card_vocab_size=(
                self.model.backbone.input_encoder.card_encoder.num_card_ids
            ),
            device=self.device,
        )
        belief_row_losses, belief_valid = normalized_sparse_belief_row_losses(
            belief_logits,
            sparse_belief,
        )
        row_tensors = _row_tensors(
            window,
            targets,
            microbatch.indices,
            device=self.device,
        )
        require_tensor_condition(
            belief_valid.eq(row_tensors.expected_belief_valid).all(),
            "reconstructed belief target validity changed",
        )
        (
            old_token_logprobs,
            old_prefix_values,
            token_advantages,
            token_returns,
            expected_mask,
        ) = _token_tensors(
            targets,
            width=int(evaluation.token_mask.shape[1]),
            device=self.device,
        )
        require_tensor_condition(
            evaluation.token_mask.eq(expected_mask).all(),
            "learner replay token mask differs from behavior",
        )
        return StatelessPpoLossInputs(
            current_token_logprobs=evaluation.token_logprobs,
            current_token_entropies=evaluation.token_entropies,
            current_prefix_values=evaluation.prefix_values,
            token_mask=evaluation.token_mask,
            old_token_logprobs=old_token_logprobs,
            old_prefix_values=old_prefix_values,
            token_advantages=token_advantages,
            token_returns=token_returns,
            current_root_values=root_values,
            old_root_values=row_tensors.old_root_values,
            root_returns=row_tensors.root_returns,
            decision_macro_weights=row_tensors.decision_macro_weights,
            belief_row_losses=belief_row_losses,
            belief_valid_mask=belief_valid,
            belief_macro_weights=row_tensors.belief_macro_weights,
        )

    def _evaluate_sequence_microbatch(
        self,
        window: StatelessOptimizerWindow,
        microbatch: StatelessMicrobatch,
        *,
        sequence_replay: SequenceReplayIndex,
    ) -> StatelessPpoLossInputs:
        """Rebuild bounded raw predecessor closures under current parameters."""
        sequence_config = self.model.config.sequence
        if self.model.sequence is None or sequence_config is None:
            raise RuntimeError("sequence evaluator requires temporal model")
        plan = plan_sequence_microbatch(
            window,
            microbatch.indices,
            max_context_blocks=sequence_config.max_context_blocks,
            replay_index=sequence_replay,
        )
        context_decisions = tuple(row.decision for row in plan.rows)
        context_batch = collate_simple_stateless_observation_rows(
            tuple(decision.actor_row for decision in context_decisions),
            device=self.device,
            deduplicate_belief=True,
        )
        context_routes = resolve_simple_exact_routes(
            context_batch.deck_signatures,
            self.model.config,
            device=self.device,
        )
        event_deltas = []
        accepted_actions = []
        for decision in context_decisions:
            if decision.public_event_delta is None or decision.accepted_action is None:
                raise ValueError("sequence context is missing raw temporal payload")
            event_deltas.append(decision.public_event_delta)
            accepted_actions.append(decision.accepted_action)
        events = collate_public_event_deltas(
            tuple(event_deltas),
            device=self.device,
        )
        actions = collate_accepted_actions(
            tuple(accepted_actions),
            device=self.device,
        )
        targets = tuple(window.targets[index] for index in microbatch.indices)
        target_rows_raw = tuple(target.decision.actor_row for target in targets)
        target_options = collate_encoded_options(
            tuple(row.options for row in target_rows_raw),
            min_counts=tuple(row.min_count for row in target_rows_raw),
            max_counts=tuple(row.max_count for row in target_rows_raw),
            device=self.device,
        )
        target_routes = resolve_simple_exact_routes(
            tuple(row.own_deck.signature for row in target_rows_raw),
            self.model.config,
            device=self.device,
        )
        target_rows = torch.tensor(
            plan.target_row_indices,
            dtype=torch.long,
            device=self.device,
        )
        block_indices = torch.tensor(
            plan.block_indices,
            dtype=torch.long,
            device=self.device,
        )
        with _learner_autocast_context(self.device, self.precision):
            snapshots = self.model.encode_observation_state(
                state=context_batch.states,
                unique_deck_card_ids=context_batch.unique_deck_card_ids,
                deck_counts=context_batch.deck_counts,
                deck_valid_mask=context_batch.deck_valid_mask,
                belief_summary=context_batch.belief_summary,
                route_plan=context_routes,
            )
            temporal_contexts = self.model.replay_sequence(
                snapshots,
                events,
                actions,
                sequence_offsets=plan.sequence_offsets,
                block_indices=block_indices,
            )
            conditioned = self.model.condition_sequence(
                snapshots,
                temporal_contexts,
            )
            target_state = select_simple_stateless_backbone_rows(
                conditioned,
                target_rows,
            )
            option_embeddings = self.model.encode_legal_options(
                target_state,
                target_options,
                route_plan=target_routes,
            )
            evaluation = self.model.heads.teacher_forced(
                target_state.policy,
                target_state.opponent_belief,
                option_embeddings,
                target_options,
                tuple(target.decision.action for target in targets),
                route_plan=target_routes,
                temperature=self.config.behavior_temperature,
            )
            root_values = self.model.heads.root_value(
                target_state.value,
                target_state.opponent_belief,
                route_plan=target_routes,
            )
            belief_logits = self.model.belief_logits(target_state)
        sparse_belief = build_sparse_belief_targets(
            tuple(target.opponent_deck for target in targets),
            tuple(
                Counter(dict(target.decision.known_opponent_counts))
                for target in targets
            ),
            card_vocab_size=(
                self.model.backbone.input_encoder.card_encoder.num_card_ids
            ),
            device=self.device,
        )
        belief_row_losses, belief_valid = normalized_sparse_belief_row_losses(
            belief_logits,
            sparse_belief,
        )
        row_tensors = _row_tensors(
            window,
            targets,
            microbatch.indices,
            device=self.device,
        )
        require_tensor_condition(
            belief_valid.eq(row_tensors.expected_belief_valid).all(),
            "sequence belief target validity changed",
        )
        (
            old_token_logprobs,
            old_prefix_values,
            token_advantages,
            token_returns,
            expected_mask,
        ) = _token_tensors(
            targets,
            width=int(evaluation.token_mask.shape[1]),
            device=self.device,
        )
        require_tensor_condition(
            evaluation.token_mask.eq(expected_mask).all(),
            "sequence replay token mask differs from behavior",
        )
        return StatelessPpoLossInputs(
            current_token_logprobs=evaluation.token_logprobs,
            current_token_entropies=evaluation.token_entropies,
            current_prefix_values=evaluation.prefix_values,
            token_mask=evaluation.token_mask,
            old_token_logprobs=old_token_logprobs,
            old_prefix_values=old_prefix_values,
            token_advantages=token_advantages,
            token_returns=token_returns,
            current_root_values=root_values,
            old_root_values=row_tensors.old_root_values,
            root_returns=row_tensors.root_returns,
            decision_macro_weights=row_tensors.decision_macro_weights,
            belief_row_losses=belief_row_losses,
            belief_valid_mask=belief_valid,
            belief_macro_weights=row_tensors.belief_macro_weights,
        )

    def _evaluate_array_microbatch(
        self,
        window: StatelessArrayOptimizerWindow,
        microbatch: StatelessMicrobatch,
        *,
        sequence_replay: ArraySequenceReplayIndex | None = None,
    ) -> StatelessPpoLossInputs:
        """Evaluate one direct compact-column microbatch on the H200."""
        if self.model.sequence is not None:
            if sequence_replay is None:
                raise RuntimeError("array sequence evaluator requires replay index")
            return self._evaluate_array_sequence_microbatch(
                window,
                microbatch,
                sequence_replay=sequence_replay,
            )
        card_vocab_size = self.model.backbone.input_encoder.card_encoder.num_card_ids
        batch = collate_stateless_array_microbatch(
            window,
            microbatch.indices,
            card_vocab_size=card_vocab_size,
            device=self.device,
        )
        routes = resolve_simple_exact_routes(
            batch.deck_signatures,
            self.model.config,
            device=self.device,
        )
        with _learner_autocast_context(self.device, self.precision):
            state = self.model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
            )
            option_embeddings = self.model.encode_legal_options(
                state,
                batch.options,
                route_plan=routes,
            )
            evaluation = self.model.heads.teacher_forced(
                state.policy,
                state.opponent_belief,
                option_embeddings,
                batch.options,
                batch.actions,
                route_plan=routes,
                temperature=self.config.behavior_temperature,
            )
            root_values = self.model.heads.root_value(
                state.value,
                state.opponent_belief,
                route_plan=routes,
            )
            belief_logits = self.model.belief_logits(state)
        belief_row_losses, belief_valid = normalized_sparse_belief_row_losses(
            belief_logits,
            batch.sparse_belief_targets,
        )
        require_tensor_condition(
            belief_valid.eq(batch.expected_belief_valid).all(),
            "array belief target validity changed",
        )
        require_tensor_condition(
            evaluation.token_mask.eq(batch.token_mask).all(),
            "array replay token mask differs from behavior",
        )
        return StatelessPpoLossInputs(
            current_token_logprobs=evaluation.token_logprobs,
            current_token_entropies=evaluation.token_entropies,
            current_prefix_values=evaluation.prefix_values,
            token_mask=evaluation.token_mask,
            old_token_logprobs=batch.old_token_logprobs,
            old_prefix_values=batch.old_prefix_values,
            token_advantages=batch.token_advantages,
            token_returns=batch.token_returns,
            current_root_values=root_values,
            old_root_values=batch.old_root_values,
            root_returns=batch.root_returns,
            decision_macro_weights=batch.decision_macro_weights,
            belief_row_losses=belief_row_losses,
            belief_valid_mask=belief_valid,
            belief_macro_weights=batch.belief_macro_weights,
        )

    def _prepare_array_sequence_microbatch(
        self,
        window: StatelessArrayOptimizerWindow,
        microbatch: StatelessMicrobatch,
        *,
        sequence_replay: ArraySequenceReplayIndex,
    ) -> _PreparedArraySequenceMicrobatch:
        """Plan and gather one microbatch into host tensors (CPU only)."""
        sequence_config = self.model.config.sequence
        if self.model.sequence is None or sequence_config is None:
            raise RuntimeError("array sequence evaluator requires temporal model")
        plan = plan_array_sequence_microbatch(
            window,
            microbatch.indices,
            max_context_blocks=sequence_config.max_context_blocks,
            replay_index=sequence_replay,
        )
        card_vocab_size = self.model.backbone.input_encoder.card_encoder.num_card_ids
        context = collate_stateless_array_sequence_context(
            window,
            plan,
            card_vocab_size=card_vocab_size,
            device=None,
        )
        targets = collate_stateless_array_sequence_targets(
            window,
            microbatch.indices,
            card_vocab_size=card_vocab_size,
            device=None,
        )
        if (
            context.input_contract_fingerprint != targets.input_contract_fingerprint
            or context.public_deck_catalog_fingerprint
            != targets.public_deck_catalog_fingerprint
        ):
            raise ValueError("array sequence context changed its static input contract")
        return _PreparedArraySequenceMicrobatch(
            microbatch=microbatch,
            plan=plan,
            context=context,
            targets=targets,
        )

    def _evaluate_array_sequence_microbatch(
        self,
        window: StatelessArrayOptimizerWindow,
        microbatch: StatelessMicrobatch,
        *,
        sequence_replay: ArraySequenceReplayIndex,
    ) -> StatelessPpoLossInputs:
        """Replay compact predecessor closures under current parameters."""
        return self._evaluate_prepared_array_sequence_microbatch(
            self._prepare_array_sequence_microbatch(
                window,
                microbatch,
                sequence_replay=sequence_replay,
            )
        )

    def _evaluate_prepared_array_sequence_microbatch(
        self,
        prepared: _PreparedArraySequenceMicrobatch,
    ) -> StatelessPpoLossInputs:
        """Move one host-collated microbatch to the device and evaluate it."""
        plan = prepared.plan
        context = _structure_to_device(prepared.context, self.device)
        targets = _structure_to_device(prepared.targets, self.device)
        context_routes = resolve_simple_exact_routes(
            context.deck_signatures,
            self.model.config,
            device=self.device,
        )
        target_routes = resolve_simple_exact_routes(
            tuple(context.deck_signatures[index] for index in plan.target_row_indices),
            self.model.config,
            device=self.device,
        )
        target_rows = torch.tensor(
            plan.target_row_indices,
            dtype=torch.long,
            device=self.device,
        )
        block_indices = torch.tensor(
            plan.block_indices,
            dtype=torch.long,
            device=self.device,
        )
        with _learner_autocast_context(self.device, self.precision):
            snapshots = self.model.encode_observation_state(
                state=context.states,
                unique_deck_card_ids=context.unique_deck_card_ids,
                deck_counts=context.deck_counts,
                deck_valid_mask=context.deck_valid_mask,
                belief_summary=context.belief_summary,
                route_plan=context_routes,
            )
            temporal_contexts = self.model.replay_sequence(
                snapshots,
                context.events,
                context.accepted_actions,
                sequence_offsets=plan.sequence_offsets,
                block_indices=block_indices,
            )
            conditioned = self.model.condition_sequence(
                snapshots,
                temporal_contexts,
            )
            target_state = select_simple_stateless_backbone_rows(
                conditioned,
                target_rows,
            )
            option_embeddings = self.model.encode_legal_options(
                target_state,
                targets.options,
                route_plan=target_routes,
            )
            evaluation = self.model.heads.teacher_forced(
                target_state.policy,
                target_state.opponent_belief,
                option_embeddings,
                targets.options,
                targets.actions,
                route_plan=target_routes,
                temperature=self.config.behavior_temperature,
            )
            root_values = self.model.heads.root_value(
                target_state.value,
                target_state.opponent_belief,
                route_plan=target_routes,
            )
            belief_logits = self.model.belief_logits(target_state)
        belief_row_losses, belief_valid = normalized_sparse_belief_row_losses(
            belief_logits,
            targets.sparse_belief_targets,
        )
        require_tensor_condition(
            belief_valid.eq(targets.expected_belief_valid).all(),
            "array sequence belief target validity changed",
        )
        require_tensor_condition(
            evaluation.token_mask.eq(targets.token_mask).all(),
            "array sequence replay token mask differs from behavior",
        )
        return StatelessPpoLossInputs(
            current_token_logprobs=evaluation.token_logprobs,
            current_token_entropies=evaluation.token_entropies,
            current_prefix_values=evaluation.prefix_values,
            token_mask=evaluation.token_mask,
            old_token_logprobs=targets.old_token_logprobs,
            old_prefix_values=targets.old_prefix_values,
            token_advantages=targets.token_advantages,
            token_returns=targets.token_returns,
            current_root_values=root_values,
            old_root_values=targets.old_root_values,
            root_returns=targets.root_returns,
            decision_macro_weights=targets.decision_macro_weights,
            belief_row_losses=belief_row_losses,
            belief_valid_mask=belief_valid,
            belief_macro_weights=targets.belief_macro_weights,
        )


def _token_tensors(
    targets: tuple[StatelessPpoTarget, ...],
    *,
    width: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    if width <= 0:
        raise RuntimeError("learner replay emitted no decode-token slots")
    shape = (len(targets), width)
    values = np.zeros((4, *shape), dtype=np.float32)
    mask = np.zeros(shape, dtype=np.bool_)
    for row, target in enumerate(targets):
        length = len(target.decision.token_logprobs)
        if length > width:
            raise RuntimeError("learner replay omitted behavior decode tokens")
        if (
            len(target.decision.prefix_values) != length
            or len(target.token_advantages) != length
            or len(target.token_returns) != length
        ):
            raise RuntimeError("learner replay token fields are misaligned")
        values[0, row, :length] = target.decision.token_logprobs
        values[1, row, :length] = target.decision.prefix_values
        values[2, row, :length] = target.token_advantages
        values[3, row, :length] = target.token_returns
        mask[row, :length] = True
    device_values = torch.from_numpy(values).to(device=device)
    return (
        device_values[0],
        device_values[1],
        device_values[2],
        device_values[3],
        torch.from_numpy(mask).to(device=device),
    )


def _logical_microbatches(
    logical_batch: StatelessLogicalBatch,
    deck_digests: Sequence[str],
    *,
    maximum_decisions: int,
    homogeneous_min_decisions: int,
    family_ids_by_deck: Mapping[str, str] | None = None,
) -> tuple[StatelessMicrobatch, ...]:
    """Schedule route-aware microbatches using global replay coordinates."""
    local_decks = tuple(deck_digests[index] for index in logical_batch.indices)
    local_family_ids: tuple[str, ...] | None = None
    if family_ids_by_deck:
        missing = tuple(sorted(set(local_decks).difference(family_ids_by_deck)))
        if missing:
            raise ValueError(
                "family-private learner rows have no family route: "
                + ", ".join(missing)
            )
        local_family_ids = tuple(family_ids_by_deck[digest] for digest in local_decks)
    local_batches = schedule_stateless_microbatches(
        local_decks,
        maximum_decisions=maximum_decisions,
        homogeneous_min_decisions=homogeneous_min_decisions,
        family_ids=local_family_ids,
    )
    return tuple(
        StatelessMicrobatch(
            indices=tuple(
                logical_batch.indices[local_index] for local_index in batch.indices
            ),
            deck_digests=batch.deck_digests,
            homogeneous=batch.homogeneous,
        )
        for batch in local_batches
    )


def _logical_macro_weights(
    deck_digests: Sequence[str],
    belief_valid: Sequence[bool],
    logical_batch: StatelessLogicalBatch,
    *,
    target_shares: Mapping[str, float] | None = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Return exact deck-macro weights normalized within one optimizer step."""
    decision_counts = Counter(deck_digests[index] for index in logical_batch.indices)
    decision_shares = normalized_present_deck_shares(
        decision_counts,
        target_shares,
    )
    decision_weights = {
        index: decision_shares[deck_digests[index]]
        / decision_counts[deck_digests[index]]
        for index in logical_batch.indices
    }
    belief_counts = Counter(
        deck_digests[index] for index in logical_batch.indices if belief_valid[index]
    )
    belief_shares = normalized_present_deck_shares(
        belief_counts,
        target_shares,
    )
    belief_weights = {
        index: (
            belief_shares[deck_digests[index]] / belief_counts[deck_digests[index]]
            if belief_valid[index]
            else 0.0
        )
        for index in logical_batch.indices
    }
    return decision_weights, belief_weights


def _replace_macro_weights(
    inputs: StatelessPpoLossInputs,
    microbatch: StatelessMicrobatch,
    *,
    decision_weights: Mapping[int, float],
    belief_weights: Mapping[int, float],
) -> StatelessPpoLossInputs:
    """Replace global-window weights with current logical-step weights."""
    return replace(
        inputs,
        decision_macro_weights=inputs.decision_macro_weights.new_tensor(
            tuple(decision_weights[index] for index in microbatch.indices)
        ),
        belief_macro_weights=inputs.belief_macro_weights.new_tensor(
            tuple(belief_weights[index] for index in microbatch.indices)
        ),
    )


def _update_report(
    *,
    update_index: int,
    decisions: int,
    fragments_seen: int,
    fragments_retained: int,
    fragments_stale: int,
    advantage_mean: float,
    advantage_std: float,
    decks: tuple[DeckMacroUpdateReport, ...],
    fresh_decisions_start: int,
    fresh_decisions_end: int,
    lr_schedule_decisions_start: int,
    lr_schedule_decisions_end: int,
    steps: Sequence[_CompletedEpoch],
) -> SimpleStatelessLearnerUpdate:
    """Build one common serializable report for either replay representation."""
    if not steps:
        raise RuntimeError("learner update completed no optimizer steps")
    step_count = len(steps)
    active_tokens = sum(step.active_tokens for step in steps)
    ratio_sum = sum(step.ratio_sum for step in steps)
    approximate_kl_sum = sum(step.approximate_kl_sum for step in steps)
    clipped_tokens = sum(step.clipped_tokens for step in steps)
    step_reports = tuple(
        StatelessOptimizerStepReport(
            optimizer_step_index=step.optimizer_step_index,
            epoch_index=step.epoch_index,
            logical_batch_index=step.logical_batch_index,
            decisions=step.decisions,
            lr_schedule_decisions_seen=step.lr_schedule_decisions_seen,
            active_decode_tokens=step.active_tokens,
            microbatches=step.batches,
            homogeneous_microbatches=step.homogeneous_batches,
            learning_rate=step.learning_rate,
            loss=step.loss,
            policy_loss=step.policy_loss,
            value_loss=step.value_loss,
            entropy=-step.entropy_loss,
            belief_loss=step.belief_loss,
            ratio_mean=step.ratio_sum / float(step.active_tokens),
            approximate_kl=(step.approximate_kl_sum / float(step.active_tokens)),
            clip_fraction=step.clipped_tokens / float(step.active_tokens),
            target_kl_exceeded=step.target_kl_exceeded,
            gradient_norm=step.gradient_norm,
            host_prepare_task_seconds=step.host_prepare_task_seconds,
            host_prepare_wait_seconds=step.host_prepare_wait_seconds,
        )
        for step in steps
    )
    return SimpleStatelessLearnerUpdate(
        update_index=update_index,
        decisions=decisions,
        active_decode_tokens=active_tokens,
        fragments_seen=fragments_seen,
        fragments_retained=fragments_retained,
        fragments_stale=fragments_stale,
        optimizer_steps=step_count,
        optimizer_step_start=steps[0].optimizer_step_index,
        optimizer_step_end=steps[-1].optimizer_step_index + 1,
        fresh_decisions_start=fresh_decisions_start,
        fresh_decisions_end=fresh_decisions_end,
        lr_schedule_decisions_start=lr_schedule_decisions_start,
        lr_schedule_decisions_end=lr_schedule_decisions_end,
        microbatches=sum(step.batches for step in steps),
        homogeneous_microbatches=sum(step.homogeneous_batches for step in steps),
        learning_rate=steps[-1].learning_rate,
        loss=sum(step.loss for step in steps) / step_count,
        policy_loss=sum(step.policy_loss for step in steps) / step_count,
        value_loss=sum(step.value_loss for step in steps) / step_count,
        entropy=-sum(step.entropy_loss for step in steps) / step_count,
        belief_loss=sum(step.belief_loss for step in steps) / step_count,
        ratio_mean=ratio_sum / float(active_tokens),
        approximate_kl=approximate_kl_sum / float(active_tokens),
        clip_fraction=clipped_tokens / float(active_tokens),
        target_kl_exceeded=any(step.target_kl_exceeded for step in steps),
        gradient_norm=max(step.gradient_norm for step in steps),
        host_prepare_task_seconds=sum(step.host_prepare_task_seconds for step in steps),
        host_prepare_wait_seconds=sum(step.host_prepare_wait_seconds for step in steps),
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
        decks=decks,
        step_reports=step_reports,
    )


def _deck_reports(
    window: StatelessOptimizerWindow,
) -> tuple[DeckMacroUpdateReport, ...]:
    counts = Counter(window.deck_digests)
    belief_counts = Counter(
        target.deck_digest for target in window.targets if target.belief_target_valid
    )
    return tuple(
        DeckMacroUpdateReport(
            deck_digest=deck,
            decisions=counts[deck],
            decision_share=counts[deck] / float(len(window.targets)),
            effective_macro_weight=sum(
                weight
                for target, weight in zip(
                    window.targets,
                    window.decision_macro_weights,
                    strict=True,
                )
                if target.deck_digest == deck
            ),
            belief_decisions=belief_counts[deck],
            belief_macro_weight=sum(
                weight
                for target, weight in zip(
                    window.targets,
                    window.belief_macro_weights,
                    strict=True,
                )
                if target.deck_digest == deck
            ),
        )
        for deck in sorted(counts)
    )


def _array_deck_reports(
    window: StatelessArrayOptimizerWindow,
) -> tuple[DeckMacroUpdateReport, ...]:
    """Summarize exact macro weights without materializing target objects."""
    deck_values = np.asarray(window.deck_digests, dtype=np.str_)
    reports: list[DeckMacroUpdateReport] = []
    for deck in np.unique(deck_values):
        mask = deck_values == deck
        belief_mask = mask & window.belief_target_valid
        count = int(np.count_nonzero(mask))
        reports.append(
            DeckMacroUpdateReport(
                deck_digest=str(deck),
                decisions=count,
                decision_share=count / float(window.decision_count),
                effective_macro_weight=float(
                    window.decision_macro_weights[mask].sum(dtype=np.float64)
                ),
                belief_decisions=int(np.count_nonzero(belief_mask)),
                belief_macro_weight=float(
                    window.belief_macro_weights[belief_mask].sum(dtype=np.float64)
                ),
            )
        )
    return tuple(reports)


def _learner_autocast_context(
    device: torch.device,
    precision: StatelessLearnerPrecision,
) -> AbstractContextManager[None]:
    """Return the configured CUDA forward context without touching loss math."""
    if precision not in {"fp32", "bf16"}:
        raise ValueError("stateless learner precision must be fp32 or bf16")
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _move_fp32_master_model(model: _ModelT, device: torch.device) -> _ModelT:
    """Move a learner model while keeping every floating master parameter FP32."""
    model.to(device=device, dtype=torch.float32)
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise TypeError(f"learner master parameter {name} must use float32")
    return model


def stateless_hybrid_shared_learning_rate(
    *,
    update_index: int,
    warmup_start_update: int,
    warmup_updates: int,
    initial_learning_rate: float,
    target_learning_rate: float,
) -> float:
    """Linearly thaw shared parameters over complete rollout windows."""
    if update_index < warmup_start_update:
        raise ValueError("hybrid shared-LR cursor precedes its warmup anchor")
    if warmup_start_update < 0 or warmup_updates < 2:
        raise ValueError("hybrid shared-LR warmup requires at least two updates")
    for value in (initial_learning_rate, target_learning_rate):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("hybrid shared learning rates must be finite and positive")
    if initial_learning_rate > target_learning_rate:
        raise ValueError("hybrid shared learning rate must not decay during thaw")
    offset = min(update_index - warmup_start_update, warmup_updates - 1)
    progress = offset / float(warmup_updates - 1)
    return initial_learning_rate + progress * (
        target_learning_rate - initial_learning_rate
    )


def _validate_optimizer_scope_learning_rates(
    *,
    trainable_scope: Literal["full_model", "private_only", "hybrid"],
    private_learning_rate: float | None,
    shared_initial_learning_rate: float | None,
    shared_learning_rate: float | None,
    shared_warmup_start_update: int | None,
    shared_warmup_updates: int | None,
) -> None:
    hybrid_values = (
        shared_initial_learning_rate,
        shared_learning_rate,
        shared_warmup_start_update,
        shared_warmup_updates,
    )
    if trainable_scope == "full_model":
        if private_learning_rate is not None or any(
            value is not None for value in hybrid_values
        ):
            raise ValueError("full-model learner cannot declare fixed scope rates")
        return
    if private_learning_rate is None or (
        not math.isfinite(private_learning_rate) or private_learning_rate <= 0.0
    ):
        raise ValueError("private learning rate must be finite and positive")
    if trainable_scope == "private_only":
        if any(value is not None for value in hybrid_values):
            raise ValueError("private-only learner cannot declare shared rates")
        return
    if trainable_scope != "hybrid" or any(value is None for value in hybrid_values):
        raise ValueError("hybrid learner requires its complete shared-LR schedule")
    assert shared_initial_learning_rate is not None
    assert shared_learning_rate is not None
    assert shared_warmup_start_update is not None
    assert shared_warmup_updates is not None
    stateless_hybrid_shared_learning_rate(
        update_index=shared_warmup_start_update,
        warmup_start_update=shared_warmup_start_update,
        warmup_updates=shared_warmup_updates,
        initial_learning_rate=shared_initial_learning_rate,
        target_learning_rate=shared_learning_rate,
    )


def _require_hybrid_optimizer_groups(optimizer: torch.optim.Optimizer) -> None:
    roles = tuple(
        group.get(STATELESS_OPTIMIZER_GROUP_ROLE_KEY)
        for group in optimizer.param_groups
    )
    if roles != (STATELESS_PRIVATE_GROUP_ROLE, STATELESS_SHARED_GROUP_ROLE):
        raise ValueError("hybrid optimizer group roles or ordering changed")
    if any(not group["params"] for group in optimizer.param_groups):
        raise ValueError("hybrid optimizer contains an empty parameter group")


def _require_fp32_master_gradients(model: torch.nn.Module) -> None:
    """Reject any gradient that escaped the FP32 master-parameter contract."""
    for name, parameter in model.named_parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise TypeError(f"learner master parameter {name} must use float32")
        gradient = parameter.grad
        if (
            gradient is not None
            and gradient.is_floating_point()
            and gradient.dtype != torch.float32
        ):
            raise TypeError(f"learner master gradient {name} must use float32")


def _require_fp32_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    """Reject reduced-precision Adam statistics before they can mutate masters."""
    for state in optimizer.state.values():
        for name, value in state.items():
            if (
                isinstance(value, Tensor)
                and value.is_floating_point()
                and value.dtype != torch.float32
            ):
                raise TypeError(f"learner optimizer state {name} must use float32")


def _require_finite_fp32_loss(loss: StatelessPpoLoss) -> None:
    """Keep PPO ratios and all scalar objective reductions in FP32."""
    values = (
        loss.loss,
        loss.policy_loss,
        loss.value_loss,
        loss.entropy_loss,
        loss.belief_loss,
        loss.ratio_sum,
        loss.approximate_kl_sum,
    )
    if any(value.dtype != torch.float32 for value in values):
        raise TypeError("stateless learner loss reductions must use float32")
    require_tensor_condition(
        torch.stack(tuple(value.detach() for value in values)).isfinite().all(),
        "stateless learner loss is not finite",
    )


def _validate_training_device(
    device: torch.device,
    *,
    require_h200: bool,
) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("reusable stateless RL training requires local CUDA")
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise RuntimeError("stateless learner CUDA device does not exist")
    index = torch.cuda.current_device() if device.index is None else device.index
    name = torch.cuda.get_device_name(index)
    if require_h200 and "H200" not in name.upper():
        raise RuntimeError("reusable stateless RL training requires NVIDIA H200")


def _adamw_execution_kwargs(
    device: torch.device,
    *,
    fused: bool,
) -> dict[str, bool]:
    """Resolve the explicit AdamW kernel without changing optimizer state."""
    if not fused:
        return {}
    if device.type != "cuda":
        raise ValueError("fused AdamW requires a CUDA learner")
    return {"fused": True}


__all__ = [
    "DeckMacroUpdateReport",
    "SimpleStatelessLearner",
    "SimpleStatelessLearnerUpdate",
    "StatelessOptimizerStepReport",
]
