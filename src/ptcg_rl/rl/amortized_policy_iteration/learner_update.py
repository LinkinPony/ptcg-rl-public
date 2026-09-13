"""H200 optimizer lanes for real Retrace and counterfactual CMPO learning."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
from torch import Tensor

from ptcg_rl.model import AgentPolicyValueNet, build_agent_policy_value_net
from ptcg_rl.model.action_value import wdl_expected_score
from ptcg_rl.profiling import StageTimer, time_stage
from ptcg_rl.rl.amortized_policy_iteration.contracts import (
    AmortizedPolicyIterationConfig,
)
from ptcg_rl.rl.amortized_policy_iteration.counterfactual import (
    CounterfactualRootTarget,
    build_counterfactual_targets,
)
from ptcg_rl.rl.amortized_policy_iteration.improvement_policy import (
    cmpo_cross_entropy,
    cmpo_distribution,
)
from ptcg_rl.rl.amortized_policy_iteration.retrace import (
    tensorized_distributional_retrace_targets,
)
from ptcg_rl.rl.amortized_policy_iteration.tensor_batch import (
    InformationSetBatch,
    collate_information_sets,
    ordered_rows,
    select_information_set_batch,
)
from ptcg_rl.rl.experience import GameTrajectory
from ptcg_rl.rl.learner import (
    PolicyIterationTrajectoryBatch,
    collate_policy_iteration_trajectories,
)

AutocastMode = Literal["bf16", "off"]


@dataclass(frozen=True, slots=True)
class PolicyIterationUpdate:
    """One additional optimizer update and its direct diagnostics."""

    kind: Literal["real_retrace", "counterfactual_cmpo"]
    rows: int
    roots: int
    loss: float
    q_wdl_loss: float
    state_wdl_loss: float
    cmpo_loss: float
    grad_norm: float
    improvement_rows: int = 0
    valid_worlds: int = 0
    omitted_worlds: int = 0


@dataclass(frozen=True, slots=True)
class _RetraceSequence:
    """One horizon-bounded run and its target-network bootstrap rows."""

    rows: tuple[int, ...]
    next_rows: tuple[int, ...]
    terminal_mask: tuple[bool, ...]


@dataclass(frozen=True, slots=True)
class _RetraceMicrobatch:
    """Whole Retrace sequences packed under one current-model forward."""

    sequences: tuple[_RetraceSequence, ...]

    @property
    def rows(self) -> tuple[int, ...]:
        """Return source rows in packed sequence order."""
        return tuple(row for sequence in self.sequences for row in sequence.rows)

    @property
    def next_rows(self) -> tuple[int, ...]:
        """Return target-state bootstrap rows aligned with ``rows``."""
        return tuple(row for sequence in self.sequences for row in sequence.next_rows)

    @property
    def terminal_mask(self) -> tuple[bool, ...]:
        """Return true only for actual episode endpoints."""
        return tuple(
            terminal
            for sequence in self.sequences
            for terminal in sequence.terminal_mask
        )

    @property
    def sequence_lengths(self) -> tuple[int, ...]:
        """Return the packed recurrence boundaries."""
        return tuple(len(sequence.rows) for sequence in self.sequences)


class AmortizedPolicyIterationLearner:
    """Own a frozen target network and run the two selected learner lanes."""

    def __init__(
        self,
        model: AgentPolicyValueNet,
        *,
        config: AmortizedPolicyIterationConfig,
        device: torch.device | str,
        autocast: AutocastMode,
        grad_clip_norm: float,
    ) -> None:
        if not config.enabled or not model.config.action_value.enabled:
            raise ValueError("policy-iteration learner requires an enabled Q head")
        self._model = model
        self._config = config
        self._device = torch.device(device)
        self._autocast = autocast
        self._grad_clip_norm = grad_clip_norm
        self._target_model = build_agent_policy_value_net(model.config).to(self._device)
        self._target_model.load_state_dict(model.state_dict(), strict=True)
        self._target_model.eval()
        self._target_model.requires_grad_(False)
        self._pending: deque[CounterfactualRootTarget] = deque()
        self._optimizer_updates = 0
        self._restored_pending_targets = False
        self._restored_training_schema_version: int | None = None

    @property
    def pending_counterfactual_roots(self) -> int:
        """Return student-safe targets waiting for bounded optimizer work."""
        return len(self._pending)

    @property
    def restored_pending_targets(self) -> bool:
        """Return whether an exact sidecar, rather than replay, restored pending work."""
        return self._restored_pending_targets

    @property
    def restored_training_schema_version(self) -> int | None:
        """Return the loaded auxiliary schema, if this learner was restored."""
        return self._restored_training_schema_version

    def training_state_dict(self) -> dict[str, Any]:
        """Return the target-network state required for an exact learner resume."""
        return {
            "schema_version": 2,
            "optimizer_updates": self._optimizer_updates,
            "target_model_state_dict": {
                name: tensor.detach().to(device="cpu", copy=True).contiguous()
                for name, tensor in self._target_model.state_dict().items()
            },
            # These targets were durably constructed but have not yet crossed
            # an optimizer step. Persisting the bounded student-safe backlog
            # prevents exact resume from replaying already consumed targets.
            "pending_targets": tuple(self._pending),
        }

    def load_training_state_dict(self, payload: Any) -> None:
        """Restore the persistent target critic from a bound resume sidecar."""
        if not isinstance(payload, dict) or payload.get("schema_version") not in (
            1,
            2,
        ):
            raise ValueError("policy-iteration resume state has an invalid schema")
        optimizer_updates = payload.get("optimizer_updates")
        if (
            not isinstance(optimizer_updates, int)
            or isinstance(optimizer_updates, bool)
            or optimizer_updates < 0
        ):
            raise ValueError("policy-iteration optimizer update count is invalid")
        state = payload.get("target_model_state_dict")
        if not isinstance(state, dict):
            raise ValueError("policy-iteration target model state is missing")
        self._target_model.load_state_dict(state, strict=True)
        self._target_model.to(self._device)
        self._target_model.eval()
        self._target_model.requires_grad_(False)
        self._optimizer_updates = optimizer_updates
        self._pending.clear()
        self._restored_training_schema_version = int(payload["schema_version"])
        if payload["schema_version"] == 2:
            pending = payload.get("pending_targets")
            if not isinstance(pending, (tuple, list)) or any(
                not isinstance(target, CounterfactualRootTarget) for target in pending
            ):
                raise ValueError("policy-iteration pending target state is invalid")
            self._pending.extend(pending)
            self._restored_pending_targets = True
        else:
            self._restored_pending_targets = False

    def ingest_native_results(
        self,
        results: Sequence[Any],
        *,
        timer: StageTimer | None = None,
    ) -> tuple[dict[str, int], tuple[CounterfactualRootTarget, ...]]:
        """Bootstrap native leaf rows on H200 and retain only aggregated targets."""
        from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
            NativeReanalysisResult,
        )

        typed = tuple(
            result for result in results if isinstance(result, NativeReanalysisResult)
        )
        invalid_type = len(results) - len(typed)
        with time_stage(timer, "learner_api_native_target_build"):
            targets = build_counterfactual_targets(
                typed,
                target_model=self._target_model,
                device=self._device,
                autocast=self._autocast,
            )
        self._pending.extend(targets)
        return (
            {
                "results_seen": len(results),
                "invalid_result_type": invalid_type,
                "result_errors": sum(bool(result.error_message) for result in typed),
                "targets_admitted": len(targets),
                "pending_targets": len(self._pending),
            },
            targets,
        )

    def admit_replay_targets(
        self,
        targets: Sequence[CounterfactualRootTarget],
    ) -> None:
        """Re-admit verified retained targets after an exact infrastructure resume."""
        self._pending.extend(targets)

    def update_real_trajectories(
        self,
        trajectories: Sequence[GameTrajectory],
        *,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        timer: StageTimer | None = None,
    ) -> PolicyIterationUpdate | None:
        """Ground categorical Q on every ordinary and improved real trajectory."""
        with time_stage(timer, "learner_api_real_collate"):
            batch = collate_policy_iteration_trajectories(
                trajectories,
                max_decisions=self._config.learner.real_max_decisions,
                device=self._device,
            )
        if batch is None:
            return None
        information_sets = InformationSetBatch(
            states=batch.states,
            options=batch.options,
            decks=batch.decks,
        )
        action_groups = tuple((action,) for action in batch.actions)
        action_ordered_rows = ordered_rows(batch.options)
        with (
            torch.inference_mode(),
            time_stage(timer, "learner_api_real_target_forward"),
            self._autocast_context(self._target_model),
        ):
            target = self._target_model.evaluate_action_values(
                batch.states,
                batch.options,
                action_groups,
                ordered_rows=action_ordered_rows,
                decks=batch.decks,
                validate_candidate_actions=False,
            )
            target_action_q = target.probabilities.float()
            target_state = torch.softmax(target.state_logits.float(), dim=-1)
        with time_stage(timer, "learner_api_real_sequence_plan"):
            microbatches = _real_retrace_microbatches(
                batch,
                horizon=self._config.retrace.horizon,
                microbatch_size=self._config.learner.real_microbatch_size,
            )
        optimizer.zero_grad(set_to_none=True)
        q_total = 0.0
        state_total = 0.0
        row_count = batch.row_count
        for microbatch in microbatches:
            indices = torch.tensor(
                microbatch.rows,
                dtype=torch.long,
                device=self._device,
            )
            next_indices = torch.tensor(
                microbatch.next_rows,
                dtype=torch.long,
                device=self._device,
            )
            inputs = select_information_set_batch(
                information_sets,
                indices,
                deck_indices=microbatch.rows,
            )
            actions = tuple(batch.actions[index] for index in microbatch.rows)
            with (
                time_stage(timer, "learner_api_real_current_forward"),
                self._autocast_context(self._model),
            ):
                evaluation = self._model.evaluate_action_values(
                    inputs.states,
                    inputs.options,
                    tuple((action,) for action in actions),
                    ordered_rows=action_ordered_rows.index_select(0, indices),
                    decks=inputs.decks,
                    validate_candidate_actions=False,
                )
            if evaluation.action_logprobs is None:
                raise RuntimeError("real Retrace evaluation omitted policy density")
            with time_stage(timer, "learner_api_real_retrace"):
                target_wdl = tensorized_distributional_retrace_targets(
                    action_q=target_action_q.index_select(0, indices),
                    next_state_value=target_state.index_select(0, next_indices),
                    terminal_wdl=batch.terminal_wdl.index_select(0, indices),
                    behavior_probability=batch.behavior_probabilities.index_select(
                        0,
                        indices,
                    ),
                    target_probability=evaluation.action_logprobs.detach()
                    .float()
                    .exp()
                    .clamp(min=1.0e-30, max=1.0),
                    sequence_lengths=microbatch.sequence_lengths,
                    terminal_mask=torch.tensor(
                        microbatch.terminal_mask,
                        dtype=torch.bool,
                        device=self._device,
                    ),
                    config=self._config.retrace,
                )
            with self._autocast_context(self._model):
                q_loss = _soft_cross_entropy(
                    evaluation.logits,
                    target_wdl,
                )
                state_loss = _soft_cross_entropy(
                    evaluation.state_logits,
                    batch.terminal_wdl.index_select(0, indices),
                )
                loss = (
                    self._config.losses.retrace_wdl * q_loss
                    + self._config.losses.state_wdl * state_loss
                )
                scaled = loss * (len(microbatch.rows) / row_count)
            _ensure_finite(scaled, name="real policy-iteration loss")
            with time_stage(timer, "learner_api_real_backward"):
                torch.autograd.backward(scaled)
            q_total += float(q_loss.detach()) * len(microbatch.rows)
            state_total += float(state_loss.detach()) * len(microbatch.rows)
        grad_norm = self._finish_optimizer_step(
            optimizer,
            lr_scheduler,
            timer=timer,
        )
        total = (
            self._config.losses.retrace_wdl * q_total
            + self._config.losses.state_wdl * state_total
        ) / row_count
        return PolicyIterationUpdate(
            kind="real_retrace",
            rows=row_count,
            roots=0,
            loss=total,
            q_wdl_loss=q_total / row_count,
            state_wdl_loss=state_total / row_count,
            cmpo_loss=0.0,
            grad_norm=grad_norm,
            improvement_rows=sum(
                kind == "improvement" for kind in batch.behavior_kinds
            ),
        )

    def update_counterfactuals(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        timer: StageTimer | None = None,
    ) -> tuple[PolicyIterationUpdate, ...]:
        """Train Q and amortize CMPO from a fixed bounded number of ready roots."""
        updates = []
        for _ in range(self._config.learner.counterfactual_updates_per_iteration):
            roots = tuple(
                self._pending.popleft()
                for _index in range(
                    min(
                        len(self._pending),
                        self._config.learner.counterfactual_roots_per_update,
                    )
                )
            )
            if not roots:
                break
            updates.append(
                self._update_counterfactual_batch(
                    roots,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    timer=timer,
                )
            )
        return tuple(updates)

    def _update_counterfactual_batch(
        self,
        roots: tuple[CounterfactualRootTarget, ...],
        *,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        timer: StageTimer | None = None,
    ) -> PolicyIterationUpdate:
        batch = collate_information_sets(
            states=tuple(row.root.state for row in roots),
            options=tuple(row.root.options for row in roots),
            min_counts=tuple(row.root.min_count for row in roots),
            max_counts=tuple(row.root.max_count for row in roots),
            decks=tuple(row.root.deck for row in roots),
            device=self._device,
        )
        action_groups = tuple(row.actions for row in roots)
        target_wdl = torch.tensor(
            [wdl for row in roots for wdl in row.target_wdl],
            dtype=torch.float32,
            device=self._device,
        )
        q_prop = torch.tensor(
            [value for row in roots for value in row.proposal_probabilities],
            dtype=torch.float32,
            device=self._device,
        )
        old_probabilities = torch.tensor(
            [value for row in roots for value in row.old_policy_probabilities],
            dtype=torch.float32,
            device=self._device,
        )
        optimizer.zero_grad(set_to_none=True)
        with (
            time_stage(timer, "learner_api_counterfactual_forward"),
            self._autocast_context(self._model),
        ):
            conditioned = self._model.encode_conditioned_state(
                batch.states,
                batch.decks,
            )
            context = self._model.policy_context_from_conditioned(
                conditioned,
                batch.options,
            )
            evaluation = self._model.evaluate_action_values_from_context(
                context,
                batch.options,
                action_groups,
                ordered_rows=ordered_rows(batch.options),
                decks=batch.decks,
            )
            if evaluation.action_logprobs is None:
                raise RuntimeError("counterfactual update omitted policy density")
            q_loss = _soft_cross_entropy(evaluation.logits, target_wdl)
            state_scores = wdl_expected_score(
                torch.softmax(
                    self._model.action_value_state_logits_from_conditioned(
                        conditioned
                    ).float(),
                    dim=-1,
                )
            ).detach()
            improvement = cmpo_distribution(
                wdl_expected_score(target_wdl),
                state_scores,
                old_probabilities,
                q_prop,
                candidate_counts=evaluation.candidate_counts,
                exhaustive_rows=tuple(row.exhaustive for row in roots),
                config=self._config.cmpo,
            )
            policy_loss = cmpo_cross_entropy(
                evaluation.action_logprobs.float(),
                improvement,
            )
            loss = (
                self._config.losses.counterfactual_wdl * q_loss
                + self._config.losses.cmpo * policy_loss
            )
        _ensure_finite(loss, name="counterfactual policy-iteration loss")
        with time_stage(timer, "learner_api_counterfactual_backward"):
            torch.autograd.backward(loss)
        grad_norm = self._finish_optimizer_step(
            optimizer,
            lr_scheduler,
            timer=timer,
        )
        return PolicyIterationUpdate(
            kind="counterfactual_cmpo",
            rows=sum(len(row.actions) for row in roots),
            roots=len(roots),
            loss=float(loss.detach()),
            q_wdl_loss=float(q_loss.detach()),
            state_wdl_loss=0.0,
            cmpo_loss=float(policy_loss.detach()),
            grad_norm=grad_norm,
            valid_worlds=sum(row.valid_worlds for row in roots),
            omitted_worlds=sum(row.omitted_worlds for row in roots),
        )

    def _finish_optimizer_step(
        self,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        *,
        timer: StageTimer | None = None,
    ) -> float:
        with time_stage(timer, "learner_api_grad_clip"):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self._model.parameters(),
                max_norm=self._grad_clip_norm,
                error_if_nonfinite=True,
            )
        with time_stage(timer, "learner_api_optimizer"):
            optimizer.step()
        _advance_active_optimizer_group_clocks(optimizer)
        with time_stage(timer, "learner_api_lr_scheduler"):
            lr_scheduler.step()
        self._optimizer_updates += 1
        if self._optimizer_updates % self._config.target_network.update_interval == 0:
            with time_stage(timer, "learner_api_target_update"):
                self._update_target_model()
        return float(grad_norm.detach())

    def _update_target_model(self) -> None:
        tau = self._config.target_network.polyak
        with torch.no_grad():
            for target_parameter, source_parameter in zip(
                self._target_model.parameters(),
                self._model.parameters(),
                strict=True,
            ):
                target_parameter.lerp_(source_parameter, tau)
            for target_buffer, source_buffer in zip(
                self._target_model.buffers(),
                self._model.buffers(),
                strict=True,
            ):
                target_buffer.copy_(source_buffer)

    def _autocast_context(
        self,
        model: AgentPolicyValueNet,
    ) -> AbstractContextManager[None]:
        if self._autocast == "off" or next(model.parameters()).device.type != "cuda":
            return nullcontext()
        return cast(
            AbstractContextManager[None],
            torch.autocast(device_type="cuda", dtype=torch.bfloat16),
        )


def _real_retrace_microbatches(
    batch: PolicyIterationTrajectoryBatch,
    *,
    horizon: int,
    microbatch_size: int,
) -> tuple[_RetraceMicrobatch, ...]:
    """Pack complete horizon runs without crossing immutable identities."""
    if horizon <= 0 or microbatch_size <= 0:
        raise ValueError("real Retrace horizon and microbatch size must be positive")
    grouped: dict[tuple[str, int, str], list[int]] = defaultdict(list)
    for row, identity in enumerate(
        zip(batch.game_ids, batch.seats, batch.route_ids, strict=True)
    ):
        grouped[identity].append(row)

    sequences: list[_RetraceSequence] = []
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: batch.decision_indices[row])
        for chunk_start in range(0, len(group_rows), horizon):
            rows = tuple(group_rows[chunk_start : chunk_start + horizon])
            for previous, current in zip(rows, rows[1:], strict=False):
                if (
                    batch.decision_indices[current]
                    != batch.decision_indices[previous] + 1
                ):
                    raise ValueError("Retrace decisions are not contiguous")
            next_rows: list[int] = []
            terminal_mask: list[bool] = []
            for local, row in enumerate(rows):
                sequence_position = chunk_start + local
                terminal = sequence_position + 1 == len(group_rows)
                next_rows.append(row if terminal else group_rows[sequence_position + 1])
                terminal_mask.append(terminal)
            sequences.append(
                _RetraceSequence(
                    rows=rows,
                    next_rows=tuple(next_rows),
                    terminal_mask=tuple(terminal_mask),
                )
            )

    packed: list[_RetraceMicrobatch] = []
    pending: list[_RetraceSequence] = []
    pending_rows = 0
    for sequence in sequences:
        sequence_rows = len(sequence.rows)
        if pending and pending_rows + sequence_rows > microbatch_size:
            packed.append(_RetraceMicrobatch(sequences=tuple(pending)))
            pending = []
            pending_rows = 0
        pending.append(sequence)
        pending_rows += sequence_rows
        if pending_rows >= microbatch_size:
            packed.append(_RetraceMicrobatch(sequences=tuple(pending)))
            pending = []
            pending_rows = 0
    if pending:
        packed.append(_RetraceMicrobatch(sequences=tuple(pending)))
    if sum(len(microbatch.rows) for microbatch in packed) != batch.row_count:
        raise RuntimeError("real Retrace sequence plan did not cover every row")
    return tuple(packed)


def _soft_cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.shape != targets.shape or logits.ndim != 2 or logits.shape[1] != 3:
        raise ValueError("categorical W/D/L logits and targets must align")
    return (
        -(targets.detach() * torch.log_softmax(logits.float(), dim=-1))
        .sum(dim=-1)
        .mean()
    )


def _ensure_finite(value: Tensor, *, name: str) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{name} is non-finite")


def _advance_active_optimizer_group_clocks(
    optimizer: torch.optim.Optimizer,
) -> None:
    for group in optimizer.param_groups:
        if "scheduler_active_updates" not in group:
            continue
        if any(parameter.grad is not None for parameter in group["params"]):
            group["scheduler_active_updates"] = (
                int(group["scheduler_active_updates"]) + 1
            )


__all__ = ["AmortizedPolicyIterationLearner", "PolicyIterationUpdate"]
