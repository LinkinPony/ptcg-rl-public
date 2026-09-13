"""Fixed-budget dense-teacher objectives for topology transitions."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor


class TransitionDistillationConfig(BaseModel):
    """Pre-registered optimizer budget and independent teacher targets."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    optimizer_updates: int = 0
    sequence_kl_coef: float = 1.0
    prefix_kl_coef: float = 0.0
    root_value_coef: float = 0.0
    prefix_value_coef: float = 0.0
    action_wdl_coef: float = 0.0
    engine_return_coef: float = 0.0

    @field_validator("optimizer_updates")
    @classmethod
    def nonnegative_updates(cls, value: int) -> int:
        """Reject a negative fixed transition budget."""
        if value < 0:
            raise ValueError("transition distillation updates must be non-negative")
        return value

    @field_validator(
        "sequence_kl_coef",
        "prefix_kl_coef",
        "root_value_coef",
        "prefix_value_coef",
        "action_wdl_coef",
        "engine_return_coef",
    )
    @classmethod
    def nonnegative_coefficient(cls, value: float) -> float:
        """Reject non-finite or negative objective weights."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(
                "distillation coefficients must be finite and non-negative"
            )
        return value

    @model_validator(mode="after")
    def active_budget_has_objective(self) -> TransitionDistillationConfig:
        """Do not spend an enabled transition budget on an empty objective."""
        if self.optimizer_updates > 0 and not any(
            coefficient > 0.0
            for coefficient in (
                self.sequence_kl_coef,
                self.prefix_kl_coef,
                self.root_value_coef,
                self.prefix_value_coef,
                self.action_wdl_coef,
                self.engine_return_coef,
            )
        ):
            raise ValueError("active transition distillation has no objective")
        return self

    def active(self, update_index: int) -> bool:
        """Return whether this exact optimizer index belongs to distillation."""
        if update_index < 0:
            raise ValueError("optimizer update index must be non-negative")
        return update_index < self.optimizer_updates


@dataclass(frozen=True)
class TransitionDistillationLosses:
    """Independent mean losses and their reduction-unit counts."""

    sequence_kl: Tensor
    prefix_kl: Tensor
    count_kl: Tensor
    root_value: Tensor
    prefix_value: Tensor
    action_wdl: Tensor
    engine_return: Tensor
    rows: int
    prefixes: int
    count_rows: int
    value_prefixes: int


@dataclass(frozen=True)
class PolicyForwardKLLosses:
    """Teacher-to-student policy KL over the visited legal prefixes."""

    row_kl: Tensor
    sequence_kl: Tensor
    prefix_kl: Tensor
    count_kl: Tensor
    rows: int
    prefixes: int
    count_rows: int


def visited_option_prefix_mask(
    actions: Sequence[Sequence[int]],
    *,
    max_counts: Tensor,
    count_mask: Tensor,
) -> Tensor:
    """Return real option/STOP decisions, excluding forced max termination."""
    rows = len(actions)
    if (
        max_counts.ndim != 1
        or count_mask.ndim != 1
        or max_counts.shape != count_mask.shape
        or int(max_counts.shape[0]) != rows
        or count_mask.dtype != torch.bool
    ):
        raise ValueError("policy KL prefix inputs must align with action rows")
    count_rows = tuple(
        bool(value) for value in count_mask.detach().to(device="cpu").tolist()
    )
    maximum_counts = tuple(
        int(value) for value in max_counts.detach().to(device="cpu").tolist()
    )
    prefix_counts = tuple(
        len(action) if count_row else len(action) + int(len(action) < maximum_count)
        for action, maximum_count, count_row in zip(
            actions,
            maximum_counts,
            count_rows,
            strict=True,
        )
    )
    width = max(prefix_counts, default=0)
    mask = torch.zeros(
        (rows, width),
        dtype=torch.bool,
        device=max_counts.device,
    )
    for row_index, prefix_count in enumerate(prefix_counts):
        mask[row_index, :prefix_count] = True
    return mask


def policy_forward_kl_losses(
    *,
    student_step_logits: Sequence[Tensor],
    teacher_step_logits: Sequence[Tensor],
    student_count_logits: Tensor | None,
    teacher_count_logits: Tensor | None,
    option_prefix_mask: Tensor,
    count_mask: Tensor,
    reference: Tensor,
    rows: int,
) -> PolicyForwardKLLosses:
    """Compute forward KL with the shared transition-distillation semantics."""
    if rows <= 0:
        raise ValueError("policy forward KL requires at least one row")
    option_row_kl, option_kl_sum, option_prefixes = _visited_prefix_forward_kl(
        student_step_logits,
        teacher_step_logits,
        option_prefix_mask,
        rows=rows,
        reference=reference,
    )
    count_row_kl, count_kl, count_rows = _count_forward_kl(
        student_count_logits,
        teacher_count_logits,
        count_mask,
        rows=rows,
        reference=reference,
    )
    row_kl = option_row_kl + count_row_kl
    prefixes = option_prefixes + count_rows
    return PolicyForwardKLLosses(
        row_kl=row_kl,
        sequence_kl=row_kl.mean(),
        prefix_kl=(
            (option_kl_sum + count_row_kl.sum()) / prefixes
            if prefixes > 0
            else reference.sum() * 0.0
        ),
        count_kl=count_kl,
        rows=rows,
        prefixes=prefixes,
        count_rows=count_rows,
    )


def transition_distillation_losses(
    *,
    student_step_logits: Sequence[Tensor],
    teacher_step_logits: Sequence[Tensor],
    student_count_logits: Tensor | None,
    teacher_count_logits: Tensor | None,
    option_prefix_mask: Tensor,
    count_mask: Tensor,
    student_values: Tensor,
    teacher_values: Tensor,
    student_prefix_values: Tensor | None,
    teacher_prefix_values: Tensor | None,
    token_mask: Tensor | None,
    actions: Sequence[Sequence[int]],
    returns: Tensor,
    student_action_wdl_logits: Tensor | None = None,
    teacher_action_wdl_logits: Tensor | None = None,
) -> TransitionDistillationLosses:
    """Compute forward teacher KL plus root/prefix/Q/return regression targets."""
    rows = len(actions)
    if rows <= 0:
        raise ValueError("transition distillation requires at least one row")
    policy_kl = policy_forward_kl_losses(
        student_step_logits=student_step_logits,
        teacher_step_logits=teacher_step_logits,
        student_count_logits=student_count_logits,
        teacher_count_logits=teacher_count_logits,
        option_prefix_mask=option_prefix_mask,
        count_mask=count_mask,
        reference=student_values,
        rows=rows,
    )
    root_value = _mean_squared_error(
        student_values,
        teacher_values,
        label="root values",
    )
    prefix_value, value_prefixes = _prefix_value_loss(
        student_prefix_values,
        teacher_prefix_values,
        token_mask,
    )
    action_wdl = _optional_forward_kl(
        student_action_wdl_logits,
        teacher_action_wdl_logits,
        reference=student_values,
        label="action WDL",
    )
    engine_return = _mean_squared_error(
        student_values,
        returns.to(device=student_values.device, dtype=student_values.dtype),
        label="engine returns",
    )
    return TransitionDistillationLosses(
        sequence_kl=policy_kl.sequence_kl,
        prefix_kl=policy_kl.prefix_kl,
        count_kl=policy_kl.count_kl,
        root_value=root_value,
        prefix_value=prefix_value,
        action_wdl=action_wdl,
        engine_return=engine_return,
        rows=rows,
        prefixes=policy_kl.prefixes,
        count_rows=policy_kl.count_rows,
        value_prefixes=value_prefixes,
    )


def _visited_prefix_forward_kl(
    student_steps: Sequence[Tensor],
    teacher_steps: Sequence[Tensor],
    prefix_mask: Tensor,
    *,
    rows: int,
    reference: Tensor,
) -> tuple[Tensor, Tensor, int]:
    if len(student_steps) != len(teacher_steps):
        raise ValueError("student and teacher decode steps must align")
    if (
        prefix_mask.dtype != torch.bool
        or prefix_mask.ndim != 2
        or prefix_mask.shape != (rows, len(student_steps))
    ):
        raise ValueError("transition option-prefix mask must align with logits")
    per_row = reference.new_zeros((rows,), dtype=torch.float32)
    prefix_sum = per_row.new_zeros(())
    prefix_count = 0
    for step_index, (student, teacher) in enumerate(
        zip(student_steps, teacher_steps, strict=True)
    ):
        if student.ndim != 2 or student.shape != teacher.shape:
            raise ValueError("student and teacher next-option logits must align")
        step_mask = prefix_mask[:, step_index].to(device=student.device)
        active = torch.nonzero(step_mask, as_tuple=False).flatten()
        if int(active.numel()) == 0:
            continue
        step_kl = _forward_categorical_kl(
            student.index_select(0, active),
            teacher.index_select(0, active),
        )
        per_row = per_row.index_add(0, active, step_kl)
        prefix_sum = prefix_sum + step_kl.sum()
        prefix_count += int(active.numel())
    return per_row, prefix_sum, prefix_count


def _count_forward_kl(
    student: Tensor | None,
    teacher: Tensor | None,
    count_mask: Tensor,
    *,
    rows: int,
    reference: Tensor,
) -> tuple[Tensor, Tensor, int]:
    if (
        count_mask.dtype != torch.bool
        or count_mask.ndim != 1
        or count_mask.shape[0] != rows
    ):
        raise ValueError("transition count mask must have shape [rows]")
    active = torch.nonzero(
        count_mask.to(device=reference.device),
        as_tuple=False,
    ).flatten()
    count = int(active.numel())
    per_row = reference.new_zeros((rows,), dtype=torch.float32)
    if count == 0:
        return per_row, reference.sum() * 0.0, 0
    if (
        student is None
        or teacher is None
        or student.ndim != 2
        or student.shape != teacher.shape
        or student.shape[0] != rows
    ):
        raise ValueError("student and teacher count logits must align")
    active = active.to(device=student.device)
    active_kl = _forward_categorical_kl(
        student.index_select(0, active),
        teacher.index_select(0, active),
    )
    per_row = per_row.index_add(0, active, active_kl)
    return per_row, active_kl.mean(), count


def _prefix_value_loss(
    student: Tensor | None,
    teacher: Tensor | None,
    token_mask: Tensor | None,
) -> tuple[Tensor, int]:
    if student is None or teacher is None or token_mask is None:
        raise ValueError("transition distillation requires prefix-value evidence")
    if (
        student.ndim != 2
        or student.shape != teacher.shape
        or token_mask.shape != student.shape
        or token_mask.dtype != torch.bool
    ):
        raise ValueError("transition prefix values and masks must align")
    mask = token_mask.to(device=student.device)
    count = int(mask.sum().item())
    if count <= 0:
        raise ValueError("transition distillation has no active prefix value")
    student_active = student.masked_select(mask).float()
    teacher_active = teacher.to(device=student.device).masked_select(mask).float()
    return torch.mean((student_active - teacher_active).square()), count


def _optional_forward_kl(
    student: Tensor | None,
    teacher: Tensor | None,
    *,
    reference: Tensor,
    label: str,
) -> Tensor:
    if student is None and teacher is None:
        return reference.sum() * 0.0
    if student is None or teacher is None or student.ndim != 2:
        raise ValueError(f"student and teacher {label} logits must align")
    if student.shape != teacher.shape:
        raise ValueError(f"student and teacher {label} logits must align")
    return _forward_categorical_kl(student, teacher).mean()


def _forward_categorical_kl(student_logits: Tensor, teacher_logits: Tensor) -> Tensor:
    student_logprobs = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_logprobs = torch.log_softmax(teacher_logits.float(), dim=-1)
    teacher_probs = torch.softmax(teacher_logits.float(), dim=-1)
    finite = torch.isfinite(student_logprobs) & torch.isfinite(teacher_logprobs)
    safe_probs = torch.where(finite, teacher_probs, torch.zeros_like(teacher_probs))
    safe_teacher = torch.where(
        finite,
        teacher_logprobs,
        torch.zeros_like(teacher_logprobs),
    )
    safe_student = torch.where(
        finite,
        student_logprobs,
        torch.zeros_like(student_logprobs),
    )
    return (safe_probs * (safe_teacher - safe_student)).sum(dim=-1)


def _mean_squared_error(student: Tensor, teacher: Tensor, *, label: str) -> Tensor:
    if student.shape != teacher.shape:
        raise ValueError(f"student and teacher {label} must align")
    return torch.mean((student.float() - teacher.detach().float()).square())


__all__ = [
    "PolicyForwardKLLosses",
    "TransitionDistillationConfig",
    "TransitionDistillationLosses",
    "policy_forward_kl_losses",
    "transition_distillation_losses",
    "visited_option_prefix_mask",
]
