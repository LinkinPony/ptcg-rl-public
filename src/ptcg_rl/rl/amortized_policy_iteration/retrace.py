"""Boundary-safe distributional Retrace targets for real trajectories."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

from ptcg_rl.rl.amortized_policy_iteration.contracts import RetraceConfig
from ptcg_rl.rl.amortized_policy_iteration.target_builder import Wdl, validate_wdl


@dataclass(frozen=True, slots=True)
class RetraceStep:
    """One contiguous acting-seat decision used by the off-policy recurrence."""

    episode_id: str
    seat: int
    route_id: str
    decision_index: int
    behavior_probability: float
    target_probability: float
    action_q: Wdl
    next_state_value: Wdl
    terminal_wdl: Wdl | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.route_id:
            raise ValueError("Retrace sequence identities must be non-empty")
        if self.seat not in (0, 1):
            raise ValueError("Retrace seat must be zero or one")
        if self.decision_index < 0:
            raise ValueError("Retrace decision index must be non-negative")
        for name, probability in (
            ("behavior", self.behavior_probability),
            ("target", self.target_probability),
        ):
            if not math.isfinite(probability) or probability <= 0.0:
                raise ValueError(f"Retrace {name} probability must be positive")
            if probability > 1.0:
                raise ValueError(f"Retrace {name} probability cannot exceed one")
        validate_wdl(self.action_q)
        validate_wdl(self.next_state_value)
        if self.terminal_wdl is not None:
            validate_wdl(self.terminal_wdl)


def distributional_retrace_targets(
    steps: tuple[RetraceStep, ...],
    *,
    config: RetraceConfig,
) -> tuple[Wdl, ...]:
    """Apply a clipped off-policy recurrence and project each result to W/D/L."""
    validate_retrace_sequence(steps, horizon=config.horizon)
    action_q = torch.tensor(
        [step.action_q for step in steps],
        dtype=torch.float64,
    )
    next_values = torch.tensor(
        [step.next_state_value for step in steps],
        dtype=torch.float64,
    )
    ratios = torch.tensor(
        [step.target_probability / step.behavior_probability for step in steps],
        dtype=torch.float64,
    )
    rho = ratios.clamp(max=config.rho_clip)
    traces = config.lambda_ * ratios.clamp(max=config.c_clip)
    targets = torch.empty_like(action_q)

    for index in range(len(steps) - 1, -1, -1):
        step = steps[index]
        bootstrap = (
            torch.tensor(step.terminal_wdl, dtype=torch.float64)
            if step.terminal_wdl is not None
            else next_values[index]
        )
        corrected = action_q[index] + rho[index] * (
            config.gamma * bootstrap - action_q[index]
        )
        if step.terminal_wdl is None and index + 1 < len(steps):
            corrected = corrected + (
                config.gamma
                * traces[index]
                * (targets[index + 1] - action_q[index + 1])
            )
        targets[index] = project_probability_simplex(corrected)
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in targets)


def tensorized_distributional_retrace_targets(
    *,
    action_q: Tensor,
    next_state_value: Tensor,
    terminal_wdl: Tensor,
    behavior_probability: Tensor,
    target_probability: Tensor,
    sequence_lengths: Sequence[int],
    terminal_mask: Tensor,
    config: RetraceConfig,
) -> Tensor:
    """Apply distributional Retrace to packed sequences on one device.

    Rows belonging to one sequence must be contiguous. ``sequence_lengths``
    describes those packed runs, which lets the recurrence advance over every
    sequence in parallel while retaining the exact horizon and terminal
    semantics of :func:`distributional_retrace_targets`.
    """
    _validate_tensorized_retrace_inputs(
        action_q=action_q,
        next_state_value=next_state_value,
        terminal_wdl=terminal_wdl,
        behavior_probability=behavior_probability,
        target_probability=target_probability,
        sequence_lengths=sequence_lengths,
        terminal_mask=terminal_mask,
        horizon=config.horizon,
    )
    row_count = int(action_q.shape[0])
    lengths = torch.as_tensor(
        tuple(sequence_lengths),
        dtype=torch.long,
        device=action_q.device,
    )
    max_length = max(int(length) for length in sequence_lengths)
    positions = torch.arange(max_length, device=action_q.device)
    valid = positions.unsqueeze(0) < lengths.unsqueeze(1)
    offsets = torch.cumsum(lengths, dim=0) - lengths
    packed_indices = offsets.unsqueeze(1) + positions.unsqueeze(0)
    safe_indices = packed_indices.clamp(max=row_count - 1)

    compute_dtype = torch.float64
    padded_action_q = action_q.to(dtype=compute_dtype)[safe_indices]
    padded_next_value = next_state_value.to(dtype=compute_dtype)[safe_indices]
    padded_terminal = terminal_wdl.to(dtype=compute_dtype)[safe_indices]
    padded_terminal_mask = terminal_mask[safe_indices] & valid
    ratios = (
        target_probability.to(dtype=compute_dtype)[safe_indices]
        / behavior_probability.to(dtype=compute_dtype)[safe_indices]
    )
    rho = ratios.clamp(max=config.rho_clip)
    traces = config.lambda_ * ratios.clamp(max=config.c_clip)
    targets = torch.zeros_like(padded_action_q)

    for position in range(max_length - 1, -1, -1):
        position_valid = valid[:, position]
        terminal = padded_terminal_mask[:, position]
        bootstrap = torch.where(
            terminal.unsqueeze(1),
            padded_terminal[:, position],
            padded_next_value[:, position],
        )
        corrected = padded_action_q[:, position] + rho[:, position].unsqueeze(1) * (
            config.gamma * bootstrap - padded_action_q[:, position]
        )
        if position + 1 < max_length:
            has_successor = position_valid & ~terminal & (position + 1 < lengths)
            correction = (
                config.gamma
                * traces[:, position].unsqueeze(1)
                * (targets[:, position + 1] - padded_action_q[:, position + 1])
            )
            corrected = corrected + torch.where(
                has_successor.unsqueeze(1),
                correction,
                torch.zeros_like(correction),
            )
        projected = _project_probability_simplex_rows_unchecked(corrected)
        targets[:, position] = torch.where(
            position_valid.unsqueeze(1),
            projected,
            torch.zeros_like(projected),
        )
    return targets[valid].to(dtype=action_q.dtype)


def _validate_tensorized_retrace_inputs(
    *,
    action_q: Tensor,
    next_state_value: Tensor,
    terminal_wdl: Tensor,
    behavior_probability: Tensor,
    target_probability: Tensor,
    sequence_lengths: Sequence[int],
    terminal_mask: Tensor,
    horizon: int,
) -> None:
    """Validate the compact tensor recurrence without copying rows to CPU."""
    if horizon <= 0:
        raise ValueError("Retrace horizon must be positive")
    if action_q.ndim != 2 or action_q.shape[1] != 3 or action_q.shape[0] == 0:
        raise ValueError("tensorized Retrace requires non-empty [rows, 3] action Q")
    row_count = int(action_q.shape[0])
    for name, values in (
        ("next state value", next_state_value),
        ("terminal W/D/L", terminal_wdl),
    ):
        if values.shape != action_q.shape:
            raise ValueError(f"tensorized Retrace {name} must align with action Q")
        if values.device != action_q.device:
            raise ValueError(f"tensorized Retrace {name} must share one device")
    for name, values in (
        ("behavior", behavior_probability),
        ("target", target_probability),
    ):
        if values.shape != (row_count,):
            raise ValueError(f"tensorized Retrace {name} probabilities must align")
        if values.device != action_q.device:
            raise ValueError(
                f"tensorized Retrace {name} probabilities must share one device"
            )
    if terminal_mask.shape != (row_count,) or terminal_mask.dtype != torch.bool:
        raise ValueError("tensorized Retrace terminal mask must be boolean per row")
    if terminal_mask.device != action_q.device:
        raise ValueError("tensorized Retrace terminal mask must share one device")
    lengths = tuple(int(length) for length in sequence_lengths)
    if not lengths or any(length <= 0 or length > horizon for length in lengths):
        raise ValueError("tensorized Retrace sequence lengths must fit the horizon")
    if sum(lengths) != row_count:
        raise ValueError("tensorized Retrace sequence lengths must cover every row")
    wdl_tensors = (action_q, next_state_value, terminal_wdl)
    valid_wdl = torch.stack(
        tuple(
            torch.isfinite(values).all()
            & (values >= 0.0).all()
            & torch.isclose(
                values.sum(dim=1),
                torch.ones(row_count, device=values.device, dtype=values.dtype),
                atol=1.0e-6,
                rtol=0.0,
            ).all()
            for values in wdl_tensors
        )
    ).all()
    valid_probabilities = (
        torch.isfinite(behavior_probability).all()
        & torch.isfinite(target_probability).all()
        & (behavior_probability > 0.0).all()
        & (behavior_probability <= 1.0).all()
        & (target_probability > 0.0).all()
        & (target_probability <= 1.0).all()
    )
    if not bool((valid_wdl & valid_probabilities).item()):
        raise ValueError(
            "tensorized Retrace requires normalized W/D/L values and finite "
            "probabilities in (0, 1]"
        )


def project_probability_simplex_rows(values: Tensor) -> Tensor:
    """Project a batch of finite three-vectors onto the probability simplex."""
    if values.ndim != 2 or int(values.shape[1]) != 3:
        raise ValueError("batched simplex projection requires [rows, 3] values")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("simplex projection requires finite values")
    return _project_probability_simplex_rows_unchecked(values)


def _project_probability_simplex_rows_unchecked(values: Tensor) -> Tensor:
    """Project already-validated rows without synchronizing the accelerator."""
    sorted_values, _ = torch.sort(values, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=1) - 1.0
    divisors = torch.arange(
        1,
        int(values.shape[1]) + 1,
        dtype=values.dtype,
        device=values.device,
    )
    active = sorted_values - cumulative / divisors > 0.0
    rho = active.sum(dim=1, dtype=torch.long) - 1
    threshold = cumulative.gather(1, rho.unsqueeze(1)).squeeze(1) / (
        rho.to(dtype=values.dtype) + 1.0
    )
    projected = torch.clamp(values - threshold.unsqueeze(1), min=0.0)
    return projected / projected.sum(dim=1, keepdim=True)


def validate_retrace_sequence(
    steps: tuple[RetraceStep, ...],
    *,
    horizon: int,
) -> None:
    """Reject any sequence crossing episode, acting seat, or private route."""
    if horizon <= 0:
        raise ValueError("Retrace horizon must be positive")
    if not steps:
        raise ValueError("Retrace sequence must not be empty")
    if len(steps) > horizon:
        raise ValueError("Retrace sequence exceeds its configured horizon")
    identity = (steps[0].episode_id, steps[0].seat, steps[0].route_id)
    for index, step in enumerate(steps):
        if (step.episode_id, step.seat, step.route_id) != identity:
            raise ValueError("Retrace sequence crossed an immutable boundary")
        if index > 0 and step.decision_index != steps[index - 1].decision_index + 1:
            raise ValueError("Retrace decisions are not contiguous")
        if step.terminal_wdl is not None and index != len(steps) - 1:
            raise ValueError("Retrace sequence continues after terminal outcome")


def project_probability_simplex(values: Tensor) -> Tensor:
    """Project one finite vector onto the probability simplex."""
    if values.ndim != 1 or int(values.shape[0]) != 3:
        raise ValueError("simplex projection requires one W/D/L vector")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("simplex projection requires finite values")
    sorted_values, _ = torch.sort(values, descending=True)
    cumulative = torch.cumsum(sorted_values, dim=0) - 1.0
    indices = torch.arange(
        1,
        int(values.shape[0]) + 1,
        dtype=values.dtype,
        device=values.device,
    )
    active = sorted_values - cumulative / indices > 0.0
    rho = int(torch.nonzero(active, as_tuple=False)[-1].item())
    threshold = cumulative[rho] / float(rho + 1)
    projected = torch.clamp(values - threshold, min=0.0)
    return projected / projected.sum()


__all__ = [
    "RetraceStep",
    "distributional_retrace_targets",
    "project_probability_simplex",
    "project_probability_simplex_rows",
    "tensorized_distributional_retrace_targets",
    "validate_retrace_sequence",
]
