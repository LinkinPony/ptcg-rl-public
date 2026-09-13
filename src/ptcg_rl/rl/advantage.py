"""Advantage estimation utilities for PPO-style RL updates."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class GaeConfig(BaseModel):
    """Configuration for policy advantages and critic value targets."""

    model_config = ConfigDict(extra="forbid")

    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_target_mode: Literal["gae", "monte_carlo"] = "gae"
    credit_unit: Literal["decision", "decode_token"] = "decision"
    intra_prompt_gamma: float = 1.0
    intra_prompt_gae_lambda: float = 0.0
    normalize_advantages: bool = True
    normalize_epsilon: float = 1e-8
    prize_diff_shaping_beta: float = 0.0

    @field_validator(
        "gamma",
        "gae_lambda",
        "intra_prompt_gamma",
        "intra_prompt_gae_lambda",
    )
    @classmethod
    def valid_unit_interval(cls, value: float) -> float:
        """Reject discount parameters outside [0, 1]."""
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("value must be finite and in [0, 1]")
        return value

    @field_validator("normalize_epsilon")
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject non-positive normalization guards."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("normalize_epsilon must be finite and positive")
        return value

    @field_validator("prize_diff_shaping_beta")
    @classmethod
    def valid_shaping_beta(cls, value: float) -> float:
        """Reject invalid shaping coefficients."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("prize_diff_shaping_beta must be finite and non-negative")
        return value


@dataclass(frozen=True)
class AdvantageStep:
    """One policy decision with rollout metadata needed for GAE."""

    game_id: str
    seat: int
    decision_index: int
    terminal_reward: float
    value_pred: float
    potential: float | None = None
    terminal_potential: float | None = None


@dataclass(frozen=True)
class AdvantageEstimate:
    """Policy GAE and configured value target for one input decision."""

    input_index: int
    game_id: str
    seat: int
    decision_index: int
    advantage: float
    normalized_advantage: float
    return_value: float


@dataclass(frozen=True)
class AdvantageBatch:
    """Batch of advantage estimates in the same order as the input steps."""

    estimates: tuple[AdvantageEstimate, ...]
    advantage_mean: float
    advantage_std: float

    @property
    def advantages(self) -> tuple[float, ...]:
        """Return raw advantages aligned to input order."""
        return tuple(estimate.advantage for estimate in self.estimates)

    @property
    def normalized_advantages(self) -> tuple[float, ...]:
        """Return normalized advantages aligned to input order."""
        return tuple(estimate.normalized_advantage for estimate in self.estimates)

    @property
    def returns(self) -> tuple[float, ...]:
        """Return critic value targets aligned to input order."""
        return tuple(estimate.return_value for estimate in self.estimates)


@dataclass(frozen=True)
class _IndexedStep:
    input_index: int
    step: AdvantageStep


def advantage_step_from_row(row: Mapping[str, object]) -> AdvantageStep:
    """Build an ``AdvantageStep`` from a finalized RL trajectory row."""
    return AdvantageStep(
        game_id=str(_required(row, "game_id")),
        seat=_as_int(row.get("rollout_seat", row.get("player_index"))),
        decision_index=_as_int(_required(row, "decision_index")),
        terminal_reward=_as_float(_required(row, "reward")),
        value_pred=_as_float(_required(row, "value_pred")),
        potential=_as_optional_float(row.get("current_prize_diff")),
        terminal_potential=_as_optional_float(row.get("terminal_prize_diff")),
    )


def compute_gae_from_rows(
    rows: Sequence[Mapping[str, object]],
    config: GaeConfig | None = None,
) -> AdvantageBatch:
    """Compute GAE from finalized RL trajectory rows."""
    return compute_gae(
        tuple(advantage_step_from_row(row) for row in rows),
        config=config,
    )


def compute_gae(
    steps: Sequence[AdvantageStep],
    config: GaeConfig | None = None,
) -> AdvantageBatch:
    """Compute GAE grouped by game and rollout seat."""
    cfg = config or GaeConfig()
    if not steps:
        return AdvantageBatch(estimates=(), advantage_mean=0.0, advantage_std=0.0)

    grouped: defaultdict[tuple[str, int], list[_IndexedStep]] = defaultdict(list)
    for input_index, step in enumerate(steps):
        _validate_step(step)
        grouped[(step.game_id, step.seat)].append(
            _IndexedStep(input_index=input_index, step=step)
        )

    estimates_by_input: dict[int, AdvantageEstimate] = {}
    for trajectory_steps in grouped.values():
        for estimate in _trajectory_gae(
            sorted(trajectory_steps, key=lambda item: item.step.decision_index),
            cfg,
        ):
            estimates_by_input[estimate.input_index] = estimate

    estimates = tuple(estimates_by_input[index] for index in range(len(steps)))
    advantages = tuple(estimate.advantage for estimate in estimates)
    advantage_mean, advantage_std = _mean_and_std(advantages)
    if cfg.normalize_advantages and advantage_std > cfg.normalize_epsilon:
        normalize_mean = advantage_mean
        scale = advantage_std
    elif cfg.normalize_advantages:
        normalize_mean = advantage_mean
        scale = math.inf
    else:
        normalize_mean = 0.0
        scale = 1.0

    normalized = tuple((value - normalize_mean) / scale for value in advantages)
    normalized_estimates = tuple(
        AdvantageEstimate(
            input_index=estimate.input_index,
            game_id=estimate.game_id,
            seat=estimate.seat,
            decision_index=estimate.decision_index,
            advantage=estimate.advantage,
            normalized_advantage=normalized[index],
            return_value=estimate.return_value,
        )
        for index, estimate in enumerate(estimates)
    )
    return AdvantageBatch(
        estimates=normalized_estimates,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
    )


def _trajectory_gae(
    indexed_steps: Sequence[_IndexedStep],
    config: GaeConfig,
) -> tuple[AdvantageEstimate, ...]:
    rewards = _trajectory_rewards(indexed_steps, config)
    values = np.asarray(
        [indexed.step.value_pred for indexed in indexed_steps],
        dtype=np.float64,
    )
    next_values = np.zeros_like(values)
    if len(values) > 1:
        next_values[:-1] = values[1:]
    deltas = rewards + config.gamma * next_values - values
    advantages = _discounted_cumsum(deltas, config.gamma * config.gae_lambda)
    if config.value_target_mode == "monte_carlo":
        returns = _discounted_cumsum(rewards, config.gamma)
    else:
        returns = advantages + values

    return tuple(
        AdvantageEstimate(
            input_index=indexed.input_index,
            game_id=indexed.step.game_id,
            seat=indexed.step.seat,
            decision_index=indexed.step.decision_index,
            advantage=float(advantages[position]),
            normalized_advantage=float(advantages[position]),
            return_value=float(returns[position]),
        )
        for position, indexed in enumerate(indexed_steps)
    )


def _trajectory_rewards(
    indexed_steps: Sequence[_IndexedStep],
    config: GaeConfig,
) -> np.ndarray:
    rewards = np.zeros(len(indexed_steps), dtype=np.float64)
    rewards[-1] = indexed_steps[-1].step.terminal_reward
    if config.prize_diff_shaping_beta == 0.0:
        return rewards

    potentials = _required_float_array(
        [indexed.step.potential for indexed in indexed_steps],
        "potential",
    )
    next_potentials = np.empty_like(potentials)
    if len(potentials) > 1:
        next_potentials[:-1] = potentials[1:]
    terminal_potential = indexed_steps[-1].step.terminal_potential
    if terminal_potential is None:
        raise ValueError("next potential is required when shaping is enabled")
    next_potentials[-1] = terminal_potential
    rewards += config.prize_diff_shaping_beta * (next_potentials - potentials)
    return rewards


def _discounted_cumsum(values: np.ndarray, discount: float) -> np.ndarray:
    if discount == 0.0:
        return values.copy()
    if discount == 1.0:
        return np.cumsum(values[::-1], dtype=np.float64)[::-1]
    powers = np.power(discount, np.arange(len(values), dtype=np.float64))
    if powers[-1] == 0.0:
        return _discounted_cumsum_scan(values, discount)
    reversed_result = np.cumsum(values[::-1] / powers, dtype=np.float64) * powers
    if not np.isfinite(reversed_result).all():
        return _discounted_cumsum_scan(values, discount)
    return reversed_result[::-1]


def _discounted_cumsum_scan(values: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(values, dtype=np.float64)
    running = 0.0
    for position in range(len(values) - 1, -1, -1):
        running = float(values[position]) + discount * running
        result[position] = running
    return result


def _required_float_array(
    values: Sequence[float | None],
    name: str,
) -> np.ndarray:
    required: list[float] = []
    for value in values:
        if value is None:
            raise ValueError(f"{name} is required when shaping is enabled")
        required.append(float(value))
    return np.asarray(required, dtype=np.float64)


def _validate_step(step: AdvantageStep) -> None:
    if step.decision_index < 0:
        raise ValueError("decision_index must be non-negative")
    for name, value in (
        ("terminal_reward", step.terminal_reward),
        ("value_pred", step.value_pred),
        ("potential", step.potential),
        ("terminal_potential", step.terminal_potential),
    ):
        if value is not None and not math.isfinite(value):
            raise ValueError(f"{name} must be finite when set")


def _mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return (mean, math.sqrt(max(0.0, variance)))


def _required(row: Mapping[str, object], key: str) -> object:
    try:
        return row[key]
    except KeyError as exc:
        raise ValueError(f"missing trajectory row field: {key}") from exc


def _as_int(value: object) -> int:
    if value is None:
        raise ValueError("expected integer value, got None")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("expected finite integer value")
        return int(value)
    if isinstance(value, str):
        return int(value)
    raise ValueError(f"expected integer-compatible value, got {type(value).__name__}")


def _as_float(value: object) -> float:
    if value is None:
        raise ValueError("expected float value, got None")
    if not isinstance(value, int | float | str):
        raise ValueError(f"expected float-compatible value, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("expected finite float value")
    return result


def _as_optional_float(value: object) -> float | None:
    if value is None:
        return None
    return _as_float(value)
