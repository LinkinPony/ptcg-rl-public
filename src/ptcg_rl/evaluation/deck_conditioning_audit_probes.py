"""Numerical and CUDA eager probes for deck-conditioned models."""

from __future__ import annotations

import math
import statistics
import time
from collections.abc import Sequence
from dataclasses import fields, replace
from typing import Any

import torch
from torch import Tensor

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.decks import DeckBatch
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model import (
    AgentNetworkOutput,
    AgentPolicyValueNet,
    OptionBatch,
    StateBatch,
)
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE


class _Comparison:
    """Accumulate finite numerical and mask differences."""

    def __init__(self, *, atol: float, rtol: float) -> None:
        self.atol = atol
        self.rtol = rtol
        self.maximum_absolute_error = 0.0
        self.maximum_relative_error = 0.0
        self.total_absolute_error = 0.0
        self.finite_values = 0
        self.mask_disagreements = 0
        self.tolerance_violations = 0

    def update(self, expected: Tensor, actual: Tensor) -> None:
        """Add one aligned tensor pair."""
        if expected.shape != actual.shape:
            raise ValueError(
                f"audit tensor shape mismatch: {expected.shape} != {actual.shape}"
            )
        expected_finite = torch.isfinite(expected)
        actual_finite = torch.isfinite(actual)
        self.mask_disagreements += int(
            torch.count_nonzero(expected_finite != actual_finite).item()
        )
        finite = expected_finite & actual_finite
        if not bool(finite.any()):
            return
        expected_values = expected[finite].float()
        actual_values = actual[finite].float()
        difference = (expected_values - actual_values).abs()
        relative = difference / expected_values.abs().clamp_min(1.0e-12)
        tolerance = self.atol + self.rtol * expected_values.abs()
        self.tolerance_violations += int(
            torch.count_nonzero(difference > tolerance).item()
        )
        self.maximum_absolute_error = max(
            self.maximum_absolute_error,
            float(difference.max().item()),
        )
        self.maximum_relative_error = max(
            self.maximum_relative_error,
            float(relative.max().item()),
        )
        self.total_absolute_error += float(difference.sum().item())
        self.finite_values += int(difference.numel())

    def as_dict(self) -> dict[str, float | int]:
        """Return a JSON-safe summary."""
        return {
            "finite_values": self.finite_values,
            "mask_disagreements": self.mask_disagreements,
            "tolerance_violations": self.tolerance_violations,
            "max_abs": self.maximum_absolute_error,
            "max_rel": self.maximum_relative_error,
            "mean_abs": (
                self.total_absolute_error / self.finite_values
                if self.finite_values
                else 0.0
            ),
        }


def migration_audit(
    legacy: AgentPolicyValueNet,
    candidate: AgentPolicyValueNet,
    *,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    actions: Sequence[Sequence[int]],
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare every policy/value/auxiliary migration output."""
    with torch.inference_mode():
        expected_output = legacy(states, options)
        actual_output = candidate(states, options, decks)
        expected_eval = legacy.evaluate_actions(states, options, actions)
        actual_eval = candidate.evaluate_actions(
            states,
            options,
            actions,
            decks=decks,
        )
        expected_actions = legacy.greedy_decode(states, options)
        actual_actions = candidate.greedy_decode(states, options, decks)
    comparison = _Comparison(atol=atol, rtol=rtol)
    for field in fields(AgentNetworkOutput):
        comparison.update(
            getattr(expected_output, field.name),
            getattr(actual_output, field.name),
        )
    for field_name in (
        "action_logprobs",
        "entropies",
        "values",
        "token_logprobs",
        "token_entropies",
        "prefix_values",
    ):
        expected = getattr(expected_eval, field_name)
        actual = getattr(actual_eval, field_name)
        if expected is not None and actual is not None:
            comparison.update(expected, actual)
    for expected, actual in zip(
        expected_eval.step_logits,
        actual_eval.step_logits,
        strict=True,
    ):
        comparison.update(expected, actual)
    metrics = comparison.as_dict()
    greedy_disagreements = sum(
        expected != actual
        for expected, actual in zip(expected_actions, actual_actions, strict=True)
    )
    passed = (
        metrics["mask_disagreements"] == 0
        and metrics["tolerance_violations"] == 0
        and greedy_disagreements == 0
    )
    result = {
        **metrics,
        "greedy_disagreements": greedy_disagreements,
        "rows": len(expected_actions),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"legacy migration audit failed: {result}")
    return result


def mixed_eager_audit(
    model: AgentPolicyValueNet,
    *,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Compare a mixed-deck CUDA batch with independent row forwards."""
    with torch.inference_mode():
        mixed = model(states, options, decks)
        mixed_actions = model.greedy_decode(states, options, decks)
        grouped_outputs = []
        grouped_actions: list[tuple[int, ...]] = []
        for row in range(len(decks)):
            row_states = _select_state_rows(states, row)
            row_options = _select_option_rows(options, row)
            row_decks = decks.select((row,))
            grouped_outputs.append(model(row_states, row_options, row_decks))
            grouped_actions.extend(
                model.greedy_decode(row_states, row_options, row_decks)
            )
    comparison = _Comparison(atol=atol, rtol=rtol)
    for field in fields(AgentNetworkOutput):
        expected = getattr(mixed, field.name)
        actual = torch.cat(
            [getattr(output, field.name) for output in grouped_outputs],
            dim=0,
        )
        comparison.update(expected, actual)
    metrics = comparison.as_dict()
    greedy_disagreements = sum(
        expected != actual
        for expected, actual in zip(mixed_actions, grouped_actions, strict=True)
    )
    passed = (
        metrics["mask_disagreements"] == 0
        and metrics["tolerance_violations"] == 0
        and greedy_disagreements == 0
    )
    result = {
        **metrics,
        "greedy_disagreements": greedy_disagreements,
        "rows": len(decks),
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"mixed eager audit failed: {result}")
    return result


def benchmark_model(
    model: AgentPolicyValueNet,
    *,
    states: StateBatch,
    options: OptionBatch,
    decks: DeckBatch | None,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    """Measure eager inference without creating a performance gate."""
    with torch.inference_mode():
        for _ in range(warmup):
            model(states, options, decks)
        synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = (
            torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        )
        latencies: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter()
            model(states, options, decks)
            synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
        peak_memory = (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        )
    return {
        **latency_summary(latencies),
        "batch_size": int(states.card_ids.shape[0]),
        "decisions_per_second": (
            int(states.card_ids.shape[0]) * 1000.0 / statistics.mean(latencies)
        ),
        "peak_activation_bytes": max(0, peak_memory - baseline_memory),
    }


def parameter_summary(
    legacy: AgentPolicyValueNet,
    candidate: AgentPolicyValueNet,
) -> dict[str, int]:
    """Count shared and conditioning-specific parameters."""
    legacy_parameters = sum(parameter.numel() for parameter in legacy.parameters())
    candidate_parameters = sum(
        parameter.numel() for parameter in candidate.parameters()
    )
    conditioning_parameters = sum(
        parameter.numel()
        for name, parameter in candidate.named_parameters()
        if name.startswith(
            (
                "deck_encoder.",
                "deck_input_projection.",
                "state_encoder.private_adapters.",
                "private_policy_adapters.",
                "private_root_value_heads.",
                "private_prefix_value_heads.",
            )
        )
    )
    return {
        "legacy": legacy_parameters,
        "conditioned": candidate_parameters,
        "increment": candidate_parameters - legacy_parameters,
        "conditioning": conditioning_parameters,
        "fp16_increment_bytes": 2 * (candidate_parameters - legacy_parameters),
    }


def model_inputs(
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[StateBatch, OptionBatch]:
    """Construct reusable, non-run-specific legal synthetic model inputs."""
    states = StateBatch(
        card_ids=torch.tensor([[0, 1]] * batch_size, device=device),
        areas=torch.zeros((batch_size, 2), dtype=torch.long, device=device),
        owner_roles=torch.tensor([[1, 1]] * batch_size, device=device),
        token_kinds=torch.tensor([[0, 2]] * batch_size, device=device),
        scalars=torch.zeros(
            (batch_size, 2, TOKEN_SCALAR_SIZE),
            device=device,
        ),
        last_attack_ids=torch.zeros((batch_size, 2), dtype=torch.long, device=device),
        padding_mask=torch.zeros((batch_size, 2), dtype=torch.bool, device=device),
    )
    option_count = 3
    options = OptionBatch(
        option_types=torch.zeros(
            (batch_size, option_count), dtype=torch.long, device=device
        ),
        contexts=torch.zeros(
            (batch_size, option_count), dtype=torch.long, device=device
        ),
        entity_slots=torch.zeros(
            (batch_size, option_count, 2), dtype=torch.long, device=device
        ),
        entity_slot_mask=torch.zeros(
            (batch_size, option_count, 2), dtype=torch.bool, device=device
        ),
        attack_ids=torch.zeros(
            (batch_size, option_count), dtype=torch.long, device=device
        ),
        card_ids=torch.tensor(
            [[1, 2, 3]] * batch_size, dtype=torch.long, device=device
        ),
        scalars=torch.zeros(
            (batch_size, option_count, SCALAR_FEATURE_SIZE), device=device
        ),
        dynamic_effect_features=torch.zeros(
            (batch_size, option_count, DYNAMIC_EFFECT_FEATURE_SIZE),
            device=device,
        ),
        dynamic_effect_masks=torch.zeros(
            (batch_size, option_count), dtype=torch.bool, device=device
        ),
        valid_options=torch.ones(
            (batch_size, option_count), dtype=torch.bool, device=device
        ),
        min_counts=torch.ones(batch_size, dtype=torch.long, device=device),
        max_counts=torch.full((batch_size,), 3, dtype=torch.long, device=device),
    )
    return states, options


def latency_summary(values: Sequence[float]) -> dict[str, float]:
    """Summarize one non-empty latency sample."""
    ordered = sorted(float(value) for value in values)
    return {
        "median_ms": statistics.median(ordered),
        "p95_ms": ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)],
        "max_ms": max(ordered),
        "mean_ms": statistics.mean(ordered),
    }


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA timing while leaving CPU unchanged."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _select_state_rows(states: StateBatch, row: int) -> StateBatch:
    return replace(
        states,
        **{
            field.name: value[row : row + 1]
            for field in fields(StateBatch)
            if isinstance((value := getattr(states, field.name)), Tensor)
        },
    )


def _select_option_rows(options: OptionBatch, row: int) -> OptionBatch:
    return replace(
        options,
        **{
            field.name: getattr(options, field.name)[row : row + 1]
            for field in fields(OptionBatch)
        },
    )


__all__ = [
    "benchmark_model",
    "latency_summary",
    "migration_audit",
    "mixed_eager_audit",
    "model_inputs",
    "parameter_summary",
    "synchronize",
]
