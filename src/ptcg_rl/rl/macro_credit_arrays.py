"""Compact two-level CSR storage for executed-macro hindsight rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.rl.macro_credit import (
    MACRO_AGGREGATION_FINGERPRINT,
    MACRO_CONTINUATION_SUMMARY_FINGERPRINT,
    MACRO_ENDPOINT_COUNT,
    ExecutedMacroTarget,
    MacroEndpoint,
    build_executed_macro_targets,
)


@dataclass(frozen=True, eq=False)
class ExecutedMacroArrayBlock:
    """Root index plus CSR continuation references and fixed-width targets."""

    decision_macro_indices: np.ndarray
    root_decision_positions: np.ndarray
    continuation_offsets: np.ndarray
    continuation_decision_positions: np.ndarray
    aggregate_effect_features: np.ndarray
    endpoints: np.ndarray
    macro_decision_counts: np.ndarray
    engine_steps: np.ndarray
    terminal_root_outcomes: np.ndarray
    behavior_controller_masks: np.ndarray
    aggregation_fingerprint: str = MACRO_AGGREGATION_FINGERPRINT
    continuation_summary_fingerprint: str = MACRO_CONTINUATION_SUMMARY_FINGERPRINT

    @property
    def row_count(self) -> int:
        """Return the number of root-only macro rows."""
        return int(self.root_decision_positions.shape[0])

    def row_at(self, row_index: int) -> ExecutedMacroTarget:
        """Reconstruct one immutable target from its CSR row."""
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError("executed macro row index is out of range")
        start = int(self.continuation_offsets[row_index])
        stop = int(self.continuation_offsets[row_index + 1])
        outcome = int(self.terminal_root_outcomes[row_index])
        return ExecutedMacroTarget(
            root_decision_position=int(self.root_decision_positions[row_index]),
            continuation_decision_positions=tuple(
                int(value) for value in self.continuation_decision_positions[start:stop]
            ),
            aggregate_effect_features=tuple(
                float(value) for value in self.aggregate_effect_features[row_index]
            ),
            endpoint=MacroEndpoint(int(self.endpoints[row_index])),
            macro_decision_count=int(self.macro_decision_counts[row_index]),
            engine_steps=int(self.engine_steps[row_index]),
            terminal_root_outcome=None if outcome == -2 else outcome,
            macro_sample_of_behavior_controller=bool(
                self.behavior_controller_masks[row_index]
            ),
        )

    def row_for_decision(self, decision_position: int) -> ExecutedMacroTarget | None:
        """Return the root macro attached to one decision position, if present."""
        row_index = int(self.decision_macro_indices[decision_position])
        return None if row_index < 0 else self.row_at(row_index)


def build_executed_macro_array_block(
    decisions: tuple[Any, ...] | list[Any],
    *,
    seats_reward: tuple[float, float],
) -> ExecutedMacroArrayBlock | None:
    """Build an explicit schema-10 block, including a valid empty row table."""
    targets = build_executed_macro_targets(decisions, seats_reward=seats_reward)
    if targets is None:
        return None
    decision_macro_indices = np.full(len(decisions), -1, dtype=np.int32)
    roots: list[int] = []
    offsets = [0]
    continuations: list[int] = []
    effects: list[tuple[float, ...]] = []
    endpoints: list[int] = []
    decision_counts: list[int] = []
    engine_steps: list[int] = []
    outcomes: list[int] = []
    behavior_masks: list[bool] = []
    for row_index, target in enumerate(targets):
        root = target.root_decision_position
        if decision_macro_indices[root] >= 0:
            raise ValueError("one decision cannot own multiple executed macros")
        decision_macro_indices[root] = row_index
        roots.append(root)
        continuations.extend(target.continuation_decision_positions)
        offsets.append(len(continuations))
        effects.append(target.aggregate_effect_features)
        endpoints.append(int(target.endpoint))
        decision_counts.append(target.macro_decision_count)
        engine_steps.append(target.engine_steps)
        outcomes.append(
            -2 if target.terminal_root_outcome is None else target.terminal_root_outcome
        )
        behavior_masks.append(target.macro_sample_of_behavior_controller)
    block = ExecutedMacroArrayBlock(
        decision_macro_indices=decision_macro_indices,
        root_decision_positions=np.asarray(roots, dtype=np.int32),
        continuation_offsets=np.asarray(offsets, dtype=np.int32),
        continuation_decision_positions=np.asarray(continuations, dtype=np.int32),
        aggregate_effect_features=np.asarray(
            effects,
            dtype=np.float32,
        ).reshape((-1, DYNAMIC_EFFECT_FEATURE_SIZE)),
        endpoints=np.asarray(endpoints, dtype=np.uint8),
        macro_decision_counts=np.asarray(decision_counts, dtype=np.uint16),
        engine_steps=np.asarray(engine_steps, dtype=np.int32),
        terminal_root_outcomes=np.asarray(outcomes, dtype=np.int8),
        behavior_controller_masks=np.asarray(behavior_masks, dtype=np.bool_),
    )
    validate_executed_macro_array_block(block, decision_count=len(decisions))
    return block


def validate_executed_macro_array_block(
    block: ExecutedMacroArrayBlock,
    *,
    decision_count: int,
) -> None:
    """Reject corrupt CSR chains and mismatched aggregation identities."""
    if decision_count <= 0:
        raise ValueError("executed macro block requires trajectory decisions")
    if block.aggregation_fingerprint != MACRO_AGGREGATION_FINGERPRINT:
        raise ValueError("executed macro aggregation identity is unsupported")
    if block.continuation_summary_fingerprint != MACRO_CONTINUATION_SUMMARY_FINGERPRINT:
        raise ValueError("executed macro continuation identity is unsupported")
    row_count = block.row_count
    if (
        block.decision_macro_indices.dtype != np.int32
        or block.decision_macro_indices.shape != (decision_count,)
    ):
        raise ValueError("macro decision indices must be int32 [decision_count]")
    if (
        block.root_decision_positions.dtype != np.int32
        or block.root_decision_positions.shape != (row_count,)
    ):
        raise ValueError("macro root positions must be int32 [row_count]")
    if (
        block.continuation_offsets.dtype != np.int32
        or block.continuation_offsets.shape != (row_count + 1,)
        or int(block.continuation_offsets[0]) != 0
        or bool(
            np.any(block.continuation_offsets[1:] < block.continuation_offsets[:-1])
        )
    ):
        raise ValueError("macro continuation offsets are invalid")
    continuation_count = int(block.continuation_offsets[-1])
    if (
        block.continuation_decision_positions.dtype != np.int32
        or block.continuation_decision_positions.shape != (continuation_count,)
    ):
        raise ValueError("macro continuation positions do not align with offsets")
    expected_fields = (
        (
            block.aggregate_effect_features,
            np.float32,
            (row_count, DYNAMIC_EFFECT_FEATURE_SIZE),
        ),
        (block.endpoints, np.uint8, (row_count,)),
        (block.macro_decision_counts, np.uint16, (row_count,)),
        (block.engine_steps, np.int32, (row_count,)),
        (block.terminal_root_outcomes, np.int8, (row_count,)),
        (block.behavior_controller_masks, np.bool_, (row_count,)),
    )
    for values, dtype, shape in expected_fields:
        if values.dtype != dtype or values.shape != shape:
            raise ValueError("executed macro fixed-width field is misaligned")
    if not bool(np.isfinite(block.aggregate_effect_features).all()):
        raise ValueError("executed macro effects must be finite")
    if row_count and (
        bool(np.any(block.root_decision_positions < 0))
        or bool(np.any(block.root_decision_positions >= decision_count))
        or len({int(value) for value in block.root_decision_positions}) != row_count
    ):
        raise ValueError("executed macro roots are invalid")
    if bool(np.any(block.decision_macro_indices < -1)) or bool(
        np.any(block.decision_macro_indices >= row_count)
    ):
        raise ValueError("executed macro decision mapping is out of range")
    referenced = {
        int(value) for value in block.decision_macro_indices if int(value) >= 0
    }
    if referenced != set(range(row_count)):
        raise ValueError("executed macro table contains unreferenced rows")
    for row_index in range(row_count):
        target = block.row_at(row_index)
        if target.root_decision_position != int(
            block.root_decision_positions[row_index]
        ):
            raise ValueError("executed macro root reconstruction failed")
        if (
            int(block.decision_macro_indices[target.root_decision_position])
            != row_index
        ):
            raise ValueError("executed macro root mapping is inconsistent")
        if (
            target.endpoint is not MacroEndpoint.TERMINAL
            and int(block.terminal_root_outcomes[row_index]) != -2
        ):
            raise ValueError("non-terminal macro has a terminal outcome")
    if row_count and not bool(block.behavior_controller_masks.all()):
        raise ValueError("executed macro rows must be behavior-controller samples")
    if row_count and not bool((block.endpoints < MACRO_ENDPOINT_COUNT).all()):
        raise ValueError("executed macro endpoint is out of range")


def executed_macro_array_field_names() -> tuple[str, ...]:
    """Return ndarray field names in stable transport order."""
    return (
        "decision_macro_indices",
        "root_decision_positions",
        "continuation_offsets",
        "continuation_decision_positions",
        "aggregate_effect_features",
        "endpoints",
        "macro_decision_counts",
        "engine_steps",
        "terminal_root_outcomes",
        "behavior_controller_masks",
    )


__all__ = [
    "ExecutedMacroArrayBlock",
    "build_executed_macro_array_block",
    "executed_macro_array_field_names",
    "validate_executed_macro_array_block",
]
