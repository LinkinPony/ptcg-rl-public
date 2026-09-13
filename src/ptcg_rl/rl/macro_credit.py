"""Executed-macro hindsight targets and their deterministic aggregation contract."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ptcg_rl.agent.search.root_information_tensorizer import (
    RootInformationTensorizerConfig,
)
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.factual_schema import (
    FACTUAL_NEXT_CONTEXT_COUNT,
    FACTUAL_NEXT_CONTEXT_OOV,
    FactualActorRelation,
)
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_BINARY_INDICES,
    DYNAMIC_EFFECT_FEATURE_NAMES,
    DYNAMIC_EFFECT_FEATURE_SIZE,
)
from ptcg_rl.engine.macro_credit_schema import (
    MACRO_ACTION_COUNT_BUCKETS,
    MACRO_CONTINUATION_SUMMARY_SIZE,
    MACRO_ENDPOINT_COUNT,
    MACRO_IDENTITY_BUCKET_COUNT,
    MACRO_OPTION_TYPE_COUNT,
    MacroEndpoint,
)

_SIGNED_SUM_FEATURES = frozenset(
    {
        "energy_delta_norm",
        "opponent_energy_delta_norm",
    }
)
_MAX_FEATURES = frozenset({"opponent_bench_max_damage_norm"})
_BINARY_INDICES = frozenset(DYNAMIC_EFFECT_BINARY_INDICES)
_AGGREGATION_OPERATIONS = tuple(
    "max"
    if index in _BINARY_INDICES or name in _MAX_FEATURES
    else "signed_clipped_sum"
    if name in _SIGNED_SUM_FEATURES
    else "clipped_sum"
    for index, name in enumerate(DYNAMIC_EFFECT_FEATURE_NAMES)
)


def macro_aggregation_fingerprint() -> str:
    """Return the content identity of the per-feature aggregation semantics."""
    payload = {
        "schema": "executed-macro-effect-aggregation-v1",
        "features": tuple(
            {"name": name, "operation": operation}
            for name, operation in zip(
                DYNAMIC_EFFECT_FEATURE_NAMES,
                _AGGREGATION_OPERATIONS,
                strict=True,
            )
        ),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def continuation_summary_fingerprint() -> str:
    """Return the fixed identity of the compact continuation featurizer."""
    payload = {
        "schema": "executed-macro-continuation-summary-v1",
        "context_count": FACTUAL_NEXT_CONTEXT_COUNT,
        "option_type_count": MACRO_OPTION_TYPE_COUNT + 1,
        "identity_buckets": MACRO_IDENTITY_BUCKET_COUNT,
        "action_count_buckets": MACRO_ACTION_COUNT_BUCKETS,
        "scalars": (
            "decision_count_norm",
            "engine_steps_norm",
            "selected_options_norm",
            "last_action_count_norm",
        ),
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


MACRO_AGGREGATION_FINGERPRINT = macro_aggregation_fingerprint()
MACRO_CONTINUATION_SUMMARY_FINGERPRINT = continuation_summary_fingerprint()
_SHA256 = re.compile(r"[0-9a-f]{64}")


class MacroTeacherIdentityConfig(BaseModel):
    """Static native teacher identities accepted by the schema-10 learner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    constructor_fingerprint: str
    scorer_fingerprint: str
    controller_fingerprint: str
    planner_fingerprint: str

    @field_validator(
        "constructor_fingerprint",
        "scorer_fingerprint",
        "controller_fingerprint",
        "planner_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require a canonical content identity."""
        if _SHA256.fullmatch(value) is None:
            raise ValueError("macro teacher identities must be lowercase SHA-256")
        return value


class MacroCreditConfig(BaseModel):
    """Hydra-backed schema-10 actor and learner contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    native_teacher_enabled: bool = False
    native_teacher_sample_probability: float = 1.0
    native_teacher_sampling_seed: int = 0
    async_queue_batches: int = 2
    root_information_tensorizer: RootInformationTensorizerConfig | None = None
    native_teacher_identity: MacroTeacherIdentityConfig | None = None
    handoff_score_mode: Literal["engine_only", "root_value_adapter"] = "engine_only"
    aggregation_fingerprint: str = MACRO_AGGREGATION_FINGERPRINT
    continuation_summary_fingerprint: str = MACRO_CONTINUATION_SUMMARY_FINGERPRINT

    @field_validator("async_queue_batches")
    @classmethod
    def positive_queue_size(cls, value: int) -> int:
        """Reject an unusable bounded teacher queue."""
        if value <= 0:
            raise ValueError("macro teacher queue size must be positive")
        return value

    @field_validator("native_teacher_sample_probability")
    @classmethod
    def valid_teacher_sample_probability(cls, value: float) -> float:
        """Require one fixed, non-adaptive post-behavior sampling rate."""
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("macro teacher sample probability must be in [0, 1]")
        return value

    @model_validator(mode="after")
    def valid_contract(self) -> MacroCreditConfig:
        """Bind enabled lanes to the implemented immutable featurizers."""
        if self.native_teacher_enabled and not self.enabled:
            raise ValueError("native macro teacher requires macro credit")
        if self.native_teacher_enabled and self.native_teacher_sample_probability <= 0.0:
            raise ValueError("native macro teacher requires positive sampling")
        if self.native_teacher_enabled != (self.native_teacher_identity is not None):
            raise ValueError(
                "native macro teacher identity must be present exactly when enabled"
            )
        if self.enabled and self.root_information_tensorizer is None:
            raise ValueError("macro credit requires a root-information tensorizer")
        if self.enabled and self.handoff_score_mode != "root_value_adapter":
            raise ValueError(
                "schema-10 mainline requires root-value-adapter handoff scoring"
            )
        if self.aggregation_fingerprint != MACRO_AGGREGATION_FINGERPRINT:
            raise ValueError("macro aggregation fingerprint is unsupported")
        if (
            self.continuation_summary_fingerprint
            != MACRO_CONTINUATION_SUMMARY_FINGERPRINT
        ):
            raise ValueError("macro continuation fingerprint is unsupported")
        return self


@dataclass(frozen=True, slots=True)
class ExecutedMacroTarget:
    """One root-only factual macro row referencing its continuation decisions."""

    root_decision_position: int
    continuation_decision_positions: tuple[int, ...]
    aggregate_effect_features: tuple[float, ...]
    endpoint: MacroEndpoint
    macro_decision_count: int
    engine_steps: int
    terminal_root_outcome: int | None
    macro_sample_of_behavior_controller: bool = True

    def __post_init__(self) -> None:
        """Reject incomplete chains and non-finite aggregate targets."""
        if self.root_decision_position < 0:
            raise ValueError("macro root position must be non-negative")
        if not self.continuation_decision_positions:
            raise ValueError("executed macro requires a continuation decision")
        if any(
            position <= self.root_decision_position
            for position in self.continuation_decision_positions
        ):
            raise ValueError("macro continuation positions must follow the root")
        if tuple(sorted(self.continuation_decision_positions)) != (
            self.continuation_decision_positions
        ):
            raise ValueError("macro continuation positions must be ordered")
        effects = tuple(float(value) for value in self.aggregate_effect_features)
        if len(effects) != DYNAMIC_EFFECT_FEATURE_SIZE or not all(
            math.isfinite(value) for value in effects
        ):
            raise ValueError("executed macro effects are invalid")
        if self.macro_decision_count != len(self.continuation_decision_positions) + 1:
            raise ValueError("macro decision count differs from its chain")
        if self.engine_steps < self.macro_decision_count:
            raise ValueError("macro engine steps are below its decision count")
        if not self.macro_sample_of_behavior_controller:
            raise ValueError("executed macro rows must identify behavior sampling")
        outcome = self.terminal_root_outcome
        if self.endpoint is MacroEndpoint.TERMINAL:
            if outcome not in (-1, 0, 1):
                raise ValueError("terminal macro requires a root-perspective outcome")
        elif outcome is not None:
            raise ValueError("non-terminal macro cannot carry a terminal outcome")
        object.__setattr__(self, "aggregate_effect_features", effects)


def aggregate_macro_effects(
    rows: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Aggregate exact per-decision facts with explicit per-feature semantics."""
    if not rows:
        raise ValueError("macro aggregation requires at least one factual row")
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != DYNAMIC_EFFECT_FEATURE_SIZE:
        raise ValueError("macro factual rows have an invalid feature width")
    if not bool(np.isfinite(values).all()):
        raise ValueError("macro factual rows must be finite")
    aggregate = np.zeros(DYNAMIC_EFFECT_FEATURE_SIZE, dtype=np.float64)
    for index, operation in enumerate(_AGGREGATION_OPERATIONS):
        column = values[:, index]
        if operation == "max":
            aggregate[index] = float(column.max())
        elif operation == "signed_clipped_sum":
            aggregate[index] = float(np.clip(column.sum(), -1.0, 1.0))
        else:
            aggregate[index] = float(np.clip(column.sum(), 0.0, 1.0))
    return tuple(float(value) for value in aggregate)


def build_executed_macro_targets(
    decisions: Sequence[Any],
    *,
    seats_reward: tuple[float, float],
) -> tuple[ExecutedMacroTarget, ...] | None:
    """Assemble every complete same-seat strategic chain in one finished game."""
    if not decisions:
        return None
    enabled = tuple(
        bool(getattr(decision, "macro_credit_enabled", False)) for decision in decisions
    )
    if any(enabled) and not all(enabled):
        raise ValueError("cannot mix schema-10 and legacy decisions in one game")
    if not any(enabled):
        return None
    targets = tuple(getattr(decision, "factual_target", None) for decision in decisions)
    if any(target is None for target in targets):
        raise ValueError("schema-10 macro assembly requires dense factual targets")
    steps = tuple(
        getattr(decision, "factual_transition_steps", None) for decision in decisions
    )
    if any(step is None or int(cast(int, step)) <= 0 for step in steps):
        raise ValueError("schema-10 macro assembly requires factual step counts")

    rows: list[ExecutedMacroTarget] = []
    for root_position, (root, raw_target) in enumerate(
        zip(decisions, targets, strict=True)
    ):
        target = cast(Any, raw_target)
        if not _starts_macro(target) or _is_continuation_decision(
            decisions,
            targets,
            root_position,
        ):
            continue
        root_seat = int(root.seat)
        expected_decision_index = int(root.decision_index) + 1
        continuation_positions: list[int] = []
        chain_targets = [target]
        endpoint: MacroEndpoint | None = None
        for position in range(root_position + 1, len(decisions)):
            continuation = decisions[position]
            if int(continuation.seat) != root_seat:
                raise ValueError("executed macro crossed seats before its endpoint")
            if int(continuation.decision_index) != expected_decision_index:
                raise ValueError("executed macro decision sequence is discontinuous")
            expected_decision_index += 1
            continuation_target = cast(Any, targets[position])
            continuation_positions.append(position)
            chain_targets.append(continuation_target)
            endpoint = macro_endpoint_for_target(continuation_target)
            if endpoint is not None:
                break
        if endpoint is None:
            raise ValueError("executed macro did not reach a semantic endpoint")
        terminal_outcome = (
            int(seats_reward[root_seat]) if endpoint is MacroEndpoint.TERMINAL else None
        )
        chain_positions = (root_position, *continuation_positions)
        rows.append(
            ExecutedMacroTarget(
                root_decision_position=root_position,
                continuation_decision_positions=tuple(continuation_positions),
                aggregate_effect_features=aggregate_macro_effects(
                    tuple(
                        cast(Any, targets[position]).effect_features
                        for position in chain_positions
                    )
                ),
                endpoint=endpoint,
                macro_decision_count=len(chain_positions),
                engine_steps=sum(
                    int(cast(int, steps[position])) for position in chain_positions
                ),
                terminal_root_outcome=terminal_outcome,
            )
        )
    return tuple(rows)


def macro_endpoint_for_target(target: Any) -> MacroEndpoint | None:
    """Map a decision-local successor to a semantic macro endpoint."""
    relation = FactualActorRelation(int(target.actor_relation))
    context = int(target.next_context)
    if relation is FactualActorRelation.TERMINAL:
        return MacroEndpoint.TERMINAL
    if relation is FactualActorRelation.OTHER_SEAT:
        return MacroEndpoint.TURN_HANDOFF
    if relation is FactualActorRelation.SAME_SEAT and context == int(
        SelectContext.MAIN
    ):
        return MacroEndpoint.SAME_SEAT_MAIN
    return None


def continuation_summary_from_array_block(
    block: Any,
    continuation_positions: Sequence[int],
    *,
    engine_steps: int,
) -> np.ndarray:
    """Featurize a referenced continuation chain without persisting copied inputs."""
    positions = tuple(int(value) for value in continuation_positions)
    if not positions:
        raise ValueError("macro continuation summary requires decision positions")
    context_width = FACTUAL_NEXT_CONTEXT_COUNT
    option_width = MACRO_OPTION_TYPE_COUNT + 1
    identity_start = context_width + option_width
    action_start = identity_start + MACRO_IDENTITY_BUCKET_COUNT
    scalar_start = action_start + MACRO_ACTION_COUNT_BUCKETS
    summary = np.zeros(MACRO_CONTINUATION_SUMMARY_SIZE, dtype=np.float32)
    selected_total = 0
    last_action_count = 0
    for position in positions:
        context = int(block.options.contexts[position, 0])
        context = (
            context
            if 0 <= context < FACTUAL_NEXT_CONTEXT_COUNT - 1
            else FACTUAL_NEXT_CONTEXT_OOV
        )
        summary[context] += 1.0
        action = block.action_at(position)
        last_action_count = len(action)
        selected_total += len(action)
        summary[action_start + min(len(action), MACRO_ACTION_COUNT_BUCKETS - 1)] += 1.0
        for option_index in action:
            option_type = int(block.options.option_types[position, option_index])
            option_type = min(max(option_type, 0), MACRO_OPTION_TYPE_COUNT)
            summary[context_width + option_type] += 1.0
            identity = (
                context,
                option_type,
                int(block.options.card_ids[position, option_index]),
                int(block.options.attack_ids[position, option_index]),
            )
            bucket = _identity_bucket(identity)
            summary[identity_start + bucket] += 1.0
    decision_count = len(positions)
    summary[:context_width] /= float(decision_count)
    if selected_total:
        summary[context_width:action_start] /= float(selected_total)
    summary[action_start:scalar_start] /= float(decision_count)
    summary[scalar_start:] = np.asarray(
        (
            min(decision_count / 8.0, 1.0),
            min(int(engine_steps) / 32.0, 1.0),
            min(selected_total / 16.0, 1.0),
            min(last_action_count / 8.0, 1.0),
        ),
        dtype=np.float32,
    )
    return summary


def _starts_macro(target: Any) -> bool:
    return FactualActorRelation(
        int(target.actor_relation)
    ) is FactualActorRelation.SAME_SEAT and int(target.next_context) != int(
        SelectContext.MAIN
    )


def _is_continuation_decision(
    decisions: Sequence[Any],
    targets: Sequence[Any],
    position: int,
) -> bool:
    """Return whether a row is already inside the preceding macro chain."""
    if position <= 0:
        return False
    previous = decisions[position - 1]
    current = decisions[position]
    if int(previous.seat) != int(current.seat) or int(
        previous.decision_index
    ) + 1 != int(current.decision_index):
        return False
    previous_target = cast(Any, targets[position - 1])
    return _starts_macro(previous_target)


def _identity_bucket(identity: tuple[int, int, int, int]) -> int:
    encoded = ":".join(str(value) for value in identity).encode()
    digest = hashlib.sha256(b"macro-continuation-identity-v1\x00" + encoded).digest()
    return int.from_bytes(digest[:4], "little") % MACRO_IDENTITY_BUCKET_COUNT


__all__ = [
    "MACRO_AGGREGATION_FINGERPRINT",
    "MACRO_CONTINUATION_SUMMARY_FINGERPRINT",
    "MACRO_CONTINUATION_SUMMARY_SIZE",
    "MACRO_ENDPOINT_COUNT",
    "ExecutedMacroTarget",
    "MacroCreditConfig",
    "MacroEndpoint",
    "aggregate_macro_effects",
    "build_executed_macro_targets",
    "continuation_summary_from_array_block",
    "macro_endpoint_for_target",
]
