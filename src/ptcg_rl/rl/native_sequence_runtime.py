"""Transactional sequence bookkeeping at the native engine boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ptcg_rl.rl.native_policy_trace import (
    NativePolicyNumpyActionBatch,
    NativePolicyNumpyTrace,
)
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_actor import GeneralistSequenceActorPolicy
from ptcg_rl.rl.stateless_actor import (
    StatelessActorBatchTrace,
    StatelessActorDecisionTrace,
)
from ptcg_rl.rl.stateless_fragment import StatelessFragmentIdentity


@dataclass(frozen=True, slots=True)
class NativePendingSequenceDecision:
    """One provisional action waiting for source-engine acceptance."""

    actor: GeneralistSequenceActorPolicy
    arena_row: int
    row: SimpleStatelessActorRow
    trace: StatelessActorDecisionTrace


def native_trace_from_actor_batch(
    identity: StatelessFragmentIdentity,
    trace: StatelessActorBatchTrace,
) -> NativePolicyNumpyTrace:
    """Convert object actor evidence to the native compact host contract."""
    decisions = trace.decisions
    if not decisions:
        raise ValueError("native sequence trace requires decisions")
    action_lengths = np.asarray(
        [len(decision.action) for decision in decisions],
        dtype=np.int64,
    )
    token_lengths = np.asarray(
        [len(decision.token_logprobs) for decision in decisions],
        dtype=np.int64,
    )
    return NativePolicyNumpyTrace(
        identity=identity,
        action_offsets=_offsets(action_lengths),
        action_choices=np.asarray(
            [choice for decision in decisions for choice in decision.action],
            dtype=np.int32,
        ),
        action_logprobs=np.asarray(
            [decision.action_logprob for decision in decisions],
            dtype=np.float32,
        ),
        token_offsets=_offsets(token_lengths),
        token_logprobs=np.asarray(
            [
                value
                for decision in decisions
                for value in decision.token_logprobs
            ],
            dtype=np.float32,
        ),
        prefix_values=np.asarray(
            [
                value
                for decision in decisions
                for value in decision.prefix_values
            ],
            dtype=np.float32,
        ),
        root_values=np.asarray(
            [decision.root_value for decision in decisions],
            dtype=np.float32,
        ),
        stop_sampled=np.asarray(
            [decision.stop_sampled for decision in decisions],
            dtype=np.bool_,
        ),
    )


def native_actions_from_actor_batch(
    identity: StatelessFragmentIdentity,
    trace: StatelessActorBatchTrace,
) -> NativePolicyNumpyActionBatch:
    """Convert sequence opponent actions without retaining PPO evidence."""
    decisions = trace.decisions
    if not decisions:
        raise ValueError("native sequence actions require decisions")
    lengths = np.asarray(
        [len(decision.action) for decision in decisions],
        dtype=np.int64,
    )
    return NativePolicyNumpyActionBatch(
        identity=identity,
        action_offsets=_offsets(lengths),
        action_choices=np.asarray(
            [choice for decision in decisions for choice in decision.action],
            dtype=np.int32,
        ),
    )


def _offsets(lengths: np.ndarray) -> np.ndarray:
    offsets = np.zeros(lengths.size + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    return offsets


__all__ = [
    "NativePendingSequenceDecision",
    "native_actions_from_actor_batch",
    "native_trace_from_actor_batch",
]
