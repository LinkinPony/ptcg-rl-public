"""Typed boundary for online engine-counterfactual policy targets."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ptcg_rl.actions.selection import is_legal_action, normalize_action_order
from ptcg_rl.context import GameContextFeatures, GameContextSnapshot
from ptcg_rl.engine.search_evidence import SearchEvidence
from ptcg_rl.engine.vector_battle import Deck

if TYPE_CHECKING:
    from ptcg_rl.rl.macro_teacher import MacroTeacherEvidence


@dataclass(frozen=True)
class EngineTeacherTarget:
    """One complete legal Select response recommended by engine search.

    The target is auxiliary supervision only. It must never replace the sampled
    behavior action or its log-probability in a PPO trajectory.
    """

    action: tuple[int, ...]
    confidence: float = 1.0
    weight: float = 1.0
    search_evidence: SearchEvidence | None = None
    macro_evidence: MacroTeacherEvidence | None = None

    def __post_init__(self) -> None:
        """Reject targets that cannot define a finite weighted objective."""
        normalized = tuple(int(index) for index in self.action)
        if normalized != self.action:
            object.__setattr__(self, "action", normalized)
        if not math.isfinite(self.confidence) or not 0.0 < self.confidence <= 1.0:
            raise ValueError("engine teacher confidence must be in (0, 1]")
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("engine teacher weight must be finite and positive")
        if (
            self.search_evidence is not None
            and self.action not in self.search_evidence.actions
        ):
            raise ValueError(
                "engine teacher target action must be present in search evidence"
            )
        if self.macro_evidence is not None:
            if self.action != self.macro_evidence.target_action:
                raise ValueError("macro teacher target differs from its evidence")
            if self.search_evidence is None:
                raise ValueError("macro teacher target requires compatible search rows")


@dataclass(frozen=True)
class EngineTeacherRequest:
    """Public-information decision context passed to an online teacher."""

    game_id: str
    seat: int
    observation: Mapping[str, Any]
    context_features: GameContextFeatures
    context_snapshot: GameContextSnapshot
    counterparty_context_snapshot: GameContextSnapshot
    own_deck: Deck
    behavior_action: tuple[int, ...]

    def __post_init__(self) -> None:
        """Enforce the private-information boundary at request construction."""
        if self.seat not in (0, 1):
            raise ValueError("engine teacher request seat must be 0 or 1")
        if self.context_snapshot.player_index != self.seat:
            raise ValueError("engine teacher root context has the wrong perspective")
        if self.counterparty_context_snapshot.player_index != 1 - self.seat:
            raise ValueError(
                "engine teacher counterparty context has the wrong perspective"
            )
        if self.counterparty_context_snapshot.own_deck_counts:
            raise ValueError(
                "engine teacher counterparty context must not contain a private deck"
            )


class EngineTeacherProducer(Protocol):
    """Injectable actor-side engine planner surface."""

    def produce(self, request: EngineTeacherRequest) -> EngineTeacherTarget | None:
        """Return a complete counterfactual target or no target on weak evidence."""


class EngineTeacherBatchProducer(EngineTeacherProducer, Protocol):
    """Optional actor-step batch extension for an engine teacher producer."""

    def produce_batch(
        self,
        requests: Sequence[EngineTeacherRequest],
    ) -> Sequence[EngineTeacherTarget | None]:
        """Return targets in exactly the same order as the requests."""


@dataclass(frozen=True)
class EngineTeacherBatchCompletion:
    """One completed non-blocking teacher batch."""

    batch_id: int
    targets: tuple[EngineTeacherTarget | None, ...]


@runtime_checkable
class AsyncEngineTeacherProducer(Protocol):
    """Actor-side teacher lane that never blocks rollout submission."""

    def submit_async(
        self,
        requests: Sequence[EngineTeacherRequest],
    ) -> int | None:
        """Queue a batch, returning its identity or ``None`` on saturation."""

    def poll_completed(self) -> tuple[EngineTeacherBatchCompletion, ...]:
        """Return all currently completed batches without waiting."""

    def drain(self) -> tuple[EngineTeacherBatchCompletion, ...]:
        """Wait for all accepted batches and return their completions."""

    def summary(self) -> Mapping[str, Any]:
        """Return non-blocking diagnostic counters."""

    def close(self) -> None:
        """Release background and native resources."""


def validate_engine_teacher_target(
    select: Any,
    target: EngineTeacherTarget,
) -> EngineTeacherTarget:
    """Validate a canonical planner target against the serving prompt."""
    action = normalize_action_order(select, target.action)
    if action != target.action:
        raise ValueError("engine teacher action is not canonical for this prompt")
    if not is_legal_action(select, action):
        raise ValueError(f"engine teacher returned an illegal action: {action}")
    if target.search_evidence is not None:
        for candidate in target.search_evidence.actions:
            normalized = normalize_action_order(select, candidate)
            if normalized != candidate:
                raise ValueError(
                    "engine teacher search candidate is not canonical for this prompt"
                )
            if not is_legal_action(select, candidate):
                raise ValueError(
                    f"engine teacher search candidate is illegal: {candidate}"
                )
    return target


def optional_target_arrays(
    targets: Sequence[EngineTeacherTarget | None],
) -> (
    tuple[
        tuple[int, ...],
        tuple[int, ...],
        tuple[float, ...],
        tuple[float, ...],
        tuple[bool, ...],
    ]
    | None
):
    """Flatten sparse targets while retaining an explicit per-row validity mask."""
    if not any(target is not None for target in targets):
        return None
    offsets = [0]
    indices: list[int] = []
    confidences: list[float] = []
    weights: list[float] = []
    masks: list[bool] = []
    for target in targets:
        valid = target is not None
        if target is not None:
            indices.extend(target.action)
        offsets.append(len(indices))
        confidences.append(0.0 if target is None else target.confidence)
        weights.append(0.0 if target is None else target.weight)
        masks.append(valid)
    return (
        tuple(offsets),
        tuple(indices),
        tuple(confidences),
        tuple(weights),
        tuple(masks),
    )


__all__ = [
    "AsyncEngineTeacherProducer",
    "EngineTeacherBatchProducer",
    "EngineTeacherBatchCompletion",
    "EngineTeacherProducer",
    "EngineTeacherRequest",
    "EngineTeacherTarget",
    "optional_target_arrays",
    "validate_engine_teacher_target",
]
