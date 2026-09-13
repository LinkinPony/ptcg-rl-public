"""Data contracts for complete engine continuation planning."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from ptcg_rl.agent.search.macro import MacroEndpoint, MacroTransition
from ptcg_rl.engine.effect_types import EffectSummary


class ContinuationProposalPolicy(Protocol):
    """Policy surface used only to order bounded continuation expansion."""

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Return one complete greedy selection for the prompt."""

    def rank_actions(
        self,
        observation: Any,
        *,
        top_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return policy-ranked complete selections for the prompt."""


@dataclass(frozen=True)
class ContinuationPlannerLimits:
    """Strict per-root bounds for strategic continuation expansion."""

    exhaustive_action_cap: int = 256
    beam_width: int = 8
    node_cap: int = 512
    forced_step_cap: int = 12
    strategic_step_cap: int = 8

    def __post_init__(self) -> None:
        for name in (
            "exhaustive_action_cap",
            "beam_width",
            "node_cap",
            "forced_step_cap",
            "strategic_step_cap",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class PromptExpansion:
    """Auditable candidate construction at one strategic prompt."""

    action_path: tuple[tuple[int, ...], ...]
    prompt_key: tuple[Any, ...]
    context: int
    legal_action_count: int
    candidates: tuple[tuple[int, ...], ...]
    sources: tuple[tuple[str, ...], ...]
    exhaustive: bool
    ordered: bool


@dataclass(frozen=True)
class ContinuationValueDecks:
    """Full deck bindings used by the two actor-perspective critics."""

    root_full_deck: tuple[int, ...]
    sampled_opponent_full_deck: tuple[int, ...]

    @classmethod
    def from_sequences(
        cls,
        *,
        root_full_deck: Sequence[int],
        sampled_opponent_full_deck: Sequence[int],
    ) -> ContinuationValueDecks:
        """Freeze explicit per-world deck routes before Search begins."""
        return cls(
            root_full_deck=tuple(int(card_id) for card_id in root_full_deck),
            sampled_opponent_full_deck=tuple(
                int(card_id) for card_id in sampled_opponent_full_deck
            ),
        )


@dataclass(frozen=True)
class ContinuationValueRequest:
    """One legal actor-perspective nonterminal critic request."""

    observation: Any
    deck: tuple[int, ...]
    perspective_player_index: int
    root_value_sign: int

    def __post_init__(self) -> None:
        if self.perspective_player_index not in (0, 1):
            raise ValueError("value perspective must be player 0 or 1")
        if self.root_value_sign not in (-1, 1):
            raise ValueError("root_value_sign must be -1 or 1")


@dataclass(frozen=True)
class CompleteContinuation:
    """One root selection advanced to a semantic leaf or explicit truncation."""

    root_action: tuple[int, ...]
    action_path: tuple[tuple[int, ...], ...]
    endpoint: MacroEndpoint
    leaf_observation: Any | None
    value_request: ContinuationValueRequest | None
    summaries: tuple[EffectSummary, ...]
    forced_steps: int
    strategic_steps: int
    stop_detail: str
    error: str | None = None

    @property
    def complete(self) -> bool:
        """Whether this path reached a value-comparable semantic endpoint."""
        return self.endpoint in {
            MacroEndpoint.TERMINAL,
            MacroEndpoint.SAME_SEAT_MAIN,
            MacroEndpoint.TURN_HANDOFF,
        }

    @property
    def value_comparable(self) -> bool:
        """Whether this endpoint has exact or actor-perspective value evidence."""
        if self.endpoint == MacroEndpoint.TERMINAL:
            return self.leaf_observation is not None
        if self.endpoint in {
            MacroEndpoint.SAME_SEAT_MAIN,
            MacroEndpoint.TURN_HANDOFF,
        }:
            return self.value_request is not None
        return False

    def as_macro_transition(
        self,
        *,
        state_pool_peak: int = 0,
        state_leaks: int = 0,
    ) -> MacroTransition:
        """Adapt the complete path to existing engine-grounded scorers."""
        return MacroTransition(
            root_action=self.root_action,
            endpoint=self.endpoint,
            leaf_observation=self.leaf_observation,
            summaries=self.summaries,
            steps=len(self.action_path),
            forced_steps=self.forced_steps,
            continuation_steps=self.strategic_steps,
            stop_detail=self.stop_detail,
            state_pool_peak=state_pool_peak,
            state_leaks=state_leaks,
            error=self.error,
        )


@dataclass(frozen=True)
class CompleteContinuationPlan:
    """All retained leaves and structural validity for one root action."""

    root_action: tuple[int, ...]
    leaves: tuple[CompleteContinuation, ...]
    prompt_expansions: tuple[PromptExpansion, ...]
    nodes_expanded: int
    proposal_errors: int
    complete_coverage: bool
    exhaustive: bool
    stop_reason: str
    state_pool_peak: int
    state_leaks: int

    @property
    def complete_leaves(self) -> tuple[CompleteContinuation, ...]:
        """Return only terminal, handoff, and same-seat MAIN leaves."""
        return tuple(leaf for leaf in self.leaves if leaf.complete)

    @property
    def valid(self) -> bool:
        """Whether all retained branches reached comparable engine leaves."""
        return (
            self.complete_coverage
            and self.state_leaks == 0
            and bool(self.complete_leaves)
            and all(leaf.value_comparable for leaf in self.complete_leaves)
        )

    @property
    def exact(self) -> bool:
        """Whether every legal continuation branch was retained."""
        return self.valid and self.exhaustive

    @property
    def search_coverage(self) -> float:
        """Return a smooth geometric retained/legal prompt-space fraction."""
        if not self.prompt_expansions:
            return 1.0
        ratios = tuple(
            min(
                1.0,
                float(len(prompt.candidates))
                / float(max(1, prompt.legal_action_count)),
            )
            for prompt in self.prompt_expansions
        )
        mean_log = sum(math.log(max(ratio, 1.0e-12)) for ratio in ratios) / len(
            ratios
        )
        return math.exp(mean_log)


__all__ = [
    "CompleteContinuation",
    "CompleteContinuationPlan",
    "ContinuationValueDecks",
    "ContinuationValueRequest",
    "ContinuationPlannerLimits",
    "ContinuationProposalPolicy",
    "PromptExpansion",
]
