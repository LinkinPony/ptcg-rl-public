"""Engine-grounded immediate-win overlay for bounded policy evaluation."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.actions.selection import forced_action, is_legal_action
from ptcg_rl.agent.probe import (
    RuntimeProbeResult,
    all_worlds_verified_lethal,
    core_option_candidates,
    run_runtime_probe_features,
)
from ptcg_rl.agent.search.config import SearchRuntimeConfig
from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context import GameContext


class ImmediateWinOverrideConfig(BaseModel):
    """Conservative diagnostic overlay applied after policy decoding."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["disabled", "shadow", "apply"] = "disabled"
    worlds: int = 3
    manual_coin: bool = True
    seed: int = 0
    sampler: BeliefSamplerConfig = Field(
        default_factory=lambda: BeliefSamplerConfig(mode="rule")
    )

    @field_validator("worlds")
    @classmethod
    def positive_worlds(cls, value: int) -> int:
        """Require at least one independent determinization."""
        if value <= 0:
            raise ValueError("immediate-win worlds must be positive")
        return value


class _ArenaAgent(Protocol):
    """Minimal policy surface wrapped by the diagnostic overlay."""

    name: str

    def act(self, observation: Any) -> Sequence[int]:
        """Return one complete engine action."""


def select_verified_immediate_win(
    *,
    select: Any,
    base_action: Sequence[int],
    probe_result: RuntimeProbeResult,
) -> tuple[tuple[int, ...], str]:
    """Prefer a verified terminal win, preserving the base action on ties."""
    normalized_base = tuple(int(index) for index in base_action)
    candidates = list(core_option_candidates(select))
    if normalized_base in candidates:
        candidates.remove(normalized_base)
        candidates.insert(0, normalized_base)
    for candidate in candidates:
        if not is_legal_action(select, candidate):
            continue
        if all_worlds_verified_lethal(candidate, probe_result):
            reason = (
                "base_verified_win"
                if candidate == normalized_base
                else "alternative_verified_win"
            )
            return candidate, reason
    return normalized_base, "no_verified_win"


class ImmediateWinOverrideAgent:
    """Run public-information engine probes after a base policy decision.

    This Python Search API path is deliberately an evaluation diagnostic. A
    production hot path should move equivalent batching into the native ABI.
    """

    def __init__(
        self,
        base_agent: _ArenaAgent,
        *,
        config: ImmediateWinOverrideConfig,
        game_seed: int,
    ) -> None:
        """Create one game-local overlay around an already-started agent."""
        if config.mode == "disabled":
            raise ValueError("disabled immediate-win overlays must not be wrapped")
        self.name = base_agent.name
        self._base_agent = base_agent
        self._config = config
        self._rng = random.Random(config.seed + game_seed)
        self._sampler = BeliefSampler(config=config.sampler)
        self._context = GameContext()
        self._own_deck: tuple[int, ...] = ()
        self._last_telemetry: dict[str, Any] = {}
        self.immediate_win_probe_calls = 0
        self.immediate_win_missed_decisions = 0
        self.immediate_win_base_verified_decisions = 0
        self.immediate_win_applied_decisions = 0

    @property
    def immediate_win_mode(self) -> str:
        """Expose the overlay mode for flat game artifacts."""
        return self._config.mode

    def __getattr__(self, name: str) -> Any:
        """Preserve release-process identity and execution diagnostics."""
        return getattr(self._base_agent, name)

    def reset(self) -> None:
        """Reset public evidence and delegate any base reset hook."""
        self._context.reset()
        reset = getattr(self._base_agent, "reset", None)
        if callable(reset):
            reset()

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Bind the acting seat and exact own deck used for determinization."""
        if own_deck is None:
            raise ValueError("immediate-win probing requires the exact own deck")
        self._own_deck = tuple(int(card_id) for card_id in own_deck)
        self._context.reset(player_index=player_index, own_deck=self._own_deck)
        begin_game = getattr(self._base_agent, "begin_game", None)
        if callable(begin_game):
            begin_game(player_index=player_index, own_deck=own_deck)

    def act(self, observation: Any) -> Sequence[int]:
        """Return argmax or a conservatively verified immediate terminal win."""
        started_at = time.perf_counter()
        context_features = self._context.update(observation)
        base_action = tuple(int(index) for index in self._base_agent.act(observation))
        base_telemetry = self._base_telemetry()
        select = _field(observation, "select")
        reason = "not_eligible"
        recommended_action = base_action
        probe_seconds = 0.0
        if (
            forced_action(select) is None
            and core_option_candidates(select)
            and _field(observation, "search_begin_input") is not None
        ):
            probe_started_at = time.perf_counter()
            probe_result = run_runtime_probe_features(
                observation,
                context_features,
                your_deck=self._own_deck,
                sampler=self._sampler,
                rng=self._rng,
                config=SearchRuntimeConfig(
                    enabled=True,
                    conservative_override_enabled=True,
                    worlds=self._config.worlds,
                    top_k=1,
                    manual_coin=self._config.manual_coin,
                    sampler=self._config.sampler,
                ),
            )
            probe_seconds = time.perf_counter() - probe_started_at
            self.immediate_win_probe_calls += 1
            if probe_result is None:
                reason = "no_probe_result"
            else:
                recommended_action, reason = select_verified_immediate_win(
                    select=select,
                    base_action=base_action,
                    probe_result=probe_result,
                )

        recommendation_changed = recommended_action != base_action
        if reason == "base_verified_win":
            self.immediate_win_base_verified_decisions += 1
        elif recommendation_changed:
            self.immediate_win_missed_decisions += 1
        served_action = base_action
        if recommendation_changed and self._config.mode == "apply":
            served_action = recommended_action
            self.immediate_win_applied_decisions += 1
        whole_act_seconds = time.perf_counter() - started_at
        self._last_telemetry = {
            **base_telemetry,
            "whole_act_seconds": whole_act_seconds,
            "actual_search_seconds": probe_seconds,
            "probe_seconds": probe_seconds,
            "search_start_remaining_overage_time": _field(
                observation,
                "remainingOverageTime",
            ),
            "stop_reason": f"immediate_win:{reason}",
            "recommended_action_changed": recommendation_changed,
            "action_changed": served_action != base_action,
            "fallback_available": True,
            "state_leaks": 0,
        }
        return served_action

    def last_act_telemetry(self) -> Mapping[str, Any]:
        """Expose timing and intervention evidence to the arena aggregator."""
        return self._last_telemetry

    def _base_telemetry(self) -> dict[str, Any]:
        telemetry_method = getattr(self._base_agent, "last_act_telemetry", None)
        telemetry = telemetry_method() if callable(telemetry_method) else None
        return dict(telemetry) if isinstance(telemetry, Mapping) else {}


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


__all__ = [
    "ImmediateWinOverrideAgent",
    "ImmediateWinOverrideConfig",
    "select_verified_immediate_win",
]
