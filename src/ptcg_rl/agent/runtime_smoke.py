"""End-to-end Battle API smoke test for the Kaggle runtime agent."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.runtime import ActTimeConfig, PolicyRuntimeAgent
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.session import BattleSession


class RuntimeBattleSmokeConfig(BaseModel):
    """Config for one local Battle API runtime smoke game."""

    model_config = ConfigDict(extra="forbid")

    deck0_path: Path = Path("data/sample_submission/deck.csv")
    deck1_path: Path = Path("data/sample_submission/deck.csv")
    checkpoint0_path: Path | None = None
    checkpoint1_path: Path | None = None
    max_steps: int = 10_000
    max_total_action_seconds: float = 600.0
    seed: int = 0
    prewarm_engine: bool = True
    fail_on_truncated: bool = True
    output_path: Path | None = Path("outputs/agent/runtime_smoke/summary.json")

    @field_validator("max_steps")
    @classmethod
    def valid_max_steps(cls, value: int) -> int:
        """Reject non-positive step limits."""
        if value <= 0:
            raise ValueError("max_steps must be positive")
        return value

    @field_validator("max_total_action_seconds")
    @classmethod
    def valid_max_total_action_seconds(cls, value: float) -> float:
        """Reject invalid wall-clock budgets."""
        if value <= 0.0:
            raise ValueError("max_total_action_seconds must be positive")
        return value


class BattleSessionLike(Protocol):
    """Subset of ``BattleSession`` used by the runtime smoke."""

    @property
    def observation_dict(self) -> Mapping[str, Any]:
        """Current raw battle observation."""
        ...

    @property
    def start_data(self) -> Any:
        """Battle-start metadata."""
        ...

    def __enter__(self) -> Self:
        """Enter the battle lifecycle."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the battle lifecycle."""
        ...

    def select(self, select: Sequence[int]) -> Mapping[str, Any]:
        """Advance one battle select prompt."""
        ...


BattleSessionFactory = Callable[[Sequence[int], Sequence[int]], BattleSessionLike]


@dataclass(frozen=True)
class _RuntimeSmokeState:
    """Final runtime-smoke counters."""

    report: dict[str, Any]
    should_fail: bool


def run_runtime_battle_smoke(
    config: RuntimeBattleSmokeConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    """Run one local battle through two ``PolicyRuntimeAgent`` instances."""
    deck0 = records.read_deck(records.repo_path(config.deck0_path))
    deck1 = records.read_deck(records.repo_path(config.deck1_path))
    agents = (
        PolicyRuntimeAgent(
            config=_agent_config(
                deck_path=config.deck0_path,
                checkpoint_path=config.checkpoint0_path,
                seed=config.seed + 17,
                prewarm_engine=config.prewarm_engine,
            )
        ),
        PolicyRuntimeAgent(
            config=_agent_config(
                deck_path=config.deck1_path,
                checkpoint_path=config.checkpoint1_path,
                seed=config.seed + 31,
                prewarm_engine=config.prewarm_engine,
            )
        ),
    )
    factory = battle_session_factory or _default_battle_session
    state = _run_battle(config, deck0=deck0, deck1=deck1, agents=agents, factory=factory)
    _write_report(config.output_path, state.report)
    if state.should_fail:
        raise RuntimeError(
            "runtime battle smoke failed: "
            f"{state.report['summary']['terminal_reason']}"
        )
    return state.report


def _run_battle(
    config: RuntimeBattleSmokeConfig,
    *,
    deck0: Sequence[int],
    deck1: Sequence[int],
    agents: tuple[PolicyRuntimeAgent, PolicyRuntimeAgent],
    factory: BattleSessionFactory,
) -> _RuntimeSmokeState:
    decisions = [0, 0]
    action_seconds = [0.0, 0.0]
    illegal_actions = [0, 0]
    prewarm_seconds = [0.0, 0.0]
    prewarm_errors = ["", ""]
    terminal_reason = "max_steps"
    winner_index = -1
    steps_played = 0

    with factory(deck0, deck1) as battle:
        _raise_deck_error_if_any(battle.start_data)
        observation = battle.observation_dict
        for step in range(config.max_steps):
            steps_played = step
            winner_index = _result_index(observation)
            if winner_index >= 0:
                terminal_reason = "finished"
                break

            player_index = _player_index(observation)
            if player_index not in (0, 1):
                terminal_reason = "invalid_player"
                break

            select = _field(observation, "select")
            start_time = time.perf_counter()
            action = tuple(int(index) for index in agents[player_index].act(observation))
            elapsed = time.perf_counter() - start_time
            decisions[player_index] += 1
            action_seconds[player_index] += elapsed
            prewarm_seconds[player_index] = agents[player_index].last_prewarm_seconds
            if agents[player_index].last_prewarm_error is not None:
                prewarm_errors[player_index] = str(
                    agents[player_index].last_prewarm_error
                )

            if not is_legal_action(select, action):
                illegal_actions[player_index] += 1
                terminal_reason = "illegal_action"
                break
            if sum(action_seconds) > config.max_total_action_seconds:
                terminal_reason = "action_timeout_budget"
                break
            observation = battle.select(action)
        else:
            winner_index = _result_index(observation)

    total_action_seconds = sum(action_seconds)
    should_fail = (
        sum(illegal_actions) > 0
        or total_action_seconds > config.max_total_action_seconds
        or (config.fail_on_truncated and terminal_reason != "finished")
    )
    return _RuntimeSmokeState(
        report={
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": config.model_dump(mode="json"),
            "summary": {
                "terminal_reason": terminal_reason,
                "winner_index": winner_index,
                "steps": steps_played,
                "decisions": decisions,
                "illegal_actions": illegal_actions,
                "action_seconds": action_seconds,
                "total_action_seconds": total_action_seconds,
                "mean_action_seconds": _safe_rate(
                    total_action_seconds,
                    sum(decisions),
                ),
                "prewarm_seconds": prewarm_seconds,
                "prewarm_errors": prewarm_errors,
                "search_sessions_opened": 0,
            },
        },
        should_fail=should_fail,
    )


def _agent_config(
    *,
    deck_path: Path,
    checkpoint_path: Path | None,
    seed: int,
    prewarm_engine: bool,
) -> ActTimeConfig:
    return ActTimeConfig(
        deck_path=deck_path,
        checkpoint_path=checkpoint_path,
        seed=seed,
        prewarm_engine=prewarm_engine,
    )


def _default_battle_session(
    deck0: Sequence[int],
    deck1: Sequence[int],
) -> BattleSessionLike:
    return BattleSession(deck0, deck1)


def _raise_deck_error_if_any(start_data: object) -> None:
    error_player = int(getattr(start_data, "errorPlayer", -1))
    if error_player < 0:
        return
    raise ValueError(
        f"deck error: player={error_player} "
        f"type={getattr(start_data, 'errorType', None)}"
    )


def _result_index(observation: Mapping[str, Any]) -> int:
    return _int_field(_field(observation, "current"), "result", -1)


def _player_index(observation: Mapping[str, Any]) -> int:
    return _int_field(_field(observation, "current"), "yourIndex", -1)


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def _write_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    output_path = records.repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
