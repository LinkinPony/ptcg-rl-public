"""Smoke runner for opponent-pool adapters."""

from __future__ import annotations

import json
import os
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.opponents.spec import (
    BattleAgent,
    OpponentPoolConfig,
    OpponentSpec,
    build_opponent,
    opponent_registry,
    select_opponents,
)
from ptcg_rl.training.arena import (
    BattleSessionFactory,
    GamePlan,
    run_arena_game,
)
from ptcg_rl.training.arena_decks import ArenaDeck, DeckPoolConfig, load_deck_pool

_DECK_PATH_ENV = "POKEMON_TCG_DECK_PATH"


class OpponentSmokeConfig(BaseModel):
    """Hydra-backed config for opponent-pool smoke checks."""

    model_config = ConfigDict(extra="forbid")

    opponent_pool: OpponentPoolConfig = Field(default_factory=OpponentPoolConfig)
    deck_path: Path = Path("data/sample_submission/deck.csv")
    output_path: Path | None = Path("outputs/opponents/smoke/summary.json")
    games_per_opponent: int = 1
    stateful_games: int = 2
    max_steps_per_game: int = 10_000
    seed: int = 0
    fail_on_error: bool = True
    fail_on_illegal: bool = True
    fail_on_truncated: bool = True

    @field_validator("games_per_opponent", "stateful_games", "max_steps_per_game")
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive smoke limits."""
        if value <= 0:
            raise ValueError("smoke limits must be positive")
        return value


def run_opponent_smoke(
    config: OpponentSmokeConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    """Run selected opponents against random and write a JSON smoke summary."""
    registry = opponent_registry(policy_opponents=config.opponent_pool.policy_opponents)
    specs = select_opponents(config.opponent_pool)
    if not specs:
        raise ValueError("opponent smoke selected no opponents")

    default_deck = _single_deck(config.deck_path)
    random_spec = registry["random"]
    rows: list[dict[str, Any]] = []
    game_index = 0
    factory = battle_session_factory
    for opponent_index, spec in enumerate(specs):
        opponent_deck = _deck_for_spec(spec, default_deck)
        opponent = build_opponent(spec, seed=config.seed + 1000 + opponent_index)
        baseline = build_opponent(random_spec, seed=config.seed + 2000 + opponent_index)
        game_count = _game_count(config, spec)
        for repeat in range(game_count):
            plan = GamePlan(
                game_index=game_index,
                seed=config.seed + game_index,
                candidate_deck=default_deck,
                opponent_deck=opponent_deck,
                candidate_seat=0,
            )
            rows.append(
                _run_one_game(
                    plan,
                    spec=spec,
                    opponent=opponent,
                    baseline=baseline,
                    max_steps=config.max_steps_per_game,
                    factory=factory,
                    repeat=repeat,
                )
            )
            game_index += 1

    opponent_rows = _opponent_summary_rows(rows)
    summary = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": _overall_summary(opponent_rows, rows),
        "opponents": opponent_rows,
        "games": rows,
    }
    _write_summary(config.output_path, summary)
    _raise_if_failed(config, summary)
    return summary


def _run_one_game(
    plan: GamePlan,
    *,
    spec: OpponentSpec,
    opponent: BattleAgent,
    baseline: BattleAgent,
    max_steps: int,
    factory: BattleSessionFactory | None,
    repeat: int,
) -> dict[str, Any]:
    try:
        with _deck_path_env(plan.opponent_deck.source):
            row = run_arena_game(
                plan,
                candidate_agent=baseline,
                opponent_agent=opponent,
                max_steps=max_steps,
                battle_session_factory=factory or _default_battle_session,
                on_agent_error=_stop_on_agent_error,
                reset_agents=True,
            )
        output = dict(row)
    except Exception as exc:  # noqa: BLE001 - smoke must report all adapter failures.
        output = _error_row(plan, spec=spec, exc=exc)
    output.update(_spec_fields(spec, repeat=repeat))
    return output


def _stop_on_agent_error(
    exc: Exception,
    player_index: int,
    agent: object,
    observation: Mapping[str, Any],
) -> Sequence[int] | None:
    del exc, player_index, agent, observation
    return None


def _error_row(plan: GamePlan, *, spec: OpponentSpec, exc: Exception) -> dict[str, Any]:
    return {
        "game_index": plan.game_index,
        "seed": plan.seed,
        "candidate_agent": "random",
        "opponent_agent": spec.name,
        "candidate_seat": plan.candidate_seat,
        "winner_index": -1,
        "candidate_result": "truncated",
        "terminal_reason": "agent_error",
        "steps": 0,
        "candidate_deck_id": plan.candidate_deck.deck_id,
        "candidate_deck_hash": plan.candidate_deck.deck_hash,
        "candidate_deck_label": plan.candidate_deck.label,
        "candidate_deck_signature": plan.candidate_deck.signature,
        "opponent_deck_id": plan.opponent_deck.deck_id,
        "opponent_deck_hash": plan.opponent_deck.deck_hash,
        "opponent_deck_label": plan.opponent_deck.label,
        "opponent_deck_signature": plan.opponent_deck.signature,
        "candidate_decisions": 0,
        "opponent_decisions": 0,
        "candidate_action_seconds": 0.0,
        "opponent_action_seconds": 0.0,
        "candidate_mean_action_seconds": 0.0,
        "opponent_mean_action_seconds": 0.0,
        "candidate_illegal_actions": 0,
        "opponent_illegal_actions": 0,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _opponent_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["opponent_agent"]), []).append(row)

    output: list[dict[str, Any]] = []
    for name, group in sorted(grouped.items()):
        first = group[0]
        games = len(group)
        terminal_reasons = Counter(str(row["terminal_reason"]) for row in group)
        opponent_decisions = sum(int(row["opponent_decisions"]) for row in group)
        opponent_seconds = sum(float(row["opponent_action_seconds"]) for row in group)
        opponent_illegal = sum(int(row["opponent_illegal_actions"]) for row in group)
        error_games = terminal_reasons.get("agent_error", 0)
        truncated_games = sum(
            1 for row in group if str(row["terminal_reason"]) != "finished"
        )
        output.append(
            {
                "name": name,
                "tier": int(first["opponent_tier"]),
                "source": str(first["opponent_source"]),
                "vector_safe": bool(first["opponent_vector_safe"]),
                "requires_search": bool(first["opponent_requires_search"]),
                "deck_path": str(first["opponent_spec_deck_path"]),
                "games": games,
                "finished_games": terminal_reasons.get("finished", 0),
                "finished": terminal_reasons.get("finished", 0) == games,
                "terminal_reasons": dict(sorted(terminal_reasons.items())),
                "illegal_actions": opponent_illegal,
                "error_games": error_games,
                "truncated_games": truncated_games,
                "mean_act_seconds": _safe_rate(opponent_seconds, opponent_decisions),
                "decisions": opponent_decisions,
            }
        )
    return output


def _overall_summary(
    opponent_rows: Sequence[Mapping[str, Any]],
    game_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    failed = [
        str(row["name"])
        for row in opponent_rows
        if not bool(row["finished"])
        or int(row["illegal_actions"]) > 0
        or int(row["error_games"]) > 0
    ]
    return {
        "opponents": len(opponent_rows),
        "games": len(game_rows),
        "finished_games": sum(int(row["finished_games"]) for row in opponent_rows),
        "illegal_actions": sum(int(row["illegal_actions"]) for row in opponent_rows),
        "error_games": sum(int(row["error_games"]) for row in opponent_rows),
        "failed_opponents": failed,
        "all_finished": all(bool(row["finished"]) for row in opponent_rows),
    }


def _raise_if_failed(config: OpponentSmokeConfig, summary: Mapping[str, Any]) -> None:
    summary_row = _mapping(summary["summary"])
    failed_opponents = list(summary_row["failed_opponents"])
    should_fail = (
        (config.fail_on_error and int(summary_row["error_games"]) > 0)
        or (config.fail_on_illegal and int(summary_row["illegal_actions"]) > 0)
        or (config.fail_on_truncated and not bool(summary_row["all_finished"]))
    )
    if should_fail:
        raise RuntimeError(
            "opponent smoke failed: " + ", ".join(str(name) for name in failed_opponents)
        )


def _game_count(config: OpponentSmokeConfig, spec: OpponentSpec) -> int:
    if spec.source == "public_opponent":
        return max(config.games_per_opponent, config.stateful_games)
    return config.games_per_opponent


def _deck_for_spec(spec: OpponentSpec, default_deck: ArenaDeck) -> ArenaDeck:
    if spec.deck_path is None:
        return default_deck
    return _single_deck(spec.deck_path)


def _single_deck(path: Path) -> ArenaDeck:
    decks = load_deck_pool(DeckPoolConfig(deck_paths=(path,)))
    if len(decks) != 1:
        raise ValueError(f"expected one deck from {path}, got {len(decks)}")
    return decks[0]


def _spec_fields(spec: OpponentSpec, *, repeat: int) -> dict[str, Any]:
    return {
        "opponent_tier": spec.tier,
        "opponent_source": spec.source,
        "opponent_vector_safe": spec.vector_safe,
        "opponent_requires_search": spec.requires_search,
        "opponent_spec_deck_path": (
            records.display_path(records.repo_path(spec.deck_path))
            if spec.deck_path is not None
            else ""
        ),
        "opponent_smoke_repeat": repeat,
    }


def _write_summary(path: Path | None, summary: Mapping[str, Any]) -> None:
    if path is None:
        return
    resolved = records.repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


@contextmanager
def _deck_path_env(path: str) -> Iterator[None]:
    previous = os.environ.get(_DECK_PATH_ENV)
    os.environ[_DECK_PATH_ENV] = path
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(_DECK_PATH_ENV, None)
        else:
            os.environ[_DECK_PATH_ENV] = previous


def _default_battle_session(
    deck0: Sequence[int],
    deck1: Sequence[int],
) -> Any:
    from ptcg_rl.engine.session import BattleSession

    return BattleSession(deck0, deck1)


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


def _mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping, got {type(value).__name__}")
    return value
