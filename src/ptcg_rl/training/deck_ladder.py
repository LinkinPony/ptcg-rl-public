"""Deck-vs-deck ladder evaluation for one fixed policy checkpoint."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.actions.selection import forced_action
from ptcg_rl.context import ContextBeliefTracker, OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.training.arena import (
    BattleSessionFactory,
    GamePlan,
    run_arena_game,
)
from ptcg_rl.training.arena_decks import DeckPoolConfig, load_deck_pool
from ptcg_rl.training.arena_utils import field_value
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolve_training_output_dir,
    resolved_training_config_dump,
)


class _PolicyLike(Protocol):
    """Minimal policy interface used by deck ladder agents."""

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Return one legal action for the current observation."""


class DeckLadderConfig(BaseModel):
    """Hydra-backed config for fixed-policy deck ladder evaluation."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: Path = Path(
        "outputs/training/bc/full_h200_100epoch/checkpoint_best.pt"
    )
    device: str = "cpu"
    belief_summary_path: Path | None = None
    deck_pool: DeckPoolConfig = Field(
        default_factory=lambda: DeckPoolConfig(
            deck_paths=(Path("data/sample_submission/deck.csv"),),
        )
    )
    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    games_per_pair: int = 50
    max_steps_per_game: int = 300
    num_workers: int = 1
    seed: int = 0
    elo_initial: float = 1500.0
    elo_k: float = 32.0
    compression: str = "zstd"
    fail_on_agent_error: bool = False

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("device must be non-empty")
        return normalized

    @field_validator("games_per_pair", "max_steps_per_game", "num_workers")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive runner limits."""
        if value <= 0:
            raise ValueError("deck ladder integer limits must be positive")
        return value

    @field_validator("elo_k")
    @classmethod
    def valid_elo_k(cls, value: float) -> float:
        """Reject negative Elo K factors."""
        if value < 0.0:
            raise ValueError("elo_k must be non-negative")
        return value


@dataclass(frozen=True)
class DeckLadderGamePlan:
    """One fixed-policy game between two candidate decks."""

    arena_plan: GamePlan
    repeat_index: int


@dataclass(frozen=True)
class _DeckLadderTask:
    """Pickleable worker task payload."""

    plan: DeckLadderGamePlan
    checkpoint_path: Path
    device: str
    max_steps: int
    belief_summary_path: Path | None = None


class _PolicyDeckAgent:
    """Arena agent wrapper around one shared checkpoint policy."""

    def __init__(
        self,
        *,
        name: str,
        policy: _PolicyLike,
        belief: OpponentBeliefFeatureConfig | None = None,
    ) -> None:
        """Wrap one shared policy with per-game context tracking."""
        self.name = name
        self.policy = policy
        self._tracker = ContextBeliefTracker(belief=belief)

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Reset per-game context with seat and deck metadata."""
        self._tracker.begin_game(player_index=player_index, own_deck=own_deck)
        bind_deck = getattr(self.policy, "bind_own_deck", None)
        if own_deck is not None and callable(bind_deck):
            bind_deck(own_deck)

    def act(self, observation: Any) -> Sequence[int]:
        """Return a greedy policy action for the current observation."""
        action = forced_action(field_value(observation, "select"))
        context_observation = self._tracker.observation_with_context(observation)
        if action is not None:
            return action
        return self.policy.select_action(context_observation)


_POLICY_CACHE: dict[tuple[str, str], _PolicyLike] = {}


def run_deck_ladder(
    config: DeckLadderConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    """Run a fixed-policy deck ladder and write artifacts."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="deck_ladder",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = expand_deck_ladder_plans(config)
    if not plans:
        raise ValueError("deck ladder produced no game plans")

    tasks = tuple(
        _DeckLadderTask(
            plan=plan,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            max_steps=config.max_steps_per_game,
            belief_summary_path=config.belief_summary_path,
        )
        for plan in plans
    )
    rows = _run_tasks(config, tasks, battle_session_factory=battle_session_factory)
    rows.sort(key=lambda row: int(row["game_index"]))

    standings_rows = _apply_online_elo(rows, config=config)
    matchup_rows = deck_ladder_matchup_rows(rows)
    games_path = output_dir / "games.parquet"
    matchups_path = output_dir / "matchups.parquet"
    standings_path = output_dir / "standings.parquet"
    _write_parquet(games_path, rows, compression=config.compression)
    _write_parquet(matchups_path, matchup_rows, compression=config.compression)
    _write_parquet(standings_path, standings_rows, compression=config.compression)

    summary = deck_ladder_summary(
        config,
        game_rows=rows,
        matchup_rows=matchup_rows,
        standings_rows=standings_rows,
        output_dir=output_dir,
        games_path=games_path,
        matchups_path=matchups_path,
        standings_path=standings_path,
    )
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary["summary_path"] = records.display_path(summary_path)
    summary["report_path"] = records.display_path(report_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_deck_ladder_report(summary), encoding="utf-8")
    if config.fail_on_agent_error and int(summary["agent_error_games"]) > 0:
        raise RuntimeError(
            f"deck ladder had agent errors: {summary['agent_error_games']}"
        )
    return summary


def expand_deck_ladder_plans(
    config: DeckLadderConfig,
) -> tuple[DeckLadderGamePlan, ...]:
    """Expand deck pairs and repeated games with alternating seats."""
    decks = load_deck_pool(config.deck_pool)
    if len(decks) < 2:
        raise ValueError("deck ladder requires at least two decks")
    plans: list[DeckLadderGamePlan] = []
    for left_index, deck_a in enumerate(decks):
        for deck_b in decks[left_index + 1 :]:
            for repeat_index in range(config.games_per_pair):
                arena_plan = GamePlan(
                    game_index=len(plans),
                    seed=config.seed + len(plans),
                    candidate_deck=deck_a,
                    opponent_deck=deck_b,
                    candidate_seat=repeat_index % 2,
                )
                plans.append(
                    DeckLadderGamePlan(
                        arena_plan=arena_plan,
                        repeat_index=repeat_index,
                    )
                )
    return tuple(plans)


def deck_ladder_matchup_rows(
    game_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate deck ladder rows by unordered deck pair."""
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in game_rows:
        key = (str(row["deck_a_id"]), str(row["deck_b_id"]))
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        first = rows[0]
        games = len(rows)
        scores = [_score(row["deck_a_result"]) for row in rows]
        deck_a_score_rate = sum(scores) / float(games)
        result_counts = Counter(str(row["deck_a_result"]) for row in rows)
        output.append(
            {
                "deck_a_id": first["deck_a_id"],
                "deck_a_hash": first["deck_a_hash"],
                "deck_a_label": first["deck_a_label"],
                "deck_b_id": first["deck_b_id"],
                "deck_b_hash": first["deck_b_hash"],
                "deck_b_label": first["deck_b_label"],
                "games": games,
                "deck_a_wins": result_counts.get("win", 0),
                "deck_b_wins": result_counts.get("loss", 0),
                "draws": result_counts.get("draw", 0),
                "truncated": result_counts.get("truncated", 0),
                "deck_a_score_rate": deck_a_score_rate,
                "deck_b_score_rate": 1.0 - deck_a_score_rate,
                "agent_error_games": sum(
                    1 for row in rows if row["terminal_reason"] == "agent_error"
                ),
            }
        )
    output.sort(key=lambda row: (row["deck_a_label"], row["deck_b_label"]))
    return output


def deck_ladder_summary(
    config: DeckLadderConfig,
    *,
    game_rows: Sequence[Mapping[str, Any]],
    matchup_rows: Sequence[Mapping[str, Any]],
    standings_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    games_path: Path,
    matchups_path: Path,
    standings_path: Path,
) -> dict[str, Any]:
    """Build the JSON summary for one deck ladder run."""
    if not game_rows:
        raise ValueError("cannot summarize an empty deck ladder run")
    return {
        "games": len(game_rows),
        "matchups": len(matchup_rows),
        "decks": len(standings_rows),
        "checkpoint_path": records.display_path(records.repo_path(config.checkpoint_path)),
        "device": config.device,
        "games_per_pair": config.games_per_pair,
        "elo_initial": config.elo_initial,
        "elo_k": config.elo_k,
        "num_workers": config.num_workers,
        "max_steps_per_game": config.max_steps_per_game,
        "agent_error_games": sum(
            1 for row in game_rows if row["terminal_reason"] == "agent_error"
        ),
        "truncated_games": sum(
            1 for row in game_rows if row["deck_a_result"] == "truncated"
        ),
        "run": config.run.model_dump(mode="json"),
        "output_dir": records.display_path(output_dir),
        "games_path": records.display_path(games_path),
        "matchups_path": records.display_path(matchups_path),
        "standings_path": records.display_path(standings_path),
        "standings": [dict(row) for row in standings_rows],
        "matchup_rows": [dict(row) for row in matchup_rows],
        "config": resolved_training_config_dump(
            config,
            task_name="deck_ladder",
            run=config.run,
            output_dir=config.output_dir,
        ),
    }


def render_deck_ladder_report(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown report from a deck ladder summary."""
    run = _mapping(summary["run"])
    lines = [
        f"# Deck Ladder Report: {run['version']}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Games | {summary['games']} |",
        f"| Decks | {summary['decks']} |",
        f"| Matchups | {summary['matchups']} |",
        f"| Elo initial | {float(summary['elo_initial']):.1f} |",
        f"| Elo K | {float(summary['elo_k']):.1f} |",
        f"| Truncated games | {summary['truncated_games']} |",
        f"| Agent error games | {summary['agent_error_games']} |",
        "",
        "## Standings",
        "",
        "| Rank | Deck | Elo | Delta | Games | W-L-D-T | Score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in _sequence(summary["standings"]):
        data = _mapping(row)
        lines.append(
            "| {rank} | {deck} | {elo:.1f} | {delta:+.1f} | {games} | "
            "{wins}-{losses}-{draws}-{truncated} | {score} |".format(
                rank=data["rank"],
                deck=data["deck_label"],
                elo=float(data["elo"]),
                delta=float(data["elo_delta"]),
                games=data["games"],
                wins=data["wins"],
                losses=data["losses"],
                draws=data["draws"],
                truncated=data["truncated"],
                score=_pct(float(data["score_rate"])),
            )
        )
    lines.extend(
        [
            "",
            "## Matchups",
            "",
            "| Deck A | Deck B | Games | A Wins | B Wins | Draw | Trunc | A Score |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in _sequence(summary["matchup_rows"]):
        data = _mapping(row)
        lines.append(
            "| {deck_a} | {deck_b} | {games} | {a_wins} | {b_wins} | "
            "{draws} | {truncated} | {score} |".format(
                deck_a=data["deck_a_label"],
                deck_b=data["deck_b_label"],
                games=data["games"],
                a_wins=data["deck_a_wins"],
                b_wins=data["deck_b_wins"],
                draws=data["draws"],
                truncated=data["truncated"],
                score=_pct(float(data["deck_a_score_rate"])),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _run_tasks(
    config: DeckLadderConfig,
    tasks: Sequence[_DeckLadderTask],
    *,
    battle_session_factory: BattleSessionFactory | None,
) -> list[dict[str, Any]]:
    if config.num_workers <= 1 or battle_session_factory is not None:
        return [
            _run_deck_ladder_task(task, battle_session_factory=battle_session_factory)
            for task in tasks
        ]

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        mp_context=context,
    ) as executor:
        return list(executor.map(_run_deck_ladder_task, tasks))


def _run_deck_ladder_task(
    task: _DeckLadderTask,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    plan = task.plan.arena_plan
    try:
        belief = None
        if task.belief_summary_path is not None:
            belief = OpponentBeliefFeatureConfig(
                deck_signature_summary_path=records.repo_path(
                    task.belief_summary_path
                ),
            )
        deck_a_agent = _PolicyDeckAgent(
            name=f"deck:{plan.candidate_deck.label}",
            policy=_deck_checkpoint_policy(
                task.checkpoint_path,
                device=task.device,
                own_deck=plan.candidate_deck.cards,
            ),
            belief=belief,
        )
        deck_b_agent = _PolicyDeckAgent(
            name=f"deck:{plan.opponent_deck.label}",
            policy=_deck_checkpoint_policy(
                task.checkpoint_path,
                device=task.device,
                own_deck=plan.opponent_deck.cards,
            ),
            belief=belief,
        )
        row = run_arena_game(
            plan,
            candidate_agent=deck_a_agent,
            opponent_agent=deck_b_agent,
            max_steps=task.max_steps,
            battle_session_factory=battle_session_factory or _default_battle_session,
            on_agent_error=_stop_on_agent_error,
            reset_agents=False,
        )
        output = dict(row)
    except Exception as exc:  # noqa: BLE001 - ladder must isolate game failures.
        output = _error_row(task, exc)
    _add_metadata_fields(output, task)
    return output


def _deck_checkpoint_policy(
    checkpoint_path: Path,
    *,
    device: str,
    own_deck: Sequence[int],
) -> _PolicyLike:
    resolved = records.repo_path(checkpoint_path)
    signature = records.deck_signature(list(own_deck))
    key = (str(resolved), f"{device}:{signature}")
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    policy = _checkpoint_policy(resolved, device=device)
    bind_own_deck = getattr(policy, "bind_own_deck", None)
    if callable(bind_own_deck):
        bind_own_deck(own_deck)
    _POLICY_CACHE[key] = policy
    return policy


def _checkpoint_policy(checkpoint_path: Path, *, device: str) -> _PolicyLike:
    """Load one policy; exact-deck caching is handled by its caller."""
    from ptcg_rl.agent.runtime import CheckpointPolicy

    return CheckpointPolicy(records.repo_path(checkpoint_path), device=device)


def _stop_on_agent_error(
    exc: Exception,
    player_index: int,
    agent: object,
    observation: Mapping[str, Any],
) -> Sequence[int] | None:
    del exc, player_index, agent, observation
    return None


def _add_metadata_fields(row: dict[str, Any], task: _DeckLadderTask) -> None:
    plan = task.plan.arena_plan
    deck_a_result = str(row.get("candidate_result", "truncated"))
    row.update(
        {
            "checkpoint_path": records.display_path(
                records.repo_path(task.checkpoint_path)
            ),
            "deck_ladder_repeat": task.plan.repeat_index,
            "deck_a_seat": plan.candidate_seat,
            "deck_b_seat": 1 - plan.candidate_seat,
            "deck_a_id": plan.candidate_deck.deck_id,
            "deck_a_hash": plan.candidate_deck.deck_hash,
            "deck_a_label": plan.candidate_deck.label,
            "deck_a_signature": plan.candidate_deck.signature,
            "deck_a_source": plan.candidate_deck.source,
            "deck_b_id": plan.opponent_deck.deck_id,
            "deck_b_hash": plan.opponent_deck.deck_hash,
            "deck_b_label": plan.opponent_deck.label,
            "deck_b_signature": plan.opponent_deck.signature,
            "deck_b_source": plan.opponent_deck.source,
            "deck_a_result": deck_a_result,
            "deck_b_result": _inverse_result(deck_a_result),
            "deck_a_score": _score(deck_a_result),
            "deck_b_score": 1.0 - _score(deck_a_result),
        }
    )


def _error_row(task: _DeckLadderTask, exc: Exception) -> dict[str, Any]:
    plan = task.plan.arena_plan
    return {
        "game_index": plan.game_index,
        "seed": plan.seed,
        "candidate_agent": f"deck:{plan.candidate_deck.label}",
        "opponent_agent": f"deck:{plan.opponent_deck.label}",
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


def _apply_online_elo(
    rows: Sequence[dict[str, Any]],
    *,
    config: DeckLadderConfig,
) -> list[dict[str, Any]]:
    deck_stats: dict[str, dict[str, Any]] = {}
    ratings: dict[str, float] = {}
    for row in rows:
        _ensure_deck_stats(deck_stats, row, prefix="deck_a", config=config)
        _ensure_deck_stats(deck_stats, row, prefix="deck_b", config=config)

    for deck_id in deck_stats:
        ratings[deck_id] = config.elo_initial

    for row in rows:
        deck_a_id = str(row["deck_a_id"])
        deck_b_id = str(row["deck_b_id"])
        before_a = ratings[deck_a_id]
        before_b = ratings[deck_b_id]
        score_a = _score(row["deck_a_result"])
        expected_a = _elo_expected(before_a, before_b)
        after_a = before_a + config.elo_k * (score_a - expected_a)
        after_b = before_b + config.elo_k * ((1.0 - score_a) - (1.0 - expected_a))
        ratings[deck_a_id] = after_a
        ratings[deck_b_id] = after_b
        row.update(
            {
                "deck_a_elo_before": before_a,
                "deck_b_elo_before": before_b,
                "deck_a_elo_after": after_a,
                "deck_b_elo_after": after_b,
                "deck_a_expected_score": expected_a,
                "deck_b_expected_score": 1.0 - expected_a,
            }
        )
        _observe_deck(deck_stats[deck_a_id], row["deck_a_result"])
        _observe_deck(deck_stats[deck_b_id], row["deck_b_result"])

    standings = []
    for deck_id, stats in deck_stats.items():
        games = int(stats["games"])
        score_total = float(stats["score_total"])
        elo = ratings[deck_id]
        standings.append(
            {
                "deck_id": deck_id,
                "deck_hash": stats["deck_hash"],
                "deck_label": stats["deck_label"],
                "deck_signature": stats["deck_signature"],
                "deck_source": stats["deck_source"],
                "games": games,
                "wins": int(stats["wins"]),
                "losses": int(stats["losses"]),
                "draws": int(stats["draws"]),
                "truncated": int(stats["truncated"]),
                "score_rate": score_total / float(games) if games > 0 else 0.0,
                "elo": elo,
                "elo_delta": elo - config.elo_initial,
            }
        )
    standings.sort(
        key=lambda row: (
            -float(row["elo"]),
            -float(row["score_rate"]),
            str(row["deck_label"]),
        )
    )
    for rank, row in enumerate(standings, start=1):
        row["rank"] = rank
    return standings


def _ensure_deck_stats(
    deck_stats: dict[str, dict[str, Any]],
    row: Mapping[str, Any],
    *,
    prefix: str,
    config: DeckLadderConfig,
) -> None:
    deck_id = str(row[f"{prefix}_id"])
    if deck_id in deck_stats:
        return
    deck_stats[deck_id] = {
        "deck_hash": str(row[f"{prefix}_hash"]),
        "deck_label": str(row[f"{prefix}_label"]),
        "deck_signature": str(row[f"{prefix}_signature"]),
        "deck_source": str(row[f"{prefix}_source"]),
        "games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "truncated": 0,
        "score_total": 0.0,
        "elo": config.elo_initial,
    }


def _observe_deck(stats: dict[str, Any], result: Any) -> None:
    result_text = str(result)
    stats["games"] = int(stats["games"]) + 1
    if result_text == "win":
        stats["wins"] = int(stats["wins"]) + 1
    elif result_text == "loss":
        stats["losses"] = int(stats["losses"]) + 1
    elif result_text == "draw":
        stats["draws"] = int(stats["draws"]) + 1
    else:
        stats["truncated"] = int(stats["truncated"]) + 1
    stats["score_total"] = float(stats["score_total"]) + _score(result_text)


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _default_battle_session(deck0: Sequence[int], deck1: Sequence[int]) -> Any:
    return BattleSession(deck0, deck1)


def _inverse_result(result: str) -> str:
    if result == "win":
        return "loss"
    if result == "loss":
        return "win"
    return result


def _score(result: Any) -> float:
    if result == "win":
        return 1.0
    if result == "loss":
        return 0.0
    return 0.5


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression: str,
) -> None:
    if not rows:
        raise ValueError(f"cannot write empty Parquet table: {path}")
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression=compression)
