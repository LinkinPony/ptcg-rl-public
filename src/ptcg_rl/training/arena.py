"""Local Battle API arena for agent-vs-agent evaluation."""

from __future__ import annotations

import json
import math
import random
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, Self

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.actions.selection import is_legal_action, random_legal_action
from ptcg_rl.agent.search.budget import (
    ActTimeLedger,
    ActTimeLedgerConfig,
    ActTimeTimeoutError,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.training.arena_agents import (
    ArenaAgent,
    ArenaAgentConfig,
    build_arena_agent,
)
from ptcg_rl.training.arena_decks import ArenaDeck, DeckPoolConfig, load_deck_pool
from ptcg_rl.training.arena_telemetry import RuntimeTelemetryAccumulator
from ptcg_rl.training.arena_utils import field_value, int_field
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolve_training_output_dir,
    resolved_training_config_dump,
)


class BattleSessionLike(Protocol):
    """Subset of ``BattleSession`` used by the arena."""

    @property
    def observation_dict(self) -> Mapping[str, Any]:
        """Current raw engine observation."""

    @property
    def start_data(self) -> Any:
        """Engine battle-start metadata."""

    def __enter__(self) -> Self:
        """Enter the battle lifecycle."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Finish the battle lifecycle."""

    def select(self, select: Sequence[int]) -> Mapping[str, Any]:
        """Advance the battle with one select response."""


BattleSessionFactory = Callable[[Sequence[int], Sequence[int]], BattleSessionLike]
AgentErrorHandler = Callable[
    [Exception, int, ArenaAgent, Mapping[str, Any]],
    Sequence[int] | None,
]


class ArenaConfig(BaseModel):
    """Config for running local battle arena games."""

    model_config = ConfigDict(extra="forbid")

    candidate_decks: DeckPoolConfig = Field(
        default_factory=lambda: DeckPoolConfig(
            deck_paths=(Path("data/sample_submission/deck.csv"),),
        )
    )
    opponent_decks: DeckPoolConfig = Field(
        default_factory=lambda: DeckPoolConfig(
            deck_paths=(Path("data/sample_submission/deck.csv"),),
        )
    )
    candidate_agent: ArenaAgentConfig = Field(
        default_factory=lambda: ArenaAgentConfig(kind="random", label="candidate_random")
    )
    opponent_agent: ArenaAgentConfig = Field(
        default_factory=lambda: ArenaAgentConfig(kind="random", label="opponent_random")
    )
    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    games_per_pair: int = 2
    max_steps_per_game: int = 10000
    mirror_sides: bool = True
    seed: int = 0
    compression: str = "zstd"
    act_time_ledger: ActTimeLedgerConfig = Field(
        default_factory=lambda: ActTimeLedgerConfig(enabled=False)
    )

    @field_validator("games_per_pair", "max_steps_per_game")
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive arena limits."""
        if value <= 0:
            raise ValueError("arena limits must be positive")
        return value


@dataclass(frozen=True)
class GamePlan:
    """One concrete battle pairing."""

    game_index: int
    seed: int
    candidate_deck: ArenaDeck
    opponent_deck: ArenaDeck
    candidate_seat: int


def run_arena(
    config: ArenaConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    """Run configured arena games and write Parquet/JSON artifacts."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="arena",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_deck_pool(config.candidate_decks)
    opponents = load_deck_pool(config.opponent_decks)
    if not candidates or not opponents:
        raise ValueError("arena deck pools must both be non-empty")

    factory = battle_session_factory or _default_battle_session
    candidate_agent = build_arena_agent(
        config.candidate_agent,
        seed=config.seed + 11,
    )
    opponent_agent = build_arena_agent(
        config.opponent_agent,
        seed=config.seed + 29,
    )
    plans = _game_plans(config, candidates, opponents)
    rows = [
        run_arena_game(
            plan,
            candidate_agent=candidate_agent,
            opponent_agent=opponent_agent,
            max_steps=config.max_steps_per_game,
            battle_session_factory=factory,
            act_time_ledger_config=config.act_time_ledger,
        )
        for plan in plans
    ]

    games_path = output_dir / "games.parquet"
    matchups_path = output_dir / "matchups.parquet"
    matchup_rows = matchup_summary_rows(rows)
    _write_parquet(games_path, rows, compression=config.compression)
    _write_parquet(matchups_path, matchup_rows, compression=config.compression)
    summary = arena_summary(
        rows,
        matchup_rows=matchup_rows,
        games_path=games_path,
        matchups_path=matchups_path,
    )
    summary["run"] = config.run.model_dump(mode="json")
    summary["output_dir"] = records.display_path(output_dir)
    summary["config"] = resolved_training_config_dump(
        config,
        task_name="arena",
        run=config.run,
        output_dir=config.output_dir,
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_arena_game(
    plan: GamePlan,
    *,
    candidate_agent: ArenaAgent,
    opponent_agent: ArenaAgent,
    max_steps: int,
    battle_session_factory: BattleSessionFactory,
    on_agent_error: AgentErrorHandler | None = None,
    reset_agents: bool = False,
    act_time_ledger_config: ActTimeLedgerConfig | None = None,
) -> dict[str, Any]:
    """Run one local battle and return a flat result row."""
    if reset_agents:
        _reset_agent(candidate_agent)
        _reset_agent(opponent_agent)
    deck_by_seat = _seat_decks(plan)
    agent_by_seat = _seat_agents(plan, candidate_agent, opponent_agent)
    for seat, agent in enumerate(agent_by_seat):
        _begin_agent_game(agent, player_index=seat, own_deck=deck_by_seat[seat].cards)
    fallback_rng = random.Random(plan.seed + 1009)
    act_time_ledger = ActTimeLedger(
        act_time_ledger_config or ActTimeLedgerConfig(enabled=False)
    )
    decisions = [0, 0]
    illegal_actions = [0, 0]
    action_seconds = [0.0, 0.0]
    runtime_telemetry = [
        RuntimeTelemetryAccumulator(
            expected=callable(getattr(agent, "last_act_telemetry", None))
        )
        for agent in agent_by_seat
    ]
    terminal_reason = "max_steps"
    winner_index = -1
    steps_played = 0
    error_type = ""
    error_message = ""
    error_player_index = -1
    error_actor = ""

    with battle_session_factory(deck_by_seat[0].cards, deck_by_seat[1].cards) as battle:
        _raise_deck_error_if_any(battle.start_data)
        observation: Mapping[str, Any] = battle.observation_dict
        for step in range(max_steps):
            steps_played = step
            winner_index = _result_index(observation)
            if winner_index >= 0:
                terminal_reason = "finished"
                break

            player_index = _player_index(observation)
            if player_index not in (0, 1):
                terminal_reason = "invalid_player"
                break

            select = field_value(observation, "select")
            acting_agent = agent_by_seat[player_index]
            try:
                callback_observation = act_time_ledger.observation_for(
                    observation,
                    player_index,
                )
            except ActTimeTimeoutError as exc:
                winner_index = 1 - player_index
                terminal_reason = "act_time_timeout"
                error_type = type(exc).__name__
                error_message = str(exc)
                error_player_index = player_index
                error_actor = (
                    "candidate" if player_index == plan.candidate_seat else "opponent"
                )
                break
            start_time = time.perf_counter()
            try:
                action = tuple(
                    int(index) for index in acting_agent.act(callback_observation)
                )
            except Exception as exc:
                elapsed = time.perf_counter() - start_time
                decisions[player_index] += 1
                action_seconds[player_index] += elapsed
                error_type = type(exc).__name__
                error_message = str(exc)
                error_player_index = player_index
                error_actor = (
                    "candidate" if player_index == plan.candidate_seat else "opponent"
                )
                if on_agent_error is None:
                    raise
                replacement = on_agent_error(
                    exc,
                    player_index,
                    acting_agent,
                    callback_observation,
                )
                if replacement is None:
                    terminal_reason = "agent_error"
                    break
                action = tuple(int(index) for index in replacement)
            else:
                elapsed = time.perf_counter() - start_time
                decisions[player_index] += 1
                action_seconds[player_index] += elapsed

            runtime_telemetry[player_index].record(acting_agent, elapsed)

            try:
                act_time_ledger.charge(player_index, elapsed)
            except ActTimeTimeoutError as exc:
                winner_index = 1 - player_index
                terminal_reason = "act_time_timeout"
                error_type = type(exc).__name__
                error_message = str(exc)
                error_player_index = player_index
                error_actor = (
                    "candidate" if player_index == plan.candidate_seat else "opponent"
                )
                break

            if not is_legal_action(select, action):
                illegal_actions[player_index] += 1
                action = random_legal_action(select, rng=fallback_rng)
            observation = battle.select(action)
        else:
            winner_index = _result_index(observation)

    candidate_decisions = decisions[plan.candidate_seat]
    opponent_seat = 1 - plan.candidate_seat
    opponent_decisions = decisions[opponent_seat]
    row = {
        "game_index": plan.game_index,
        "seed": plan.seed,
        "candidate_agent": candidate_agent.name,
        "opponent_agent": opponent_agent.name,
        "candidate_seat": plan.candidate_seat,
        "winner_index": winner_index,
        "candidate_result": _candidate_result(winner_index, plan.candidate_seat, terminal_reason),
        "terminal_reason": terminal_reason,
        "steps": steps_played,
        "candidate_deck_id": plan.candidate_deck.deck_id,
        "candidate_deck_hash": plan.candidate_deck.deck_hash,
        "candidate_deck_label": plan.candidate_deck.label,
        "candidate_deck_signature": plan.candidate_deck.signature,
        "opponent_deck_id": plan.opponent_deck.deck_id,
        "opponent_deck_hash": plan.opponent_deck.deck_hash,
        "opponent_deck_label": plan.opponent_deck.label,
        "opponent_deck_signature": plan.opponent_deck.signature,
        "candidate_decisions": candidate_decisions,
        "opponent_decisions": opponent_decisions,
        "candidate_action_seconds": action_seconds[plan.candidate_seat],
        "opponent_action_seconds": action_seconds[opponent_seat],
        "candidate_mean_action_seconds": _safe_rate(
            action_seconds[plan.candidate_seat],
            candidate_decisions,
        ),
        "opponent_mean_action_seconds": _safe_rate(
            action_seconds[opponent_seat],
            opponent_decisions,
        ),
        "candidate_illegal_actions": illegal_actions[plan.candidate_seat],
        "opponent_illegal_actions": illegal_actions[opponent_seat],
        "candidate_act_time_used_seconds": act_time_ledger.used(plan.candidate_seat),
        "opponent_act_time_used_seconds": act_time_ledger.used(opponent_seat),
        "candidate_remaining_overage_time": act_time_ledger.remaining(
            plan.candidate_seat
        ),
        "opponent_remaining_overage_time": act_time_ledger.remaining(opponent_seat),
        "act_time_startup_charge_seconds": (
            act_time_ledger.config.startup_charge_seconds
            if act_time_ledger.config.enabled
            else 0.0
        ),
        "act_time_timeout_seat": (
            error_player_index if terminal_reason == "act_time_timeout" else -1
        ),
        "error_type": error_type,
        "error_message": error_message,
        "error_player_index": error_player_index,
        "error_actor": error_actor,
    }
    row.update(runtime_telemetry[plan.candidate_seat].row_fields("candidate"))
    row.update(runtime_telemetry[opponent_seat].row_fields("opponent"))
    return row


def matchup_summary_rows(game_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate game rows into candidate-vs-opponent matchup records."""
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for row in game_rows:
        key = (
            str(row["candidate_agent"]),
            str(row["opponent_agent"]),
            str(row["candidate_deck_id"]),
            str(row["opponent_deck_id"]),
        )
        grouped.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for rows in grouped.values():
        first = rows[0]
        scores = [_score(row["candidate_result"]) for row in rows]
        wins = sum(1 for row in rows if row["candidate_result"] == "win")
        losses = sum(1 for row in rows if row["candidate_result"] == "loss")
        draws = sum(1 for row in rows if row["candidate_result"] == "draw")
        truncated = sum(1 for row in rows if row["candidate_result"] == "truncated")
        games = len(rows)
        score_rate = sum(scores) / float(games)
        ci_low, ci_high = _normal_ci(score_rate, games)
        output.append(
            {
                "candidate_agent": first["candidate_agent"],
                "opponent_agent": first["opponent_agent"],
                "candidate_deck_id": first["candidate_deck_id"],
                "candidate_deck_hash": first["candidate_deck_hash"],
                "candidate_deck_label": first["candidate_deck_label"],
                "opponent_deck_id": first["opponent_deck_id"],
                "opponent_deck_hash": first["opponent_deck_hash"],
                "opponent_deck_label": first["opponent_deck_label"],
                "games": games,
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "truncated": truncated,
                "win_rate": wins / float(games),
                "score_rate": score_rate,
                "score_ci95_low": ci_low,
                "score_ci95_high": ci_high,
                "candidate_mean_action_seconds": _safe_rate(
                    sum(float(row["candidate_action_seconds"]) for row in rows),
                    sum(int(row["candidate_decisions"]) for row in rows),
                ),
                "opponent_mean_action_seconds": _safe_rate(
                    sum(float(row["opponent_action_seconds"]) for row in rows),
                    sum(int(row["opponent_decisions"]) for row in rows),
                ),
                "candidate_illegal_actions": sum(
                    int(row["candidate_illegal_actions"]) for row in rows
                ),
                "opponent_illegal_actions": sum(
                    int(row["opponent_illegal_actions"]) for row in rows
                ),
            }
        )
    output.sort(key=lambda row: (row["score_rate"], row["games"]), reverse=True)
    return output


def arena_summary(
    game_rows: Sequence[Mapping[str, Any]],
    *,
    matchup_rows: Sequence[Mapping[str, Any]],
    games_path: Path,
    matchups_path: Path,
) -> dict[str, Any]:
    """Build a compact JSON summary for one arena run."""
    if not game_rows:
        raise ValueError("cannot summarize an empty arena run")
    result_counts = Counter(str(row["candidate_result"]) for row in game_rows)
    games = len(game_rows)
    score_rate = sum(_score(row["candidate_result"]) for row in game_rows) / float(games)
    ci_low, ci_high = _normal_ci(score_rate, games)
    return {
        "games": games,
        "matchups": len(matchup_rows),
        "wins": result_counts.get("win", 0),
        "losses": result_counts.get("loss", 0),
        "draws": result_counts.get("draw", 0),
        "truncated": result_counts.get("truncated", 0),
        "candidate_win_rate": result_counts.get("win", 0) / float(games),
        "candidate_score_rate": score_rate,
        "candidate_score_ci95_low": ci_low,
        "candidate_score_ci95_high": ci_high,
        "candidate_mean_action_seconds": _safe_rate(
            sum(float(row["candidate_action_seconds"]) for row in game_rows),
            sum(int(row["candidate_decisions"]) for row in game_rows),
        ),
        "opponent_mean_action_seconds": _safe_rate(
            sum(float(row["opponent_action_seconds"]) for row in game_rows),
            sum(int(row["opponent_decisions"]) for row in game_rows),
        ),
        "candidate_illegal_actions": sum(
            int(row["candidate_illegal_actions"]) for row in game_rows
        ),
        "opponent_illegal_actions": sum(
            int(row["opponent_illegal_actions"]) for row in game_rows
        ),
        "games_path": records.display_path(games_path),
        "matchups_path": records.display_path(matchups_path),
    }


def _default_battle_session(
    deck0: Sequence[int],
    deck1: Sequence[int],
) -> BattleSessionLike:
    return BattleSession(deck0, deck1)


def _game_plans(
    config: ArenaConfig,
    candidates: Sequence[ArenaDeck],
    opponents: Sequence[ArenaDeck],
) -> tuple[GamePlan, ...]:
    plans: list[GamePlan] = []
    seats = (0, 1) if config.mirror_sides else (0,)
    for candidate in candidates:
        for opponent in opponents:
            for _repeat in range(config.games_per_pair):
                for candidate_seat in seats:
                    plans.append(
                        GamePlan(
                            game_index=len(plans),
                            seed=config.seed + len(plans),
                            candidate_deck=candidate,
                            opponent_deck=opponent,
                            candidate_seat=candidate_seat,
                        )
                    )
    return tuple(plans)


def _seat_decks(plan: GamePlan) -> tuple[ArenaDeck, ArenaDeck]:
    if plan.candidate_seat == 0:
        return (plan.candidate_deck, plan.opponent_deck)
    return (plan.opponent_deck, plan.candidate_deck)


def _seat_agents(
    plan: GamePlan,
    candidate_agent: ArenaAgent,
    opponent_agent: ArenaAgent,
) -> tuple[ArenaAgent, ArenaAgent]:
    if plan.candidate_seat == 0:
        return (candidate_agent, opponent_agent)
    return (opponent_agent, candidate_agent)


def _reset_agent(agent: ArenaAgent) -> None:
    reset = getattr(agent, "reset", None)
    if callable(reset):
        reset()


def _begin_agent_game(
    agent: ArenaAgent,
    *,
    player_index: int,
    own_deck: Sequence[int],
) -> None:
    begin_game = getattr(agent, "begin_game", None)
    if callable(begin_game):
        begin_game(player_index=player_index, own_deck=own_deck)


def _raise_deck_error_if_any(start_data: object) -> None:
    error_player = int(getattr(start_data, "errorPlayer", -1))
    if error_player < 0:
        return
    raise ValueError(
        f"deck error: player={error_player} "
        f"type={getattr(start_data, 'errorType', None)}"
    )


def _candidate_result(
    winner_index: int,
    candidate_seat: int,
    terminal_reason: str,
) -> str:
    if terminal_reason not in {"finished", "act_time_timeout"}:
        return "truncated"
    if winner_index == candidate_seat:
        return "win"
    if winner_index == 2:
        return "draw"
    return "loss"


def _result_index(observation: Mapping[str, Any]) -> int:
    return int_field(field_value(observation, "current"), "result", -1)


def _player_index(observation: Mapping[str, Any]) -> int:
    return int_field(field_value(observation, "current"), "yourIndex", -1)


def _score(result: Any) -> float:
    if result == "win":
        return 1.0
    if result == "draw":
        return 0.5
    return 0.0


def _normal_ci(rate: float, count: int) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 0.0)
    radius = 1.96 * math.sqrt(max(0.0, rate * (1.0 - rate)) / float(count))
    return (max(0.0, rate - radius), min(1.0, rate + radius))


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / float(denominator)


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
