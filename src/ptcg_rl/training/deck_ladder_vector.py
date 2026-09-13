"""Vectorized deck-vs-deck ladder evaluation for one fixed checkpoint."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow.parquet as pq
import torch
from pydantic import ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.vector_battle import (
    FinishedGame,
    VectorGame,
    finish_pointer_battle,
    load_cg_sim_lib,
    result_index,
    select_pointer_battle,
    start_pointer_battle,
)
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    build_agent_policy_value_net,
)
from ptcg_rl.profiling import StageTimer
from ptcg_rl.rl.collection import AutocastMode, ModelRolloutPolicy
from ptcg_rl.rl.rollout import (
    RolloutActors,
    RolloutBeliefConfig,
    RolloutProbeConfig,
    RolloutStepper,
)
from ptcg_rl.training.deck_ladder import (
    DeckLadderConfig,
    DeckLadderGamePlan,
    _apply_online_elo,
    _inverse_result,
    _score,
    _write_parquet,
    deck_ladder_matchup_rows,
    deck_ladder_summary,
    expand_deck_ladder_plans,
    render_deck_ladder_report,
)
from ptcg_rl.training.run_config import resolve_training_output_dir


class VectorDeckLadderConfig(DeckLadderConfig):
    """Config for batched-policy deck ladder evaluation."""

    model_config = ConfigDict(extra="forbid")

    num_concurrent_games: int = 512
    autocast: AutocastMode = "bf16"
    compile_model: bool = False
    profile_sync_cuda: bool = False
    resume: bool = True
    flush_interval_games: int = 25

    @field_validator("num_concurrent_games", "flush_interval_games")
    @classmethod
    def valid_positive_vector_int(cls, value: int) -> int:
        """Reject invalid vector runner integer limits."""
        if value <= 0:
            raise ValueError("vector runner integer limits must be positive")
        return value


@dataclass(frozen=True)
class _PoolPlan:
    """One finite vector-pool game assignment."""

    ladder_plan: DeckLadderGamePlan
    deck_pair: tuple[tuple[int, ...], tuple[int, ...]]


class _FiniteVectorDeckLadderPool:
    """Finite ``VectorBattlePool`` variant with max-step truncation."""

    def __init__(
        self,
        plans: Sequence[DeckLadderGamePlan],
        *,
        num_games: int,
        max_steps: int,
        timer: StageTimer | None = None,
    ) -> None:
        """Start up to ``num_games`` finite ladder games."""
        if num_games <= 0:
            raise ValueError("num_games must be positive")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._plans = tuple(_pool_plan(plan) for plan in plans)
        self._max_live = int(num_games)
        self._max_steps = int(max_steps)
        self._timer = timer
        self._lib = load_cg_sim_lib()
        self._next_plan = 0
        self._games: dict[str, VectorGame] = {}
        self._closed = False
        self.plans_by_game_id: dict[str, DeckLadderGamePlan] = {}
        try:
            while len(self._games) < self._max_live and self._next_plan < len(
                self._plans
            ):
                self._start_next_game()
        except Exception:
            self.close()
            raise

    def __enter__(self) -> _FiniteVectorDeckLadderPool:
        """Return this pool for ``with`` blocks."""
        return self

    def __exit__(self, *unused: object) -> None:
        """Close live battles when leaving a context manager."""
        del unused
        self.close()

    def pending(self) -> list[VectorGame]:
        """Return games waiting for a select response."""
        self._raise_if_closed()
        return [
            game
            for game in self._games.values()
            if result_index(game.observation) < 0 and game.steps < self._max_steps
        ]

    def submit(self, game_id: str, action: Sequence[int]) -> None:
        """Submit one action to a live game."""
        self._raise_if_closed()
        game = self._games[game_id]
        if result_index(game.observation) >= 0:
            raise RuntimeError(f"cannot submit to finished game: {game_id}")
        if game.steps >= self._max_steps:
            raise RuntimeError(f"cannot submit to max-step game: {game_id}")
        game.observation = select_pointer_battle(
            self._lib,
            game.battle_ptr,
            action,
            include_search_input=False,
            timer=self._timer,
        )
        game.steps += 1

    def finished(self) -> list[FinishedGame]:
        """Return terminal or max-step games and keep the pool filled."""
        self._raise_if_closed()
        output: list[FinishedGame] = []
        for game_id, game in list(self._games.items()):
            winner_index = result_index(game.observation)
            truncated = winner_index < 0 and game.steps >= self._max_steps
            if winner_index < 0 and not truncated:
                continue
            output.append(
                FinishedGame(
                    game_id=game.game_id,
                    battle_ptr=game.battle_ptr,
                    deck_pair=game.deck_pair,
                    observation=dict(game.observation),
                    winner_index=winner_index if winner_index >= 0 else -1,
                    steps=game.steps,
                )
            )
            finish_pointer_battle(self._lib, game.battle_ptr)
            del self._games[game_id]
            if self._next_plan < len(self._plans):
                self._start_next_game()
        return output

    def close(self) -> None:
        """Finish all live battle pointers."""
        if self._closed:
            return
        for game in list(self._games.values()):
            finish_pointer_battle(self._lib, game.battle_ptr)
        self._games.clear()
        self._closed = True

    def _start_next_game(self) -> None:
        plan = self._plans[self._next_plan]
        self._next_plan += 1
        game_id = f"g{plan.ladder_plan.arena_plan.game_index}"
        self.plans_by_game_id[game_id] = plan.ladder_plan
        self._games[game_id] = start_pointer_battle(
            self._lib,
            plan.deck_pair,
            game_id=game_id,
            include_search_input=False,
            timer=self._timer,
        )

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("vector deck ladder pool is closed")


@dataclass
class _DeckLadderVectorRecorder:
    """Collect and flush finished vector games without decision payloads."""

    rows: list[dict[str, Any]]
    plans_by_game_id: Mapping[str, DeckLadderGamePlan]
    config: VectorDeckLadderConfig
    flush_callback: Callable[[Sequence[Mapping[str, Any]], bool], dict[str, Any]]
    flush_interval_games: int
    _unflushed_games: int = 0

    def record(self, decision: object) -> None:
        """Ignore per-action trajectory data during evaluation."""
        del decision

    def finalize(self, finished: FinishedGame) -> None:
        """Append and periodically persist one terminal game."""
        plan = self.plans_by_game_id.get(finished.game_id)
        if plan is None:
            raise RuntimeError(f"finished unknown vector ladder game: {finished.game_id}")
        self.rows.append(_finished_row(finished, plan, config=self.config))
        self._unflushed_games += 1
        if self._unflushed_games >= self.flush_interval_games:
            self.flush(complete=False)

    def flush(self, *, complete: bool) -> dict[str, Any]:
        """Persist all collected rows."""
        self._unflushed_games = 0
        return self.flush_callback(self.rows, complete)


def run_vector_deck_ladder(config: VectorDeckLadderConfig) -> dict[str, Any]:
    """Run a fixed-policy ladder with vectorized engine stepping and batched policy."""
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
    games_path = output_dir / "games.parquet"
    matchups_path = output_dir / "matchups.parquet"
    standings_path = output_dir / "standings.parquet"
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    progress_path = output_dir / "progress.json"

    resumed_rows = (
        _load_resume_rows(games_path, plans=plans, config=config)
        if config.resume
        else []
    )
    remaining_plans = _remaining_plans(plans, resumed_rows)

    start_time = time.perf_counter()
    timer = StageTimer()
    rollout_summary: Any | None = None

    def flush_artifacts(
        rows: Sequence[Mapping[str, Any]],
        complete: bool,
    ) -> dict[str, Any]:
        return _write_ladder_artifacts(
            config,
            rows=rows,
            output_dir=output_dir,
            games_path=games_path,
            matchups_path=matchups_path,
            standings_path=standings_path,
            summary_path=summary_path,
            report_path=report_path,
            progress_path=progress_path,
            timer=timer,
            start_time=start_time,
            rollout_summary=rollout_summary,
            total_games=len(plans),
            resumed_games=len(resumed_rows),
            complete=complete,
        )

    if not remaining_plans:
        return flush_artifacts(resumed_rows, True)

    device = _resolve_device(config.device)
    timer = StageTimer(
        synchronize=(
            torch.cuda.synchronize
            if config.profile_sync_cuda and device.type == "cuda"
            else None
        )
    )
    policy = _load_rollout_policy(
        config.checkpoint_path,
        device=device,
        autocast=config.autocast,
        compile_model=config.compile_model,
    )

    recorder = _DeckLadderVectorRecorder(
        rows=list(resumed_rows),
        plans_by_game_id={},
        config=config,
        flush_callback=flush_artifacts,
        flush_interval_games=config.flush_interval_games,
    )
    with _FiniteVectorDeckLadderPool(
        remaining_plans,
        num_games=min(config.num_concurrent_games, len(remaining_plans)),
        max_steps=config.max_steps_per_game,
        timer=timer,
    ) as pool:
        recorder.plans_by_game_id = pool.plans_by_game_id
        stepper = RolloutStepper(
            pool=pool,
            actors=RolloutActors(mode="self_play", candidate_policy=policy),
            recorder=recorder,
            temperature=0.0,
            device=device,
            timer=timer,
            probe_config=RolloutProbeConfig(enabled=False),
            belief_config=_belief_config(config.belief_summary_path),
            seed=config.seed,
            record_policy_decisions=False,
        )
        rollout_summary = stepper.run_until(
            total_finished_games=len(remaining_plans),
            max_iterations=_max_iterations(config, len(remaining_plans)),
        )

    rows = recorder.rows
    if len(rows) != len(plans):
        recorder.flush(complete=False)
        raise RuntimeError(
            f"vector deck ladder completed {len(rows)} games, expected {len(plans)}"
        )
    return recorder.flush(complete=True)


def _pool_plan(plan: DeckLadderGamePlan) -> _PoolPlan:
    arena_plan = plan.arena_plan
    if arena_plan.candidate_seat == 0:
        deck_pair = (arena_plan.candidate_deck.cards, arena_plan.opponent_deck.cards)
    else:
        deck_pair = (arena_plan.opponent_deck.cards, arena_plan.candidate_deck.cards)
    return _PoolPlan(
        ladder_plan=plan,
        deck_pair=(
            tuple(int(card) for card in deck_pair[0]),
            tuple(int(card) for card in deck_pair[1]),
        ),
    )


def _load_resume_rows(
    games_path: Path,
    *,
    plans: Sequence[DeckLadderGamePlan],
    config: VectorDeckLadderConfig,
) -> list[dict[str, Any]]:
    """Load and validate completed rows for a resumable vector ladder run."""
    if not games_path.exists():
        return []
    plan_by_index = {int(plan.arena_plan.game_index): plan for plan in plans}
    rows = [dict(row) for row in pq.read_table(games_path).to_pylist()]
    seen: set[int] = set()
    for row in rows:
        game_index = int(row["game_index"])
        if game_index in seen:
            raise ValueError(f"duplicate resumed game_index in {games_path}: {game_index}")
        seen.add(game_index)
        plan = plan_by_index.get(game_index)
        if plan is None:
            raise ValueError(
                f"resumed game_index {game_index} is not in current ladder plan"
            )
        if not _resume_row_matches_plan(row, plan, config=config):
            raise ValueError(
                "existing games.parquet does not match current ladder config at "
                f"game_index={game_index}; use a new output_dir or remove the stale file"
            )
    rows.sort(key=lambda row: int(row["game_index"]))
    return rows


def _remaining_plans(
    plans: Sequence[DeckLadderGamePlan],
    completed_rows: Sequence[Mapping[str, Any]],
) -> tuple[DeckLadderGamePlan, ...]:
    """Return plans that do not already have a completed game row."""
    completed = {int(row["game_index"]) for row in completed_rows}
    return tuple(
        plan
        for plan in plans
        if int(plan.arena_plan.game_index) not in completed
    )


def _resume_row_matches_plan(
    row: Mapping[str, Any],
    plan: DeckLadderGamePlan,
    *,
    config: VectorDeckLadderConfig,
) -> bool:
    """Return whether one resumed row belongs to one current plan."""
    arena_plan = plan.arena_plan
    expected_checkpoint = records.display_path(records.repo_path(config.checkpoint_path))
    return (
        int(row["seed"]) == int(arena_plan.seed)
        and int(row["candidate_seat"]) == int(arena_plan.candidate_seat)
        and int(row["deck_ladder_repeat"]) == int(plan.repeat_index)
        and str(row["deck_a_id"]) == str(arena_plan.candidate_deck.deck_id)
        and str(row["deck_b_id"]) == str(arena_plan.opponent_deck.deck_id)
        and str(row["checkpoint_path"]) == expected_checkpoint
    )


def _write_ladder_artifacts(
    config: VectorDeckLadderConfig,
    *,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    games_path: Path,
    matchups_path: Path,
    standings_path: Path,
    summary_path: Path,
    report_path: Path,
    progress_path: Path,
    timer: StageTimer,
    start_time: float,
    rollout_summary: Any | None,
    total_games: int,
    resumed_games: int,
    complete: bool,
) -> dict[str, Any]:
    """Write resumable ladder artifacts and return the latest summary."""
    if not rows:
        progress = _progress_payload(
            total_games=total_games,
            completed_games=0,
            resumed_games=resumed_games,
            complete=False,
            elapsed_seconds=time.perf_counter() - start_time,
        )
        _write_json_atomic(progress_path, progress)
        return progress

    game_rows = [dict(row) for row in rows]
    game_rows.sort(key=lambda row: int(row["game_index"]))
    standings_rows = _apply_online_elo(game_rows, config=config)
    matchup_rows = deck_ladder_matchup_rows(game_rows)
    _write_parquet_atomic(games_path, game_rows, compression=config.compression)
    _write_parquet_atomic(matchups_path, matchup_rows, compression=config.compression)
    _write_parquet_atomic(standings_path, standings_rows, compression=config.compression)

    elapsed_seconds = time.perf_counter() - start_time
    new_games = max(0, len(game_rows) - resumed_games)
    summary = deck_ladder_summary(
        config,
        game_rows=game_rows,
        matchup_rows=matchup_rows,
        standings_rows=standings_rows,
        output_dir=output_dir,
        games_path=games_path,
        matchups_path=matchups_path,
        standings_path=standings_path,
    )
    summary.update(
        {
            "runner": "vector",
            "complete": bool(complete),
            "total_games": int(total_games),
            "completed_games": len(game_rows),
            "remaining_games": max(0, int(total_games) - len(game_rows)),
            "resumed_games": int(resumed_games),
            "new_games": int(new_games),
            "elapsed_seconds": elapsed_seconds,
            "games_per_second": new_games / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0,
            "num_concurrent_games": config.num_concurrent_games,
            "autocast": config.autocast,
            "compile_model": config.compile_model,
            "resume": config.resume,
            "flush_interval_games": config.flush_interval_games,
            "rollout": rollout_summary.__dict__ if rollout_summary is not None else None,
            "profile": timer.summary(),
            "profile_synchronizations": timer.synchronizations,
            "progress_path": records.display_path(progress_path),
        }
    )
    summary["summary_path"] = records.display_path(summary_path)
    summary["report_path"] = records.display_path(report_path)
    _write_json_atomic(summary_path, summary)
    _write_json_atomic(
        progress_path,
        _progress_payload(
            total_games=total_games,
            completed_games=len(game_rows),
            resumed_games=resumed_games,
            complete=complete,
            elapsed_seconds=elapsed_seconds,
        ),
    )
    report_path.write_text(render_deck_ladder_report(summary), encoding="utf-8")
    return summary


def _progress_payload(
    *,
    total_games: int,
    completed_games: int,
    resumed_games: int,
    complete: bool,
    elapsed_seconds: float,
) -> dict[str, Any]:
    """Return compact progress metadata for a resumable ladder run."""
    remaining_games = max(0, int(total_games) - int(completed_games))
    return {
        "complete": bool(complete),
        "total_games": int(total_games),
        "completed_games": int(completed_games),
        "remaining_games": remaining_games,
        "resumed_games": int(resumed_games),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _write_parquet_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression: str,
) -> None:
    """Write a Parquet table through a same-directory temporary path."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    _write_parquet(tmp_path, rows, compression=compression)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a same-directory temporary path."""
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _finished_row(
    finished: FinishedGame,
    plan: DeckLadderGamePlan,
    *,
    config: VectorDeckLadderConfig,
) -> dict[str, Any]:
    arena_plan = plan.arena_plan
    terminal_reason = "finished" if finished.winner_index >= 0 else "max_steps"
    deck_a_result = _candidate_result(
        finished.winner_index,
        arena_plan.candidate_seat,
        terminal_reason,
    )
    opponent_seat = 1 - arena_plan.candidate_seat
    return {
        "game_index": arena_plan.game_index,
        "seed": arena_plan.seed,
        "candidate_agent": f"deck:{arena_plan.candidate_deck.label}",
        "opponent_agent": f"deck:{arena_plan.opponent_deck.label}",
        "candidate_seat": arena_plan.candidate_seat,
        "winner_index": finished.winner_index,
        "candidate_result": deck_a_result,
        "terminal_reason": terminal_reason,
        "steps": finished.steps,
        "candidate_deck_id": arena_plan.candidate_deck.deck_id,
        "candidate_deck_hash": arena_plan.candidate_deck.deck_hash,
        "candidate_deck_label": arena_plan.candidate_deck.label,
        "candidate_deck_signature": arena_plan.candidate_deck.signature,
        "opponent_deck_id": arena_plan.opponent_deck.deck_id,
        "opponent_deck_hash": arena_plan.opponent_deck.deck_hash,
        "opponent_deck_label": arena_plan.opponent_deck.label,
        "opponent_deck_signature": arena_plan.opponent_deck.signature,
        "candidate_decisions": 0,
        "opponent_decisions": 0,
        "candidate_action_seconds": 0.0,
        "opponent_action_seconds": 0.0,
        "candidate_mean_action_seconds": 0.0,
        "opponent_mean_action_seconds": 0.0,
        "candidate_illegal_actions": 0,
        "opponent_illegal_actions": 0,
        "error_type": "",
        "error_message": "",
        "checkpoint_path": records.display_path(records.repo_path(config.checkpoint_path)),
        "deck_ladder_repeat": plan.repeat_index,
        "deck_a_seat": arena_plan.candidate_seat,
        "deck_b_seat": opponent_seat,
        "deck_a_id": arena_plan.candidate_deck.deck_id,
        "deck_a_hash": arena_plan.candidate_deck.deck_hash,
        "deck_a_label": arena_plan.candidate_deck.label,
        "deck_a_signature": arena_plan.candidate_deck.signature,
        "deck_a_source": arena_plan.candidate_deck.source,
        "deck_b_id": arena_plan.opponent_deck.deck_id,
        "deck_b_hash": arena_plan.opponent_deck.deck_hash,
        "deck_b_label": arena_plan.opponent_deck.label,
        "deck_b_signature": arena_plan.opponent_deck.signature,
        "deck_b_source": arena_plan.opponent_deck.source,
        "deck_a_result": deck_a_result,
        "deck_b_result": _inverse_result(deck_a_result),
        "deck_a_score": _score(deck_a_result),
        "deck_b_score": 1.0 - _score(deck_a_result),
    }


def _candidate_result(
    winner_index: int,
    candidate_seat: int,
    terminal_reason: str,
) -> Literal["win", "loss", "draw", "truncated"]:
    if terminal_reason != "finished" or winner_index < 0:
        return "truncated"
    if winner_index == 2:
        return "draw"
    return "win" if winner_index == candidate_seat else "loss"


def _belief_config(path: Path | None) -> RolloutBeliefConfig:
    if path is None:
        return RolloutBeliefConfig(enabled=False)
    return RolloutBeliefConfig(
        enabled=True,
        deck_signature_summary_path=records.repo_path(path),
    )


def _max_iterations(config: VectorDeckLadderConfig, games: int) -> int:
    batches = math.ceil(games / max(1, config.num_concurrent_games))
    return max(config.max_steps_per_game * (batches + 2), config.max_steps_per_game + 1)


def _resolve_device(raw_device: str) -> torch.device:
    normalized = raw_device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "gpu":
        normalized = "cuda"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {raw_device}")
    return device


def _load_rollout_policy(
    checkpoint_path: Path,
    *,
    device: torch.device,
    autocast: AutocastMode,
    compile_model: bool,
) -> ModelRolloutPolicy:
    checkpoint = torch.load(records.repo_path(checkpoint_path), map_location="cpu")
    model_config = _checkpoint_model_config(checkpoint) or AgentNetworkConfig()
    model = build_agent_policy_value_net(model_config).to(device)
    incompatible = model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    allowed_missing = {
        "opponent_hand_head.weight",
        "opponent_hand_head.bias",
    } | LEGACY_STATE_ENCODER_MISSING_KEYS
    if missing - allowed_missing or unexpected:
        raise RuntimeError("checkpoint state dict is incompatible with model")
    model.eval()
    if compile_model:
        model = cast(Any, torch.compile(model, mode="reduce-overhead"))
    return ModelRolloutPolicy(model, autocast=autocast)


def _checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig | None:
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        model_config = full_config.get("model")
        if isinstance(model_config, Mapping):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_lightning_model_prefix(cast(Mapping[str, Any], value))
        return _strip_lightning_model_prefix(cast(Mapping[str, Any], checkpoint))
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {str(key)[6:]: value for key, value in state_dict.items()}
    return state_dict
