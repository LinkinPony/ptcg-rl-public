"""Kaggle-like runtime deck ladder evaluation."""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.runtime import ActTimeConfig, PolicyRuntimeAgent, SelectPolicy
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.training.arena import (
    BattleSessionFactory,
    GamePlan,
    run_arena_game,
)
from ptcg_rl.training.arena_decks import ArenaDeck, DeckPoolConfig, load_deck_pool
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolve_training_output_dir,
    resolved_training_config_dump,
)

ScheduleKind = Literal["uniform", "recent_weighted"]


class _RuntimePolicyLike(SelectPolicy, Protocol):
    """Runtime checkpoint policy surface used by conservative overrides."""

    def rank_actions(
        self,
        observation: Any,
        *,
        top_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return high-priority complete actions for runtime verification."""


class RecentOpponentWeightsConfig(BaseModel):
    """Recent replay deck-prevalence weights for Kaggle-meta reporting."""

    model_config = ConfigDict(extra="forbid")

    path: Path | None = None
    weight_field: str = "games"
    default_weight: float = 0.0
    min_weight: float = 0.0

    @field_validator("weight_field")
    @classmethod
    def valid_field_name(cls, value: str) -> str:
        """Reject empty CSV/JSON field names."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("weight_field must be non-empty")
        return normalized

    @field_validator("default_weight", "min_weight")
    @classmethod
    def valid_non_negative(cls, value: float) -> float:
        """Reject negative weights."""
        if value < 0.0:
            raise ValueError("recent weights must be non-negative")
        return value


class RuntimeDeckLadderScheduleConfig(BaseModel):
    """Game scheduling knobs for runtime deck ladder evaluation."""

    model_config = ConfigDict(extra="forbid")

    uniform_games_per_pair: int = 2
    recent_weighted_extra_games: int = 0
    mirror_sides: bool = True
    recent_weights: RecentOpponentWeightsConfig = Field(
        default_factory=RecentOpponentWeightsConfig,
    )

    @field_validator("uniform_games_per_pair", "recent_weighted_extra_games")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative schedule counts."""
        if value < 0:
            raise ValueError("runtime deck ladder game counts must be non-negative")
        return value

    @model_validator(mode="after")
    def has_games(self) -> RuntimeDeckLadderScheduleConfig:
        """Require at least one requested game source."""
        if self.uniform_games_per_pair <= 0 and self.recent_weighted_extra_games <= 0:
            raise ValueError("runtime deck ladder schedule produced no requested games")
        return self


class RuntimeDeckLadderConcurrencyConfig(BaseModel):
    """Process-level concurrency configuration."""

    model_config = ConfigDict(extra="forbid")

    num_workers: int | Literal["auto"] = "auto"
    max_workers_cap: int = 12
    worker_memory_gb: float = 4.0
    gpu_workers_per_device: int = 2
    task_chunk_size: int | None = None

    @field_validator("num_workers")
    @classmethod
    def valid_num_workers(cls, value: int | str) -> int | str:
        """Reject invalid worker counts."""
        if isinstance(value, int) and value <= 0:
            raise ValueError("num_workers must be positive or 'auto'")
        if isinstance(value, str) and value != "auto":
            raise ValueError("num_workers must be positive or 'auto'")
        return value

    @field_validator("max_workers_cap", "gpu_workers_per_device")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive integer limits."""
        if value <= 0:
            raise ValueError("concurrency integer limits must be positive")
        return value

    @field_validator("worker_memory_gb")
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject invalid memory estimates."""
        if value <= 0.0:
            raise ValueError("worker_memory_gb must be positive")
        return value

    @field_validator("task_chunk_size")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject invalid chunk sizes."""
        if value is not None and value <= 0:
            raise ValueError("task_chunk_size must be positive when set")
        return value


class RuntimeDeckLadderConfig(BaseModel):
    """Hydra-backed config for Kaggle-like runtime deck ladder evaluation."""

    model_config = ConfigDict(extra="forbid")

    checkpoint_path: Path = Path("outputs/submission/agent_checkpoint.pt")
    device: str = "cuda"
    belief_summary_path: Path | None = None
    public_catalog_manifest_path: Path | None = None
    deck_pool: DeckPoolConfig = Field(
        default_factory=lambda: DeckPoolConfig(
            deck_paths=(Path("data/sample_submission/deck.csv"),),
        )
    )
    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    schedule: RuntimeDeckLadderScheduleConfig = Field(
        default_factory=RuntimeDeckLadderScheduleConfig,
    )
    concurrency: RuntimeDeckLadderConcurrencyConfig = Field(
        default_factory=RuntimeDeckLadderConcurrencyConfig,
    )
    act_time: ActTimeConfig = Field(default_factory=ActTimeConfig)
    max_steps_per_game: int = 300
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

    @field_validator("max_steps_per_game")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive runner limits."""
        if value <= 0:
            raise ValueError("runtime deck ladder limits must be positive")
        return value

    @field_validator("elo_k")
    @classmethod
    def valid_elo_k(cls, value: float) -> float:
        """Reject negative Elo K factors."""
        if value < 0.0:
            raise ValueError("elo_k must be non-negative")
        return value


@dataclass(frozen=True)
class RuntimeDeckLadderGamePlan:
    """One runtime game between two candidate decks."""

    arena_plan: GamePlan
    repeat_index: int
    schedule_kind: ScheduleKind
    pair_sampling_weight: float


@dataclass(frozen=True)
class _RuntimeDeckLadderTask:
    """Pickleable worker task payload."""

    plan: RuntimeDeckLadderGamePlan
    checkpoint_path: Path
    device: str
    max_steps: int
    act_time: ActTimeConfig
    belief_summary_path: Path | None = None
    public_catalog_manifest_path: Path | None = None


@dataclass(frozen=True)
class _RuntimeDeckLadderTaskChunk:
    """A small batch of games assigned to one process future."""

    tasks: tuple[_RuntimeDeckLadderTask, ...]


@dataclass(frozen=True)
class _RecentDeckWeights:
    """Loaded recent replay weights keyed by multiple deck identifiers."""

    path: Path
    weights_by_key: Mapping[str, float]
    rows: int
    total_weight: float


@dataclass
class _RuntimeAgentStats:
    """Per-game runtime telemetry collected by the local wrapper."""

    decisions: int = 0
    probe_seconds: float = 0.0
    probe_errors: int = 0
    belief_errors: int = 0
    policy_errors: int = 0
    verified_lethal_overrides: int = 0
    avoid_self_loss_overrides: int = 0
    used_random_fallback: bool = False
    prewarm_errors: int = 0
    engine_prewarm_errors: int = 0

    def row_fields(self, prefix: str) -> dict[str, Any]:
        """Return flat game-row fields for one side."""
        return {
            f"{prefix}_runtime_decisions": self.decisions,
            f"{prefix}_runtime_probe_seconds": self.probe_seconds,
            f"{prefix}_runtime_probe_errors": self.probe_errors,
            f"{prefix}_runtime_belief_errors": self.belief_errors,
            f"{prefix}_runtime_policy_errors": self.policy_errors,
            f"{prefix}_runtime_verified_lethal_overrides": (
                self.verified_lethal_overrides
            ),
            f"{prefix}_runtime_avoid_self_loss_overrides": (
                self.avoid_self_loss_overrides
            ),
            f"{prefix}_runtime_used_random_fallback": self.used_random_fallback,
            f"{prefix}_runtime_prewarm_errors": self.prewarm_errors,
            f"{prefix}_runtime_engine_prewarm_errors": self.engine_prewarm_errors,
        }


class _RuntimeDeckAgent:
    """Arena agent wrapper around the Kaggle act-time runtime."""

    def __init__(
        self,
        *,
        name: str,
        config: ActTimeConfig,
        policy: _RuntimePolicyLike,
    ) -> None:
        self.name = name
        self._base_config = config
        self._policy = policy
        self._agent: PolicyRuntimeAgent | None = None
        self.stats = _RuntimeAgentStats()
        self.reset()

    def reset(self) -> None:
        """Recreate per-game runtime state without reloading model weights."""
        self.stats = _RuntimeAgentStats()
        self._agent = PolicyRuntimeAgent(
            config=self._base_config,
            policy=self._policy,
        )

    def begin_game(
        self,
        *,
        player_index: int | None = None,
        own_deck: Sequence[int] | None = None,
    ) -> None:
        """Seed runtime context with the local BattleSession deck metadata."""
        if self._agent is None:
            self.reset()
        cast(Any, self._agent).begin_game(
            player_index=player_index,
            own_deck=own_deck,
        )

    def act(self, observation: Any) -> Sequence[int]:
        """Return the runtime agent action and collect act-time telemetry."""
        if self._agent is None:
            self.reset()
        agent = self._agent
        if agent is None:
            raise RuntimeError("runtime agent was not initialized")
        _clear_last_action_status(agent)
        action = agent.act(observation)
        self._observe_runtime_status(agent)
        return tuple(int(index) for index in action)

    def _observe_runtime_status(self, agent: PolicyRuntimeAgent) -> None:
        self.stats.decisions += 1
        self.stats.probe_seconds += max(0.0, float(agent.last_probe_seconds))
        if agent.last_probe_error is not None:
            self.stats.probe_errors += 1
        if agent.last_belief_error is not None:
            self.stats.belief_errors += 1
        if agent.last_policy_error is not None:
            self.stats.policy_errors += 1
        if agent.last_override_reason == "verified_lethal":
            self.stats.verified_lethal_overrides += 1
        elif agent.last_override_reason == "avoid_verified_self_loss":
            self.stats.avoid_self_loss_overrides += 1
        status = agent.runtime_status()
        self.stats.used_random_fallback = self.stats.used_random_fallback or bool(
            status.get("used_random_fallback"),
        )
        if status.get("prewarm_error") is not None:
            self.stats.prewarm_errors += 1
        if status.get("engine_prewarm_error") is not None:
            self.stats.engine_prewarm_errors += 1


_POLICY_CACHE: dict[tuple[str, str], _RuntimePolicyLike] = {}


def run_runtime_deck_ladder(
    config: RuntimeDeckLadderConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    """Run a Kaggle-like runtime deck ladder and write artifacts."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="runtime_deck_ladder",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recent_weights = load_recent_deck_weights(config.schedule.recent_weights)
    plans = expand_runtime_deck_ladder_plans(config, recent_weights=recent_weights)
    if not plans:
        raise ValueError("runtime deck ladder produced no game plans")

    tasks = tuple(
        _RuntimeDeckLadderTask(
            plan=plan,
            checkpoint_path=config.checkpoint_path,
            device=config.device,
            max_steps=config.max_steps_per_game,
            act_time=config.act_time,
            belief_summary_path=config.belief_summary_path,
            public_catalog_manifest_path=config.public_catalog_manifest_path,
        )
        for plan in plans
    )
    worker_count = resolve_runtime_worker_count(
        config.concurrency,
        device=config.device,
    )
    progress = _RuntimeDeckLadderProgress(
        output_dir / "progress.json",
        config=config,
        worker_count=worker_count,
    )
    rows = _run_tasks(
        config,
        tasks,
        worker_count=worker_count,
        battle_session_factory=battle_session_factory,
        progress=progress,
    )
    rows.sort(key=lambda row: int(row["game_index"]))

    uniform_rows = [row for row in rows if row["schedule_kind"] == "uniform"]
    standings_uniform = _apply_online_elo(uniform_rows or rows, config=config)
    matchup_rows = runtime_deck_ladder_matchup_rows(rows)
    standings_recent = recent_weighted_standings(
        rows,
        deck_weights=recent_weights,
        config=config,
    )

    games_path = output_dir / "games.parquet"
    matchups_path = output_dir / "matchups.parquet"
    standings_uniform_path = output_dir / "standings_uniform.parquet"
    standings_recent_path = output_dir / "standings_recent_weighted.parquet"
    _write_parquet(games_path, rows, compression=config.compression)
    _write_parquet(matchups_path, matchup_rows, compression=config.compression)
    _write_parquet(
        standings_uniform_path,
        standings_uniform,
        compression=config.compression,
    )
    if standings_recent:
        _write_parquet(
            standings_recent_path,
            standings_recent,
            compression=config.compression,
        )

    summary = runtime_deck_ladder_summary(
        config,
        game_rows=rows,
        matchup_rows=matchup_rows,
        standings_uniform=standings_uniform,
        standings_recent=standings_recent,
        output_dir=output_dir,
        games_path=games_path,
        matchups_path=matchups_path,
        standings_uniform_path=standings_uniform_path,
        standings_recent_path=standings_recent_path if standings_recent else None,
        recent_weights=recent_weights,
        worker_count=worker_count,
    )
    summary["progress_path"] = records.display_path(progress.path)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary["summary_path"] = records.display_path(summary_path)
    summary["report_path"] = records.display_path(report_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(
        render_runtime_deck_ladder_report(summary),
        encoding="utf-8",
    )
    if config.fail_on_agent_error and int(summary["agent_error_games"]) > 0:
        raise RuntimeError(
            f"runtime deck ladder had agent errors: {summary['agent_error_games']}"
        )
    return summary


def expand_runtime_deck_ladder_plans(
    config: RuntimeDeckLadderConfig,
    *,
    recent_weights: _RecentDeckWeights | None = None,
) -> tuple[RuntimeDeckLadderGamePlan, ...]:
    """Expand uniform and recent-weighted deck-pair schedules."""
    decks = load_deck_pool(config.deck_pool)
    if len(decks) < 2:
        raise ValueError("runtime deck ladder requires at least two decks")
    pairs = _deck_pairs(decks)
    plans: list[RuntimeDeckLadderGamePlan] = []
    pair_repeats: dict[tuple[str, str], int] = {}
    for deck_a, deck_b in pairs:
        pair_key = _pair_key(deck_a, deck_b)
        pair_weight = _pair_sampling_weight(deck_a, deck_b, recent_weights, config)
        for _ in range(config.schedule.uniform_games_per_pair):
            repeat_index = pair_repeats.get(pair_key, 0)
            pair_repeats[pair_key] = repeat_index + 1
            plans.append(
                _game_plan(
                    config,
                    deck_a=deck_a,
                    deck_b=deck_b,
                    repeat_index=repeat_index,
                    game_index=len(plans),
                    schedule_kind="uniform",
                    pair_sampling_weight=pair_weight,
                )
            )

    extra_games = config.schedule.recent_weighted_extra_games
    if extra_games > 0 and recent_weights is not None:
        rng = random.Random(config.seed + 9173)
        weighted_pairs = _sample_weighted_pairs(
            pairs,
            total_games=extra_games,
            recent_weights=recent_weights,
            config=config,
            rng=rng,
        )
        for deck_a, deck_b in weighted_pairs:
            pair_key = _pair_key(deck_a, deck_b)
            repeat_index = pair_repeats.get(pair_key, 0)
            pair_repeats[pair_key] = repeat_index + 1
            plans.append(
                _game_plan(
                    config,
                    deck_a=deck_a,
                    deck_b=deck_b,
                    repeat_index=repeat_index,
                    game_index=len(plans),
                    schedule_kind="recent_weighted",
                    pair_sampling_weight=_pair_sampling_weight(
                        deck_a,
                        deck_b,
                        recent_weights,
                        config,
                    ),
                )
            )
    return tuple(plans)


def load_recent_deck_weights(
    config: RecentOpponentWeightsConfig,
) -> _RecentDeckWeights | None:
    """Load optional recent replay deck weights from CSV or JSON."""
    if config.path is None:
        return None
    path = records.repo_path(config.path)
    if not path.exists():
        raise FileNotFoundError(f"recent weights not found: {path}")
    if path.suffix.lower() == ".json":
        rows = _json_weight_rows(path, config=config)
    else:
        rows = _csv_weight_rows(path, config=config)
    weights_by_key: dict[str, float] = {}
    total_weight = 0.0
    for keys, weight in rows:
        if weight < config.min_weight:
            continue
        total_weight += weight
        for key in keys:
            if key:
                weights_by_key[key] = max(weight, weights_by_key.get(key, 0.0))
    return _RecentDeckWeights(
        path=path,
        weights_by_key=weights_by_key,
        rows=len(rows),
        total_weight=total_weight,
    )


def runtime_deck_ladder_matchup_rows(
    game_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate runtime ladder rows by unordered deck pair."""
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
        deck_a_seats = _deck_seat_stats(rows, deck_prefix="deck_a")
        deck_b_seats = _deck_seat_stats(rows, deck_prefix="deck_b")
        output.append(
            {
                "deck_a_id": first["deck_a_id"],
                "deck_a_hash": first["deck_a_hash"],
                "deck_a_label": first["deck_a_label"],
                "deck_b_id": first["deck_b_id"],
                "deck_b_hash": first["deck_b_hash"],
                "deck_b_label": first["deck_b_label"],
                "games": games,
                "uniform_games": sum(
                    1 for row in rows if row["schedule_kind"] == "uniform"
                ),
                "recent_weighted_games": sum(
                    1 for row in rows if row["schedule_kind"] == "recent_weighted"
                ),
                "deck_a_wins": result_counts.get("win", 0),
                "deck_b_wins": result_counts.get("loss", 0),
                "draws": result_counts.get("draw", 0),
                "truncated": result_counts.get("truncated", 0),
                "deck_a_score_rate": deck_a_score_rate,
                "deck_b_score_rate": 1.0 - deck_a_score_rate,
                "agent_error_games": sum(
                    1 for row in rows if row["terminal_reason"] == "agent_error"
                ),
                **deck_a_seats,
                **deck_b_seats,
                "deck_a_mean_action_seconds": _safe_rate(
                    sum(float(row["candidate_action_seconds"]) for row in rows),
                    sum(int(row["candidate_decisions"]) for row in rows),
                ),
                "deck_b_mean_action_seconds": _safe_rate(
                    sum(float(row["opponent_action_seconds"]) for row in rows),
                    sum(int(row["opponent_decisions"]) for row in rows),
                ),
                "deck_a_runtime_probe_seconds": sum(
                    float(row["candidate_runtime_probe_seconds"]) for row in rows
                ),
                "deck_b_runtime_probe_seconds": sum(
                    float(row["opponent_runtime_probe_seconds"]) for row in rows
                ),
                "deck_a_runtime_overrides": sum(
                    int(row["candidate_runtime_verified_lethal_overrides"])
                    + int(row["candidate_runtime_avoid_self_loss_overrides"])
                    for row in rows
                ),
                "deck_b_runtime_overrides": sum(
                    int(row["opponent_runtime_verified_lethal_overrides"])
                    + int(row["opponent_runtime_avoid_self_loss_overrides"])
                    for row in rows
                ),
            }
        )
    output.sort(key=lambda row: (row["deck_a_label"], row["deck_b_label"]))
    return output


def _deck_seat_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    deck_prefix: str,
) -> dict[str, int | float]:
    """Return additive result counts for one deck in each player seat."""
    output: dict[str, int | float] = {}
    for seat in (0, 1):
        seat_rows = [row for row in rows if int(row[f"{deck_prefix}_seat"]) == seat]
        counts = Counter(str(row[f"{deck_prefix}_result"]) for row in seat_rows)
        resolved = counts.get("win", 0) + counts.get("draw", 0) + counts.get("loss", 0)
        field_prefix = f"{deck_prefix}_seat_{seat}"
        output.update(
            {
                f"{field_prefix}_games": len(seat_rows),
                f"{field_prefix}_wins": counts.get("win", 0),
                f"{field_prefix}_draws": counts.get("draw", 0),
                f"{field_prefix}_losses": counts.get("loss", 0),
                f"{field_prefix}_truncated": counts.get("truncated", 0),
                f"{field_prefix}_score_rate": _safe_rate(
                    counts.get("win", 0) + 0.5 * counts.get("draw", 0),
                    resolved,
                ),
            }
        )
    return output


def recent_weighted_standings(
    game_rows: Sequence[Mapping[str, Any]],
    *,
    deck_weights: _RecentDeckWeights | None,
    config: RuntimeDeckLadderConfig,
) -> list[dict[str, Any]]:
    """Build per-deck scores weighted by recent opponent prevalence."""
    if deck_weights is None:
        return []
    deck_stats = _deck_metadata(game_rows, config=config)
    pair_scores = _pair_score_rates(game_rows)
    output: list[dict[str, Any]] = []
    for deck_id, stats in deck_stats.items():
        weighted_score_sum = 0.0
        opponent_weight_sum = 0.0
        observed_games = 0
        for pair, pair_score in pair_scores.items():
            if deck_id not in pair:
                continue
            opponent_id = pair[1] if pair[0] == deck_id else pair[0]
            opponent_stats = deck_stats[opponent_id]
            opponent_weight = _deck_weight_from_stats(
                opponent_stats,
                deck_weights,
                config,
            )
            if opponent_weight <= 0.0:
                continue
            score = pair_score.deck_a_score
            if pair_score.deck_a_id != deck_id:
                score = 1.0 - score
            weighted_score_sum += opponent_weight * score
            opponent_weight_sum += opponent_weight
            observed_games += pair_score.games
        score_rate = _safe_rate(weighted_score_sum, opponent_weight_sum)
        output.append(
            {
                "deck_id": deck_id,
                "deck_hash": stats["deck_hash"],
                "deck_label": stats["deck_label"],
                "deck_signature": stats["deck_signature"],
                "deck_source": stats["deck_source"],
                "recent_weight": _deck_weight_from_stats(
                    stats,
                    deck_weights,
                    config,
                ),
                "weighted_opponent_weight": opponent_weight_sum,
                "weighted_score_rate": score_rate,
                "observed_games": observed_games,
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["weighted_score_rate"]),
            -float(row["weighted_opponent_weight"]),
            str(row["deck_label"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["rank"] = rank
    return output


def runtime_deck_ladder_summary(
    config: RuntimeDeckLadderConfig,
    *,
    game_rows: Sequence[Mapping[str, Any]],
    matchup_rows: Sequence[Mapping[str, Any]],
    standings_uniform: Sequence[Mapping[str, Any]],
    standings_recent: Sequence[Mapping[str, Any]],
    output_dir: Path,
    games_path: Path,
    matchups_path: Path,
    standings_uniform_path: Path,
    standings_recent_path: Path | None,
    recent_weights: _RecentDeckWeights | None,
    worker_count: int,
) -> dict[str, Any]:
    """Build the JSON summary for one runtime deck ladder run."""
    if not game_rows:
        raise ValueError("cannot summarize an empty runtime deck ladder run")
    return {
        "runner": "runtime_deck_ladder",
        "estimand": "same_checkpoint_pairwise_deck_diagnostic",
        "ranking_status": "diagnostic_only",
        "ranking_limitations": [
            "the same checkpoint pilots both sides",
            "online Elo depends on game order",
            "recent-weighted scores omit unrepresented meta tail",
        ],
        "games": len(game_rows),
        "uniform_games": sum(
            1 for row in game_rows if row["schedule_kind"] == "uniform"
        ),
        "recent_weighted_games": sum(
            1 for row in game_rows if row["schedule_kind"] == "recent_weighted"
        ),
        "matchups": len(matchup_rows),
        "decks": len(standings_uniform),
        "checkpoint_path": records.display_path(
            records.repo_path(config.checkpoint_path)
        ),
        "device": config.device,
        "belief_summary_path": (
            records.display_path(records.repo_path(config.belief_summary_path))
            if config.belief_summary_path is not None
            else None
        ),
        "elo_initial": config.elo_initial,
        "elo_k": config.elo_k,
        "num_workers": worker_count,
        "num_workers_config": config.concurrency.num_workers,
        "max_steps_per_game": config.max_steps_per_game,
        "agent_error_games": sum(
            1 for row in game_rows if row["terminal_reason"] == "agent_error"
        ),
        "truncated_games": sum(
            1 for row in game_rows if row["deck_a_result"] == "truncated"
        ),
        "candidate_runtime_probe_seconds": sum(
            float(row["candidate_runtime_probe_seconds"]) for row in game_rows
        ),
        "opponent_runtime_probe_seconds": sum(
            float(row["opponent_runtime_probe_seconds"]) for row in game_rows
        ),
        "runtime_verified_lethal_overrides": sum(
            int(row["candidate_runtime_verified_lethal_overrides"])
            + int(row["opponent_runtime_verified_lethal_overrides"])
            for row in game_rows
        ),
        "runtime_avoid_self_loss_overrides": sum(
            int(row["candidate_runtime_avoid_self_loss_overrides"])
            + int(row["opponent_runtime_avoid_self_loss_overrides"])
            for row in game_rows
        ),
        "recent_weights_loaded": recent_weights is not None,
        "recent_weights_path": (
            records.display_path(recent_weights.path)
            if recent_weights is not None
            else None
        ),
        "recent_weights_rows": recent_weights.rows if recent_weights is not None else 0,
        "recent_weights_total": (
            recent_weights.total_weight if recent_weights is not None else 0.0
        ),
        "run": config.run.model_dump(mode="json"),
        "output_dir": records.display_path(output_dir),
        "games_path": records.display_path(games_path),
        "matchups_path": records.display_path(matchups_path),
        "standings_uniform_path": records.display_path(standings_uniform_path),
        "standings_recent_weighted_path": (
            records.display_path(standings_recent_path)
            if standings_recent_path is not None
            else None
        ),
        "standings_uniform": [dict(row) for row in standings_uniform],
        "standings_recent_weighted": [dict(row) for row in standings_recent],
        "matchup_rows": [dict(row) for row in matchup_rows],
        "config": resolved_training_config_dump(
            config,
            task_name="runtime_deck_ladder",
            run=config.run,
            output_dir=config.output_dir,
        ),
    }


def render_runtime_deck_ladder_report(summary: Mapping[str, Any]) -> str:
    """Render a compact Markdown report from a runtime ladder summary."""
    run = _mapping(summary["run"])
    lines = [
        f"# Runtime Deck Ladder Report: {run['version']}",
        "",
        "> Diagnostic only: this measures deck interactions under one shared "
        "checkpoint. Use the exact-bundle posterior evaluator for selection.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Games | {summary['games']} |",
        f"| Uniform games | {summary['uniform_games']} |",
        f"| Recent-weighted extra games | {summary['recent_weighted_games']} |",
        f"| Decks | {summary['decks']} |",
        f"| Matchups | {summary['matchups']} |",
        f"| Workers | {summary['num_workers']} |",
        f"| Truncated games | {summary['truncated_games']} |",
        f"| Agent error games | {summary['agent_error_games']} |",
        f"| Verified lethal overrides | {summary['runtime_verified_lethal_overrides']} |",
        f"| Avoid self-loss overrides | {summary['runtime_avoid_self_loss_overrides']} |",
        "",
        "## Uniform Elo",
        "",
        "| Rank | Deck | Elo | Delta | Games | W-L-D-T | Score |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in _sequence(summary["standings_uniform"]):
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
    if summary.get("recent_weights_loaded"):
        lines.extend(
            [
                "",
                "## Recent-Weighted Score",
                "",
                "| Rank | Deck | Weighted Score | Opp Weight | Replay Weight | Games |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for row in _sequence(summary["standings_recent_weighted"]):
            data = _mapping(row)
            lines.append(
                "| {rank} | {deck} | {score} | {opp_weight:.1f} | "
                "{weight:.1f} | {games} |".format(
                    rank=data["rank"],
                    deck=data["deck_label"],
                    score=_pct(float(data["weighted_score_rate"])),
                    opp_weight=float(data["weighted_opponent_weight"]),
                    weight=float(data["recent_weight"]),
                    games=data["observed_games"],
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


def resolve_runtime_worker_count(
    config: RuntimeDeckLadderConcurrencyConfig,
    *,
    device: str = "cpu",
) -> int:
    """Resolve an explicit or auto process count."""
    if isinstance(config.num_workers, int):
        return config.num_workers
    cpu_count = os.cpu_count() or 1
    available_gb = _available_memory_gb()
    memory_workers = max(1, int(available_gb // config.worker_memory_gb))
    device_workers = config.max_workers_cap
    if device.startswith("cuda"):
        device_workers = max(1, _cuda_device_count() * config.gpu_workers_per_device)
    return max(
        1, min(cpu_count, config.max_workers_cap, memory_workers, device_workers)
    )


def _run_tasks(
    config: RuntimeDeckLadderConfig,
    tasks: Sequence[_RuntimeDeckLadderTask],
    *,
    worker_count: int,
    battle_session_factory: BattleSessionFactory | None,
    progress: _RuntimeDeckLadderProgress,
) -> list[dict[str, Any]]:
    progress.start(total_games=len(tasks))
    if worker_count <= 1 or battle_session_factory is not None:
        rows = []
        for task in tasks:
            row = _run_runtime_deck_ladder_task(
                task,
                battle_session_factory=battle_session_factory,
            )
            rows.append(row)
            progress.record(row)
        progress.finish()
        return rows

    chunks = _task_chunks(
        tasks,
        chunk_size=_task_chunk_size(
            config,
            task_count=len(tasks),
            worker_count=worker_count,
        ),
    )
    context = mp.get_context("spawn")
    rows = []
    with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
        futures = tuple(executor.submit(_run_task_chunk, chunk) for chunk in chunks)
        for future in as_completed(futures):
            chunk_rows = future.result()
            rows.extend(chunk_rows)
            for row in chunk_rows:
                progress.record(row)
    progress.finish()
    return rows


def _run_task_chunk(chunk: _RuntimeDeckLadderTaskChunk) -> list[dict[str, Any]]:
    return [_run_runtime_deck_ladder_task(task) for task in chunk.tasks]


def _run_runtime_deck_ladder_task(
    task: _RuntimeDeckLadderTask,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    plan = task.plan.arena_plan
    try:
        deck_a_agent = _RuntimeDeckAgent(
            name=f"runtime:{plan.candidate_deck.label}",
            config=_runtime_config(
                task,
                deck=plan.candidate_deck,
                seed=plan.seed + 101,
            ),
            policy=_deck_checkpoint_policy(
                task.checkpoint_path,
                device=task.device,
                own_deck=plan.candidate_deck.cards,
                public_catalog_manifest_path=(task.public_catalog_manifest_path),
            ),
        )
        deck_b_agent = _RuntimeDeckAgent(
            name=f"runtime:{plan.opponent_deck.label}",
            config=_runtime_config(
                task,
                deck=plan.opponent_deck,
                seed=plan.seed + 202,
            ),
            policy=_deck_checkpoint_policy(
                task.checkpoint_path,
                device=task.device,
                own_deck=plan.opponent_deck.cards,
                public_catalog_manifest_path=(task.public_catalog_manifest_path),
            ),
        )
        row = run_arena_game(
            plan,
            candidate_agent=deck_a_agent,
            opponent_agent=deck_b_agent,
            max_steps=task.max_steps,
            battle_session_factory=battle_session_factory or _default_battle_session,
            on_agent_error=_stop_on_agent_error,
            reset_agents=True,
        )
        output = dict(row)
        output.update(deck_a_agent.stats.row_fields("candidate"))
        output.update(deck_b_agent.stats.row_fields("opponent"))
    except Exception as exc:  # noqa: BLE001 - ladder must isolate game failures.
        output = _error_row(task, exc)
    _add_metadata_fields(output, task)
    return output


def _runtime_config(
    task: _RuntimeDeckLadderTask,
    *,
    deck: ArenaDeck,
    seed: int,
) -> ActTimeConfig:
    config = task.act_time.model_copy(deep=True)
    update: dict[str, Any] = {
        "checkpoint_path": records.repo_path(task.checkpoint_path),
        "seed": seed,
    }
    deck_path = _deck_path_if_readable(deck)
    if deck_path is not None:
        update["deck_path"] = deck_path
    config = config.model_copy(update=update)
    if task.belief_summary_path is not None:
        belief_path = records.repo_path(task.belief_summary_path)
        config = config.model_copy(deep=True)
        config.belief.deck_signature_summary_path = belief_path
        config.search.sampler.prior_deck_signature_summary_path = belief_path
    return config


def _deck_checkpoint_policy(
    checkpoint_path: Path,
    *,
    device: str,
    own_deck: Sequence[int],
    public_catalog_manifest_path: Path | None,
) -> _RuntimePolicyLike:
    resolved = records.repo_path(checkpoint_path)
    signature = records.deck_signature(list(own_deck))
    catalog_path = (
        None
        if public_catalog_manifest_path is None
        else records.repo_path(public_catalog_manifest_path)
    )
    key = (
        str(resolved),
        f"{device}:{signature}:{'' if catalog_path is None else str(catalog_path)}",
    )
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    policy = (
        _checkpoint_policy(resolved, device=device)
        if catalog_path is None
        else _routed_stateless_policy(
            resolved,
            public_catalog_manifest_path=catalog_path,
            device=device,
        )
    )
    bind_own_deck = getattr(policy, "bind_own_deck", None)
    if callable(bind_own_deck):
        bind_own_deck(own_deck)
    _POLICY_CACHE[key] = policy
    return policy


def _routed_stateless_policy(
    checkpoint_path: Path,
    *,
    public_catalog_manifest_path: Path,
    device: str,
) -> _RuntimePolicyLike:
    from ptcg_rl.agent.simple_stateless_runtime import routed_stateless_policy

    return cast(
        _RuntimePolicyLike,
        routed_stateless_policy(
            checkpoint_path,
            public_catalog_manifest_path=public_catalog_manifest_path,
            device=device,
        ),
    )


def _checkpoint_policy(
    checkpoint_path: Path,
    *,
    device: str,
) -> _RuntimePolicyLike:
    """Load one policy; exact-deck caching is handled by its caller."""
    from ptcg_rl.agent.runtime import CheckpointPolicy

    return cast(
        _RuntimePolicyLike,
        CheckpointPolicy(records.repo_path(checkpoint_path), device=device),
    )


def _stop_on_agent_error(
    exc: Exception,
    player_index: int,
    agent: object,
    observation: Mapping[str, Any],
) -> Sequence[int] | None:
    del exc, player_index, agent, observation
    return None


def _add_metadata_fields(row: dict[str, Any], task: _RuntimeDeckLadderTask) -> None:
    plan = task.plan.arena_plan
    deck_a_result = str(row.get("candidate_result", "truncated"))
    row.update(
        {
            "checkpoint_path": records.display_path(
                records.repo_path(task.checkpoint_path)
            ),
            "schedule_kind": task.plan.schedule_kind,
            "pair_sampling_weight": task.plan.pair_sampling_weight,
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
            "deck_a_runtime_probe_seconds": float(
                row.get("candidate_runtime_probe_seconds", 0.0)
            ),
            "deck_b_runtime_probe_seconds": float(
                row.get("opponent_runtime_probe_seconds", 0.0)
            ),
        }
    )


def _error_row(task: _RuntimeDeckLadderTask, exc: Exception) -> dict[str, Any]:
    plan = task.plan.arena_plan
    row: dict[str, Any] = {
        "game_index": plan.game_index,
        "seed": plan.seed,
        "candidate_agent": f"runtime:{plan.candidate_deck.label}",
        "opponent_agent": f"runtime:{plan.opponent_deck.label}",
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
        "error_player_index": -1,
        "error_actor": "infrastructure",
    }
    row.update(_RuntimeAgentStats().row_fields("candidate"))
    row.update(_RuntimeAgentStats().row_fields("opponent"))
    return row


def _apply_online_elo(
    rows: Sequence[dict[str, Any]],
    *,
    config: RuntimeDeckLadderConfig,
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
    config: RuntimeDeckLadderConfig,
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


@dataclass(frozen=True)
class _PairScore:
    deck_a_id: str
    deck_b_id: str
    deck_a_score: float
    games: int


def _pair_score_rates(
    game_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], _PairScore]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in game_rows:
        key = (str(row["deck_a_id"]), str(row["deck_b_id"]))
        grouped.setdefault(key, []).append(row)
    return {
        key: _PairScore(
            deck_a_id=key[0],
            deck_b_id=key[1],
            deck_a_score=sum(_score(row["deck_a_result"]) for row in rows)
            / float(len(rows)),
            games=len(rows),
        )
        for key, rows in grouped.items()
    }


def _deck_metadata(
    game_rows: Sequence[Mapping[str, Any]],
    *,
    config: RuntimeDeckLadderConfig,
) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for row in game_rows:
        _ensure_deck_stats(stats, row, prefix="deck_a", config=config)
        _ensure_deck_stats(stats, row, prefix="deck_b", config=config)
    return stats


def _deck_pairs(decks: Sequence[ArenaDeck]) -> tuple[tuple[ArenaDeck, ArenaDeck], ...]:
    pairs: list[tuple[ArenaDeck, ArenaDeck]] = []
    for left_index, deck_a in enumerate(decks):
        for deck_b in decks[left_index + 1 :]:
            pairs.append((deck_a, deck_b))
    return tuple(pairs)


def _game_plan(
    config: RuntimeDeckLadderConfig,
    *,
    deck_a: ArenaDeck,
    deck_b: ArenaDeck,
    repeat_index: int,
    game_index: int,
    schedule_kind: ScheduleKind,
    pair_sampling_weight: float,
) -> RuntimeDeckLadderGamePlan:
    candidate_seat = repeat_index % 2 if config.schedule.mirror_sides else 0
    arena_plan = GamePlan(
        game_index=game_index,
        seed=config.seed + game_index,
        candidate_deck=deck_a,
        opponent_deck=deck_b,
        candidate_seat=candidate_seat,
    )
    return RuntimeDeckLadderGamePlan(
        arena_plan=arena_plan,
        repeat_index=repeat_index,
        schedule_kind=schedule_kind,
        pair_sampling_weight=pair_sampling_weight,
    )


def _sample_weighted_pairs(
    pairs: Sequence[tuple[ArenaDeck, ArenaDeck]],
    *,
    total_games: int,
    recent_weights: _RecentDeckWeights,
    config: RuntimeDeckLadderConfig,
    rng: random.Random,
) -> tuple[tuple[ArenaDeck, ArenaDeck], ...]:
    weights = [
        _pair_sampling_weight(deck_a, deck_b, recent_weights, config)
        for deck_a, deck_b in pairs
    ]
    if sum(weights) <= 0.0:
        return ()
    return tuple(rng.choices(list(pairs), weights=weights, k=total_games))


def _pair_sampling_weight(
    deck_a: ArenaDeck,
    deck_b: ArenaDeck,
    weights: _RecentDeckWeights | None,
    config: RuntimeDeckLadderConfig,
) -> float:
    if weights is None:
        return 1.0
    return max(
        _deck_weight(deck_a, weights, config),
        _deck_weight(deck_b, weights, config),
    )


def _deck_weight(
    deck: ArenaDeck,
    weights: _RecentDeckWeights,
    config: RuntimeDeckLadderConfig,
) -> float:
    for key in _deck_keys(
        deck_id=deck.deck_id,
        deck_hash=deck.deck_hash,
        deck_label=deck.label,
        deck_signature=deck.signature,
        deck_source=deck.source,
    ):
        if key in weights.weights_by_key:
            return weights.weights_by_key[key]
    return config.schedule.recent_weights.default_weight


def _deck_weight_from_stats(
    stats: Mapping[str, Any],
    weights: _RecentDeckWeights,
    config: RuntimeDeckLadderConfig,
) -> float:
    for key in _deck_keys(
        deck_id=str(stats["deck_id"]) if "deck_id" in stats else "",
        deck_hash=str(stats["deck_hash"]),
        deck_label=str(stats["deck_label"]),
        deck_signature=str(stats["deck_signature"]),
        deck_source=str(stats["deck_source"]),
    ):
        if key in weights.weights_by_key:
            return weights.weights_by_key[key]
    return config.schedule.recent_weights.default_weight


def _deck_keys(
    *,
    deck_id: str,
    deck_hash: str,
    deck_label: str,
    deck_signature: str,
    deck_source: str,
) -> tuple[str, ...]:
    keys = (
        deck_id,
        deck_hash,
        deck_label,
        deck_signature,
        deck_source,
        Path(deck_source).name,
        Path(deck_source).stem,
    )
    return tuple(key for key in keys if key)


def _pair_key(deck_a: ArenaDeck, deck_b: ArenaDeck) -> tuple[str, str]:
    return (deck_a.deck_id, deck_b.deck_id)


def _csv_weight_rows(
    path: Path,
    *,
    config: RecentOpponentWeightsConfig,
) -> list[tuple[tuple[str, ...], float]]:
    rows: list[tuple[tuple[str, ...], float]] = []
    with path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            weight = _float_text(row.get(config.weight_field))
            keys = tuple(
                value
                for value in (
                    row.get("deck_signature"),
                    row.get("deck_hash"),
                    row.get("deck_label"),
                    row.get("known_deck"),
                    row.get("deck_id"),
                    row.get("name"),
                )
                if value
            )
            if keys:
                rows.append((keys, weight))
    return rows


def _json_weight_rows(
    path: Path,
    *,
    config: RecentOpponentWeightsConfig,
) -> list[tuple[tuple[str, ...], float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        mapping_rows: list[tuple[tuple[str, ...], float]] = []
        for key, value in payload.items():
            if isinstance(value, Mapping):
                weight = _float_text(value.get(config.weight_field))
            else:
                weight = _float_text(value)
            mapping_rows.append(((str(key),), weight))
        return mapping_rows
    if isinstance(payload, Sequence) and not isinstance(payload, str):
        sequence_rows: list[tuple[tuple[str, ...], float]] = []
        for item in payload:
            if not isinstance(item, Mapping):
                continue
            weight = _float_text(item.get(config.weight_field))
            keys = tuple(
                str(value)
                for value in (
                    item.get("deck_signature"),
                    item.get("deck_hash"),
                    item.get("deck_label"),
                    item.get("known_deck"),
                    item.get("deck_id"),
                    item.get("name"),
                )
                if value
            )
            if keys:
                sequence_rows.append((keys, weight))
        return sequence_rows
    raise ValueError(f"unsupported recent weights JSON shape: {path}")


class _RuntimeDeckLadderProgress:
    """Small JSON progress writer for long runtime deck ladders."""

    def __init__(
        self,
        path: Path,
        *,
        config: RuntimeDeckLadderConfig,
        worker_count: int,
    ) -> None:
        self.path = path
        self._config = config
        self._worker_count = worker_count
        self._start_time = 0.0
        self._total_games = 0
        self._completed_games = 0
        self._agent_error_games = 0
        self._last_game_index: int | None = None
        self._last_write_completed = -1

    def start(self, *, total_games: int) -> None:
        """Record the start of a runtime ladder run."""
        self._start_time = time.perf_counter()
        self._total_games = int(total_games)
        self._completed_games = 0
        self._agent_error_games = 0
        self._last_game_index = None
        self._write(status="running", force=True)

    def record(self, row: Mapping[str, Any]) -> None:
        """Record one completed game."""
        self._completed_games += 1
        if str(row.get("terminal_reason", "")) == "agent_error":
            self._agent_error_games += 1
        raw_game_index = row.get("game_index")
        if raw_game_index is not None:
            self._last_game_index = int(raw_game_index)
        self._write(status="running", force=False)

    def finish(self) -> None:
        """Record completion of all games."""
        self._write(status="completed", force=True)

    def _write(self, *, status: str, force: bool) -> None:
        if (
            not force
            and self._completed_games != self._total_games
            and self._completed_games - self._last_write_completed < 25
        ):
            return
        self._last_write_completed = self._completed_games
        elapsed = max(0.0, time.perf_counter() - self._start_time)
        remaining_games = max(0, self._total_games - self._completed_games)
        payload = {
            "status": status,
            "complete": status == "completed",
            "total_games": self._total_games,
            "completed_games": self._completed_games,
            "remaining_games": remaining_games,
            "agent_error_games": self._agent_error_games,
            "last_game_index": self._last_game_index,
            "elapsed_seconds": elapsed,
            "games_per_second": (
                self._completed_games / elapsed if elapsed > 0.0 else 0.0
            ),
            "num_workers": self._worker_count,
            "device": self._config.device,
            "updated_at_unix": time.time(),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_atomic(self.path, payload)


def _task_chunks(
    tasks: Sequence[_RuntimeDeckLadderTask],
    *,
    chunk_size: int,
) -> tuple[_RuntimeDeckLadderTaskChunk, ...]:
    return tuple(
        _RuntimeDeckLadderTaskChunk(tasks=tuple(tasks[index : index + chunk_size]))
        for index in range(0, len(tasks), chunk_size)
    )


def _task_chunk_size(
    config: RuntimeDeckLadderConfig,
    *,
    task_count: int,
    worker_count: int,
) -> int:
    if config.concurrency.task_chunk_size is not None:
        return config.concurrency.task_chunk_size
    if task_count <= worker_count:
        return 1
    return max(1, min(8, task_count // (worker_count * 4)))


def _available_memory_gb() -> float:
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                parts = line.split()
                if len(parts) >= 2:
                    return float(parts[1]) / (1024.0 * 1024.0)
    return float(os.cpu_count() or 1) * 4.0


def _cuda_device_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if not devices or devices == ["-1"]:
            return 0
        return len(devices)
    try:
        result = subprocess.run(
            ("nvidia-smi", "-L"),
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return 1
    if result.returncode != 0:
        return 1
    return max(1, sum(1 for line in result.stdout.splitlines() if line.strip()))


def _clear_last_action_status(agent: PolicyRuntimeAgent) -> None:
    agent.last_probe_seconds = 0.0
    agent.last_probe_error = None
    agent.last_override_reason = None
    agent.last_policy_error = None


def _deck_path_if_readable(deck: ArenaDeck) -> Path | None:
    source = records.repo_path(Path(deck.source))
    if source.exists() and source.is_file():
        return source
    return None


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


def _elo_expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))


def _safe_rate(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / float(denominator)


def _float_text(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


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


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small JSON file atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
