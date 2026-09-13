"""Tiered opponent-pool gauntlet runner for checkpoint evaluation."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.context import OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.opponents.builtin import RandomOpponent
from ptcg_rl.opponents.spec import (
    BattleAgent,
    OpponentPoolConfig,
    OpponentSpec,
    build_opponent,
    select_opponents,
)
from ptcg_rl.training.arena import (
    BattleSessionFactory,
    GamePlan,
    run_arena_game,
)
from ptcg_rl.training.arena_agents import ArenaAgent, PolicyGreedyAgent
from ptcg_rl.training.arena_decks import ArenaDeck, DeckPoolConfig, load_deck_pool
from ptcg_rl.training.gauntlet_reports import (
    gauntlet_matchup_rows,
    gauntlet_summary,
    gauntlet_tier_rows,
    render_gauntlet_report,
)
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolve_training_output_dir,
)

CandidateMode = Literal["random", "policy_greedy", "runtime"]
OpponentDeckMode = Literal["mirror", "pool"]

_DECK_PATH_ENV = "POKEMON_TCG_DECK_PATH"
_CANDIDATE_CACHE: dict[
    tuple[CandidateMode, str, str, str, str, str, int],
    ArenaAgent,
] = {}
_OPPONENT_CACHE: dict[tuple[str, str, str, str, int], BattleAgent] = {}
_CONFIGURED_CUDA_MEMORY_LIMIT_GB: float | None = None


class ResettableAgent(Protocol):
    """Optional per-game reset hook used by stateful local agents."""

    def reset(self) -> None:
        """Reset per-game state."""


class GauntletCandidateConfig(BaseModel):
    """Candidate agent configuration for a gauntlet run."""

    model_config = ConfigDict(extra="forbid")

    mode: CandidateMode = "policy_greedy"
    label: str | None = None
    checkpoint_path: Path | None = Path(
        "outputs/training/bc/full_h200_100epoch/checkpoint_best.pt"
    )
    device: str = "cpu"
    overage_seconds: float = 600.0
    belief_summary_path: Path | None = None

    @field_validator("overage_seconds")
    @classmethod
    def valid_overage_seconds(cls, value: float) -> float:
        """Reject negative overage budgets."""
        if value < 0.0:
            raise ValueError("overage_seconds must be non-negative")
        return value

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("device must be non-empty")
        return normalized


class GauntletOpponentSuiteConfig(BaseModel):
    """Opponent deck and game-count controls."""

    model_config = ConfigDict(extra="forbid")

    opponent_deck_mode: OpponentDeckMode = "mirror"
    games_per_matchup: int = 100
    games_per_matchup_overrides: dict[str, int] = Field(default_factory=dict)

    @field_validator("games_per_matchup")
    @classmethod
    def valid_games_per_matchup(cls, value: int) -> int:
        """Reject non-positive matchup counts."""
        if value <= 0:
            raise ValueError("games_per_matchup must be positive")
        return value

    @field_validator("games_per_matchup_overrides")
    @classmethod
    def valid_overrides(cls, value: dict[str, int]) -> dict[str, int]:
        """Reject non-positive per-opponent matchup counts."""
        invalid = {name: count for name, count in value.items() if count <= 0}
        if invalid:
            raise ValueError(f"games_per_matchup_overrides must be positive: {invalid}")
        return value


class GauntletConfig(BaseModel):
    """Hydra-backed config for tiered opponent-pool evaluation."""

    model_config = ConfigDict(extra="forbid")

    candidate: GauntletCandidateConfig = Field(default_factory=GauntletCandidateConfig)
    candidate_decks: DeckPoolConfig = Field(
        default_factory=lambda: DeckPoolConfig(
            deck_paths=(Path("data/sample_submission/deck.csv"),),
        )
    )
    opponent_pool: OpponentPoolConfig = Field(default_factory=OpponentPoolConfig)
    opponent_deck_pool: DeckPoolConfig = Field(
        default_factory=lambda: DeckPoolConfig(
            deck_paths=(Path("data/sample_submission/deck.csv"),),
        )
    )
    opponent_suite: GauntletOpponentSuiteConfig = Field(
        default_factory=GauntletOpponentSuiteConfig
    )
    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    num_workers: int = 1
    max_steps_per_game: int = 10_000
    mirror_sides: bool = True
    seed: int = 0
    compression: str = "zstd"
    task_chunk_size: int | None = None
    cuda_memory_limit_gb: float | None = None
    fail_on_agent_error: bool = False

    @field_validator("num_workers", "max_steps_per_game")
    @classmethod
    def valid_positive(cls, value: int) -> int:
        """Reject non-positive runner limits."""
        if value <= 0:
            raise ValueError("gauntlet runner limits must be positive")
        return value

    @field_validator("task_chunk_size")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject invalid optional runner chunk sizes."""
        if value is not None and value <= 0:
            raise ValueError("gauntlet task_chunk_size must be positive when set")
        return value

    @field_validator("cuda_memory_limit_gb")
    @classmethod
    def valid_optional_positive_float(cls, value: float | None) -> float | None:
        """Reject invalid optional CUDA memory budgets."""
        if value is not None and value <= 0.0:
            raise ValueError("cuda_memory_limit_gb must be positive when set")
        return value


@dataclass(frozen=True)
class GauntletGamePlan:
    """One candidate-vs-opponent game with opponent-pool metadata."""

    arena_plan: GamePlan
    opponent_spec: OpponentSpec
    repeat_index: int


@dataclass(frozen=True)
class _GauntletTask:
    """Pickleable worker task payload."""

    plan: GauntletGamePlan
    candidate: GauntletCandidateConfig
    max_steps: int
    cuda_memory_limit_gb: float | None = None


class _RuntimeCandidateAgent:
    """PolicyRuntimeAgent wrapper with explicit per-game reset."""

    def __init__(
        self,
        *,
        name: str,
        deck_path: Path | None,
        checkpoint_path: Path | None,
        seed: int,
        belief_summary_path: Path | None = None,
    ) -> None:
        self.name = name
        self._deck_path = deck_path
        self._checkpoint_path = checkpoint_path
        self._seed = seed
        self._belief_summary_path = belief_summary_path
        self._agent: Any | None = None
        self.reset()

    def reset(self) -> None:
        """Recreate runtime state before each local battle."""
        from ptcg_rl.agent.runtime import ActTimeConfig, PolicyRuntimeAgent

        config = ActTimeConfig()
        if self._belief_summary_path is not None:
            config = config.model_copy(deep=True)
            config.belief.deck_signature_summary_path = self._belief_summary_path
            config.search.sampler.prior_deck_signature_summary_path = (
                self._belief_summary_path
            )
        self._agent = PolicyRuntimeAgent(
            config=config,
            deck_path=self._deck_path,
            checkpoint_path=self._checkpoint_path,
            seed=self._seed,
        )

    def act(self, observation: Any) -> Sequence[int]:
        """Return the runtime agent action."""
        if self._agent is None:
            self.reset()
        action = cast(Any, self._agent).act(observation)
        return tuple(int(index) for index in action)


def run_gauntlet(
    config: GauntletConfig,
    *,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    """Run a tiered opponent-pool gauntlet and write artifacts."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="arena",
            run=config.run,
            output_dir=config.output_dir,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = expand_gauntlet_plans(config)
    if not plans:
        raise ValueError("gauntlet produced no game plans")

    tasks = tuple(
        _GauntletTask(
            plan=plan,
            candidate=config.candidate,
            max_steps=config.max_steps_per_game,
            cuda_memory_limit_gb=_task_cuda_memory_limit_gb(config),
        )
        for plan in plans
    )
    progress = _GauntletProgress(output_dir / "progress.json", config=config)
    rows = _run_tasks(
        config,
        tasks,
        battle_session_factory=battle_session_factory,
        progress=progress,
    )
    rows.sort(key=lambda row: int(row["game_index"]))

    matchup_rows = gauntlet_matchup_rows(rows)
    tier_rows = gauntlet_tier_rows(rows)
    games_path = output_dir / "games.parquet"
    matchups_path = output_dir / "matchups.parquet"
    _write_parquet(games_path, rows, compression=config.compression)
    _write_parquet(matchups_path, matchup_rows, compression=config.compression)

    summary = gauntlet_summary(
        config,
        game_rows=rows,
        matchup_rows=matchup_rows,
        tier_rows=tier_rows,
        output_dir=output_dir,
        games_path=games_path,
        matchups_path=matchups_path,
    )
    summary["progress_path"] = records.display_path(progress.path)
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path.write_text(render_gauntlet_report(summary), encoding="utf-8")
    if config.fail_on_agent_error and int(summary["agent_error_games"]) > 0:
        raise RuntimeError(f"gauntlet had agent errors: {summary['agent_error_games']}")
    return summary


def expand_gauntlet_plans(config: GauntletConfig) -> tuple[GauntletGamePlan, ...]:
    """Expand candidate decks, opponent specs, decks, seats, and repeats."""
    candidate_decks = load_deck_pool(config.candidate_decks)
    if not candidate_decks:
        raise ValueError("candidate_decks must not be empty")
    opponent_specs = select_opponents(config.opponent_pool)
    if not opponent_specs:
        raise ValueError("opponent_pool selected no opponents")
    pooled_opponent_decks = _load_opponent_deck_pool(config)

    plans: list[GauntletGamePlan] = []
    for candidate_deck in candidate_decks:
        for spec in opponent_specs:
            opponent_decks = _opponent_decks_for(
                spec,
                candidate_deck=candidate_deck,
                pooled_opponent_decks=pooled_opponent_decks,
                deck_mode=config.opponent_suite.opponent_deck_mode,
            )
            games = config.opponent_suite.games_per_matchup_overrides.get(
                spec.name,
                config.opponent_suite.games_per_matchup,
            )
            for opponent_deck in opponent_decks:
                for repeat_index, candidate_seat in enumerate(
                    _candidate_seats(games, mirror_sides=config.mirror_sides)
                ):
                    arena_plan = GamePlan(
                        game_index=len(plans),
                        seed=config.seed + len(plans),
                        candidate_deck=candidate_deck,
                        opponent_deck=opponent_deck,
                        candidate_seat=candidate_seat,
                    )
                    plans.append(
                        GauntletGamePlan(
                            arena_plan=arena_plan,
                            opponent_spec=spec,
                            repeat_index=repeat_index,
                        )
                    )
    return tuple(plans)


class _GauntletProgress:
    """Small JSON progress writer for long gauntlet evaluations."""

    def __init__(self, path: Path, *, config: GauntletConfig) -> None:
        self.path = path
        self._config = config
        self._start_time = 0.0
        self._total_games = 0
        self._completed_games = 0
        self._agent_error_games = 0
        self._last_game_index: int | None = None
        self._last_write_completed = -1

    def start(self, *, total_games: int) -> None:
        """Record the start of a gauntlet run."""
        self._start_time = time.perf_counter()
        self._total_games = int(total_games)
        self._completed_games = 0
        self._agent_error_games = 0
        self._last_game_index = None
        self._write(status="running", force=True)

    def record(self, row: Mapping[str, Any]) -> None:
        """Record one completed gauntlet game."""
        self._completed_games += 1
        if str(row.get("terminal_reason", "")) == "agent_error":
            self._agent_error_games += 1
        raw_game_index = row.get("game_index")
        if raw_game_index is not None:
            self._last_game_index = int(raw_game_index)
        self._write(status="running", force=False)

    def finish(self) -> None:
        """Record completion of all gauntlet games."""
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
            "num_workers": self._config.num_workers,
            "candidate_mode": self._config.candidate.mode,
            "candidate_device": self._config.candidate.device,
            "updated_at_unix": time.time(),
            "updated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        _write_json_atomic(self.path, payload)


def _run_tasks(
    config: GauntletConfig,
    tasks: Sequence[_GauntletTask],
    *,
    battle_session_factory: BattleSessionFactory | None,
    progress: _GauntletProgress,
) -> list[dict[str, Any]]:
    progress.start(total_games=len(tasks))
    if config.num_workers <= 1 or battle_session_factory is not None:
        rows: list[dict[str, Any]] = []
        for task in tasks:
            row = _run_gauntlet_task(
                task,
                battle_session_factory=battle_session_factory,
            )
            rows.append(row)
            progress.record(row)
        progress.finish()
        return rows

    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=config.num_workers,
        mp_context=context,
    ) as executor:
        futures = tuple(executor.submit(_run_gauntlet_task, task) for task in tasks)
        rows = []
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            progress.record(row)
    progress.finish()
    return rows


def _run_gauntlet_task(
    task: _GauntletTask,
    battle_session_factory: BattleSessionFactory | None = None,
) -> dict[str, Any]:
    _configure_cuda_memory_limit(task.cuda_memory_limit_gb)
    plan = task.plan.arena_plan
    try:
        candidate_agent = _candidate_agent(task.candidate, plan.candidate_deck, plan.seed)
        opponent_agent = _opponent_agent(task.plan.opponent_spec, plan.seed)
        with _deck_path_env(plan.opponent_deck.source):
            row = run_arena_game(
                plan,
                candidate_agent=candidate_agent,
                opponent_agent=opponent_agent,
                max_steps=task.max_steps,
                battle_session_factory=battle_session_factory or _default_battle_session,
                on_agent_error=_stop_on_agent_error,
                reset_agents=True,
            )
        output = dict(row)
    except Exception as exc:  # noqa: BLE001 - gauntlet must isolate game failures.
        output = _error_row(task, exc)
    _add_metadata_fields(output, task)
    return output


def _stop_on_agent_error(
    exc: Exception,
    player_index: int,
    agent: object,
    observation: Mapping[str, Any],
) -> Sequence[int] | None:
    del exc, player_index, agent, observation
    return None


def _candidate_agent(
    config: GauntletCandidateConfig,
    candidate_deck: ArenaDeck,
    seed: int,
) -> ArenaAgent:
    deck_path = _existing_source_path(candidate_deck)
    key = _candidate_cache_key(config, deck_path=deck_path, seed=seed)
    cached = _CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached

    label = config.label or f"candidate_{config.mode}"
    belief_summary_path = (
        records.repo_path(config.belief_summary_path)
        if config.belief_summary_path is not None
        else None
    )
    if config.mode == "random":
        agent: ArenaAgent = RandomOpponent(name=label, rng=random.Random(seed))
    elif config.mode == "policy_greedy":
        if config.checkpoint_path is None:
            raise ValueError("policy_greedy candidate requires checkpoint_path")
        belief = (
            OpponentBeliefFeatureConfig(
                deck_signature_summary_path=belief_summary_path,
            )
            if belief_summary_path is not None
            else None
        )
        agent = PolicyGreedyAgent(
            name=label,
            checkpoint_path=records.repo_path(config.checkpoint_path),
            device=config.device,
            belief=belief,
        )
    elif config.mode == "runtime":
        agent = _RuntimeCandidateAgent(
            name=label,
            deck_path=deck_path,
            checkpoint_path=(
                records.repo_path(config.checkpoint_path)
                if config.checkpoint_path is not None
                else None
            ),
            seed=seed,
            belief_summary_path=belief_summary_path,
        )
    else:
        raise ValueError(f"unsupported candidate mode: {config.mode}")
    _CANDIDATE_CACHE[key] = agent
    return agent


def _candidate_cache_key(
    config: GauntletCandidateConfig,
    *,
    deck_path: Path | None,
    seed: int,
) -> tuple[CandidateMode, str, str, str, str, str, int]:
    checkpoint_key = (
        str(records.repo_path(config.checkpoint_path))
        if config.checkpoint_path is not None
        else ""
    )
    label_key = config.label or ""
    belief_key = (
        str(records.repo_path(config.belief_summary_path))
        if config.belief_summary_path is not None
        else ""
    )
    if config.mode == "policy_greedy":
        return (config.mode, checkpoint_key, label_key, config.device, "", belief_key, 0)
    if config.mode == "runtime":
        return (
            config.mode,
            checkpoint_key,
            label_key,
            config.device,
            str(deck_path or ""),
            belief_key,
            0,
        )
    return (config.mode, checkpoint_key, label_key, config.device, "", belief_key, seed)


def _opponent_agent(spec: OpponentSpec, seed: int) -> BattleAgent:
    key = _opponent_cache_key(spec, seed=seed)
    cached = _OPPONENT_CACHE.get(key)
    if cached is not None:
        return cached
    agent = build_opponent(spec, seed=seed + 29)
    _OPPONENT_CACHE[key] = agent
    return agent


def _opponent_cache_key(
    spec: OpponentSpec,
    *,
    seed: int,
) -> tuple[str, str, str, str, int]:
    checkpoint_key = (
        str(records.repo_path(spec.checkpoint_path))
        if spec.checkpoint_path is not None
        else ""
    )
    seed_key = seed if spec.source == "builtin" else 0
    return (spec.name, spec.source, checkpoint_key, spec.device, seed_key)


def _add_metadata_fields(row: dict[str, Any], task: _GauntletTask) -> None:
    spec = task.plan.opponent_spec
    plan = task.plan.arena_plan
    candidate_action_seconds = float(row.get("candidate_action_seconds", 0.0))
    row.update(
        {
            "candidate_mode": task.candidate.mode,
            "candidate_device": task.candidate.device,
            "candidate_overage_seconds": task.candidate.overage_seconds,
            "candidate_overage_used_seconds": candidate_action_seconds,
            "candidate_overage_exceeded": (
                candidate_action_seconds > task.candidate.overage_seconds
            ),
            "opponent_tier": spec.tier,
            "opponent_source": spec.source,
            "opponent_vector_safe": spec.vector_safe,
            "opponent_requires_search": spec.requires_search,
            "opponent_spec_deck_path": (
                records.display_path(records.repo_path(spec.deck_path))
                if spec.deck_path is not None
                else ""
            ),
            "opponent_matchup_repeat": task.plan.repeat_index,
            "candidate_deck_source": plan.candidate_deck.source,
            "opponent_deck_source": plan.opponent_deck.source,
        }
    )


def _error_row(task: _GauntletTask, exc: Exception) -> dict[str, Any]:
    plan = task.plan.arena_plan
    return {
        "game_index": plan.game_index,
        "seed": plan.seed,
        "candidate_agent": task.candidate.label or f"candidate_{task.candidate.mode}",
        "opponent_agent": task.plan.opponent_spec.name,
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


def _load_opponent_deck_pool(config: GauntletConfig) -> tuple[ArenaDeck, ...]:
    if config.opponent_suite.opponent_deck_mode != "pool":
        return ()
    decks = load_deck_pool(config.opponent_deck_pool)
    if not decks:
        raise ValueError("opponent_deck_pool must not be empty in pool mode")
    return decks


def _opponent_decks_for(
    spec: OpponentSpec,
    *,
    candidate_deck: ArenaDeck,
    pooled_opponent_decks: Sequence[ArenaDeck],
    deck_mode: OpponentDeckMode,
) -> tuple[ArenaDeck, ...]:
    if spec.deck_path is not None:
        return (_single_deck(spec.deck_path),)
    if deck_mode == "mirror":
        return (candidate_deck,)
    return tuple(pooled_opponent_decks)


def _single_deck(path: Path) -> ArenaDeck:
    decks = load_deck_pool(DeckPoolConfig(deck_paths=(path,)))
    if len(decks) != 1:
        raise ValueError(f"expected one deck from {path}, got {len(decks)}")
    return decks[0]


def _candidate_seats(games: int, *, mirror_sides: bool) -> tuple[int, ...]:
    if not mirror_sides:
        return tuple(0 for _ in range(games))
    return tuple(index % 2 for index in range(games))


def _existing_source_path(deck: ArenaDeck) -> Path | None:
    source_path = records.repo_path(Path(deck.source))
    return source_path if source_path.exists() else None


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


def _default_battle_session(deck0: Sequence[int], deck1: Sequence[int]) -> Any:
    from ptcg_rl.engine.session import BattleSession

    return BattleSession(deck0, deck1)


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


def _task_chunk_size(config: GauntletConfig, *, task_count: int) -> int:
    if config.task_chunk_size is not None:
        return config.task_chunk_size
    if task_count <= config.num_workers:
        return 1
    return max(1, min(16, task_count // (config.num_workers * 4)))


def _task_cuda_memory_limit_gb(config: GauntletConfig) -> float | None:
    if config.cuda_memory_limit_gb is None:
        return None
    return config.cuda_memory_limit_gb / float(config.num_workers)


def _configure_cuda_memory_limit(cuda_memory_limit_gb: float | None) -> None:
    global _CONFIGURED_CUDA_MEMORY_LIMIT_GB  # noqa: PLW0603
    if cuda_memory_limit_gb is None:
        return
    if cuda_memory_limit_gb == _CONFIGURED_CUDA_MEMORY_LIMIT_GB:
        return
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    device_index = torch.cuda.current_device()
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    limit_bytes = int(cuda_memory_limit_gb * (1024**3))
    fraction = min(1.0, max(0.0, limit_bytes / float(total_bytes)))
    if fraction > 0.0:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_index)
        _CONFIGURED_CUDA_MEMORY_LIMIT_GB = cuda_memory_limit_gb
