"""Deployment-native head-to-head evaluation for immutable release archives."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import shutil
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.agent.search.budget import ActTimeLedgerConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.session import BattleSession
from ptcg_rl.evaluation.immediate_win_override import (
    ImmediateWinOverrideAgent,
    ImmediateWinOverrideConfig,
)
from ptcg_rl.evaluation.release_agent import (
    IsolatedReleaseAgent,
    ReleaseCapsule,
    bridge_path,
    cleanup_release_cache,
    materialize_release_capsule,
)
from ptcg_rl.evaluation.search_identity import (
    environment_identity,
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.evaluation.streaming_games import StreamingGameStore
from ptcg_rl.training.arena import GamePlan, run_arena_game
from ptcg_rl.training.arena_agents import ArenaAgent
from ptcg_rl.training.arena_decks import DeckPoolConfig, load_deck_pool
from ptcg_rl.training.run_config import (
    TrainingRunConfig,
    resolve_training_output_dir,
)


class ReleaseParticipantConfig(BaseModel):
    """One immutable release manifest selected for direct H2H."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str
    manifest_path: Path
    immediate_win_override: ImmediateWinOverrideConfig = Field(
        default_factory=ImmediateWinOverrideConfig
    )

    @field_validator("label")
    @classmethod
    def nonempty_label(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("release participant label must be non-empty")
        return cleaned

    @field_validator("manifest_path")
    @classmethod
    def immutable_manifest_path(cls, value: Path) -> Path:
        if any("latest" in part.lower() for part in value.parts):
            raise ValueError("release participant cannot reference a latest alias")
        return value


class ReleaseH2HDecisionConfig(BaseModel):
    """Predeclared posterior interpretation for one fixed-size campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_games: int = 400
    rope_score: float = 0.02
    posterior_threshold: float = 0.975
    posterior_samples: int = 200_000

    @field_validator("minimum_games", "posterior_samples")
    @classmethod
    def positive_counts(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("decision counts must be positive")
        return value

    @field_validator("rope_score")
    @classmethod
    def valid_rope(cls, value: float) -> float:
        if not 0.0 <= value < 0.5:
            raise ValueError("rope_score must be in [0, 0.5)")
        return value

    @field_validator("posterior_threshold")
    @classmethod
    def valid_threshold(cls, value: float) -> float:
        if not 0.5 < value < 1.0:
            raise ValueError("posterior_threshold must be in (0.5, 1)")
        return value


class ReleaseH2HConfig(BaseModel):
    """Resolved deployment-native direct H2H campaign."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run: TrainingRunConfig = Field(default_factory=TrainingRunConfig)
    output_dir: Path | None = None
    protocol: Literal["RELEASE-H2H-v1"] = "RELEASE-H2H-v1"
    experiment_id: str
    stage: Literal["smoke", "formal"]
    candidate: ReleaseParticipantConfig
    opponent: ReleaseParticipantConfig
    games: int
    num_workers: int = 1
    partition_worker_cpus: bool = False
    cpu_threads_per_worker: int | None = None
    max_steps_per_game: int = 1000
    result_shard_size: int = 16
    seed: int = 0
    compression: str = "zstd"
    startup_timeout_seconds: float = 300.0
    action_timeout_seconds: float = 30.0
    fail_on_participant_fault: bool = True
    fail_on_unresolved: bool = True
    cleanup_runtime_cache: bool = True
    temp_root: Path = Path("tmp/release_h2h")
    act_time_ledger: ActTimeLedgerConfig = Field(default_factory=ActTimeLedgerConfig)
    decision: ReleaseH2HDecisionConfig = Field(default_factory=ReleaseH2HDecisionConfig)

    @field_validator(
        "games",
        "num_workers",
        "max_steps_per_game",
        "result_shard_size",
    )
    @classmethod
    def positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("release H2H counts must be positive")
        return value

    @field_validator("cpu_threads_per_worker")
    @classmethod
    def positive_optional_threads(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("cpu_threads_per_worker must be positive")
        return value

    @field_validator("startup_timeout_seconds", "action_timeout_seconds")
    @classmethod
    def positive_timeouts(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("release H2H timeouts must be positive")
        return value

    @field_validator("experiment_id")
    @classmethod
    def immutable_experiment_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or "latest" in cleaned.lower():
            raise ValueError("experiment_id must be immutable and non-empty")
        return cleaned

    @model_validator(mode="after")
    def balanced_distinct_campaign(self) -> ReleaseH2HConfig:
        if self.games % 2:
            raise ValueError("release H2H requires an even game count")
        if self.candidate == self.opponent:
            raise ValueError("release H2H participants must be distinct")
        if self.decision.minimum_games > self.games:
            raise ValueError("decision.minimum_games cannot exceed planned games")
        if self.cpu_threads_per_worker is not None and not self.partition_worker_cpus:
            raise ValueError(
                "cpu_threads_per_worker requires partition_worker_cpus=true"
            )
        return self


class ReleaseH2HShardContext(BaseModel):
    """Distributed execution identity attached to every shard row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    campaign_fingerprint: str
    shard_id: str
    host_label: str
    runtime_fingerprint: str
    requested_resource: Literal["cpu", "cuda"]

    @field_validator(
        "campaign_fingerprint",
        "shard_id",
        "host_label",
        "runtime_fingerprint",
    )
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("distributed shard identity fields must be non-empty")
        return normalized


@dataclass(frozen=True)
class _ReleaseGameTask:
    plan: GamePlan
    candidate: ReleaseCapsule
    opponent: ReleaseCapsule
    output_dir: Path
    work_root: Path
    max_steps_per_game: int
    startup_timeout_seconds: float
    action_timeout_seconds: float
    act_time_ledger: ActTimeLedgerConfig
    candidate_immediate_win_override: ImmediateWinOverrideConfig
    opponent_immediate_win_override: ImmediateWinOverrideConfig


class _FailingAgent:
    """Turn a startup exception into an attributable first-act participant fault."""

    def __init__(self, name: str, error: BaseException) -> None:
        self.name = name
        self._error = error

    def act(self, observation: Any) -> Sequence[int]:
        del observation
        raise RuntimeError(f"release startup failed: {self._error}") from self._error


def run_release_h2h(
    config: ReleaseH2HConfig,
    *,
    game_indices: Sequence[int] | None = None,
    shard_context: ReleaseH2HShardContext | None = None,
) -> dict[str, Any]:
    """Run or resume one exact release-archive H2H campaign."""
    output_dir = records.repo_path(
        resolve_training_output_dir(
            task_name="release_h2h",
            run=config.run,
            output_dir=config.output_dir,
        )
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_root = records.repo_path(config.temp_root / "runtime_cache").resolve()
    candidate = materialize_release_capsule(
        label=config.candidate.label,
        manifest_path=config.candidate.manifest_path,
        cache_root=cache_root,
    )
    opponent = materialize_release_capsule(
        label=config.opponent.label,
        manifest_path=config.opponent.manifest_path,
        cache_root=cache_root,
    )
    capsules = (candidate, opponent)
    try:
        _validate_pair(candidate, opponent)
        candidate_deck = _one_deck(candidate)
        opponent_deck = _one_deck(opponent)
        assigned_indices = _resolve_game_indices(config.games, game_indices)
        plans = tuple(
            GamePlan(
                game_index=index,
                seed=config.seed + index,
                candidate_deck=candidate_deck,
                opponent_deck=opponent_deck,
                candidate_seat=index % 2,
            )
            for index in assigned_indices
        )
        worker_cpu_sets = _resolve_worker_cpu_sets(config)
        identity = _evaluation_identity(
            config,
            candidate,
            opponent,
            worker_cpu_sets=worker_cpu_sets,
            game_indices=assigned_indices,
            shard_context=shard_context,
        )
        evaluation_fingerprint = str(identity["evaluation_fingerprint"])
        write_identity_atomic(output_dir / "fingerprints.json", identity)
        write_identity_atomic(
            output_dir / "resolved_config.json",
            config.model_dump(mode="json"),
        )
        work_root = records.repo_path(
            config.temp_root / "work" / evaluation_fingerprint
        ).resolve()
        summary = _run_games(
            config,
            plans=plans,
            candidate=candidate,
            opponent=opponent,
            output_dir=output_dir,
            work_root=work_root,
            identity=identity,
            worker_cpu_sets=worker_cpu_sets,
            expected_game_indices=(
                assigned_indices if game_indices is not None else None
            ),
        )
        score = _score_games(
            records.repo_path(Path(str(summary["games_path"]))),
            decision=config.decision,
        )
        write_identity_atomic(output_dir / "score.json", score)
        summary["score"] = score
        summary["score_path"] = records.display_path(output_dir / "score.json")
        summary_path = output_dir / "summary.json"
        summary["summary_path"] = records.display_path(summary_path)
        write_identity_atomic(summary_path, summary)
        participant_faults = int(score["quality"]["participant_faults"])
        unresolved = int(score["quality"]["unresolved_games"])
        if config.fail_on_participant_fault and participant_faults:
            raise RuntimeError(
                f"release H2H had {participant_faults} participant fault(s)"
            )
        if config.fail_on_unresolved and unresolved:
            raise RuntimeError(f"release H2H had {unresolved} unresolved game(s)")
        return summary
    finally:
        if config.cleanup_runtime_cache:
            cleanup_release_cache(capsules)


def _run_games(
    config: ReleaseH2HConfig,
    *,
    plans: Sequence[GamePlan],
    candidate: ReleaseCapsule,
    opponent: ReleaseCapsule,
    output_dir: Path,
    work_root: Path,
    identity: Mapping[str, Any],
    worker_cpu_sets: Sequence[Sequence[int]],
    expected_game_indices: Sequence[int] | None,
) -> dict[str, Any]:
    evaluation_fingerprint = str(identity["evaluation_fingerprint"])
    with StreamingGameStore(
        output_dir,
        evaluation_fingerprint=evaluation_fingerprint,
        total_games=config.games,
        compression=config.compression,
        result_shard_size=config.result_shard_size,
        expected_game_indices=expected_game_indices,
    ) as store:
        tasks = tuple(
            _ReleaseGameTask(
                plan=plan,
                candidate=candidate,
                opponent=opponent,
                output_dir=output_dir,
                work_root=work_root,
                max_steps_per_game=config.max_steps_per_game,
                startup_timeout_seconds=config.startup_timeout_seconds,
                action_timeout_seconds=config.action_timeout_seconds,
                act_time_ledger=config.act_time_ledger,
                candidate_immediate_win_override=(
                    config.candidate.immediate_win_override
                ),
                opponent_immediate_win_override=(
                    config.opponent.immediate_win_override
                ),
            )
            for plan in plans
            if plan.game_index not in store.completed_indices
        )
        buffered_rows: list[dict[str, Any]] = []
        games_finished = store.resumed_games
        try:
            for row in _iter_rows(
                tasks,
                num_workers=config.num_workers,
                worker_cpu_sets=worker_cpu_sets,
            ):
                _add_identity(row, identity)
                buffered_rows.append(row)
                games_finished += 1
                if len(buffered_rows) >= config.result_shard_size:
                    store.append(buffered_rows)
                    buffered_rows.clear()
                _write_live_status(
                    output_dir,
                    store.progress_payload(games_finished=games_finished),
                    config=config,
                )
        except BaseException:
            if buffered_rows:
                store.append(buffered_rows)
            raise
        if buffered_rows:
            store.append(buffered_rows)
        games_path = store.compact()
        _write_live_status(
            output_dir,
            store.progress_payload(games_finished=games_finished),
            config=config,
            status="completed_games",
        )
        return {
            "runner": "release_h2h",
            "protocol": config.protocol,
            "experiment_id": config.experiment_id,
            "stage": config.stage,
            "games": len(store.completed_indices),
            "planned_games": len(plans),
            "resumed_games": store.resumed_games,
            "result_parts": store.result_parts,
            "terminal_reason_counts": store.terminal_reason_counts,
            "error_actor_counts": store.error_actor_counts,
            "evaluation_fingerprint": evaluation_fingerprint,
            "worker_cpu_sets": [list(cpu_set) for cpu_set in worker_cpu_sets],
            "candidate": candidate.identity(),
            "opponent": opponent.identity(),
            "games_path": records.display_path(games_path),
            "games_parts_dir": records.display_path(store.parts_dir),
            "progress_path": records.display_path(store.progress_path),
            "output_dir": records.display_path(output_dir),
        }


def _iter_rows(
    tasks: Sequence[_ReleaseGameTask],
    *,
    num_workers: int,
    worker_cpu_sets: Sequence[Sequence[int]],
) -> Iterator[dict[str, Any]]:
    if num_workers <= 1:
        for task in tasks:
            yield _run_task(task)
        return
    context = mp.get_context("spawn")
    slot_queue = context.Queue()
    slots: Sequence[Sequence[int] | None] = (
        worker_cpu_sets if worker_cpu_sets else (None,) * num_workers
    )
    for cpu_set in slots:
        slot_queue.put(tuple(cpu_set) if cpu_set is not None else None)
    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(slot_queue,),
    ) as executor:
        task_iterator = iter(tasks)
        futures: set[Future[dict[str, Any]]] = set()
        for _ in range(num_workers):
            initial_task = next(task_iterator, None)
            if initial_task is None:
                break
            futures.add(executor.submit(_run_task, initial_task))
        while futures:
            completed, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in completed:
                yield future.result()
                replacement = next(task_iterator, None)
                if replacement is not None:
                    futures.add(executor.submit(_run_task, replacement))


def _run_task(task: _ReleaseGameTask) -> dict[str, Any]:
    plan = task.plan
    game_dir = task.output_dir / "episode_logs" / f"game-{plan.game_index:06d}"
    game_work = task.work_root / f"game-{plan.game_index:06d}"
    candidate_log = game_dir / "candidate.log"
    opponent_log = game_dir / "opponent.log"
    agents: dict[str, ArenaAgent] = {}
    live_agents: list[IsolatedReleaseAgent] = []

    def start(
        role: str,
        capsule: ReleaseCapsule,
        log_path: Path,
        immediate_win: ImmediateWinOverrideConfig,
    ) -> None:
        try:
            agent = IsolatedReleaseAgent(
                capsule,
                log_path=log_path,
                work_dir=game_work / role,
                startup_timeout_seconds=task.startup_timeout_seconds,
                action_timeout_seconds=task.action_timeout_seconds,
            )
        except Exception as exc:  # Package startup is participant behavior.
            agents[role] = _FailingAgent(capsule.label, exc)
        else:
            agents[role] = (
                ImmediateWinOverrideAgent(
                    agent,
                    config=immediate_win,
                    game_seed=plan.seed,
                )
                if immediate_win.mode != "disabled"
                else agent
            )
            live_agents.append(agent)

    start_order = (
        (
            (
                "candidate",
                task.candidate,
                candidate_log,
                task.candidate_immediate_win_override,
            ),
            (
                "opponent",
                task.opponent,
                opponent_log,
                task.opponent_immediate_win_override,
            ),
        )
        if plan.candidate_seat == 0
        else (
            (
                "opponent",
                task.opponent,
                opponent_log,
                task.opponent_immediate_win_override,
            ),
            (
                "candidate",
                task.candidate,
                candidate_log,
                task.candidate_immediate_win_override,
            ),
        )
    )
    try:
        for role, capsule, log_path, immediate_win in start_order:
            start(role, capsule, log_path, immediate_win)
        row = run_arena_game(
            plan,
            candidate_agent=agents["candidate"],
            opponent_agent=agents["opponent"],
            max_steps=task.max_steps_per_game,
            battle_session_factory=_battle_session,
            on_agent_error=_stop_on_agent_error,
            reset_agents=False,
            act_time_ledger_config=task.act_time_ledger,
        )
    finally:
        for agent in live_agents:
            agent.close()
        shutil.rmtree(game_work, ignore_errors=True)
    if row["terminal_reason"] == "agent_error" and row["error_player_index"] in {
        0,
        1,
    }:
        winner_index = 1 - int(row["error_player_index"])
        row["winner_index"] = winner_index
        row["candidate_result"] = (
            "win" if winner_index == plan.candidate_seat else "loss"
        )
        row["terminal_reason"] = "participant_fault"
    candidate_agent = agents["candidate"]
    opponent_agent = agents["opponent"]
    row.update(
        {
            "candidate_bundle_id": task.candidate.bundle_id,
            "opponent_bundle_id": task.opponent.bundle_id,
            "candidate_manifest_sha256": task.candidate.manifest_sha256,
            "opponent_manifest_sha256": task.opponent.manifest_sha256,
            "candidate_submission_fingerprint": (task.candidate.submission_fingerprint),
            "opponent_submission_fingerprint": (task.opponent.submission_fingerprint),
            "candidate_runtime_sha256": task.candidate.runtime_sha256,
            "opponent_runtime_sha256": task.opponent.runtime_sha256,
            "candidate_main_sha256": task.candidate.main_sha256,
            "opponent_main_sha256": task.opponent.main_sha256,
            "candidate_capsule_fingerprint": (task.candidate.capsule_fingerprint),
            "opponent_capsule_fingerprint": (task.opponent.capsule_fingerprint),
            "candidate_process_id": _agent_int(candidate_agent, "process_id", -1),
            "opponent_process_id": _agent_int(opponent_agent, "process_id", -1),
            "candidate_release_startup_seconds": _agent_float(
                candidate_agent,
                "startup_seconds",
            ),
            "opponent_release_startup_seconds": _agent_float(
                opponent_agent,
                "startup_seconds",
            ),
            "candidate_policy_loaded_observed": bool(
                getattr(candidate_agent, "policy_loaded_observed", False)
            ),
            "opponent_policy_loaded_observed": bool(
                getattr(opponent_agent, "policy_loaded_observed", False)
            ),
            "candidate_log_path": records.display_path(candidate_log),
            "opponent_log_path": records.display_path(opponent_log),
            "execution_mode": "fresh-process-per-seat-episode",
            "environment_provenance": "native-reconstructed",
            "worker_cpu_affinity": json.dumps(
                sorted(int(cpu) for cpu in os.sched_getaffinity(0))
            ),
            "worker_omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "candidate_torch_num_threads": _agent_execution_int(
                candidate_agent,
                "torch_num_threads",
            ),
            "opponent_torch_num_threads": _agent_execution_int(
                opponent_agent,
                "torch_num_threads",
            ),
            "candidate_policy_device": _agent_execution_str(
                candidate_agent,
                "policy_device",
            ),
            "opponent_policy_device": _agent_execution_str(
                opponent_agent,
                "policy_device",
            ),
            "candidate_immediate_win_mode": _agent_str(
                candidate_agent,
                "immediate_win_mode",
                "disabled",
            ),
            "candidate_immediate_win_probe_calls": _agent_int(
                candidate_agent,
                "immediate_win_probe_calls",
                0,
            ),
            "candidate_immediate_win_missed_decisions": _agent_int(
                candidate_agent,
                "immediate_win_missed_decisions",
                0,
            ),
            "candidate_immediate_win_base_verified_decisions": _agent_int(
                candidate_agent,
                "immediate_win_base_verified_decisions",
                0,
            ),
            "candidate_immediate_win_applied_decisions": _agent_int(
                candidate_agent,
                "immediate_win_applied_decisions",
                0,
            ),
        }
    )
    return row


def _score_games(
    games_path: Path,
    *,
    decision: ReleaseH2HDecisionConfig,
) -> dict[str, Any]:
    rows = cast(list[dict[str, Any]], pq.read_table(games_path).to_pylist())
    resolved = [
        row for row in rows if row.get("candidate_result") in {"win", "draw", "loss"}
    ]
    unresolved = len(rows) - len(resolved)
    seat_counts: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
    for row in resolved:
        seat_counts[int(row["candidate_seat"])][str(row["candidate_result"])] += 1
    rng = np.random.default_rng(0)
    seat_scores: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for seat in (0, 1):
        counts = seat_counts[seat]
        posterior = rng.dirichlet(
            np.asarray(
                [
                    counts["win"] + 0.5,
                    counts["draw"] + 0.5,
                    counts["loss"] + 0.5,
                ],
                dtype=np.float64,
            ),
            size=decision.posterior_samples,
        )
        seat_scores.append(posterior[:, 0] + 0.5 * posterior[:, 1])
    score_samples = 0.5 * (seat_scores[0] + seat_scores[1])
    observed_score = (
        sum(
            1.0
            if row["candidate_result"] == "win"
            else 0.5
            if row["candidate_result"] == "draw"
            else 0.0
            for row in resolved
        )
        / len(resolved)
        if resolved
        else 0.0
    )
    practical_superior = float(np.mean(score_samples > 0.5 + decision.rope_score))
    practical_inferior = float(np.mean(score_samples < 0.5 - decision.rope_score))
    practically_equivalent = float(
        np.mean(np.abs(score_samples - 0.5) <= decision.rope_score)
    )
    interpretation = "insufficient_games"
    if len(resolved) >= decision.minimum_games:
        if practical_superior >= decision.posterior_threshold:
            interpretation = "candidate_practically_superior"
        elif practical_inferior >= decision.posterior_threshold:
            interpretation = "candidate_practically_inferior"
        elif practically_equivalent >= decision.posterior_threshold:
            interpretation = "practically_equivalent"
        else:
            interpretation = "inconclusive"
    participant_faults = sum(
        int(row.get("terminal_reason") == "participant_fault") for row in rows
    )
    policy_not_loaded_games = sum(
        int(
            not bool(row.get("candidate_policy_loaded_observed"))
            or not bool(row.get("opponent_policy_loaded_observed"))
        )
        for row in rows
    )
    immediate_win_shadow = _summarize_immediate_win_shadow(rows)
    return {
        "protocol": "RELEASE-H2H-WDL-v1",
        "games": len(rows),
        "resolved_games": len(resolved),
        "counts": dict(Counter(str(row["candidate_result"]) for row in resolved)),
        "per_seat_counts": {str(seat): dict(seat_counts[seat]) for seat in (0, 1)},
        "candidate_score_rate": observed_score,
        "posterior_mean_score": float(np.mean(score_samples)),
        "posterior_ci95": [
            float(np.quantile(score_samples, 0.025)),
            float(np.quantile(score_samples, 0.975)),
        ],
        "probability_candidate_above_half": float(np.mean(score_samples > 0.5)),
        "probability_candidate_practically_superior": practical_superior,
        "probability_candidate_practically_inferior": practical_inferior,
        "probability_practically_equivalent": practically_equivalent,
        "interpretation": interpretation,
        "decision": decision.model_dump(mode="json"),
        "quality": {
            "participant_faults": participant_faults,
            "unresolved_games": unresolved,
            "policy_not_loaded_games": policy_not_loaded_games,
        },
        "immediate_win_shadow": immediate_win_shadow,
    }


def _summarize_immediate_win_shadow(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Derive the guaranteed sample-local uplift from shadow interventions."""
    shadow_rows = [
        row for row in rows if row.get("candidate_immediate_win_mode") == "shadow"
    ]
    missed_rows = [
        row
        for row in shadow_rows
        if int(row.get("candidate_immediate_win_missed_decisions", 0)) > 0
    ]
    resolved = [
        row for row in shadow_rows if row.get("candidate_result") in {"win", "draw", "loss"}
    ]
    baseline_points = sum(
        1.0
        if row.get("candidate_result") == "win"
        else 0.5
        if row.get("candidate_result") == "draw"
        else 0.0
        for row in resolved
    )
    guaranteed_gain = sum(
        1.0
        if row.get("candidate_result") == "loss"
        else 0.5
        if row.get("candidate_result") == "draw"
        else 0.0
        for row in missed_rows
    )
    denominator = len(resolved)
    return {
        "mode": "shadow" if shadow_rows else "disabled",
        "games": len(shadow_rows),
        "resolved_games": denominator,
        "probe_calls": sum(
            int(row.get("candidate_immediate_win_probe_calls", 0))
            for row in shadow_rows
        ),
        "base_verified_win_decisions": sum(
            int(row.get("candidate_immediate_win_base_verified_decisions", 0))
            for row in shadow_rows
        ),
        "missed_win_decisions": sum(
            int(row.get("candidate_immediate_win_missed_decisions", 0))
            for row in shadow_rows
        ),
        "games_with_missed_win": len(missed_rows),
        "missed_game_result_counts": dict(
            Counter(str(row.get("candidate_result")) for row in missed_rows)
        ),
        "guaranteed_additional_score_points": guaranteed_gain,
        "baseline_score_rate": (
            baseline_points / denominator if denominator else None
        ),
        "counterfactual_score_rate": (
            (baseline_points + guaranteed_gain) / denominator
            if denominator
            else None
        ),
        "guaranteed_uplift": (
            guaranteed_gain / denominator if denominator else None
        ),
    }


def score_release_h2h_games(
    games_path: Path,
    *,
    decision: ReleaseH2HDecisionConfig,
) -> dict[str, Any]:
    """Score one verified, possibly distributed, release H2H games file."""
    return _score_games(records.repo_path(games_path), decision=decision)


def _evaluation_identity(
    config: ReleaseH2HConfig,
    candidate: ReleaseCapsule,
    opponent: ReleaseCapsule,
    *,
    worker_cpu_sets: Sequence[Sequence[int]],
    game_indices: Sequence[int],
    shard_context: ReleaseH2HShardContext | None,
) -> dict[str, Any]:
    referee_assets = {
        records.display_path(path): file_sha256(path) for path in _referee_asset_paths()
    }
    referee_fingerprint = fingerprint_payload(referee_assets)
    bridge_sha256 = file_sha256(bridge_path())
    environment = _execution_environment()
    semantic_config = config.model_dump(
        mode="json",
        exclude={
            "output_dir",
            "temp_root",
            "compression",
            "result_shard_size",
            "cleanup_runtime_cache",
        },
    )
    payload = {
        "protocol": config.protocol,
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "semantic_config": semantic_config,
        "candidate": candidate.identity(),
        "opponent": opponent.identity(),
        "referee_fingerprint": referee_fingerprint,
        "bridge_sha256": bridge_sha256,
        "environment": environment,
        "worker_cpu_sets": [list(cpu_set) for cpu_set in worker_cpu_sets],
        "game_indices_fingerprint": fingerprint_payload(
            {"game_indices": list(game_indices)}
        ),
        "game_indices_count": len(game_indices),
        "distributed_shard": (
            shard_context.model_dump(mode="json")
            if shard_context is not None
            else None
        ),
    }
    return {
        **payload,
        "evaluation_fingerprint": fingerprint_payload(payload),
        "referee_assets": referee_assets,
    }


def _execution_environment() -> dict[str, Any]:
    identity = environment_identity()
    try:
        import torch

        identity.update(
            {
                "torch_cuda": str(torch.version.cuda),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count()),
                "cuda_device_name": (
                    str(torch.cuda.get_device_name(0))
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        identity.update(
            {
                "torch_cuda": None,
                "cuda_available": False,
                "cuda_device_count": 0,
                "cuda_device_name": None,
            }
        )
    return identity


def _referee_asset_paths() -> tuple[Path, ...]:
    paths = (
        records.repo_path(Path("data/sample_submission/cg/libcg.so")).resolve(),
        records.repo_path(Path("data/sample_submission/cg/game.py")).resolve(),
        records.repo_path(Path("src/ptcg_rl/engine/session.py")).resolve(),
        records.repo_path(Path("src/ptcg_rl/training/arena.py")).resolve(),
        Path(__file__).resolve(),
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"referee identity asset is missing: {path}")
    return paths


def _add_identity(row: dict[str, Any], identity: Mapping[str, Any]) -> None:
    environment = cast(Mapping[str, Any], identity["environment"])
    row.update(
        {
            "experiment_protocol": identity["protocol"],
            "experiment_id": identity["experiment_id"],
            "experiment_stage": identity["stage"],
            "evaluation_fingerprint": identity["evaluation_fingerprint"],
            "referee_fingerprint": identity["referee_fingerprint"],
            "bridge_sha256": identity["bridge_sha256"],
            "environment_fingerprint": fingerprint_payload(environment),
            "host_platform": environment.get("platform"),
            "git_revision": environment.get("git_revision"),
            "git_dirty": environment.get("git_dirty"),
            "cuda_device_name": environment.get("cuda_device_name"),
        }
    )
    shard = identity.get("distributed_shard")
    if isinstance(shard, Mapping):
        row.update(
            {
                "distributed_campaign_fingerprint": shard.get(
                    "campaign_fingerprint"
                ),
                "distributed_shard_id": shard.get("shard_id"),
                "distributed_host_label": shard.get("host_label"),
                "distributed_runtime_fingerprint": shard.get(
                    "runtime_fingerprint"
                ),
                "distributed_requested_resource": shard.get(
                    "requested_resource"
                ),
            }
        )


def _validate_pair(candidate: ReleaseCapsule, opponent: ReleaseCapsule) -> None:
    if candidate.submission_fingerprint == opponent.submission_fingerprint:
        raise ValueError("release H2H participants resolve to the same submission")
    if candidate.deck_sha256 != opponent.deck_sha256:
        raise ValueError("direct release H2H requires the same exact deck SHA256")


def _one_deck(capsule: ReleaseCapsule) -> Any:
    decks = load_deck_pool(DeckPoolConfig(deck_paths=(capsule.deck_path,)))
    if len(decks) != 1:
        raise ValueError(f"release capsule resolved {len(decks)} decks")
    return decks[0]


def _battle_session(deck0: Sequence[int], deck1: Sequence[int]) -> BattleSession:
    return BattleSession(deck0, deck1)


def _stop_on_agent_error(
    exc: Exception,
    player_index: int,
    agent: ArenaAgent,
    observation: Mapping[str, Any],
) -> None:
    del exc, player_index, agent, observation
    return None


def _agent_int(agent: ArenaAgent, name: str, default: int) -> int:
    value = getattr(agent, name, default)
    return int(value) if isinstance(value, int) else default


def _agent_str(agent: ArenaAgent, name: str, default: str) -> str:
    value = getattr(agent, name, default)
    return value if isinstance(value, str) else default


def _agent_float(agent: ArenaAgent, name: str) -> float:
    value = getattr(agent, name, 0.0)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _agent_execution_int(agent: ArenaAgent, name: str) -> int:
    status = getattr(agent, "execution_status", {})
    if not isinstance(status, Mapping):
        return 0
    value = status.get(name)
    return int(value) if isinstance(value, int) else 0


def _agent_execution_str(agent: ArenaAgent, name: str) -> str:
    status = getattr(agent, "execution_status", {})
    if not isinstance(status, Mapping):
        return ""
    value = status.get(name)
    return str(value) if value is not None else ""


def _resolve_worker_cpu_sets(
    config: ReleaseH2HConfig,
) -> tuple[tuple[int, ...], ...]:
    if not config.partition_worker_cpus:
        return ()
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("worker CPU partitioning requires sched affinity support")
    available = tuple(sorted(int(cpu) for cpu in os.sched_getaffinity(0)))
    threads = config.cpu_threads_per_worker or len(available) // config.num_workers
    if threads <= 0 or threads * config.num_workers > len(available):
        raise ValueError(
            "worker CPU partitions exceed available affinity: "
            f"workers={config.num_workers}, threads={threads}, available={available}"
        )
    return tuple(
        available[index * threads : (index + 1) * threads]
        for index in range(config.num_workers)
    )


def _resolve_game_indices(
    total_games: int,
    game_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if game_indices is None:
        return tuple(range(total_games))
    materialized = tuple(int(index) for index in game_indices)
    if not materialized:
        raise ValueError("release H2H shard must contain at least one game")
    if len(materialized) != len(set(materialized)):
        raise ValueError("release H2H shard game indices must be unique")
    invalid = sorted(index for index in materialized if not 0 <= index < total_games)
    if invalid:
        raise ValueError(f"release H2H shard game indices are out of range: {invalid}")
    return tuple(sorted(materialized))


def _initialize_worker(slot_queue: Any) -> None:
    cpu_set = slot_queue.get()
    if cpu_set is None:
        return
    normalized = tuple(int(cpu) for cpu in cpu_set)
    os.sched_setaffinity(0, normalized)
    threads = str(len(normalized))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = threads


def _write_live_status(
    output_dir: Path,
    progress: Mapping[str, Any],
    *,
    config: ReleaseH2HConfig,
    status: str = "running_games",
) -> None:
    write_identity_atomic(
        output_dir / "status.json",
        {
            "status": status,
            "experiment_id": config.experiment_id,
            "stage": config.stage,
            **progress,
        },
    )


__all__ = [
    "ReleaseH2HConfig",
    "ReleaseH2HDecisionConfig",
    "ReleaseParticipantConfig",
    "ReleaseH2HShardContext",
    "run_release_h2h",
    "score_release_h2h_games",
]
