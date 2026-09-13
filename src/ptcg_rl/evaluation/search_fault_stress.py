"""Controlled S2 runtime fallback and lifecycle fault injection."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import ptcg_rl.agent.runtime as runtime_module
from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.probe import enumerate_select_actions
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.candidates import CandidateSet
from ptcg_rl.agent.search.macro import MacroEndpoint
from ptcg_rl.agent.search.reranker import (
    MacroSearchResult,
    MacroWorldEvaluation,
)
from ptcg_rl.agent.search.scoring import PairedActionScore, PairedRerankDecision
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import (
    SearchCampaignIdentityConfig,
    build_search_campaign_identity,
    file_sha256,
    write_identity_atomic,
)
from ptcg_rl.evaluation.search_stress import build_stress_agent
from ptcg_rl.evaluation.search_stress_config import SearchTraceStressConfig
from ptcg_rl.evaluation.search_stress_trace import (
    FrozenTraceObservation,
    iter_frozen_trace,
    resolve_stress_replays,
)

_FAULTS = ("search_exception", "partial_world_coverage", "state_pool_pressure")


def run_search_fault_stress(config: SearchTraceStressConfig) -> dict[str, Any]:
    """Inject three search failures and verify legal saved-action fallback."""
    replay_paths = resolve_stress_replays(config.replay_paths, config.replay_glob)
    source = _first_branching_main(
        replay_paths,
        team_name=config.team_name,
        chunk_size=config.chunk_size,
    )
    output_dir = records.repo_path(config.output_dir)
    _prepare_output(output_dir, overwrite=config.overwrite)
    identity = _identity(config, replay_paths)
    policy = CheckpointPolicy(
        records.repo_path(config.checkpoint_path), device=config.device
    )
    deck = records.read_deck(records.repo_path(config.deck_path))
    observation = dict(source.observation)
    observation["remainingOverageTime"] = config.references.total_overage_seconds
    rows = [
        _run_fault(
            config,
            fault,
            source=source,
            observation=observation,
            deck=deck,
            policy=policy,
            identity=identity,
        )
        for fault in _FAULTS
    ]
    _write_rows(output_dir / "fault_runs.parquet", rows, config.compression)
    checks = {
        "all_legal": all(bool(row["legal"]) for row in rows),
        "saved_action_fallback": all(bool(row["saved_action_fallback"]) for row in rows),
        "fallback_available": all(bool(row["fallback_available"]) for row in rows),
        "search_exception_classified": rows[0]["search_error"] == "RuntimeError",
        "partial_coverage_blocked": rows[1]["override_gate_reason"]
        == "incomplete_coverage",
        "state_pool_pressure_blocked": rows[2]["override_gate_reason"]
        == "state_pool_limit",
        "state_leaks_zero": all(int(row["state_leaks"]) == 0 for row in rows),
    }
    summary = {
        "protocol": "ITS-EVAL-v1-S2-FAULT",
        "experiment_id": config.experiment_id,
        "campaign_fp": identity["campaign_fp"],
        "stage_fp": identity["stage_fp"],
        "runner_complete": len(rows) == len(_FAULTS),
        "decision_role": "runtime_safety_diagnostic",
        "checks": checks,
        "diagnostic_warnings": [
            name for name, observed in checks.items() if not observed
        ],
        "rows": rows,
    }
    write_identity_atomic(output_dir / "fingerprints.json", identity)
    write_identity_atomic(output_dir / "environment.json", identity["environment"])
    write_identity_atomic(output_dir / "summary.json", summary)
    write_identity_atomic(
        output_dir / "manifest.json",
        {
            **identity,
            "runner_complete": summary["runner_complete"],
            "diagnostic_warnings": summary["diagnostic_warnings"],
            "output_files": {
                "fault_runs": {
                    "path": records.display_path(output_dir / "fault_runs.parquet"),
                    "sha256": file_sha256(output_dir / "fault_runs.parquet"),
                }
            },
        },
    )
    return summary


def _run_fault(
    config: SearchTraceStressConfig,
    fault: str,
    *,
    source: FrozenTraceObservation,
    observation: Mapping[str, Any],
    deck: Sequence[int],
    policy: CheckpointPolicy,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    agent = build_stress_agent(
        config,
        "override",
        policy=policy,
        clock=time.perf_counter,
    )
    agent.begin_game(player_index=_seat(observation), own_deck=deck)
    searcher = _searcher_for_fault(fault)
    with patch.object(runtime_module, "PairedMacroSearcher", searcher):
        action = tuple(int(index) for index in agent.act(observation))
    telemetry = agent.last_act_telemetry()
    base_action = tuple(agent.last_base_action or ())
    return {
        "campaign_fp": identity["campaign_fp"],
        "stage_fp": identity["stage_fp"],
        "fault": fault,
        "source_episode_id": source.source_episode_id,
        "source_step": source.source_step,
        "seat": _seat(observation),
        "base_action": list(base_action),
        "served_action": list(action),
        "legal": is_legal_action(observation.get("select"), action),
        "saved_action_fallback": action == base_action,
        "search_error": (
            type(agent.last_search_error).__name__
            if agent.last_search_error is not None
            else None
        ),
        "stop_reason": telemetry.get("stop_reason"),
        "override_gate_reason": telemetry.get("override_gate_reason"),
        "worlds_requested": int(telemetry.get("worlds_requested", 0) or 0),
        "worlds_completed": int(telemetry.get("worlds_completed", 0) or 0),
        "state_pool_peak": int(telemetry.get("state_pool_peak", 0) or 0),
        "state_leaks": int(telemetry.get("state_leaks", 0) or 0),
        "fallback_available": bool(telemetry.get("fallback_available", False)),
        "deadline_overshoot_seconds": float(
            telemetry.get("deadline_overshoot_seconds", 0.0) or 0.0
        ),
    }


def _searcher_for_fault(fault: str) -> type[Any]:
    if fault == "search_exception":
        return _ExceptionSearcher
    if fault == "partial_world_coverage":
        return _PartialCoverageSearcher
    if fault == "state_pool_pressure":
        return _StatePoolPressureSearcher
    raise ValueError(f"unknown search fault: {fault}")


class _ExceptionSearcher:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def run(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("injected search failure")


class _PartialCoverageSearcher:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def run(
        self,
        observation: Any,
        *,
        greedy_action: Sequence[int],
        deadline: float,
    ) -> MacroSearchResult:
        del deadline
        return injected_search_result(
            observation,
            greedy_action=greedy_action,
            complete=False,
            state_pool_peak=2,
        )


class _StatePoolPressureSearcher:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    def run(
        self,
        observation: Any,
        *,
        greedy_action: Sequence[int],
        deadline: float,
    ) -> MacroSearchResult:
        del deadline
        return injected_search_result(
            observation,
            greedy_action=greedy_action,
            complete=True,
            state_pool_peak=129,
        )


def injected_search_result(
    observation: Any,
    *,
    greedy_action: Sequence[int],
    complete: bool,
    state_pool_peak: int,
) -> MacroSearchResult:
    """Build adversarial paired evidence that recommends a legal alternative."""
    greedy = tuple(int(index) for index in greedy_action)
    select = observation.get("select") if isinstance(observation, Mapping) else None
    alternatives = [
        action
        for action in enumerate_select_actions(select, max_actions=64)
        if action != greedy
    ]
    if not alternatives:
        raise ValueError("fault injection root has no non-greedy legal action")
    selected = alternatives[0]
    actions = (greedy, selected)
    evaluations = tuple(
        MacroWorldEvaluation(
            action=action,
            world_index=world,
            endpoint=MacroEndpoint.SAME_SEAT_MAIN,
            steps=1,
            engine_score=0.5 if action == selected else 0.0,
            critic_value=0.0,
            stop_detail="injected",
            transition=None,
        )
        for world in range(3 if complete else 1)
        for action in actions
    )
    score = PairedActionScore(
        action=selected,
        paired_worlds=3,
        mean_delta=0.5,
        std_delta=0.0,
        robust_delta=0.5,
        downside_cvar=0.5,
        minimum_delta=0.5,
    )
    return MacroSearchResult(
        candidates=CandidateSet(
            actions=actions,
            sources=(("greedy",), ("fault_injection",)),
        ),
        evaluations=evaluations,
        decision=PairedRerankDecision(
            greedy_action=greedy,
            selected_action=selected,
            reason="injected_recommendation",
            action_scores=(score,),
        ),
        worlds_requested=3,
        worlds_sampled=3,
        complete_coverage=complete,
        stop_reason="complete" if complete else "incomplete_coverage",
        transitions=len(evaluations),
        engine_sessions=len(evaluations),
        state_pool_peak=state_pool_peak,
        state_leaks=0,
    )


def _first_branching_main(
    replay_paths: Sequence[Path], *, team_name: str, chunk_size: int
) -> FrozenTraceObservation:
    for seat in (0, 1):
        for item in iter_frozen_trace(
            replay_paths,
            team_name=team_name,
            seat=seat,
            global_step_limit=20_000,
            chunk_size=chunk_size,
        ):
            if not isinstance(item, FrozenTraceObservation):
                continue
            select = item.observation.get("select")
            if not isinstance(select, Mapping) or int(select.get("context", -1)) != 0:
                continue
            if len(enumerate_select_actions(select, max_actions=3)) >= 2:
                return item
    raise ValueError("replay panel has no branching MAIN callback")


def _identity(
    config: SearchTraceStressConfig, replay_paths: Sequence[Path]
) -> dict[str, Any]:
    belief_path = (
        config.belief.deck_signature_summary_path
        or config.sampler.prior_deck_signature_summary_path
    )
    return build_search_campaign_identity(
        SearchCampaignIdentityConfig(
            experiment_id=config.experiment_id,
            stage="S2",
            deck_path=config.deck_path,
            checkpoint_path=config.checkpoint_path,
            belief_path=belief_path,
            resolved_search=config.macro,
            runtime_definition={"faults": _FAULTS, "device": config.device},
            replay_paths=tuple(replay_paths),
            replay_definition={"kind": "controlled runtime fault root"},
            runtime_source_paths=config.runtime_source_paths,
            engine_asset_paths=config.engine_asset_paths,
            stage_parameters={"faults": _FAULTS},
        )
    )


def _seat(observation: Mapping[str, Any]) -> int:
    current = observation.get("current")
    return int(current.get("yourIndex", 0)) if isinstance(current, Mapping) else 0


def _prepare_output(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "fault_runs.parquet",
        "fingerprints.json",
        "environment.json",
        "summary.json",
        "manifest.json",
    )
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"immutable S2 fault output exists: {existing[0]}")
    if overwrite:
        for path in existing:
            path.unlink()


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], compression: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression=compression)
    temporary.replace(path)
