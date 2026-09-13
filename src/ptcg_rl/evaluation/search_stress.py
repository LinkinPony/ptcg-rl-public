"""S2 virtual-clock frozen-trace and overage-boundary stress runner."""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.runtime import ActTimeConfig, CheckpointPolicy, PolicyRuntimeAgent
from ptcg_rl.agent.search.budget import ActTimeLedger, ActTimeLedgerConfig
from ptcg_rl.agent.search.config import SearchRuntimeConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import (
    SearchCampaignIdentityConfig,
    build_search_campaign_identity,
    file_sha256,
    write_identity_atomic,
)
from ptcg_rl.evaluation.search_stress_config import (
    SearchTraceCellConfig,
    SearchTraceStressConfig,
)
from ptcg_rl.evaluation.search_stress_io import SearchStressWriter
from ptcg_rl.evaluation.search_stress_metrics import (
    StressRunStats,
    action_row,
    attach_action_equivalence,
    run_row,
    stress_cell_checks_observed,
    stress_summary,
)
from ptcg_rl.evaluation.search_stress_trace import (
    FrozenTraceCompletion,
    FrozenTraceObservation,
    ScaledVirtualClock,
    first_trace_observation,
    iter_frozen_trace,
    resolve_stress_replays,
)


def run_search_trace_stress(config: SearchTraceStressConfig) -> dict[str, Any]:
    """Execute all frozen trace, slowdown, seat, and overage-boundary cells."""
    _configure_reference_cpu(config)
    replay_paths = resolve_stress_replays(config.replay_paths, config.replay_glob)
    output_dir = records.repo_path(config.output_dir)
    _prepare_output_dir(output_dir, overwrite=config.overwrite)
    belief_path = (
        config.belief.deck_signature_summary_path
        or config.sampler.prior_deck_signature_summary_path
    )
    identity = build_search_campaign_identity(
        SearchCampaignIdentityConfig(
            experiment_id=config.experiment_id,
            stage="S2",
            deck_path=config.deck_path,
            checkpoint_path=config.checkpoint_path,
            belief_path=belief_path,
            resolved_search=config.macro,
            runtime_definition={
                "device": config.device,
                "precision": config.precision,
                "belief": config.belief.model_dump(mode="json"),
                "sampler": config.sampler.model_dump(mode="json"),
                "torch_num_threads": config.torch_num_threads,
                "cpu_affinity": config.cpu_affinity,
                "references": config.references.model_dump(mode="json"),
            },
            replay_paths=replay_paths,
            replay_definition={
                "kind": "same-seat concatenated synthetic budget trace",
                "team_name": config.team_name,
                "cells": [cell.model_dump(mode="json") for cell in config.cells],
            },
            runtime_source_paths=config.runtime_source_paths,
            engine_asset_paths=config.engine_asset_paths,
            stage_parameters={
                "cells": [cell.model_dump(mode="json") for cell in config.cells],
                "overage_boundaries": config.overage_boundaries,
                "seed": config.seed,
            },
        )
    )
    deck = records.read_deck(records.repo_path(config.deck_path))
    run_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    with SearchStressWriter(output_dir, compression=config.compression) as writer:
        for cell in config.cells:
            run_rows.append(
                _run_trace_cell(
                    config,
                    cell,
                    replay_paths=replay_paths,
                    deck=deck,
                    identity=identity,
                    writer=writer,
                )
            )
        for seat in (0, 1):
            source = first_trace_observation(
                replay_paths,
                team_name=config.team_name,
                seat=seat,
                chunk_size=config.chunk_size,
            )
            for remaining in config.overage_boundaries:
                run_rows.append(
                    _run_boundary_cell(
                        config,
                        source,
                        remaining=remaining,
                        deck=deck,
                        identity=identity,
                        writer=writer,
                    )
                )
        attach_action_equivalence(run_rows)
        for row in run_rows:
            row["safety_checks_observed"] = stress_cell_checks_observed(row, config)
        writer.write_runs(run_rows)
    summary = stress_summary(
        config,
        identity,
        run_rows,
        elapsed_seconds=time.perf_counter() - started,
    )
    _write_artifacts(output_dir, config, identity, summary)
    return summary


def _run_trace_cell(
    config: SearchTraceStressConfig,
    cell: SearchTraceCellConfig,
    *,
    replay_paths: Sequence[Path],
    deck: Sequence[int],
    identity: Mapping[str, Any],
    writer: SearchStressWriter,
) -> dict[str, Any]:
    policy = CheckpointPolicy(records.repo_path(config.checkpoint_path), device=config.device)
    clock = ScaledVirtualClock(cell.slowdown_factor)
    agent = build_stress_agent(config, cell.controller, policy=policy, clock=clock)
    agent.begin_game(player_index=cell.seat, own_deck=deck)
    ledger = ActTimeLedger(
        ActTimeLedgerConfig(
            enabled=True,
            total_seconds=config.references.total_overage_seconds,
            startup_charge_seconds=config.references.startup_charge_seconds,
        )
    )
    stats = StressRunStats()
    completion = FrozenTraceCompletion(0, 0, 0)
    timed_out = False
    real_started = time.perf_counter()
    virtual_started = clock()
    for item in iter_frozen_trace(
        replay_paths,
        team_name=config.team_name,
        seat=cell.seat,
        global_step_limit=cell.global_steps,
        chunk_size=config.chunk_size,
    ):
        if isinstance(item, FrozenTraceCompletion):
            completion = item
            break
        completion = FrozenTraceCompletion(
            global_steps=item.global_step + 1,
            callbacks=stats.callbacks,
            source_replays=item.source_replay_index + 1,
        )
        remaining_before = ledger.remaining(cell.seat)
        callback_observation = ledger.observation_for(item.observation, cell.seat)
        real_before = time.perf_counter()
        virtual_before = clock()
        error_type: str | None = None
        try:
            action = tuple(int(index) for index in agent.act(callback_observation))
        except Exception as exc:  # Defensive evidence: PolicyRuntimeAgent should contain it.
            action = ()
            error_type = type(exc).__name__
        real_elapsed = time.perf_counter() - real_before
        virtual_elapsed = max(0.0, clock() - virtual_before)
        try:
            ledger.charge(cell.seat, virtual_elapsed)
        except Exception as exc:
            timed_out = True
            error_type = error_type or type(exc).__name__
        remaining_after = ledger.remaining(cell.seat)
        select = item.observation.get("select")
        legal = is_legal_action(select, action)
        telemetry = agent.last_act_telemetry()
        errors = _agent_errors(agent, error_type)
        stats.observe(
            global_step=item.global_step,
            action=action,
            legal=legal,
            telemetry=telemetry,
            errors=errors,
        )
        writer.write_action(
            action_row(
                identity,
                run_kind="trace",
                cell_id=cell.cell_id,
                controller=cell.controller,
                seat=cell.seat,
                global_steps=cell.global_steps,
                slowdown_factor=cell.slowdown_factor,
                source=item,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                real_elapsed=real_elapsed,
                virtual_elapsed=virtual_elapsed,
                action=action,
                legal=legal,
                timed_out=timed_out,
                error_type=error_type,
                telemetry=telemetry,
            )
        )
        if timed_out:
            break
    real_elapsed_total = time.perf_counter() - real_started
    virtual_elapsed_total = max(0.0, clock() - virtual_started)
    return run_row(
        identity,
        run_kind="trace",
        cell_id=cell.cell_id,
        controller=cell.controller,
        seat=cell.seat,
        global_steps_requested=cell.global_steps,
        global_steps_completed=completion.global_steps,
        slowdown_factor=cell.slowdown_factor,
        boundary_remaining=None,
        source_replays=completion.source_replays,
        real_elapsed=real_elapsed_total,
        virtual_elapsed=virtual_elapsed_total,
        final_remaining=ledger.remaining(cell.seat),
        timed_out=timed_out,
        stats=stats,
    )


def _run_boundary_cell(
    config: SearchTraceStressConfig,
    source: FrozenTraceObservation,
    *,
    remaining: float,
    deck: Sequence[int],
    identity: Mapping[str, Any],
    writer: SearchStressWriter,
) -> dict[str, Any]:
    current = source.observation.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("boundary source has no current player mapping")
    seat = int(current["yourIndex"])
    cell_id = f"boundary_s{seat}_r{remaining:g}"
    policy = CheckpointPolicy(records.repo_path(config.checkpoint_path), device=config.device)
    clock = ScaledVirtualClock(1.0)
    agent = build_stress_agent(
        config,
        "override",
        policy=policy,
        clock=clock,
        prewarm_on_startup=False,
    )
    agent.begin_game(player_index=seat, own_deck=deck)
    observation = dict(source.observation)
    observation["remainingOverageTime"] = remaining
    real_before = time.perf_counter()
    virtual_before = clock()
    error_type: str | None = None
    try:
        action = tuple(agent.act(observation))
    except Exception as exc:
        action = ()
        error_type = type(exc).__name__
    real_elapsed = time.perf_counter() - real_before
    virtual_elapsed = max(0.0, clock() - virtual_before)
    legal = is_legal_action(observation.get("select"), action)
    telemetry = agent.last_act_telemetry()
    stats = StressRunStats()
    stats.observe(
        global_step=source.global_step,
        action=action,
        legal=legal,
        telemetry=telemetry,
        errors=_agent_errors(agent, error_type),
    )
    writer.write_action(
        action_row(
            identity,
            run_kind="boundary",
            cell_id=cell_id,
            controller="override",
            seat=seat,
            global_steps=0,
            slowdown_factor=1.0,
            source=source,
            remaining_before=remaining,
            remaining_after=max(0.0, remaining - virtual_elapsed),
            real_elapsed=real_elapsed,
            virtual_elapsed=virtual_elapsed,
            action=action,
            legal=legal,
            timed_out=False,
            error_type=error_type,
            telemetry=telemetry,
        )
    )
    return run_row(
        identity,
        run_kind="boundary",
        cell_id=cell_id,
        controller="override",
        seat=seat,
        global_steps_requested=0,
        global_steps_completed=0,
        slowdown_factor=1.0,
        boundary_remaining=remaining,
        source_replays=1,
        real_elapsed=real_elapsed,
        virtual_elapsed=virtual_elapsed,
        final_remaining=max(0.0, remaining - virtual_elapsed),
        timed_out=False,
        stats=stats,
    )


def build_stress_agent(
    config: SearchTraceStressConfig,
    controller: str,
    *,
    policy: CheckpointPolicy,
    clock: Callable[[], float],
    prewarm_on_startup: bool = True,
) -> PolicyRuntimeAgent:
    """Build the exact checkpoint/belief/runtime bundle for one stress mode."""
    belief = config.belief.model_copy(deep=True)
    if belief.deck_signature_summary_path is not None:
        belief.deck_signature_summary_path = records.repo_path(
            belief.deck_signature_summary_path
        )
    sampler = config.sampler
    if sampler.prior_deck_signature_summary_path is not None:
        sampler = sampler.model_copy(
            update={
                "prior_deck_signature_summary_path": records.repo_path(
                    sampler.prior_deck_signature_summary_path
                )
            }
        )
    macro = config.macro.model_copy(update={"mode": controller})
    runtime = ActTimeConfig(
        deck_path=records.repo_path(config.deck_path),
        checkpoint_path=records.repo_path(config.checkpoint_path),
        seed=config.seed,
        prewarm_on_startup=prewarm_on_startup,
        belief=belief,
        search=SearchRuntimeConfig(
            enabled=True,
            worlds=macro.worlds,
            top_k=macro.top_k,
            manual_coin=macro.manual_coin,
            sampler=sampler,
            macro=macro,
        ),
    )
    return PolicyRuntimeAgent(
        config=runtime,
        policy=policy,
        clock=clock,
        strict_runtime_errors=True,
    )


def _configure_reference_cpu(config: SearchTraceStressConfig) -> None:
    os.environ["OMP_NUM_THREADS"] = str(config.torch_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(config.torch_num_threads)
    available = set(os.sched_getaffinity(0))
    requested = set(config.cpu_affinity)
    if not requested <= available:
        raise ValueError(f"requested CPU affinity is unavailable: {sorted(requested - available)}")
    os.sched_setaffinity(0, requested)
    import torch

    torch.set_num_threads(config.torch_num_threads)
    with contextlib.suppress(RuntimeError):
        torch.set_num_interop_threads(1)


def _agent_errors(agent: PolicyRuntimeAgent, outer_error: str | None) -> tuple[str, ...]:
    values = [outer_error or ""]
    for error in (
        agent.last_policy_error,
        agent.last_belief_error,
        agent.last_probe_error,
        agent.last_search_error,
    ):
        if error is not None:
            values.append(f"{type(error).__name__}:{error}")
    return tuple(values)


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    names = (
        "actions.parquet",
        "stress_runs.parquet",
        "resolved_config.json",
        "fingerprints.json",
        "environment.json",
        "summary.json",
        "manifest.json",
    )
    existing = [output_dir / name for name in names if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"immutable S2 output already exists: {existing[0]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def _write_artifacts(
    output_dir: Path,
    config: SearchTraceStressConfig,
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    _write_json(output_dir / "resolved_config.json", config.model_dump(mode="json"))
    write_identity_atomic(output_dir / "fingerprints.json", identity)
    write_identity_atomic(output_dir / "environment.json", identity["environment"])
    write_identity_atomic(output_dir / "summary.json", summary)
    outputs = {
        name: {
            "path": records.display_path(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in {
            "actions": output_dir / "actions.parquet",
            "stress_runs": output_dir / "stress_runs.parquet",
        }.items()
    }
    write_identity_atomic(
        output_dir / "manifest.json",
        {
            **identity,
            "runner_complete": summary["runner_complete"],
            "diagnostic_warnings": summary["diagnostic_warnings"],
            "output_files": outputs,
        },
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
