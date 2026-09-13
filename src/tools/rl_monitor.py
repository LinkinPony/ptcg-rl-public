"""Print a compact live status summary for an RL training run.

Run with:
    PYTHONPATH=src python src/tools/rl_monitor.py outputs/rl/formal_longrun
    PYTHONPATH=src python src/tools/rl_monitor.py outputs/rl/formal_longrun --watch 30
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

_DEFAULT_RUNTIME_STALE_SECONDS = 60.0
_OLDEST_LIVE_GAMES_LIMIT = 4


def main() -> None:
    """Parse CLI arguments and print run status."""
    args = _parse_args()
    run_dir = Path(args.run_dir)
    while True:
        status = build_status(run_dir)
        if args.json:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(format_status(status))
        if args.watch is None:
            return
        time.sleep(args.watch)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", help="RL run directory to inspect.")
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        help="Refresh interval in seconds. Omit for a single snapshot.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw synthesized status object.",
    )
    return parser.parse_args()


def build_status(run_dir: Path) -> dict[str, Any]:
    """Build a normalized status object from run artifacts."""
    runtime_status_path = run_dir / "runtime_status.json"
    runtime_monitor_path = run_dir / "runtime_monitor.json"
    runtime_status = _read_json(runtime_status_path)
    runtime_monitor = _read_json(runtime_monitor_path)
    runtime_summary = _mapping(
        runtime_status.get("summary") or runtime_monitor.get("summary")
    )
    latest_weights_path = run_dir / "weights" / "latest.json"
    latest_weights = _read_json(latest_weights_path)
    summary = _read_json(run_dir / "summary.json")
    learner_status_path = run_dir / "learner_status.json"
    learner_status = _read_json(learner_status_path)
    inference = _read_json(run_dir / "inference_summary.json")
    curriculum = _read_json(run_dir / "curriculum" / "central_summary.json")
    native_distributed = _read_json(
        run_dir / "control" / "native_distributed_status.json"
    )
    learner_supervisor = _read_json(
        run_dir / "control" / "learner_supervisor.json"
    )
    last_sample = _mapping(runtime_summary.get("last_sample"))
    runtime_failure = _mapping(runtime_summary.get("terminal_failure"))
    runtime_update_age = _path_age_seconds(
        runtime_status_path if runtime_status else runtime_monitor_path
    )
    latest_status = (
        {**latest_weights, "age_seconds": _path_age_seconds(latest_weights_path)}
        if latest_weights
        else {}
    )
    learner_status_data = (
        {**learner_status, "age_seconds": _path_age_seconds(learner_status_path)}
        if learner_status
        else {}
    )
    return {
        "run_dir": str(run_dir),
        "status": _run_status(
            summary,
            last_sample,
            runtime_update_age_seconds=runtime_update_age,
            runtime_write_interval_seconds=_optional_float(
                runtime_summary.get("live_write_interval_seconds")
            ),
            runtime_failure=runtime_failure,
            learner_supervisor=learner_supervisor,
        ),
        "updated_age_seconds": runtime_update_age,
        "runtime": runtime_summary,
        "latest_weights": latest_status,
        "learner_status": learner_status_data,
        "inference": inference,
        "actors": _build_actor_status(run_dir),
        "summary": summary,
        "curriculum": curriculum,
        "native_distributed": native_distributed,
        "learner_supervisor": learner_supervisor,
    }


def format_status(status: Mapping[str, Any]) -> str:
    """Return a human-readable one-screen status summary."""
    runtime = _mapping(status.get("runtime"))
    latest = _mapping(status.get("latest_weights"))
    learner_status = _mapping(status.get("learner_status"))
    inference = _mapping(status.get("inference"))
    actors = _mapping(status.get("actors"))
    summary = _mapping(status.get("summary"))
    curriculum = _mapping(status.get("curriculum"))
    native_distributed = _mapping(status.get("native_distributed"))
    learner_supervisor = _mapping(status.get("learner_supervisor"))
    last_sample = _mapping(runtime.get("last_sample"))

    lines = [
        f"run: {status.get('run_dir')} [{status.get('status')}]",
        _format_learner_supervisor(learner_supervisor),
        f"runtime_status_age: {_fmt_seconds(status.get('updated_age_seconds'))}",
        _format_latest(latest),
        _format_learner_status(learner_status),
        _format_inference(inference),
        _format_runtime(runtime, last_sample),
        _format_gpu_phases(runtime),
        _format_processes(last_sample),
        _format_actors(actors),
        _format_native_distributed(native_distributed),
        _format_curriculum(curriculum),
    ]
    if summary:
        lines.append(_format_final_summary(summary))
    return "\n".join(line for line in lines if line)


def _format_learner_supervisor(status: Mapping[str, Any]) -> str:
    if not status:
        return ""
    checkpoint = _mapping(status.get("resume_checkpoint"))
    checkpoint_text = (
        "-" if not checkpoint else f"v{checkpoint.get('version', '?')}"
    )
    return (
        "learner_supervisor: "
        f"state={status.get('state')} "
        f"restarts={_to_int(status.get('restart_count'))} "
        f"exit={status.get('learner_exitcode')} "
        f"backoff={_fmt_seconds(status.get('backoff_seconds'))} "
        f"checkpoint={checkpoint_text}"
    )


def _format_native_distributed(status: Mapping[str, Any]) -> str:
    if not status:
        return ""
    coordinator = _mapping(status.get("coordinator"))
    workers = _mapping(status.get("workers"))
    if not coordinator:
        return "native_distributed: no coordinator status"
    return (
        "native_distributed: "
        f"state={coordinator.get('window_state')} "
        f"workers={coordinator.get('connected_workers')} "
        f"degraded={coordinator.get('degraded_workers')} "
        f"decisions={_to_int(coordinator.get('accepted_decisions'))}/"
        f"{_to_int(coordinator.get('target_decisions'))} "
        f"rate={_to_float(coordinator.get('decisions_per_second')):.1f}/s "
        f"eta={_fmt_seconds(coordinator.get('estimated_seconds_remaining'))} "
        f"shards={_to_int(coordinator.get('shards_completed'))}/"
        f"{_to_int(coordinator.get('shards_issued'))} "
        f"retries={_to_int(coordinator.get('retries'))} "
        f"parts={_to_int(coordinator.get('parts_accepted'))} "
        f"bytes/decision={_to_float(coordinator.get('bytes_per_decision')):.1f} "
        f"worker_metrics={len(workers)}"
    )


def _format_latest(latest: Mapping[str, Any]) -> str:
    if not latest:
        return "weights: not published yet"
    metadata = _mapping(latest.get("metadata"))
    stale = _to_int(metadata.get("stale_decisions"))
    kept = _to_int(metadata.get("kept_decisions"))
    total = stale + kept
    stale_pct = (100.0 * stale / total) if total else 0.0
    early_stop = _mapping(metadata.get("early_stop"))
    early = ""
    if early_stop.get("early_stopped"):
        early = (
            f" early_stop=e{early_stop.get('epoch')} "
            f"kl={_to_float(early_stop.get('approx_kl')):.5f}"
        )
    return (
        "weights: "
        f"v{latest.get('version', '?')} "
        f"age={_fmt_seconds(latest.get('age_seconds'))} "
        f"updates={metadata.get('updates', '?')} "
        f"kept={kept} stale={stale} ({stale_pct:.3f}%)"
        f"{early}"
    )


def _format_learner_status(learner_status: Mapping[str, Any]) -> str:
    if not learner_status:
        return "learner: no live status yet"
    last_update = _mapping(learner_status.get("last_update"))
    early_stop = _mapping(learner_status.get("early_stop"))
    staleness_refill = _mapping(learner_status.get("staleness_refill"))
    kl = f" kl={_to_float(last_update.get('approx_kl')):.5f}" if last_update else ""
    early = " early_stop=yes" if early_stop.get("early_stopped") else ""
    refill = ""
    if staleness_refill:
        refill = (
            " refill("
            f"raw={_format_optional_int(staleness_refill.get('accumulated_raw_decisions'))} "
            f"stale={_format_optional_int(staleness_refill.get('stale_decisions'))} "
            "retained="
            f"{_format_optional_int(staleness_refill.get('retained_budget_decisions'))} "
            f"remaining={_format_optional_int(staleness_refill.get('remaining_decisions'))} "
            f"topups={_format_optional_int(staleness_refill.get('topup_chunks'))}"
            ")"
        )
    version = (
        learner_status.get("published_version")
        or learner_status.get("publish_version")
        or "?"
    )
    return (
        "learner: "
        f"phase={learner_status.get('phase')} "
        f"age={_fmt_seconds(learner_status.get('age_seconds'))} "
        f"iter={learner_status.get('iteration')} "
        f"policy_v={learner_status.get('current_policy_version')} "
        f"target_v={version} "
        f"epoch={learner_status.get('epoch', '-')} "
        f"updates={_to_int(learner_status.get('updates') or learner_status.get('iteration_updates'))}"
        f"{kl}{early}{refill}"
    )


def _format_inference(inference: Mapping[str, Any]) -> str:
    if not inference:
        return "inference: no summary yet"
    last_sync = _mapping(inference.get("last_sync"))
    current_weight_version = (
        inference.get("current_weight_version")
        or last_sync.get("current_weight_version")
        or last_sync.get("loaded_weight_version")
    )
    stage_fraction = _mapping(inference.get("server_stage_fraction"))
    bucket_waste = _mapping(inference.get("bucket_padding_waste_fraction"))
    snapshot_pool = _mapping(last_sync.get("snapshot_pool"))
    resident_versions = snapshot_pool.get("resident_versions", ())
    leases_by_version = _mapping(snapshot_pool.get("leases_by_version"))
    deferred = last_sync.get("deferred_weight_version")
    deferred_reason = last_sync.get("deferred_weight_reason")
    gap = last_sync.get("snapshot_min_version_gap")
    snapshot_status = ""
    if snapshot_pool:
        snapshot_status = (
            f" resident={resident_versions}"
            f" leases={dict(leases_by_version)}"
            f" deferred={deferred}({deferred_reason or '-'})"
            f" gap={gap}"
        )
    return (
        "inference: "
        f"{_to_int(inference.get('decisions'))} decisions "
        f"{_to_float(inference.get('decisions_per_second')):.1f} dec/s "
        f"batches={_to_int(inference.get('policy_batches'))} "
        f"p50_batch={_to_float(inference.get('policy_batch_p50')):.0f} "
        f"p95_latency={_to_float(inference.get('request_latency_p95_ms')):.1f}ms "
        f"sample={100.0 * _to_float(stage_fraction.get('sample')):.1f}% "
        f"shape_top={_top_count_label(_mapping(inference.get('shape_histogram')))} "
        f"bucket_top={_top_count_label(_mapping(inference.get('bucket_histogram')))} "
        f"bucket_pad={100.0 * _to_float(bucket_waste.get('cross')):.1f}% "
        f"svc_ms_top={_top_count_label(_mapping(inference.get('service_time_histogram_ms')))} "
        f"current_v={current_weight_version}"
        f"{snapshot_status}"
    )


def _format_runtime(
    runtime: Mapping[str, Any],
    last_sample: Mapping[str, Any],
) -> str:
    if not runtime:
        return "runtime: no monitor sample yet"
    queues = _mapping(last_sample.get("queues"))
    gpu = _mapping(last_sample.get("gpu"))
    power_watts = _optional_float(gpu.get("power_watts"))
    utilization = _optional_float(gpu.get("utilization_percent"))
    gpu_power = f" gpu_power={power_watts:.0f}W" if power_watts is not None else ""
    gpu_util = f" gpu_util={utilization:.0f}%" if utilization is not None else ""
    return (
        "runtime: "
        f"samples={_to_int(runtime.get('sample_count'))} "
        f"elapsed={_fmt_seconds(last_sample.get('elapsed_seconds'))} "
        f"q.traj={queues.get('trajectory')} "
        f"q.inf_req={queues.get('inference_request')} "
        f"gpu_used={_to_float(gpu.get('memory_used_mb')) / 1000.0:.1f}GB "
        f"gpu_max={_to_float(runtime.get('max_gpu_memory_used_mb')) / 1000.0:.1f}GB"
        f"{gpu_power}{gpu_util}"
    )


def _format_gpu_phases(runtime: Mapping[str, Any]) -> str:
    phase_summary = _mapping(runtime.get("gpu_phase_summary"))
    if not phase_summary:
        return ""
    parts = []
    for phase, raw_stats in sorted(phase_summary.items()):
        stats = _mapping(raw_stats)
        parts.append(
            f"{phase}:occ={100.0 * _to_float(stats.get('gpu_power_occupancy_fraction')):.1f}% "
            f"util={_to_float(stats.get('mean_utilization_percent')):.1f}% "
            f"power={_to_float(stats.get('mean_power_watts')):.0f}W "
            f"inf={_to_float(stats.get('inference_decisions_per_second')):.1f}/s"
        )
    return "gpu_by_phase: " + "; ".join(parts)


def _format_processes(last_sample: Mapping[str, Any]) -> str:
    if not last_sample:
        return ""
    actors = _sequence(last_sample.get("actors"))
    alive_actors = sum(1 for actor in actors if _mapping(actor).get("exitcode") is None)
    learner = _mapping(last_sample.get("learner"))
    inference = _mapping(last_sample.get("inference"))
    actors_cpu = sum(_to_float(_mapping(actor).get("cpu_percent")) for actor in actors)
    return (
        "processes: "
        f"actors_alive={alive_actors}/{len(actors)} "
        f"learner_exit={learner.get('exitcode')} "
        f"inference_exit={inference.get('exitcode')} "
        f"actors_cpu={actors_cpu:.1f}% "
        f"learner_cpu={_to_float(learner.get('cpu_percent')):.1f}% "
        f"inference_cpu={_to_float(inference.get('cpu_percent')):.1f}%"
    )


def _format_actors(actors: Mapping[str, Any]) -> str:
    if not actors:
        return "actors: no live summaries yet"
    stage_timings = _mapping(actors.get("rollout_stage_timings"))
    policy_wait = _mapping(stage_timings.get("rollout_policy_wait"))
    encode = _mapping(stage_timings.get("rollout_encode_collate"))
    elapsed = _to_float(actors.get("actor_elapsed_seconds"))
    policy_wait_fraction = (
        _to_float(policy_wait.get("seconds")) / elapsed if elapsed > 0.0 else 0.0
    )
    encode_fraction = (
        _to_float(encode.get("seconds")) / elapsed if elapsed > 0.0 else 0.0
    )
    rollout_features = _mapping(actors.get("rollout_features"))
    live_steps = ""
    if "live_game_count" in rollout_features:
        oldest_rows = _sequence(rollout_features.get("oldest_live_games"))
        oldest = _mapping(oldest_rows[0]) if oldest_rows else {}
        oldest_label = "-"
        if oldest:
            oldest_label = (
                f"a{_to_int(oldest.get('actor_index'))}/"
                f"{oldest.get('game_id')}:{_to_int(oldest.get('steps'))}"
            )
        live_steps = (
            " "
            f"live_steps=count={_to_int(rollout_features.get('live_game_count'))} "
            f"max={_to_int(rollout_features.get('live_game_steps_max'))} "
            f"peak={_to_int(rollout_features.get('max_live_game_steps_seen'))} "
            f"oldest={oldest_label}"
        )
    stale_recurrent = ""
    if "stale_recurrent_recycle_polls" in rollout_features:
        stale_recurrent = (
            " "
            "stale_recurrent="
            f"recycled={_to_int(rollout_features.get('stale_recurrent_games_recycled'))} "
            "deferred="
            f"{_to_int(rollout_features.get('stale_recurrent_games_deferred_pending_evidence'))} "
            "sequences="
            f"{_to_int(rollout_features.get('stale_recurrent_sequences_released'))} "
            "versions="
            f"{_to_int(rollout_features.get('stale_recurrent_last_learner_version'))}/"
            f"{_to_int(rollout_features.get('stale_recurrent_last_served_version'))} "
            "max_age="
            f"{_to_int(rollout_features.get('stale_recurrent_max_candidate_version_age'))}"
        )
    return (
        "actors: "
        f"summaries={_to_int(actors.get('actors'))} "
        f"queued={_to_int(actors.get('queued_trajectories'))} "
        f"decisions={_to_int(actors.get('recorded_decisions'))} "
        f"q_full={_to_int(actors.get('queue_full_retries'))} "
        f"policy_wait={_fmt_seconds(policy_wait.get('seconds'))}"
        f"({100.0 * policy_wait_fraction:.1f}%) "
        f"encode={_fmt_seconds(encode.get('seconds'))}"
        f"({100.0 * encode_fraction:.1f}%)"
        f"{live_steps}"
        f"{stale_recurrent}"
    )


def _format_curriculum(curriculum: Mapping[str, Any]) -> str:
    if not curriculum:
        return "curriculum: no central summary yet"
    assigned = _mapping(curriculum.get("assigned"))
    finished = _mapping(curriculum.get("finished"))
    summary = (
        "curriculum: "
        f"status={curriculum.get('status')} "
        f"frozen_members={curriculum.get('frozen_members')} "
        f"assigned={dict(assigned)} "
        f"finished={dict(finished)}"
    )
    deck_summary = _format_curriculum_decks(_mapping(curriculum.get("decks")))
    if deck_summary:
        return f"{summary}\n{deck_summary}"
    return summary


def _format_curriculum_decks(decks: Mapping[str, Any]) -> str:
    if not decks:
        return ""
    rows = ["curriculum_decks:"]
    for label, raw_stats in sorted(decks.items()):
        stats = _mapping(raw_stats)
        rows.append(
            "  "
            f"{label}: "
            f"assigned={_to_int(stats.get('assigned'))} "
            f"actual={100.0 * _to_float(stats.get('assigned_fraction')):.1f}% "
            f"target={_format_optional_percent(_curriculum_target(stats))} "
            f"finished={_to_int(stats.get('finished'))} "
            f"wr={_format_optional_percent(stats.get('winrate'))} "
            f"ema={_format_optional_percent(stats.get('winrate_ema'))}"
        )
    return "\n".join(rows)


def _curriculum_target(stats: Mapping[str, Any]) -> Any:
    target = stats.get("target_probability")
    if target is not None:
        return target
    return stats.get("configured_probability")


def _format_final_summary(summary: Mapping[str, Any]) -> str:
    supervisor = _mapping(summary.get("supervisor"))
    learner = _mapping(summary.get("learner"))
    throughput = _mapping(summary.get("throughput"))
    all_throughput = _mapping(throughput.get("all"))
    measured_throughput = _mapping(throughput.get("measured"))
    throughput_text = ""
    if all_throughput:
        throughput_text = (
            f" kept={_to_float(all_throughput.get('kept_decisions_per_second')):.1f}/s"
        )
    if measured_throughput:
        throughput_text += (
            " measured="
            f"{_to_float(measured_throughput.get('kept_decisions_per_second')):.1f}/s"
            f" warmup={_to_int(throughput.get('warmup_iterations'))}"
        )
    return (
        "summary: "
        f"learner_exit={supervisor.get('learner_exitcode')} "
        f"actor_restarts={supervisor.get('actor_restarts')} "
        f"published_v={learner.get('published_version')} "
        f"updates={learner.get('updates')}"
        f"{throughput_text}"
    )


def _run_status(
    summary: Mapping[str, Any],
    last_sample: Mapping[str, Any],
    *,
    runtime_update_age_seconds: float | None,
    runtime_write_interval_seconds: float | None,
    runtime_failure: Mapping[str, Any],
    learner_supervisor: Mapping[str, Any],
) -> str:
    supervisor_state = str(learner_supervisor.get("state") or "")
    if supervisor_state == "running":
        return "running"
    if supervisor_state == "backoff":
        return f"failed(learner={learner_supervisor.get('learner_exitcode')})"
    if supervisor_state == "failed":
        return f"failed(learner={learner_supervisor.get('learner_exitcode')})"
    if supervisor_state == "completed":
        return "completed"
    if supervisor_state == "stopped":
        return f"stopped({learner_supervisor.get('learner_exitcode')})"
    supervisor = _mapping(summary.get("supervisor"))
    if supervisor:
        learner_exit = supervisor.get("learner_exitcode")
        return "completed" if learner_exit == 0 else f"stopped({learner_exit})"
    if runtime_failure:
        error_type = str(runtime_failure.get("error_type") or "unknown")
        return f"failed({error_type})"
    learner = _mapping(last_sample.get("learner"))
    inference = _mapping(last_sample.get("inference"))
    learner_exit = learner.get("exitcode")
    inference_exit = inference.get("exitcode")
    if learner_exit is not None:
        return f"failed(learner={learner_exit})"
    if inference and inference_exit is not None:
        return f"failed(inference={inference_exit})"
    stale_after = _DEFAULT_RUNTIME_STALE_SECONDS
    if runtime_write_interval_seconds is not None:
        stale_after = max(stale_after, 3.0 * runtime_write_interval_seconds)
    if (
        runtime_update_age_seconds is not None
        and runtime_update_age_seconds > stale_after
    ):
        return f"stale({_fmt_seconds(runtime_update_age_seconds)})"
    if learner and learner_exit is None:
        return "running"
    return "unknown"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return cast(dict[str, Any], raw)


def _build_actor_status(run_dir: Path) -> dict[str, Any]:
    summaries = tuple(
        summary
        for path in _actor_summary_paths(run_dir)
        if (summary := _read_json(path))
    )
    if not summaries:
        return {}
    totals = Counter[str]()
    actor_elapsed_seconds = 0.0
    rollout_stage_seconds: dict[str, float] = {}
    rollout_stage_counts: dict[str, int] = {}
    saw_live_game_steps = False
    live_game_count = 0
    live_game_steps_max = 0
    max_live_game_steps_seen = 0
    oldest_live_games: list[dict[str, Any]] = []
    saw_stale_recurrent_recycling = False
    stale_recurrent_totals = Counter[str]()
    stale_recurrent_last_learner_version = -1
    stale_recurrent_last_served_version = -1
    stale_recurrent_oldest_candidate_policy_version: int | None = None
    stale_recurrent_max_candidate_version_age = 0
    for fallback_actor_index, summary in enumerate(summaries):
        stats = _mapping(summary.get("stats"))
        totals["queued_trajectories"] += _to_int(stats.get("queued_trajectories"))
        totals["queue_full_retries"] += _to_int(stats.get("queue_full_retries"))
        totals["recorded_decisions"] += _to_int(stats.get("recorded_decisions"))
        totals["finished_games"] += _to_int(stats.get("finished_games"))
        actor_elapsed_seconds += _to_float(stats.get("elapsed_seconds"))
        _accumulate_stage_timings(
            _mapping(stats.get("stage_timings")),
            seconds_by_stage=rollout_stage_seconds,
            counts_by_stage=rollout_stage_counts,
        )
        rollout_features = _mapping(stats.get("rollout_features"))
        if "live_game_count" in rollout_features:
            saw_live_game_steps = True
            live_game_count += max(
                0,
                _to_int(rollout_features.get("live_game_count")),
            )
            live_game_steps_max = max(
                live_game_steps_max,
                _to_int(rollout_features.get("live_game_steps_max")),
            )
            max_live_game_steps_seen = max(
                max_live_game_steps_seen,
                _to_int(rollout_features.get("max_live_game_steps_seen")),
            )
            actor_index = _to_int(summary.get("actor_index", fallback_actor_index))
            for raw_game in _sequence(rollout_features.get("oldest_live_games")):
                game = _mapping(raw_game)
                if not game:
                    continue
                oldest_live_games.append(
                    {
                        "actor_index": actor_index,
                        "game_id": str(game.get("game_id", "")),
                        "steps": max(0, _to_int(game.get("steps"))),
                    }
                )
        if "stale_recurrent_recycle_polls" in rollout_features:
            saw_stale_recurrent_recycling = True
            for key in (
                "stale_recurrent_recycle_polls",
                "stale_recurrent_candidate_sequences_examined",
                "stale_recurrent_games_recycled",
                "stale_recurrent_games_deferred_pending_evidence",
                "stale_recurrent_sequences_released",
            ):
                stale_recurrent_totals[key] += _to_int(rollout_features.get(key))
            stale_recurrent_last_learner_version = max(
                stale_recurrent_last_learner_version,
                _to_int(rollout_features.get("stale_recurrent_last_learner_version")),
            )
            stale_recurrent_last_served_version = max(
                stale_recurrent_last_served_version,
                _to_int(rollout_features.get("stale_recurrent_last_served_version")),
            )
            oldest_candidate_version = _to_int(
                rollout_features.get("stale_recurrent_oldest_candidate_policy_version")
            )
            if oldest_candidate_version >= 0:
                stale_recurrent_oldest_candidate_policy_version = (
                    oldest_candidate_version
                    if stale_recurrent_oldest_candidate_policy_version is None
                    else min(
                        stale_recurrent_oldest_candidate_policy_version,
                        oldest_candidate_version,
                    )
                )
            stale_recurrent_max_candidate_version_age = max(
                stale_recurrent_max_candidate_version_age,
                _to_int(
                    rollout_features.get("stale_recurrent_max_candidate_version_age")
                ),
            )
    oldest_live_games.sort(
        key=lambda row: (
            -int(row["steps"]),
            int(row["actor_index"]),
            str(row["game_id"]),
        )
    )
    actor_status = {
        "actors": len(summaries),
        "queued_trajectories": totals["queued_trajectories"],
        "queue_full_retries": totals["queue_full_retries"],
        "recorded_decisions": totals["recorded_decisions"],
        "finished_games": totals["finished_games"],
        "actor_elapsed_seconds": actor_elapsed_seconds,
        "rollout_stage_timings": _stage_timing_summary(
            rollout_stage_seconds,
            rollout_stage_counts,
        ),
    }
    aggregated_rollout_features: dict[str, Any] = {}
    if saw_live_game_steps:
        aggregated_rollout_features.update(
            {
                "live_game_count": live_game_count,
                "live_game_steps_max": live_game_steps_max,
                "max_live_game_steps_seen": max_live_game_steps_seen,
                "oldest_live_games": oldest_live_games[:_OLDEST_LIVE_GAMES_LIMIT],
            }
        )
    if saw_stale_recurrent_recycling:
        aggregated_rollout_features.update(stale_recurrent_totals)
        aggregated_rollout_features.update(
            {
                "stale_recurrent_last_learner_version": (
                    stale_recurrent_last_learner_version
                ),
                "stale_recurrent_last_served_version": (
                    stale_recurrent_last_served_version
                ),
                "stale_recurrent_oldest_candidate_policy_version": (
                    -1
                    if stale_recurrent_oldest_candidate_policy_version is None
                    else stale_recurrent_oldest_candidate_policy_version
                ),
                "stale_recurrent_max_candidate_version_age": (
                    stale_recurrent_max_candidate_version_age
                ),
            }
        )
    if aggregated_rollout_features:
        actor_status["rollout_features"] = aggregated_rollout_features
    return actor_status


def _actor_summary_paths(run_dir: Path) -> tuple[Path, ...]:
    paths = sorted(run_dir.glob("actor_summary*.json"))
    return tuple(path for path in paths if path.name.startswith("actor_summary"))


def _accumulate_stage_timings(
    stage_timings: Mapping[str, Any],
    *,
    seconds_by_stage: dict[str, float],
    counts_by_stage: dict[str, int],
) -> None:
    for raw_stage, raw_timing in stage_timings.items():
        if not isinstance(raw_stage, str) or not isinstance(raw_timing, Mapping):
            continue
        seconds_by_stage[raw_stage] = seconds_by_stage.get(raw_stage, 0.0) + _to_float(
            raw_timing.get("seconds")
        )
        counts_by_stage[raw_stage] = counts_by_stage.get(raw_stage, 0) + _to_int(
            raw_timing.get("count")
        )


def _stage_timing_summary(
    seconds_by_stage: Mapping[str, float],
    counts_by_stage: Mapping[str, int],
) -> dict[str, dict[str, float | int]]:
    return {
        stage: {
            "seconds": seconds,
            "count": counts_by_stage.get(stage, 0),
            "mean_ms": (
                1000.0 * seconds / float(counts_by_stage.get(stage, 0))
                if counts_by_stage.get(stage, 0) > 0
                else 0.0
            ),
        }
        for stage, seconds in sorted(seconds_by_stage.items())
    }


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _path_age_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def _fmt_seconds(value: Any) -> str:
    seconds = _to_float(value)
    if seconds <= 0.0:
        return "n/a"
    if seconds < 120.0:
        return f"{seconds:.0f}s"
    if seconds < 7200.0:
        return f"{seconds / 60.0:.1f}m"
    return f"{seconds / 3600.0:.1f}h"


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_optional_int(value: Any) -> str:
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "-"


def _format_optional_percent(value: Any) -> str:
    number = _optional_float(value)
    if number is None:
        return "-"
    return f"{100.0 * number:.1f}%"


def _top_count_label(histogram: Mapping[str, Any]) -> str:
    if not histogram:
        return "-"
    key, value = max(histogram.items(), key=lambda item: _to_int(item[1]))
    return f"{key}:{_to_int(value)}"


if __name__ == "__main__":
    main()
