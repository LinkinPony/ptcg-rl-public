"""Learner-empty async actor and inference-server benchmark."""

from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Any, cast

import torch.multiprocessing as torch_mp
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.rl.experience import GameTrajectory
from ptcg_rl.rl.training import (
    RLTrainConfig,
    _actor_queue_summary,
    _async_actor_worker,
    _async_curriculum_worker,
    _async_inference_worker,
    _inference_summary_path,
    _read_optional_summary,
    _stop_managed_process,
    _uses_central_curriculum,
    resolve_rl_train_output_dir,
)


class ActorInferenceBenchmarkConfig(BaseModel):
    """Config for running async actors against the central inference service."""

    model_config = ConfigDict(extra="forbid")

    duration_seconds: float = 60.0
    target_inference_decisions: int | None = None
    drain_timeout_seconds: float = 0.01
    output_path: Path | None = None

    @field_validator("duration_seconds", "drain_timeout_seconds")
    @classmethod
    def valid_positive_float(cls, value: float) -> float:
        """Reject invalid benchmark timing settings."""
        if value <= 0.0:
            raise ValueError("benchmark timing settings must be positive")
        return value

    @field_validator("target_inference_decisions")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject invalid optional decision target."""
        if value is not None and value <= 0:
            raise ValueError("target_inference_decisions must be positive when set")
        return value


def run_actor_inference_benchmark(
    train_config: RLTrainConfig,
    benchmark_config: ActorInferenceBenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run async actors plus inference server without a learner process."""
    bench = benchmark_config or ActorInferenceBenchmarkConfig()
    config = _benchmark_train_config(train_config)
    output_dir = deck_records.repo_path(resolve_rl_train_output_dir(config))
    output_dir.mkdir(parents=True, exist_ok=True)

    context = torch_mp.get_context("spawn")
    trajectory_queue = context.Queue(maxsize=config.execution.trajectory_queue_maxsize)
    inference_request_queue = context.Queue()
    inference_response_queues = [
        context.Queue() for _index in range(config.execution.actors)
    ]
    central_curriculum = _uses_central_curriculum(config)
    curriculum_request_queue = context.Queue() if central_curriculum else None
    curriculum_response_queues = (
        [context.Queue() for _index in range(config.execution.actors)]
        if central_curriculum
        else None
    )
    config_data = config.model_dump(mode="python")

    inference_process = cast(
        Any,
        context.Process(
            target=_async_inference_worker,
            kwargs={
                "config_data": config_data,
                "output_dir": str(output_dir),
                "request_queue": inference_request_queue,
                "response_queues": inference_response_queues,
            },
        ),
    )
    actor_processes = [
        cast(
            Any,
            context.Process(
                target=_async_actor_worker,
                kwargs={
                    "config_data": config_data,
                    "output_dir": str(output_dir),
                    "trajectory_queue": trajectory_queue,
                    "pool_factory": None,
                    "actor_index": actor_index,
                    "inference_request_queue": inference_request_queue,
                    "inference_response_queue": inference_response_queues[actor_index],
                    "curriculum_request_queue": curriculum_request_queue,
                    "curriculum_response_queue": (
                        None
                        if curriculum_response_queues is None
                        else curriculum_response_queues[actor_index]
                    ),
                },
            ),
        )
        for actor_index in range(config.execution.actors)
    ]
    curriculum_process = (
        cast(
            Any,
            context.Process(
                target=_async_curriculum_worker,
                kwargs={
                    "config_data": config_data,
                    "output_dir": str(output_dir),
                    "request_queue": curriculum_request_queue,
                    "response_queues": curriculum_response_queues,
                },
            ),
        )
        if central_curriculum
        else None
    )

    started = time.perf_counter()
    drained_trajectories = 0
    drained_decisions = 0
    if curriculum_process is not None:
        curriculum_process.start()
    inference_process.start()
    for process in actor_processes:
        process.start()
    try:
        while True:
            drained = _drain_trajectory_queue(
                trajectory_queue,
                timeout_seconds=bench.drain_timeout_seconds,
            )
            drained_trajectories += len(drained)
            drained_decisions += sum(
                trajectory.decision_count for trajectory in drained
            )
            elapsed_seconds = time.perf_counter() - started
            if elapsed_seconds >= bench.duration_seconds:
                break
            if _target_reached(output_dir, bench.target_inference_decisions):
                break
            if not inference_process.is_alive():
                break
            if any(process.exitcode not in (None, 0) for process in actor_processes):
                break
    finally:
        for process in actor_processes:
            _stop_managed_process(
                process,
                join_timeout_seconds=(
                    config.execution.async_supervisor.process_join_timeout_seconds
                ),
            )
        _stop_managed_process(
            inference_process,
            join_timeout_seconds=(
                config.execution.async_supervisor.process_join_timeout_seconds
            ),
        )
        if curriculum_process is not None:
            _stop_managed_process(
                curriculum_process,
                join_timeout_seconds=(
                    config.execution.async_supervisor.process_join_timeout_seconds
                ),
            )

    elapsed_seconds = time.perf_counter() - started
    inference_summary = _read_optional_summary(_inference_summary_path(output_dir))
    if inference_summary is not None:
        inference_summary["process_exitcode"] = inference_process.exitcode
    summary = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "actor_inference_benchmark",
        "output_dir": deck_records.display_path(output_dir),
        "elapsed_seconds": elapsed_seconds,
        "drained_trajectories": drained_trajectories,
        "drained_decisions": drained_decisions,
        "drained_decisions_per_second": (
            drained_decisions / elapsed_seconds if elapsed_seconds > 0.0 else 0.0
        ),
        "actor_queue": _actor_queue_summary(
            output_dir,
            actor_count=config.execution.actors,
        ),
        "inference": inference_summary,
        "processes": {
            "actor_exitcodes": [process.exitcode for process in actor_processes],
            "inference_exitcode": inference_process.exitcode,
            "curriculum_exitcode": (
                None if curriculum_process is None else curriculum_process.exitcode
            ),
        },
        "config": {
            "benchmark": bench.model_dump(mode="json"),
            "train": config.model_dump(mode="json"),
        },
    }
    output_path = bench.output_path or output_dir / "actor_inference_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return summary


def _benchmark_train_config(config: RLTrainConfig) -> RLTrainConfig:
    execution = config.execution.model_copy(
        update={"mode": "async", "inference_server": True}
    )
    return config.model_copy(update={"execution": execution})


def _drain_trajectory_queue(
    trajectory_queue: Any,
    *,
    timeout_seconds: float,
) -> tuple[GameTrajectory, ...]:
    drained: list[GameTrajectory] = []
    while True:
        try:
            item = trajectory_queue.get(timeout=timeout_seconds if not drained else 0.0)
        except queue.Empty:
            break
        if not isinstance(item, GameTrajectory):
            raise TypeError(f"expected GameTrajectory, got {type(item).__name__}")
        drained.append(item)
    return tuple(drained)


def _target_reached(output_dir: Path, target_decisions: int | None) -> bool:
    if target_decisions is None:
        return False
    summary = _read_inference_summary(output_dir)
    if summary is None:
        return False
    return int(summary.get("decisions", 0)) >= target_decisions


def _read_inference_summary(output_dir: Path) -> dict[str, Any] | None:
    path = _inference_summary_path(output_dir)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, dict):
        return None
    return raw
