"""Remote actor+inference worker runner for distributed async training."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import FrameType
from typing import Any, cast

import torch
import torch.multiprocessing as torch_mp

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.rl import training as training_module
from ptcg_rl.rl.async_runtime import ManagedProcess
from ptcg_rl.rl.distributed.client import (
    DistributedTrajectorySender,
    DistributedTrajectorySenderConfig,
    DistributedWeightClient,
    DistributedWeightClientConfig,
)
from ptcg_rl.rl.distributed.compatibility import DistributedModelCompatibility
from ptcg_rl.rl.training import RLTrainConfig, resolve_rl_train_output_dir


def distributed_worker_output_dir(root_output_dir: Path, worker_id: str) -> Path:
    """Derive the sole artifact directory for a distributed worker."""
    return root_output_dir / "distributed_workers" / worker_id


def run_distributed_actor_worker(config: RLTrainConfig) -> dict[str, Any]:
    """Run a remote actor+inference worker for distributed async training."""
    config = training_module._resolve_training_model_config(config)
    if not config.distributed.worker_enabled:
        raise ValueError("distributed.worker_enabled must be true for worker mode")
    if config.execution.actors <= 0:
        raise ValueError("distributed workers require at least one actor")
    torch.manual_seed(config.seed + 10_000)
    context = torch_mp.get_context("spawn")
    root_output_dir = deck_records.repo_path(resolve_rl_train_output_dir(config))
    output_dir = distributed_worker_output_dir(
        root_output_dir,
        config.distributed.worker_id,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    training_module._maybe_seed_curriculum_anchor(config, output_dir=output_dir)
    config_data = config.model_dump(mode="python")
    transport = config.distributed.transport
    trajectory_queue = context.Queue(maxsize=config.execution.trajectory_queue_maxsize)
    compatibility = DistributedModelCompatibility.from_model_config(config.model)
    weight_client = DistributedWeightClient(
        DistributedWeightClientConfig(
            endpoint=_endpoint(transport.coordinator_host, transport.weight_port),
            worker_id=config.distributed.worker_id,
            weights_dir=output_dir / "weights",
            poll_interval_seconds=transport.weight_poll_interval_seconds,
            summary_path=output_dir / "distributed_weight_summary.json",
            compatibility=compatibility,
        )
    )
    sender = DistributedTrajectorySender(
        trajectory_queue,
        DistributedTrajectorySenderConfig(
            endpoint=_endpoint(transport.coordinator_host, transport.trajectory_port),
            worker_id=config.distributed.worker_id,
            batch_decisions=transport.trajectory_batch_decisions,
            flush_interval_seconds=transport.trajectory_flush_interval_seconds,
            flow_control_sleep_seconds=transport.flow_control_sleep_seconds,
            summary_path=output_dir / "distributed_sender_summary.json",
            compatibility=compatibility,
        ),
        flow_control_provider=weight_client.flow_control,
    )
    weight_client.poll_once()
    weight_client.start()
    sender.start()

    inference_request_queue = context.Queue() if config.execution.inference_server else None
    inference_response_queues = (
        [context.Queue() for _index in range(config.execution.actors)]
        if config.execution.inference_server
        else None
    )
    central_curriculum = training_module._uses_central_curriculum(config)
    curriculum_request_queue = context.Queue() if central_curriculum else None
    curriculum_response_queues = (
        [context.Queue() for _index in range(config.execution.actors)]
        if central_curriculum
        else None
    )
    curriculum_process = _maybe_start_curriculum_process(
        config=config,
        config_data=config_data,
        output_dir=str(output_dir),
        context=context,
        request_queue=curriculum_request_queue,
        response_queues=curriculum_response_queues,
    )
    inference_process = _maybe_start_inference_process(
        config=config,
        config_data=config_data,
        output_dir=str(output_dir),
        context=context,
        request_queue=inference_request_queue,
        response_queues=inference_response_queues,
    )
    actors = _start_actor_processes(
        config=config,
        config_data=config_data,
        output_dir=str(output_dir),
        context=context,
        trajectory_queue=trajectory_queue,
        inference_request_queue=inference_request_queue,
        inference_response_queues=inference_response_queues,
        curriculum_request_queue=curriculum_request_queue,
        curriculum_response_queues=curriculum_response_queues,
    )
    started = time.perf_counter()
    restarts = [0 for _index in actors]
    recycles = [0 for _index in actors]
    polls = 0
    curriculum_summary: dict[str, Any] | None = None
    stop_event = threading.Event()
    previous_signal_handlers = _install_stop_signal_handlers(stop_event)
    try:
        polls = _monitor_actor_processes(
            actors,
            restarts=restarts,
            recycles=recycles,
            config=config,
            should_stop=stop_event.is_set,
            actor_factory=lambda index: _actor_process(
                config_data=config_data,
                output_dir=str(output_dir),
                context=context,
                actor_index=index,
                trajectory_queue=trajectory_queue,
                inference_request_queue=inference_request_queue,
                inference_response_queue=(
                    None
                    if inference_response_queues is None
                    else inference_response_queues[index]
                ),
                curriculum_request_queue=curriculum_request_queue,
                curriculum_response_queue=(
                    None
                    if curriculum_response_queues is None
                    else curriculum_response_queues[index]
                ),
            ),
        )
    finally:
        try:
            curriculum_summary = _stop_worker_children(
                config=config,
                output_dir=output_dir,
                actors=actors,
                inference_process=inference_process,
                curriculum_process=curriculum_process,
                curriculum_request_queue=curriculum_request_queue,
            )
        finally:
            try:
                sender.close()
                weight_client.close()
            finally:
                _restore_signal_handlers(previous_signal_handlers)
    summary = {
        "status": "stopped" if stop_event.is_set() else "completed",
        "mode": "distributed_actor_worker",
        "worker_id": config.distributed.worker_id,
        "output_dir": deck_records.display_path(output_dir),
        "elapsed_seconds": time.perf_counter() - started,
        "polls": polls,
        "actor_restarts_by_actor": restarts,
        "actor_recycles_by_actor": recycles,
        "sender": sender.summary(),
        "weight_client": weight_client.summary(),
        "curriculum": curriculum_summary,
    }
    training_module._write_summary(output_dir / "distributed_worker_summary.json", summary)
    return summary


def _maybe_start_curriculum_process(
    *,
    config: RLTrainConfig,
    config_data: dict[str, Any],
    output_dir: str,
    context: Any,
    request_queue: Any | None,
    response_queues: Sequence[Any] | None,
) -> ManagedProcess | None:
    if not training_module._uses_central_curriculum(config):
        return None
    if request_queue is None or response_queues is None:
        raise RuntimeError("curriculum queues were not initialized")
    process = cast(
        ManagedProcess,
        context.Process(
            target=training_module._async_curriculum_worker,
            kwargs={
                "config_data": config_data,
                "output_dir": output_dir,
                "request_queue": request_queue,
                "response_queues": response_queues,
            },
        ),
    )
    process.start()
    return process


def _maybe_start_inference_process(
    *,
    config: RLTrainConfig,
    config_data: dict[str, Any],
    output_dir: str,
    context: Any,
    request_queue: Any | None,
    response_queues: Sequence[Any] | None,
) -> ManagedProcess | None:
    if not config.execution.inference_server:
        return None
    if request_queue is None or response_queues is None:
        raise RuntimeError("inference queues were not initialized")
    process = cast(
        ManagedProcess,
        context.Process(
            target=training_module._async_inference_worker,
            kwargs={
                "config_data": config_data,
                "output_dir": output_dir,
                "request_queue": request_queue,
                "response_queues": response_queues,
            },
        ),
    )
    process.start()
    return process


def _start_actor_processes(
    *,
    config: RLTrainConfig,
    config_data: dict[str, Any],
    output_dir: str,
    context: Any,
    trajectory_queue: Any,
    inference_request_queue: Any | None,
    inference_response_queues: Sequence[Any] | None,
    curriculum_request_queue: Any | None,
    curriculum_response_queues: Sequence[Any] | None,
) -> list[ManagedProcess]:
    actors = [
        _actor_process(
            config_data=config_data,
            output_dir=output_dir,
            context=context,
            actor_index=index,
            trajectory_queue=trajectory_queue,
            inference_request_queue=inference_request_queue,
            inference_response_queue=(
                None if inference_response_queues is None else inference_response_queues[index]
            ),
            curriculum_request_queue=curriculum_request_queue,
            curriculum_response_queue=(
                None
                if curriculum_response_queues is None
                else curriculum_response_queues[index]
            ),
        )
        for index in range(config.execution.actors)
    ]
    for actor in actors:
        actor.start()
    return actors


def _actor_process(
    *,
    config_data: dict[str, Any],
    output_dir: str,
    context: Any,
    actor_index: int,
    trajectory_queue: Any,
    inference_request_queue: Any | None,
    inference_response_queue: Any | None,
    curriculum_request_queue: Any | None,
    curriculum_response_queue: Any | None,
) -> ManagedProcess:
    return cast(
        ManagedProcess,
        context.Process(
            target=training_module._async_actor_worker,
            kwargs={
                "config_data": config_data,
                "output_dir": output_dir,
                "trajectory_queue": trajectory_queue,
                "pool_factory": None,
                "actor_index": actor_index,
                "inference_request_queue": inference_request_queue,
                "inference_response_queue": inference_response_queue,
                "curriculum_request_queue": curriculum_request_queue,
                "curriculum_response_queue": curriculum_response_queue,
            },
        ),
    )


def _monitor_actor_processes(
    actors: list[ManagedProcess],
    *,
    restarts: list[int],
    recycles: list[int],
    config: RLTrainConfig,
    actor_factory: Any,
    should_stop: Callable[[], bool] | None = None,
) -> int:
    polls = 0
    stop_requested = should_stop or (lambda: False)
    while True:
        if stop_requested():
            return polls
        time.sleep(config.execution.async_supervisor.poll_interval_seconds)
        polls += 1
        if stop_requested():
            return polls
        for index, actor in enumerate(tuple(actors)):
            if actor.exitcode is None:
                continue
            if actor.exitcode == 0:
                recycles[index] += 1
                replacement = actor_factory(index)
                replacement.start()
                actors[index] = replacement
                continue
            if restarts[index] >= config.execution.async_supervisor.actor_restart_limit:
                raise RuntimeError(
                    f"distributed actor {index} exited with code {actor.exitcode} "
                    "and exceeded failure restart limit"
                )
            restarts[index] += 1
            replacement = actor_factory(index)
            replacement.start()
            actors[index] = replacement
        if (
            config.execution.async_supervisor.max_polls is not None
            and polls >= config.execution.async_supervisor.max_polls
        ):
            return polls


def _install_stop_signal_handlers(
    stop_event: threading.Event,
) -> dict[signal.Signals, Any]:
    """Translate process termination signals into orderly worker shutdown."""
    if threading.current_thread() is not threading.main_thread():
        return {}

    def request_stop(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    previous: dict[signal.Signals, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_stop)
    return previous


def _restore_signal_handlers(
    previous: Mapping[signal.Signals, Any],
) -> None:
    """Restore host signal behavior after the worker has stopped its children."""
    if threading.current_thread() is not threading.main_thread():
        return
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _stop_worker_children(
    *,
    config: RLTrainConfig,
    output_dir: Path,
    actors: Sequence[ManagedProcess],
    inference_process: ManagedProcess | None,
    curriculum_process: ManagedProcess | None,
    curriculum_request_queue: Any | None,
) -> dict[str, Any] | None:
    """Stop producers, finalize curriculum state, then stop inference."""
    join_timeout = config.execution.async_supervisor.process_join_timeout_seconds
    _stop_processes(actors, join_timeout_seconds=join_timeout)
    curriculum_summary: dict[str, Any] | None = None
    if curriculum_process is not None:
        if curriculum_request_queue is None:
            raise RuntimeError("curriculum process is missing its request queue")
        curriculum_summary = training_module._finish_async_curriculum_worker(
            config,
            output_dir=output_dir,
            process=curriculum_process,
            request_queue=curriculum_request_queue,
        )
    _stop_processes((inference_process,), join_timeout_seconds=join_timeout)
    return curriculum_summary


def _stop_processes(
    processes: Sequence[ManagedProcess | None],
    *,
    join_timeout_seconds: float,
) -> None:
    for process in processes:
        if process is None:
            continue
        training_module._stop_managed_process(
            process,
            join_timeout_seconds=join_timeout_seconds,
        )


def _endpoint(host: str, port: int) -> str:
    return f"tcp://{host}:{int(port)}"
