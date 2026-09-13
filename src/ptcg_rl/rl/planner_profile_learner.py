"""Concurrent fixed full-objective learner-kernel execution on H200."""

from __future__ import annotations

import gc
import multiprocessing as mp
import queue
import time
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from typing import Any

import torch

from ptcg_rl.evaluation.planner_profile_config import IntegratedPlannerProfileConfig
from ptcg_rl.evaluation.planner_profile_workloads import PlannerProfileRuntimeConfig
from ptcg_rl.rl.planner_profile_inference import load_profile_model
from ptcg_rl.rl.planner_profile_learner_update import (
    ProfileKernelUpdateResult,
    build_profile_kernel_batch_config,
    build_profile_kernel_ppo_config,
    load_fixed_profile_anchor,
    run_profile_kernel_update,
    validate_fixed_kernel_contract,
)
from ptcg_rl.rl.planner_profile_learner_workload import (
    prepare_profile_learner_kernel,
)


@dataclass(frozen=True, slots=True)
class ProfileLearnerResult:
    """Measured fixed learner-kernel work from a discarded model copy.

    ``kernel_rows`` counts unique synthetic rows per outer kernel cycle, not
    on-policy decisions and not the two PPO epoch replays of each row. This is
    a contention microbenchmark; its one-step trajectories do not reproduce a
    production state distribution or multi-step GAE.
    """

    warmup_updates: int
    updates: int
    optimizer_steps: int
    kernel_rows: int
    elapsed_seconds: float
    peak_vram_bytes: int
    workload_fingerprint: str
    planner_rows_per_update: int
    root_value_rows_per_update: int
    unique_root_value_model_inputs_per_update: int

    @property
    def kernel_rows_per_second(self) -> float:
        """Return unique fixed-kernel rows completed per measured second."""
        if self.elapsed_seconds <= 0.0:
            return 0.0
        return self.kernel_rows / self.elapsed_seconds


@dataclass(slots=True)
class ConcurrentProfileLearner:
    """Spawn one fixed synthetic learner kernel beside the serving process."""

    process: BaseProcess
    start_event: Any
    result_queue: Any
    workload_fingerprint: str
    result: ProfileLearnerResult | None = None

    @classmethod
    def start(
        cls,
        *,
        config: IntegratedPlannerProfileConfig,
        runtime: PlannerProfileRuntimeConfig,
    ) -> ConcurrentProfileLearner:
        """Prepare the isolated discarded-model kernel and wait for readiness."""
        workload = runtime.learner
        if workload is None:
            raise ValueError("concurrent profile learner requires an H200 workload")
        context = mp.get_context("spawn")
        status_queue = context.Queue(maxsize=1)
        result_queue = context.Queue(maxsize=1)
        start_event = context.Event()
        process = context.Process(
            target=_learner_main,
            kwargs={
                "config": config,
                "runtime": runtime,
                "start_event": start_event,
                "status_queue": status_queue,
                "result_queue": result_queue,
            },
            name="planner-profile-learner-kernel",
        )
        process.start()
        try:
            status = status_queue.get(timeout=1800.0)
        except queue.Empty as exc:
            _terminate_process(process)
            raise TimeoutError("profile learner kernel did not become ready") from exc
        if not isinstance(status, Mapping) or status.get("status") != "ready":
            process.join(timeout=10.0)
            if process.is_alive():
                _terminate_process(process)
            detail = (
                status.get("traceback", status)
                if isinstance(status, Mapping)
                else status
            )
            raise RuntimeError(f"profile learner kernel failed to start: {detail}")
        fingerprint = str(status.get("workload_fingerprint", ""))
        if len(fingerprint) != 64:
            _terminate_process(process)
            raise RuntimeError("profile learner kernel omitted workload identity")
        return cls(
            process=process,
            start_event=start_event,
            result_queue=result_queue,
            workload_fingerprint=fingerprint,
        )

    def begin(self) -> None:
        """Release the prepared learner at deployment measurement start."""
        self.start_event.set()

    def finish(self) -> ProfileLearnerResult:
        """Wait for all fixed measured cycles and return their actual counters."""
        if self.result is not None:
            return self.result
        try:
            payload = self.result_queue.get(timeout=1800.0)
        except queue.Empty as exc:
            _terminate_process(self.process)
            raise TimeoutError("profile learner kernel did not finish") from exc
        self.process.join(timeout=30.0)
        if self.process.is_alive():
            _terminate_process(self.process)
            raise RuntimeError("profile learner kernel did not stop cleanly")
        if not isinstance(payload, Mapping) or payload.get("status") != "complete":
            detail = (
                payload.get("traceback", payload)
                if isinstance(payload, Mapping)
                else payload
            )
            raise RuntimeError(f"profile learner kernel failed: {detail}")
        self.result = ProfileLearnerResult(
            warmup_updates=int(payload["warmup_updates"]),
            updates=int(payload["updates"]),
            optimizer_steps=int(payload["optimizer_steps"]),
            kernel_rows=int(payload["kernel_rows"]),
            elapsed_seconds=float(payload["elapsed_seconds"]),
            peak_vram_bytes=int(payload["peak_vram_bytes"]),
            workload_fingerprint=str(payload["workload_fingerprint"]),
            planner_rows_per_update=int(payload["planner_rows_per_update"]),
            root_value_rows_per_update=int(payload["root_value_rows_per_update"]),
            unique_root_value_model_inputs_per_update=int(
                payload["unique_root_value_model_inputs_per_update"]
            ),
        )
        if self.result.workload_fingerprint != self.workload_fingerprint:
            raise RuntimeError("profile learner kernel changed after readiness")
        return self.result

    def close(self) -> None:
        """Terminate only an unfinished child during backend error cleanup."""
        if self.process.is_alive():
            _terminate_process(self.process)


def _learner_main(
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    start_event: Any,
    status_queue: Any,
    result_queue: Any,
) -> None:
    ready_sent = False
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("profile learner kernel has no visible CUDA device")
        workload = runtime.learner
        if workload is None:
            raise ValueError("profile learner kernel has no workload")
        validate_fixed_kernel_contract(workload)
        checkpoint = torch.load(
            config.checkpoint_path,
            map_location="cpu",
            mmap=True,
        )
        if not isinstance(checkpoint, Mapping):
            raise TypeError("profile checkpoint must be a mapping")
        training_payload_raw = checkpoint.get("training_config")
        if not isinstance(training_payload_raw, Mapping):
            raise ValueError("profile checkpoint has no training_config")
        training_payload = dict(training_payload_raw)
        source_update_index = _checkpoint_optimizer_update_index(checkpoint)
        from ptcg_rl.rl import training as training_module

        train_config = training_module.RLTrainConfig.model_validate(training_payload)
        del checkpoint, training_payload_raw
        gc.collect()

        training_module._configure_gpu_worker_runtime(train_config)
        learner_seed = int(train_config.seed) + 2
        torch.manual_seed(learner_seed)
        model = load_profile_model(
            config.checkpoint_path,
            device="cuda",
            expected_model_fingerprint=config.expected_model_fingerprint,
        )
        training_module._maybe_compile_learner_evaluate_actions(
            model,
            enabled=train_config.learner.compile_evaluate_actions,
        )
        anchor = load_fixed_profile_anchor(
            workload,
            campaign=config,
            device="cuda",
        )
        resolved = runtime.planner.resolve_for_lease(
            model_fingerprint=config.expected_model_fingerprint,
            policy_version=config.policy_version,
            proposal_version=config.proposal_version,
        )
        batch_config = build_profile_kernel_batch_config(
            train_config,
            workload=workload,
            resolved=resolved,
        )
        prepared = prepare_profile_learner_kernel(
            config=config,
            runtime=runtime,
            workload=workload,
            resolved=resolved,
            model=model,
            batch_config=batch_config,
        )
        training_module._reset_torch_random_stream(learner_seed)
        optimizer = training_module._build_ppo_optimizer(
            train_config,
            model=model,
            device=torch.device("cuda"),
        )
        planned_steps = _scheduler_horizon(
            train_config,
            source_update_index=source_update_index,
            kernel_steps=(workload.warmup_updates + workload.updates)
            * workload.ppo_epochs,
        )
        scheduler = training_module._build_lr_scheduler(
            optimizer,
            config=train_config,
            planned_steps=planned_steps,
        )
        training_module._set_lr_scheduler_progress(
            optimizer=optimizer,
            lr_scheduler=scheduler,
            completed_updates=source_update_index,
        )
        ppo_config = build_profile_kernel_ppo_config(training_payload, workload)
        model.train()
        update_index = source_update_index
        cycle_index = 0
        for _ in range(workload.warmup_updates):
            warmup = run_profile_kernel_update(
                model=model,
                anchor=anchor,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_result=prepared.batch_result,
                batch_config=batch_config,
                ppo_config=ppo_config,
                ppo_epochs=workload.ppo_epochs,
                policy_version=config.policy_version,
                kernel_cycle_index=cycle_index,
                first_update_index=update_index,
            )
            _validate_cycle_result(warmup, workload.ppo_epochs)
            update_index = warmup.next_update_index
            cycle_index += 1
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        status_queue.put(
            {
                "status": "ready",
                "workload_fingerprint": prepared.workload_fingerprint,
            }
        )
        ready_sent = True
        if not start_event.wait(timeout=1800.0):
            raise TimeoutError("profile learner kernel received no start signal")
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer_steps = 0
        kernel_rows = 0
        for _ in range(workload.updates):
            measured = run_profile_kernel_update(
                model=model,
                anchor=anchor,
                optimizer=optimizer,
                scheduler=scheduler,
                batch_result=prepared.batch_result,
                batch_config=batch_config,
                ppo_config=ppo_config,
                ppo_epochs=workload.ppo_epochs,
                policy_version=config.policy_version,
                kernel_cycle_index=cycle_index,
                first_update_index=update_index,
            )
            _validate_cycle_result(measured, workload.ppo_epochs)
            update_index = measured.next_update_index
            cycle_index += 1
            optimizer_steps += measured.optimizer_steps
            kernel_rows += workload.kernel_rows_per_update
        torch.cuda.synchronize()
        elapsed = max(time.perf_counter() - started, 1.0e-9)
        result_queue.put(
            {
                "status": "complete",
                "warmup_updates": workload.warmup_updates,
                "updates": workload.updates,
                "optimizer_steps": optimizer_steps,
                "kernel_rows": kernel_rows,
                "elapsed_seconds": elapsed,
                "peak_vram_bytes": torch.cuda.max_memory_reserved(),
                "workload_fingerprint": prepared.workload_fingerprint,
                "planner_rows_per_update": prepared.planner_rows,
                "root_value_rows_per_update": prepared.root_value_rows,
                "unique_root_value_model_inputs_per_update": (
                    prepared.unique_root_value_model_inputs
                ),
            }
        )
    except BaseException:
        payload = {"status": "error", "traceback": traceback.format_exc()}
        target_queue = result_queue if ready_sent else status_queue
        target_queue.put(payload)
        raise


def _checkpoint_optimizer_update_index(checkpoint: Mapping[str, Any]) -> int:
    metadata = checkpoint.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("profile checkpoint has no training metadata")
    raw = metadata.get("total_optimizer_updates")
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
        raise ValueError("profile checkpoint has no valid optimizer update clock")
    return raw


def _scheduler_horizon(
    train_config: Any,
    *,
    source_update_index: int,
    kernel_steps: int,
) -> int:
    explicit = train_config.optimizer.scheduler_total_updates
    minimum = source_update_index + kernel_steps
    if explicit is None:
        return max(1, minimum)
    if explicit < minimum:
        raise ValueError("profile scheduler horizon ends inside the fixed kernel")
    return int(explicit)


def _validate_cycle_result(
    result: ProfileKernelUpdateResult,
    ppo_epochs: int,
) -> None:
    expected_optimizer_rows = ppo_epochs * 1_024
    if result.optimizer_steps != ppo_epochs:
        raise RuntimeError("profile learner kernel optimizer step count changed")
    if (
        result.ppo_rows,
        result.factual_rows,
        result.planner_rows,
        result.planner_applicable_rows,
        result.root_value_rows,
    ) != (expected_optimizer_rows,) * 5:
        raise RuntimeError("profile learner kernel objective counters changed")


def _terminate_process(process: BaseProcess) -> None:
    process.terminate()
    process.join(timeout=10.0)


__all__ = ["ConcurrentProfileLearner", "ProfileLearnerResult"]
