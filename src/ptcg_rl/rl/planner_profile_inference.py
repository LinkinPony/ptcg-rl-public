"""Isolated exact-model inference serving for integrated planner profiling."""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

import torch

from ptcg_rl.agent.search.root_information_tensorizer import (
    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
)
from ptcg_rl.model.network import AgentNetworkConfig, AgentPolicyValueNet
from ptcg_rl.model.weights_only_migration_classification import (
    checkpoint_model_config_payload,
    checkpoint_state_dict,
)
from ptcg_rl.rl.collection import ModelRolloutPolicy
from ptcg_rl.rl.inference_server import (
    InferenceClientConfig,
    InferenceRequestPurpose,
    InferenceServerConfig,
    RemoteInferencePolicy,
    run_inference_server_step,
)
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeConfig


def load_profile_model(
    checkpoint_path: Path,
    *,
    device: torch.device | str,
    expected_model_fingerprint: str | None = None,
) -> AgentPolicyValueNet:
    """Strictly construct the checkpoint's serving architecture on one device."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = AgentNetworkConfig.model_validate(
        checkpoint_model_config_payload(checkpoint)
    )
    from ptcg_rl.model.network import build_agent_policy_value_net

    model = build_agent_policy_value_net(config)
    incompatible = model.load_state_dict(
        checkpoint_state_dict(checkpoint), strict=False
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "profile checkpoint differs from its serving architecture: "
            f"missing={sorted(incompatible.missing_keys)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    actual_fingerprint = canonical_model_state_fingerprint(model)
    if (
        expected_model_fingerprint is not None
        and actual_fingerprint != expected_model_fingerprint
    ):
        raise RuntimeError(
            "profile constructed model differs from its preregistered identity"
        )
    model.to(device)
    model.eval()
    return model


@dataclass(slots=True)
class ProfileInferenceServer:
    """One spawned CUDA inference server and actor-local queue clients."""

    process: BaseProcess
    request_queue: Any
    response_queues: Mapping[str, Any]
    stop_event: Any
    status_queue: Any
    clients: Mapping[str, RemoteInferencePolicy]
    _closed: bool = False

    @classmethod
    def start(
        cls,
        *,
        checkpoint_path: Path,
        expected_model_fingerprint: str,
        policy_version: int,
        proposal_version: int,
        runtime: ResolvedPlannerRuntimeConfig,
        actor_purposes: Mapping[str, InferenceRequestPurpose],
    ) -> ProfileInferenceServer:
        """Spawn CUDA ownership before constructing any actor clients."""
        context = mp.get_context("spawn")
        request_queue = context.Queue(maxsize=runtime.batching.inference_queue_capacity)
        response_queues = {
            actor_id: context.Queue(maxsize=runtime.batching.inference_queue_capacity)
            for actor_id in actor_purposes
        }
        stop_event = context.Event()
        status_queue = context.Queue(maxsize=2)
        server_config = InferenceServerConfig(
            max_batch=(
                runtime.batching.actor_count
                * runtime.batching.max_root_rows_per_request
            ),
            max_wait_ms=runtime.batching.inference_batch_wait_ms,
            max_planner_candidate_rows=(runtime.batching.candidate_microbatch_rows),
            max_planner_proposal_rows=(runtime.batching.proposal_microbatch_rows),
            max_root_information_rows=(runtime.batching.root_value_microbatch_rows),
            planner_context_ttl_seconds=max(
                runtime.deadlines.request_timeout_seconds
                + runtime.deadlines.cleanup_timeout_seconds,
                1.0,
            ),
        )
        process = context.Process(
            target=_inference_server_main,
            kwargs={
                "checkpoint_path": checkpoint_path,
                "expected_model_fingerprint": expected_model_fingerprint,
                "policy_version": policy_version,
                "proposal_version": proposal_version,
                "planner_context_capacity": runtime.contexts.retained_root_rows,
                "request_queue": request_queue,
                "response_queues": response_queues,
                "stop_event": stop_event,
                "status_queue": status_queue,
                "server_config": server_config,
            },
            name="planner-profile-inference",
        )
        process.start()
        try:
            status = status_queue.get(timeout=180.0)
        except queue.Empty as exc:
            stop_event.set()
            process.join(timeout=10.0)
            raise TimeoutError("profile inference server did not become ready") from exc
        if not isinstance(status, Mapping) or status.get("status") != "ready":
            stop_event.set()
            process.join(timeout=10.0)
            detail = (
                status.get("traceback", status)
                if isinstance(status, Mapping)
                else status
            )
            raise RuntimeError(f"profile inference server failed to start: {detail}")
        client_config = InferenceClientConfig(
            response_timeout_seconds=max(
                runtime.deadlines.request_timeout_seconds
                + runtime.deadlines.cleanup_timeout_seconds,
                1.0,
            ),
            response_retries=0,
        )
        clients = {
            actor_id: RemoteInferencePolicy(
                actor_id=actor_id,
                policy_id="profile",
                request_queue=request_queue,
                response_queue=response_queues[actor_id],
                config=client_config,
                request_purpose=purpose,
                planner_model_version_lease=(
                    policy_version if purpose == "planner_behavior" else None
                ),
                planner_tensor_schema_fingerprint=(
                    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT
                    if purpose == "planner_behavior"
                    else ""
                ),
                planner_inference_device_type=(
                    "cuda" if purpose == "planner_behavior" else None
                ),
            )
            for actor_id, purpose in actor_purposes.items()
        }
        return cls(
            process=process,
            request_queue=request_queue,
            response_queues=response_queues,
            stop_event=stop_event,
            status_queue=status_queue,
            clients=clients,
        )

    def close(self) -> None:
        """Stop serving only after all actor calls and context releases drain."""
        if self._closed:
            return
        self._closed = True
        self.stop_event.set()
        self.process.join(timeout=30.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10.0)
            raise RuntimeError("profile inference server did not stop cleanly")
        if self.process.exitcode not in (0, None):
            raise RuntimeError(
                f"profile inference server exited with code {self.process.exitcode}"
            )


def _inference_server_main(
    *,
    checkpoint_path: Path,
    expected_model_fingerprint: str,
    policy_version: int,
    proposal_version: int,
    planner_context_capacity: int,
    request_queue: Any,
    response_queues: Mapping[str, Any],
    stop_event: Any,
    status_queue: Any,
    server_config: InferenceServerConfig,
) -> None:
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("profile inference child has no visible CUDA device")
        model = load_profile_model(
            checkpoint_path,
            device="cuda",
            expected_model_fingerprint=expected_model_fingerprint,
        )
        policy = ModelRolloutPolicy(
            model,
            policy_version=policy_version,
            autocast="bf16",
            planner_context_capacity=planner_context_capacity,
            verified_model_fingerprint=expected_model_fingerprint,
            proposal_version=proposal_version,
        )
        status_queue.put({"status": "ready"})
        while not stop_event.is_set():
            run_inference_server_step(
                request_queue=request_queue,
                response_queues=response_queues,
                policies={"profile": policy},
                config=server_config,
            )
        # Drain requests that crossed the stop boundary before process exit.
        while True:
            stats = run_inference_server_step(
                request_queue=request_queue,
                response_queues=response_queues,
                policies={"profile": policy},
                config=server_config.model_copy(update={"max_wait_ms": 0.0}),
            )
            if stats.requests == 0:
                break
    except BaseException:
        status_queue.put({"status": "error", "traceback": traceback.format_exc()})
        raise


__all__ = ["ProfileInferenceServer", "load_profile_model"]
