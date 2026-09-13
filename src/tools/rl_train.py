"""Hydra entry point for the H200 learner or a native collection worker.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/rl_train.py

Use ``--config-name rl/train/base`` for the one-iteration smoke profile.
"""

# ruff: noqa: E402

from __future__ import annotations

import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType
from typing import Any, NoReturn, cast

from ptcg_rl.training.cuda_allocator import configure_cuda_allocator

configure_cuda_allocator()

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.stateless_training import run_simple_stateless_training
from ptcg_rl.rl.stateless_training_config import SimpleStatelessTrainingConfig
from ptcg_rl.rl.training import RLTrainConfig, run_rl_training
from ptcg_rl.stack_sampling import maybe_start_stack_sampler
from ptcg_rl.training.host_policy import require_cuda_training_host


def _raise_shutdown_interrupt(
    signum: int,
    _frame: FrameType | None,
) -> NoReturn:
    """Turn SIGTERM into stack unwinding so child cleanup finally blocks run."""
    if signum != signal.SIGTERM:
        raise ValueError(f"unexpected shutdown signal: {signum}")
    raise KeyboardInterrupt("SIGTERM requested RL training shutdown")


@contextmanager
def _sigterm_as_keyboard_interrupt() -> Iterator[None]:
    """Install and restore the supervisor SIGTERM cleanup handler."""
    previous = signal.signal(signal.SIGTERM, _raise_shutdown_interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/train/default",
)
def main(hydra_config: DictConfig) -> None:
    """Run PPO RL training."""
    maybe_start_stack_sampler()
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    resolved = cast(dict[str, Any], raw_config)
    if resolved.get("trainer") == "simple_stateless":
        stateless_config = SimpleStatelessTrainingConfig.model_validate(resolved)
        role = os.environ.get("PTCG_RL_TRAINING_ROLE", "learner")
        if role == "collection-worker":
            from ptcg_rl.rl.native_distributed.lifecycle import (
                NATIVE_WORKER_RECYCLE_EXIT_CODE,
                NativeWorkerProcessRecycleError,
            )
            from ptcg_rl.rl.native_distributed.worker import (
                run_native_collection_worker,
            )

            worker_id = _required_environment("PTCG_RL_WORKER_ID")
            coordinator_host = _required_environment("PTCG_RL_COORDINATOR_HOST")
            worker_profile = _required_environment("PTCG_RL_WORKER_PROFILE")
            try:
                with _sigterm_as_keyboard_interrupt():
                    run_native_collection_worker(
                        stateless_config,
                        worker_id=worker_id,
                        coordinator_host=coordinator_host,
                        worker_profile=worker_profile,
                    )
            except NativeWorkerProcessRecycleError:
                raise SystemExit(NATIVE_WORKER_RECYCLE_EXIT_CODE) from None
            return
        if role != "learner":
            raise ValueError(f"unsupported RL training role: {role}")
        if stateless_config.validate_only:
            print(run_simple_stateless_training(stateless_config))
            return
        require_cuda_training_host()
        with _sigterm_as_keyboard_interrupt():
            print(run_simple_stateless_training(stateless_config))
        return
    config = RLTrainConfig.model_validate(resolved)
    require_cuda_training_host()
    with _sigterm_as_keyboard_interrupt():
        print(run_rl_training(config))


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"collection-worker role requires {name}")
    return value


if __name__ == "__main__":
    main()
