"""Microbenchmark for vectorized rollout stepping throughput."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.batch import DeckBatch
from ptcg_rl.engine.vector_battle import VectorBattlePool
from ptcg_rl.model import (
    LEGACY_STATE_ENCODER_MISSING_KEYS,
    AgentNetworkConfig,
    AgentPolicyValueNet,
    OptionBatch,
    StateBatch,
    build_agent_policy_value_net,
)
from ptcg_rl.rl.rollout import (
    ListRolloutRecorder,
    RolloutActors,
    RolloutRunSummary,
    RolloutStepper,
)

PolicyKind = Literal["model", "min_count"]


class RolloutBenchmarkConfig(BaseModel):
    """Hydra-backed config for rollout throughput benchmarking."""

    model_config = ConfigDict(extra="forbid")

    deck_path: Path = Path("data/sample_submission/deck.csv")
    output_path: Path | None = Path("outputs/rl/rollout_benchmark/summary.json")
    num_concurrent_games: int = 64
    total_recorded_decisions: int = 512
    max_iterations: int = 100_000
    temperature: float = 1.0
    seed: int = 0
    device: str = "auto"
    policy_kind: PolicyKind = "model"
    checkpoint_path: Path | None = None
    model: AgentNetworkConfig = Field(default_factory=AgentNetworkConfig)
    gpu_sample_interval_seconds: float = 0.2
    min_recorded_decisions_per_second: float | None = None

    @field_validator("num_concurrent_games", "total_recorded_decisions", "max_iterations")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive benchmark limits."""
        if value <= 0:
            raise ValueError("benchmark limits must be positive")
        return value

    @field_validator("temperature", "gpu_sample_interval_seconds")
    @classmethod
    def valid_non_negative_float(cls, value: float) -> float:
        """Reject invalid non-negative floats."""
        if value < 0.0:
            raise ValueError("float benchmark settings must be non-negative")
        return value

    @field_validator("min_recorded_decisions_per_second")
    @classmethod
    def valid_optional_non_negative_float(cls, value: float | None) -> float | None:
        """Reject negative optional throughput thresholds."""
        if value is not None and value < 0.0:
            raise ValueError("throughput threshold must be non-negative")
        return value


class _MinCountPolicy:
    """Policy that picks the first minCount options; useful for engine-only smoke."""

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        del temperature
        if len(decks) != int(states.card_ids.shape[0]):
            raise ValueError("benchmark deck batch is misaligned")
        actions = tuple(
            tuple(range(int(options.min_counts[row].item())))
            for row in range(int(options.valid_options.shape[0]))
        )
        batch_size = len(actions)
        device = options.valid_options.device
        return (
            actions,
            torch.zeros(batch_size, dtype=torch.float32, device=device),
            torch.zeros(batch_size, dtype=torch.float32, device=device),
        )


class _ModelRolloutPolicy:
    """Thin wrapper around ``AgentPolicyValueNet`` for rollout benchmarking."""

    def __init__(self, model: AgentPolicyValueNet) -> None:
        self._model = model

    def sample_decode(
        self,
        states: StateBatch,
        options: OptionBatch,
        decks: DeckBatch,
        *,
        temperature: float = 1.0,
    ) -> tuple[tuple[tuple[int, ...], ...], Tensor, Tensor]:
        return self._model.sample_decode(
            states,
            options,
            decks,
            temperature=temperature,
        )


class _GpuUtilizationSampler:
    """Best-effort ``nvidia-smi`` GPU utilization sampler."""

    def __init__(self, *, device_index: int, interval_seconds: float) -> None:
        self._device_index = int(device_index)
        self._interval_seconds = max(float(interval_seconds), 0.05)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[int] = []
        self.error: str = ""

    def __enter__(self) -> _GpuUtilizationSampler:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def summary(self) -> dict[str, Any]:
        """Return compact utilization stats."""
        if not self.samples:
            return {
                "samples": 0,
                "mean_percent": None,
                "max_percent": None,
                "error": self.error,
            }
        return {
            "samples": len(self.samples),
            "mean_percent": sum(self.samples) / float(len(self.samples)),
            "max_percent": max(self.samples),
            "error": self.error,
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            value = _query_gpu_utilization(self._device_index)
            if value is None:
                if not self.error:
                    self.error = "nvidia-smi utilization query unavailable"
                return
            self.samples.append(value)
            self._stop.wait(self._interval_seconds)


def run_rollout_benchmark(config: RolloutBenchmarkConfig) -> dict[str, Any]:
    """Run a rollout stepping throughput benchmark and write a JSON summary."""
    torch.manual_seed(config.seed)
    device = _resolve_device(config.device)
    if device.type == "cuda":
        torch.cuda.set_device(_cuda_device_index(device))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(_cuda_device_index(device))

    deck = tuple(records.read_deck(records.repo_path(config.deck_path)))
    policy = _build_policy(config, device=device)
    recorder = ListRolloutRecorder()
    gpu_context: _GpuUtilizationSampler | _NullGpuSampler
    gpu_context = (
        _GpuUtilizationSampler(
            device_index=_cuda_device_index(device),
            interval_seconds=config.gpu_sample_interval_seconds,
        )
        if device.type == "cuda"
        else _NullGpuSampler()
    )

    with VectorBattlePool(
        config.num_concurrent_games,
        lambda: (deck, deck),
    ) as pool:
        stepper = RolloutStepper(
            pool=pool,
            actors=RolloutActors(mode="self_play", candidate_policy=policy),
            recorder=recorder,
            temperature=config.temperature,
            device=device,
        )
        _synchronize_if_cuda(device)
        start = time.perf_counter()
        with gpu_context:
            run_summary = stepper.run_until(
                total_recorded_decisions=config.total_recorded_decisions,
                max_iterations=config.max_iterations,
            )
            _synchronize_if_cuda(device)
        elapsed_seconds = time.perf_counter() - start
        live_games = len(pool.live_games)

    rates = _rate_summary(run_summary, elapsed_seconds)
    cuda_summary = _cuda_summary(device)
    output = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": config.model_dump(mode="json"),
        "device": str(device),
        "policy_kind": config.policy_kind,
        "elapsed_seconds": elapsed_seconds,
        "run": _run_summary_dict(run_summary),
        "rates": rates,
        "recorder": {
            "decisions": len(recorder.decisions),
            "finished_games": len(recorder.finished_games),
        },
        "live_games_after_run": live_games,
        "cuda": cuda_summary,
        "gpu_utilization": gpu_context.summary(),
    }
    threshold = config.min_recorded_decisions_per_second
    if threshold is not None and rates["recorded_decisions_per_second"] < threshold:
        output["passed"] = False
        _write_summary(config.output_path, output)
        raise RuntimeError(
            "rollout benchmark below threshold: "
            f"{rates['recorded_decisions_per_second']:.2f} < {threshold:.2f}"
        )
    output["passed"] = True
    _write_summary(config.output_path, output)
    return output


class _NullGpuSampler:
    def __enter__(self) -> _NullGpuSampler:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> None:
        del exc_type, exc, traceback

    def summary(self) -> dict[str, Any]:
        return {
            "samples": 0,
            "mean_percent": None,
            "max_percent": None,
            "error": "not a cuda device",
        }


def _build_policy(
    config: RolloutBenchmarkConfig,
    *,
    device: torch.device,
) -> _MinCountPolicy | _ModelRolloutPolicy:
    if config.policy_kind == "min_count":
        return _MinCountPolicy()
    model = build_agent_policy_value_net(_model_config(config)).to(device)
    if config.checkpoint_path is not None:
        checkpoint = torch.load(records.repo_path(config.checkpoint_path), map_location="cpu")
        incompatible = model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=False)
        allowed_missing = {
            "opponent_hand_head.weight",
            "opponent_hand_head.bias",
        } | LEGACY_STATE_ENCODER_MISSING_KEYS
        missing = set(incompatible.missing_keys)
        unexpected = set(incompatible.unexpected_keys)
        if missing - allowed_missing or unexpected:
            raise RuntimeError("checkpoint state dict is incompatible with benchmark model")
    model.eval()
    return _ModelRolloutPolicy(model)


def _model_config(config: RolloutBenchmarkConfig) -> AgentNetworkConfig:
    if config.checkpoint_path is None:
        return config.model
    checkpoint = torch.load(records.repo_path(config.checkpoint_path), map_location="cpu")
    checkpoint_config = _checkpoint_model_config(checkpoint)
    return checkpoint_config or config.model


def _resolve_device(raw_device: str) -> torch.device:
    normalized = raw_device.strip().lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "gpu":
        normalized = "cuda"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device is unavailable: {raw_device}")
    return device


def _checkpoint_state_dict(checkpoint: Any) -> Mapping[str, Any]:
    if isinstance(checkpoint, Mapping):
        for key in ("model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, Mapping):
                return _strip_lightning_model_prefix(cast(Mapping[str, Any], value))
        return _strip_lightning_model_prefix(cast(Mapping[str, Any], checkpoint))
    raise TypeError("checkpoint must be a state_dict or contain model_state_dict")


def _strip_lightning_model_prefix(state_dict: Mapping[str, Any]) -> Mapping[str, Any]:
    if not state_dict:
        return state_dict
    if all(str(key).startswith("model.") for key in state_dict):
        return {str(key).removeprefix("model."): value for key, value in state_dict.items()}
    return state_dict


def _checkpoint_model_config(checkpoint: Any) -> AgentNetworkConfig | None:
    if not isinstance(checkpoint, Mapping):
        return None
    for key in ("model_config", "agent_network_config", "network_config"):
        value = checkpoint.get(key)
        if isinstance(value, AgentNetworkConfig):
            return value
        if isinstance(value, Mapping):
            return AgentNetworkConfig.model_validate(value)
    full_config = checkpoint.get("config")
    if isinstance(full_config, Mapping):
        model_config = full_config.get("model")
        if isinstance(model_config, Mapping):
            return AgentNetworkConfig.model_validate(model_config)
    return None


def _run_summary_dict(summary: RolloutRunSummary) -> dict[str, int]:
    return {
        "iterations": summary.iterations,
        "forced_actions": summary.forced_actions,
        "scripted_actions": summary.scripted_actions,
        "policy_actions": summary.policy_actions,
        "recorded_decisions": summary.recorded_decisions,
        "finished_games": summary.finished_games,
        "engine_submissions": (
            summary.forced_actions + summary.scripted_actions + summary.policy_actions
        ),
    }


def _rate_summary(
    summary: RolloutRunSummary,
    elapsed_seconds: float,
) -> dict[str, float]:
    return {
        "recorded_decisions_per_second": _rate(
            summary.recorded_decisions,
            elapsed_seconds,
        ),
        "policy_actions_per_second": _rate(summary.policy_actions, elapsed_seconds),
        "engine_submissions_per_second": _rate(
            summary.forced_actions + summary.scripted_actions + summary.policy_actions,
            elapsed_seconds,
        ),
        "finished_games_per_second": _rate(summary.finished_games, elapsed_seconds),
    }


def _cuda_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        return {
            "available": torch.cuda.is_available(),
            "device_name": None,
            "max_memory_allocated_mb": None,
            "max_memory_reserved_mb": None,
        }
    return {
        "available": True,
        "device_name": torch.cuda.get_device_name(_cuda_device_index(device)),
        "max_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(_cuda_device_index(device)) / 1_000_000.0
        ),
        "max_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(_cuda_device_index(device)) / 1_000_000.0
        ),
    }


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(_cuda_device_index(device))


def _cuda_device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"not a CUDA device: {device}")
    return torch.cuda.current_device() if device.index is None else int(device.index)


def _query_gpu_utilization(device_index: int) -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()[0:1]
    if not first_line:
        return None
    try:
        return int(first_line[0].strip())
    except ValueError:
        return None


def _write_summary(path: Path | None, summary: Mapping[str, Any]) -> None:
    if path is None:
        return
    resolved = records.repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _rate(count: int, elapsed_seconds: float) -> float:
    if elapsed_seconds <= 0.0:
        return 0.0
    return float(count) / elapsed_seconds
