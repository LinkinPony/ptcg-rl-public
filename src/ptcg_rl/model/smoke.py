"""Smoke tests and CPU micro-benchmarks for the agent policy/value model."""

from __future__ import annotations

import glob
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps.records import extract_replay_rows
from ptcg_rl.model.network import (
    AgentNetworkConfig,
    AgentPolicyValueNet,
    build_agent_policy_value_net,
)
from ptcg_rl.model.policy import KNOWN_CONTEXT_COUNT, KNOWN_OPTION_TYPE_COUNT
from ptcg_rl.training.bc_dataset import (
    BCSample,
    collate_bc_samples,
    sample_from_step_row,
)


class AgentModelSmokeConfig(BaseModel):
    """Hydra-backed config for real-replay model smoke tests."""

    model_config = ConfigDict(extra="forbid")

    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = "data/external/kaggle_top_episodes_daily/*/*.json"
    max_replays: int | None = 2
    max_rows: int = 64
    batch_size: int = 8
    warmup_batches: int = 1
    device: str = "cpu"
    model: AgentNetworkConfig = AgentNetworkConfig()
    output_path: Path | None = Path("outputs/model/agent_smoke/summary.json")
    drop_forced_actions: bool = True
    normalize_unordered_actions: bool = True
    fast_prefix_bytes: int = 65_536
    chunk_size: int = 1 << 20
    greedy_decode: bool = True

    @field_validator("max_replays")
    @classmethod
    def valid_optional_positive(cls, value: int | None) -> int | None:
        """Reject non-positive optional limits."""
        if value is not None and value <= 0:
            raise ValueError("optional limits must be positive when set")
        return value

    @field_validator("max_rows", "batch_size", "fast_prefix_bytes", "chunk_size")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive required limits."""
        if value <= 0:
            raise ValueError("limits must be positive")
        return value

    @field_validator("warmup_batches")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject invalid non-negative counters."""
        if value < 0:
            raise ValueError("warmup_batches must be non-negative")
        return value


@dataclass
class _SmokeStats:
    """Mutable counters collected while probing model batches."""

    replays_seen: int = 0
    rows_seen: int = 0
    samples: int = 0
    skipped_rows: int = 0
    batches: int = 0
    warmup_batches: int = 0
    measured_batches: int = 0
    measured_samples: int = 0
    forward_seconds: float = 0.0
    max_tokens: int = 0
    max_options: int = 0
    extraction_counters: Counter[str] = field(default_factory=Counter)
    select_counts: Counter[tuple[int, int]] = field(default_factory=Counter)
    option_type_counts: Counter[int] = field(default_factory=Counter)
    oov_option_type_count: int = 0
    oov_context_count: int = 0

    def add_sample(self, sample: BCSample) -> None:
        """Record one tensor-ready sample."""
        self.samples += 1
        self.max_tokens = max(self.max_tokens, len(sample.state.card_ids))
        self.max_options = max(self.max_options, len(sample.options))
        self.select_counts[(sample.select_type, sample.select_context)] += 1
        if sample.select_context < 0 or sample.select_context >= KNOWN_CONTEXT_COUNT:
            self.oov_context_count += 1
        for option in sample.options:
            self.option_type_counts[int(option.option_type)] += 1
            if option.option_type < 0 or option.option_type >= KNOWN_OPTION_TYPE_COUNT:
                self.oov_option_type_count += 1

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-serializable smoke summary metrics."""
        seconds_per_batch = (
            self.forward_seconds / self.measured_batches
            if self.measured_batches > 0
            else None
        )
        seconds_per_sample = (
            self.forward_seconds / self.samples if self.samples > 0 else None
        )
        return {
            "replays_seen": self.replays_seen,
            "rows_seen": self.rows_seen,
            "samples": self.samples,
            "skipped_rows": self.skipped_rows,
            "batches": self.batches,
            "warmup_batches": self.warmup_batches,
            "measured_batches": self.measured_batches,
            "measured_samples": self.measured_samples,
            "forward_seconds": self.forward_seconds,
            "seconds_per_batch": seconds_per_batch,
            "seconds_per_sample": (
                self.forward_seconds / self.measured_samples
                if self.measured_samples > 0
                else seconds_per_sample
            ),
            "max_tokens": self.max_tokens,
            "max_options": self.max_options,
            "oov_context_count": self.oov_context_count,
            "oov_option_type_count": self.oov_option_type_count,
            "extraction_counters": dict(sorted(self.extraction_counters.items())),
            "select_counts": _counter_rows(
                self.select_counts,
                key_names=("select_type", "select_context"),
            ),
            "option_type_counts": [
                {"option_type": key, "count": count}
                for key, count in sorted(self.option_type_counts.items())
            ],
        }


def run_agent_model_smoke(config: AgentModelSmokeConfig) -> dict[str, Any]:
    """Run real-replay smoke checks and write an optional summary JSON."""
    replay_paths = _resolve_replay_paths(config)
    stats = _new_stats()
    device = torch.device(config.device)
    model = build_agent_policy_value_net(config.model).to(device)
    model.eval()

    pending: list[BCSample] = []
    for replay_path in replay_paths:
        if stats.samples >= config.max_rows:
            break
        stats.replays_seen += 1
        rows, counters = extract_replay_rows(
            replay_path,
            drop_forced_actions=config.drop_forced_actions,
            normalize_unordered_actions=config.normalize_unordered_actions,
            fast_prefix_bytes=config.fast_prefix_bytes,
            chunk_size=config.chunk_size,
        )
        stats.extraction_counters.update(counters)
        for row in rows:
            if stats.samples >= config.max_rows:
                break
            _append_sample(row, pending, stats)
            if len(pending) >= config.batch_size:
                _run_forward_batch(config, model, pending, stats, device=device)
                pending.clear()

    if pending:
        _run_forward_batch(config, model, pending, stats, device=device)

    summary = stats.as_dict()
    output_path = config.output_path
    if output_path is not None:
        _write_json(deck_records.repo_path(output_path), summary)
    return summary


def run_agent_model_smoke_from_rows(
    rows: Iterable[Mapping[str, Any]],
    config: AgentModelSmokeConfig,
) -> dict[str, Any]:
    """Run the same model smoke path from already extracted compact rows."""
    stats = _new_stats()
    device = torch.device(config.device)
    model = build_agent_policy_value_net(config.model).to(device)
    model.eval()
    pending: list[BCSample] = []

    for row in rows:
        if stats.samples >= config.max_rows:
            break
        _append_sample(row, pending, stats)
        if len(pending) >= config.batch_size:
            _run_forward_batch(config, model, pending, stats, device=device)
            pending.clear()

    if pending:
        _run_forward_batch(config, model, pending, stats, device=device)
    return stats.as_dict()


def _new_stats() -> _SmokeStats:
    return _SmokeStats()


def _append_sample(
    row: Mapping[str, Any],
    pending: list[BCSample],
    stats: _SmokeStats,
) -> None:
    stats.rows_seen += 1
    sample = sample_from_step_row(row)
    if sample is None:
        stats.skipped_rows += 1
        return
    stats.add_sample(sample)
    pending.append(sample)


def _run_forward_batch(
    config: AgentModelSmokeConfig,
    model: AgentPolicyValueNet,
    samples: Sequence[BCSample],
    stats: _SmokeStats,
    *,
    device: torch.device,
) -> None:
    batch = collate_bc_samples(samples, device=device)
    _sync_if_cuda(device)
    with torch.inference_mode():
        start = time.perf_counter()
        output = model(batch.states, batch.options)
        _sync_if_cuda(device)
        elapsed = time.perf_counter() - start
        if config.greedy_decode:
            actions = model.greedy_decode(batch.states, batch.options)
            _assert_greedy_actions_legal(
                actions,
                min_counts=batch.options.min_counts,
                max_counts=batch.options.max_counts,
                valid_options=batch.options.valid_options,
            )

    expected_shape = (
        len(samples),
        int(batch.options.valid_options.shape[1]) + 1,
    )
    if tuple(output.policy_logits.shape) != expected_shape:
        raise AssertionError(
            f"policy logits shape {tuple(output.policy_logits.shape)} "
            f"!= expected {expected_shape}"
        )
    if tuple(output.value.shape) != (len(samples),):
        raise AssertionError(f"value shape mismatch: {tuple(output.value.shape)}")
    if bool(torch.isnan(output.policy_logits).any().item()):
        raise AssertionError("policy logits contain NaN")
    if bool(torch.isnan(output.value).any().item()):
        raise AssertionError("value predictions contain NaN")

    stats.batches += 1
    if stats.batches <= config.warmup_batches:
        stats.warmup_batches += 1
    else:
        stats.measured_batches += 1
        stats.measured_samples += len(samples)
        stats.forward_seconds += elapsed


def _assert_greedy_actions_legal(
    actions: Sequence[Sequence[int]],
    *,
    min_counts: torch.Tensor,
    max_counts: torch.Tensor,
    valid_options: torch.Tensor,
) -> None:
    for index, action in enumerate(actions):
        if len(action) < int(min_counts[index].item()):
            raise AssertionError(f"decoded action below minCount at batch row {index}")
        if len(action) > int(max_counts[index].item()):
            raise AssertionError(f"decoded action above maxCount at batch row {index}")
        if len(set(action)) != len(action):
            raise AssertionError(f"decoded action has duplicate indices at batch row {index}")
        for option_index in action:
            if option_index < 0 or option_index >= int(valid_options.shape[1]):
                raise AssertionError(f"decoded action out of range at batch row {index}")
            if not bool(valid_options[index, option_index].item()):
                raise AssertionError(f"decoded action chose padding at batch row {index}")


def _resolve_replay_paths(config: AgentModelSmokeConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        paths = tuple(deck_records.repo_path(path) for path in config.replay_paths)
    else:
        pattern = str(deck_records.repo_path(Path(config.replay_glob)))
        paths = tuple(Path(path) for path in sorted(glob.glob(pattern)))
    if config.max_replays is not None:
        paths = paths[: config.max_replays]
    if not paths:
        raise ValueError("no replay paths matched smoke configuration")
    return paths


def _counter_rows(
    counter: Counter[tuple[int, int]],
    *,
    key_names: tuple[str, str],
) -> list[dict[str, int]]:
    first_name, second_name = key_names
    rows: list[dict[str, int]] = []
    for (first_value, second_value), count in counter.items():
        rows.append(
            {
                first_name: first_value,
                second_name: second_value,
                "count": count,
            }
        )
    return sorted(rows, key=lambda row: (-row["count"], row[first_name], row[second_name]))


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(data), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
