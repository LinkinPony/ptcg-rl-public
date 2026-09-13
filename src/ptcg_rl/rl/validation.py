"""Validation utilities for rollout trajectory artifacts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.model import AgentNetworkConfig, build_agent_policy_value_net
from ptcg_rl.training import KaggleStepDataConfig
from ptcg_rl.training.bc_dataset import iter_bc_batches


class RolloutArtifactValidationError(RuntimeError):
    """Raised when rollout artifacts fail validation."""


class RolloutArtifactValidationConfig(BaseModel):
    """Config for rollout artifact validation."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: Path = Path("outputs/rl/rollouts/dev/manifest.json")
    batch_size: int = 16
    read_batch_size: int = 4096
    max_forward_batches: int = 1
    device: str = "cpu"
    min_games: int = 1
    min_rows: int = 1
    min_godview_count_match_rate: float = 0.0
    model: AgentNetworkConfig = Field(default_factory=AgentNetworkConfig)

    @field_validator(
        "batch_size",
        "read_batch_size",
        "min_games",
        "min_rows",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive validation limits."""
        if value <= 0:
            raise ValueError("validation limits must be positive")
        return value

    @field_validator("max_forward_batches")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative forward-batch limits."""
        if value < 0:
            raise ValueError("max_forward_batches must be non-negative")
        return value

    @field_validator("min_godview_count_match_rate")
    @classmethod
    def valid_rate(cls, value: float) -> float:
        """Reject invalid rate thresholds."""
        if value < 0.0 or value > 1.0:
            raise ValueError("min_godview_count_match_rate must be in [0, 1]")
        return value

    @field_validator("device")
    @classmethod
    def valid_device(cls, value: str) -> str:
        """Reject empty device strings."""
        if not value.strip():
            raise ValueError("device must be non-empty")
        return value


@dataclass
class _RowValidationState:
    rows: int = 0
    unknown_game_rows: int = 0
    reward_mismatches: int = 0
    result_mismatches: int = 0
    invalid_seat_rows: int = 0
    godview_available_rows: int = 0
    godview_count_match_rows: int = 0
    rows_by_game: Counter[str] = field(default_factory=Counter)
    rows_by_game_seat: Counter[tuple[str, int]] = field(default_factory=Counter)
    episode_lengths: defaultdict[tuple[str, int], Counter[int]] = field(
        default_factory=lambda: defaultdict(Counter)
    )


def validate_rollout_artifacts(
    config: RolloutArtifactValidationConfig,
) -> dict[str, Any]:
    """Validate rollout shards, games metadata, and BC/model compatibility."""
    manifest_path = deck_records.repo_path(config.manifest_path)
    manifest = _read_manifest(manifest_path)
    shard_paths = _shard_paths(manifest)
    games_path = _games_path(manifest)
    games = _read_games(games_path, read_batch_size=config.read_batch_size)
    state, shard_summaries = _validate_row_shards(
        shard_paths,
        games=games,
        read_batch_size=config.read_batch_size,
    )
    consistency = _consistency_summary(state, games)
    godview = _godview_summary(state)
    forward = _run_forward_smoke(config) if config.max_forward_batches > 0 else None
    summary = {
        "manifest_path": deck_records.display_path(manifest_path),
        "games_path": deck_records.display_path(games_path),
        "shards": shard_summaries,
        "games": len(games),
        "rows": state.rows,
        "consistency": consistency,
        "godview": godview,
        "bc_forward": forward,
    }
    failures = _validation_failures(
        config,
        manifest=manifest,
        shard_count=len(shard_paths),
        game_count=len(games),
        row_count=state.rows,
        consistency=consistency,
        godview=godview,
        forward=forward,
    )
    if failures:
        raise RolloutArtifactValidationError("; ".join(failures))
    return summary


def _read_manifest(manifest_path: Path) -> Mapping[str, Any]:
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"manifest must be an object: {manifest_path}")
    return manifest


def _shard_paths(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    shards = manifest.get("shards", ())
    if not isinstance(shards, Sequence):
        raise ValueError("manifest shards must be a sequence")
    paths = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        path = shard.get("path")
        if isinstance(path, str) and path:
            paths.append(deck_records.repo_path(Path(path)))
    if not paths:
        raise ValueError("manifest contains no row shards")
    return tuple(paths)


def _games_path(manifest: Mapping[str, Any]) -> Path:
    games = manifest.get("games")
    if not isinstance(games, Mapping):
        raise ValueError("manifest is missing games metadata")
    path = games.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("manifest games metadata is missing path")
    return deck_records.repo_path(Path(path))


def _read_games(
    games_path: Path,
    *,
    read_batch_size: int,
) -> dict[str, Mapping[str, Any]]:
    games: dict[str, Mapping[str, Any]] = {}
    for row in _iter_parquet_rows(games_path, read_batch_size=read_batch_size):
        game_id = _string(row.get("game_id"))
        if not game_id:
            raise ValueError(f"games row is missing game_id: {games_path}")
        if game_id in games:
            raise ValueError(f"duplicate game_id in games parquet: {game_id}")
        games[game_id] = row
    return games


def _validate_row_shards(
    shard_paths: Sequence[Path],
    *,
    games: Mapping[str, Mapping[str, Any]],
    read_batch_size: int,
) -> tuple[_RowValidationState, list[dict[str, Any]]]:
    state = _RowValidationState()
    shard_summaries: list[dict[str, Any]] = []
    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        shard_rows = int(parquet_file.metadata.num_rows)
        shard_summaries.append(
            {
                "path": deck_records.display_path(shard_path),
                "rows": shard_rows,
            }
        )
        for batch in parquet_file.iter_batches(batch_size=read_batch_size):
            for row in batch.to_pylist():
                _validate_row(row, games=games, state=state)
    return state, shard_summaries


def _validate_row(
    row: Mapping[str, Any],
    *,
    games: Mapping[str, Mapping[str, Any]],
    state: _RowValidationState,
) -> None:
    state.rows += 1
    game_id = _string(row.get("game_id"))
    game = games.get(game_id)
    if game is None:
        state.unknown_game_rows += 1
        return

    seat = _optional_int(row.get("player_index"))
    if seat not in (0, 1):
        state.invalid_seat_rows += 1
        return

    state.rows_by_game[game_id] += 1
    state.rows_by_game_seat[(game_id, seat)] += 1
    episode_length = _optional_int(row.get("episode_length"))
    if episode_length is not None:
        state.episode_lengths[(game_id, seat)][episode_length] += 1

    expected_reward = _reward_for_seat(seat, _int_value(game.get("winner_index")))
    reward = _optional_float(row.get("reward"))
    if reward is None or abs(reward - expected_reward) > 1e-6:
        state.reward_mismatches += 1
    if _string(row.get("result")) != _result_label(expected_reward):
        state.result_mismatches += 1

    if bool(row.get("godview_opp_hand_available")):
        state.godview_available_rows += 1
    if bool(row.get("godview_opp_hand_count_match")):
        state.godview_count_match_rows += 1


def _consistency_summary(
    state: _RowValidationState,
    games: Mapping[str, Mapping[str, Any]],
) -> dict[str, int]:
    game_row_mismatches = 0
    seat_count_mismatches = 0
    episode_length_mismatches = 0
    for game_id, game in games.items():
        expected_rows = _int_value(game.get("rows"))
        if state.rows_by_game[game_id] != expected_rows:
            game_row_mismatches += 1
        for seat in (0, 1):
            expected_seat_rows = _int_value(game.get(f"seat{seat}_decisions"))
            actual_seat_rows = state.rows_by_game_seat[(game_id, seat)]
            if actual_seat_rows != expected_seat_rows:
                seat_count_mismatches += 1
            lengths = state.episode_lengths.get((game_id, seat), Counter())
            for episode_length, count in lengths.items():
                if episode_length != expected_seat_rows:
                    episode_length_mismatches += count
    return {
        "unknown_game_rows": state.unknown_game_rows,
        "invalid_seat_rows": state.invalid_seat_rows,
        "reward_mismatches": state.reward_mismatches,
        "result_mismatches": state.result_mismatches,
        "game_row_mismatches": game_row_mismatches,
        "seat_count_mismatches": seat_count_mismatches,
        "episode_length_mismatches": episode_length_mismatches,
    }


def _godview_summary(state: _RowValidationState) -> dict[str, float | int]:
    return {
        "available_rows": state.godview_available_rows,
        "count_match_rows": state.godview_count_match_rows,
        "available_rate": _rate(state.godview_available_rows, state.rows),
        "count_match_rate": _rate(state.godview_count_match_rows, state.rows),
    }


def _run_forward_smoke(
    config: RolloutArtifactValidationConfig,
) -> dict[str, int]:
    device = _resolve_device(config.device)
    data_config = KaggleStepDataConfig(
        manifest_path=config.manifest_path,
        batch_size=config.batch_size,
        validation_fraction=0.0,
        shuffle_shards=False,
        shuffle_buffer_rows=1,
    )
    model = build_agent_policy_value_net(config.model).to(device)
    model.eval()
    batches = 0
    samples = 0
    with torch.no_grad():
        for batch in iter_bc_batches(
            data_config,
            split="train",
            epoch=0,
            seed=0,
            device=device,
            max_batches=config.max_forward_batches,
        ):
            output = model(batch.states, batch.options)
            valid_logits = output.policy_logits[:, :-1][batch.options.valid_options]
            if valid_logits.numel() == 0:
                raise RolloutArtifactValidationError("BC forward batch has no options")
            if not bool(torch.isfinite(valid_logits).all().item()):
                raise RolloutArtifactValidationError("BC forward logits are not finite")
            if not bool(torch.isfinite(output.value).all().item()):
                raise RolloutArtifactValidationError("BC forward values are not finite")
            batches += 1
            samples += len(batch.actions)
    return {"batches": batches, "samples": samples}


def _validation_failures(
    config: RolloutArtifactValidationConfig,
    *,
    manifest: Mapping[str, Any],
    shard_count: int,
    game_count: int,
    row_count: int,
    consistency: Mapping[str, int],
    godview: Mapping[str, float | int],
    forward: Mapping[str, int] | None,
) -> list[str]:
    failures: list[str] = []
    manifest_summary = manifest.get("summary")
    if isinstance(manifest_summary, Mapping):
        _check_equal(
            failures,
            "manifest summary rows",
            _optional_int(manifest_summary.get("rows")),
            row_count,
        )
        _check_equal(
            failures,
            "manifest summary games",
            _optional_int(manifest_summary.get("games")),
            game_count,
        )
        _check_equal(
            failures,
            "manifest summary shards",
            _optional_int(manifest_summary.get("shards")),
            shard_count,
        )
    else:
        failures.append("manifest summary is missing")
    if game_count < config.min_games:
        failures.append(f"games {game_count} < min_games {config.min_games}")
    if row_count < config.min_rows:
        failures.append(f"rows {row_count} < min_rows {config.min_rows}")
    for name, count in consistency.items():
        if count:
            failures.append(f"{name}={count}")
    count_match_rate = float(godview["count_match_rate"])
    if count_match_rate < config.min_godview_count_match_rate:
        failures.append(
            "godview count_match_rate "
            f"{count_match_rate:.4f} < {config.min_godview_count_match_rate:.4f}"
        )
    if (
        config.max_forward_batches > 0
        and (forward is None or int(forward.get("batches", 0)) <= 0)
    ):
        failures.append("BC forward smoke produced no batches")
    return failures


def _iter_parquet_rows(
    path: Path,
    *,
    read_batch_size: int,
) -> Iterator[Mapping[str, Any]]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=read_batch_size):
        yield from batch.to_pylist()


def _check_equal(
    failures: list[str],
    label: str,
    actual: int | None,
    expected: int,
) -> None:
    if actual != expected:
        failures.append(f"{label} {actual} != {expected}")


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


def _reward_for_seat(seat: int, winner_index: int) -> float:
    if winner_index == 2:
        return 0.0
    return 1.0 if winner_index == seat else -1.0


def _result_label(reward: float) -> str:
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _string(value: Any) -> str:
    return "" if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _int_value(value: Any) -> int:
    return int(value) if value is not None else 0


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
