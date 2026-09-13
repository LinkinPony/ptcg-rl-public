"""Streaming exact-pilot teacher dataset construction."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.training.teacher_replays import PublicPilotEpisode


class TeacherOutcomeWeights(BaseModel):
    """Outcome multipliers that retain every public-pilot episode."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    win: float = 1.0
    loss: float = 1.0
    draw: float = 1.0
    other: float = 1.0

    @field_validator("win", "loss", "draw", "other")
    @classmethod
    def positive_weight(cls, value: float) -> float:
        """Reject zero or negative teacher row weights."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("teacher outcome weights must be finite and positive")
        return value


class PublicPilotTeacherConfig(BaseModel):
    """Config for filtering exact pilot/deck rows into temporal shards."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_manifest_path: Path
    replay_manifest_path: Path
    output_dir: Path
    submission_id: int
    team_name: str
    deck_path: Path
    deck_hash: str
    validation_fraction: float = 0.20
    target_weight: float = 4.0
    outcome_weights: TeacherOutcomeWeights = TeacherOutcomeWeights()
    read_batch_size: int = 2048
    rows_per_shard: int = 50_000
    compression: str = "zstd"
    min_episode_coverage: float = 0.95

    @field_validator("submission_id", "read_batch_size", "rows_per_shard")
    @classmethod
    def positive_int(cls, value: int) -> int:
        """Reject non-positive identifiers and batch sizes."""
        if value <= 0:
            raise ValueError("identifiers and batch sizes must be positive")
        return value

    @field_validator("target_weight")
    @classmethod
    def positive_target_weight(cls, value: float) -> float:
        """Require a finite positive target-pilot weight."""
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("target_weight must be finite and positive")
        return value

    @field_validator("validation_fraction")
    @classmethod
    def split_probability(cls, value: float) -> float:
        """Require a non-empty temporal train and validation split."""
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        return value

    @field_validator("min_episode_coverage")
    @classmethod
    def coverage_probability(cls, value: float) -> float:
        """Restrict the episode coverage reference to (0, 1]."""
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError("min_episode_coverage must be in (0, 1]")
        return value

    @field_validator("team_name", "deck_hash", "compression")
    @classmethod
    def non_empty_string(cls, value: str) -> str:
        """Reject empty teacher identity and storage strings."""
        if not value.strip():
            raise ValueError("strings must be non-empty")
        return value.strip()


@dataclass
class _TeacherShardWriter:
    """Bounded-memory Parquet writer with atomic shard publication."""

    staging_dir: Path
    final_dir: Path
    schema: pa.Schema
    rows_per_shard: int
    compression: str
    shards: list[dict[str, Any]] = field(default_factory=list)
    _writer: pq.ParquetWriter | None = field(default=None, init=False)
    _temporary_path: Path | None = field(default=None, init=False)
    _rows_in_shard: int = field(default=0, init=False)

    def write(self, table: pa.Table) -> None:
        """Stream an Arrow table across fixed-size output shards."""
        offset = 0
        while offset < table.num_rows:
            self._ensure_open()
            capacity = self.rows_per_shard - self._rows_in_shard
            count = min(capacity, table.num_rows - offset)
            if self._writer is None:
                raise RuntimeError("teacher shard writer failed to open")
            self._writer.write_table(table.slice(offset, count))
            self._rows_in_shard += count
            offset += count
            if self._rows_in_shard >= self.rows_per_shard:
                self._close_current()

    def close(self) -> None:
        """Publish the final partial shard."""
        if self._writer is not None:
            self._close_current()

    def _ensure_open(self) -> None:
        if self._writer is not None:
            return
        shard_index = len(self.shards)
        self._temporary_path = self.staging_dir / f"steps-{shard_index:05d}.tmp"
        self._writer = pq.ParquetWriter(
            self._temporary_path,
            self.schema,
            compression=self.compression,
        )
        self._rows_in_shard = 0

    def _close_current(self) -> None:
        if self._writer is None or self._temporary_path is None:
            return
        self._writer.close()
        shard_index = len(self.shards)
        published = self.staging_dir / f"steps-{shard_index:05d}.parquet"
        os.replace(self._temporary_path, published)
        final_path = self.final_dir / published.name
        self.shards.append(
            {
                "path": deck_records.display_path(final_path),
                "rows": self._rows_in_shard,
                "bytes": published.stat().st_size,
                "sha256": _sha256(published),
            }
        )
        self._writer = None
        self._temporary_path = None
        self._rows_in_shard = 0


def build_public_pilot_teacher(config: PublicPilotTeacherConfig) -> dict[str, Any]:
    """Filter one exact public pilot and publish a temporal teacher manifest."""
    source_manifest_path = deck_records.repo_path(config.source_manifest_path)
    replay_manifest_path = deck_records.repo_path(config.replay_manifest_path)
    output_dir = deck_records.repo_path(config.output_dir)
    existing_manifest = output_dir / "manifest.json"
    identity = _build_identity(config, source_manifest_path, replay_manifest_path)
    if existing_manifest.exists():
        manifest = _read_json(existing_manifest)
        if manifest.get("identity") != identity:
            raise ValueError(
                "existing teacher dataset identity differs; choose a new output_dir"
            )
        _verify_output_manifest(manifest)
        return manifest

    source_manifest = _read_json(source_manifest_path)
    replay_manifest = _read_json(replay_manifest_path)
    episodes = _load_episodes(replay_manifest, submission_id=config.submission_id)
    split_by_episode = _temporal_episode_split(
        episodes,
        validation_fraction=config.validation_fraction,
    )
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    exact_signature = deck_records.deck_signature(
        deck_records.read_deck(deck_records.repo_path(config.deck_path))
    )
    if deck_records.signature_hash(exact_signature) != config.deck_hash:
        raise ValueError("configured deck_hash does not match deck_path")
    source_shards = _source_shards(source_manifest)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    schema = _teacher_schema(pq.ParquetFile(source_shards[0]).schema_arrow)
    writer = _TeacherShardWriter(
        staging_dir=staging_dir,
        final_dir=output_dir,
        schema=schema,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
    )
    counters: Counter[str] = Counter()
    observed_episodes: set[int] = set()
    try:
        for shard_path in source_shards:
            _stream_source_shard(
                shard_path,
                config=config,
                exact_signature=exact_signature,
                episode_by_id=episode_by_id,
                split_by_episode=split_by_episode,
                writer=writer,
                counters=counters,
                observed_episodes=observed_episodes,
            )
        writer.close()
        if counters["rows"] <= 0:
            raise ValueError("no exact public-pilot rows matched the teacher filters")
        coverage = len(observed_episodes) / len(episodes)
        manifest = {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "identity": identity,
            "config": config.model_dump(mode="json"),
            "source_manifest": deck_records.display_path(source_manifest_path),
            "replay_manifest": deck_records.display_path(replay_manifest_path),
            "output_dir": deck_records.display_path(output_dir),
            "deck_signature": exact_signature,
            "behavior_kind": "public_teacher",
            "split_column": "teacher_split",
            "sample_weight_column": "teacher_sample_weight",
            "summary": {
                **dict(sorted(counters.items())),
                "episodes_expected": len(episodes),
                "episodes_observed": len(observed_episodes),
                "episode_coverage": coverage,
                "episode_coverage_reference": config.min_episode_coverage,
                "diagnostic_warnings": (
                    ["episode_coverage"]
                    if coverage < config.min_episode_coverage
                    else []
                ),
                "train_episodes": sum(
                    split == "train" for split in split_by_episode.values()
                ),
                "validation_episodes": sum(
                    split == "validation" for split in split_by_episode.values()
                ),
            },
            "schema": schema.to_string(),
            "shards": writer.shards,
        }
        _write_json(staging_dir / "manifest.json", manifest)
        os.replace(staging_dir, output_dir)
        return manifest
    except BaseException:
        writer.close()
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _stream_source_shard(
    path: Path,
    *,
    config: PublicPilotTeacherConfig,
    exact_signature: str,
    episode_by_id: Mapping[int, PublicPilotEpisode],
    split_by_episode: Mapping[int, str],
    writer: _TeacherShardWriter,
    counters: Counter[str],
    observed_episodes: set[int],
) -> None:
    parquet_file = pq.ParquetFile(path)
    expected_source_schema = pa.schema(
        list(writer.schema)[: len(parquet_file.schema_arrow)]
    )
    if not parquet_file.schema_arrow.remove_metadata().equals(
        expected_source_schema.remove_metadata()
    ):
        raise ValueError(f"source teacher shard schema mismatch: {path}")
    for batch in parquet_file.iter_batches(batch_size=config.read_batch_size):
        table = pa.Table.from_batches([batch])
        compute = cast(Any, pc)
        team_mask = compute.equal(
            compute.utf8_lower(table["team_name"]),
            config.team_name.casefold(),
        )
        deck_mask = compute.equal(table["deck_signature"], exact_signature)
        filtered = table.filter(compute.and_(team_mask, deck_mask))
        if filtered.num_rows <= 0:
            continue
        episode_ids = [int(value) for value in filtered["episode_id"].to_pylist()]
        missing = sorted(set(episode_ids) - episode_by_id.keys())
        if missing:
            raise ValueError(
                f"source rows contain episodes outside frozen inventory: {missing[:5]}"
            )
        results = [
            str(value or "other").lower() for value in filtered["result"].to_pylist()
        ]
        splits = [split_by_episode[episode_id] for episode_id in episode_ids]
        create_times = [
            episode_by_id[episode_id].create_time_utc.isoformat()
            for episode_id in episode_ids
        ]
        weights = [
            config.target_weight * _outcome_weight(config.outcome_weights, result)
            for result in results
        ]
        augmented = filtered
        for name, values, data_type in (
            (
                "teacher_submission_id",
                [config.submission_id] * filtered.num_rows,
                pa.int64(),
            ),
            ("teacher_episode_create_time", create_times, pa.string()),
            ("teacher_split", splits, pa.string()),
            ("teacher_sample_weight", weights, pa.float32()),
            ("behavior_kind", ["public_teacher"] * filtered.num_rows, pa.string()),
        ):
            augmented = augmented.append_column(name, pa.array(values, type=data_type))
        writer.write(augmented)
        observed_episodes.update(episode_ids)
        counters["rows"] += filtered.num_rows
        for split in splits:
            counters[f"rows_{split}"] += 1
        for result in results:
            counters[f"result_{result}"] += 1
        for status in filtered["status"].to_pylist():
            counters[f"status_{str(status or 'unknown').lower()}"] += 1


def _teacher_schema(source_schema: pa.Schema) -> pa.Schema:
    additions = (
        pa.field("teacher_submission_id", pa.int64()),
        pa.field("teacher_episode_create_time", pa.string()),
        pa.field("teacher_split", pa.string()),
        pa.field("teacher_sample_weight", pa.float32()),
        pa.field("behavior_kind", pa.string()),
    )
    collisions = set(source_schema.names) & {field.name for field in additions}
    if collisions:
        raise ValueError(
            f"teacher metadata columns already exist: {sorted(collisions)}"
        )
    return pa.schema([*source_schema, *additions])


def _temporal_episode_split(
    episodes: Sequence[PublicPilotEpisode],
    *,
    validation_fraction: float,
) -> dict[int, str]:
    if len(episodes) < 2:
        raise ValueError("temporal teacher split requires at least two episodes")
    ordered = sorted(episodes, key=lambda row: (row.create_time_utc, row.episode_id))
    validation_count = min(
        len(ordered) - 1,
        max(1, math.ceil(len(ordered) * validation_fraction)),
    )
    validation_ids = {row.episode_id for row in ordered[-validation_count:]}
    return {
        row.episode_id: ("validation" if row.episode_id in validation_ids else "train")
        for row in ordered
    }


def _load_episodes(
    manifest: Mapping[str, Any],
    *,
    submission_id: int,
) -> tuple[PublicPilotEpisode, ...]:
    config = manifest.get("config")
    if (
        not isinstance(config, Mapping)
        or int(config.get("submission_id", 0)) != submission_id
    ):
        raise ValueError("replay manifest submission id mismatch")
    if not bool(manifest.get("inventory_frozen", False)):
        raise ValueError("teacher source replay inventory must be frozen")
    raw_episodes = manifest.get("episodes")
    if not isinstance(raw_episodes, Sequence) or isinstance(raw_episodes, str):
        raise ValueError("replay manifest episodes must be a list")
    episodes = tuple(PublicPilotEpisode.model_validate(row) for row in raw_episodes)
    if not episodes:
        raise ValueError("replay manifest has no episodes")
    if len({row.episode_id for row in episodes}) != len(episodes):
        raise ValueError("replay manifest contains duplicate episode ids")
    return episodes


def _source_shards(manifest: Mapping[str, Any]) -> tuple[Path, ...]:
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, str):
        raise ValueError("source step manifest shards must be a list")
    paths: list[Path] = []
    for row in raw_shards:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise ValueError("invalid source step shard entry")
        path = deck_records.repo_path(Path(str(row["path"])))
        if not path.exists():
            raise FileNotFoundError(f"source step shard does not exist: {path}")
        paths.append(path)
    if not paths:
        raise ValueError("source step manifest has no shards")
    return tuple(paths)


def _outcome_weight(config: TeacherOutcomeWeights, result: str) -> float:
    if result == "win":
        return config.win
    if result == "loss":
        return config.loss
    if result == "draw":
        return config.draw
    return config.other


def _build_identity(
    config: PublicPilotTeacherConfig,
    source_manifest_path: Path,
    replay_manifest_path: Path,
) -> dict[str, Any]:
    return {
        "config_sha256": hashlib.sha256(
            json.dumps(
                config.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "source_manifest_sha256": _sha256(source_manifest_path),
        "replay_manifest_sha256": _sha256(replay_manifest_path),
    }


def _verify_output_manifest(manifest: Mapping[str, Any]) -> None:
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, Sequence) or isinstance(raw_shards, str):
        raise ValueError("teacher manifest shards must be a list")
    for row in raw_shards:
        if not isinstance(row, Mapping):
            raise ValueError("invalid teacher shard entry")
        path = deck_records.repo_path(Path(str(row.get("path", ""))))
        if not path.exists() or path.stat().st_size != int(row.get("bytes", -1)):
            raise FileNotFoundError(f"teacher shard is missing or truncated: {path}")
        if _sha256(path) != str(row.get("sha256", "")):
            raise ValueError(f"teacher shard SHA256 mismatch: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
