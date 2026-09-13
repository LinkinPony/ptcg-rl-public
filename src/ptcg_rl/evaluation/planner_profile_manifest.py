"""Immutable manifest validation for the planner profiling corpus."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Protocol, TypeVar

from ptcg_rl.context import GameContextFeatures
from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_context import (
    PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION,
    PROFILE_OBSERVATION_CODEC_VERSION,
)

PLANNER_PROFILE_CORPUS_SCHEMA_VERSION = 3


class _CorpusRecord(Protocol):
    """Fields required to validate one streamed corpus row."""

    @property
    def replay_sha256(self) -> str: ...

    @property
    def source_context_match(self) -> bool: ...


_CorpusRecordT = TypeVar("_CorpusRecordT", bound=_CorpusRecord, covariant=True)


class _CorpusReader(Protocol[_CorpusRecordT]):
    """Reader surface required for manifest validation."""

    path: Path
    sha256: str
    rows: int

    def __iter__(self) -> Iterator[_CorpusRecordT]: ...


@dataclass(frozen=True, slots=True)
class PlannerProfileCorpusValidation:
    """Validated manifest identity and streamed corpus provenance."""

    manifest_sha256: str
    validated_rows: int
    source_context_match_rows: int
    source_context_mismatch_rows: int
    replay_assets: int
    replay_archives: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "validated_rows": self.validated_rows,
            "source_context_match_rows": self.source_context_match_rows,
            "source_context_mismatch_rows": self.source_context_mismatch_rows,
            "replay_assets": self.replay_assets,
            "replay_archives": self.replay_archives,
        }


def validate_planner_profile_corpus_manifest(
    reader: _CorpusReader[_CorpusRecord],
    manifest_path: Path,
    *,
    expected_sha256: str,
    shape_counts: Mapping[str, int],
) -> PlannerProfileCorpusValidation:
    """Bind streamed exact roots to their immutable build provenance."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"planner corpus manifest not found: {manifest_path}")
    manifest_sha256 = file_sha256(manifest_path)
    if manifest_sha256 != expected_sha256:
        raise ValueError("planner corpus manifest fingerprint differs from config")
    with manifest_path.open("r", encoding="utf-8") as source:
        raw = json.load(source)
    if not isinstance(raw, Mapping):
        raise ValueError("planner corpus manifest must be a mapping")

    manifest_corpus_path = raw.get("corpus_path")
    if (
        not isinstance(manifest_corpus_path, str)
        or Path(manifest_corpus_path) != reader.path
    ):
        raise ValueError("planner corpus manifest names another corpus path")
    if _manifest_sha256(raw, "corpus_sha256") != reader.sha256:
        raise ValueError("planner corpus manifest names another corpus fingerprint")
    if _manifest_integer(raw, "corpus_schema_version") != (
        PLANNER_PROFILE_CORPUS_SCHEMA_VERSION
    ):
        raise ValueError("planner corpus manifest schema is incompatible")
    if _manifest_integer(raw, "observation_codec_version") != (
        PROFILE_OBSERVATION_CODEC_VERSION
    ):
        raise ValueError("planner corpus manifest observation codec is incompatible")
    if _manifest_integer(raw, "context_snapshot_codec_version") != (
        PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION
    ):
        raise ValueError("planner corpus manifest context codec is incompatible")
    manifest_rows = _manifest_integer(raw, "rows")
    if manifest_rows != reader.rows:
        raise ValueError("planner corpus manifest row count differs from Parquet")
    manifest_shapes = _manifest_count_mapping(raw, "shape_counts")
    expected_shapes = {
        str(name): int(count) for name, count in sorted(shape_counts.items())
    }
    if manifest_shapes != expected_shapes:
        raise ValueError("planner corpus manifest shape counts differ from Parquet")

    if _manifest_integer(raw, "source_files") <= 0:
        raise ValueError("planner corpus manifest has no source files")
    if _manifest_integer(raw, "source_rows_scanned") < manifest_rows:
        raise ValueError("planner corpus manifest scanned fewer rows than it retained")
    if _manifest_integer(raw, "chance_evidence_files") <= 0:
        raise ValueError("planner corpus manifest has no chance-evidence files")
    if _manifest_integer(raw, "chance_case_ids") <= 0:
        raise ValueError("planner corpus manifest has no chance-evidence roots")
    if _manifest_integer(raw, "rows_per_shape") <= 0:
        raise ValueError("planner corpus manifest has no per-shape capacity")
    seed = raw.get("seed")
    if not isinstance(seed, str) or not seed.strip():
        raise ValueError("planner corpus manifest has no deterministic seed")

    replay_assets = _manifest_sha_mapping(raw, "replay_assets", allow_empty=False)
    replay_archives = _manifest_sha_mapping(raw, "replay_archives", allow_empty=True)
    if len(replay_assets) > manifest_rows:
        raise ValueError("planner corpus manifest has more replays than roots")

    decoded = 0
    source_context_match_rows = 0
    record_replay_sha256s: set[str] = set()
    for record in reader:
        decoded += 1
        source_context_match_rows += int(record.source_context_match)
        record_replay_sha256s.add(record.replay_sha256)
    if decoded != manifest_rows:
        raise ValueError("planner corpus decoded row count differs from manifest")
    if record_replay_sha256s != set(replay_assets.values()):
        raise ValueError("planner corpus replay provenance differs from manifest")

    source_context_mismatch_rows = decoded - source_context_match_rows
    if _manifest_integer(raw, "source_context_match_rows") != (
        source_context_match_rows
    ):
        raise ValueError("planner corpus context-match count differs from manifest")
    if _manifest_integer(raw, "source_context_mismatch_rows") != (
        source_context_mismatch_rows
    ):
        raise ValueError("planner corpus context-drift count differs from manifest")
    drift_counts = _manifest_count_mapping(
        raw,
        "source_context_drift_field_counts",
    )
    valid_drift_fields = {field.name for field in fields(GameContextFeatures)}
    if any(name not in valid_drift_fields for name in drift_counts):
        raise ValueError("planner corpus manifest names an unknown context field")
    if any(count > source_context_mismatch_rows for count in drift_counts.values()):
        raise ValueError("planner corpus context-field drift exceeds drift rows")
    if source_context_mismatch_rows:
        if (
            not drift_counts
            or sum(drift_counts.values()) < source_context_mismatch_rows
        ):
            raise ValueError("planner corpus manifest omits context drift provenance")
    elif drift_counts:
        raise ValueError("planner corpus manifest claims drift without drift rows")
    comparison_note = raw.get("source_context_comparison_note")
    if not isinstance(comparison_note, str) or not comparison_note.strip():
        raise ValueError("planner corpus manifest omits context provenance semantics")

    return PlannerProfileCorpusValidation(
        manifest_sha256=manifest_sha256,
        validated_rows=decoded,
        source_context_match_rows=source_context_match_rows,
        source_context_mismatch_rows=source_context_mismatch_rows,
        replay_assets=len(replay_assets),
        replay_archives=len(replay_archives),
    )


def _manifest_integer(manifest: Mapping[str, Any], name: str) -> int:
    value = manifest.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"planner corpus manifest {name} must be non-negative")
    return value


def _manifest_sha256(manifest: Mapping[str, Any], name: str) -> str:
    value = manifest.get(name)
    if not isinstance(value, str) or not _is_sha256(value):
        raise ValueError(f"planner corpus manifest {name} must be SHA-256")
    return value


def _manifest_count_mapping(
    manifest: Mapping[str, Any],
    name: str,
) -> dict[str, int]:
    value = manifest.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"planner corpus manifest {name} must be a mapping")
    result: dict[str, int] = {}
    for raw_key, raw_count in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"planner corpus manifest {name} has an invalid key")
        if (
            isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise ValueError(f"planner corpus manifest {name} counts must be positive")
        result[raw_key] = raw_count
    return dict(sorted(result.items()))


def _manifest_sha_mapping(
    manifest: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool,
) -> dict[str, str]:
    value = manifest.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"planner corpus manifest {name} must be a mapping")
    result: dict[str, str] = {}
    for raw_key, raw_sha256 in value.items():
        if not isinstance(raw_key, str) or not raw_key:
            raise ValueError(f"planner corpus manifest {name} has an invalid key")
        if not isinstance(raw_sha256, str) or not _is_sha256(raw_sha256):
            raise ValueError(f"planner corpus manifest {name} has an invalid SHA-256")
        result[raw_key] = raw_sha256
    if not result and not allow_empty:
        raise ValueError(f"planner corpus manifest {name} cannot be empty")
    return dict(sorted(result.items()))


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


__all__ = [
    "PLANNER_PROFILE_CORPUS_SCHEMA_VERSION",
    "PlannerProfileCorpusValidation",
    "validate_planner_profile_corpus_manifest",
]
