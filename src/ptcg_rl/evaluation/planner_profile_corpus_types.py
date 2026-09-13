"""Shared immutable types and column names for planner profile corpora."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ptcg_rl.context import GameContextFeatures, GameContextSnapshot
from ptcg_rl.evaluation.consequence_audit_sampling import CaseLocator
from ptcg_rl.evaluation.planner_profile_config import PlannerDecisionShape
from ptcg_rl.evaluation.planner_profile_manifest import (
    PLANNER_PROFILE_CORPUS_SCHEMA_VERSION,
)

CORPUS_SCHEMA_VERSION = PLANNER_PROFILE_CORPUS_SCHEMA_VERSION
PROFILE_CASE_ID = "profile_case_id"
PROFILE_SHAPES = "profile_decision_shapes"
PROFILE_SOURCE_ROW = "profile_source_row_index"
PROFILE_SCHEMA_VERSION = "profile_corpus_schema_version"
PROFILE_OBSERVATION = "profile_observation_payload"
PROFILE_OBSERVATION_FINGERPRINT = "profile_observation_fingerprint"
PROFILE_OBSERVATION_CODEC = "profile_observation_codec_version"
PROFILE_CONTEXT_SNAPSHOT = "profile_context_snapshot_payload"
PROFILE_CONTEXT_FINGERPRINT = "profile_context_snapshot_fingerprint"
PROFILE_CONTEXT_CODEC = "profile_context_snapshot_codec_version"
PROFILE_REPLAY_SHA256 = "profile_replay_sha256"
PROFILE_SOURCE_CONTEXT_MATCH = "profile_source_context_match"
SCAN_COLUMNS = (
    "date",
    "episode_id",
    "step_index",
    "player_index",
    "select_context",
    "select_min_count",
    "select_max_count",
    "select_option_count",
)


@dataclass(frozen=True, slots=True)
class PlannerProfileCorpusRecord:
    """One reconstructed deployable root from a compact Parquet row."""

    row_id: str
    source_date: str
    source_episode_id: int
    source_step_index: int
    shapes: tuple[PlannerDecisionShape, ...]
    observation: Mapping[str, Any]
    context_features: GameContextFeatures
    context_snapshot: GameContextSnapshot
    observation_fingerprint: str
    context_snapshot_fingerprint: str
    replay_sha256: str
    source_context_match: bool
    own_deck: tuple[int, ...]
    executed_action: tuple[int, ...]
    final_root_outcome: int
    source_row_index: int


@dataclass(frozen=True, slots=True)
class PlannerProfileCorpusManifest:
    """Small identity and coverage record for a built corpus."""

    corpus_path: str
    corpus_sha256: str
    corpus_schema_version: int
    rows: int
    shape_counts: Mapping[str, int]
    source_files: int
    source_rows_scanned: int
    chance_evidence_files: int
    chance_case_ids: int
    rows_per_shape: int
    seed: str
    observation_codec_version: int
    context_snapshot_codec_version: int
    replay_assets: Mapping[str, str]
    replay_archives: Mapping[str, str]
    source_context_match_rows: int
    source_context_mismatch_rows: int
    source_context_drift_field_counts: Mapping[str, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "corpus_path": self.corpus_path,
            "corpus_sha256": self.corpus_sha256,
            "corpus_schema_version": self.corpus_schema_version,
            "rows": self.rows,
            "shape_counts": dict(sorted(self.shape_counts.items())),
            "source_files": self.source_files,
            "source_rows_scanned": self.source_rows_scanned,
            "chance_evidence_files": self.chance_evidence_files,
            "chance_case_ids": self.chance_case_ids,
            "rows_per_shape": self.rows_per_shape,
            "seed": self.seed,
            "observation_codec_version": self.observation_codec_version,
            "context_snapshot_codec_version": self.context_snapshot_codec_version,
            "replay_assets": dict(sorted(self.replay_assets.items())),
            "replay_archives": dict(sorted(self.replay_archives.items())),
            "source_context_match_rows": self.source_context_match_rows,
            "source_context_mismatch_rows": self.source_context_mismatch_rows,
            "source_context_drift_field_counts": dict(
                sorted(self.source_context_drift_field_counts.items())
            ),
            "source_context_comparison_note": (
                "The compact selectors are legacy schema-v5 rows. Drift is a "
                "diagnostic against replay-derived current GameContext semantics; "
                "the exact replay observation and snapshot are authoritative."
            ),
        }


@dataclass(frozen=True, slots=True)
class SelectedSourceRoot:
    """Compact source evidence cross-checked against the exact replay root."""

    locator: CaseLocator
    own_deck: tuple[int, ...]
    context_features: GameContextFeatures
    search_begin_input: str | bytes


@dataclass(frozen=True, slots=True)
class ExactProfileRoot:
    """Versioned exact observation and online context payloads for one root."""

    executed_action: tuple[int, ...]
    observation_payload: bytes
    observation_fingerprint: str
    context_snapshot_payload: bytes
    context_snapshot_fingerprint: str
    replay_sha256: str
    source_context_match: bool
    source_context_drift_fields: tuple[str, ...]


__all__ = [
    "CORPUS_SCHEMA_VERSION",
    "ExactProfileRoot",
    "PlannerProfileCorpusManifest",
    "PlannerProfileCorpusRecord",
    "PROFILE_CASE_ID",
    "PROFILE_CONTEXT_CODEC",
    "PROFILE_CONTEXT_FINGERPRINT",
    "PROFILE_CONTEXT_SNAPSHOT",
    "PROFILE_OBSERVATION",
    "PROFILE_OBSERVATION_CODEC",
    "PROFILE_OBSERVATION_FINGERPRINT",
    "PROFILE_REPLAY_SHA256",
    "PROFILE_SCHEMA_VERSION",
    "PROFILE_SHAPES",
    "PROFILE_SOURCE_CONTEXT_MATCH",
    "PROFILE_SOURCE_ROW",
    "SCAN_COLUMNS",
    "SelectedSourceRoot",
]
