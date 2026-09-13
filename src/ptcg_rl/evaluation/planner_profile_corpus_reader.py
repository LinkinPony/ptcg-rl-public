"""Streaming decode and integrity checks for immutable planner corpora."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.context import (
    GameContext,
    context_features_from_observation,
    context_features_from_row,
)
from ptcg_rl.decks.identity import parse_canonical_signature
from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_config import (
    REQUIRED_PLANNER_PROFILE_SHAPES,
    PlannerDecisionShape,
)
from ptcg_rl.evaluation.planner_profile_context import (
    PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION,
    PROFILE_OBSERVATION_CODEC_VERSION,
    decode_profile_observation,
    profile_observation_fingerprint,
    verified_profile_context_snapshot,
)
from ptcg_rl.evaluation.planner_profile_corpus_types import (
    CORPUS_SCHEMA_VERSION,
    PROFILE_CASE_ID,
    PROFILE_CONTEXT_CODEC,
    PROFILE_CONTEXT_FINGERPRINT,
    PROFILE_CONTEXT_SNAPSHOT,
    PROFILE_OBSERVATION,
    PROFILE_OBSERVATION_CODEC,
    PROFILE_OBSERVATION_FINGERPRINT,
    PROFILE_REPLAY_SHA256,
    PROFILE_SCHEMA_VERSION,
    PROFILE_SHAPES,
    PROFILE_SOURCE_CONTEXT_MATCH,
    PROFILE_SOURCE_ROW,
    PlannerProfileCorpusRecord,
)


class PlannerProfileCorpusReader:
    """Stream immutable selected rows without materializing the corpus."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        batch_size: int = 256,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("planner corpus reader batch_size must be positive")
        if not path.is_file():
            raise FileNotFoundError(f"planner decision corpus not found: {path}")
        actual = file_sha256(path)
        if actual != expected_sha256:
            raise ValueError("planner decision corpus fingerprint differs from config")
        parquet_file = pq.ParquetFile(path)
        _require_profile_columns(parquet_file.schema_arrow.names, path)
        self.path = path
        self.sha256 = actual
        self.rows = int(parquet_file.metadata.num_rows)
        self._batch_size = int(batch_size)

    def __iter__(self) -> Iterator[PlannerProfileCorpusRecord]:
        parquet_file = pq.ParquetFile(self.path)
        for batch in parquet_file.iter_batches(
            batch_size=self._batch_size,
            use_threads=True,
        ):
            for raw in batch.to_pylist():
                yield _corpus_record(cast(Mapping[str, Any], raw))

    def shape_counts(self) -> Mapping[str, int]:
        """Count the small profile label column in a streaming pass."""
        counts: Counter[str] = Counter()
        parquet_file = pq.ParquetFile(self.path)
        for batch in parquet_file.iter_batches(
            batch_size=self._batch_size,
            columns=[PROFILE_SHAPES],
            use_threads=True,
        ):
            for labels in batch.column(0).to_pylist():
                counts.update(str(label) for label in labels or ())
        return dict(sorted(counts.items()))

    def validate_records(self) -> int:
        """Decode every exact root payload before starting mutable work."""
        decoded = sum(1 for _record in self)
        if decoded != self.rows:
            raise ValueError("planner corpus decoded row count differs from metadata")
        return decoded


def exact_prompt_identity(
    observation: Mapping[str, Any],
) -> tuple[int, int, int, int, int]:
    """Return the persisted identity fields for one exact replay prompt."""
    current = observation.get("current")
    select = observation.get("select")
    if not isinstance(current, Mapping) or not isinstance(select, Mapping):
        raise ValueError("exact profile root lacks current/select mappings")
    options = select.get("option")
    if not isinstance(options, Sequence) or isinstance(
        options,
        (str, bytes, bytearray),
    ):
        raise ValueError("exact profile root has malformed select options")
    return (
        int(current.get("yourIndex", -1)),
        int(select.get("context", -1)),
        int(select.get("minCount", -1)),
        int(select.get("maxCount", -1)),
        len(options),
    )


def _corpus_record(row: Mapping[str, Any]) -> PlannerProfileCorpusRecord:
    row_id = str(row[PROFILE_CASE_ID])
    raw_shapes = tuple(str(value) for value in row[PROFILE_SHAPES])
    if not raw_shapes or any(
        value not in REQUIRED_PLANNER_PROFILE_SHAPES for value in raw_shapes
    ):
        raise ValueError(f"planner corpus row {row_id} has invalid decision shapes")
    if int(row[PROFILE_SCHEMA_VERSION]) != CORPUS_SCHEMA_VERSION:
        raise ValueError(f"planner corpus row {row_id} has another schema version")
    if int(row[PROFILE_OBSERVATION_CODEC]) != PROFILE_OBSERVATION_CODEC_VERSION:
        raise ValueError(f"planner corpus row {row_id} has another observation codec")
    if int(row[PROFILE_CONTEXT_CODEC]) != PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION:
        raise ValueError(f"planner corpus row {row_id} has another context codec")
    observation_payload = bytes(row[PROFILE_OBSERVATION])
    observation_fingerprint = str(row[PROFILE_OBSERVATION_FINGERPRINT])
    if profile_observation_fingerprint(observation_payload) != (
        observation_fingerprint
    ):
        raise ValueError(f"planner corpus row {row_id} observation is corrupted")
    observation = decode_profile_observation(observation_payload)
    snapshot_fingerprint = str(row[PROFILE_CONTEXT_FINGERPRINT])
    context_snapshot = verified_profile_context_snapshot(
        bytes(row[PROFILE_CONTEXT_SNAPSHOT]),
        expected_fingerprint=snapshot_fingerprint,
    )
    state_token = observation.get("search_begin_input")
    if not isinstance(state_token, (str, bytes)) or not state_token:
        raise ValueError(f"planner corpus row {row_id} lacks an engine state token")
    if state_token != row.get("search_begin_input"):
        raise ValueError(f"planner corpus row {row_id} changes the engine state")
    prompt_identity = exact_prompt_identity(observation)
    persisted_prompt_identity = (
        int(row.get("player_index", -1)),
        int(row.get("select_context", -1)),
        int(row.get("select_min_count", -1)),
        int(row.get("select_max_count", -1)),
        int(row.get("select_option_count", -1)),
    )
    if prompt_identity != persisted_prompt_identity:
        raise ValueError(f"planner corpus row {row_id} changes the select prompt")
    executed_action = _executed_action(row, observation, row_id=row_id)
    final_root_outcome = _final_root_outcome(row, row_id=row_id)
    own = parse_canonical_signature(str(row.get("deck_signature") or ""))
    if context_snapshot.player_index != int(row.get("player_index", -1)):
        raise ValueError(f"planner corpus row {row_id} changes player perspective")
    expected_deck_counts = tuple(sorted(Counter(own.card_ids).items()))
    if context_snapshot.own_deck_counts != expected_deck_counts:
        raise ValueError(f"planner corpus row {row_id} changes the registered deck")
    context_features = context_features_from_observation(observation)
    source_context_match = bool(row[PROFILE_SOURCE_CONTEXT_MATCH])
    if source_context_match != (context_features == context_features_from_row(row)):
        raise ValueError(f"planner corpus row {row_id} has false context provenance")
    if GameContext.from_snapshot(context_snapshot).features(observation) != (
        context_features
    ):
        raise ValueError(f"planner corpus row {row_id} snapshot is not root-aligned")
    replay_sha256 = str(row[PROFILE_REPLAY_SHA256])
    if len(replay_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in replay_sha256
    ):
        raise ValueError(f"planner corpus row {row_id} has invalid replay identity")
    return PlannerProfileCorpusRecord(
        row_id=row_id,
        source_date=str(row["date"]),
        source_episode_id=int(row["episode_id"]),
        source_step_index=int(row["step_index"]),
        shapes=cast(tuple[PlannerDecisionShape, ...], raw_shapes),
        observation=observation,
        context_features=context_features,
        context_snapshot=context_snapshot,
        observation_fingerprint=observation_fingerprint,
        context_snapshot_fingerprint=snapshot_fingerprint,
        replay_sha256=replay_sha256,
        source_context_match=source_context_match,
        own_deck=own.card_ids,
        executed_action=executed_action,
        final_root_outcome=final_root_outcome,
        source_row_index=int(row[PROFILE_SOURCE_ROW]),
    )


def _executed_action(
    row: Mapping[str, Any],
    observation: Mapping[str, Any],
    *,
    row_id: str,
) -> tuple[int, ...]:
    raw_action = row.get("action")
    if not isinstance(raw_action, Sequence) or isinstance(raw_action, (str, bytes)):
        raise ValueError(f"planner corpus row {row_id} lacks an executed action")
    if any(
        isinstance(index, bool) or not isinstance(index, int) for index in raw_action
    ):
        raise ValueError(f"planner corpus row {row_id} has a non-integer action")
    action = tuple(raw_action)
    select = observation.get("select")
    if not isinstance(select, Mapping) or not is_legal_action(select, action):
        raise ValueError(f"planner corpus row {row_id} has an illegal executed action")
    return action


def _final_root_outcome(row: Mapping[str, Any], *, row_id: str) -> int:
    raw_reward = row.get("reward")
    if (
        isinstance(raw_reward, bool)
        or not isinstance(raw_reward, (int, float))
        or not math.isfinite(float(raw_reward))
        or float(raw_reward) not in (-1.0, 0.0, 1.0)
    ):
        raise ValueError(
            f"planner corpus row {row_id} reward is not a root-player W/D/L outcome"
        )
    return int(raw_reward)


def _require_profile_columns(names: Sequence[str], path: Path) -> None:
    required = {
        PROFILE_CASE_ID,
        PROFILE_SHAPES,
        PROFILE_SOURCE_ROW,
        "search_begin_input",
        "date",
        "episode_id",
        "step_index",
        "action",
        "reward",
        "deck_signature",
        PROFILE_SCHEMA_VERSION,
        PROFILE_OBSERVATION,
        PROFILE_OBSERVATION_FINGERPRINT,
        PROFILE_OBSERVATION_CODEC,
        PROFILE_CONTEXT_SNAPSHOT,
        PROFILE_CONTEXT_FINGERPRINT,
        PROFILE_CONTEXT_CODEC,
        PROFILE_REPLAY_SHA256,
        PROFILE_SOURCE_CONTEXT_MATCH,
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"planner corpus {path} is missing columns: {missing}")


__all__ = [
    "PlannerProfileCorpusReader",
    "PlannerProfileCorpusRecord",
    "exact_prompt_identity",
]
