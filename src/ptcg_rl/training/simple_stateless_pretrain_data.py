"""Compact, recoverable NPZ shards for public replay pretraining."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import uuid
from collections import Counter
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Literal, Self, TypeAlias, cast

import numpy as np
import numpy.typing as npt
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from ptcg_rl.actions.encoding import (
    SCALAR_FEATURE_SIZE,
    EncodedOptionArrayFeatures,
)
from ptcg_rl.belief.public_catalog import PublicDeckPosteriorArrays
from ptcg_rl.context.public_event_arrays import (
    PublicEventArrayBlock,
    build_public_event_array_block,
    validate_public_event_array_block,
)
from ptcg_rl.context.public_events import PublicEventDelta
from ptcg_rl.decks.identity import CanonicalDeck, canonicalize_deck
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS
from ptcg_rl.model.sequence.action import (
    ACCEPTED_ACTION_SCHEMA_VERSION,
    AcceptedActionRecord,
)
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE, StateTokenArrayFeatures
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity

LEGACY_PRETRAINING_SHARD_SCHEMA: Final[
    Literal["simple-stateless-replay-pretraining-npz-v1"]
] = "simple-stateless-replay-pretraining-npz-v1"
PRETRAINING_SHARD_SCHEMA: Final[
    Literal["simple-stateless-replay-pretraining-npz-v2"]
] = "simple-stateless-replay-pretraining-npz-v2"
LEGACY_TEMPORAL_PRETRAINING_SHARD_SCHEMA: Final[
    Literal["simple-stateless-sequence-pretraining-npz-v3"]
] = "simple-stateless-sequence-pretraining-npz-v3"
TEMPORAL_PRETRAINING_SHARD_SCHEMA: Final[
    Literal["simple-stateless-sequence-pretraining-npz-v4"]
] = "simple-stateless-sequence-pretraining-npz-v4"
TEMPORAL_PRETRAINING_SHARD_SCHEMAS: Final[frozenset[str]] = frozenset(
    {
        LEGACY_TEMPORAL_PRETRAINING_SHARD_SCHEMA,
        TEMPORAL_PRETRAINING_SHARD_SCHEMA,
    }
)


def is_temporal_pretraining_format(value: str) -> bool:
    """Return whether a dataset uses either supported causal replay schema."""
    return value in TEMPORAL_PRETRAINING_SHARD_SCHEMAS

ReplaySplit: TypeAlias = Literal["train", "validation", "test"]
REPLAY_SPLITS: Final[tuple[ReplaySplit, ...]] = (
    "train",
    "validation",
    "test",
)
_SPLIT_TO_CODE: Final[dict[ReplaySplit, int]] = {
    split: index for index, split in enumerate(REPLAY_SPLITS)
}
_CODE_TO_SPLIT: Final[dict[int, ReplaySplit]] = {
    code: split for split, code in _SPLIT_TO_CODE.items()
}
_METADATA_COLUMN_DTYPES: Final[dict[str, npt.DTypeLike]] = {
    "episode_ids": np.int64,
    "player_indices": np.int8,
    "date_indices": np.int8,
    "split_codes": np.int8,
    "outcomes": np.float32,
    "example_weights": np.float32,
    "route_expert_ids": np.dtype("<U64"),
    "decision_indices": np.int32,
    "sequence_offsets": np.int64,
    "state_offsets": np.int64,
    "option_offsets": np.int64,
}
_ROW_OFFSET_METADATA_COLUMNS: Final[frozenset[str]] = frozenset(
    {"state_offsets", "option_offsets"}
)

_V1_ARRAY_KEYS = frozenset(
    {
        "schema_version",
        "episode_ids",
        "player_indices",
        "step_indices",
        "team_indices",
        "date_indices",
        "replay_indices",
        "own_decks",
        "opponent_decks",
        "outcomes",
        "example_weights",
        "state_offsets",
        "state_card_ids",
        "state_areas",
        "state_owner_roles",
        "state_token_kinds",
        "state_scalars",
        "state_last_attack_ids",
        "state_entity_slots",
        "attachment_offsets",
        "attachment_card_ids",
        "attachment_parent_indices",
        "attachment_kinds",
        "option_offsets",
        "option_types",
        "option_contexts",
        "option_entity_slots",
        "option_entity_slot_masks",
        "option_attack_ids",
        "option_card_ids",
        "option_scalars",
        "option_dynamic_effect_features",
        "option_dynamic_effect_masks",
        "min_counts",
        "max_counts",
        "belief_offsets",
        "belief_card_ids",
        "belief_expected_counts",
        "belief_scalars",
        "known_offsets",
        "known_card_ids",
        "known_counts",
        "action_offsets",
        "action_choices",
    }
)
_V2_ARRAY_KEYS = _V1_ARRAY_KEYS | {"split_codes"}
_V3_ARRAY_KEYS = _V2_ARRAY_KEYS | {
    "decision_indices",
    "source_steps",
    "source_replay_sha256s",
    "route_expert_ids",
    "sequence_offsets",
    "sequence_episode_ids",
    "sequence_seats",
    "model_config_fingerprint",
    "exact_registry_fingerprint",
    "public_catalog_fingerprint",
    "input_contract_fingerprint",
    "event_contract_fingerprint",
    "sequence_contract_fingerprint",
    "engine_fact_available",
    "event_offsets",
    "event_types",
    "event_actor_roles",
    "event_from_areas",
    "event_to_areas",
    "event_card_ids",
    "event_serials",
    "event_entity_mask",
    "event_attack_ids",
    "event_attack_id_mask",
    "event_values",
    "event_value_mask",
    "event_categorical_values",
    "event_overflow_offsets",
    "event_overflow_types",
    "event_overflow_actor_roles",
    "event_overflow_counts",
    "accepted_action_offsets",
    "accepted_action_schema_versions",
    "accepted_action_stable_ids",
    "accepted_action_prompt_contexts",
    "accepted_action_ordered",
    "accepted_action_stop_sampled",
    "accepted_action_fallback",
    "accepted_action_option_types",
    "accepted_action_option_contexts",
    "accepted_action_card_ids",
    "accepted_action_attack_ids",
    "accepted_action_option_scalars",
    "accepted_action_entity_card_ids",
    "accepted_action_entity_areas",
    "accepted_action_entity_owner_roles",
    "accepted_action_entity_token_kinds",
    "accepted_action_entity_scalars",
}
_V4_ARRAY_KEYS = _V3_ARRAY_KEYS | {"engine_fact_producer_fingerprint"}


@dataclass(frozen=True)
class ReplayPretrainingExample:
    """One leak-free demonstrated decision and learner-only targets."""

    episode_id: int
    player_index: int
    step_index: int
    team_index: int
    date_index: int
    replay_index: int
    split: ReplaySplit
    actor_row: SimpleStatelessActorRow
    action: tuple[int, ...]
    opponent_deck: tuple[int, ...]
    known_opponent_counts: tuple[tuple[int, int], ...]
    outcome: float
    example_weight: float
    decision_index: int | None = None
    source_replay_sha256: str | None = None
    route_expert_id: str | None = None
    accepted_action: AcceptedActionRecord | None = None


class PretrainingPartRecord(BaseModel):
    """One immutable compact data part."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    sha256: str
    size_bytes: int = Field(ge=1)
    examples: int = Field(ge=1)
    source_replays: int = Field(ge=1)
    split_examples: dict[ReplaySplit, int] = Field(default_factory=dict)

    @field_validator("filename")
    @classmethod
    def valid_filename(cls, value: str) -> str:
        """Keep parts relocatable under one directory."""
        if Path(value).name != value or not value.endswith(".npz"):
            raise ValueError("pretraining part filename is invalid")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a lowercase SHA-256."""
        return _validate_sha256(value)

    @field_validator("split_examples")
    @classmethod
    def valid_split_examples(
        cls,
        value: dict[ReplaySplit, int],
    ) -> dict[ReplaySplit, int]:
        """Validate optional V2 per-part split counts."""
        if value and (
            set(value) != set(REPLAY_SPLITS)
            or any(count < 0 for count in value.values())
            or sum(value.values()) <= 0
        ):
            raise ValueError("pretraining part split counts are invalid")
        return value

    @model_validator(mode="after")
    def coherent_split_examples(self) -> Self:
        """Keep optional V2 counts aligned with the physical row count."""
        if self.split_examples and sum(self.split_examples.values()) != self.examples:
            raise ValueError("pretraining part split counts differ from examples")
        return self


class RejectedReplayRecord(BaseModel):
    """One source replay excluded after bounded, recorded extraction attempts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_index: int = Field(ge=0)
    episode_id: int = Field(ge=0)
    date: str = Field(min_length=1)
    split: ReplaySplit = "train"
    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    expected_sha256: str | None
    observed_sha256: str | None
    attempts: int = Field(ge=1, le=5)
    error_kind: str = Field(min_length=1, max_length=128)
    error_message: str = Field(min_length=1, max_length=2048)

    @field_validator("relative_path")
    @classmethod
    def safe_relative_path(cls, value: str) -> str:
        """Keep source provenance relative and non-escaping."""
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("rejected replay path escapes the source root")
        return value

    @field_validator("expected_sha256", "observed_sha256")
    @classmethod
    def valid_optional_sha256(cls, value: str | None) -> str | None:
        """Validate source hashes while allowing an unreadable file."""
        return None if value is None else _validate_sha256(value)


class ReplayPretrainingDatasetManifest(BaseModel):
    """Durable source cursor and immutable compact-part identity."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "simple-stateless-replay-pretraining-npz-v1",
        "simple-stateless-replay-pretraining-npz-v2",
        "simple-stateless-sequence-pretraining-npz-v3",
        "simple-stateless-sequence-pretraining-npz-v4",
    ] = PRETRAINING_SHARD_SCHEMA
    source_manifest_sha256: str
    replay_manifest_sha256: str
    top_teams_sha256: str | None
    episode_teams_sha256: str | None = None
    source_selection: Literal[
        "team_allowlist", "episode_team_bindings", "all_sides"
    ] = "team_allowlist"
    public_catalog_fingerprint: str
    input_contract_fingerprint: str
    model_config_fingerprint: str | None = None
    exact_registry_fingerprint: str | None = None
    event_contract_fingerprint: str | None = None
    sequence_contract_fingerprint: str | None = None
    engine_fact_producer_fingerprint: str | None = None
    target_deck_digest: str | None = None
    source_replays: int = Field(ge=1)
    source_cursor: int = Field(default=0, ge=0)
    examples_committed: int = Field(default=0, ge=0)
    complete: bool = False
    counters: dict[str, int] = Field(default_factory=dict)
    split_assignment_fingerprint: str | None = None
    split_source_replays: dict[ReplaySplit, int] = Field(default_factory=dict)
    split_examples_committed: dict[ReplaySplit, int] = Field(default_factory=dict)
    dates: tuple[str, ...]
    teams: tuple[str, ...]
    parts: tuple[PretrainingPartRecord, ...] = ()
    rejections: tuple[RejectedReplayRecord, ...] = ()

    @field_validator(
        "source_manifest_sha256",
        "replay_manifest_sha256",
        "top_teams_sha256",
        "episode_teams_sha256",
        "public_catalog_fingerprint",
        "input_contract_fingerprint",
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "event_contract_fingerprint",
        "sequence_contract_fingerprint",
        "engine_fact_producer_fingerprint",
        "target_deck_digest",
        "split_assignment_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str | None) -> str | None:
        """Require all dataset identity fields to be SHA-256."""
        return None if value is None else _validate_sha256(value)

    @model_validator(mode="after")
    def coherent_cursor(self) -> Self:
        """Keep the durable cursor and rejected source identities consistent."""
        if self.source_selection in {"team_allowlist", "episode_team_bindings"}:
            if self.top_teams_sha256 is None or not self.teams:
                raise ValueError(
                    "explicit-selection dataset has no cohort fingerprint or teams"
                )
        elif self.top_teams_sha256 is not None:
            raise ValueError("all-sides dataset cannot carry a team cohort fingerprint")
        if (
            self.source_selection == "episode_team_bindings"
        ) != (self.episode_teams_sha256 is not None):
            raise ValueError(
                "episode-team selection has an inconsistent binding fingerprint"
            )
        if self.source_cursor > self.source_replays:
            raise ValueError("dataset source cursor exceeds its inventory")
        rejected_indices = tuple(item.source_index for item in self.rejections)
        if rejected_indices != tuple(sorted(set(rejected_indices))) or any(
            index >= self.source_cursor for index in rejected_indices
        ):
            raise ValueError("dataset rejections do not precede its source cursor")
        if any(value < 0 for value in self.counters.values()):
            raise ValueError("dataset counters cannot be negative")
        if self.format in {
            PRETRAINING_SHARD_SCHEMA,
            *TEMPORAL_PRETRAINING_SHARD_SCHEMAS,
        }:
            if self.split_assignment_fingerprint is None:
                raise ValueError("V2 pretraining dataset has no split fingerprint")
            for counts, total, label in (
                (
                    self.split_source_replays,
                    self.source_replays,
                    "source replay",
                ),
                (
                    self.split_examples_committed,
                    self.examples_committed,
                    "example",
                ),
            ):
                if (
                    set(counts) != set(REPLAY_SPLITS)
                    or any(count < 0 for count in counts.values())
                    or sum(counts.values()) != total
                ):
                    raise ValueError(f"V2 pretraining {label} split counts are invalid")
            if any(not part.split_examples for part in self.parts):
                raise ValueError("pretraining part has no split counts")
            temporal_contracts = (
                self.model_config_fingerprint,
                self.exact_registry_fingerprint,
                self.event_contract_fingerprint,
                self.sequence_contract_fingerprint,
            )
            if self.format in TEMPORAL_PRETRAINING_SHARD_SCHEMAS and any(
                value is None for value in temporal_contracts
            ):
                raise ValueError("temporal pretraining dataset has incomplete contracts")
            if (
                self.format == LEGACY_TEMPORAL_PRETRAINING_SHARD_SCHEMA
                and self.engine_fact_producer_fingerprint is not None
            ):
                raise ValueError("V3 pretraining dataset cannot bind engine facts")
            if self.format == PRETRAINING_SHARD_SCHEMA and any(
                value is not None
                for value in (
                    *temporal_contracts,
                    self.engine_fact_producer_fingerprint,
                )
            ):
                raise ValueError("V2 pretraining dataset carries temporal contracts")
        elif (
            self.target_deck_digest is not None
            or self.split_assignment_fingerprint is not None
            or self.split_source_replays
            or self.split_examples_committed
            or any(part.split_examples for part in self.parts)
            or self.model_config_fingerprint is not None
            or self.exact_registry_fingerprint is not None
            or self.event_contract_fingerprint is not None
            or self.sequence_contract_fingerprint is not None
            or self.engine_fact_producer_fingerprint is not None
        ):
            raise ValueError("legacy pretraining dataset carries V2 split metadata")
        return self

    @property
    def fingerprint(self) -> str:
        """Hash the complete immutable dataset once extraction is complete."""
        if not self.complete or self.source_cursor != self.source_replays:
            raise ValueError("incomplete pretraining dataset has no final fingerprint")
        legacy = self.format == LEGACY_PRETRAINING_SHARD_SCHEMA
        payload = self.model_dump(mode="json", exclude={"complete"})
        if self.source_selection == "team_allowlist":
            # Preserve immutable V1/V2/V3 fingerprints published before source
            # selection became explicit.
            payload.pop("source_selection")
        if self.format not in TEMPORAL_PRETRAINING_SHARD_SCHEMAS:
            for field in (
                "model_config_fingerprint",
                "exact_registry_fingerprint",
                "event_contract_fingerprint",
                "sequence_contract_fingerprint",
            ):
                payload.pop(field)
        if self.format != TEMPORAL_PRETRAINING_SHARD_SCHEMA:
            # This field did not exist when V1-V3 identities were published.
            payload.pop("engine_fact_producer_fingerprint")
        if legacy:
            payload.pop("split_assignment_fingerprint")
            payload.pop("split_source_replays")
            payload.pop("split_examples_committed")
            payload.pop("target_deck_digest")
            for part in payload["parts"]:
                part.pop("split_examples")
            for rejection in payload["rejections"]:
                rejection.pop("split")
        domain = (
            b"ptcg-rl/simple-stateless-pretraining-dataset/v1\x00"
            if legacy
            else (
                b"ptcg-rl/simple-stateless-sequence-pretraining-dataset/v4\x00"
                if self.format == TEMPORAL_PRETRAINING_SHARD_SCHEMA
                else (
                    b"ptcg-rl/simple-stateless-sequence-pretraining-dataset/v3\x00"
                    if self.format == LEGACY_TEMPORAL_PRETRAINING_SHARD_SCHEMA
                    else b"ptcg-rl/simple-stateless-pretraining-dataset/v2\x00"
                )
            )
        )
        return hashlib.sha256(
            domain
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    def split_examples(self, split: ReplaySplit) -> int:
        """Return durable split counts, treating V1 rows as training-only."""
        if self.format == LEGACY_PRETRAINING_SHARD_SCHEMA:
            return self.examples_committed if split == "train" else 0
        return self.split_examples_committed[split]


@dataclass(frozen=True)
class LoadedPretrainingPart:
    """Validated no-pickle arrays for one compact part."""

    path: Path
    arrays: Mapping[str, npt.NDArray[np.generic]]

    @property
    def example_count(self) -> int:
        """Return flattened decision rows."""
        return int(self.arrays["episode_ids"].shape[0])


@dataclass(frozen=True)
class _PendingPretrainingPart:
    """One complete source batch awaiting ordered durable publication."""

    future: Future[PretrainingPartRecord | None] | None
    examples: int
    source_replays: int
    split_examples: dict[ReplaySplit, int]
    counters: dict[str, int]
    rejections: tuple[RejectedReplayRecord, ...]
    uncompressed_bytes: int


class ReplayPretrainingShardWriter:
    """Publish complete-replay groups with a recoverable contiguous cursor."""

    def __init__(
        self,
        output_dir: Path,
        *,
        identity: ReplayPretrainingDatasetManifest,
        shard_rows: int,
        resume: bool,
        shard_uncompressed_bytes: int | None = None,
        compression_workers: int = 2,
    ) -> None:
        """Create or resume one compact dataset."""
        self.output_dir = output_dir
        self.parts_dir = output_dir / "parts"
        self.manifest_path = output_dir / "manifest.json"
        self.shard_rows = shard_rows
        self.shard_uncompressed_bytes = shard_uncompressed_bytes
        self.compression_workers = compression_workers
        self._compressor = ThreadPoolExecutor(
            max_workers=compression_workers,
            thread_name_prefix="pretraining-part",
        )
        self._pending_parts: list[_PendingPretrainingPart] = []
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self._examples: list[ReplayPretrainingExample] = []
        self._uncompressed_bytes = 0
        self._source_replays = 0
        self._counters: Counter[str] = Counter()
        self._rejections: list[RejectedReplayRecord] = []
        if resume and self.manifest_path.is_file():
            existing = load_pretraining_manifest(self.manifest_path)
            _validate_resume_identity(existing, identity)
            self.manifest = existing
            self._verify_parts()
        else:
            if self.manifest_path.exists() or any(self.parts_dir.iterdir()):
                raise FileExistsError("pretraining dataset directory is not empty")
            self.manifest = identity
            self._write_manifest()

    def add_replay(
        self,
        examples: Sequence[ReplayPretrainingExample],
        *,
        counters: Mapping[str, int],
        rejection: RejectedReplayRecord | None = None,
    ) -> None:
        """Buffer one fully processed source replay."""
        if self.manifest.complete:
            raise RuntimeError("cannot append to a complete pretraining dataset")
        self._commit_ready_parts()
        if rejection is not None:
            expected_index = self.manifest.source_cursor + self._source_replays
            if rejection.source_index != expected_index:
                raise ValueError("rejected replay differs from the source cursor")
            if examples:
                raise ValueError("rejected replay cannot publish training examples")
            self._rejections.append(rejection)
        self._examples.extend(examples)
        self._uncompressed_bytes += sum(_resident_size(example) for example in examples)
        self._source_replays += 1
        self._counters.update(counters)
        if (
            len(self._examples) >= self.shard_rows
            or (
                self.shard_uncompressed_bytes is not None
                and self._uncompressed_bytes >= self.shard_uncompressed_bytes
            )
            or (self._source_replays >= 64 and not self._examples)
        ):
            self.flush()

    @property
    def buffered_examples(self) -> int:
        """Return examples not yet represented by the durable manifest."""
        return len(self._examples) + sum(part.examples for part in self._pending_parts)

    @property
    def buffered_uncompressed_bytes(self) -> int:
        """Return the bounded resident estimate for the current part."""
        return self._uncompressed_bytes + sum(
            part.uncompressed_bytes for part in self._pending_parts
        )

    def flush(self) -> PretrainingPartRecord | None:
        """Queue one complete part, applying ordered compression backpressure."""
        if self._source_replays == 0:
            return None
        committed: PretrainingPartRecord | None = None
        if len(self._pending_parts) >= self.compression_workers:
            committed = self._commit_oldest_part()
        part_split_examples = _split_counts(example.split for example in self._examples)
        examples = tuple(self._examples)
        part_index = len(self.manifest.parts) + len(self._pending_parts)
        future: Future[PretrainingPartRecord | None] | None = None
        if self._examples:
            future = self._compressor.submit(
                write_pretraining_part,
                examples,
                identity=self.manifest,
                part_index=part_index,
                parts_dir=self.parts_dir,
                source_replays=self._source_replays,
            )
        self._pending_parts.append(
            _PendingPretrainingPart(
                future=future,
                examples=len(examples),
                source_replays=self._source_replays,
                split_examples=part_split_examples,
                counters=dict(self._counters),
                rejections=tuple(self._rejections),
                uncompressed_bytes=self._uncompressed_bytes,
            )
        )
        self._examples.clear()
        self._uncompressed_bytes = 0
        self._source_replays = 0
        self._counters.clear()
        self._rejections.clear()
        return committed

    def _commit_oldest_part(self) -> PretrainingPartRecord | None:
        """Publish the oldest compressed part and only then advance its cursor."""
        pending = self._pending_parts.pop(0)
        record = None if pending.future is None else pending.future.result()
        if (record is None) != (pending.examples == 0):
            raise RuntimeError("pretraining compression changed part row ownership")
        updated_parts = (
            self.manifest.parts if record is None else self.manifest.parts + (record,)
        )
        proposed = self.manifest.model_copy(
            update={
                "source_cursor": self.manifest.source_cursor + pending.source_replays,
                "examples_committed": (
                    self.manifest.examples_committed + pending.examples
                ),
                "split_examples_committed": {
                    split: (
                        self.manifest.split_examples_committed[split]
                        + pending.split_examples[split]
                    )
                    for split in REPLAY_SPLITS
                },
                "counters": dict(
                    Counter(self.manifest.counters) + Counter(pending.counters)
                ),
                "parts": updated_parts,
                "rejections": (self.manifest.rejections + pending.rejections),
            }
        )
        self._publish_manifest(proposed)
        return record

    def _commit_ready_parts(self) -> None:
        """Advance completed compression tasks without blocking extraction."""
        while self._pending_parts:
            future = self._pending_parts[0].future
            if future is not None and not future.done():
                break
            self._commit_oldest_part()

    def commit_prebuilt_batch(
        self,
        *,
        part: PretrainingPartRecord | None,
        source_replays: int,
        counters: Mapping[str, int],
        rejections: Sequence[RejectedReplayRecord],
    ) -> ReplayPretrainingDatasetManifest:
        """Commit one worker-built part without transferring examples over IPC."""
        if self.manifest.complete:
            raise RuntimeError("cannot append to a complete pretraining dataset")
        if (
            self._examples
            or self._source_replays
            or self._pending_parts
            or self._rejections
        ):
            raise RuntimeError("prebuilt batches cannot mix with buffered examples")
        if source_replays <= 0:
            raise ValueError("prebuilt batch must advance at least one source replay")
        start = self.manifest.source_cursor
        stop = start + source_replays
        if stop > self.manifest.source_replays:
            raise ValueError("prebuilt batch exceeds the source inventory")
        rejection_records = tuple(rejections)
        rejection_indices = tuple(item.source_index for item in rejection_records)
        if rejection_indices != tuple(sorted(set(rejection_indices))) or any(
            index < start or index >= stop for index in rejection_indices
        ):
            raise ValueError("prebuilt batch rejections differ from its source range")
        accepted_source_replays = source_replays - len(rejection_records)
        if part is not None:
            if part.source_replays != accepted_source_replays:
                raise ValueError(
                    "prebuilt part source count differs from accepted replays"
                )
            path = self.parts_dir / part.filename
            if (
                not path.is_file()
                or path.stat().st_size != part.size_bytes
                or file_sha256(path) != part.sha256
            ):
                raise ValueError("prebuilt pretraining part failed verification")
        split_examples = (
            dict.fromkeys(REPLAY_SPLITS, 0)
            if part is None
            else part.split_examples
        )
        proposed = self.manifest.model_copy(
            update={
                "source_cursor": stop,
                "examples_committed": (
                    self.manifest.examples_committed
                    + (0 if part is None else part.examples)
                ),
                "split_examples_committed": {
                    split: (
                        self.manifest.split_examples_committed[split]
                        + split_examples[split]
                    )
                    for split in REPLAY_SPLITS
                },
                "counters": dict(
                    Counter(self.manifest.counters) + Counter(counters)
                ),
                "parts": (
                    self.manifest.parts
                    if part is None
                    else self.manifest.parts + (part,)
                ),
                "rejections": self.manifest.rejections + rejection_records,
            }
        )
        self._publish_manifest(proposed)
        return self.manifest

    def close(self) -> ReplayPretrainingDatasetManifest:
        """Commit the final buffer and mark the full source cursor complete."""
        self.flush()
        try:
            while self._pending_parts:
                self._commit_oldest_part()
        finally:
            self._compressor.shutdown(wait=True)
        if self.manifest.source_cursor != self.manifest.source_replays:
            raise RuntimeError("pretraining extraction stopped before all replays")
        if self.manifest.examples_committed <= 0:
            raise RuntimeError("pretraining extraction produced no examples")
        self._publish_manifest(self.manifest.model_copy(update={"complete": True}))
        return self.manifest

    def _verify_parts(self) -> None:
        """Verify every committed part before advancing a resumed cursor."""
        examples = 0
        replay_count = 0
        split_examples: Counter[ReplaySplit] = Counter()
        for record in self.manifest.parts:
            path = self.parts_dir / record.filename
            if (
                not path.is_file()
                or path.stat().st_size != record.size_bytes
                or file_sha256(path) != record.sha256
            ):
                raise ValueError("committed pretraining part failed verification")
            loaded = load_pretraining_part(path)
            if loaded.example_count != record.examples:
                raise ValueError("pretraining part example count changed")
            loaded_split_examples = {
                split: int(pretraining_split_indices(loaded, split).size)
                for split in REPLAY_SPLITS
            }
            if loaded_split_examples != record.split_examples:
                raise ValueError("pretraining part split counts changed")
            examples += record.examples
            replay_count += record.source_replays
            split_examples.update(record.split_examples)
        if examples != self.manifest.examples_committed:
            raise ValueError("pretraining manifest example count is inconsistent")
        if replay_count > self.manifest.source_cursor:
            raise ValueError("pretraining part replay counts exceed source cursor")
        if {
            split: split_examples[split] for split in REPLAY_SPLITS
        } != self.manifest.split_examples_committed:
            raise ValueError("pretraining manifest split example counts changed")

    def _write_manifest(self) -> None:
        self._publish_manifest(self.manifest)

    def _publish_manifest(
        self,
        manifest: ReplayPretrainingDatasetManifest,
    ) -> None:
        atomic_write_bytes(
            self.manifest_path,
            json_payload(manifest.model_dump(mode="json")),
            overwrite=True,
        )
        self.manifest = manifest


def load_pretraining_manifest(path: Path) -> ReplayPretrainingDatasetManifest:
    """Load one validated compact-dataset manifest."""
    with path.open(encoding="utf-8") as handle:
        return ReplayPretrainingDatasetManifest.model_validate(json.load(handle))


def load_pretraining_part(path: Path) -> LoadedPretrainingPart:
    """Load and structurally validate one compact NPZ part."""
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: np.asarray(source[key]) for key in source.files}
    validate_pretraining_arrays(arrays)
    return LoadedPretrainingPart(path=path, arrays=arrays)


def load_pretraining_metadata_columns(
    path: Path,
    *,
    columns: Collection[str],
) -> dict[str, npt.NDArray[np.generic]]:
    """Load validated scalar metadata without inflating tensor payload columns."""
    requested = frozenset(columns) | {"episode_ids"}
    unsupported = requested - _METADATA_COLUMN_DTYPES.keys()
    if unsupported:
        raise ValueError(
            "unsupported pretraining metadata columns: "
            f"{sorted(unsupported)}"
        )
    with np.load(path, allow_pickle=False) as source:
        version = np.asarray(source["schema_version"]).tolist()
        expected_keys = (
            _V1_ARRAY_KEYS
            if version == [1]
            else (
                _V2_ARRAY_KEYS
                if version == [2]
                else (
                    _V3_ARRAY_KEYS
                    if version == [3]
                    else _V4_ARRAY_KEYS
                    if version == [4]
                    else None
                )
            )
        )
        if expected_keys is None:
            raise ValueError("unsupported pretraining part schema")
        if set(source.files) != expected_keys:
            raise ValueError("pretraining part arrays differ from schema")
        missing = requested - set(source.files)
        if missing:
            raise ValueError(
                "pretraining metadata columns are absent from the schema: "
                f"{sorted(missing)}"
            )
        arrays = {key: np.asarray(source[key]) for key in requested}
    rows = int(arrays["episode_ids"].shape[0])
    if rows <= 0:
        raise ValueError("pretraining metadata has no rows")
    for key, array in arrays.items():
        expected_dtype = np.dtype(_METADATA_COLUMN_DTYPES[key])
        expected_shape = (
            (rows + 1,)
            if key in _ROW_OFFSET_METADATA_COLUMNS
            else None if key == "sequence_offsets" else (rows,)
        )
        if (
            (expected_shape is not None and array.shape != expected_shape)
            or (
                key == "sequence_offsets"
                and (
                    array.ndim != 1
                    or len(array) < 2
                    or len(array) > rows + 1
                )
            )
            or array.dtype != expected_dtype
        ):
            raise ValueError(
                "pretraining metadata column is misaligned: "
                f"{key} shape={array.shape} dtype={array.dtype}"
            )
    if np.any(arrays["episode_ids"] < 0):
        raise ValueError("pretraining metadata episode IDs are invalid")
    player_indices = arrays.get("player_indices")
    if player_indices is not None and not np.isin(player_indices, (0, 1)).all():
        raise ValueError("pretraining metadata player indices are invalid")
    date_indices = arrays.get("date_indices")
    if date_indices is not None and np.any(date_indices < 0):
        raise ValueError("pretraining metadata date indices are invalid")
    split_codes = arrays.get("split_codes")
    if split_codes is not None and not np.isin(
        split_codes,
        tuple(_CODE_TO_SPLIT),
    ).all():
        raise ValueError("pretraining metadata split codes are invalid")
    outcomes = arrays.get("outcomes")
    if outcomes is not None and not np.isin(outcomes, (-1.0, 0.0, 1.0)).all():
        raise ValueError("pretraining metadata outcomes are invalid")
    example_weights = arrays.get("example_weights")
    if example_weights is not None and (
        not np.isfinite(example_weights).all()
        or np.any(example_weights <= 0.0)
    ):
        raise ValueError("pretraining metadata example weights are invalid")
    decision_indices = arrays.get("decision_indices")
    if decision_indices is not None and np.any(decision_indices < 0):
        raise ValueError("pretraining metadata decision indices are invalid")
    for key in (*_ROW_OFFSET_METADATA_COLUMNS, "sequence_offsets"):
        offsets = arrays.get(key)
        if offsets is None:
            continue
        if (
            offsets[0] != 0
            or np.any(offsets[1:] < offsets[:-1])
            or (key == "sequence_offsets" and offsets[-1] != rows)
        ):
            raise ValueError(f"pretraining metadata {key} are invalid")
    return arrays


def load_temporal_pretraining_geometry(path: Path) -> LoadedPretrainingPart:
    """Load only the compact arrays required to plan bounded temporal batches."""
    arrays = load_pretraining_metadata_columns(
        path,
        columns=(
            "decision_indices",
            "sequence_offsets",
            "state_offsets",
            "option_offsets",
            "split_codes",
        ),
    )
    return LoadedPretrainingPart(path=path, arrays=arrays)


def pretraining_split_indices(
    part: LoadedPretrainingPart,
    split: ReplaySplit,
) -> npt.NDArray[np.int64]:
    """Return row indices for one durable episode-level split."""
    codes = part.arrays.get("split_codes")
    if codes is None:
        if split != "train":
            return np.asarray([], dtype=np.int64)
        return np.arange(part.example_count, dtype=np.int64)
    return np.flatnonzero(
        np.asarray(codes, dtype=np.int64) == _SPLIT_TO_CODE[split]
    ).astype(np.int64, copy=False)


def iter_pretraining_rows(
    part: LoadedPretrainingPart,
    indices: Sequence[int],
    *,
    catalog_fingerprint: str,
    input_contract_fingerprint: str,
) -> Iterator[ReplayPretrainingExample]:
    """Reconstruct selected numeric rows without reading any raw replay JSON."""
    arrays = part.arrays
    deck_cache: dict[tuple[int, ...], CanonicalDeck] = {}
    temporal_events = (
        _public_event_block(arrays)
        if np.asarray(arrays["schema_version"]).tolist() in ([3], [4])
        else None
    )
    for raw_index in indices:
        row = int(raw_index)
        own_cards = tuple(int(value) for value in arrays["own_decks"][row])
        own_deck = deck_cache.get(own_cards)
        if own_deck is None:
            own_deck = canonicalize_deck(own_cards)
            deck_cache[own_cards] = own_deck
        state_start, state_end = _row_bounds(arrays["state_offsets"], row)
        attachment_start, attachment_end = _row_bounds(
            arrays["attachment_offsets"], row
        )
        option_start, option_end = _row_bounds(arrays["option_offsets"], row)
        belief_start, belief_end = _row_bounds(arrays["belief_offsets"], row)
        known_start, known_end = _row_bounds(arrays["known_offsets"], row)
        action_start, action_end = _row_bounds(arrays["action_offsets"], row)
        belief_scalars = arrays["belief_scalars"][row]
        event_delta = (
            PublicEventDelta()
            if temporal_events is None
            else temporal_events.delta_at(row)
        )
        decision_index = (
            None if temporal_events is None else int(arrays["decision_indices"][row])
        )
        actor_row = SimpleStatelessActorRow(
            state=StateTokenArrayFeatures(
                card_ids=arrays["state_card_ids"][state_start:state_end],
                areas=arrays["state_areas"][state_start:state_end],
                owner_roles=arrays["state_owner_roles"][state_start:state_end],
                token_kinds=arrays["state_token_kinds"][state_start:state_end],
                scalars=arrays["state_scalars"][state_start:state_end],
                last_attack_ids=arrays["state_last_attack_ids"][state_start:state_end],
                attachment_card_ids=arrays["attachment_card_ids"][
                    attachment_start:attachment_end
                ],
                attachment_parent_indices=arrays["attachment_parent_indices"][
                    attachment_start:attachment_end
                ],
                attachment_kinds=arrays["attachment_kinds"][
                    attachment_start:attachment_end
                ],
                entity_slots=arrays["state_entity_slots"][state_start:state_end],
            ),
            options=EncodedOptionArrayFeatures(
                option_types=arrays["option_types"][option_start:option_end],
                contexts=arrays["option_contexts"][option_start:option_end],
                entity_slots=arrays["option_entity_slots"][option_start:option_end],
                entity_slot_mask=arrays["option_entity_slot_masks"][
                    option_start:option_end
                ],
                attack_ids=arrays["option_attack_ids"][option_start:option_end],
                card_ids=arrays["option_card_ids"][option_start:option_end],
                scalars=arrays["option_scalars"][option_start:option_end],
                dynamic_effect_features=arrays["option_dynamic_effect_features"][
                    option_start:option_end
                ],
                dynamic_effect_masks=arrays["option_dynamic_effect_masks"][
                    option_start:option_end
                ],
            ),
            min_count=int(arrays["min_counts"][row]),
            max_count=int(arrays["max_counts"][row]),
            own_deck=own_deck,
            belief_summary=PublicDeckPosteriorArrays(
                card_ids=np.asarray(
                    arrays["belief_card_ids"][belief_start:belief_end],
                    dtype=np.int32,
                ),
                expected_counts=np.asarray(
                    arrays["belief_expected_counts"][belief_start:belief_end],
                    dtype=np.float32,
                ),
                entropy=float(belief_scalars[0]),
                compatible_deck_count=int(belief_scalars[1]),
                public_evidence_count=int(belief_scalars[2]),
                unknown_probability=float(belief_scalars[3]),
            ),
            catalog_fingerprint=catalog_fingerprint,
            input_contract_fingerprint=input_contract_fingerprint,
            public_event_delta=event_delta,
            engine_fact_producer_fingerprint=(
                None
                if temporal_events is None
                or "engine_fact_producer_fingerprint" not in arrays
                or not str(arrays["engine_fact_producer_fingerprint"][0])
                else str(arrays["engine_fact_producer_fingerprint"][0])
            ),
            sequence_identity=(
                None
                if decision_index is None
                else SequenceDecisionIdentity(
                    game_id=str(int(arrays["episode_ids"][row])),
                    seat=int(arrays["player_indices"][row]),  # type: ignore[arg-type]
                    decision_index=decision_index,
                    request_id=(
                        f"{int(arrays['episode_ids'][row])}:"
                        f"{int(arrays['player_indices'][row])}:{decision_index}"
                    ),
                )
            ),
        )
        yield ReplayPretrainingExample(
            episode_id=int(arrays["episode_ids"][row]),
            player_index=int(arrays["player_indices"][row]),
            step_index=int(arrays["step_indices"][row]),
            team_index=int(arrays["team_indices"][row]),
            date_index=int(arrays["date_indices"][row]),
            replay_index=int(arrays["replay_indices"][row]),
            split=_row_split(arrays, row),
            actor_row=actor_row,
            action=tuple(
                int(value)
                for value in arrays["action_choices"][action_start:action_end]
            ),
            opponent_deck=tuple(int(value) for value in arrays["opponent_decks"][row]),
            known_opponent_counts=tuple(
                (int(card_id), int(count))
                for card_id, count in zip(
                    arrays["known_card_ids"][known_start:known_end],
                    arrays["known_counts"][known_start:known_end],
                    strict=True,
                )
            ),
            outcome=float(arrays["outcomes"][row]),
            example_weight=float(arrays["example_weights"][row]),
            decision_index=decision_index,
            source_replay_sha256=(
                None
                if temporal_events is None
                else str(arrays["source_replay_sha256s"][row])
            ),
            route_expert_id=(
                None
                if temporal_events is None or not str(arrays["route_expert_ids"][row])
                else str(arrays["route_expert_ids"][row])
            ),
            accepted_action=(
                None if temporal_events is None else _accepted_action(arrays, row)
            ),
        )


def validate_pretraining_arrays(
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> None:
    """Validate compact shapes and action/deck invariants."""
    version = np.asarray(arrays.get("schema_version")).tolist()
    expected_keys = (
        _V1_ARRAY_KEYS
        if version == [1]
        else (
            _V2_ARRAY_KEYS
            if version == [2]
            else (
                _V3_ARRAY_KEYS
                if version == [3]
                else _V4_ARRAY_KEYS
                if version == [4]
                else None
            )
        )
    )
    if expected_keys is None:
        raise ValueError("unsupported pretraining part schema")
    if set(arrays) != expected_keys:
        raise ValueError("pretraining part arrays differ from schema")
    rows = int(arrays["episode_ids"].shape[0])
    scalar_rows = (
        "player_indices",
        "step_indices",
        "team_indices",
        "date_indices",
        "replay_indices",
        "outcomes",
        "example_weights",
        "min_counts",
        "max_counts",
    )
    if rows <= 0 or any(arrays[key].shape != (rows,) for key in scalar_rows):
        raise ValueError("pretraining scalar columns are misaligned")
    if version in ([2], [3], [4]):
        split_codes = np.asarray(arrays["split_codes"], dtype=np.int64)
        if (
            split_codes.shape != (rows,)
            or not np.isin(
                split_codes,
                tuple(_CODE_TO_SPLIT),
            ).all()
        ):
            raise ValueError("pretraining split codes are invalid")
    if arrays["own_decks"].shape != (rows, 60) or arrays["opponent_decks"].shape != (
        rows,
        60,
    ):
        raise ValueError("pretraining exact decks must have shape [rows, 60]")
    if arrays["belief_scalars"].shape != (rows, 4):
        raise ValueError("pretraining belief scalar shape is invalid")
    _validate_ragged(arrays, rows, prefix="state", values=("card_ids",))
    _validate_ragged(arrays, rows, prefix="attachment", values=("card_ids",))
    _validate_ragged(arrays, rows, prefix="option", values=("types",))
    _validate_ragged(arrays, rows, prefix="belief", values=("card_ids",))
    _validate_ragged(arrays, rows, prefix="known", values=("card_ids",))
    _validate_ragged(arrays, rows, prefix="action", values=("choices",))
    state_count = len(arrays["state_card_ids"])
    option_count = len(arrays["option_types"])
    attachment_count = len(arrays["attachment_card_ids"])
    if arrays["state_scalars"].shape != (state_count, TOKEN_SCALAR_SIZE):
        raise ValueError("pretraining state scalar shape is invalid")
    if arrays["state_entity_slots"].shape != (state_count,):
        raise ValueError("pretraining state entity slots are misaligned")
    if arrays["attachment_parent_indices"].shape != (attachment_count,):
        raise ValueError("pretraining attachment parents are misaligned")
    if arrays["attachment_kinds"].shape != (attachment_count,):
        raise ValueError("pretraining attachment kinds are misaligned")
    if arrays["option_entity_slots"].shape != (
        option_count,
        MAX_ENTITY_SLOTS,
    ):
        raise ValueError("pretraining option entity slots are misaligned")
    if arrays["option_entity_slot_masks"].shape != (
        option_count,
        MAX_ENTITY_SLOTS,
    ):
        raise ValueError("pretraining option entity masks are misaligned")
    if arrays["option_scalars"].shape != (option_count, SCALAR_FEATURE_SIZE):
        raise ValueError("pretraining option scalar shape is invalid")
    if arrays["option_dynamic_effect_features"].shape != (
        option_count,
        DYNAMIC_EFFECT_FEATURE_SIZE,
    ):
        raise ValueError("pretraining option effect shape is invalid")
    if not np.isin(arrays["outcomes"], (-1.0, 0.0, 1.0)).all():
        raise ValueError("pretraining outcomes must be win/draw/loss")
    own_decks = np.asarray(arrays["own_decks"], dtype=np.int64)
    opponent_decks = np.asarray(arrays["opponent_decks"], dtype=np.int64)
    example_weights = np.asarray(arrays["example_weights"], dtype=np.float64)
    if (
        np.any(own_decks <= 0)
        or np.any(opponent_decks <= 0)
        or np.any(example_weights <= 0.0)
    ):
        raise ValueError("pretraining decks and weights must be positive")
    option_offsets = np.asarray(arrays["option_offsets"], dtype=np.int64)
    action_offsets = np.asarray(arrays["action_offsets"], dtype=np.int64)
    option_lengths = option_offsets[1:] - option_offsets[:-1]
    action_lengths = action_offsets[1:] - action_offsets[:-1]
    minimums = np.asarray(arrays["min_counts"], dtype=np.int64)
    maximums = np.asarray(arrays["max_counts"], dtype=np.int64)
    if np.any(action_lengths < minimums) or np.any(action_lengths > maximums):
        raise ValueError("pretraining action cardinality is illegal")
    for row in range(rows):
        start, end = int(action_offsets[row]), int(action_offsets[row + 1])
        choices = np.asarray(arrays["action_choices"][start:end], dtype=np.int64)
        if len(np.unique(choices)) != len(choices) or np.any(
            (choices < 0) | (choices >= option_lengths[row])
        ):
            raise ValueError("pretraining action option index is illegal")
    if version in ([3], [4]):
        _validate_temporal_arrays(arrays, rows=rows, version=int(version[0]))


def _validate_temporal_arrays(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    *,
    rows: int,
    version: int,
) -> None:
    """Fail closed on broken sequence boundaries or raw temporal payloads."""
    for key in ("decision_indices", "source_steps", "engine_fact_available"):
        if arrays[key].shape != (rows,):
            raise ValueError("temporal pretraining row columns are misaligned")
    engine_fact_available = np.asarray(
        arrays["engine_fact_available"],
        dtype=np.bool_,
    )
    dynamic_effect_masks = np.asarray(
        arrays["option_dynamic_effect_masks"],
        dtype=np.bool_,
    )
    if version == 3 and (
        engine_fact_available.any() or dynamic_effect_masks.any()
    ):
        raise ValueError("V3 temporal pretraining cannot persist engine facts")
    if version == 4:
        producer_values = np.asarray(arrays["engine_fact_producer_fingerprint"])
        if producer_values.shape != (1,):
            raise ValueError("V4 engine-fact producer contract is not scalar")
        producer_fingerprint = str(producer_values[0])
        if producer_fingerprint:
            _validate_sha256(producer_fingerprint)
            if not engine_fact_available.all():
                raise ValueError("V4 engine-fact producer is not bound to every row")
        elif engine_fact_available.any() or dynamic_effect_masks.any():
            raise ValueError("V4 engine facts have no producer fingerprint")
    for key in (
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "public_catalog_fingerprint",
        "input_contract_fingerprint",
        "event_contract_fingerprint",
        "sequence_contract_fingerprint",
    ):
        values = np.asarray(arrays[key])
        if values.shape != (1,):
            raise ValueError("temporal pretraining part contract is not scalar")
        _validate_sha256(str(values[0]))
    for value in np.asarray(arrays["source_replay_sha256s"]):
        _validate_sha256(str(value))
    for value in np.asarray(arrays["route_expert_ids"]):
        if str(value):
            _validate_sha256(str(value))

    offsets = np.asarray(arrays["sequence_offsets"], dtype=np.int64)
    episode_ids = np.asarray(arrays["sequence_episode_ids"], dtype=np.int64)
    seats = np.asarray(arrays["sequence_seats"], dtype=np.int64)
    if (
        offsets.ndim != 1
        or offsets.size < 2
        or offsets[0] != 0
        or offsets[-1] != rows
        or np.any(offsets[1:] <= offsets[:-1])
        or episode_ids.shape != (offsets.size - 1,)
        or seats.shape != (offsets.size - 1,)
        or not np.isin(seats, (0, 1)).all()
    ):
        raise ValueError("temporal pretraining sequence offsets are invalid")
    decisions = np.asarray(arrays["decision_indices"], dtype=np.int64)
    row_episodes = np.asarray(arrays["episode_ids"], dtype=np.int64)
    row_seats = np.asarray(arrays["player_indices"], dtype=np.int64)
    split_codes = np.asarray(arrays["split_codes"], dtype=np.int64)
    for sequence, (start, stop) in enumerate(
        zip(offsets[:-1], offsets[1:], strict=True)
    ):
        expected = np.arange(stop - start, dtype=np.int64)
        if (
            not np.array_equal(decisions[start:stop], expected)
            or not np.all(row_episodes[start:stop] == episode_ids[sequence])
            or not np.all(row_seats[start:stop] == seats[sequence])
            or not np.all(split_codes[start:stop] == split_codes[start])
        ):
            raise ValueError(
                "temporal pretraining sequence identity or split is discontinuous"
            )

    events = _public_event_block(arrays)
    validate_public_event_array_block(events)
    if events.decision_count != rows:
        raise ValueError("temporal event rows differ from decisions")
    _validate_ragged(
        arrays,
        rows,
        prefix="accepted_action",
        values=("option_types",),
    )
    action_offsets = np.asarray(arrays["action_offsets"], dtype=np.int64)
    accepted_offsets = np.asarray(
        arrays["accepted_action_offsets"],
        dtype=np.int64,
    )
    if not np.array_equal(
        action_offsets[1:] - action_offsets[:-1],
        accepted_offsets[1:] - accepted_offsets[:-1],
    ):
        raise ValueError("accepted actions differ from teacher action cardinality")
    row_fields = (
        "accepted_action_schema_versions",
        "accepted_action_stable_ids",
        "accepted_action_prompt_contexts",
        "accepted_action_ordered",
        "accepted_action_stop_sampled",
        "accepted_action_fallback",
    )
    if any(arrays[key].shape != (rows,) for key in row_fields):
        raise ValueError("accepted-action row fields are misaligned")
    if not np.all(
        np.asarray(
            arrays["accepted_action_schema_versions"],
            dtype=np.int64,
        )
        == ACCEPTED_ACTION_SCHEMA_VERSION
    ):
        raise ValueError("accepted-action schema version changed")
    accepted = int(accepted_offsets[-1])
    vector_fields = (
        "accepted_action_option_contexts",
        "accepted_action_card_ids",
        "accepted_action_attack_ids",
    )
    if any(arrays[key].shape != (accepted,) for key in vector_fields):
        raise ValueError("accepted-action categorical fields are misaligned")
    if arrays["accepted_action_option_scalars"].shape != (
        accepted,
        SCALAR_FEATURE_SIZE,
    ):
        raise ValueError("accepted-action option scalars are misaligned")
    entity_shape = (accepted, MAX_ENTITY_SLOTS)
    if any(
        arrays[key].shape != entity_shape
        for key in (
            "accepted_action_entity_card_ids",
            "accepted_action_entity_areas",
            "accepted_action_entity_owner_roles",
            "accepted_action_entity_token_kinds",
        )
    ):
        raise ValueError("accepted-action entity columns are misaligned")
    if arrays["accepted_action_entity_scalars"].shape != (
        accepted,
        MAX_ENTITY_SLOTS,
        TOKEN_SCALAR_SIZE,
    ):
        raise ValueError("accepted-action entity scalars are misaligned")
    for row in range(rows):
        _accepted_action(arrays, row)


def _public_event_block(
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> PublicEventArrayBlock:
    """Build the canonical public-event array view over one compact part."""
    return PublicEventArrayBlock(
        event_offsets=np.asarray(arrays["event_offsets"]),
        event_types=np.asarray(arrays["event_types"]),
        actor_roles=np.asarray(arrays["event_actor_roles"]),
        from_areas=np.asarray(arrays["event_from_areas"]),
        to_areas=np.asarray(arrays["event_to_areas"]),
        card_ids=np.asarray(arrays["event_card_ids"]),
        serials=np.asarray(arrays["event_serials"]),
        entity_mask=np.asarray(arrays["event_entity_mask"]),
        attack_ids=np.asarray(arrays["event_attack_ids"]),
        attack_id_mask=np.asarray(arrays["event_attack_id_mask"]),
        values=np.asarray(arrays["event_values"]),
        value_mask=np.asarray(arrays["event_value_mask"]),
        categorical_values=np.asarray(arrays["event_categorical_values"]),
        overflow_offsets=np.asarray(arrays["event_overflow_offsets"]),
        overflow_event_types=np.asarray(arrays["event_overflow_types"]),
        overflow_actor_roles=np.asarray(arrays["event_overflow_actor_roles"]),
        overflow_counts=np.asarray(arrays["event_overflow_counts"]),
    )


def _accepted_action(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    row: int,
) -> AcceptedActionRecord:
    """Reconstruct one stable accepted action from compact columns."""
    start, stop = _row_bounds(arrays["accepted_action_offsets"], row)
    return AcceptedActionRecord(
        schema_version=int(arrays["accepted_action_schema_versions"][row]),
        stable_identity=str(arrays["accepted_action_stable_ids"][row]),
        prompt_context=int(arrays["accepted_action_prompt_contexts"][row]),
        option_types=tuple(
            int(value) for value in arrays["accepted_action_option_types"][start:stop]
        ),
        option_contexts=tuple(
            int(value)
            for value in arrays["accepted_action_option_contexts"][start:stop]
        ),
        card_ids=tuple(
            int(value) for value in arrays["accepted_action_card_ids"][start:stop]
        ),
        attack_ids=tuple(
            int(value) for value in arrays["accepted_action_attack_ids"][start:stop]
        ),
        option_scalars=tuple(
            tuple(float(value) for value in values)
            for values in arrays["accepted_action_option_scalars"][start:stop]
        ),
        entity_card_ids=tuple(
            tuple(int(value) for value in values)
            for values in arrays["accepted_action_entity_card_ids"][start:stop]
        ),
        entity_areas=tuple(
            tuple(int(value) for value in values)
            for values in arrays["accepted_action_entity_areas"][start:stop]
        ),
        entity_owner_roles=tuple(
            tuple(int(value) for value in values)
            for values in arrays["accepted_action_entity_owner_roles"][start:stop]
        ),
        entity_token_kinds=tuple(
            tuple(int(value) for value in values)
            for values in arrays["accepted_action_entity_token_kinds"][start:stop]
        ),
        entity_scalars=tuple(
            tuple(float(value) for value in values.reshape(-1))
            for values in arrays["accepted_action_entity_scalars"][start:stop]
        ),
        ordered=bool(arrays["accepted_action_ordered"][row]),
        stop_sampled=bool(arrays["accepted_action_stop_sampled"][row]),
        accepted=True,
        fallback=bool(arrays["accepted_action_fallback"][row]),
    )


def file_sha256(path: Path) -> str:
    """Stream one file fingerprint."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_pretraining_part(
    examples: tuple[ReplayPretrainingExample, ...],
    *,
    identity: ReplayPretrainingDatasetManifest,
    part_index: int,
    parts_dir: Path,
    source_replays: int,
) -> PretrainingPartRecord:
    """Build, compress, verify, and atomically publish one immutable part."""
    arrays = _examples_to_arrays(examples, identity=identity)
    validate_pretraining_arrays(arrays)
    provisional_filename = f"part-{part_index:08d}.npz"
    temporary = _temporary_path(provisional_filename)
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
            handle.flush()
            os.fsync(handle.fileno())
        sha256 = file_sha256(temporary)
        filename = f"part-{part_index:08d}-{sha256[:16]}.npz"
        destination = parts_dir / filename
        record = PretrainingPartRecord(
            filename=filename,
            sha256=sha256,
            size_bytes=temporary.stat().st_size,
            examples=len(examples),
            source_replays=source_replays,
            split_examples=_split_counts(example.split for example in examples),
        )
        if destination.exists():
            if (
                destination.stat().st_size != record.size_bytes
                or file_sha256(destination) != record.sha256
            ):
                raise FileExistsError(
                    f"content-addressed pretraining part differs: {destination}"
                )
        else:
            os.replace(temporary, destination)
            _fsync_directory(parts_dir)
        return record
    finally:
        temporary.unlink(missing_ok=True)


def _resident_size(value: object, seen: set[int] | None = None) -> int:
    """Estimate resident bytes without materializing another compact part."""
    visited = set() if seen is None else seen
    identity = id(value)
    if identity in visited:
        return 0
    visited.add(identity)
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        return size + sum(
            _resident_size(key, visited) + _resident_size(item, visited)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list, set, frozenset)):
        return size + sum(_resident_size(item, visited) for item in value)
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return size + _resident_size(attributes, visited)
    return size


def _examples_to_arrays(
    examples: tuple[ReplayPretrainingExample, ...],
    *,
    identity: ReplayPretrainingDatasetManifest | None = None,
) -> dict[str, npt.NDArray[np.generic]]:
    states = tuple(example.actor_row.state for example in examples)
    options = tuple(example.actor_row.options for example in examples)
    beliefs = tuple(example.actor_row.belief_summary for example in examples)
    if not all(isinstance(item, PublicDeckPosteriorArrays) for item in beliefs):
        raise TypeError("pretraining persistence requires array-backed belief rows")
    array_beliefs = cast(tuple[PublicDeckPosteriorArrays, ...], beliefs)
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "schema_version": np.asarray([2], dtype=np.int16),
        "episode_ids": np.asarray(
            [example.episode_id for example in examples], dtype=np.int64
        ),
        "player_indices": np.asarray(
            [example.player_index for example in examples], dtype=np.int8
        ),
        "step_indices": np.asarray(
            [example.step_index for example in examples], dtype=np.int32
        ),
        "team_indices": np.asarray(
            [example.team_index for example in examples], dtype=np.int16
        ),
        "date_indices": np.asarray(
            [example.date_index for example in examples], dtype=np.int8
        ),
        "replay_indices": np.asarray(
            [example.replay_index for example in examples], dtype=np.int32
        ),
        "split_codes": np.asarray(
            [_SPLIT_TO_CODE[example.split] for example in examples],
            dtype=np.int8,
        ),
        "own_decks": np.asarray(
            [example.actor_row.own_deck.card_ids for example in examples],
            dtype=np.int32,
        ),
        "opponent_decks": np.asarray(
            [example.opponent_deck for example in examples], dtype=np.int32
        ),
        "outcomes": np.asarray(
            [example.outcome for example in examples], dtype=np.float32
        ),
        "example_weights": np.asarray(
            [example.example_weight for example in examples], dtype=np.float32
        ),
        "state_offsets": _offsets([len(state.card_ids) for state in states]),
        "state_card_ids": _concat(states, "card_ids", np.int32),
        "state_areas": _concat(states, "areas", np.int16),
        "state_owner_roles": _concat(states, "owner_roles", np.int8),
        "state_token_kinds": _concat(states, "token_kinds", np.int8),
        "state_scalars": _concat_2d(states, "scalars", TOKEN_SCALAR_SIZE, np.float32),
        "state_last_attack_ids": _concat(states, "last_attack_ids", np.int32),
        "state_entity_slots": _concat(states, "entity_slots", np.int16),
        "attachment_offsets": _offsets(
            [len(state.attachment_card_ids) for state in states]
        ),
        "attachment_card_ids": _concat(states, "attachment_card_ids", np.int32),
        "attachment_parent_indices": _concat(
            states, "attachment_parent_indices", np.int16
        ),
        "attachment_kinds": _concat(states, "attachment_kinds", np.int8),
        "option_offsets": _offsets([len(option) for option in options]),
        "option_types": _concat(options, "option_types", np.int16),
        "option_contexts": _concat(options, "contexts", np.int16),
        "option_entity_slots": _concat_2d(
            options, "entity_slots", MAX_ENTITY_SLOTS, np.int16
        ),
        "option_entity_slot_masks": _concat_2d(
            options, "entity_slot_mask", MAX_ENTITY_SLOTS, np.bool_
        ),
        "option_attack_ids": _concat(options, "attack_ids", np.int32),
        "option_card_ids": _concat(options, "card_ids", np.int32),
        "option_scalars": _concat_2d(
            options, "scalars", SCALAR_FEATURE_SIZE, np.float32
        ),
        "option_dynamic_effect_features": _concat_2d(
            options,
            "dynamic_effect_features",
            DYNAMIC_EFFECT_FEATURE_SIZE,
            np.float32,
        ),
        "option_dynamic_effect_masks": _concat(
            options, "dynamic_effect_masks", np.bool_
        ),
        "min_counts": np.asarray(
            [example.actor_row.min_count for example in examples], dtype=np.int16
        ),
        "max_counts": np.asarray(
            [example.actor_row.max_count for example in examples], dtype=np.int16
        ),
        "belief_offsets": _offsets([len(item.card_ids) for item in array_beliefs]),
        "belief_card_ids": _concat(array_beliefs, "card_ids", np.int32),
        "belief_expected_counts": _concat(array_beliefs, "expected_counts", np.float32),
        "belief_scalars": np.asarray(
            [
                (
                    item.entropy,
                    item.compatible_deck_count,
                    item.public_evidence_count,
                    item.unknown_probability,
                )
                for item in array_beliefs
            ],
            dtype=np.float32,
        ),
        "known_offsets": _offsets(
            [len(example.known_opponent_counts) for example in examples]
        ),
        "known_card_ids": np.asarray(
            [
                card_id
                for example in examples
                for card_id, _count in example.known_opponent_counts
            ],
            dtype=np.int32,
        ),
        "known_counts": np.asarray(
            [
                count
                for example in examples
                for _card_id, count in example.known_opponent_counts
            ],
            dtype=np.int16,
        ),
        "action_offsets": _offsets([len(example.action) for example in examples]),
        "action_choices": np.asarray(
            [choice for example in examples for choice in example.action],
            dtype=np.int16,
        ),
    }
    if identity is not None and is_temporal_pretraining_format(identity.format):
        arrays["schema_version"] = np.asarray(
            [4 if identity.format == TEMPORAL_PRETRAINING_SHARD_SCHEMA else 3],
            dtype=np.int16,
        )
        arrays.update(_temporal_example_arrays(examples, identity=identity))
    return arrays


def _temporal_example_arrays(
    examples: tuple[ReplayPretrainingExample, ...],
    *,
    identity: ReplayPretrainingDatasetManifest,
) -> dict[str, npt.NDArray[np.generic]]:
    """Flatten causal EVENT/STATE/ACTION history without repeating prefixes."""
    contracts = (
        identity.model_config_fingerprint,
        identity.exact_registry_fingerprint,
        identity.event_contract_fingerprint,
        identity.sequence_contract_fingerprint,
    )
    if any(value is None for value in contracts):
        raise ValueError("temporal pretraining identity is incomplete")
    if any(
        example.decision_index is None
        or example.source_replay_sha256 is None
        or example.actor_row.public_event_delta is None
        or example.accepted_action is None
        for example in examples
    ):
        raise ValueError("temporal pretraining row is incomplete")
    expected_engine_fact_fingerprint = identity.engine_fact_producer_fingerprint
    observed_engine_fact_fingerprints = {
        example.actor_row.engine_fact_producer_fingerprint for example in examples
    }
    if observed_engine_fact_fingerprints != {expected_engine_fact_fingerprint}:
        raise ValueError("temporal engine-fact producer identity changed within a part")
    event_deltas = cast(
        tuple[PublicEventDelta, ...],
        tuple(example.actor_row.public_event_delta for example in examples),
    )
    accepted_actions = cast(
        tuple[AcceptedActionRecord, ...],
        tuple(example.accepted_action for example in examples),
    )
    decision_indices = cast(
        tuple[int, ...],
        tuple(example.decision_index for example in examples),
    )
    events = build_public_event_array_block(event_deltas)
    action_lengths = tuple(len(action.option_types) for action in accepted_actions)
    action_count = sum(action_lengths)
    sequence_offsets = [0]
    sequence_episode_ids: list[int] = []
    sequence_seats: list[int] = []
    previous_key: tuple[int, int] | None = None
    previous_clock = -1
    for row, (example, clock) in enumerate(
        zip(examples, decision_indices, strict=True)
    ):
        key = (example.episode_id, example.player_index)
        if key != previous_key:
            if previous_key is not None:
                sequence_offsets.append(row)
            sequence_episode_ids.append(example.episode_id)
            sequence_seats.append(example.player_index)
            previous_key = key
            previous_clock = -1
        if clock != previous_clock + 1:
            raise ValueError("temporal pretraining decision clock is discontinuous")
        previous_clock = clock
    sequence_offsets.append(len(examples))

    def accepted_vector(
        field: str,
        *,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[np.generic]:
        return np.asarray(
            [value for action in accepted_actions for value in getattr(action, field)],
            dtype=dtype,
        )

    def accepted_matrix(
        field: str,
        *,
        width: int,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[np.generic]:
        values = [
            values for action in accepted_actions for values in getattr(action, field)
        ]
        return np.asarray(values, dtype=dtype).reshape(action_count, width)

    arrays = {
        "decision_indices": np.asarray(
            decision_indices,
            dtype=np.int32,
        ),
        "source_steps": np.asarray(
            [example.step_index for example in examples],
            dtype=np.int32,
        ),
        "source_replay_sha256s": _strings(
            [str(example.source_replay_sha256) for example in examples]
        ),
        "route_expert_ids": _strings(
            [example.route_expert_id or "" for example in examples]
        ),
        "sequence_offsets": np.asarray(sequence_offsets, dtype=np.int64),
        "sequence_episode_ids": np.asarray(sequence_episode_ids, dtype=np.int64),
        "sequence_seats": np.asarray(sequence_seats, dtype=np.int8),
        "model_config_fingerprint": _strings([str(contracts[0])]),
        "exact_registry_fingerprint": _strings([str(contracts[1])]),
        "public_catalog_fingerprint": _strings([identity.public_catalog_fingerprint]),
        "input_contract_fingerprint": _strings([identity.input_contract_fingerprint]),
        "event_contract_fingerprint": _strings([str(contracts[2])]),
        "sequence_contract_fingerprint": _strings([str(contracts[3])]),
        "engine_fact_available": np.asarray(
            [
                example.actor_row.engine_fact_producer_fingerprint is not None
                for example in examples
            ],
            dtype=np.bool_,
        ),
        "event_offsets": events.event_offsets,
        "event_types": events.event_types,
        "event_actor_roles": events.actor_roles,
        "event_from_areas": events.from_areas,
        "event_to_areas": events.to_areas,
        "event_card_ids": events.card_ids,
        "event_serials": events.serials,
        "event_entity_mask": events.entity_mask,
        "event_attack_ids": events.attack_ids,
        "event_attack_id_mask": events.attack_id_mask,
        "event_values": events.values,
        "event_value_mask": events.value_mask,
        "event_categorical_values": events.categorical_values,
        "event_overflow_offsets": events.overflow_offsets,
        "event_overflow_types": events.overflow_event_types,
        "event_overflow_actor_roles": events.overflow_actor_roles,
        "event_overflow_counts": events.overflow_counts,
        "accepted_action_offsets": _offsets(action_lengths),
        "accepted_action_schema_versions": np.asarray(
            [action.schema_version for action in accepted_actions],
            dtype=np.int16,
        ),
        "accepted_action_stable_ids": _strings(
            [action.stable_identity for action in accepted_actions]
        ),
        "accepted_action_prompt_contexts": np.asarray(
            [action.prompt_context for action in accepted_actions],
            dtype=np.int16,
        ),
        "accepted_action_ordered": np.asarray(
            [action.ordered for action in accepted_actions],
            dtype=np.bool_,
        ),
        "accepted_action_stop_sampled": np.asarray(
            [action.stop_sampled for action in accepted_actions],
            dtype=np.bool_,
        ),
        "accepted_action_fallback": np.asarray(
            [action.fallback for action in accepted_actions],
            dtype=np.bool_,
        ),
        "accepted_action_option_types": accepted_vector(
            "option_types",
            dtype=np.int16,
        ),
        "accepted_action_option_contexts": accepted_vector(
            "option_contexts",
            dtype=np.int16,
        ),
        "accepted_action_card_ids": accepted_vector(
            "card_ids",
            dtype=np.int32,
        ),
        "accepted_action_attack_ids": accepted_vector(
            "attack_ids",
            dtype=np.int32,
        ),
        "accepted_action_option_scalars": accepted_matrix(
            "option_scalars",
            width=SCALAR_FEATURE_SIZE,
            dtype=np.float32,
        ),
        "accepted_action_entity_card_ids": accepted_matrix(
            "entity_card_ids",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int32,
        ),
        "accepted_action_entity_areas": accepted_matrix(
            "entity_areas",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int16,
        ),
        "accepted_action_entity_owner_roles": accepted_matrix(
            "entity_owner_roles",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int8,
        ),
        "accepted_action_entity_token_kinds": accepted_matrix(
            "entity_token_kinds",
            width=MAX_ENTITY_SLOTS,
            dtype=np.int8,
        ),
        "accepted_action_entity_scalars": accepted_matrix(
            "entity_scalars",
            width=MAX_ENTITY_SLOTS * TOKEN_SCALAR_SIZE,
            dtype=np.float32,
        ).reshape(action_count, MAX_ENTITY_SLOTS, TOKEN_SCALAR_SIZE),
    }
    if identity.format == TEMPORAL_PRETRAINING_SHARD_SCHEMA:
        arrays["engine_fact_producer_fingerprint"] = _strings(
            [expected_engine_fact_fingerprint or ""]
        )
    return arrays


def _concat(
    rows: Sequence[object],
    field: str,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.generic]:
    values = [np.asarray(getattr(row, field), dtype=dtype) for row in rows]
    return np.concatenate(values) if values else np.asarray([], dtype=dtype)


def _concat_2d(
    rows: Sequence[object],
    field: str,
    width: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.generic]:
    values = [np.asarray(getattr(row, field), dtype=dtype) for row in rows]
    if not values:
        return np.empty((0, width), dtype=dtype)
    result = np.concatenate(values, axis=0)
    return result.reshape(-1, width)


def _offsets(lengths: Sequence[int]) -> npt.NDArray[np.int64]:
    return np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.cumsum(np.asarray(lengths, dtype=np.int64)),
        )
    )


def _strings(values: Sequence[str]) -> npt.NDArray[np.str_]:
    """Persist Unicode values without object arrays or pickle."""
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{max(width, 1)}")


def _row_bounds(
    offsets: npt.NDArray[np.generic],
    row: int,
) -> tuple[int, int]:
    return (int(offsets[row]), int(offsets[row + 1]))


def _row_split(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    row: int,
) -> ReplaySplit:
    codes = arrays.get("split_codes")
    if codes is None:
        return "train"
    try:
        return _CODE_TO_SPLIT[int(codes[row])]
    except KeyError as error:
        raise ValueError("pretraining row has an invalid split code") from error


def _split_counts(values: Iterable[ReplaySplit]) -> dict[ReplaySplit, int]:
    counts = Counter(values)
    return {split: int(counts[split]) for split in REPLAY_SPLITS}


def _validate_ragged(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    rows: int,
    *,
    prefix: str,
    values: tuple[str, ...],
) -> None:
    offsets = np.asarray(arrays[f"{prefix}_offsets"], dtype=np.int64)
    if (
        offsets.shape != (rows + 1,)
        or offsets[0] != 0
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"pretraining {prefix} offsets are invalid")
    expected = int(offsets[-1])
    if any(len(arrays[f"{prefix}_{value}"]) != expected for value in values):
        raise ValueError(f"pretraining {prefix} values are misaligned")


def _validate_resume_identity(
    existing: ReplayPretrainingDatasetManifest,
    expected: ReplayPretrainingDatasetManifest,
) -> None:
    fields = (
        "format",
        "source_manifest_sha256",
        "replay_manifest_sha256",
        "top_teams_sha256",
        "episode_teams_sha256",
        "source_selection",
        "public_catalog_fingerprint",
        "input_contract_fingerprint",
        "model_config_fingerprint",
        "exact_registry_fingerprint",
        "event_contract_fingerprint",
        "sequence_contract_fingerprint",
        "engine_fact_producer_fingerprint",
        "target_deck_digest",
        "source_replays",
        "split_assignment_fingerprint",
        "split_source_replays",
        "dates",
        "teams",
    )
    if any(getattr(existing, field) != getattr(expected, field) for field in fields):
        raise ValueError("pretraining dataset resume identity changed")


def _temporary_path(filename: str) -> Path:
    root = Path(__file__).resolve().parents[3] / "tmp" / "pretraining_shards"
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{uuid.uuid4().hex}-{filename}"


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("artifact identity must be lowercase SHA-256")
    return normalized


__all__ = [
    "LEGACY_TEMPORAL_PRETRAINING_SHARD_SCHEMA",
    "LEGACY_PRETRAINING_SHARD_SCHEMA",
    "LoadedPretrainingPart",
    "PRETRAINING_SHARD_SCHEMA",
    "TEMPORAL_PRETRAINING_SHARD_SCHEMA",
    "TEMPORAL_PRETRAINING_SHARD_SCHEMAS",
    "PretrainingPartRecord",
    "REPLAY_SPLITS",
    "RejectedReplayRecord",
    "ReplaySplit",
    "ReplayPretrainingDatasetManifest",
    "ReplayPretrainingExample",
    "ReplayPretrainingShardWriter",
    "file_sha256",
    "is_temporal_pretraining_format",
    "iter_pretraining_rows",
    "load_pretraining_manifest",
    "load_pretraining_metadata_columns",
    "load_pretraining_part",
    "load_temporal_pretraining_geometry",
    "pretraining_split_indices",
    "validate_pretraining_arrays",
    "write_pretraining_part",
]
