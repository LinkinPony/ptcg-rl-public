"""Compact streaming NPZ shards for clean stateless PPO fragments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.actions.encoding import SCALAR_FEATURE_SIZE
from ptcg_rl.belief.public_catalog import PublicDeckPosteriorArrays
from ptcg_rl.context.public_event_arrays import build_public_event_array_block
from ptcg_rl.context.public_events import (
    PUBLIC_EVENT_CATEGORICAL_SIZE,
    PUBLIC_EVENT_ENTITY_COUNT,
)
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model.policy import MAX_ENTITY_SLOTS
from ptcg_rl.model.state_encoder import TOKEN_SCALAR_SIZE
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.stateless_fragment import StatelessFragment

STATELESS_FRAGMENT_ARRAY_SCHEMA: Literal["stateless-fragment-npz-v1"] = (
    "stateless-fragment-npz-v1"
)
SEQUENCE_FRAGMENT_ARRAY_SCHEMA: Literal["sequence-fragment-npz-v2"] = (
    "sequence-fragment-npz-v2"
)

_ARRAY_KEYS = frozenset(
    {
        "schema_version",
        "fragment_ids",
        "fragment_decision_offsets",
        "game_ids",
        "seats",
        "start_decision_indices",
        "own_decks",
        "opponent_decks",
        "own_deck_digests",
        "opponent_deck_digests",
        "curriculum_generations",
        "assignment_ids",
        "opponent_artifact_fingerprints",
        "terminal",
        "truncated",
        "bootstrap_values",
        "terminal_rewards",
        "horizons",
        "behavior_policy_versions",
        "behavior_policy_fingerprints",
        "model_config_fingerprints",
        "action_schema_fingerprints",
        "public_context_fingerprints",
        "card_catalog_fingerprints",
        "public_deck_catalog_fingerprints",
        "exact_registry_fingerprints",
        "belief_target_semantics_fingerprints",
        "input_contract_fingerprints",
        "resolved_config_fingerprints",
        "decision_fragment_indices",
        "decision_indices",
        "action_logprobs",
        "root_values",
        "rewards",
        "stop_sampled",
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
        "option_entity_slot_mask",
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
        "token_offsets",
        "token_logprobs",
        "prefix_values",
    }
)
_SEQUENCE_ARRAY_KEYS = _ARRAY_KEYS | frozenset(
    {
        "fragment_schema_versions",
        "sequence_contract_fingerprints",
        "engine_fact_producer_fingerprints",
        "sequence_request_ids",
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
        "accepted_action_stable_ids",
        "accepted_action_prompt_contexts",
        "accepted_action_ordered",
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
)
STATELESS_FRAGMENT_ARRAY_KEYS = _ARRAY_KEYS
SEQUENCE_FRAGMENT_ARRAY_KEYS = _SEQUENCE_ARRAY_KEYS


class FragmentPartRecord(BaseModel):
    """One immutable compact part in the recovery manifest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    filename: str
    sha256: str
    size_bytes: int = Field(ge=1)
    fragments: int = Field(ge=1)
    decisions: int = Field(ge=1)

    @field_validator("filename")
    @classmethod
    def simple_filename(cls, value: str) -> str:
        """Keep parts relocatable under one directory."""
        if Path(value).name != value or not value.endswith(".npz"):
            raise ValueError("fragment part filename is invalid")
        return value

    @field_validator("sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require a complete immutable part fingerprint."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("fragment part fingerprint must be SHA-256")
        return normalized


class CompactFragmentManifest(BaseModel):
    """Small durable cursor for a stream of immutable NPZ parts."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    format: Literal[
        "stateless-fragment-npz-v1",
        "sequence-fragment-npz-v2",
    ] = "stateless-fragment-npz-v1"
    static_contract_fingerprint: str
    horizon: int = Field(ge=1)
    fragments_per_part: int = Field(ge=1)
    fragments_committed: int = Field(default=0, ge=0)
    decisions_committed: int = Field(default=0, ge=0)
    parts: tuple[FragmentPartRecord, ...] = ()

    @field_validator("static_contract_fingerprint")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require one immutable run-level fragment contract."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("fragment static contract must be SHA-256")
        return normalized


@dataclass(frozen=True)
class CompactFragmentPart:
    """Loaded no-pickle arrays after structural validation."""

    path: Path | None
    arrays: Mapping[str, npt.NDArray[np.generic]]

    @property
    def fragment_count(self) -> int:
        """Return the number of fragment contexts."""
        return int(self.arrays["fragment_ids"].shape[0])

    @property
    def decision_count(self) -> int:
        """Return the number of flattened decisions."""
        return int(self.arrays["decision_indices"].shape[0])


class CompactFragmentShardWriter:
    """Bounded-memory writer with immutable parts and exact resume cursor."""

    def __init__(
        self,
        output_dir: Path,
        *,
        static_contract_fingerprint: str,
        horizon: int,
        fragments_per_part: int,
        resume: bool = False,
        sequence: bool = False,
    ) -> None:
        """Create or resume one compact fragment stream."""
        if horizon <= 0 or fragments_per_part <= 0:
            raise ValueError("fragment horizon and part size must be positive")
        self.output_dir = output_dir
        self.parts_dir = output_dir / "parts"
        self.manifest_path = output_dir / "manifest.json"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self._buffer: list[StatelessFragment] = []
        self._closed = False
        self._fragment_ids: set[str] = set()
        if resume:
            self.manifest = load_compact_fragment_manifest(self.manifest_path)
            expected = (
                (
                    SEQUENCE_FRAGMENT_ARRAY_SCHEMA
                    if sequence
                    else STATELESS_FRAGMENT_ARRAY_SCHEMA
                ),
                static_contract_fingerprint,
                horizon,
                fragments_per_part,
            )
            actual = (
                self.manifest.format,
                self.manifest.static_contract_fingerprint,
                self.manifest.horizon,
                self.manifest.fragments_per_part,
            )
            if actual != expected:
                raise ValueError("fragment writer resume contract mismatch")
            self._verify_committed_parts()
        else:
            if self.manifest_path.exists() or any(self.parts_dir.iterdir()):
                raise FileExistsError("fragment output directory is not empty")
            manifest_format: Literal[
                "stateless-fragment-npz-v1",
                "sequence-fragment-npz-v2",
            ] = (
                    SEQUENCE_FRAGMENT_ARRAY_SCHEMA
                    if sequence
                    else STATELESS_FRAGMENT_ARRAY_SCHEMA
                )
            self.manifest = CompactFragmentManifest(
                format=manifest_format,
                static_contract_fingerprint=static_contract_fingerprint,
                horizon=horizon,
                fragments_per_part=fragments_per_part,
            )
            self._write_manifest()

    def add(self, fragment: StatelessFragment) -> None:
        """Buffer one complete terminal/truncated GAE unit."""
        self._ensure_open()
        if (
            fragment.identity.static_contract_fingerprint
            != self.manifest.static_contract_fingerprint
            or fragment.identity.horizon != self.manifest.horizon
        ):
            raise ValueError("fragment differs from writer static contract")
        if fragment.fragment_id in self._fragment_ids or any(
            buffered.fragment_id == fragment.fragment_id for buffered in self._buffer
        ):
            raise ValueError("duplicate fragment identity")
        self._buffer.append(fragment)
        if len(self._buffer) >= self.manifest.fragments_per_part:
            self.flush()

    def import_part(
        self,
        source: Path,
        *,
        fragments_by_id: Mapping[str, StatelessFragment],
    ) -> FragmentPartRecord:
        """Atomically adopt an already-encoded worker part without recompression."""
        self._ensure_open()
        self.flush()
        part = load_compact_fragment_part(source)
        fragment_ids = tuple(str(value) for value in part.arrays["fragment_ids"])
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("imported part contains duplicate fragment identity")
        if any(fragment_id in self._fragment_ids for fragment_id in fragment_ids):
            raise ValueError("duplicate fragment identity")
        try:
            fragments = tuple(
                fragments_by_id[fragment_id] for fragment_id in fragment_ids
            )
        except KeyError as error:
            raise ValueError(
                "imported part contains an unknown fragment identity"
            ) from error
        if any(
            fragment.identity.static_contract_fingerprint
            != self.manifest.static_contract_fingerprint
            or fragment.identity.horizon != self.manifest.horizon
            for fragment in fragments
        ):
            raise ValueError("imported part differs from writer static contract")
        if sum(len(fragment.decisions) for fragment in fragments) != (
            part.decision_count
        ):
            raise ValueError("imported part decision count changed")
        return self._adopt_loaded_part(part)

    def import_validated_part(
        self,
        part: CompactFragmentPart,
        *,
        static_contract_fingerprint: str,
    ) -> FragmentPartRecord:
        """Adopt a semantically validated array part without rebuilding objects.

        The caller must have included ``part`` in an array-native replay
        validation pass.  Binding the returned static contract to this writer
        prevents a validated part from being redirected into another run.
        """
        self._ensure_open()
        self.flush()
        if static_contract_fingerprint != self.manifest.static_contract_fingerprint:
            raise ValueError("imported array part differs from writer static contract")
        return self._adopt_loaded_part(part)

    def _adopt_loaded_part(
        self,
        part: CompactFragmentPart,
    ) -> FragmentPartRecord:
        """Atomically publish one loaded, structurally validated part."""
        source = part.path
        if source is None:
            raise ValueError("in-memory fragment parts cannot be persisted")
        fragment_ids = tuple(str(value) for value in part.arrays["fragment_ids"])
        if len(fragment_ids) != len(set(fragment_ids)):
            raise ValueError("imported part contains duplicate fragment identity")
        if any(fragment_id in self._fragment_ids for fragment_id in fragment_ids):
            raise ValueError("duplicate fragment identity")
        part_index = len(self.manifest.parts)
        filename = f"part-{part_index:08d}.npz"
        destination = self.parts_dir / filename
        if destination.exists():
            raise FileExistsError(destination)
        record = FragmentPartRecord(
            filename=filename,
            sha256=_file_sha256(source),
            size_bytes=source.stat().st_size,
            fragments=part.fragment_count,
            decisions=part.decision_count,
        )
        _publish_file_atomic(source, destination)
        _fsync_directory(self.parts_dir)
        proposed = self.manifest.model_copy(
            update={
                "fragments_committed": (
                    self.manifest.fragments_committed + record.fragments
                ),
                "decisions_committed": (
                    self.manifest.decisions_committed + record.decisions
                ),
                "parts": self.manifest.parts + (record,),
            }
        )
        atomic_write_bytes(
            self.manifest_path,
            json_payload(proposed.model_dump(mode="json")),
            overwrite=True,
        )
        self.manifest = proposed
        self._fragment_ids.update(fragment_ids)
        source.unlink(missing_ok=True)
        return record

    def flush(self) -> FragmentPartRecord | None:
        """Atomically publish the current bounded fragment buffer."""
        self._ensure_open()
        if not self._buffer:
            return None
        part_index = len(self.manifest.parts)
        filename = f"part-{part_index:08d}.npz"
        destination = self.parts_dir / filename
        if destination.exists():
            raise FileExistsError(destination)
        arrays = _fragment_arrays(tuple(self._buffer))
        _validate_fragment_arrays(arrays)
        temporary_root = (
            Path(__file__).resolve().parents[3] / "tmp" / "stateless_fragments"
        )
        temporary_root.mkdir(parents=True, exist_ok=True)
        temporary = temporary_root / f"{uuid.uuid4().hex}.npz"
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)  # type: ignore[arg-type]
                handle.flush()
                os.fsync(handle.fileno())
            record = FragmentPartRecord(
                filename=filename,
                sha256=_file_sha256(temporary),
                size_bytes=temporary.stat().st_size,
                fragments=len(self._buffer),
                decisions=sum(len(fragment.decisions) for fragment in self._buffer),
            )
            _publish_file_atomic(temporary, destination)
            _fsync_directory(self.parts_dir)
        finally:
            temporary.unlink(missing_ok=True)
        proposed = self.manifest.model_copy(
            update={
                "fragments_committed": (
                    self.manifest.fragments_committed + record.fragments
                ),
                "decisions_committed": (
                    self.manifest.decisions_committed + record.decisions
                ),
                "parts": self.manifest.parts + (record,),
            }
        )
        atomic_write_bytes(
            self.manifest_path,
            json_payload(proposed.model_dump(mode="json")),
            overwrite=True,
        )
        self.manifest = proposed
        self._fragment_ids.update(fragment.fragment_id for fragment in self._buffer)
        self._buffer.clear()
        return record

    def close(self) -> CompactFragmentManifest:
        """Flush remaining fragments and close the writer."""
        if self._closed:
            return self.manifest
        self.flush()
        self._closed = True
        return self.manifest

    def _verify_committed_parts(self) -> None:
        fragments = 0
        decisions = 0
        for index, record in enumerate(self.manifest.parts):
            expected_name = f"part-{index:08d}.npz"
            if record.filename != expected_name:
                raise ValueError("fragment part sequence is not contiguous")
            path = self.parts_dir / record.filename
            if (
                not path.is_file()
                or path.stat().st_size != record.size_bytes
                or _file_sha256(path) != record.sha256
            ):
                raise ValueError("committed fragment part identity changed")
            part = load_compact_fragment_part(path)
            if (
                part.fragment_count != record.fragments
                or part.decision_count != record.decisions
            ):
                raise ValueError("fragment part counts differ from manifest")
            self._fragment_ids.update(
                str(value) for value in part.arrays["fragment_ids"]
            )
            fragments += record.fragments
            decisions += record.decisions
        if (
            fragments != self.manifest.fragments_committed
            or decisions != self.manifest.decisions_committed
        ):
            raise ValueError("fragment manifest committed counters are corrupt")

    def _write_manifest(self) -> None:
        atomic_write_bytes(
            self.manifest_path,
            json_payload(self.manifest.model_dump(mode="json")),
            overwrite=True,
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("fragment writer is closed")


def load_compact_fragment_manifest(path: Path) -> CompactFragmentManifest:
    """Load a compact stream manifest without reading repository config."""
    return CompactFragmentManifest.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )


def load_compact_fragment_part(path: Path) -> CompactFragmentPart:
    """Load, detach, and validate one immutable no-pickle NPZ part."""
    with np.load(path, allow_pickle=False) as loaded:
        arrays = {key: np.asarray(loaded[key]).copy() for key in loaded.files}
    _validate_fragment_arrays(arrays)
    return CompactFragmentPart(path=path, arrays=arrays)


def _fragment_arrays(
    fragments: Sequence[StatelessFragment],
) -> dict[str, npt.NDArray[np.generic]]:
    decisions = tuple(
        decision for fragment in fragments for decision in fragment.decisions
    )
    fragment_decision_offsets = _offsets(
        [len(fragment.decisions) for fragment in fragments]
    )
    state_lengths = [len(decision.actor_row.state.card_ids) for decision in decisions]
    attachment_lengths = [
        len(decision.actor_row.state.attachment_card_ids) for decision in decisions
    ]
    option_lengths = [len(decision.actor_row.options) for decision in decisions]
    belief_rows = [
        _belief_arrays(decision.actor_row.belief_summary) for decision in decisions
    ]
    belief_lengths = [len(card_ids) for card_ids, _counts in belief_rows]
    known_lengths = [len(decision.known_opponent_counts) for decision in decisions]
    action_lengths = [len(decision.action) for decision in decisions]
    token_lengths = [len(decision.token_logprobs) for decision in decisions]
    schema_versions = {fragment.identity.schema_version for fragment in fragments}
    if len(schema_versions) != 1:
        raise ValueError("fragment part mixes trajectory schema versions")
    schema_version = schema_versions.pop()
    arrays: dict[str, npt.NDArray[np.generic]] = {
        "schema_version": np.asarray([schema_version], dtype=np.int16),
        "fragment_ids": _strings([fragment.fragment_id for fragment in fragments]),
        "fragment_decision_offsets": fragment_decision_offsets,
        "game_ids": _strings([fragment.context.game_id for fragment in fragments]),
        "seats": np.asarray(
            [fragment.context.seat for fragment in fragments],
            dtype=np.int8,
        ),
        "start_decision_indices": np.asarray(
            [fragment.context.start_decision_index for fragment in fragments],
            dtype=np.int64,
        ),
        "own_decks": np.asarray(
            [fragment.context.own_deck for fragment in fragments],
            dtype=np.int32,
        ),
        "opponent_decks": np.asarray(
            [fragment.context.opponent_deck for fragment in fragments],
            dtype=np.int32,
        ),
        "own_deck_digests": _strings(
            [fragment.context.own_deck_digest for fragment in fragments]
        ),
        "opponent_deck_digests": _strings(
            [fragment.context.opponent_deck_digest for fragment in fragments]
        ),
        "curriculum_generations": np.asarray(
            [fragment.context.curriculum_generation for fragment in fragments],
            dtype=np.int64,
        ),
        "assignment_ids": _strings(
            [fragment.context.assignment_id for fragment in fragments]
        ),
        "opponent_artifact_fingerprints": _strings(
            [fragment.context.opponent_artifact_fingerprint for fragment in fragments]
        ),
        "terminal": np.asarray(
            [fragment.terminal for fragment in fragments],
            dtype=np.bool_,
        ),
        "truncated": np.asarray(
            [fragment.truncated for fragment in fragments],
            dtype=np.bool_,
        ),
        "bootstrap_values": np.asarray(
            [fragment.bootstrap_value for fragment in fragments],
            dtype=np.float32,
        ),
        "terminal_rewards": np.asarray(
            [fragment.terminal_reward for fragment in fragments],
            dtype=np.float32,
        ),
        "horizons": np.asarray(
            [fragment.identity.horizon for fragment in fragments],
            dtype=np.int32,
        ),
        "behavior_policy_versions": np.asarray(
            [fragment.identity.behavior_policy_version for fragment in fragments],
            dtype=np.int64,
        ),
        **_fragment_identity_string_arrays(fragments),
        "decision_fragment_indices": np.repeat(
            np.arange(len(fragments), dtype=np.int32),
            np.diff(fragment_decision_offsets),
        ),
        "decision_indices": np.asarray(
            [decision.decision_index for decision in decisions],
            dtype=np.int64,
        ),
        "action_logprobs": np.asarray(
            [decision.action_logprob for decision in decisions],
            dtype=np.float32,
        ),
        "root_values": np.asarray(
            [decision.root_value for decision in decisions],
            dtype=np.float32,
        ),
        "rewards": np.asarray(
            [decision.reward for decision in decisions],
            dtype=np.float32,
        ),
        "stop_sampled": np.asarray(
            [decision.stop_sampled for decision in decisions],
            dtype=np.bool_,
        ),
        "state_offsets": _offsets(state_lengths),
        "state_card_ids": _concatenate(
            [decision.actor_row.state.card_ids for decision in decisions],
            dtype=np.int32,
        ),
        "state_areas": _concatenate(
            [decision.actor_row.state.areas for decision in decisions],
            dtype=np.int16,
        ),
        "state_owner_roles": _concatenate(
            [decision.actor_row.state.owner_roles for decision in decisions],
            dtype=np.int8,
        ),
        "state_token_kinds": _concatenate(
            [decision.actor_row.state.token_kinds for decision in decisions],
            dtype=np.int8,
        ),
        "state_scalars": _concatenate_rows(
            [decision.actor_row.state.scalars for decision in decisions],
            width=TOKEN_SCALAR_SIZE,
            dtype=np.float32,
        ),
        "state_last_attack_ids": _concatenate(
            [decision.actor_row.state.last_attack_ids for decision in decisions],
            dtype=np.int32,
        ),
        "state_entity_slots": _concatenate(
            [decision.actor_row.state.entity_slots for decision in decisions],
            dtype=np.uint8,
        ),
        "attachment_offsets": _offsets(attachment_lengths),
        "attachment_card_ids": _concatenate(
            [decision.actor_row.state.attachment_card_ids for decision in decisions],
            dtype=np.int32,
        ),
        "attachment_parent_indices": _concatenate(
            [
                decision.actor_row.state.attachment_parent_indices
                for decision in decisions
            ],
            dtype=np.int32,
        ),
        "attachment_kinds": _concatenate(
            [decision.actor_row.state.attachment_kinds for decision in decisions],
            dtype=np.int8,
        ),
        "option_offsets": _offsets(option_lengths),
        **_option_arrays(decisions),
        "min_counts": np.asarray(
            [decision.actor_row.min_count for decision in decisions],
            dtype=np.int16,
        ),
        "max_counts": np.asarray(
            [decision.actor_row.max_count for decision in decisions],
            dtype=np.int16,
        ),
        "belief_offsets": _offsets(belief_lengths),
        "belief_card_ids": _concatenate(
            [card_ids for card_ids, _counts in belief_rows],
            dtype=np.int32,
        ),
        "belief_expected_counts": _concatenate(
            [counts for _card_ids, counts in belief_rows],
            dtype=np.float32,
        ),
        "belief_scalars": np.asarray(
            [
                (
                    decision.actor_row.belief_summary.entropy,
                    decision.actor_row.belief_summary.compatible_deck_count,
                    decision.actor_row.belief_summary.public_evidence_count,
                    decision.actor_row.belief_summary.unknown_probability,
                )
                for decision in decisions
            ],
            dtype=np.float32,
        ),
        "known_offsets": _offsets(known_lengths),
        "known_card_ids": np.asarray(
            [
                card_id
                for decision in decisions
                for card_id, _count in decision.known_opponent_counts
            ],
            dtype=np.int32,
        ),
        "known_counts": np.asarray(
            [
                count
                for decision in decisions
                for _card_id, count in decision.known_opponent_counts
            ],
            dtype=np.int16,
        ),
        "action_offsets": _offsets(action_lengths),
        "action_choices": np.asarray(
            [choice for decision in decisions for choice in decision.action],
            dtype=np.int32,
        ),
        "token_offsets": _offsets(token_lengths),
        "token_logprobs": np.asarray(
            [value for decision in decisions for value in decision.token_logprobs],
            dtype=np.float32,
        ),
        "prefix_values": np.asarray(
            [value for decision in decisions for value in decision.prefix_values],
            dtype=np.float32,
        ),
    }
    if schema_version == 2:
        arrays.update(_sequence_arrays(fragments, decisions))
    return arrays


def _sequence_arrays(
    fragments: Sequence[StatelessFragment],
    decisions: Sequence[Any],
) -> dict[str, npt.NDArray[np.generic]]:
    """Flatten raw EVENT and stable accepted ACTION truth for schema V2."""
    event_deltas = []
    accepted_actions = []
    for decision in decisions:
        if (
            decision.public_event_delta is None
            or decision.accepted_action is None
            or decision.actor_row.sequence_identity is None
        ):
            raise ValueError("sequence fragment decision is incomplete")
        event_deltas.append(decision.public_event_delta)
        accepted_actions.append(decision.accepted_action)
    events = build_public_event_array_block(tuple(event_deltas))
    accepted_count = sum(len(action.option_types) for action in accepted_actions)

    def accepted_matrix(
        field: str,
        *,
        width: int,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[np.generic]:
        values = [
            row
            for action in accepted_actions
            for row in getattr(action, field)
        ]
        return np.asarray(values, dtype=dtype).reshape(accepted_count, width)

    def accepted_vector(
        field: str,
        *,
        dtype: npt.DTypeLike,
    ) -> npt.NDArray[np.generic]:
        return np.asarray(
            [
                value
                for action in accepted_actions
                for value in getattr(action, field)
            ],
            dtype=dtype,
        )

    return {
        "fragment_schema_versions": np.asarray(
            [fragment.identity.schema_version for fragment in fragments],
            dtype=np.int16,
        ),
        "sequence_contract_fingerprints": _strings(
            [
                str(fragment.identity.sequence_contract_fingerprint)
                for fragment in fragments
            ]
        ),
        "engine_fact_producer_fingerprints": _strings(
            [
                decision.actor_row.engine_fact_producer_fingerprint or ""
                for decision in decisions
            ]
        ),
        "sequence_request_ids": _strings(
            [
                decision.actor_row.sequence_identity.request_id
                for decision in decisions
            ]
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
        "accepted_action_entity_scalars": np.asarray(
            [
                row
                for action in accepted_actions
                for row in action.entity_scalars
            ],
            dtype=np.float32,
        ).reshape(
            accepted_count,
            MAX_ENTITY_SLOTS,
            TOKEN_SCALAR_SIZE,
        ),
    }


def _fragment_identity_string_arrays(
    fragments: Sequence[StatelessFragment],
) -> dict[str, npt.NDArray[np.str_]]:
    fields = {
        "behavior_policy_fingerprints": "behavior_policy_fingerprint",
        "model_config_fingerprints": "model_config_fingerprint",
        "action_schema_fingerprints": "action_schema_fingerprint",
        "public_context_fingerprints": "public_context_fingerprint",
        "card_catalog_fingerprints": "card_catalog_fingerprint",
        "public_deck_catalog_fingerprints": "public_deck_catalog_fingerprint",
        "exact_registry_fingerprints": "exact_registry_fingerprint",
        "belief_target_semantics_fingerprints": ("belief_target_semantics_fingerprint"),
        "input_contract_fingerprints": "input_contract_fingerprint",
        "resolved_config_fingerprints": "resolved_config_fingerprint",
    }
    return {
        output: _strings(
            [str(getattr(fragment.identity, field)) for fragment in fragments]
        )
        for output, field in fields.items()
    }


def _belief_arrays(
    summary: Any,
) -> tuple[npt.NDArray[np.generic], npt.NDArray[np.generic]]:
    if isinstance(summary, PublicDeckPosteriorArrays):
        return (summary.card_ids, summary.expected_counts)
    return (
        np.asarray(
            [item.card_id for item in summary.expected_remaining],
            dtype=np.int32,
        ),
        np.asarray(
            [item.expected_count for item in summary.expected_remaining],
            dtype=np.float32,
        ),
    )


def _option_arrays(
    decisions: Sequence[Any],
) -> dict[str, npt.NDArray[np.generic]]:
    options = [decision.actor_row.options for decision in decisions]
    return {
        "option_types": _concatenate(
            [option.option_types for option in options],
            dtype=np.int16,
        ),
        "option_contexts": _concatenate(
            [option.contexts for option in options],
            dtype=np.int16,
        ),
        "option_entity_slots": _concatenate_rows(
            [option.entity_slots for option in options],
            width=MAX_ENTITY_SLOTS,
            dtype=np.int32,
        ),
        "option_entity_slot_mask": _concatenate_rows(
            [option.entity_slot_mask for option in options],
            width=MAX_ENTITY_SLOTS,
            dtype=np.bool_,
        ),
        "option_attack_ids": _concatenate(
            [option.attack_ids for option in options],
            dtype=np.int32,
        ),
        "option_card_ids": _concatenate(
            [option.card_ids for option in options],
            dtype=np.int32,
        ),
        "option_scalars": _concatenate_rows(
            [option.scalars for option in options],
            width=SCALAR_FEATURE_SIZE,
            dtype=np.float32,
        ),
        "option_dynamic_effect_features": _concatenate_rows(
            [option.dynamic_effect_features for option in options],
            width=DYNAMIC_EFFECT_FEATURE_SIZE,
            dtype=np.float32,
        ),
        "option_dynamic_effect_masks": _concatenate(
            [option.dynamic_effect_masks for option in options],
            dtype=np.bool_,
        ),
    }


def _validate_fragment_arrays(
    arrays: Mapping[str, npt.NDArray[np.generic]],
) -> None:
    raw_version = np.asarray(arrays.get("schema_version", ())).tolist()
    if raw_version not in ([1], [2]):
        raise ValueError("unsupported fragment array schema version")
    schema_version = int(raw_version[0])
    expected_keys = _SEQUENCE_ARRAY_KEYS if schema_version == 2 else _ARRAY_KEYS
    if set(arrays) != expected_keys:
        missing = sorted(expected_keys - set(arrays))
        extra = sorted(set(arrays) - expected_keys)
        raise ValueError(
            f"fragment array schema mismatch: missing={missing}, extra={extra}"
        )
    fragments = len(arrays["fragment_ids"])
    decisions = len(arrays["decision_indices"])
    if fragments <= 0 or decisions <= 0:
        raise ValueError("fragment part cannot be empty")
    _validate_offsets(
        arrays["fragment_decision_offsets"],
        rows=fragments,
        values=decisions,
        name="fragment decisions",
    )
    if arrays["own_decks"].shape != (fragments, 60) or arrays[
        "opponent_decks"
    ].shape != (fragments, 60):
        raise ValueError("fragment deck contexts must have shape [fragments, 60]")
    fragment_fields = (
        "game_ids",
        "seats",
        "start_decision_indices",
        "own_deck_digests",
        "opponent_deck_digests",
        "curriculum_generations",
        "assignment_ids",
        "opponent_artifact_fingerprints",
        "terminal",
        "truncated",
        "bootstrap_values",
        "terminal_rewards",
        "horizons",
        "behavior_policy_versions",
        "behavior_policy_fingerprints",
        "model_config_fingerprints",
        "action_schema_fingerprints",
        "public_context_fingerprints",
        "card_catalog_fingerprints",
        "public_deck_catalog_fingerprints",
        "exact_registry_fingerprints",
        "belief_target_semantics_fingerprints",
        "input_contract_fingerprints",
        "resolved_config_fingerprints",
    )
    if any(len(arrays[field]) != fragments for field in fragment_fields):
        raise ValueError("fragment metadata arrays are misaligned")
    terminal = arrays["terminal"].astype(np.bool_)
    truncated = arrays["truncated"].astype(np.bool_)
    if np.any(terminal == truncated):
        raise ValueError("fragment endpoint flags are invalid")
    lengths = np.diff(arrays["fragment_decision_offsets"].astype(np.int64))
    horizons = arrays["horizons"].astype(np.int64)
    if np.any(lengths <= 0) or np.any(lengths > horizons):
        raise ValueError("fragment decision lengths exceed horizon")
    if np.any(terminal & (arrays["bootstrap_values"] != 0.0)):
        raise ValueError("terminal fragment contains a bootstrap")
    decision_fields = (
        "decision_fragment_indices",
        "action_logprobs",
        "root_values",
        "rewards",
        "stop_sampled",
        "min_counts",
        "max_counts",
        "belief_scalars",
    )
    if any(len(arrays[field]) != decisions for field in decision_fields):
        raise ValueError("decision metadata arrays are misaligned")
    if arrays["belief_scalars"].shape != (decisions, 4):
        raise ValueError("belief scalar rows have the wrong width")
    ragged = (
        ("state", "state_offsets", "state_card_ids"),
        ("attachment", "attachment_offsets", "attachment_card_ids"),
        ("option", "option_offsets", "option_types"),
        ("belief", "belief_offsets", "belief_card_ids"),
        ("known", "known_offsets", "known_card_ids"),
        ("action", "action_offsets", "action_choices"),
        ("token", "token_offsets", "token_logprobs"),
    )
    for name, offsets, values in ragged:
        _validate_offsets(
            arrays[offsets],
            rows=decisions,
            values=len(arrays[values]),
            name=name,
        )
    state_values = len(arrays["state_card_ids"])
    if (
        arrays["state_scalars"].shape != (state_values, TOKEN_SCALAR_SIZE)
        or arrays["state_areas"].shape != (state_values,)
        or arrays["state_owner_roles"].shape != (state_values,)
        or arrays["state_token_kinds"].shape != (state_values,)
        or arrays["state_last_attack_ids"].shape != (state_values,)
        or arrays["state_entity_slots"].shape != (state_values,)
    ):
        raise ValueError("state flattened arrays are misaligned")
    attachments = len(arrays["attachment_card_ids"])
    if (
        len(arrays["attachment_parent_indices"]) != attachments
        or len(arrays["attachment_kinds"]) != attachments
    ):
        raise ValueError("attachment flattened arrays are misaligned")
    option_values = len(arrays["option_types"])
    if (
        len(arrays["option_contexts"]) != option_values
        or arrays["option_entity_slots"].shape != (option_values, MAX_ENTITY_SLOTS)
        or arrays["option_entity_slot_mask"].shape != (option_values, MAX_ENTITY_SLOTS)
        or len(arrays["option_attack_ids"]) != option_values
        or len(arrays["option_card_ids"]) != option_values
        or arrays["option_scalars"].shape != (option_values, SCALAR_FEATURE_SIZE)
        or arrays["option_dynamic_effect_features"].shape
        != (option_values, DYNAMIC_EFFECT_FEATURE_SIZE)
        or len(arrays["option_dynamic_effect_masks"]) != option_values
    ):
        raise ValueError("option flattened arrays are misaligned")
    if len(arrays["belief_expected_counts"]) != len(arrays["belief_card_ids"]):
        raise ValueError("belief sparse arrays are misaligned")
    if len(arrays["known_counts"]) != len(arrays["known_card_ids"]):
        raise ValueError("known-card sparse arrays are misaligned")
    if len(arrays["prefix_values"]) != len(arrays["token_logprobs"]):
        raise ValueError("decode-token arrays are misaligned")
    if schema_version == 2:
        _validate_sequence_arrays(
            arrays,
            fragments=fragments,
            decisions=decisions,
        )
    token_offsets = np.asarray(arrays["token_offsets"], dtype=np.int64)
    if np.any(np.diff(token_offsets) <= 0):
        raise ValueError("stored action log-prob differs from token sum")
    token_logprobs = np.asarray(arrays["token_logprobs"], dtype=np.float64)
    token_sums = np.add.reduceat(token_logprobs, token_offsets[:-1])
    if not np.allclose(
        token_sums,
        np.asarray(arrays["action_logprobs"], dtype=np.float64),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise ValueError("stored action log-prob differs from token sum")


def _validate_sequence_arrays(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    *,
    fragments: int,
    decisions: int,
) -> None:
    """Validate V2 raw temporal payloads without inspecting real config."""
    if (
        arrays["fragment_schema_versions"].shape != (fragments,)
        or np.any(arrays["fragment_schema_versions"] != 2)
        or len(arrays["sequence_contract_fingerprints"]) != fragments
    ):
        raise ValueError("sequence fragment identity arrays are misaligned")
    decision_fields = (
        "engine_fact_producer_fingerprints",
        "sequence_request_ids",
        "accepted_action_stable_ids",
        "accepted_action_prompt_contexts",
        "accepted_action_ordered",
        "accepted_action_fallback",
    )
    if any(len(arrays[field]) != decisions for field in decision_fields):
        raise ValueError("sequence decision identity arrays are misaligned")
    _validate_offsets(
        arrays["event_offsets"],
        rows=decisions,
        values=len(arrays["event_types"]),
        name="public events",
    )
    event_count = len(arrays["event_types"])
    if (
        len(arrays["event_actor_roles"]) != event_count
        or len(arrays["event_from_areas"]) != event_count
        or len(arrays["event_to_areas"]) != event_count
        or arrays["event_card_ids"].shape
        != (event_count, PUBLIC_EVENT_ENTITY_COUNT)
        or arrays["event_serials"].shape
        != (event_count, PUBLIC_EVENT_ENTITY_COUNT)
        or arrays["event_entity_mask"].shape
        != (event_count, PUBLIC_EVENT_ENTITY_COUNT)
        or len(arrays["event_attack_ids"]) != event_count
        or len(arrays["event_attack_id_mask"]) != event_count
        or len(arrays["event_values"]) != event_count
        or len(arrays["event_value_mask"]) != event_count
        or arrays["event_categorical_values"].shape
        != (event_count, PUBLIC_EVENT_CATEGORICAL_SIZE)
    ):
        raise ValueError("sequence public-event arrays are misaligned")
    _validate_offsets(
        arrays["event_overflow_offsets"],
        rows=decisions,
        values=len(arrays["event_overflow_types"]),
        name="public event overflow",
    )
    overflow_count = len(arrays["event_overflow_types"])
    if (
        len(arrays["event_overflow_actor_roles"]) != overflow_count
        or len(arrays["event_overflow_counts"]) != overflow_count
    ):
        raise ValueError("sequence public-event overflow arrays are misaligned")
    selected = len(arrays["action_choices"])
    selected_fields = (
        "accepted_action_option_types",
        "accepted_action_option_contexts",
        "accepted_action_card_ids",
        "accepted_action_attack_ids",
    )
    if any(len(arrays[field]) != selected for field in selected_fields):
        raise ValueError("accepted-action categorical arrays are misaligned")
    if (
        arrays["accepted_action_option_scalars"].shape
        != (selected, SCALAR_FEATURE_SIZE)
        or arrays["accepted_action_entity_card_ids"].shape
        != (selected, MAX_ENTITY_SLOTS)
        or arrays["accepted_action_entity_areas"].shape
        != (selected, MAX_ENTITY_SLOTS)
        or arrays["accepted_action_entity_owner_roles"].shape
        != (selected, MAX_ENTITY_SLOTS)
        or arrays["accepted_action_entity_token_kinds"].shape
        != (selected, MAX_ENTITY_SLOTS)
        or arrays["accepted_action_entity_scalars"].shape
        != (selected, MAX_ENTITY_SLOTS, TOKEN_SCALAR_SIZE)
    ):
        raise ValueError("accepted-action semantic arrays are misaligned")


def _validate_offsets(
    offsets: npt.NDArray[np.generic],
    *,
    rows: int,
    values: int,
    name: str,
) -> None:
    normalized = np.asarray(offsets, dtype=np.int64)
    if (
        normalized.shape != (rows + 1,)
        or normalized[0] != 0
        or normalized[-1] != values
        or np.any(normalized[1:] < normalized[:-1])
    ):
        raise ValueError(f"{name} offsets are invalid")


def _offsets(lengths: Sequence[int]) -> npt.NDArray[np.int64]:
    result = np.zeros(len(lengths) + 1, dtype=np.int64)
    result[1:] = np.cumsum(np.asarray(lengths, dtype=np.int64))
    return result


def _concatenate(
    values: Sequence[npt.ArrayLike],
    *,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.generic]:
    if not values:
        return np.asarray([], dtype=dtype)
    return np.concatenate([np.asarray(value, dtype=dtype) for value in values])


def _concatenate_rows(
    values: Sequence[npt.ArrayLike],
    *,
    width: int,
    dtype: npt.DTypeLike,
) -> npt.NDArray[np.generic]:
    if not values:
        return np.empty((0, width), dtype=dtype)
    result = np.concatenate(
        [np.asarray(value, dtype=dtype) for value in values],
        axis=0,
    )
    if result.shape != (result.shape[0], width):
        raise ValueError("flattened row arrays have the wrong width")
    return result


def _strings(values: Sequence[str]) -> npt.NDArray[np.str_]:
    width = max(1, *(len(value) for value in values))
    return np.asarray(values, dtype=f"U{width}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file_atomic(source: Path, destination: Path) -> None:
    """Atomically publish across filesystems without partial final visibility."""
    if source.stat().st_dev == destination.parent.stat().st_dev:
        os.replace(source, destination)
        return
    pending = destination.parent / (f".{destination.name}.pending-{uuid.uuid4().hex}")
    try:
        with source.open("rb") as source_handle, pending.open("xb") as target:
            shutil.copyfileobj(source_handle, target, length=1024 * 1024)
            target.flush()
            os.fsync(target.fileno())
        if pending.stat().st_size != source.stat().st_size or _file_sha256(
            pending
        ) != _file_sha256(source):
            raise RuntimeError("cross-filesystem fragment staging changed bytes")
        os.replace(pending, destination)
    finally:
        pending.unlink(missing_ok=True)


__all__ = [
    "SEQUENCE_FRAGMENT_ARRAY_SCHEMA",
    "SEQUENCE_FRAGMENT_ARRAY_KEYS",
    "STATELESS_FRAGMENT_ARRAY_SCHEMA",
    "STATELESS_FRAGMENT_ARRAY_KEYS",
    "CompactFragmentManifest",
    "CompactFragmentPart",
    "CompactFragmentShardWriter",
    "FragmentPartRecord",
    "load_compact_fragment_manifest",
    "load_compact_fragment_part",
]
