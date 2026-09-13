"""Typed compact replay streams for roots, engine evidence, and Q targets."""

from __future__ import annotations

import pickle
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import msgpack
import numpy as np
import numpy.typing as npt

from ptcg_rl.actions.encoding import (
    ENTITY_SLOT_FEATURE_SIZE,
    EncodedOptionArrayFeatures,
)
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.model.state_encoder import StateTokenArrayFeatures
from ptcg_rl.rl.amortized_policy_iteration.belief_reanalysis import (
    LeafActorInput,
    NativeReanalysisJob,
    NativeReanalysisResult,
    ReanalysisRoot,
    StudentReanalysisRoot,
)
from ptcg_rl.rl.amortized_policy_iteration.counterfactual import (
    CounterfactualRootTarget,
)
from ptcg_rl.rl.amortized_policy_iteration.replay import (
    AtomicReplayShardWriter,
    BoundedAsyncReplayShardWriter,
    ReplayIdentity,
    ReplayShardKind,
)

_PACKED_RECORD_SCHEMA = 1
_MAX_BUFFER_BYTES = 64 << 20


@dataclass(frozen=True, slots=True)
class PolicyIterationReplaySummary:
    """Committed and currently buffered logical rows by evidence kind."""

    committed_rows: dict[str, int]
    buffered_rows: dict[str, int]
    inflight_rows: dict[str, int]
    last_write_seconds: dict[str, float]


class _PackedReplayStream:
    """Buffer MessagePack records and publish fixed-row compact NPZ parts."""

    def __init__(
        self,
        directory: Path,
        *,
        kind: ReplayShardKind,
        identity: ReplayIdentity,
        rows_per_shard: int,
        compress: bool,
        max_committed_shards: int,
        async_writes: bool = False,
    ) -> None:
        writer = AtomicReplayShardWriter(
            directory,
            kind=kind,
            identity=identity,
            compress=compress,
            max_committed_shards=max_committed_shards,
        )
        self._writer: AtomicReplayShardWriter | BoundedAsyncReplayShardWriter = (
            BoundedAsyncReplayShardWriter(writer) if async_writes else writer
        )
        self._rows_per_shard = rows_per_shard
        self._buffer: list[bytes] = []
        self._buffer_bytes = 0

    @property
    def committed_rows(self) -> int:
        return self._writer.committed_rows

    @property
    def buffered_rows(self) -> int:
        return len(self._buffer)

    @property
    def inflight_rows(self) -> int:
        if isinstance(self._writer, BoundedAsyncReplayShardWriter):
            return self._writer.pending_rows
        return 0

    @property
    def last_write_seconds(self) -> float:
        timing = self._writer.last_timing
        return 0.0 if timing is None else timing.total_seconds

    def append(self, records: Sequence[Mapping[str, Any]]) -> None:
        for record in records:
            packed = msgpack.packb(
                {
                    "record_schema": _PACKED_RECORD_SCHEMA,
                    **dict(record),
                },
                use_bin_type=True,
            )
            if len(packed) > _MAX_BUFFER_BYTES:
                raise ValueError("one packed replay record exceeds the memory cap")
            if self._buffer and self._buffer_bytes + len(packed) > _MAX_BUFFER_BYTES:
                self._flush_prefix(len(self._buffer))
            self._buffer.append(packed)
            self._buffer_bytes += len(packed)
            if len(self._buffer) >= self._rows_per_shard:
                self._flush_prefix(self._rows_per_shard)

    def flush(self) -> None:
        """Submit buffered rows and wait until they are manifest-visible."""
        if self._buffer:
            self._flush_prefix(len(self._buffer))
        self.barrier()

    def flush_async(self) -> None:
        """Submit buffered rows without waiting for durable publication."""
        if self._buffer:
            self._flush_prefix(len(self._buffer))

    def barrier(self) -> None:
        """Wait for an in-flight replay part, if asynchronous writes are enabled."""
        if isinstance(self._writer, BoundedAsyncReplayShardWriter):
            self._writer.barrier()

    def close(self) -> None:
        """Commit buffered rows and stop an optional replay writer thread."""
        self.flush()
        if isinstance(self._writer, BoundedAsyncReplayShardWriter):
            self._writer.close()

    def iter_records(self) -> Iterator[Mapping[str, Any]]:
        for part in self._writer.iter_parts():
            payloads = cast(np.ndarray, part["record_payload"])
            offsets = part.get("record_offsets")
            if offsets is None:
                sizes = cast(np.ndarray, part["record_size"])
                records = (
                    bytes(payload)[: int(size)]
                    for payload, size in zip(payloads, sizes, strict=True)
                )
            else:
                typed_offsets = cast(np.ndarray, offsets)
                records = (
                    bytes(payloads[int(start) : int(end)])
                    for start, end in zip(
                        typed_offsets[:-1],
                        typed_offsets[1:],
                        strict=True,
                    )
                )
            for raw in records:
                record = msgpack.unpackb(raw, raw=False, strict_map_key=False)
                if not isinstance(record, dict) or record.get("record_schema") != (
                    _PACKED_RECORD_SCHEMA
                ):
                    raise RuntimeError("replay packed record schema is invalid")
                yield record

    def _flush_prefix(self, count: int) -> None:
        rows = self._buffer[:count]
        joined = b"".join(rows)
        payload = np.frombuffer(joined, dtype=np.uint8)
        offsets = np.zeros(len(rows) + 1, dtype=np.uint64)
        np.cumsum(
            np.asarray([len(row) for row in rows], dtype=np.uint64),
            out=offsets[1:],
        )
        columns: dict[str, npt.NDArray[np.generic]] = {
            "record_payload": payload,
            "record_offsets": offsets,
        }
        if isinstance(self._writer, BoundedAsyncReplayShardWriter):
            self._writer.append(columns, copy=False)
        else:
            self._writer.append(columns)
        del self._buffer[:count]
        self._buffer_bytes = sum(len(row) for row in self._buffer)


class PolicyIterationReplayStore:
    """Persist separately typed policy-iteration evidence and recover targets."""

    _KINDS: tuple[ReplayShardKind, ...] = (
        "root",
        "protected_root",
        "candidate",
        "consequence",
        "target",
    )

    def __init__(
        self,
        directory: Path,
        *,
        identity: ReplayIdentity,
        rows_per_shard: int,
        compress: bool,
        max_committed_shards: int,
        async_writes: bool = False,
    ) -> None:
        self._streams = {
            kind: _PackedReplayStream(
                directory / kind,
                kind=kind,
                identity=identity,
                rows_per_shard=rows_per_shard,
                compress=compress,
                max_committed_shards=max_committed_shards,
                async_writes=async_writes,
            )
            for kind in self._KINDS
        }

    def record_roots(self, roots: Sequence[ReanalysisRoot]) -> None:
        """Archive student and engine-only root material in disjoint streams."""
        self._streams["root"].append(
            tuple(_student_root_record(root.student) for root in roots)
        )
        self._streams["protected_root"].append(
            tuple(
                {
                    "root_id": root.student.root_id,
                    "observation_json": root.protected.observation_json,
                    "context_snapshots_pickle": pickle.dumps(
                        root.protected.context_snapshots,
                        protocol=5,
                    ),
                    "deck_pair": [
                        list(deck.card_ids) for deck in root.protected.deck_pair
                    ],
                }
                for root in roots
            )
        )

    def record_jobs(self, jobs: Sequence[NativeReanalysisJob]) -> None:
        """Persist exact proposal probabilities and completeness markers."""
        records = []
        for job in jobs:
            for candidate_index, candidate in enumerate(job.proposal.candidates):
                records.append(
                    {
                        "root_id": job.root.student.root_id,
                        "candidate_index": candidate_index,
                        "action": list(candidate.action),
                        "q_prop": candidate.q_prop,
                        "old_policy_probability": candidate.behavior_probability,
                        "sources": list(candidate.sources),
                        "legal_action_count": job.proposal.legal_action_count,
                        "exhaustive": job.proposal.exhaustive,
                        "proposal_policy_version": job.proposal_policy_version,
                        "producer_contract_fingerprint": (
                            job.producer_contract_fingerprint
                        ),
                        "stochastic_seed": job.stochastic_seed,
                    }
                )
        self._streams["candidate"].append(records)

    def record_results(self, results: Sequence[NativeReanalysisResult]) -> None:
        """Persist per-world engine evidence only in the protected consequence lane."""
        records = []
        for result in results:
            if result.error_message:
                records.append(
                    {
                        "root_id": result.root.student.root_id,
                        "error_message": result.error_message,
                        "engine_library_fingerprint": (
                            result.engine_library_fingerprint
                        ),
                        "native_abi_fingerprint": result.native_abi_fingerprint,
                        "native_seconds": result.native_seconds,
                    }
                )
                continue
            for cell in result.cells:
                records.append(
                    {
                        "root_id": result.root.student.root_id,
                        "candidate_id": cell.candidate_id,
                        "particle_id": cell.particle_id,
                        "candidate_index": cell.candidate_index,
                        "world_index": cell.world_index,
                        "sampling_weight": cell.sampling_weight,
                        "endpoint": int(cell.endpoint),
                        "error": cell.error,
                        "root_player": cell.root_player,
                        "leaf_player": cell.leaf_player,
                        "engine_result": cell.engine_result,
                        "transition_steps": cell.transition_steps,
                        "forced_steps": cell.forced_steps,
                        "leaf": (
                            None if cell.leaf is None else _leaf_record(cell.leaf)
                        ),
                        "engine_library_fingerprint": (
                            result.engine_library_fingerprint
                        ),
                        "native_abi_fingerprint": result.native_abi_fingerprint,
                        "native_seconds": result.native_seconds,
                    }
                )
        self._streams["consequence"].append(records)

    def record_targets(self, targets: Sequence[CounterfactualRootTarget]) -> None:
        """Persist aggregated targets after all particle identity has been removed."""
        self._streams["target"].append(
            tuple(
                {
                    "root": _student_root_record(target.root),
                    "actions": [list(action) for action in target.actions],
                    "target_wdl": [list(wdl) for wdl in target.target_wdl],
                    "proposal_probabilities": list(target.proposal_probabilities),
                    "old_policy_probabilities": list(target.old_policy_probabilities),
                    "exhaustive": target.exhaustive,
                    "proposal_policy_version": target.proposal_policy_version,
                    "valid_worlds": target.valid_worlds,
                    "omitted_worlds": target.omitted_worlds,
                }
                for target in targets
            )
        )

    def recover_targets(self, *, limit: int) -> tuple[CounterfactualRootTarget, ...]:
        """Load verified retained student targets without opening protected streams."""
        if limit <= 0:
            raise ValueError("replay target recovery limit must be positive")
        recovered: deque[CounterfactualRootTarget] = deque(maxlen=limit)
        for record in self._streams["target"].iter_records():
            recovered.append(_decode_target_record(record))
        return tuple(recovered)

    def flush(self) -> None:
        """Commit every complete in-memory evidence row."""
        for stream in self._streams.values():
            stream.flush()

    def flush_async(self) -> None:
        """Submit buffered evidence while allowing one part per stream in flight."""
        for stream in self._streams.values():
            stream.flush_async()

    def barrier(self) -> None:
        """Wait until every submitted replay part is manifest-visible."""
        for stream in self._streams.values():
            stream.barrier()

    def close(self) -> None:
        """Commit all evidence and cleanly stop optional background writers."""
        failures: list[Exception] = []
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            if len(failures) == 1:
                raise failures[0]
            raise ExceptionGroup("multiple replay streams failed to close", failures)

    def summary(self) -> PolicyIterationReplaySummary:
        """Return exact committed and volatile row counts."""
        return PolicyIterationReplaySummary(
            committed_rows={
                kind: stream.committed_rows for kind, stream in self._streams.items()
            },
            buffered_rows={
                kind: stream.buffered_rows for kind, stream in self._streams.items()
            },
            inflight_rows={
                kind: stream.inflight_rows for kind, stream in self._streams.items()
            },
            last_write_seconds={
                kind: stream.last_write_seconds
                for kind, stream in self._streams.items()
            },
        )


def _student_root_record(root: StudentReanalysisRoot) -> dict[str, Any]:
    return {
        "root_id": root.root_id,
        "game_id": root.game_id,
        "seat": root.seat,
        "decision_index": root.decision_index,
        "state": _state_record(root.state),
        "options": _option_record(root.options),
        "min_count": root.min_count,
        "max_count": root.max_count,
        "deck": list(root.deck.card_ids),
        "behavior_kind": root.behavior_kind,
        "behavior_action": list(root.behavior_action),
        "behavior_probability": root.behavior_probability,
        "sampling_temperature": root.sampling_temperature,
        "policy_version": root.policy_version,
    }


def _state_record(state: Any) -> dict[str, Any]:
    return {
        name: _array_record(np.asarray(getattr(state, name)))
        for name in (
            "card_ids",
            "areas",
            "owner_roles",
            "token_kinds",
            "scalars",
            "last_attack_ids",
            "attachment_card_ids",
            "attachment_parent_indices",
            "attachment_kinds",
            "entity_slots",
        )
    }


def _option_record(options: Any) -> dict[str, Any]:
    if isinstance(options, EncodedOptionArrayFeatures):
        arrays = options
    else:
        tuple_options = tuple(options)
        entity_slots = np.zeros(
            (len(tuple_options), ENTITY_SLOT_FEATURE_SIZE), dtype=np.int64
        )
        entity_mask = np.zeros(
            (len(tuple_options), ENTITY_SLOT_FEATURE_SIZE), dtype=np.bool_
        )
        for row, option in enumerate(tuple_options):
            entity_slots[row, : len(option.entity_slots)] = option.entity_slots
            entity_mask[row, : len(option.entity_slots)] = True
        arrays = EncodedOptionArrayFeatures(
            option_types=np.asarray(
                [option.option_type for option in tuple_options], dtype=np.int64
            ),
            contexts=np.asarray(
                [option.context for option in tuple_options], dtype=np.int64
            ),
            entity_slots=entity_slots,
            entity_slot_mask=entity_mask,
            attack_ids=np.asarray(
                [option.attack_id for option in tuple_options], dtype=np.int64
            ),
            card_ids=np.asarray(
                [option.card_id for option in tuple_options], dtype=np.int64
            ),
            scalars=np.asarray(
                [option.scalars for option in tuple_options], dtype=np.float32
            ),
            dynamic_effect_features=np.asarray(
                [option.dynamic_effect_features for option in tuple_options],
                dtype=np.float32,
            ),
            dynamic_effect_masks=np.asarray(
                [option.dynamic_effect_mask for option in tuple_options],
                dtype=np.bool_,
            ),
        )
    return {
        name: _array_record(np.asarray(getattr(arrays, name)))
        for name in (
            "option_types",
            "contexts",
            "entity_slots",
            "entity_slot_mask",
            "attack_ids",
            "card_ids",
            "scalars",
            "dynamic_effect_features",
            "dynamic_effect_masks",
        )
    }


def _leaf_record(leaf: LeafActorInput) -> dict[str, Any]:
    return {
        "state": _state_record(leaf.state),
        "options": _option_record(leaf.options),
        "min_count": leaf.min_count,
        "max_count": leaf.max_count,
        "deck": list(leaf.deck.card_ids),
        "observation_json": leaf.observation_json,
    }


def _array_record(values: np.ndarray) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(values)
    return {
        "dtype": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "data": contiguous.tobytes(),
    }


def _decode_array(record: Mapping[str, Any]) -> np.ndarray:
    dtype = np.dtype(str(record["dtype"]))
    shape = tuple(int(value) for value in record["shape"])
    return np.frombuffer(bytes(record["data"]), dtype=dtype).reshape(shape).copy()


def _decode_student_root(record: Mapping[str, Any]) -> StudentReanalysisRoot:
    state_raw = cast(Mapping[str, Any], record["state"])
    option_raw = cast(Mapping[str, Any], record["options"])
    state = StateTokenArrayFeatures(
        **{
            name: _decode_array(cast(Mapping[str, Any], value))
            for name, value in state_raw.items()
        },
        layout=None,
    )
    options = EncodedOptionArrayFeatures(
        **{
            name: _decode_array(cast(Mapping[str, Any], value))
            for name, value in option_raw.items()
        }
    )
    return StudentReanalysisRoot(
        root_id=str(record["root_id"]),
        game_id=str(record["game_id"]),
        seat=int(record["seat"]),
        decision_index=int(record["decision_index"]),
        state=state,
        options=options,
        min_count=int(record["min_count"]),
        max_count=int(record["max_count"]),
        deck=canonicalize_deck(record["deck"]),
        behavior_kind=cast(Any, str(record["behavior_kind"])),
        behavior_action=tuple(int(value) for value in record["behavior_action"]),
        behavior_probability=float(record["behavior_probability"]),
        sampling_temperature=float(record["sampling_temperature"]),
        policy_version=int(record["policy_version"]),
    )


def _decode_target_record(record: Mapping[str, Any]) -> CounterfactualRootTarget:
    return CounterfactualRootTarget(
        root=_decode_student_root(cast(Mapping[str, Any], record["root"])),
        actions=tuple(
            tuple(int(value) for value in action) for action in record["actions"]
        ),
        target_wdl=tuple(
            (float(row[0]), float(row[1]), float(row[2]))
            for row in record["target_wdl"]
        ),
        proposal_probabilities=tuple(
            float(value) for value in record["proposal_probabilities"]
        ),
        old_policy_probabilities=tuple(
            float(value) for value in record["old_policy_probabilities"]
        ),
        exhaustive=bool(record["exhaustive"]),
        proposal_policy_version=int(record["proposal_policy_version"]),
        valid_worlds=int(record["valid_worlds"]),
        omitted_worlds=int(record["omitted_worlds"]),
    )


__all__ = ["PolicyIterationReplayStore", "PolicyIterationReplaySummary"]
