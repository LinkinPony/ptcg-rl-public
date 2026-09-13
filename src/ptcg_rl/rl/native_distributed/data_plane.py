"""Attempt-bound envelope around the compact native fragment codec."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, cast

import msgpack
import numpy as np

from ptcg_rl.rl.native_distributed.codec import (
    DecodedNativeRolloutPart,
    decode_compact_fragment_part,
    encode_compact_fragment_part,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeCollectionAttempt,
    NativeCollectionPartIdentity,
    NativeCollectionShardLease,
    NativeCollectionWorkerManifest,
    NativeRolloutLeaseIdentity,
    NativeRolloutPartIdentity,
)
from ptcg_rl.rl.native_distributed.control import NativeRolloutProtocolError
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart

_SCHEMA = "ptcg-rl/native-collection-part/v1"
_HEADER_FIELDS = (
    "schema",
    "part",
    "payload_header_sha256",
)
_MAX_HEADER_BYTES = 1 << 16


@dataclass(frozen=True, slots=True)
class DecodedNativeCollectionPart:
    """Attempt-bound compact part retaining every borrowed ndarray frame."""

    identity: NativeCollectionPartIdentity
    payload: DecodedNativeRolloutPart
    _wire_bytes: int

    @property
    def part(self) -> CompactFragmentPart:
        """Return the verified compact fragment arrays."""
        return self.payload.part

    @property
    def payload_bytes(self) -> int:
        """Return raw ndarray payload bytes."""
        return self.payload.payload_bytes

    @property
    def wire_bytes(self) -> int:
        """Return envelope, compact header, and raw ndarray wire bytes."""
        return self._wire_bytes


def collection_part_id(
    attempt: NativeCollectionAttempt,
    *,
    shard_sequence_id: int,
    part_sequence_id: int,
) -> str:
    """Derive one stable part ID from its exact attempt coordinates."""
    material = (
        f"{attempt.attempt_id}:{attempt.lease_id}:{shard_sequence_id}:"
        f"{part_sequence_id}"
    ).encode()
    return hashlib.sha256(material).hexdigest()


def encode_native_collection_part(
    lease: NativeCollectionShardLease,
    attempt: NativeCollectionAttempt,
    worker: NativeCollectionWorkerManifest,
    part: CompactFragmentPart,
    *,
    part_sequence_id: int,
) -> tuple[bytes, bytes, tuple[memoryview, ...]]:
    """Encode one part with immutable shard and attempt identity."""
    if attempt.lease_id != lease.lease_id:
        raise ValueError("native collection attempt crossed its shard lease")
    if (
        attempt.worker_id != worker.identity.worker_id
        or attempt.worker_session_id != worker.identity.session_id
    ):
        raise ValueError("native collection attempt crossed its worker session")
    part_id = collection_part_id(
        attempt,
        shard_sequence_id=lease.shard_sequence_id,
        part_sequence_id=part_sequence_id,
    )
    identity = NativeCollectionPartIdentity(
        part_id=part_id,
        lease_id=lease.lease_id,
        attempt_id=attempt.attempt_id,
        shard_sequence_id=lease.shard_sequence_id,
        part_sequence_id=part_sequence_id,
        fragment_count=part.fragment_count,
        decision_count=part.decision_count,
    )
    legacy_lease = NativeRolloutLeaseIdentity(
        lease_id=lease.lease_id,
        worker=worker.identity,
        window=lease.window.identity,
        sequence_id=lease.shard_sequence_id,
        issued_at_unix_ns=lease.issued_at_unix_ns,
        expires_at_unix_ns=max(
            lease.issued_at_unix_ns + 1,
            attempt.expires_at_unix_ns,
        ),
    )
    legacy_identity = NativeRolloutPartIdentity(
        part_id=part_id,
        lease=legacy_lease,
        sequence_id=part_sequence_id,
        fragment_count=part.fragment_count,
        decision_count=part.decision_count,
    )
    payload_header, frames = encode_compact_fragment_part(legacy_identity, part)
    envelope = cast(
        bytes,
        msgpack.packb(
            {
                "schema": _SCHEMA,
                "part": identity.model_dump(mode="python"),
                "payload_header_sha256": hashlib.sha256(payload_header).hexdigest(),
            },
            use_bin_type=True,
        ),
    )
    return envelope, payload_header, frames


def decode_native_collection_part(
    envelope: Any,
    payload_header: Any,
    frames: list[Any] | tuple[Any, ...],
    *,
    expected_lease: NativeCollectionShardLease,
    expected_attempt: NativeCollectionAttempt,
    expected_worker: NativeCollectionWorkerManifest,
) -> DecodedNativeCollectionPart:
    """Decode and cross-check the outer attempt and inner ndarray identities."""
    envelope_view = _byte_view(envelope, name="envelope")
    if envelope_view.nbytes > _MAX_HEADER_BYTES:
        raise NativeRolloutProtocolError(
            "native collection part envelope exceeds the size limit"
        )
    try:
        raw = msgpack.unpackb(
            bytes(envelope_view),
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_strict_map,
            use_list=False,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native collection part envelope is invalid msgpack"
        ) from exc
    if not isinstance(raw, dict) or tuple(raw) != _HEADER_FIELDS:
        raise NativeRolloutProtocolError(
            "native collection part envelope fields are invalid"
        )
    values = cast(dict[str, object], raw)
    if values["schema"] != _SCHEMA:
        raise NativeRolloutProtocolError(
            "native collection part envelope schema is unsupported"
        )
    try:
        identity = NativeCollectionPartIdentity.model_validate(values["part"])
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native collection part identity is invalid"
        ) from exc
    payload_header_view = _byte_view(payload_header, name="payload header")
    claimed_hash = values["payload_header_sha256"]
    if (
        not isinstance(claimed_hash, str)
        or hashlib.sha256(payload_header_view).hexdigest() != claimed_hash
    ):
        raise NativeRolloutProtocolError(
            "native collection part payload header fingerprint differs"
        )
    if (
        identity.lease_id != expected_lease.lease_id
        or identity.attempt_id != expected_attempt.attempt_id
        or identity.shard_sequence_id != expected_lease.shard_sequence_id
        or expected_attempt.lease_id != expected_lease.lease_id
        or expected_attempt.worker_id != expected_worker.identity.worker_id
        or expected_attempt.worker_session_id != expected_worker.identity.session_id
    ):
        raise NativeRolloutProtocolError(
            "native collection part lease/attempt/worker identity differs"
        )
    expected_part_id = collection_part_id(
        expected_attempt,
        shard_sequence_id=expected_lease.shard_sequence_id,
        part_sequence_id=identity.part_sequence_id,
    )
    if identity.part_id != expected_part_id:
        raise NativeRolloutProtocolError("native collection part ID differs")
    decoded = decode_compact_fragment_part(payload_header_view, frames)
    _validate_window_semantics(
        decoded.part,
        expected_lease=expected_lease,
        expected_worker=expected_worker,
    )
    legacy = decoded.identity
    if (
        legacy.part_id != identity.part_id
        or legacy.lease.lease_id != identity.lease_id
        or legacy.lease.worker != expected_worker.identity
        or legacy.lease.window != expected_lease.window.identity
        or legacy.sequence_id != identity.part_sequence_id
        or legacy.fragment_count != identity.fragment_count
        or legacy.decision_count != identity.decision_count
    ):
        raise NativeRolloutProtocolError(
            "native collection outer and compact-part identities differ"
        )
    return DecodedNativeCollectionPart(
        identity=identity,
        payload=decoded,
        _wire_bytes=(
            envelope_view.nbytes + payload_header_view.nbytes + decoded.payload_bytes
        ),
    )


def _validate_window_semantics(
    part: CompactFragmentPart,
    *,
    expected_lease: NativeCollectionShardLease,
    expected_worker: NativeCollectionWorkerManifest,
) -> None:
    """Reject payload columns that cross the lease's immutable contracts."""
    arrays = part.arrays
    window = expected_lease.window.identity
    expected_strings = {
        "behavior_policy_fingerprints": window.behavior_policy_fingerprint,
        "resolved_config_fingerprints": window.resolved_config_fingerprint,
        "model_config_fingerprints": (
            expected_worker.identity.model_config_fingerprint
        ),
        "card_catalog_fingerprints": (
            expected_worker.identity.card_catalog_fingerprint
        ),
        "exact_registry_fingerprints": (
            expected_worker.identity.exact_registry_fingerprint
        ),
    }
    for field, expected_string in expected_strings.items():
        values = np.asarray(arrays[field])
        if values.size == 0 or np.any(values != expected_string):
            raise NativeRolloutProtocolError(
                f"native collection part {field} crossed its window"
            )
    assignments = {
        item.curriculum.assignment_id: item for item in expected_lease.assignments
    }
    assignment_ids = np.asarray(arrays["assignment_ids"])
    seats = np.asarray(arrays["seats"])
    for row, assignment_id_value in enumerate(assignment_ids):
        assignment_id = str(assignment_id_value)
        try:
            assignment = assignments[assignment_id]
        except KeyError as exc:
            raise NativeRolloutProtocolError(
                "native collection part crossed its assignment lease"
            ) from exc
        curriculum = assignment.curriculum
        seat = int(seats[row])
        if seat == curriculum.candidate_seat:
            own_deck_digest = curriculum.candidate_deck_digest
            opponent_deck_digest = curriculum.opponent_deck_digest
        elif curriculum.lane == "mirror":
            own_deck_digest = curriculum.opponent_deck_digest
            opponent_deck_digest = curriculum.candidate_deck_digest
        else:
            raise NativeRolloutProtocolError(
                "native collection part crossed assignment candidate seat"
            )
        expected_values = {
            "curriculum_generations": curriculum.generation,
            "own_deck_digests": own_deck_digest,
            "opponent_deck_digests": opponent_deck_digest,
            "opponent_artifact_fingerprints": (
                curriculum.opponent_artifact_fingerprint
            ),
        }
        for field, expected_value in expected_values.items():
            if np.asarray(arrays[field])[row] != expected_value:
                raise NativeRolloutProtocolError(
                    f"native collection part crossed assignment field {field}"
                )
    behavior_versions = np.asarray(arrays["behavior_policy_versions"])
    if behavior_versions.size == 0 or np.any(
        behavior_versions != window.behavior_policy_version
    ):
        raise NativeRolloutProtocolError(
            "native collection part behavior version crossed its window"
        )


def peek_native_collection_part_identity(
    envelope: Any,
) -> NativeCollectionPartIdentity:
    """Read only the bounded outer identity before resolving attempt context."""
    envelope_view = _byte_view(envelope, name="envelope")
    if envelope_view.nbytes > _MAX_HEADER_BYTES:
        raise NativeRolloutProtocolError(
            "native collection part envelope exceeds the size limit"
        )
    try:
        raw = msgpack.unpackb(
            bytes(envelope_view),
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_strict_map,
            use_list=False,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native collection part envelope is invalid msgpack"
        ) from exc
    if (
        not isinstance(raw, dict)
        or tuple(raw) != _HEADER_FIELDS
        or raw.get("schema") != _SCHEMA
    ):
        raise NativeRolloutProtocolError(
            "native collection part envelope fields are invalid"
        )
    try:
        return NativeCollectionPartIdentity.model_validate(raw["part"])
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native collection part identity is invalid"
        ) from exc


def _strict_map(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate msgpack map key")
        result[key] = value
    return result


def _byte_view(value: Any, *, name: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise NativeRolloutProtocolError(
            f"native collection part {name} is not bytes-like"
        ) from exc
    if not view.c_contiguous:
        raise NativeRolloutProtocolError(
            f"native collection part {name} is not contiguous"
        )
    try:
        return view.cast("B")
    except TypeError as exc:
        raise NativeRolloutProtocolError(
            f"native collection part {name} cannot be byte-cast"
        ) from exc


__all__ = [
    "DecodedNativeCollectionPart",
    "collection_part_id",
    "decode_native_collection_part",
    "encode_native_collection_part",
    "peek_native_collection_part_identity",
]
