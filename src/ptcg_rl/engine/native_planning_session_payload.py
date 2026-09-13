"""Validated zero-copy views for lane-local hierarchical planning sessions."""

from __future__ import annotations

import hashlib
import json
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import orjson

from ptcg_rl.engine.constants import SelectContext

NATIVE_PLANNING_SESSION_MAGIC = 0x31534745
NATIVE_PLANNING_SESSION_PAYLOAD_VERSION = 5
NATIVE_PLANNING_SESSION_METADATA_WIDTH = 14
NATIVE_PLANNING_SESSION_MAX_ROWS = 1 << 16
NATIVE_PLANNING_SESSION_MAX_STATE_SLOTS = 1 << 16
NATIVE_PLANNING_SESSION_MAX_ENGINE_STEPS = 1 << 24
NATIVE_PLANNING_SESSION_MAX_FORCED_STEPS = 64
NATIVE_PLANNING_SESSION_MAX_OBSERVATION_BYTES = 1 << 30
NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES = 32
NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR = (
    "cg-planner-session/v5;row_order=request_aligned;"
    "header=<8i+raw_sha256+producer_sha256;"
    "metadata=<14i:error,rules_exact,endpoint,root_player,leaf_player,"
    "leaf_context,transition_steps,forced_steps,observation_offset,"
    "observation_size,result,leaf_select_type,session_generation,state_slot;"
    "observation=root_visible_select_logs_current_json_v1;"
    "handle=lane_local_generation_slot"
)

_ABI_FINGERPRINT_DOMAIN = b"ptcg-rl/native-planning-session/abi/v1\x00"
_SCHEMA_FINGERPRINT_DOMAIN = b"ptcg-rl/native-planning-session/schema/v1\x00"
_HEADER_WIDTH = 8
_INT32_BYTES = 4
_INT32_MAX = (1 << 31) - 1
_LITTLE_ENDIAN_INT32 = np.dtype("<i4")


class NativePlanningSessionPayloadError(RuntimeError):
    """Raised when a session payload violates its hidden-state boundary."""


class NativePlanningSessionRequestKind(IntEnum):
    """Stable native request kind carried by every response."""

    OPEN = 0
    CONTINUE = 1


class NativePlanningSessionEndpoint(IntEnum):
    """Semantic endpoint reached by one session transition."""

    INVALID = 0
    TERMINAL = 1
    SAME_SEAT_MAIN = 2
    TURN_HANDOFF = 3
    ROOT_STRATEGIC_PROMPT = 4
    CHANCE_PROMPT = 5


class NativePlanningSessionMetadataColumn(IntEnum):
    """Column indices in the fixed-width session metadata matrix."""

    ERROR = 0
    RULES_EXACT = 1
    ENDPOINT = 2
    ROOT_PLAYER = 3
    LEAF_PLAYER = 4
    LEAF_CONTEXT = 5
    TRANSITION_STEPS = 6
    FORCED_STEPS = 7
    OBSERVATION_OFFSET = 8
    OBSERVATION_SIZE = 9
    RESULT = 10
    LEAF_SELECT_TYPE = 11
    SESSION_GENERATION = 12
    STATE_SLOT = 13


@dataclass(frozen=True, slots=True)
class NativePlanningSessionHandle:
    """Opaque lane-local continuation identity, never a wire artifact."""

    generation: int
    state_slot: int

    def __post_init__(self) -> None:
        """Reject sentinel and non-int32 handle components."""
        generation = _positive_int32(self.generation, "session handle generation")
        state_slot = _nonnegative_int32(
            self.state_slot,
            "session handle state_slot",
        )
        object.__setattr__(self, "generation", generation)
        object.__setattr__(self, "state_slot", state_slot)


@dataclass(frozen=True, eq=False)
class NativePlanningSessionPayload:
    """Validated request-aligned views over one immutable session response."""

    request_kind: NativePlanningSessionRequestKind
    generation: int
    metadata: npt.NDArray[np.int32]
    observation_blob: npt.NDArray[np.uint8]
    raw_request_fingerprint: str
    producer_contract_fingerprint: str
    payload_bytes: int
    _storage: bytes = field(repr=False)

    @property
    def row_count(self) -> int:
        """Return aligned transition rows."""
        return int(self.metadata.shape[0])

    def handle_at(self, row_index: int) -> NativePlanningSessionHandle | None:
        """Return a strategic continuation handle or ``None`` at a leaf."""
        row = _bounded_index(row_index, self.row_count, "row_index")
        generation = int(
            self.metadata[
                row,
                int(NativePlanningSessionMetadataColumn.SESSION_GENERATION),
            ]
        )
        slot = int(
            self.metadata[
                row,
                int(NativePlanningSessionMetadataColumn.STATE_SLOT),
            ]
        )
        if generation < 0:
            return None
        return NativePlanningSessionHandle(generation, slot)

    def observation_bytes_for_row(self, row_index: int) -> memoryview | None:
        """Return one zero-copy root-visible JSON slice."""
        row = _bounded_index(row_index, self.row_count, "row_index")
        offset = int(
            self.metadata[
                row,
                int(NativePlanningSessionMetadataColumn.OBSERVATION_OFFSET),
            ]
        )
        size = int(
            self.metadata[
                row,
                int(NativePlanningSessionMetadataColumn.OBSERVATION_SIZE),
            ]
        )
        if size == 0:
            return None
        return self.observation_blob.data[offset : offset + size]

    def decode_observation_row(self, row_index: int) -> Mapping[str, Any] | None:
        """Decode only a requested root-visible row."""
        encoded = self.observation_bytes_for_row(row_index)
        if encoded is None:
            return None
        try:
            decoded = orjson.loads(encoded)
        except orjson.JSONDecodeError as exc:
            raise NativePlanningSessionPayloadError(
                f"native planning session row {row_index} has invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise NativePlanningSessionPayloadError(
                f"native planning session row {row_index} JSON is not an object"
            )
        return cast(Mapping[str, Any], decoded)


def native_planning_session_abi_fingerprint(descriptor: str) -> str:
    """Fingerprint the exact handle, metadata, and observation contract."""
    if not isinstance(descriptor, str):
        raise TypeError("native session ABI descriptor must be a string")
    return hashlib.sha256(
        _ABI_FINGERPRINT_DOMAIN + descriptor.encode("ascii")
    ).hexdigest()


def native_planning_session_schema_fingerprint() -> str:
    """Fingerprint every Python-side v5 payload and endpoint discriminator."""
    payload = {
        "magic": NATIVE_PLANNING_SESSION_MAGIC,
        "payload_version": NATIVE_PLANNING_SESSION_PAYLOAD_VERSION,
        "metadata_width": NATIVE_PLANNING_SESSION_METADATA_WIDTH,
        "fingerprint_bytes": NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES,
        "request_kinds": {
            item.name: int(item) for item in NativePlanningSessionRequestKind
        },
        "endpoints": {
            item.name: int(item) for item in NativePlanningSessionEndpoint
        },
        "metadata_columns": {
            item.name: int(item) for item in NativePlanningSessionMetadataColumn
        },
        "abi_descriptor": NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(_SCHEMA_FINGERPRINT_DOMAIN + encoded).hexdigest()


def parse_native_planning_session_payload(
    payload: bytes | bytearray | memoryview,
    *,
    expected_kind: NativePlanningSessionRequestKind,
    expected_rows: int,
    expected_generation: int | None = None,
    expected_root_player: int | None = None,
    expected_raw_request_fingerprint: bytes | None = None,
    expected_producer_contract_fingerprint: bytes | None = None,
    max_forced_steps: int = NATIVE_PLANNING_SESSION_MAX_FORCED_STEPS,
) -> NativePlanningSessionPayload:
    """Validate one session response without decoding observation objects."""
    row_count = _positive_int32(expected_rows, "expected_rows")
    if row_count > NATIVE_PLANNING_SESSION_MAX_ROWS:
        raise ValueError("expected_rows exceeds the native session row cap")
    if not isinstance(expected_kind, NativePlanningSessionRequestKind):
        raise TypeError("expected_kind must be NativePlanningSessionRequestKind")
    storage = payload if isinstance(payload, bytes) else bytes(payload)
    fixed_header_bytes = _HEADER_WIDTH * _INT32_BYTES
    header_bytes = fixed_header_bytes + 2 * NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES
    if len(storage) < header_bytes:
        raise NativePlanningSessionPayloadError(
            "native planning session payload ended before its header"
        )
    header = np.frombuffer(
        storage,
        dtype=_LITTLE_ENDIAN_INT32,
        count=_HEADER_WIDTH,
    )
    (
        magic,
        version,
        request_kind,
        generation,
        rows,
        metadata_width,
        blob_bytes,
        reserved,
    ) = (int(value) for value in header)
    if magic != NATIVE_PLANNING_SESSION_MAGIC:
        raise NativePlanningSessionPayloadError("session payload has invalid magic")
    if version != NATIVE_PLANNING_SESSION_PAYLOAD_VERSION:
        raise NativePlanningSessionPayloadError(
            "session payload has an unsupported version"
        )
    if request_kind != int(expected_kind):
        raise NativePlanningSessionPayloadError(
            "session payload request kind differs from the call"
        )
    if generation <= 0 or (
        expected_generation is not None and generation != expected_generation
    ):
        raise NativePlanningSessionPayloadError(
            "session payload generation differs from the active lease"
        )
    if rows != row_count or metadata_width != NATIVE_PLANNING_SESSION_METADATA_WIDTH:
        raise NativePlanningSessionPayloadError(
            "session payload shape differs from the request"
        )
    if reserved != 0:
        raise NativePlanningSessionPayloadError(
            "session payload reserved header field is nonzero"
        )
    if not 0 <= blob_bytes <= NATIVE_PLANNING_SESSION_MAX_OBSERVATION_BYTES:
        raise NativePlanningSessionPayloadError(
            "session payload observation size is outside the ABI cap"
        )
    raw_stop = fixed_header_bytes + NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES
    raw_fingerprint = storage[fixed_header_bytes:raw_stop]
    producer_fingerprint = storage[raw_stop:header_bytes]
    _validate_expected_fingerprint(
        raw_fingerprint,
        expected_raw_request_fingerprint,
        "raw request",
    )
    _validate_expected_fingerprint(
        producer_fingerprint,
        expected_producer_contract_fingerprint,
        "producer contract",
    )
    metadata_bytes = rows * metadata_width * _INT32_BYTES
    if len(storage) != header_bytes + metadata_bytes + blob_bytes:
        raise NativePlanningSessionPayloadError(
            "session payload size does not match its header"
        )
    metadata = np.frombuffer(
        storage,
        dtype=_LITTLE_ENDIAN_INT32,
        count=rows * metadata_width,
        offset=header_bytes,
    ).reshape(rows, metadata_width)
    metadata.setflags(write=False)
    observation_blob = np.frombuffer(
        storage,
        dtype=np.uint8,
        count=blob_bytes,
        offset=header_bytes + metadata_bytes,
    )
    observation_blob.setflags(write=False)
    _validate_metadata(
        metadata,
        generation=generation,
        blob_bytes=blob_bytes,
        expected_root_player=expected_root_player,
        max_forced_steps=max_forced_steps,
    )
    return NativePlanningSessionPayload(
        request_kind=NativePlanningSessionRequestKind(request_kind),
        generation=generation,
        metadata=metadata,
        observation_blob=observation_blob,
        raw_request_fingerprint=raw_fingerprint.hex(),
        producer_contract_fingerprint=producer_fingerprint.hex(),
        payload_bytes=len(storage),
        _storage=storage,
    )


def _validate_metadata(
    metadata: npt.NDArray[np.int32],
    *,
    generation: int,
    blob_bytes: int,
    expected_root_player: int | None,
    max_forced_steps: int,
) -> None:
    column = NativePlanningSessionMetadataColumn
    errors = metadata[:, int(column.ERROR)]
    exact = metadata[:, int(column.RULES_EXACT)]
    endpoints = metadata[:, int(column.ENDPOINT)]
    roots = metadata[:, int(column.ROOT_PLAYER)]
    leaves = metadata[:, int(column.LEAF_PLAYER)]
    contexts = metadata[:, int(column.LEAF_CONTEXT)]
    select_types = metadata[:, int(column.LEAF_SELECT_TYPE)]
    results = metadata[:, int(column.RESULT)]
    transitions = metadata[:, int(column.TRANSITION_STEPS)]
    forced = metadata[:, int(column.FORCED_STEPS)]
    offsets = metadata[:, int(column.OBSERVATION_OFFSET)].astype(np.int64, copy=False)
    sizes = metadata[:, int(column.OBSERVATION_SIZE)].astype(np.int64, copy=False)
    handle_generations = metadata[:, int(column.SESSION_GENERATION)]
    slots = metadata[:, int(column.STATE_SLOT)]
    if np.any(errors < 0) or np.any((exact != 0) & (exact != 1)):
        raise NativePlanningSessionPayloadError(
            "session metadata has invalid error/exactness flags"
        )
    if np.any(
        (endpoints < int(NativePlanningSessionEndpoint.INVALID))
        | (endpoints > int(NativePlanningSessionEndpoint.CHANCE_PROMPT))
    ):
        raise NativePlanningSessionPayloadError("session metadata endpoint is invalid")
    if np.any((roots < 0) | (roots > 1)):
        raise NativePlanningSessionPayloadError(
            "session metadata root player is invalid"
        )
    if expected_root_player is not None and np.any(
        roots != _bounded_root_player(expected_root_player)
    ):
        raise NativePlanningSessionPayloadError(
            "session metadata root player differs from the request"
        )
    if np.any((leaves < -1) | (leaves > 1)) or np.any(
        (contexts < -1) | (contexts > 48)
    ):
        raise NativePlanningSessionPayloadError(
            "session metadata leaf prompt is invalid"
        )
    if np.any((select_types < -1) | (select_types > 11)) or np.any(
        (results < -1) | (results > 2)
    ):
        raise NativePlanningSessionPayloadError(
            "session metadata leaf result/select type is invalid"
        )
    forced_cap = _bounded_forced_steps(max_forced_steps)
    if np.any(transitions < 0) or np.any(forced < 0) or np.any(forced > forced_cap):
        raise NativePlanningSessionPayloadError("session step counts are invalid")
    successful = errors == 0
    if np.any(successful & (transitions < 1)) or np.any(transitions > forced_cap + 1):
        raise NativePlanningSessionPayloadError(
            "session transition count exceeds the request"
        )
    if np.any(forced > np.maximum(transitions - 1, 0)):
        raise NativePlanningSessionPayloadError(
            "session forced steps exceed transition steps"
        )
    if np.any(offsets < 0) or np.any(sizes < 0) or np.any(offsets + sizes > blob_bytes):
        raise NativePlanningSessionPayloadError(
            "session observation slice exceeds its blob"
        )
    expected_offsets = np.empty_like(offsets)
    expected_offsets[0] = 0
    if sizes.size > 1:
        np.cumsum(sizes[:-1], out=expected_offsets[1:])
    if (
        not np.array_equal(offsets, expected_offsets)
        or int(sizes.sum(dtype=np.int64)) != blob_bytes
    ):
        raise NativePlanningSessionPayloadError(
            "session observation slices are not contiguous request rows"
        )
    has_endpoint = endpoints != int(NativePlanningSessionEndpoint.INVALID)
    has_observation = sizes > 0
    if np.any(successful != (exact == 1)) or np.any(successful != has_endpoint):
        raise NativePlanningSessionPayloadError(
            "session success, exactness, and endpoint disagree"
        )
    if np.any(successful != has_observation):
        raise NativePlanningSessionPayloadError(
            "session success and observation presence disagree"
        )
    terminal = endpoints == int(NativePlanningSessionEndpoint.TERMINAL)
    same_main = endpoints == int(NativePlanningSessionEndpoint.SAME_SEAT_MAIN)
    handoff = endpoints == int(NativePlanningSessionEndpoint.TURN_HANDOFF)
    strategic = endpoints == int(NativePlanningSessionEndpoint.ROOT_STRATEGIC_PROMPT)
    chance = endpoints == int(NativePlanningSessionEndpoint.CHANCE_PROMPT)
    nonterminal = successful & ~terminal
    if np.any(terminal & ((leaves != -1) | (contexts != -1) | (select_types != -1))):
        raise NativePlanningSessionPayloadError(
            "terminal session rows retain a leaf prompt"
        )
    if np.any(terminal & ((results < 0) | (results > 2))) or np.any(
        nonterminal & (results != -1)
    ):
        raise NativePlanningSessionPayloadError(
            "session terminal result disagrees with its endpoint"
        )
    if np.any(
        same_main
        & (
            (leaves != roots)
            | (contexts != int(SelectContext.MAIN))
            | (select_types != 0)
        )
    ):
        raise NativePlanningSessionPayloadError(
            "same-seat MAIN session metadata is inconsistent"
        )
    if np.any(handoff & ((leaves < 0) | (leaves == roots))):
        raise NativePlanningSessionPayloadError(
            "handoff session metadata is inconsistent"
        )
    if np.any(strategic & ((leaves != roots) | (contexts < 0) | (select_types < 0))):
        raise NativePlanningSessionPayloadError(
            "strategic session metadata is inconsistent"
        )
    if np.any(chance & ((leaves < 0) | (contexts != int(SelectContext.COIN_HEAD)))):
        raise NativePlanningSessionPayloadError(
            "chance session metadata is inconsistent"
        )
    needs_handle = successful & (strategic | chance)
    has_handle = (handle_generations >= 0) | (slots >= 0)
    if np.any((handle_generations < -1) | (slots < -1)) or np.any(
        (handle_generations >= 0) != (slots >= 0)
    ):
        raise NativePlanningSessionPayloadError("session handle sentinel is invalid")
    if np.any(needs_handle != has_handle) or np.any(
        needs_handle & (handle_generations != generation)
    ):
        raise NativePlanningSessionPayloadError(
            "session continuation handle disagrees with its endpoint/generation"
        )
    active_slots = slots[needs_handle]
    if active_slots.size and len({int(value) for value in active_slots}) != int(
        active_slots.size
    ):
        raise NativePlanningSessionPayloadError(
            "session response contains duplicate newly allocated handles"
        )


def _validate_expected_fingerprint(
    actual: bytes,
    expected: bytes | None,
    name: str,
) -> None:
    if expected is None:
        return
    if len(expected) != NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES:
        raise ValueError(f"expected {name} fingerprint must contain 32 bytes")
    if actual != expected:
        raise NativePlanningSessionPayloadError(
            f"native planning session {name} fingerprint differs from request"
        )


def _positive_int32(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 0 < parsed <= _INT32_MAX:
        raise ValueError(f"{name} must be positive int32")
    return int(parsed)


def _nonnegative_int32(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 0 <= parsed <= _INT32_MAX:
        raise ValueError(f"{name} must be non-negative int32")
    return int(parsed)


def _bounded_root_player(value: Any) -> int:
    if isinstance(value, bool) or value not in (0, 1):
        raise ValueError("expected_root_player must be 0 or 1")
    return int(value)


def _bounded_forced_steps(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("max_forced_steps must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError("max_forced_steps must be an integer") from exc
    if not 0 <= parsed <= NATIVE_PLANNING_SESSION_MAX_FORCED_STEPS:
        raise ValueError("max_forced_steps is outside the session ABI cap")
    return int(parsed)


def _bounded_index(value: Any, length: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not 0 <= parsed < length:
        raise IndexError(f"{name} must be in [0, {length})")
    return int(parsed)


__all__ = [
    "NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR",
    "NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES",
    "NATIVE_PLANNING_SESSION_MAGIC",
    "NATIVE_PLANNING_SESSION_MAX_ENGINE_STEPS",
    "NATIVE_PLANNING_SESSION_MAX_FORCED_STEPS",
    "NATIVE_PLANNING_SESSION_MAX_STATE_SLOTS",
    "NATIVE_PLANNING_SESSION_MAX_OBSERVATION_BYTES",
    "NATIVE_PLANNING_SESSION_MAX_ROWS",
    "NATIVE_PLANNING_SESSION_METADATA_WIDTH",
    "NATIVE_PLANNING_SESSION_PAYLOAD_VERSION",
    "NativePlanningSessionEndpoint",
    "NativePlanningSessionHandle",
    "NativePlanningSessionMetadataColumn",
    "NativePlanningSessionPayload",
    "NativePlanningSessionPayloadError",
    "NativePlanningSessionRequestKind",
    "native_planning_session_abi_fingerprint",
    "native_planning_session_schema_fingerprint",
    "parse_native_planning_session_payload",
]
