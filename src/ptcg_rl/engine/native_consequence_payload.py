"""Validated zero-copy views for the native consequence wire format."""

from __future__ import annotations

import hashlib
import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import orjson

from ptcg_rl.engine.constants import SelectContext

NATIVE_CONSEQUENCE_MAGIC = 0x31434745
NATIVE_CONSEQUENCE_PAYLOAD_VERSION = 6
NATIVE_CONSEQUENCE_METADATA_WIDTH = 14
NATIVE_CONSEQUENCE_MAX_CELLS = 1 << 16
NATIVE_CONSEQUENCE_MAX_ENGINE_STEPS = 1 << 24
NATIVE_CONSEQUENCE_MAX_FORCED_STEPS = 64
NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES = 1 << 30
NATIVE_CONSEQUENCE_CELL_ORDER = "candidate_major"
NATIVE_CONSEQUENCE_FINGERPRINT_BYTES = 32
# Historical public name retained for callers that size the raw request hash.
NATIVE_CONSEQUENCE_REQUEST_FINGERPRINT_BYTES = NATIVE_CONSEQUENCE_FINGERPRINT_BYTES
NATIVE_CONSEQUENCE_ABI_DESCRIPTOR = (
    "cg-planner/v6;cell_order=candidate_major;"
    "header=<6i+raw_sha256+producer_sha256;"
    "metadata=<14i:error,rules_exact,endpoint,root_player,leaf_player,"
    "leaf_context,transition_steps,forced_steps,observation_offset,"
    "observation_size,result,leaf_select_type,leaf_observation_offset,"
    "leaf_observation_size;"
    "rng=request_seeded_world_stream_common_across_candidates_v1;"
    "observation=root_visible_then_leaf_actor_visible_select_logs_current_json_v1"
)
_ABI_FINGERPRINT_DOMAIN = b"ptcg-rl/native-consequence/abi/v1\x00"

_HEADER_WIDTH = 6
_INT32_BYTES = 4
_INT32_MAX = (1 << 31) - 1
_LITTLE_ENDIAN_INT32 = np.dtype("<i4")


class NativeConsequencePayloadError(RuntimeError):
    """Raised when native consequence output violates the wire contract."""


class NativeConsequenceEndpoint(IntEnum):
    """Semantic boundary reached after a candidate and forced closure."""

    INVALID = 0
    TERMINAL = 1
    SAME_SEAT_MAIN = 2
    TURN_HANDOFF = 3
    ROOT_STRATEGIC_PROMPT = 4
    CHANCE_PROMPT = 5


class NativeConsequenceMetadataColumn(IntEnum):
    """Column indices in the fixed-width consequence metadata matrix."""

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
    LEAF_OBSERVATION_OFFSET = 12
    LEAF_OBSERVATION_SIZE = 13


def native_consequence_abi_fingerprint(descriptor: str) -> str:
    """Fingerprint the exact native wire and root-observation descriptor."""
    if not isinstance(descriptor, str):
        raise TypeError("native ABI descriptor must be a string")
    encoded = descriptor.encode("ascii")
    return hashlib.sha256(_ABI_FINGERPRINT_DOMAIN + encoded).hexdigest()


@dataclass(frozen=True, eq=False)
class NativeConsequencePayload:
    """Validated zero-copy views over one immutable native payload."""

    worlds: int
    candidates: int
    metadata: npt.NDArray[np.int32]
    observation_blob: npt.NDArray[np.uint8]
    raw_request_fingerprint: str
    producer_contract_fingerprint: str
    payload_bytes: int
    _storage: bytes = field(repr=False)

    @property
    def cell_count(self) -> int:
        """Return the number of candidate-major world cells."""
        return self.worlds * self.candidates

    def row_index(self, world_index: int, candidate_index: int) -> int:
        """Resolve one world/candidate coordinate to its candidate-major row."""
        world = _bounded_index(world_index, self.worlds, "world_index")
        candidate = _bounded_index(
            candidate_index,
            self.candidates,
            "candidate_index",
        )
        return candidate * self.worlds + world

    def observation_bytes_for_row(self, row_index: int) -> memoryview | None:
        """Return a zero-copy JSON slice for one row, if the row has one."""
        row = _bounded_index(row_index, self.cell_count, "row_index")
        offset = int(
            self.metadata[
                row,
                int(NativeConsequenceMetadataColumn.OBSERVATION_OFFSET),
            ]
        )
        size = int(
            self.metadata[
                row,
                int(NativeConsequenceMetadataColumn.OBSERVATION_SIZE),
            ]
        )
        if size == 0:
            return None
        return self.observation_blob.data[offset : offset + size]

    @property
    def root_observation_blob(self) -> npt.NDArray[np.uint8]:
        """Return the contiguous root-visible prefix without copying."""
        sizes = self.metadata[
            :, int(NativeConsequenceMetadataColumn.OBSERVATION_SIZE)
        ]
        size = int(sizes.sum(dtype=np.int64))
        view = self.observation_blob[:size]
        view.setflags(write=False)
        return view

    def leaf_observation_bytes_for_row(
        self,
        row_index: int,
    ) -> memoryview | None:
        """Return the actual leaf actor's JSON slice for one row."""
        row = _bounded_index(row_index, self.cell_count, "row_index")
        offset = int(
            self.metadata[
                row,
                int(NativeConsequenceMetadataColumn.LEAF_OBSERVATION_OFFSET),
            ]
        )
        size = int(
            self.metadata[
                row,
                int(NativeConsequenceMetadataColumn.LEAF_OBSERVATION_SIZE),
            ]
        )
        if size == 0:
            return None
        return self.observation_blob.data[offset : offset + size]

    def decode_observation_row(self, row_index: int) -> Mapping[str, Any] | None:
        """Decode one row's root-visible observation with ``orjson``."""
        encoded = self.observation_bytes_for_row(row_index)
        if encoded is None:
            return None
        try:
            decoded = orjson.loads(encoded)
        except orjson.JSONDecodeError as exc:
            raise NativeConsequencePayloadError(
                f"native consequence row {row_index} has invalid JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise NativeConsequencePayloadError(
                f"native consequence row {row_index} JSON is not an object"
            )
        return cast(Mapping[str, Any], decoded)

    def decode_observation(
        self,
        world_index: int,
        candidate_index: int,
    ) -> Mapping[str, Any] | None:
        """Decode the observation at one world/candidate coordinate."""
        return self.decode_observation_row(
            self.row_index(world_index, candidate_index)
        )

    def decode_leaf_observation_row(
        self,
        row_index: int,
    ) -> Mapping[str, Any] | None:
        """Decode one row as observed by the actual nonterminal leaf actor."""
        encoded = self.leaf_observation_bytes_for_row(row_index)
        if encoded is None:
            return None
        try:
            decoded = orjson.loads(encoded)
        except orjson.JSONDecodeError as exc:
            raise NativeConsequencePayloadError(
                f"native consequence row {row_index} has invalid leaf JSON"
            ) from exc
        if not isinstance(decoded, dict):
            raise NativeConsequencePayloadError(
                f"native consequence row {row_index} leaf JSON is not an object"
            )
        return cast(Mapping[str, Any], decoded)

    def decode_leaf_observation(
        self,
        world_index: int,
        candidate_index: int,
    ) -> Mapping[str, Any] | None:
        """Decode the leaf-actor observation at one grid coordinate."""
        return self.decode_leaf_observation_row(
            self.row_index(world_index, candidate_index)
        )


def parse_native_consequence_payload(
    payload: bytes | bytearray | memoryview,
    *,
    expected_worlds: int,
    expected_candidates: int,
    expected_root_player: int | None = None,
    expected_raw_request_fingerprint: bytes | None = None,
    expected_producer_contract_fingerprint: bytes | None = None,
    max_forced_steps: int = NATIVE_CONSEQUENCE_MAX_FORCED_STEPS,
) -> NativeConsequencePayload:
    """Validate one wire payload and return zero-copy metadata/blob views."""
    expected_world_count = _positive_int32(expected_worlds, "expected_worlds")
    expected_candidate_count = _positive_int32(
        expected_candidates,
        "expected_candidates",
    )
    expected_cells = expected_world_count * expected_candidate_count
    if expected_cells > NATIVE_CONSEQUENCE_MAX_CELLS:
        raise ValueError(
            "expected candidate-by-world grid exceeds the native transition cap"
        )
    storage = payload if isinstance(payload, bytes) else bytes(payload)
    fixed_header_bytes = _HEADER_WIDTH * _INT32_BYTES
    header_bytes = fixed_header_bytes + 2 * NATIVE_CONSEQUENCE_FINGERPRINT_BYTES
    if len(storage) < header_bytes:
        raise NativeConsequencePayloadError(
            "native consequence payload ended before its header"
        )
    header = np.frombuffer(
        storage,
        dtype=_LITTLE_ENDIAN_INT32,
        count=_HEADER_WIDTH,
    )
    magic, version, worlds, candidates, metadata_width, blob_bytes = (
        int(value) for value in header
    )
    if magic != NATIVE_CONSEQUENCE_MAGIC:
        raise NativeConsequencePayloadError(
            "native consequence payload has an invalid magic"
        )
    if version != NATIVE_CONSEQUENCE_PAYLOAD_VERSION:
        raise NativeConsequencePayloadError(
            "native consequence payload has an unsupported version"
        )
    if worlds != expected_world_count or candidates != expected_candidate_count:
        raise NativeConsequencePayloadError(
            "native consequence payload dimensions do not match the request"
        )
    if metadata_width != NATIVE_CONSEQUENCE_METADATA_WIDTH:
        raise NativeConsequencePayloadError(
            "native consequence payload metadata width does not match the ABI"
        )
    if blob_bytes < 0 or blob_bytes > NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES:
        raise NativeConsequencePayloadError(
            "native consequence payload blob size is outside the ABI cap"
        )
    raw_fingerprint_stop = (
        fixed_header_bytes + NATIVE_CONSEQUENCE_FINGERPRINT_BYTES
    )
    raw_request_fingerprint_bytes = storage[
        fixed_header_bytes:raw_fingerprint_stop
    ]
    producer_contract_fingerprint_bytes = storage[
        raw_fingerprint_stop:header_bytes
    ]
    _validate_expected_fingerprint(
        raw_request_fingerprint_bytes,
        expected_raw_request_fingerprint,
        name="raw request",
    )
    _validate_expected_fingerprint(
        producer_contract_fingerprint_bytes,
        expected_producer_contract_fingerprint,
        name="producer contract",
    )

    metadata_bytes = expected_cells * metadata_width * _INT32_BYTES
    expected_payload_bytes = header_bytes + metadata_bytes + blob_bytes
    if len(storage) != expected_payload_bytes:
        raise NativeConsequencePayloadError(
            "native consequence payload size does not match its header"
        )
    metadata = np.frombuffer(
        storage,
        dtype=_LITTLE_ENDIAN_INT32,
        count=expected_cells * metadata_width,
        offset=header_bytes,
    ).reshape(expected_cells, metadata_width)
    metadata.setflags(write=False)
    blob_start = header_bytes + metadata_bytes
    observation_blob = np.frombuffer(
        storage,
        dtype=np.uint8,
        count=blob_bytes,
        offset=blob_start,
    )
    observation_blob.setflags(write=False)
    _validate_metadata(
        metadata,
        blob_bytes=blob_bytes,
        expected_root_player=expected_root_player,
        max_forced_steps=max_forced_steps,
    )
    return NativeConsequencePayload(
        worlds=worlds,
        candidates=candidates,
        metadata=metadata,
        observation_blob=observation_blob,
        raw_request_fingerprint=raw_request_fingerprint_bytes.hex(),
        producer_contract_fingerprint=(
            producer_contract_fingerprint_bytes.hex()
        ),
        payload_bytes=len(storage),
        _storage=storage,
    )


def _validate_metadata(
    metadata: npt.NDArray[np.int32],
    *,
    blob_bytes: int,
    expected_root_player: int | None,
    max_forced_steps: int,
) -> None:
    errors = metadata[:, int(NativeConsequenceMetadataColumn.ERROR)]
    if np.any(errors < 0):
        raise NativeConsequencePayloadError(
            "native consequence metadata has a negative error code"
        )
    exact = metadata[:, int(NativeConsequenceMetadataColumn.RULES_EXACT)]
    if np.any((exact != 0) & (exact != 1)):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid rules_exact flag"
        )
    endpoints = metadata[:, int(NativeConsequenceMetadataColumn.ENDPOINT)]
    if np.any(
        (endpoints < int(NativeConsequenceEndpoint.INVALID))
        | (endpoints > int(NativeConsequenceEndpoint.CHANCE_PROMPT))
    ):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid endpoint"
        )
    roots = metadata[:, int(NativeConsequenceMetadataColumn.ROOT_PLAYER)]
    if np.any((roots < 0) | (roots > 1)):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid root player"
        )
    if expected_root_player is not None:
        expected_root = _bounded_root_player(expected_root_player)
        if np.any(roots != expected_root):
            raise NativeConsequencePayloadError(
                "native consequence payload root player differs from request"
            )
    leaves = metadata[:, int(NativeConsequenceMetadataColumn.LEAF_PLAYER)]
    if np.any((leaves < -1) | (leaves > 1)):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid leaf player"
        )
    contexts = metadata[:, int(NativeConsequenceMetadataColumn.LEAF_CONTEXT)]
    select_types = metadata[
        :, int(NativeConsequenceMetadataColumn.LEAF_SELECT_TYPE)
    ]
    results = metadata[:, int(NativeConsequenceMetadataColumn.RESULT)]
    if np.any((contexts < -1) | (contexts > 48)):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid leaf context"
        )
    if np.any((select_types < -1) | (select_types > 11)):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid leaf select type"
        )
    if np.any((results < -1) | (results > 2)):
        raise NativeConsequencePayloadError(
            "native consequence metadata has an invalid result"
        )
    transition_steps = metadata[
        :, int(NativeConsequenceMetadataColumn.TRANSITION_STEPS)
    ]
    forced_steps = metadata[:, int(NativeConsequenceMetadataColumn.FORCED_STEPS)]
    if np.any(transition_steps < 0) or np.any(forced_steps < 0):
        raise NativeConsequencePayloadError(
            "native consequence metadata has a negative step count"
        )
    forced_cap = _bounded_forced_steps(max_forced_steps)
    if np.any(forced_steps > forced_cap):
        raise NativeConsequencePayloadError(
            "native consequence metadata exceeds the forced-step ABI cap"
        )
    successful = errors == 0
    if np.any(successful & (transition_steps < 1)):
        raise NativeConsequencePayloadError(
            "successful native consequence metadata has no transition"
        )
    if np.any(transition_steps > forced_cap + 1):
        raise NativeConsequencePayloadError(
            "native consequence metadata exceeds the request step cap"
        )
    maximum_forced_steps = np.maximum(transition_steps - 1, 0)
    if np.any(forced_steps > maximum_forced_steps):
        raise NativeConsequencePayloadError(
            "native consequence forced steps exceed its transitions"
        )
    root_offsets = metadata[
        :, int(NativeConsequenceMetadataColumn.OBSERVATION_OFFSET)
    ].astype(np.int64, copy=False)
    root_sizes = metadata[
        :, int(NativeConsequenceMetadataColumn.OBSERVATION_SIZE)
    ].astype(np.int64, copy=False)
    leaf_offsets = metadata[
        :, int(NativeConsequenceMetadataColumn.LEAF_OBSERVATION_OFFSET)
    ].astype(np.int64, copy=False)
    leaf_sizes = metadata[
        :, int(NativeConsequenceMetadataColumn.LEAF_OBSERVATION_SIZE)
    ].astype(np.int64, copy=False)
    if (
        np.any(root_offsets < 0)
        or np.any(root_sizes < 0)
        or np.any(leaf_offsets < 0)
        or np.any(leaf_sizes < 0)
    ):
        raise NativeConsequencePayloadError(
            "native consequence metadata has a negative observation slice"
        )
    if (
        np.any(root_offsets + root_sizes > blob_bytes)
        or np.any(leaf_offsets + leaf_sizes > blob_bytes)
    ):
        raise NativeConsequencePayloadError(
            "native consequence metadata observation slice exceeds the blob"
        )
    has_exact_rules = exact == 1
    has_endpoint = endpoints != int(NativeConsequenceEndpoint.INVALID)
    has_root_observation = root_sizes > 0
    has_leaf_observation = leaf_sizes > 0
    if np.any(successful != has_exact_rules):
        raise NativeConsequencePayloadError(
            "native consequence success and rules_exact disagree"
        )
    if np.any(successful != has_endpoint):
        raise NativeConsequencePayloadError(
            "native consequence success and endpoint disagree"
        )
    if np.any(successful != has_root_observation):
        raise NativeConsequencePayloadError(
            "native consequence success and root observation presence disagree"
        )
    expected_root_offsets = np.empty_like(root_offsets)
    expected_root_offsets[0] = 0
    if root_sizes.size > 1:
        np.cumsum(root_sizes[:-1], out=expected_root_offsets[1:])
    if not np.array_equal(root_offsets, expected_root_offsets):
        raise NativeConsequencePayloadError(
            "native consequence root observation slices are not contiguous "
            "candidate-major rows"
        )
    root_blob_bytes = int(root_sizes.sum(dtype=np.int64))
    expected_leaf_offsets = np.empty_like(leaf_offsets)
    expected_leaf_offsets[0] = root_blob_bytes
    if leaf_sizes.size > 1:
        np.cumsum(leaf_sizes[:-1], out=expected_leaf_offsets[1:])
        expected_leaf_offsets[1:] += root_blob_bytes
    if not np.array_equal(leaf_offsets, expected_leaf_offsets):
        raise NativeConsequencePayloadError(
            "native consequence leaf observation slices are not contiguous "
            "candidate-major rows after the root prefix"
        )
    if root_blob_bytes + int(leaf_sizes.sum(dtype=np.int64)) != blob_bytes:
        raise NativeConsequencePayloadError(
            "native consequence root and leaf observation slices do not cover "
            "the blob"
        )

    terminal = endpoints == int(NativeConsequenceEndpoint.TERMINAL)
    same_main = endpoints == int(NativeConsequenceEndpoint.SAME_SEAT_MAIN)
    handoff = endpoints == int(NativeConsequenceEndpoint.TURN_HANDOFF)
    strategic = endpoints == int(
        NativeConsequenceEndpoint.ROOT_STRATEGIC_PROMPT
    )
    chance = endpoints == int(NativeConsequenceEndpoint.CHANCE_PROMPT)
    nonterminal = successful & ~terminal
    if np.any(nonterminal != has_leaf_observation):
        raise NativeConsequencePayloadError(
            "nonterminal success and leaf-actor observation presence disagree"
        )
    if np.any(terminal & ((leaves != -1) | (contexts != -1) | (select_types != -1))):
        raise NativeConsequencePayloadError(
            "terminal consequence rows must not retain a leaf prompt"
        )
    if np.any(terminal & ((results < 0) | (results > 2))):
        raise NativeConsequencePayloadError(
            "terminal consequence rows need a terminal result"
        )
    if np.any(nonterminal & (results != -1)):
        raise NativeConsequencePayloadError(
            "nonterminal consequence rows cannot carry a result"
        )
    if np.any(same_main & ((leaves != roots) | (contexts != int(SelectContext.MAIN)) | (select_types != 0))):
        raise NativeConsequencePayloadError(
            "same-seat MAIN consequence metadata is inconsistent"
        )
    if np.any(handoff & ((leaves < 0) | (leaves == roots))):
        raise NativeConsequencePayloadError(
            "handoff consequence metadata is inconsistent"
        )
    if np.any(strategic & ((leaves != roots) | (contexts < 0) | (select_types < 0))):
        raise NativeConsequencePayloadError(
            "root strategic consequence metadata is inconsistent"
        )
    if np.any(chance & ((leaves < 0) | (contexts != int(SelectContext.COIN_HEAD)))):
        raise NativeConsequencePayloadError(
            "chance consequence metadata is inconsistent"
        )


def _positive_int32(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if parsed < 1 or parsed > _INT32_MAX:
        raise ValueError(f"{name} must be in [1, {_INT32_MAX}]")
    return int(parsed)


def _validate_expected_fingerprint(
    actual: bytes,
    expected: bytes | None,
    *,
    name: str,
) -> None:
    if expected is None:
        return
    if len(expected) != NATIVE_CONSEQUENCE_FINGERPRINT_BYTES:
        raise ValueError(f"expected {name} fingerprint must contain 32 bytes")
    if actual != expected:
        raise NativeConsequencePayloadError(
            f"native consequence {name} fingerprint differs from request"
        )


def _bounded_root_player(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("expected_root_player must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError("expected_root_player must be an integer") from exc
    if parsed not in (0, 1):
        raise ValueError("expected_root_player must be 0 or 1")
    return int(parsed)


def _bounded_forced_steps(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("max_forced_steps must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError("max_forced_steps must be an integer") from exc
    if parsed < 0 or parsed > NATIVE_CONSEQUENCE_MAX_FORCED_STEPS:
        raise ValueError(
            "max_forced_steps is outside the native consequence ABI cap"
        )
    return int(parsed)


def _bounded_index(value: Any, length: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if parsed < 0 or parsed >= length:
        raise IndexError(f"{name} must be in [0, {length})")
    return int(parsed)


__all__ = [
    "NATIVE_CONSEQUENCE_CELL_ORDER",
    "NATIVE_CONSEQUENCE_ABI_DESCRIPTOR",
    "NATIVE_CONSEQUENCE_FINGERPRINT_BYTES",
    "NATIVE_CONSEQUENCE_MAGIC",
    "NATIVE_CONSEQUENCE_MAX_FORCED_STEPS",
    "NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES",
    "NATIVE_CONSEQUENCE_MAX_CELLS",
    "NATIVE_CONSEQUENCE_MAX_ENGINE_STEPS",
    "NATIVE_CONSEQUENCE_METADATA_WIDTH",
    "NATIVE_CONSEQUENCE_PAYLOAD_VERSION",
    "NATIVE_CONSEQUENCE_REQUEST_FINGERPRINT_BYTES",
    "NativeConsequenceEndpoint",
    "NativeConsequenceMetadataColumn",
    "NativeConsequencePayload",
    "NativeConsequencePayloadError",
    "parse_native_consequence_payload",
    "native_consequence_abi_fingerprint",
]
