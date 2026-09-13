"""Fixed msgpack control envelopes for native rollout delivery."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import msgpack

_CONTROL_SCHEMA = "ptcg-rl/native-rollout-control/v1"
_CONTROL_FIELDS = (
    "schema",
    "message_type",
    "lease_id",
    "part_id",
    "sequence_id",
    "error_code",
)
_CONTROL_MESSAGE_TYPES = frozenset({"PART_ACK", "ABORT"})
_ABORT_ERROR_CODES = frozenset(
    {
        "DECODE_ERROR",
        "DUPLICATE_PART_ID",
        "IDENTITY_MISMATCH",
        "PROTOCOL_ERROR",
        "RECEIVER_ABORTED",
        "SEQUENCE_MISMATCH",
    }
)
_MAX_CONTROL_BYTES = 4096
_MAX_ID_LENGTH = 256


class NativeRolloutSessionError(RuntimeError):
    """Base class for fail-closed native rollout session errors."""


class NativeRolloutProtocolError(NativeRolloutSessionError):
    """A data or control message violated the bound session contract."""


class NativeRolloutRemoteAbortError(NativeRolloutSessionError):
    """The remote endpoint explicitly rejected the current part."""

    def __init__(self, error_code: str) -> None:
        super().__init__(f"native rollout receiver aborted: {error_code}")
        self.error_code = error_code


@dataclass(frozen=True, slots=True)
class _ControlEnvelope:
    message_type: Literal["PART_ACK", "ABORT"]
    lease_id: str
    part_id: str
    sequence_id: int
    error_code: str


def _encode_control(control: _ControlEnvelope) -> bytes:
    if control.message_type == "PART_ACK":
        if control.error_code != "NONE":
            raise ValueError("native rollout PART_ACK error code must be NONE")
    elif control.error_code not in _ABORT_ERROR_CODES:
        raise ValueError("native rollout ABORT error code is invalid")
    return cast(
        bytes,
        msgpack.packb(
            {
                "schema": _CONTROL_SCHEMA,
                "message_type": control.message_type,
                "lease_id": control.lease_id,
                "part_id": control.part_id,
                "sequence_id": control.sequence_id,
                "error_code": control.error_code,
            },
            use_bin_type=True,
        ),
    )


def _decode_control(frame: Any) -> _ControlEnvelope:
    try:
        view = memoryview(frame).cast("B")
    except (TypeError, ValueError) as exc:
        raise NativeRolloutProtocolError(
            "native rollout control frame is not contiguous bytes"
        ) from exc
    if view.nbytes > _MAX_CONTROL_BYTES:
        raise NativeRolloutProtocolError("native rollout control frame is too large")
    try:
        unpacked = msgpack.unpackb(
            bytes(view),
            raw=False,
            strict_map_key=True,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native rollout control frame is invalid msgpack"
        ) from exc
    if not isinstance(unpacked, dict) or tuple(unpacked) != _CONTROL_FIELDS:
        raise NativeRolloutProtocolError(
            "native rollout control envelope fields are invalid"
        )
    fields = cast(dict[str, object], unpacked)
    if fields["schema"] != _CONTROL_SCHEMA:
        raise NativeRolloutProtocolError("native rollout control schema is unsupported")
    message_type = fields["message_type"]
    lease_id = fields["lease_id"]
    part_id = fields["part_id"]
    sequence_id = fields["sequence_id"]
    error_code = fields["error_code"]
    if (
        not isinstance(message_type, str)
        or message_type not in _CONTROL_MESSAGE_TYPES
        or not _valid_control_id(lease_id)
        or not _valid_control_id(part_id)
        or type(sequence_id) is not int
        or sequence_id < 0
        or not isinstance(error_code, str)
    ):
        raise NativeRolloutProtocolError(
            "native rollout control envelope values are invalid"
        )
    if message_type == "PART_ACK":
        if error_code != "NONE":
            raise NativeRolloutProtocolError(
                "native rollout PART_ACK error code is invalid"
            )
    elif error_code not in _ABORT_ERROR_CODES:
        raise NativeRolloutProtocolError("native rollout ABORT error code is invalid")
    return _ControlEnvelope(
        message_type=cast(Literal["PART_ACK", "ABORT"], message_type),
        lease_id=cast(str, lease_id),
        part_id=cast(str, part_id),
        sequence_id=sequence_id,
        error_code=error_code,
    )


def _valid_control_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= _MAX_ID_LENGTH
        and value.strip() == value
        and all(ord(character) >= 32 for character in value)
    )


__all__ = [
    "NativeRolloutProtocolError",
    "NativeRolloutRemoteAbortError",
    "NativeRolloutSessionError",
]
