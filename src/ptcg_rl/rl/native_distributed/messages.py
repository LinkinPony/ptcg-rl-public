"""Strict small-message schemas for native distributed ZMQ channels."""

from __future__ import annotations

from typing import Any, Literal, TypeVar, cast

import msgpack
from pydantic import BaseModel, ConfigDict, Field

from ptcg_rl.rl.native_distributed.contracts import (
    NativeBfloat16ArtifactManifest,
    NativeCollectionAttempt,
    NativeCollectionPartIdentity,
    NativeCollectionShardLease,
    NativeCollectionShardResult,
    NativeCollectionWindowReceipt,
    NativeCollectionWorkerManifest,
)
from ptcg_rl.rl.native_distributed.control import NativeRolloutProtocolError

_SCHEMA = "ptcg-rl/native-distributed-message/v1"
_FIELDS = ("schema", "model", "payload")
_MAX_MESSAGE_BYTES = 1 << 24
MessageT = TypeVar("MessageT", bound=BaseModel)


class _StrictMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class NativeRegisterRequest(_StrictMessage):
    """Register one exact worker session on the control channel."""

    message_type: Literal["REGISTER"] = "REGISTER"
    manifest: NativeCollectionWorkerManifest
    sent_at_unix_ns: int = Field(ge=0)


class NativeReadyRequest(_StrictMessage):
    """Advertise preflight completion for future window admission."""

    message_type: Literal["READY"] = "READY"
    worker_id: str
    session_id: str
    sent_at_unix_ns: int = Field(ge=0)


class NativeWorkerMetrics(_StrictMessage):
    """Bounded worker runtime metrics attached to heartbeats."""

    gpu_memory_allocated_bytes: int = Field(default=0, ge=0)
    gpu_memory_reserved_bytes: int = Field(default=0, ge=0)
    gpu_memory_reclaimable_bytes: int = Field(default=0, ge=0)
    gpu_device_free_bytes: int = Field(default=0, ge=0)
    gpu_utilization_percent: float = Field(default=0.0, ge=0.0, le=100.0)
    cuda_cache_trim_count: int = Field(default=0, ge=0)
    cuda_cache_trim_released_bytes: int = Field(default=0, ge=0)
    cuda_cache_last_trim_released_bytes: int = Field(default=0, ge=0)
    process_rss_bytes: int = Field(default=0, ge=0)
    system_available_memory_bytes: int = Field(default=0, ge=0)
    cgroup_memory_current_bytes: int = Field(default=0, ge=0)
    cgroup_memory_limit_bytes: int = Field(default=0, ge=0)
    host_memory_trim_count: int = Field(default=0, ge=0)
    host_memory_trim_released_bytes: int = Field(default=0, ge=0)
    host_memory_last_trim_released_bytes: int = Field(default=0, ge=0)
    artifact_cache_entries: int = Field(default=0, ge=0)
    artifact_cache_bytes: int = Field(default=0, ge=0)
    model_cache_entries: int = Field(default=0, ge=0)
    collection_decisions_per_second: float = Field(default=0.0, ge=0.0)
    active_lease_id: str | None = None
    active_attempt_id: str | None = None


class NativeHeartbeatRequest(_StrictMessage):
    """Refresh a worker session and publish non-semantic telemetry."""

    message_type: Literal["HEARTBEAT"] = "HEARTBEAT"
    worker_id: str
    session_id: str
    metrics: NativeWorkerMetrics = Field(default_factory=NativeWorkerMetrics)
    sent_at_unix_ns: int = Field(ge=0)


class NativeWorkRequest(_StrictMessage):
    """Ask for work only after the preceding response was settled."""

    message_type: Literal["WORK_REQUEST"] = "WORK_REQUEST"
    worker_id: str
    session_id: str
    sent_at_unix_ns: int = Field(ge=0)


class NativeLeaseResponse(_StrictMessage):
    """Assign one immutable shard lease and its current attempt."""

    message_type: Literal["LEASE"] = "LEASE"
    lease: NativeCollectionShardLease
    attempt: NativeCollectionAttempt


class NativeWaitResponse(_StrictMessage):
    """Tell a worker to poll again without changing any lease.

    ``drain_hint`` advises an active attempt to seal its live games promptly.
    Cutoff games are transactionally removed from PPO rows before completion;
    unlike ``NativeDrainResponse`` the remaining attempt stays accepted.
    """

    message_type: Literal["WAIT"] = "WAIT"
    reason: str
    retry_after_seconds: float = Field(gt=0.0)
    drain_hint: bool = False


class NativeDrainResponse(_StrictMessage):
    """Stop issuing or uploading work after a window reaches its cutoff."""

    message_type: Literal["DRAIN"] = "DRAIN"
    window_id: str


class NativeAttemptFailedRequest(_StrictMessage):
    """Report a failed worker attempt without changing its assignment lease."""

    message_type: Literal["ATTEMPT_FAILED"] = "ATTEMPT_FAILED"
    worker_id: str
    session_id: str
    lease_id: str
    attempt_id: str
    reason: str
    sent_at_unix_ns: int = Field(ge=0)


class NativeShardCompleteRequest(_StrictMessage):
    """Publish terminal evidence after every compact part was ACKed."""

    message_type: Literal["SHARD_COMPLETE"] = "SHARD_COMPLETE"
    worker_id: str
    session_id: str
    result: NativeCollectionShardResult
    sent_at_unix_ns: int = Field(ge=0)


class NativeWindowReceiptMessage(_StrictMessage):
    """Broadcast final commit/abort evidence to a worker."""

    message_type: Literal["WINDOW_RECEIPT"] = "WINDOW_RECEIPT"
    receipt: NativeCollectionWindowReceipt


class NativeArtifactRequest(_StrictMessage):
    """Pull one active-window artifact by immutable manifest identity."""

    message_type: Literal["ARTIFACT_REQUEST"] = "ARTIFACT_REQUEST"
    request_id: str
    worker_id: str
    session_id: str
    window_id: str
    artifact_id: str
    expected_manifest: NativeBfloat16ArtifactManifest


class NativePartAck(_StrictMessage):
    """ACK one exact compact part only after coordinator retention.

    ``drain_hint`` advises the streaming attempt to seal its live games
    promptly. Cutoff games are removed from ACKed parts at shard completion.
    """

    message_type: Literal["PART_ACK"] = "PART_ACK"
    part: NativeCollectionPartIdentity
    drain_hint: bool = False


class NativeProtocolAbort(_StrictMessage):
    """Fail one channel request closed without accepting its payload."""

    message_type: Literal["ABORT"] = "ABORT"
    error_code: Literal[
        "DECODE_ERROR",
        "IDENTITY_MISMATCH",
        "OLD_ATTEMPT",
        "DUPLICATE_PART",
        "SEQUENCE_MISMATCH",
        "UNKNOWN_ARTIFACT",
        "WINDOW_ABORTED",
    ]
    detail: str


def encode_message(message: BaseModel) -> bytes:
    """Encode one strict Pydantic message in a fixed top-level envelope."""
    return cast(
        bytes,
        msgpack.packb(
            {
                "schema": _SCHEMA,
                "model": type(message).__name__,
                "payload": message.model_dump(mode="json"),
            },
            use_bin_type=True,
        ),
    )


def decode_message(frame: object, expected: type[MessageT]) -> MessageT:
    """Decode one exact message class and reject duplicate/unknown fields."""
    try:
        values = _decode_envelope(frame)
    except NativeRolloutProtocolError:
        raise
    if values["schema"] != _SCHEMA or values["model"] != expected.__name__:
        raise NativeRolloutProtocolError(
            "native distributed message schema/model differs"
        )
    try:
        return expected.model_validate(values["payload"])
    except Exception as exc:
        raise NativeRolloutProtocolError(
            f"native distributed {expected.__name__} payload is invalid"
        ) from exc


def message_model_name(frame: object) -> str:
    """Return the authenticated envelope model discriminator."""
    values = _decode_envelope(frame)
    model = values["model"]
    if not isinstance(model, str) or not model:
        raise NativeRolloutProtocolError("native distributed message model is invalid")
    return model


def _decode_envelope(frame: object) -> dict[str, object]:
    try:
        view = memoryview(cast(Any, frame)).cast("B")
    except (TypeError, ValueError) as exc:
        raise NativeRolloutProtocolError(
            "native distributed message is not contiguous bytes"
        ) from exc
    if view.nbytes > _MAX_MESSAGE_BYTES:
        raise NativeRolloutProtocolError(
            "native distributed message exceeds the size limit"
        )
    try:
        raw = msgpack.unpackb(
            bytes(view),
            raw=False,
            strict_map_key=True,
            object_pairs_hook=_strict_map,
            use_list=False,
        )
    except Exception as exc:
        raise NativeRolloutProtocolError(
            "native distributed message is invalid msgpack"
        ) from exc
    if not isinstance(raw, dict) or tuple(raw) != _FIELDS:
        raise NativeRolloutProtocolError(
            "native distributed message envelope fields are invalid"
        )
    return cast(dict[str, object], raw)


def _strict_map(pairs: list[tuple[object, object]]) -> dict[object, object]:
    result: dict[object, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate msgpack map key")
        result[key] = value
    return result


__all__ = [
    "NativeArtifactRequest",
    "NativeAttemptFailedRequest",
    "NativeDrainResponse",
    "NativeHeartbeatRequest",
    "NativeLeaseResponse",
    "NativePartAck",
    "NativeProtocolAbort",
    "NativeReadyRequest",
    "NativeRegisterRequest",
    "NativeShardCompleteRequest",
    "NativeWaitResponse",
    "NativeWindowReceiptMessage",
    "NativeWorkerMetrics",
    "NativeWorkRequest",
    "decode_message",
    "encode_message",
    "message_model_name",
]
