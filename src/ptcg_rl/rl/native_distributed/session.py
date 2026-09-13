"""Synchronous, memory-only delivery sessions for native rollout parts."""

from __future__ import annotations

import hashlib
from typing import Any, cast

import msgpack
import zmq

from ptcg_rl.rl.native_distributed.codec import (
    DecodedNativeRolloutPart,
    decode_compact_fragment_part,
    send_compact_fragment_part,
)
from ptcg_rl.rl.native_distributed.contracts import (
    NativeRolloutLeaseIdentity,
    NativeRolloutPartIdentity,
    NativeRolloutWindowIdentity,
    NativeRolloutWorkerIdentity,
)
from ptcg_rl.rl.native_distributed.control import (
    _ABORT_ERROR_CODES,
    NativeRolloutProtocolError,
    NativeRolloutRemoteAbortError,
    NativeRolloutSessionError,
    _ControlEnvelope,
    _decode_control,
    _encode_control,
    _valid_control_id,
)
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart


class NativeRolloutTimeoutError(NativeRolloutSessionError, TimeoutError):
    """A bounded synchronous send or receive did not complete."""


class NativeRolloutPartSink:
    """Callable worker sink that releases a source part only after strict ACK."""

    def __init__(
        self,
        socket: Any,
        *,
        lease: NativeRolloutLeaseIdentity,
        first_sequence_id: int,
        timeout_ms: int,
    ) -> None:
        """Bind one connected socket to one immutable lease and sequence."""
        _validate_session_arguments(first_sequence_id, timeout_ms)
        self._socket = socket
        self._lease = lease
        self._next_sequence_id = first_sequence_id
        self._timeout_ms = timeout_ms
        self._failed = False

    @property
    def next_sequence_id(self) -> int:
        """Return the sequence assigned to the next accepted source part."""
        return self._next_sequence_id

    def __call__(self, part: CompactFragmentPart, /) -> None:
        """Send one raw part and return only after its exact ACK."""
        self._require_active()
        sequence_id = self._next_sequence_id
        identity = NativeRolloutPartIdentity(
            part_id=_part_id(self._lease, sequence_id),
            lease=self._lease,
            sequence_id=sequence_id,
            fragment_count=part.fragment_count,
            decision_count=part.decision_count,
        )
        try:
            _wait_for(self._socket, zmq.POLLOUT, self._timeout_ms)
            send_compact_fragment_part(
                self._socket,
                identity,
                part,
                flags=zmq.NOBLOCK,
                copy=False,
            )
            _wait_for(self._socket, zmq.POLLIN, self._timeout_ms)
            response = self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
            if len(response) != 1:
                raise NativeRolloutProtocolError(
                    "native rollout control response must have exactly one frame"
                )
            control = _decode_control(response[0])
            if control.message_type == "ABORT":
                raise NativeRolloutRemoteAbortError(control.error_code)
            if control != _ack_for(identity):
                raise NativeRolloutProtocolError(
                    "native rollout PART_ACK identity does not match the part"
                )
        except Exception:
            self._failed = True
            raise
        self._next_sequence_id += 1

    def _require_active(self) -> None:
        if self._failed:
            raise NativeRolloutSessionError("native rollout worker session has failed")


class NativeRolloutPartReceiver:
    """Central receiver retaining borrowed frames until learner release."""

    def __init__(
        self,
        socket: Any,
        *,
        expected_lease: NativeRolloutLeaseIdentity,
        expected_window: NativeRolloutWindowIdentity,
        expected_worker: NativeRolloutWorkerIdentity,
        first_sequence_id: int,
        timeout_ms: int,
    ) -> None:
        """Bind one connected socket to exact lease, window, and worker IDs."""
        _validate_session_arguments(first_sequence_id, timeout_ms)
        if expected_lease.window != expected_window:
            raise ValueError("native rollout expected lease/window identity differs")
        if expected_lease.worker != expected_worker:
            raise ValueError("native rollout expected lease/worker identity differs")
        self._socket = socket
        self._expected_lease = expected_lease
        self._expected_window = expected_window
        self._expected_worker = expected_worker
        self._next_sequence_id = first_sequence_id
        self._timeout_ms = timeout_ms
        self._seen_part_ids: set[str] = set()
        self._retained: dict[str, DecodedNativeRolloutPart] = {}
        self._failed = False

    @property
    def next_sequence_id(self) -> int:
        """Return the exact sequence required from the next part."""
        return self._next_sequence_id

    @property
    def retained(self) -> tuple[DecodedNativeRolloutPart, ...]:
        """Return decoded wrappers currently owned on behalf of the learner."""
        return tuple(self._retained.values())

    def receive(self) -> DecodedNativeRolloutPart:
        """Receive, validate, retain, and acknowledge exactly one part."""
        self._require_active()
        try:
            _wait_for(self._socket, zmq.POLLIN, self._timeout_ms)
            message = self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
        except Exception:
            self._failed = True
            raise

        claimed = _claimed_control_identity(message, self._expected_lease)
        try:
            if not message:
                raise _RejectedPartError(
                    "DECODE_ERROR", "native rollout multipart message is empty"
                )
            decoded = decode_compact_fragment_part(message[0], message[1:])
            self._validate(decoded.identity)
        except _RejectedPartError as exc:
            self._abort(claimed, exc.error_code)
            raise NativeRolloutProtocolError(str(exc)) from exc
        except Exception as exc:
            self._abort(claimed, "DECODE_ERROR")
            raise NativeRolloutProtocolError(
                f"native rollout part decode failed: {exc}"
            ) from exc

        identity = decoded.identity
        self._retained[identity.part_id] = decoded
        self._seen_part_ids.add(identity.part_id)
        self._next_sequence_id += 1
        try:
            _send_control(
                self._socket,
                _ack_for(identity),
                timeout_ms=self._timeout_ms,
            )
        except Exception:
            self._failed = True
            raise
        return decoded

    def release(self, part_id: str) -> None:
        """Release central ownership after the learner has consumed a part."""
        try:
            del self._retained[part_id]
        except KeyError as exc:
            raise KeyError(f"native rollout part is not retained: {part_id}") from exc

    def abort(self, error_code: str = "RECEIVER_ABORTED") -> None:
        """Abort an active request and permanently fail this receiver."""
        self._require_active()
        control = _ControlEnvelope(
            message_type="ABORT",
            lease_id=self._expected_lease.lease_id,
            part_id="unknown",
            sequence_id=self._next_sequence_id,
            error_code=error_code,
        )
        if error_code not in _ABORT_ERROR_CODES:
            raise ValueError("native rollout abort error code is invalid")
        try:
            _send_control(self._socket, control, timeout_ms=self._timeout_ms)
        finally:
            self._failed = True

    def _validate(self, identity: NativeRolloutPartIdentity) -> None:
        if (
            identity.lease != self._expected_lease
            or identity.lease.window != self._expected_window
            or identity.lease.worker != self._expected_worker
        ):
            raise _RejectedPartError(
                "IDENTITY_MISMATCH",
                "native rollout part lease/window/worker identity differs",
            )
        if identity.sequence_id != self._next_sequence_id:
            raise _RejectedPartError(
                "SEQUENCE_MISMATCH",
                "native rollout part sequence is not contiguous",
            )
        if identity.part_id in self._seen_part_ids:
            raise _RejectedPartError(
                "DUPLICATE_PART_ID",
                "native rollout part_id was already accepted",
            )
        if identity.part_id != _part_id(identity.lease, identity.sequence_id):
            raise _RejectedPartError(
                "IDENTITY_MISMATCH",
                "native rollout part_id does not match its lease and sequence",
            )

    def _abort(self, control: _ControlEnvelope, error_code: str) -> None:
        try:
            _send_control(
                self._socket,
                _ControlEnvelope(
                    message_type="ABORT",
                    lease_id=control.lease_id,
                    part_id=control.part_id,
                    sequence_id=control.sequence_id,
                    error_code=error_code,
                ),
                timeout_ms=self._timeout_ms,
            )
        finally:
            self._failed = True

    def _require_active(self) -> None:
        if self._failed:
            raise NativeRolloutSessionError(
                "native rollout receiver session has failed"
            )


class _RejectedPartError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def _part_id(lease: NativeRolloutLeaseIdentity, sequence_id: int) -> str:
    material = f"{lease.model_dump_json()}:{sequence_id}".encode()
    return hashlib.sha256(material).hexdigest()


def _ack_for(identity: NativeRolloutPartIdentity) -> _ControlEnvelope:
    return _ControlEnvelope(
        message_type="PART_ACK",
        lease_id=identity.lease.lease_id,
        part_id=identity.part_id,
        sequence_id=identity.sequence_id,
        error_code="NONE",
    )


def _send_control(
    socket: Any,
    control: _ControlEnvelope,
    *,
    timeout_ms: int,
) -> None:
    _wait_for(socket, zmq.POLLOUT, timeout_ms)
    socket.send(
        _encode_control(control),
        flags=zmq.NOBLOCK,
        copy=True,
    )


def _claimed_control_identity(
    message: list[Any],
    expected_lease: NativeRolloutLeaseIdentity,
) -> _ControlEnvelope:
    lease_id = expected_lease.lease_id
    part_id = "unknown"
    sequence_id = expected_lease.sequence_id
    if message:
        try:
            unpacked = msgpack.unpackb(
                bytes(memoryview(message[0]).cast("B")),
                raw=False,
                strict_map_key=True,
            )
            if isinstance(unpacked, dict):
                part = unpacked.get("part")
                if isinstance(part, dict):
                    raw_part_id = part.get("part_id")
                    raw_sequence_id = part.get("sequence_id")
                    lease = part.get("lease")
                    if _valid_control_id(raw_part_id):
                        part_id = cast(str, raw_part_id)
                    if type(raw_sequence_id) is int and raw_sequence_id >= 0:
                        sequence_id = raw_sequence_id
                    if isinstance(lease, dict):
                        raw_lease_id = lease.get("lease_id")
                        if _valid_control_id(raw_lease_id):
                            lease_id = cast(str, raw_lease_id)
        except (TypeError, ValueError, msgpack.UnpackException):
            pass
    return _ControlEnvelope(
        message_type="ABORT",
        lease_id=lease_id,
        part_id=part_id,
        sequence_id=sequence_id,
        error_code="PROTOCOL_ERROR",
    )


def _wait_for(socket: Any, event: int, timeout_ms: int) -> None:
    if not socket.poll(timeout=timeout_ms, flags=event) & event:
        action = "receive" if event == zmq.POLLIN else "send"
        raise NativeRolloutTimeoutError(
            f"native rollout {action} timed out after {timeout_ms} ms"
        )


def _validate_session_arguments(first_sequence_id: int, timeout_ms: int) -> None:
    if type(first_sequence_id) is not int or first_sequence_id < 0:
        raise ValueError("native rollout first sequence must be non-negative")
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("native rollout timeout must be positive")


__all__ = [
    "NativeRolloutPartReceiver",
    "NativeRolloutPartSink",
    "NativeRolloutProtocolError",
    "NativeRolloutRemoteAbortError",
    "NativeRolloutSessionError",
    "NativeRolloutTimeoutError",
]
