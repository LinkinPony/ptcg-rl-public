"""Low-level ctypes binding for the native planning-session ABI."""

from __future__ import annotations

import ctypes
import hashlib
import operator
import os
import threading
from pathlib import Path
from typing import Any

from ptcg_rl.engine.native_planning_session_payload import (
    NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES,
    NATIVE_PLANNING_SESSION_METADATA_WIDTH,
    NativePlanningSessionPayloadError,
)

_HEADER_BYTES = 8 * 4 + 2 * NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES
_BIND_LOCK = threading.Lock()

INT_POINTER = ctypes.POINTER(ctypes.c_int)


class NativePlanningSessionLoadError(RuntimeError):
    """Raised when the v5 planning-session ABI cannot be loaded."""


class NativePlanningSessionCallError(RuntimeError):
    """Raised when native session lifecycle or execution fails."""


class CgPlannerSessionResult(ctypes.Structure):
    """Owned native response buffer returned by an ABI call."""

    _fields_ = [
        ("error", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_int),
    ]


def copy_planning_session_result(
    library: Any,
    result: CgPlannerSessionResult,
    *,
    expected_rows: int,
    max_observation_bytes: int,
) -> bytes:
    """Copy and free one bounded native result buffer."""
    if int(result.error) != 0:
        _free_result(library, result)
        raise NativePlanningSessionCallError(
            "native planning session call failed with "
            f"error={int(result.error)}: {planning_session_last_error(library)}"
        )
    if result.data is None or int(result.size) <= 0:
        _free_result(library, result)
        raise NativePlanningSessionCallError(
            "native planning session returned no payload"
        )
    maximum = (
        _HEADER_BYTES
        + expected_rows * NATIVE_PLANNING_SESSION_METADATA_WIDTH * 4
        + max_observation_bytes
    )
    if int(result.size) > maximum:
        _free_result(library, result)
        raise NativePlanningSessionPayloadError(
            "native planning session payload exceeds request capacity"
        )
    try:
        return ctypes.string_at(result.data, int(result.size))
    finally:
        library.CgPlannerSessionFree(result.data)


def bind_planning_session_library(library: Any) -> None:
    """Bind exact argument and return types once per process."""
    with _BIND_LOCK:
        names = (
            "CgPlannerSessionPayloadVersion",
            "CgPlannerSessionAbiDescriptor",
            "CgPlannerSessionLastError",
            "CgPlannerCreateSessionLane",
            "CgPlannerDestroySessionLane",
            "CgPlannerOpenSession",
            "CgPlannerContinueSession",
            "CgPlannerReleaseSessionHandles",
            "CgPlannerCloseSession",
            "CgPlannerSessionFree",
        )
        try:
            (
                payload_version,
                abi_descriptor,
                last_error,
                create_lane,
                destroy_lane,
                open_session,
                continue_session,
                release_handles,
                close_session,
                free_payload,
            ) = (getattr(library, name) for name in names)
        except AttributeError as exc:
            raise NativePlanningSessionLoadError(
                "native library has no v5 planning-session ABI; rebuild it"
            ) from exc
        payload_version.argtypes = []
        payload_version.restype = ctypes.c_int
        abi_descriptor.argtypes = []
        abi_descriptor.restype = ctypes.c_char_p
        last_error.argtypes = []
        last_error.restype = ctypes.c_char_p
        create_lane.argtypes = []
        create_lane.restype = ctypes.c_void_p
        destroy_lane.argtypes = [ctypes.c_void_p]
        destroy_lane.restype = None
        open_session.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        open_session.restype = CgPlannerSessionResult
        continue_session.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        continue_session.restype = CgPlannerSessionResult
        release_handles.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            INT_POINTER,
            ctypes.c_int,
        ]
        release_handles.restype = ctypes.c_int
        close_session.argtypes = [ctypes.c_void_p, ctypes.c_int]
        close_session.restype = ctypes.c_int
        free_payload.argtypes = [ctypes.c_void_p]
        free_payload.restype = None


def load_planning_session_library(
    library_path: Path | str | None,
) -> tuple[Any, Path]:
    """Load an explicit or repository-local native probe library."""
    candidates: list[Path] = []
    if library_path is not None:
        candidates.append(Path(library_path))
    else:
        env_path = os.environ.get("PTCG_RL_CG_PROBE_LIB")
        if env_path:
            candidates.append(Path(env_path))
        repo_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            (
                repo_root / "src" / "native" / "cg_probe" / "libcg_probe.so",
                Path.cwd() / "src" / "native" / "cg_probe" / "libcg_probe.so",
            )
        )
    attempted: list[str] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        attempted.append(str(resolved))
        if not resolved.exists():
            continue
        try:
            return ctypes.CDLL(str(resolved)), resolved.resolve()
        except OSError as exc:
            attempted[-1] = f"{resolved}: {exc}"
    raise NativePlanningSessionLoadError(
        "native planning-session library not found or loadable; tried "
        + ", ".join(attempted)
    )


def planning_session_last_error(library: Any) -> str:
    """Read the thread-local native error string."""
    raw = library.CgPlannerSessionLastError()
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else ""


def require_fingerprint_bytes(value: bytes) -> bytes:
    """Validate one raw SHA-256 identity."""
    if not isinstance(value, bytes):
        raise TypeError("producer_contract_fingerprint must be bytes")
    if len(value) != NATIVE_PLANNING_SESSION_FINGERPRINT_BYTES:
        raise ValueError("producer_contract_fingerprint must contain 32 bytes")
    return value


def sha256_file(path: Path) -> str:
    """Stream-hash an immutable native library."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def void_pointer_value(raw_pointer: Any) -> int | None:
    """Normalize a ctypes or integer pointer without accepting zero."""
    if isinstance(raw_pointer, ctypes.c_void_p):
        return raw_pointer.value
    if raw_pointer is None:
        return None
    value = operator.index(raw_pointer)
    return value if value != 0 else None


def _free_result(library: Any, result: CgPlannerSessionResult) -> None:
    if result.data is not None:
        library.CgPlannerSessionFree(result.data)


__all__ = [
    "INT_POINTER",
    "CgPlannerSessionResult",
    "NativePlanningSessionCallError",
    "NativePlanningSessionLoadError",
    "bind_planning_session_library",
    "copy_planning_session_result",
    "load_planning_session_library",
    "planning_session_last_error",
    "require_fingerprint_bytes",
    "sha256_file",
    "void_pointer_value",
]
