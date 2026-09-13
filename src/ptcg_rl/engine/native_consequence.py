"""Compact lane-based native consequence execution.

The wrapper owns one native planner lane and exposes the lane payload without
materializing one Python object per candidate-by-world cell.  Native output is
copied once into immutable Python-owned bytes; metadata and observation slices
are then zero-copy NumPy/memoryview views over that storage.
"""

from __future__ import annotations

import ctypes
import hashlib
import operator
import os
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.native_consequence_payload import (
    NATIVE_CONSEQUENCE_ABI_DESCRIPTOR,
    NATIVE_CONSEQUENCE_CELL_ORDER,
    NATIVE_CONSEQUENCE_FINGERPRINT_BYTES,
    NATIVE_CONSEQUENCE_MAGIC,
    NATIVE_CONSEQUENCE_MAX_CELLS,
    NATIVE_CONSEQUENCE_MAX_ENGINE_STEPS,
    NATIVE_CONSEQUENCE_MAX_FORCED_STEPS,
    NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES,
    NATIVE_CONSEQUENCE_METADATA_WIDTH,
    NATIVE_CONSEQUENCE_PAYLOAD_VERSION,
    NATIVE_CONSEQUENCE_REQUEST_FINGERPRINT_BYTES,
    NativeConsequenceEndpoint,
    NativeConsequenceMetadataColumn,
    NativeConsequencePayload,
    NativeConsequencePayloadError,
    native_consequence_abi_fingerprint,
    parse_native_consequence_payload,
)
from ptcg_rl.engine.native_consequence_request import (
    NATIVE_CONSEQUENCE_MAX_SELECT_COUNT,
    NATIVE_CONSEQUENCE_MAX_STATE_TOKEN_BYTES,
    PackedNativeRagged,
    nonempty_int32,
    pack_candidate_actions,
    pack_hidden_worlds,
    prepare_native_consequence_request,
)
from ptcg_rl.engine.session import HiddenInformation

_HEADER_WIDTH = 6
_INT32_BYTES = 4
_INT_POINTER = ctypes.POINTER(ctypes.c_int)
_LIBRARY_BIND_LOCK = threading.Lock()


class NativeConsequenceLoadError(RuntimeError):
    """Raised when the lane-based native planner cannot be loaded."""


class NativeConsequenceCallError(RuntimeError):
    """Raised when the native planner rejects or fails one request."""


@dataclass(frozen=True)
class NativeConsequenceTimings:
    """Disjoint wall-clock timings for one lane request."""

    pack_seconds: float
    native_call_seconds: float
    parse_seconds: float


@dataclass(frozen=True, eq=False)
class NativeConsequenceBatchResult:
    """Compact native consequence output and stage timings."""

    payload: NativeConsequencePayload
    timings: NativeConsequenceTimings

    @property
    def worlds(self) -> int:
        """Return the number of hidden worlds in the result."""
        return self.payload.worlds

    @property
    def candidates(self) -> int:
        """Return the number of candidate actions in the result."""
        return self.payload.candidates

    @property
    def world_count(self) -> int:
        """Return the number of hidden worlds in the result."""
        return self.payload.worlds

    @property
    def candidate_count(self) -> int:
        """Return the number of candidate actions in the result."""
        return self.payload.candidates

    @property
    def cell_count(self) -> int:
        """Return the number of candidate-major result cells."""
        return self.payload.cell_count

    @property
    def metadata(self) -> npt.NDArray[np.int32]:
        """Return the zero-copy cell metadata matrix."""
        return self.payload.metadata

    @property
    def observation_blob(self) -> npt.NDArray[np.uint8]:
        """Return the zero-copy concatenated observation JSON blob."""
        return self.payload.observation_blob

    @property
    def payload_bytes(self) -> int:
        """Return total wire payload bytes."""
        return self.payload.payload_bytes

    @property
    def pack_seconds(self) -> float:
        """Return request validation and input packing time."""
        return self.timings.pack_seconds

    @property
    def native_call_seconds(self) -> float:
        """Return time spent inside the native decision batch call."""
        return self.timings.native_call_seconds

    @property
    def parse_seconds(self) -> float:
        """Return native-output copy and payload validation time."""
        return self.timings.parse_seconds

    def decode_observation_row(self, row_index: int) -> Mapping[str, Any] | None:
        """Decode one candidate-major observation row."""
        return self.payload.decode_observation_row(row_index)

    def decode_observation(
        self,
        world_index: int,
        candidate_index: int,
    ) -> Mapping[str, Any] | None:
        """Decode one world/candidate observation row."""
        return self.payload.decode_observation(world_index, candidate_index)

    def decode_leaf_observation_row(
        self,
        row_index: int,
    ) -> Mapping[str, Any] | None:
        """Decode one row from the actual nonterminal leaf actor's view."""
        return self.payload.decode_leaf_observation_row(row_index)

    def decode_leaf_observation(
        self,
        world_index: int,
        candidate_index: int,
    ) -> Mapping[str, Any] | None:
        """Decode one grid cell from the actual nonterminal leaf actor's view."""
        return self.payload.decode_leaf_observation(
            world_index,
            candidate_index,
        )


class _CgPlannerResult(ctypes.Structure):
    _fields_ = [
        ("error", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_int),
    ]


class NativeConsequenceLane:
    """RAII owner for one reusable native planner lane."""

    def __init__(self, *, library_path: Path | str | None = None) -> None:
        """Load the native planner ABI and create one isolated lane."""
        self._lock = threading.RLock()
        self._lib, self._library_path = _load_library(library_path)
        _bind_library(self._lib)
        if int(self._lib.CgPlannerPayloadVersion()) != (
            NATIVE_CONSEQUENCE_PAYLOAD_VERSION
        ):
            raise NativeConsequenceLoadError(
                "native consequence payload version does not match Python"
            )
        raw_descriptor = self._lib.CgPlannerAbiDescriptor()
        descriptor = (
            raw_descriptor.decode("ascii")
            if isinstance(raw_descriptor, bytes)
            else ""
        )
        if descriptor != NATIVE_CONSEQUENCE_ABI_DESCRIPTOR:
            raise NativeConsequenceLoadError(
                "native consequence ABI descriptor does not match Python"
            )
        self._engine_library_fingerprint = _file_sha256(self._library_path)
        self._native_abi_fingerprint = native_consequence_abi_fingerprint(
            descriptor
        )
        raw_lane = self._lib.CgPlannerCreateLane()
        pointer_value = _void_pointer_value(raw_lane)
        if pointer_value is None:
            raise NativeConsequenceLoadError(
                "native consequence lane creation failed: "
                f"{_last_error(self._lib)}"
            )
        self._lane: ctypes.c_void_p | None = ctypes.c_void_p(pointer_value)

    @property
    def closed(self) -> bool:
        """Whether the native lane has been destroyed."""
        with self._lock:
            return self._lane is None

    @property
    def engine_library_fingerprint(self) -> str:
        """Return the immutable loaded native-library SHA-256 identity."""
        return self._engine_library_fingerprint

    @property
    def native_abi_fingerprint(self) -> str:
        """Return the validated wire and observation-schema identity."""
        return self._native_abi_fingerprint

    def __enter__(self) -> Self:
        """Return this lane for a context-managed request scope."""
        with self._lock:
            if self._lane is None:
                raise RuntimeError("cannot enter a closed native consequence lane")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Destroy the lane when leaving a context manager."""
        del exc_type, exc, traceback
        self.close()

    def __del__(self) -> None:
        """Best-effort native cleanup for callers that omit ``close``."""
        with suppress(Exception):
            self.close()

    def close(self) -> None:
        """Destroy the native lane exactly once, including concurrent calls."""
        with self._lock:
            lane = self._lane
            if lane is None:
                return
            self._lane = None
            self._lib.CgPlannerDestroyLane(lane)

    def run(
        self,
        state_token: bytes | str,
        *,
        hidden_worlds: Sequence[HiddenInformation],
        candidate_actions: Sequence[Sequence[int]],
        producer_contract_fingerprint: bytes,
        root_player: int,
        manual_coin: bool = False,
        stochastic_seed: int = 0,
        max_cells: int,
        max_engine_steps: int,
        max_forced_steps: int,
        max_observation_bytes: int,
    ) -> NativeConsequenceBatchResult:
        """Execute a bounded candidate-by-world consequence grid."""
        with self._lock:
            if self._lane is None:
                raise RuntimeError("cannot run a closed native consequence lane")

            pack_started = time.perf_counter()
            request = prepare_native_consequence_request(
                state_token,
                hidden_worlds=hidden_worlds,
                candidate_actions=candidate_actions,
                root_player=root_player,
                manual_coin=manual_coin,
                stochastic_seed=stochastic_seed,
                max_cells=max_cells,
                max_engine_steps=max_engine_steps,
                max_forced_steps=max_forced_steps,
                max_observation_bytes=max_observation_bytes,
            )
            contract_fingerprint = _fingerprint_bytes(
                producer_contract_fingerprint,
                name="producer_contract_fingerprint",
            )
            hidden_counts = nonempty_int32(request.hidden.counts)
            hidden_values = nonempty_int32(request.hidden.values)
            candidate_counts = nonempty_int32(request.candidates.counts)
            candidate_values = nonempty_int32(request.candidates.values)
            pack_seconds = time.perf_counter() - pack_started
            native_started = time.perf_counter()
            result = self._lib.CgPlannerDecisionBatch(
                self._lane,
                contract_fingerprint,
                len(contract_fingerprint),
                request.state_token,
                len(request.state_token),
                hidden_counts.ctypes.data_as(_INT_POINTER),
                int(request.hidden.counts.size),
                hidden_values.ctypes.data_as(_INT_POINTER),
                int(request.hidden.values.size),
                request.worlds,
                candidate_counts.ctypes.data_as(_INT_POINTER),
                int(request.candidates.counts.size),
                candidate_values.ctypes.data_as(_INT_POINTER),
                int(request.candidates.values.size),
                request.candidate_count,
                request.root_player,
                int(request.manual_coin),
                request.stochastic_seed,
                request.max_cells,
                request.max_engine_steps,
                request.max_forced_steps,
                request.max_observation_bytes,
            )
            native_call_seconds = time.perf_counter() - native_started
            parse_started = time.perf_counter()
            if int(result.error) != 0:
                _free_native_result(self._lib, result)
                raise NativeConsequenceCallError(
                    "native consequence batch failed with "
                    f"error={int(result.error)}: {_last_error(self._lib)}"
                )
            if result.data is None or int(result.size) <= 0:
                _free_native_result(self._lib, result)
                raise NativeConsequenceCallError(
                    "native consequence batch returned no payload"
                )

            payload_bytes = int(result.size)
            maximum_payload_bytes = (
                _HEADER_WIDTH * _INT32_BYTES
                + 2 * NATIVE_CONSEQUENCE_FINGERPRINT_BYTES
                + request.worlds
                * request.candidate_count
                * NATIVE_CONSEQUENCE_METADATA_WIDTH
                * _INT32_BYTES
                + request.max_observation_bytes
            )
            if payload_bytes > maximum_payload_bytes:
                _free_native_result(self._lib, result)
                raise NativeConsequencePayloadError(
                    "native consequence payload exceeds the request capacity"
                )
            try:
                payload_storage = ctypes.string_at(result.data, payload_bytes)
            finally:
                self._lib.CgPlannerFree(result.data)
            payload = parse_native_consequence_payload(
                payload_storage,
                expected_worlds=request.worlds,
                expected_candidates=request.candidate_count,
                expected_root_player=request.root_player,
                expected_raw_request_fingerprint=request.request_fingerprint,
                expected_producer_contract_fingerprint=contract_fingerprint,
                max_forced_steps=request.max_forced_steps,
            )
            parse_seconds = time.perf_counter() - parse_started

            return NativeConsequenceBatchResult(
                payload=payload,
                timings=NativeConsequenceTimings(
                    pack_seconds=pack_seconds,
                    native_call_seconds=native_call_seconds,
                    parse_seconds=parse_seconds,
                ),
            )


def _bind_library(library: Any) -> None:
    with _LIBRARY_BIND_LOCK:
        try:
            payload_version = library.CgPlannerPayloadVersion
            abi_descriptor = library.CgPlannerAbiDescriptor
            last_error = library.CgPlannerLastError
            create_lane = library.CgPlannerCreateLane
            destroy_lane = library.CgPlannerDestroyLane
            decision_batch = library.CgPlannerDecisionBatch
            free_payload = library.CgPlannerFree
        except AttributeError as exc:
            raise NativeConsequenceLoadError(
                "native probe library has no planner lane ABI; rebuild it"
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
        decision_batch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            _INT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            ctypes.c_int,
            ctypes.c_int,
            _INT_POINTER,
            ctypes.c_int,
            _INT_POINTER,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        decision_batch.restype = _CgPlannerResult
        free_payload.argtypes = [ctypes.c_void_p]
        free_payload.restype = None


def _load_library(library_path: Path | str | None) -> tuple[Any, Path]:
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
    raise NativeConsequenceLoadError(
        "native consequence library not found or not loadable; tried "
        + ", ".join(attempted)
    )


def _last_error(library: Any) -> str:
    raw = library.CgPlannerLastError()
    return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fingerprint_bytes(value: bytes, *, name: str) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError(f"{name} must be bytes")
    if len(value) != NATIVE_CONSEQUENCE_FINGERPRINT_BYTES:
        raise ValueError(f"{name} must contain 32 bytes")
    return value


def _free_native_result(library: Any, result: _CgPlannerResult) -> None:
    if result.data is not None:
        library.CgPlannerFree(result.data)


def _void_pointer_value(raw_pointer: Any) -> int | None:
    if isinstance(raw_pointer, ctypes.c_void_p):
        return raw_pointer.value
    if raw_pointer is None:
        return None
    value = operator.index(raw_pointer)
    return value if value != 0 else None


__all__ = [
    "NATIVE_CONSEQUENCE_MAGIC",
    "NATIVE_CONSEQUENCE_CELL_ORDER",
    "NATIVE_CONSEQUENCE_ABI_DESCRIPTOR",
    "NATIVE_CONSEQUENCE_MAX_CELLS",
    "NATIVE_CONSEQUENCE_MAX_ENGINE_STEPS",
    "NATIVE_CONSEQUENCE_MAX_FORCED_STEPS",
    "NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES",
    "NATIVE_CONSEQUENCE_MAX_SELECT_COUNT",
    "NATIVE_CONSEQUENCE_MAX_STATE_TOKEN_BYTES",
    "NATIVE_CONSEQUENCE_METADATA_WIDTH",
    "NATIVE_CONSEQUENCE_PAYLOAD_VERSION",
    "NATIVE_CONSEQUENCE_REQUEST_FINGERPRINT_BYTES",
    "NativeConsequenceBatchResult",
    "NativeConsequenceCallError",
    "NativeConsequenceEndpoint",
    "NativeConsequenceLane",
    "NativeConsequenceLoadError",
    "NativeConsequenceMetadataColumn",
    "NativeConsequencePayload",
    "NativeConsequencePayloadError",
    "NativeConsequenceTimings",
    "PackedNativeRagged",
    "pack_candidate_actions",
    "pack_hidden_worlds",
    "parse_native_consequence_payload",
]
