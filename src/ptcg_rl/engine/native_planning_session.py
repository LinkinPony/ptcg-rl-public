"""Pinned native sessions for exact multi-prompt hierarchical transitions."""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from ptcg_rl.engine.native_consequence_request import nonempty_int32
from ptcg_rl.engine.native_planning_session_ffi import (
    INT_POINTER,
    NativePlanningSessionCallError,
    NativePlanningSessionLoadError,
    bind_planning_session_library,
    copy_planning_session_result,
    load_planning_session_library,
    planning_session_last_error,
    require_fingerprint_bytes,
    sha256_file,
    void_pointer_value,
)
from ptcg_rl.engine.native_planning_session_payload import (
    NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR,
    NATIVE_PLANNING_SESSION_PAYLOAD_VERSION,
    NativePlanningSessionHandle,
    NativePlanningSessionPayload,
    NativePlanningSessionPayloadError,
    NativePlanningSessionRequestKind,
    native_planning_session_abi_fingerprint,
    parse_native_planning_session_payload,
)
from ptcg_rl.engine.native_planning_session_request import (
    NativePlanningSessionCaps,
    prepare_native_planning_session_continue,
    prepare_native_planning_session_open,
)
from ptcg_rl.engine.session import HiddenInformation


@dataclass(frozen=True, slots=True)
class NativePlanningSessionTimings:
    """Disjoint Python/native timings for one session transition batch."""

    pack_seconds: float
    native_call_seconds: float
    parse_seconds: float


@dataclass(frozen=True, eq=False, slots=True)
class NativePlanningSessionBatchResult:
    """Validated response plus timing and optional root-grid dimensions."""

    payload: NativePlanningSessionPayload
    timings: NativePlanningSessionTimings
    worlds: int | None = None
    candidates: int | None = None

    @property
    def row_count(self) -> int:
        """Return request-aligned rows."""
        return self.payload.row_count

    def root_row_index(self, world_index: int, candidate_index: int) -> int:
        """Resolve candidate-major coordinates for the initial root call."""
        if self.worlds is None or self.candidates is None:
            raise ValueError("continuation results have no root-grid coordinates")
        if not 0 <= world_index < self.worlds:
            raise IndexError("world_index is outside the root grid")
        if not 0 <= candidate_index < self.candidates:
            raise IndexError("candidate_index is outside the root grid")
        return candidate_index * self.worlds + world_index


class NativePlanningSession:
    """One generation pinned to its originating native lane."""

    def __init__(
        self,
        *,
        lane: NativePlanningSessionLane,
        generation: int,
        producer_contract_fingerprint: bytes,
        root_player: int,
        initial_result: NativePlanningSessionBatchResult,
    ) -> None:
        self._lane = lane
        self.generation = generation
        self.producer_contract_fingerprint = producer_contract_fingerprint
        self.root_player = root_player
        self.initial_result = initial_result
        self._closed = False

    def __enter__(self) -> Self:
        if self.closed:
            raise RuntimeError("cannot enter a closed planning session")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    @property
    def closed(self) -> bool:
        """Return whether this generation has been invalidated."""
        with self._lane._lock:  # pylint: disable=protected-access
            return self._closed or self._lane._lane is None  # pylint: disable=protected-access

    def continue_batch(
        self,
        handles: Sequence[NativePlanningSessionHandle],
        actions: Sequence[Sequence[int]],
        *,
        caps: NativePlanningSessionCaps,
    ) -> NativePlanningSessionBatchResult:
        """Advance aligned handles while retaining strategic child handles."""
        return self._lane._continue_session(  # pylint: disable=protected-access
            self,
            handles,
            actions,
            caps=caps,
        )

    def release(self, handles: Sequence[NativePlanningSessionHandle]) -> None:
        """Release unchosen states without allowing slot identity reuse."""
        self._lane._release_handles(  # pylint: disable=protected-access
            self,
            handles,
        )

    def close(self) -> None:
        """Invalidate every remaining hidden state exactly once."""
        self._lane._close_session(self)  # pylint: disable=protected-access

    def _mark_closed(self) -> None:
        self._closed = True


class NativePlanningSessionLane:
    """RAII owner for one persistent, single-generation native state arena."""

    def __init__(self, *, library_path: Path | str | None = None) -> None:
        self._lock = threading.RLock()
        self._lib, self._library_path = load_planning_session_library(library_path)
        bind_planning_session_library(self._lib)
        if int(self._lib.CgPlannerSessionPayloadVersion()) != (
            NATIVE_PLANNING_SESSION_PAYLOAD_VERSION
        ):
            raise NativePlanningSessionLoadError(
                "native planning session payload version differs from Python"
            )
        raw_descriptor = self._lib.CgPlannerSessionAbiDescriptor()
        descriptor = (
            raw_descriptor.decode("ascii") if isinstance(raw_descriptor, bytes) else ""
        )
        if descriptor != NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR:
            raise NativePlanningSessionLoadError(
                "native planning session ABI descriptor differs from Python"
            )
        self._engine_library_fingerprint = sha256_file(self._library_path)
        self._native_abi_fingerprint = native_planning_session_abi_fingerprint(
            descriptor
        )
        raw_lane = self._lib.CgPlannerCreateSessionLane()
        pointer = void_pointer_value(raw_lane)
        if pointer is None:
            raise NativePlanningSessionLoadError(
                "native planning session lane creation failed: "
                f"{planning_session_last_error(self._lib)}"
            )
        self._lane: ctypes.c_void_p | None = ctypes.c_void_p(pointer)
        self._active_session: NativePlanningSession | None = None

    def __enter__(self) -> Self:
        with self._lock:
            if self._lane is None:
                raise RuntimeError("cannot enter a closed planning-session lane")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.close()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()

    @property
    def closed(self) -> bool:
        """Whether the native lane has been destroyed."""
        with self._lock:
            return self._lane is None

    @property
    def engine_library_fingerprint(self) -> str:
        """Return the exact loaded shared-library identity."""
        return self._engine_library_fingerprint

    @property
    def native_abi_fingerprint(self) -> str:
        """Return the v5 session contract fingerprint."""
        return self._native_abi_fingerprint

    def open_session(
        self,
        state_token: bytes | str,
        *,
        hidden_worlds: Sequence[HiddenInformation],
        candidate_actions: Sequence[Sequence[int]],
        producer_contract_fingerprint: bytes,
        root_player: int,
        manual_coin: bool = False,
        max_state_slots: int,
        caps: NativePlanningSessionCaps,
    ) -> NativePlanningSession:
        """Open one generation and execute its initial root grid."""
        pack_started = time.perf_counter()
        request = prepare_native_planning_session_open(
            state_token,
            hidden_worlds=hidden_worlds,
            candidate_actions=candidate_actions,
            root_player=root_player,
            manual_coin=manual_coin,
            max_state_slots=max_state_slots,
            caps=caps,
        )
        contract = require_fingerprint_bytes(producer_contract_fingerprint)
        hidden_counts = nonempty_int32(request.hidden.counts)
        hidden_values = nonempty_int32(request.hidden.values)
        candidate_counts = nonempty_int32(request.candidates.counts)
        candidate_values = nonempty_int32(request.candidates.values)
        pack_seconds = time.perf_counter() - pack_started
        with self._lock:
            lane = self._require_lane()
            if self._active_session is not None:
                raise RuntimeError(
                    "planning-session lane already has an active request"
                )
            try:
                native_started = time.perf_counter()
                result = self._lib.CgPlannerOpenSession(
                    lane,
                    contract,
                    len(contract),
                    request.state_token,
                    len(request.state_token),
                    hidden_counts.ctypes.data_as(INT_POINTER),
                    int(request.hidden.counts.size),
                    hidden_values.ctypes.data_as(INT_POINTER),
                    int(request.hidden.values.size),
                    request.worlds,
                    candidate_counts.ctypes.data_as(INT_POINTER),
                    int(request.candidates.counts.size),
                    candidate_values.ctypes.data_as(INT_POINTER),
                    int(request.candidates.values.size),
                    request.candidate_count,
                    request.root_player,
                    int(request.manual_coin),
                    request.max_state_slots,
                    request.caps.max_engine_steps,
                    request.caps.max_forced_steps,
                    request.caps.max_observation_bytes,
                )
                native_seconds = time.perf_counter() - native_started
                payload_bytes = copy_planning_session_result(
                    self._lib,
                    result,
                    expected_rows=request.worlds * request.candidate_count,
                    max_observation_bytes=request.caps.max_observation_bytes,
                )
                parse_started = time.perf_counter()
                payload = parse_native_planning_session_payload(
                    payload_bytes,
                    expected_kind=NativePlanningSessionRequestKind.OPEN,
                    expected_rows=request.worlds * request.candidate_count,
                    expected_root_player=request.root_player,
                    expected_raw_request_fingerprint=request.request_fingerprint,
                    expected_producer_contract_fingerprint=contract,
                    max_forced_steps=request.caps.max_forced_steps,
                )
            except Exception:
                # A malformed or ambiguous response may have retained native
                # states that Python cannot name.  Destroying the lane is the
                # only complete cleanup boundary; callers can replace it.
                self._destroy_lane_unchecked()
                raise
            batch = NativePlanningSessionBatchResult(
                payload=payload,
                timings=NativePlanningSessionTimings(
                    pack_seconds=pack_seconds,
                    native_call_seconds=native_seconds,
                    parse_seconds=time.perf_counter() - parse_started,
                ),
                worlds=request.worlds,
                candidates=request.candidate_count,
            )
            session = NativePlanningSession(
                lane=self,
                generation=payload.generation,
                producer_contract_fingerprint=contract,
                root_player=request.root_player,
                initial_result=batch,
            )
            self._active_session = session
            return session

    def close(self) -> None:
        """Invalidate a live generation and destroy the lane once."""
        with self._lock:
            lane = self._lane
            if lane is None:
                return
            session = self._active_session
            if session is not None:
                error = int(self._lib.CgPlannerCloseSession(lane, session.generation))
                session._mark_closed()  # pylint: disable=protected-access
                self._active_session = None
                if error != 0:
                    # Destruction remains the final hidden-state cleanup even
                    # when the explicit close reports a stale native session.
                    native_error = planning_session_last_error(self._lib)
                    self._destroy_lane_unchecked()
                    raise NativePlanningSessionCallError(
                        f"native planning session close failed: {native_error}"
                    )
            self._destroy_lane_unchecked()

    def _continue_session(
        self,
        session: NativePlanningSession,
        handles: Sequence[NativePlanningSessionHandle],
        actions: Sequence[Sequence[int]],
        *,
        caps: NativePlanningSessionCaps,
    ) -> NativePlanningSessionBatchResult:
        pack_started = time.perf_counter()
        request = prepare_native_planning_session_continue(
            handles,
            actions,
            caps=caps,
        )
        if request.generation != session.generation:
            raise ValueError("continuation handles differ from the active session")
        parent_slots = nonempty_int32(request.parent_slots)
        action_counts = nonempty_int32(request.actions.counts)
        action_values = nonempty_int32(request.actions.values)
        pack_seconds = time.perf_counter() - pack_started
        with self._lock:
            lane = self._require_active(session)
            try:
                native_started = time.perf_counter()
                result = self._lib.CgPlannerContinueSession(
                    lane,
                    session.producer_contract_fingerprint,
                    len(session.producer_contract_fingerprint),
                    request.generation,
                    parent_slots.ctypes.data_as(INT_POINTER),
                    request.row_count,
                    action_counts.ctypes.data_as(INT_POINTER),
                    int(request.actions.counts.size),
                    action_values.ctypes.data_as(INT_POINTER),
                    int(request.actions.values.size),
                    request.caps.max_engine_steps,
                    request.caps.max_forced_steps,
                    request.caps.max_observation_bytes,
                )
                native_seconds = time.perf_counter() - native_started
                payload_bytes = copy_planning_session_result(
                    self._lib,
                    result,
                    expected_rows=request.row_count,
                    max_observation_bytes=request.caps.max_observation_bytes,
                )
                parse_started = time.perf_counter()
                payload = parse_native_planning_session_payload(
                    payload_bytes,
                    expected_kind=NativePlanningSessionRequestKind.CONTINUE,
                    expected_rows=request.row_count,
                    expected_generation=session.generation,
                    expected_root_player=session.root_player,
                    expected_raw_request_fingerprint=request.request_fingerprint,
                    expected_producer_contract_fingerprint=(
                        session.producer_contract_fingerprint
                    ),
                    max_forced_steps=request.caps.max_forced_steps,
                )
            except Exception:
                self._destroy_lane_unchecked()
                raise
            return NativePlanningSessionBatchResult(
                payload=payload,
                timings=NativePlanningSessionTimings(
                    pack_seconds=pack_seconds,
                    native_call_seconds=native_seconds,
                    parse_seconds=time.perf_counter() - parse_started,
                ),
            )

    def _release_handles(
        self,
        session: NativePlanningSession,
        handles: Sequence[NativePlanningSessionHandle],
    ) -> None:
        frozen = tuple(handles)
        if not frozen:
            return
        if any(handle.generation != session.generation for handle in frozen):
            raise ValueError("released handles differ from the active session")
        slots_tuple = tuple(handle.state_slot for handle in frozen)
        if len(set(slots_tuple)) != len(slots_tuple):
            raise ValueError("released session handles must be unique")
        slots = np.asarray(slots_tuple, dtype=np.int32)
        with self._lock:
            lane = self._require_active(session)
            error = int(
                self._lib.CgPlannerReleaseSessionHandles(
                    lane,
                    session.generation,
                    slots.ctypes.data_as(INT_POINTER),
                    int(slots.size),
                )
            )
            if error != 0:
                native_error = planning_session_last_error(self._lib)
                self._destroy_lane_unchecked()
                raise NativePlanningSessionCallError(
                    f"native session handle release failed: {native_error}"
                )

    def _close_session(self, session: NativePlanningSession) -> None:
        with self._lock:
            if session._closed:  # pylint: disable=protected-access
                return
            lane = self._require_active(session)
            error = int(self._lib.CgPlannerCloseSession(lane, session.generation))
            if error != 0:
                native_error = planning_session_last_error(self._lib)
                self._destroy_lane_unchecked()
                raise NativePlanningSessionCallError(
                    f"native planning session close failed: {native_error}"
                )
            session._mark_closed()  # pylint: disable=protected-access
            self._active_session = None

    def _require_lane(self) -> ctypes.c_void_p:
        lane = self._lane
        if lane is None:
            raise RuntimeError("native planning-session lane is closed")
        return lane

    def _require_active(
        self,
        session: NativePlanningSession,
    ) -> ctypes.c_void_p:
        lane = self._require_lane()
        if self._active_session is not session or session._closed:  # pylint: disable=protected-access
            raise RuntimeError("native planning session is not active on this lane")
        return lane

    def _destroy_lane_unchecked(self) -> None:
        """Destroy the native arena while ``self._lock`` is held."""
        lane = self._lane
        if lane is None:
            return
        session = self._active_session
        if session is not None:
            session._mark_closed()  # pylint: disable=protected-access
        self._active_session = None
        self._lib.CgPlannerDestroySessionLane(lane)
        self._lane = None


__all__ = [
    "NativePlanningSession",
    "NativePlanningSessionBatchResult",
    "NativePlanningSessionCallError",
    "NativePlanningSessionCaps",
    "NativePlanningSessionHandle",
    "NativePlanningSessionLane",
    "NativePlanningSessionLoadError",
    "NativePlanningSessionPayloadError",
    "NativePlanningSessionTimings",
]
