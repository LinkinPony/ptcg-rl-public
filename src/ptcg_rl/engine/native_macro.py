"""Native exact complete-macro execution through the bundled engine."""

from __future__ import annotations

import ctypes
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.engine.native_probe_payload import (
    NativeProbeTransition,
    parse_native_probe_payload,
)
from ptcg_rl.engine.session import HiddenInformation

NativeAction = tuple[int, ...]
NativeMacro = tuple[NativeAction, ...]
_PAYLOAD_VERSION = 2
_HIDDEN_LIST_COUNT = 6
# Per-transition error returned when a supplied macro contains an action after
# the root semantic boundary.  The native backend stops before that action.
NATIVE_MACRO_TRAILING_ACTION_ERROR = 1001


class NativeMacroLoadError(RuntimeError):
    """Raised when the native complete-macro entry point cannot be loaded."""


class _CgProbeResult(ctypes.Structure):
    _fields_ = [
        ("error", ctypes.c_int),
        ("data", ctypes.c_void_p),
        ("size", ctypes.c_int),
    ]


@dataclass(frozen=True)
class NativeMacroBatchResult:
    """Exact transitions and timing for one native macro batch."""

    transitions: tuple[NativeProbeTransition, ...]
    native_call_seconds: float
    payload_bytes: int
    worlds: int
    macro_count: int


class NativeMacroBackend:
    """Execute supplied complete action sequences without Search API JSON."""

    def __init__(self, *, library_path: Path | str | None = None) -> None:
        """Load the native benchmark library and bind its stable C ABI."""
        self._lib = _load_library(library_path)
        self._lib.CgProbeInitialize.argtypes = []
        self._lib.CgProbeInitialize.restype = None
        self._lib.CgProbeLastError.argtypes = []
        self._lib.CgProbeLastError.restype = ctypes.c_char_p
        self._lib.CgProbePayloadVersion.argtypes = []
        self._lib.CgProbePayloadVersion.restype = ctypes.c_int
        self._lib.CgProbeFree.argtypes = [ctypes.c_void_p]
        self._lib.CgProbeFree.restype = None
        try:
            macro_batch = self._lib.CgProbeMacroBatch
        except AttributeError as exc:
            raise NativeMacroLoadError(
                "native probe library has no complete-macro entry point"
            ) from exc
        macro_batch.argtypes = [
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        macro_batch.restype = _CgProbeResult
        if int(self._lib.CgProbePayloadVersion()) != _PAYLOAD_VERSION:
            raise NativeMacroLoadError("native macro payload version mismatch")
        self._lib.CgProbeInitialize()

    def run(
        self,
        state_token: bytes | str,
        *,
        hidden_worlds: Sequence[HiddenInformation],
        macros: Sequence[Sequence[Sequence[int]]],
        manual_coin: bool = False,
        complete_to_boundary: bool = False,
    ) -> NativeMacroBatchResult:
        """Execute all macros for every supplied determinized world."""
        token = (
            state_token.encode("ascii")
            if isinstance(state_token, str)
            else bytes(state_token)
        )
        canonical_worlds = tuple(hidden_worlds)
        canonical_macros = _canonical_macros(macros)
        if not token:
            raise ValueError("state token must not be empty")
        if not canonical_worlds:
            raise ValueError("hidden_worlds must not be empty")
        if not canonical_macros:
            raise ValueError("macros must not be empty")

        hidden_counts, hidden_values = _hidden_inputs(canonical_worlds)
        macro_step_counts, step_select_counts, step_select_values = _macro_inputs(
            canonical_macros
        )
        started = time.perf_counter()
        result = self._lib.CgProbeMacroBatch(
            token,
            len(token),
            _int_array(hidden_counts),
            _int_array(hidden_values),
            len(canonical_worlds),
            _int_array(macro_step_counts),
            _int_array(step_select_counts),
            _int_array(step_select_values),
            len(canonical_macros),
            int(manual_coin),
            int(complete_to_boundary),
        )
        native_call_seconds = time.perf_counter() - started
        if result.error != 0:
            raise RuntimeError(
                f"native macro batch failed with error={result.error}: "
                f"{self._last_error()}"
            )
        if result.data is None or result.size <= 0:
            raise RuntimeError("native macro batch returned no payload")
        payload_bytes = int(result.size)
        try:
            payload = ctypes.string_at(result.data, payload_bytes)
        finally:
            self._lib.CgProbeFree(result.data)
        transitions = parse_native_probe_payload(
            payload,
            expected_worlds=len(canonical_worlds),
            expected_candidates=len(canonical_macros),
        )
        return NativeMacroBatchResult(
            transitions=transitions,
            native_call_seconds=native_call_seconds,
            payload_bytes=payload_bytes,
            worlds=len(canonical_worlds),
            macro_count=len(canonical_macros),
        )

    def _last_error(self) -> str:
        raw = self._lib.CgProbeLastError()
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else ""


def _canonical_macros(
    macros: Sequence[Sequence[Sequence[int]]],
) -> tuple[NativeMacro, ...]:
    canonical = tuple(
        tuple(tuple(int(index) for index in action) for action in macro)
        for macro in macros
    )
    if any(not macro for macro in canonical):
        raise ValueError("every macro must contain at least one action")
    return canonical


def _hidden_inputs(
    worlds: Sequence[HiddenInformation],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    counts: list[int] = []
    values: list[int] = []
    for hidden in worlds:
        lists = (
            hidden.your_deck,
            hidden.your_prize,
            hidden.opponent_deck,
            hidden.opponent_prize,
            hidden.opponent_hand,
            hidden.opponent_active,
        )
        if len(lists) != _HIDDEN_LIST_COUNT:
            raise RuntimeError("internal hidden-list count mismatch")
        for cards in lists:
            counts.append(len(cards))
            values.extend(int(card_id) for card_id in cards)
    return tuple(counts), tuple(values)


def _macro_inputs(
    macros: Sequence[NativeMacro],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    macro_step_counts: list[int] = []
    step_select_counts: list[int] = []
    step_select_values: list[int] = []
    for macro in macros:
        macro_step_counts.append(len(macro))
        for action in macro:
            step_select_counts.append(len(action))
            step_select_values.extend(action)
    return (
        tuple(macro_step_counts),
        tuple(step_select_counts),
        tuple(step_select_values),
    )


def _int_array(values: Sequence[int]) -> Any:
    materialized = tuple(int(value) for value in values)
    if not materialized:
        materialized = (0,)
    return (ctypes.c_int * len(materialized))(*materialized)


def _load_library(library_path: Path | str | None) -> Any:
    candidates: list[Path] = []
    if library_path is not None:
        candidates.append(Path(library_path))
    env_path = os.environ.get("PTCG_RL_CG_PROBE_LIB")
    if env_path:
        candidates.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[3]
    candidates.append(repo_root / "src" / "native" / "cg_probe" / "libcg_probe.so")
    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        if not candidate.exists():
            continue
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as exc:
            attempted[-1] = f"{candidate}: {exc}"
    raise NativeMacroLoadError(
        "native macro library not found or not loadable; tried "
        + ", ".join(attempted)
    )


__all__ = [
    "NATIVE_MACRO_TRAILING_ACTION_ERROR",
    "NativeAction",
    "NativeMacro",
    "NativeMacroBackend",
    "NativeMacroBatchResult",
    "NativeMacroLoadError",
]
