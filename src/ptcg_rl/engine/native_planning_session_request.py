"""Strict request packing and native-recomputed fingerprints for sessions."""

from __future__ import annotations

import hashlib
import operator
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.native_consequence_request import (
    PackedNativeRagged,
    pack_candidate_actions,
    prepare_native_consequence_request,
)
from ptcg_rl.engine.native_planning_session_payload import (
    NATIVE_PLANNING_SESSION_MAX_ENGINE_STEPS,
    NATIVE_PLANNING_SESSION_MAX_FORCED_STEPS,
    NATIVE_PLANNING_SESSION_MAX_OBSERVATION_BYTES,
    NATIVE_PLANNING_SESSION_MAX_ROWS,
    NATIVE_PLANNING_SESSION_MAX_STATE_SLOTS,
    NativePlanningSessionHandle,
)
from ptcg_rl.engine.session import HiddenInformation

_OPEN_DOMAIN = b"ptcg-rl/native-planning-session-open/v1\x00"
_CONTINUE_DOMAIN = b"ptcg-rl/native-planning-session-continue/v1\x00"

Int32Array = npt.NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class NativePlanningSessionCaps:
    """Per-call caps that are also bound into native request identity."""

    max_engine_steps: int
    max_forced_steps: int
    max_observation_bytes: int

    def __post_init__(self) -> None:
        """Validate native ABI bounds."""
        _bounded_int(
            self.max_engine_steps,
            minimum=1,
            maximum=NATIVE_PLANNING_SESSION_MAX_ENGINE_STEPS,
            name="max_engine_steps",
        )
        _bounded_int(
            self.max_forced_steps,
            minimum=0,
            maximum=NATIVE_PLANNING_SESSION_MAX_FORCED_STEPS,
            name="max_forced_steps",
        )
        _bounded_int(
            self.max_observation_bytes,
            minimum=1,
            maximum=NATIVE_PLANNING_SESSION_MAX_OBSERVATION_BYTES,
            name="max_observation_bytes",
        )


@dataclass(frozen=True, slots=True)
class PreparedNativePlanningSessionOpen:
    """Contiguous inputs for one root candidates-by-world session call."""

    state_token: bytes
    hidden: PackedNativeRagged
    candidates: PackedNativeRagged
    worlds: int
    candidate_count: int
    root_player: int
    manual_coin: bool
    max_state_slots: int
    caps: NativePlanningSessionCaps
    request_fingerprint: bytes


@dataclass(frozen=True, slots=True)
class PreparedNativePlanningSessionContinue:
    """Aligned parent handles and complete actions for one continuation batch."""

    generation: int
    parent_slots: Int32Array
    actions: PackedNativeRagged
    row_count: int
    caps: NativePlanningSessionCaps
    request_fingerprint: bytes


def prepare_native_planning_session_open(
    state_token: bytes | str,
    *,
    hidden_worlds: Sequence[HiddenInformation],
    candidate_actions: Sequence[Sequence[int]],
    root_player: int,
    manual_coin: bool,
    max_state_slots: int,
    caps: NativePlanningSessionCaps,
) -> PreparedNativePlanningSessionOpen:
    """Validate and pack one root session without retaining caller objects."""
    state_slots = _bounded_int(
        max_state_slots,
        minimum=1,
        maximum=NATIVE_PLANNING_SESSION_MAX_STATE_SLOTS,
        name="max_state_slots",
    )
    worlds = tuple(hidden_worlds)
    candidates = tuple(tuple(action) for action in candidate_actions)
    cell_count = len(worlds) * len(candidates)
    if cell_count <= 0 or cell_count > state_slots:
        raise ValueError("root candidate-by-world grid must fit max_state_slots")
    if cell_count > caps.max_engine_steps:
        raise ValueError("root grid must fit max_engine_steps")
    prepared = prepare_native_consequence_request(
        state_token,
        hidden_worlds=worlds,
        candidate_actions=candidates,
        root_player=root_player,
        manual_coin=manual_coin,
        max_cells=cell_count,
        max_engine_steps=caps.max_engine_steps,
        max_forced_steps=caps.max_forced_steps,
        max_observation_bytes=caps.max_observation_bytes,
    )
    fingerprint = native_planning_session_open_fingerprint(
        state_token=prepared.state_token,
        hidden=prepared.hidden,
        candidates=prepared.candidates,
        worlds=prepared.worlds,
        candidate_count=prepared.candidate_count,
        root_player=prepared.root_player,
        manual_coin=prepared.manual_coin,
        max_state_slots=state_slots,
        caps=caps,
    )
    return PreparedNativePlanningSessionOpen(
        state_token=prepared.state_token,
        hidden=prepared.hidden,
        candidates=prepared.candidates,
        worlds=prepared.worlds,
        candidate_count=prepared.candidate_count,
        root_player=prepared.root_player,
        manual_coin=prepared.manual_coin,
        max_state_slots=state_slots,
        caps=caps,
        request_fingerprint=fingerprint,
    )


def prepare_native_planning_session_continue(
    handles: Sequence[NativePlanningSessionHandle],
    actions: Sequence[Sequence[int]],
    *,
    caps: NativePlanningSessionCaps,
) -> PreparedNativePlanningSessionContinue:
    """Pack one request-aligned continuation expansion."""
    frozen_handles = tuple(handles)
    frozen_actions = tuple(tuple(action) for action in actions)
    if not frozen_handles or len(frozen_handles) != len(frozen_actions):
        raise ValueError("continuation handles and actions must align and be nonempty")
    if len(frozen_handles) > NATIVE_PLANNING_SESSION_MAX_ROWS:
        raise ValueError("continuation row count exceeds the native cap")
    generations = {handle.generation for handle in frozen_handles}
    if len(generations) != 1:
        raise ValueError("continuation handles must share one session generation")
    if len(frozen_handles) > caps.max_engine_steps:
        raise ValueError("continuation rows must fit max_engine_steps")
    parent_slots = _readonly_int32(
        tuple(handle.state_slot for handle in frozen_handles)
    )
    packed_actions = pack_candidate_actions(frozen_actions)
    generation = next(iter(generations))
    fingerprint = native_planning_session_continue_fingerprint(
        generation=generation,
        parent_slots=parent_slots,
        actions=packed_actions,
        caps=caps,
    )
    return PreparedNativePlanningSessionContinue(
        generation=generation,
        parent_slots=parent_slots,
        actions=packed_actions,
        row_count=len(frozen_handles),
        caps=caps,
        request_fingerprint=fingerprint,
    )


def native_planning_session_open_fingerprint(
    *,
    state_token: bytes,
    hidden: PackedNativeRagged,
    candidates: PackedNativeRagged,
    worlds: int,
    candidate_count: int,
    root_player: int,
    manual_coin: bool,
    max_state_slots: int,
    caps: NativePlanningSessionCaps,
) -> bytes:
    """Mirror the native root-session fingerprint byte for byte."""
    digest = hashlib.sha256()
    digest.update(_OPEN_DOMAIN)
    digest.update(struct.pack("<I", len(state_token)))
    digest.update(state_token)
    digest.update(
        struct.pack(
            "<7iB",
            worlds,
            candidate_count,
            root_player,
            max_state_slots,
            caps.max_engine_steps,
            caps.max_forced_steps,
            caps.max_observation_bytes,
            int(manual_coin),
        )
    )
    for values in (
        hidden.counts,
        hidden.values,
        candidates.counts,
        candidates.values,
    ):
        _update_int32_array(digest, values)
    return digest.digest()


def native_planning_session_continue_fingerprint(
    *,
    generation: int,
    parent_slots: Int32Array,
    actions: PackedNativeRagged,
    caps: NativePlanningSessionCaps,
) -> bytes:
    """Mirror the native continuation fingerprint byte for byte."""
    digest = hashlib.sha256()
    digest.update(_CONTINUE_DOMAIN)
    digest.update(
        struct.pack(
            "<5i",
            generation,
            int(parent_slots.size),
            caps.max_engine_steps,
            caps.max_forced_steps,
            caps.max_observation_bytes,
        )
    )
    for values in (parent_slots, actions.counts, actions.values):
        _update_int32_array(digest, values)
    return digest.digest()


def _update_int32_array(digest: Any, values: Int32Array) -> None:
    digest.update(struct.pack("<I", int(values.size)))
    digest.update(values.astype("<i4", copy=False).tobytes())


def _readonly_int32(values: Sequence[int]) -> Int32Array:
    array = np.asarray(values, dtype=np.int32)
    array.setflags(write=False)
    return array


def _bounded_int(value: Any, *, minimum: int, maximum: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return int(parsed)


__all__ = [
    "NativePlanningSessionCaps",
    "PreparedNativePlanningSessionContinue",
    "PreparedNativePlanningSessionOpen",
    "native_planning_session_continue_fingerprint",
    "native_planning_session_open_fingerprint",
    "prepare_native_planning_session_continue",
    "prepare_native_planning_session_open",
]
