"""Strict request packing for the native consequence lane ABI."""

from __future__ import annotations

import hashlib
import operator
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from ptcg_rl.decks.identity import DECK_SIZE
from ptcg_rl.engine.native_consequence_payload import (
    NATIVE_CONSEQUENCE_MAX_CELLS,
    NATIVE_CONSEQUENCE_MAX_ENGINE_STEPS,
    NATIVE_CONSEQUENCE_MAX_FORCED_STEPS,
    NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES,
)
from ptcg_rl.engine.session import HiddenInformation

NATIVE_CONSEQUENCE_MAX_SELECT_COUNT = 128
NATIVE_CONSEQUENCE_MAX_STATE_TOKEN_BYTES = 1 << 25
NATIVE_CONSEQUENCE_REQUEST_FINGERPRINT_BYTES = 32

_HIDDEN_LIST_COUNT = 6
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1
_UINT64_MAX = (1 << 64) - 1
_STATE_TOKEN_ALPHABET = frozenset(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=-*"
)
_REQUEST_FINGERPRINT_DOMAIN = b"ptcg-rl/native-consequence-request/v5\x00"


@dataclass(frozen=True, eq=False)
class PackedNativeRagged:
    """Contiguous int32 counts and values for one ragged native argument."""

    counts: npt.NDArray[np.int32]
    values: npt.NDArray[np.int32]


@dataclass(frozen=True, eq=False)
class PreparedNativeConsequenceRequest:
    """Validated, contiguous arguments for one native lane call."""

    state_token: bytes
    hidden: PackedNativeRagged
    candidates: PackedNativeRagged
    worlds: int
    candidate_count: int
    root_player: int
    manual_coin: bool
    stochastic_seed: int
    max_cells: int
    max_engine_steps: int
    max_forced_steps: int
    max_observation_bytes: int
    request_fingerprint: bytes


def prepare_native_consequence_request(
    state_token: bytes | str,
    *,
    hidden_worlds: Sequence[HiddenInformation],
    candidate_actions: Sequence[Sequence[int]],
    root_player: int,
    manual_coin: bool,
    stochastic_seed: int = 0,
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
) -> PreparedNativeConsequenceRequest:
    """Validate caps before materializing ragged world and action buffers."""
    token = _state_token(state_token)
    root = _bounded_int(root_player, minimum=0, maximum=1, name="root_player")
    if not isinstance(manual_coin, bool):
        raise TypeError("manual_coin must be bool")
    seed = _strict_uint64(stochastic_seed, "stochastic_seed")
    cell_cap = _bounded_int(
        max_cells,
        minimum=1,
        maximum=NATIVE_CONSEQUENCE_MAX_CELLS,
        name="max_cells",
    )
    engine_step_cap = _bounded_int(
        max_engine_steps,
        minimum=1,
        maximum=NATIVE_CONSEQUENCE_MAX_ENGINE_STEPS,
        name="max_engine_steps",
    )
    forced_cap = _bounded_int(
        max_forced_steps,
        minimum=0,
        maximum=NATIVE_CONSEQUENCE_MAX_FORCED_STEPS,
        name="max_forced_steps",
    )
    observation_cap = _bounded_int(
        max_observation_bytes,
        minimum=1,
        maximum=NATIVE_CONSEQUENCE_MAX_OBSERVATION_BYTES,
        name="max_observation_bytes",
    )
    worlds_input = tuple(hidden_worlds)
    candidates_input = tuple(candidate_actions)
    worlds = len(worlds_input)
    candidate_count = len(candidates_input)
    if worlds <= 0:
        raise ValueError("hidden_worlds must not be empty")
    if candidate_count <= 0:
        raise ValueError("candidate_actions must not be empty")
    cell_count = worlds * candidate_count
    if cell_count > cell_cap:
        raise ValueError(
            "candidate-by-world grid exceeds max_cells before native call"
        )
    if cell_count > engine_step_cap:
        raise ValueError(
            "candidate-by-world grid exceeds max_engine_steps before native call"
        )
    hidden = pack_hidden_worlds(worlds_input)
    candidates = pack_candidate_actions(candidates_input)
    request_fingerprint = native_consequence_request_fingerprint(
        state_token=token,
        hidden=hidden,
        candidates=candidates,
        worlds=worlds,
        candidate_count=candidate_count,
        root_player=root,
        manual_coin=manual_coin,
        stochastic_seed=seed,
        max_cells=cell_cap,
        max_engine_steps=engine_step_cap,
        max_forced_steps=forced_cap,
        max_observation_bytes=observation_cap,
    )
    return PreparedNativeConsequenceRequest(
        state_token=token,
        hidden=hidden,
        candidates=candidates,
        worlds=worlds,
        candidate_count=candidate_count,
        root_player=root,
        manual_coin=manual_coin,
        stochastic_seed=seed,
        max_cells=cell_cap,
        max_engine_steps=engine_step_cap,
        max_forced_steps=forced_cap,
        max_observation_bytes=observation_cap,
        request_fingerprint=request_fingerprint,
    )


def native_consequence_request_fingerprint(
    *,
    state_token: bytes,
    hidden: PackedNativeRagged,
    candidates: PackedNativeRagged,
    worlds: int,
    candidate_count: int,
    root_player: int,
    manual_coin: bool,
    stochastic_seed: int = 0,
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
) -> bytes:
    """Bind all raw native request inputs to one fixed-width digest."""
    digest = hashlib.sha256()
    digest.update(_REQUEST_FINGERPRINT_DOMAIN)
    digest.update(struct.pack("<I", len(state_token)))
    digest.update(state_token)
    digest.update(
        struct.pack(
            "<7i?",
            worlds,
            candidate_count,
            root_player,
            max_cells,
            max_engine_steps,
            max_forced_steps,
            max_observation_bytes,
            manual_coin,
        )
    )
    digest.update(struct.pack("<Q", _strict_uint64(stochastic_seed, "stochastic_seed")))
    for values in (
        hidden.counts,
        hidden.values,
        candidates.counts,
        candidates.values,
    ):
        digest.update(struct.pack("<I", int(values.size)))
        digest.update(values.astype("<i4", copy=False).tobytes())
    return digest.digest()


def pack_hidden_worlds(
    hidden_worlds: Sequence[HiddenInformation],
) -> PackedNativeRagged:
    """Pack six hidden-zone lists per world into contiguous int32 arrays."""
    worlds = tuple(hidden_worlds)
    if not worlds:
        raise ValueError("hidden_worlds must not be empty")
    counts: list[int] = []
    values: list[int] = []
    for world_index, hidden in enumerate(worlds):
        if not isinstance(hidden, HiddenInformation):
            raise TypeError(
                f"hidden_worlds[{world_index}] must be HiddenInformation"
            )
        zones = (
            hidden.your_deck,
            hidden.your_prize,
            hidden.opponent_deck,
            hidden.opponent_prize,
            hidden.opponent_hand,
            hidden.opponent_active,
        )
        if len(zones) != _HIDDEN_LIST_COUNT:
            raise RuntimeError("internal hidden-list count mismatch")
        for zone_index, cards in enumerate(zones):
            if len(cards) > DECK_SIZE:
                raise ValueError(
                    f"hidden world {world_index} zone {zone_index} exceeds "
                    f"the {DECK_SIZE}-card native cap"
                )
            counts.append(len(cards))
            values.extend(
                _strict_card_id(
                    card_id,
                    f"hidden_worlds[{world_index}].zones[{zone_index}]",
                )
                for card_id in cards
            )
    return PackedNativeRagged(
        counts=_readonly_int32(counts),
        values=_readonly_int32(values),
    )


def pack_candidate_actions(
    candidate_actions: Sequence[Sequence[int]],
) -> PackedNativeRagged:
    """Pack ragged complete root actions into contiguous int32 arrays."""
    candidates = tuple(candidate_actions)
    if not candidates:
        raise ValueError("candidate_actions must not be empty")
    counts: list[int] = []
    values: list[int] = []
    for candidate_index, action in enumerate(candidates):
        if not isinstance(action, Sequence) or isinstance(
            action, (str, bytes, bytearray)
        ):
            raise TypeError(
                f"candidate_actions[{candidate_index}] must be an integer sequence"
            )
        if len(action) > NATIVE_CONSEQUENCE_MAX_SELECT_COUNT:
            raise ValueError(
                f"candidate_actions[{candidate_index}] exceeds the native "
                f"select cap {NATIVE_CONSEQUENCE_MAX_SELECT_COUNT}"
            )
        counts.append(len(action))
        values.extend(
            _strict_option_index(
                option_index,
                f"candidate_actions[{candidate_index}]",
            )
            for option_index in action
        )
    return PackedNativeRagged(
        counts=_readonly_int32(counts),
        values=_readonly_int32(values),
    )


def nonempty_int32(
    values: npt.NDArray[np.int32],
) -> npt.NDArray[np.int32]:
    """Return storage with a non-null pointer while retaining logical size."""
    if values.size:
        return values
    return np.zeros(1, dtype=np.int32)


def _state_token(value: bytes | str) -> bytes:
    if isinstance(value, str):
        try:
            token = value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("state_token string must be ASCII") from exc
    elif isinstance(value, bytes):
        token = value
    else:
        raise TypeError("state_token must be bytes or str")
    if not token:
        raise ValueError("state_token must not be empty")
    if len(token) > NATIVE_CONSEQUENCE_MAX_STATE_TOKEN_BYTES:
        raise ValueError("state_token exceeds the native state-token cap")
    if any(value not in _STATE_TOKEN_ALPHABET for value in token):
        raise ValueError("state_token contains a non-engine-base64 byte")
    return token


def _strict_card_id(value: Any, location: str) -> int:
    card_id = _strict_int32(value, location)
    if card_id <= 0:
        raise ValueError(f"{location} card IDs must be positive")
    return card_id


def _strict_option_index(value: Any, location: str) -> int:
    option_index = _strict_int32(value, location)
    if option_index < 0:
        raise ValueError(f"{location} option indices must be nonnegative")
    return option_index


def _bounded_int(value: Any, *, minimum: int, maximum: int, name: str) -> int:
    parsed = _strict_int32(value, name)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return parsed


def _strict_int32(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if parsed < _INT32_MIN or parsed > _INT32_MAX:
        raise ValueError(f"{name} is outside the int32 range")
    return int(parsed)


def _strict_uint64(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if parsed < 0 or parsed > _UINT64_MAX:
        raise ValueError(f"{name} is outside the uint64 range")
    return int(parsed)


def _readonly_int32(values: Sequence[int]) -> npt.NDArray[np.int32]:
    result = np.asarray(values, dtype=np.int32)
    result.setflags(write=False)
    return result


__all__ = [
    "NATIVE_CONSEQUENCE_MAX_SELECT_COUNT",
    "NATIVE_CONSEQUENCE_MAX_STATE_TOKEN_BYTES",
    "NATIVE_CONSEQUENCE_REQUEST_FINGERPRINT_BYTES",
    "PackedNativeRagged",
    "PreparedNativeConsequenceRequest",
    "nonempty_int32",
    "native_consequence_request_fingerprint",
    "pack_candidate_actions",
    "pack_hidden_worlds",
    "prepare_native_consequence_request",
]
