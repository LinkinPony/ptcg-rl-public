"""Versioned compact payloads for exact planner-profile root context."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from typing import Any, cast

import msgpack

from ptcg_rl.agent.search.root_information_producer import (
    context_snapshot_fingerprint,
)
from ptcg_rl.context import GameContext, GameContextSnapshot

PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION = 1
PROFILE_OBSERVATION_CODEC_VERSION = 1

_OBSERVATION_DOMAIN = b"ptcg-rl/planner-profile-observation/v1\x00"
_SNAPSHOT_FIELD_NAMES = tuple(item.name for item in fields(GameContextSnapshot))


def encode_profile_observation(observation: Mapping[str, Any]) -> bytes:
    """Encode one exact public observation with deterministic key ordering."""
    payload = {
        "codec_version": PROFILE_OBSERVATION_CODEC_VERSION,
        "observation": _canonical_value(observation),
    }
    return cast(
        bytes,
        msgpack.packb(payload, use_bin_type=True, strict_types=True),
    )


def decode_profile_observation(payload: bytes) -> Mapping[str, Any]:
    """Decode and validate one exact public observation payload."""
    raw = _unpack_mapping(payload, label="profile observation")
    if raw.get("codec_version") != PROFILE_OBSERVATION_CODEC_VERSION:
        raise ValueError("profile observation codec version is incompatible")
    observation = raw.get("observation")
    if not isinstance(observation, Mapping):
        raise ValueError("profile observation payload is not a mapping")
    required = {"current", "logs", "select", "search_begin_input", "gameContext"}
    if not required.issubset(observation):
        raise ValueError("profile observation payload lacks planner root fields")
    return cast(Mapping[str, Any], observation)


def profile_observation_fingerprint(payload: bytes) -> str:
    """Fingerprint the versioned exact observation bytes."""
    return hashlib.sha256(_OBSERVATION_DOMAIN + payload).hexdigest()


def encode_profile_context_snapshot(snapshot: GameContextSnapshot) -> bytes:
    """Encode the full online context state needed by branch continuation."""
    payload = {
        "codec_version": PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION,
        "snapshot": {
            name: _canonical_value(getattr(snapshot, name))
            for name in _SNAPSHOT_FIELD_NAMES
        },
    }
    return cast(
        bytes,
        msgpack.packb(payload, use_bin_type=True, strict_types=True),
    )


def decode_profile_context_snapshot(payload: bytes) -> GameContextSnapshot:
    """Decode a snapshot and prove it can round-trip through ``GameContext``."""
    raw = _unpack_mapping(payload, label="profile context snapshot")
    if raw.get("codec_version") != PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION:
        raise ValueError("profile context snapshot codec version is incompatible")
    snapshot_raw = raw.get("snapshot")
    if not isinstance(snapshot_raw, Mapping):
        raise ValueError("profile context snapshot payload is not a mapping")
    if set(snapshot_raw) != set(_SNAPSHOT_FIELD_NAMES):
        raise ValueError("profile context snapshot fields are incompatible")
    values = {name: _frozen_value(snapshot_raw[name]) for name in _SNAPSHOT_FIELD_NAMES}
    supporter_ids = values["supporter_card_ids"]
    if not isinstance(supporter_ids, tuple):
        raise ValueError("profile supporter-card payload is malformed")
    values["supporter_card_ids"] = frozenset(
        _strict_int(value, label="supporter card id") for value in supporter_ids
    )
    snapshot = GameContextSnapshot(**values)
    if snapshot.player_index not in (None, 0, 1):
        raise ValueError("profile context snapshot has an invalid player index")
    if GameContext.from_snapshot(snapshot).snapshot() != snapshot:
        raise ValueError("profile context snapshot does not round-trip exactly")
    return snapshot


def verified_profile_context_snapshot(
    payload: bytes,
    *,
    expected_fingerprint: str,
) -> GameContextSnapshot:
    """Decode a snapshot and bind its semantic fingerprint."""
    snapshot = decode_profile_context_snapshot(payload)
    if context_snapshot_fingerprint(snapshot) != expected_fingerprint:
        raise ValueError("profile context snapshot fingerprint differs from payload")
    return snapshot


def _unpack_mapping(payload: bytes, *, label: str) -> Mapping[str, Any]:
    if not payload:
        raise ValueError(f"{label} payload is empty")
    try:
        raw = msgpack.unpackb(payload, raw=False, strict_map_key=True)
    except (ValueError, msgpack.ExtraData) as exc:
        raise ValueError(f"{label} payload is invalid MessagePack") from exc
    if not isinstance(raw, Mapping) or any(not isinstance(key, str) for key in raw):
        raise ValueError(f"{label} envelope is not a string-keyed mapping")
    return cast(Mapping[str, Any], raw)


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("profile payload mappings require string keys")
        return {
            key: _canonical_value(value[key])
            for key in sorted(cast(Sequence[str], tuple(value)))
        }
    if isinstance(value, frozenset):
        return [_canonical_value(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, str, bytes)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("profile payload floats must be finite")
        return value
    raise TypeError(f"unsupported profile payload value: {type(value).__name__}")


def _frozen_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_frozen_value(item) for item in value)
    if isinstance(value, Mapping):
        return {str(key): _frozen_value(item) for key, item in value.items()}
    return value


def _strict_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


__all__ = [
    "PROFILE_CONTEXT_SNAPSHOT_CODEC_VERSION",
    "PROFILE_OBSERVATION_CODEC_VERSION",
    "decode_profile_context_snapshot",
    "decode_profile_observation",
    "encode_profile_context_snapshot",
    "encode_profile_observation",
    "profile_observation_fingerprint",
    "verified_profile_context_snapshot",
]
