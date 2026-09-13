"""Compact, lossless serialization for stateless curriculum snapshots."""

from __future__ import annotations

import zlib
from collections.abc import Mapping
from typing import Any, cast

import msgpack

_MAGIC = b"PTCG-RL-COMPACT-MAPPING-MSGPACK-ZLIB-V1\x00"
_MAX_DECOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def encode_compact_mapping(payload: Mapping[str, Any]) -> bytes:
    """Encode one JSON-compatible mapping without changing its values."""
    packed = msgpack.packb(
        _canonical_value(payload),
        use_bin_type=True,
        strict_types=True,
    )
    return _MAGIC + zlib.compress(packed, level=1)


def decode_compact_mapping(payload: bytes) -> dict[str, Any]:
    """Decode one bounded compact payload into its logical mapping."""
    if not payload.startswith(_MAGIC):
        raise ValueError("compact curriculum payload header is invalid")
    compressed = payload[len(_MAGIC) :]
    decompressor = zlib.decompressobj()
    packed = decompressor.decompress(compressed, _MAX_DECOMPRESSED_BYTES + 1)
    if len(packed) > _MAX_DECOMPRESSED_BYTES:
        raise ValueError("compact curriculum payload exceeds the safety limit")
    if decompressor.unconsumed_tail or not decompressor.eof:
        raise ValueError("compact curriculum payload is truncated or oversized")
    packed += decompressor.flush()
    if len(packed) > _MAX_DECOMPRESSED_BYTES:
        raise ValueError("compact curriculum payload exceeds the safety limit")
    decoded = msgpack.unpackb(
        packed,
        raw=False,
        strict_map_key=False,
    )
    if not isinstance(decoded, dict) or any(
        not isinstance(key, str) for key in decoded
    ):
        raise ValueError("compact curriculum payload must decode to a string mapping")
    return cast(dict[str, Any], decoded)


def encode_curriculum_payload(payload: Mapping[str, Any]) -> bytes:
    """Compatibility name for compact curriculum-state mappings."""
    return encode_compact_mapping(payload)


def decode_curriculum_payload(payload: bytes) -> dict[str, Any]:
    """Compatibility name for compact curriculum-state mappings."""
    return decode_compact_mapping(payload)


def _canonical_value(value: Any) -> Any:
    """Normalize JSON containers so logical equality implies byte equality."""
    if isinstance(value, Mapping):
        return {key: _canonical_value(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    return value


__all__ = [
    "decode_curriculum_payload",
    "decode_compact_mapping",
    "encode_curriculum_payload",
    "encode_compact_mapping",
]
