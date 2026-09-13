"""Compact integer keys for columnar native policy pointer joins."""

from __future__ import annotations

import numpy as np

_AREA_RADIX = 16
_OWNER_RADIX = 4
_INDEX_RADIX = 256
_SERIAL_RADIX = 1 << 32
_ATTACHMENT_PARENT_RADIX = 1 << 16
_ATTACHMENT_KIND_RADIX = 4
_ATTACHMENT_INDEX_RADIX = 256
_STADIUM_AREA = 7


def area_pointer_keys(
    batch_rows: np.ndarray,
    areas: np.ndarray,
    owners: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Encode row-local public area pointers as collision-free int64 keys."""
    canonical_owners = np.where(areas == _STADIUM_AREA, -1, owners)
    return (
        (
            batch_rows.astype(np.int64, copy=False) * _AREA_RADIX
            + areas.astype(np.int64, copy=False)
        )
        * _OWNER_RADIX
        + canonical_owners.astype(np.int64, copy=False)
        + 1
    ) * _INDEX_RADIX + indices.astype(np.int64, copy=False)


def serial_pointer_keys(
    batch_rows: np.ndarray,
    serials: np.ndarray,
) -> np.ndarray:
    """Encode row-local positive serials as collision-free int64 keys."""
    return batch_rows.astype(np.int64, copy=False) * _SERIAL_RADIX + serials.astype(
        np.int64, copy=False
    )


def attachment_pointer_keys(
    batch_rows: np.ndarray,
    parent_tokens: np.ndarray,
    kinds: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    """Encode row-local attachment pointers as collision-free int64 keys."""
    return (
        (
            batch_rows.astype(np.int64, copy=False) * _ATTACHMENT_PARENT_RADIX
            + parent_tokens.astype(np.int64, copy=False)
        )
        * _ATTACHMENT_KIND_RADIX
        + kinds.astype(np.int64, copy=False)
    ) * _ATTACHMENT_INDEX_RADIX + indices.astype(np.int64, copy=False)


def sorted_key_values(
    keys: np.ndarray,
    *values: np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Stable-sort aligned lookup columns by key."""
    if any(value.shape != keys.shape for value in values):
        raise ValueError("native lookup key/value columns must align")
    order = np.argsort(keys, kind="stable")
    return (keys[order], *(value[order] for value in values))


def exact_lookup(
    sorted_keys: np.ndarray,
    sorted_values: np.ndarray,
    query_keys: np.ndarray,
    *,
    missing: int = -1,
) -> np.ndarray:
    """Resolve exact integer keys without Python dictionaries."""
    output = np.full(query_keys.shape, missing, dtype=sorted_values.dtype)
    if sorted_keys.size == 0 or query_keys.size == 0:
        return output
    positions = np.searchsorted(sorted_keys, query_keys, side="left")
    in_bounds = positions < sorted_keys.shape[0]
    if not np.any(in_bounds):
        return output
    candidate_rows = np.flatnonzero(in_bounds)
    candidate_positions = positions[in_bounds]
    matches = sorted_keys[candidate_positions] == query_keys[in_bounds]
    if np.any(matches):
        matched_rows = candidate_rows[matches]
        output[matched_rows] = sorted_values[candidate_positions[matches]]
    return output


__all__ = [
    "area_pointer_keys",
    "attachment_pointer_keys",
    "exact_lookup",
    "serial_pointer_keys",
    "sorted_key_values",
]
