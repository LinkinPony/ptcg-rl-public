"""Small shared helpers for arena modules."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def field_value(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from a mapping or dataclass-like object."""
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def int_field(value: Any, name: str, default: int) -> int:
    """Read an integer field from a mapping or dataclass-like object."""
    raw_value = field_value(value, name, default)
    return int(raw_value) if raw_value is not None else default
