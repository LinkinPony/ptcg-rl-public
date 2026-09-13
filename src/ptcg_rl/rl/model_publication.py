"""Immutable CPU snapshots shared by policy publication transports."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from torch import Tensor

from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint


@dataclass(frozen=True, slots=True)
class PreparedModelState:
    """One content-addressed CPU snapshot ready for publication.

    The learner prepares this object once, then the disk and shared-memory
    transports serialize the same tensors and advertise the same digest.  This
    avoids hashing or transferring the approximately model-sized state once
    per transport.
    """

    state_dict: Mapping[str, Tensor]
    model_fingerprint: str
    copy_seconds: float = 0.0
    fingerprint_seconds: float = 0.0

    @property
    def preparation_seconds(self) -> float:
        """Return measured device-copy and fingerprint wall time."""
        return self.copy_seconds + self.fingerprint_seconds


def prepare_model_state(
    state_dict: Mapping[str, Any] | PreparedModelState,
) -> PreparedModelState:
    """Freeze a complete state dict on CPU and compute its canonical digest."""
    if isinstance(state_dict, PreparedModelState):
        return state_dict
    if not state_dict:
        raise ValueError("model publication requires at least one state tensor")
    snapshot: dict[str, Tensor] = {}
    copy_started_at = time.perf_counter()
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name:
            raise TypeError("model state names must be non-empty strings")
        if not isinstance(value, Tensor):
            raise TypeError(f"model state value is not a tensor: {name}")
        # ``copy=True`` is important for CPU models: plain ``detach().cpu()``
        # aliases live parameters and would not be an immutable publication.
        snapshot[name] = value.detach().to(device="cpu", copy=True).contiguous()
    copy_seconds = time.perf_counter() - copy_started_at
    fingerprint_started_at = time.perf_counter()
    fingerprint = canonical_model_state_fingerprint(snapshot)
    fingerprint_seconds = time.perf_counter() - fingerprint_started_at
    return PreparedModelState(
        state_dict=MappingProxyType(snapshot),
        model_fingerprint=fingerprint,
        copy_seconds=copy_seconds,
        fingerprint_seconds=fingerprint_seconds,
    )


def refresh_prepared_model_state(
    state_dict: Mapping[str, Any],
    previous: PreparedModelState | None,
) -> PreparedModelState:
    """Refresh a reusable CPU staging snapshot when its topology is unchanged.

    Publication transports are synchronous: after they return, the previous
    staging tensors no longer back an in-flight artifact. Reusing their storage
    avoids allocating and freeing a model-sized CPU snapshot every policy
    version, which otherwise fragments memory during long dense-private runs.
    """
    if previous is None or not _compatible_snapshot(state_dict, previous.state_dict):
        return prepare_model_state(state_dict)
    copy_started_at = time.perf_counter()
    for name, value in state_dict.items():
        if not isinstance(value, Tensor):
            raise TypeError(f"model state value is not a tensor: {name}")
        previous.state_dict[name].copy_(value.detach(), non_blocking=False)
    copy_seconds = time.perf_counter() - copy_started_at
    fingerprint_started_at = time.perf_counter()
    fingerprint = canonical_model_state_fingerprint(previous.state_dict)
    fingerprint_seconds = time.perf_counter() - fingerprint_started_at
    return PreparedModelState(
        state_dict=previous.state_dict,
        model_fingerprint=fingerprint,
        copy_seconds=copy_seconds,
        fingerprint_seconds=fingerprint_seconds,
    )


def _compatible_snapshot(
    source: Mapping[str, Any],
    target: Mapping[str, Tensor],
) -> bool:
    """Return whether an existing CPU snapshot can receive the new state."""
    if tuple(source) != tuple(target):
        return False
    return all(
        isinstance(value, Tensor)
        and target[name].device.type == "cpu"
        and target[name].shape == value.shape
        and target[name].dtype == value.dtype
        for name, value in source.items()
    )


def validate_model_fingerprint(value: str, *, name: str = "model_fingerprint") -> str:
    """Validate and return one canonical lowercase SHA-256 identity."""
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{name} must be lowercase SHA-256 hex")
    return value


__all__ = [
    "PreparedModelState",
    "prepare_model_state",
    "refresh_prepared_model_state",
    "validate_model_fingerprint",
]
