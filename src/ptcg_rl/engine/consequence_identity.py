"""Canonical identities and materialized scenario support for consequences."""

from __future__ import annotations

import hashlib
import math
import operator
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.engine.compact_consequence import (
    ROOT_OBSERVATION_ENCODING,
    ScenarioHandle,
    ScenarioSupport,
    ScenarioSupportMode,
    scenario_fingerprint,
    scenario_support_fingerprint,
)
from ptcg_rl.engine.session import HiddenInformation

_ROOT_TOKEN_DOMAIN = b"ptcg-rl/native-consequence/root-token/v1\x00"
_CANDIDATE_ACTION_DOMAIN = b"ptcg-rl/native-consequence/candidate-action/v1\x00"
_HIDDEN_WORLD_DOMAIN = b"ptcg-rl/native-consequence/hidden-world/v1\x00"
_CHANCE_SUPPORT_DOMAIN = b"ptcg-rl/native-consequence/chance-support/v1\x00"
_OBSERVATION_SCHEMA_DOMAIN = (
    b"ptcg-rl/native-consequence/root-observation-schema/v1\x00"
)
_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


@dataclass(frozen=True, slots=True)
class MaterializedScenario:
    """A public scenario handle bound to its engine hidden-zone material."""

    handle: ScenarioHandle
    hidden_information: HiddenInformation

    def __post_init__(self) -> None:
        """Reject a handle whose belief identity does not match its material."""
        actual = hidden_information_fingerprint(self.hidden_information)
        if self.handle.belief_world_fingerprint != actual:
            raise ValueError(
                "scenario belief-world fingerprint does not match hidden information"
            )

    @classmethod
    def create(
        cls,
        *,
        hidden_information: HiddenInformation,
        belief_world_handle: int,
        chance_support_handle: int,
        chance_support_identity: bytes | str,
        weight: float,
    ) -> MaterializedScenario:
        """Create a scenario with canonical content-derived fingerprints."""
        belief_fingerprint = hidden_information_fingerprint(hidden_information)
        chance_fingerprint = canonical_chance_support_fingerprint(
            chance_support_identity
        )
        return cls(
            handle=ScenarioHandle(
                belief_world_handle=belief_world_handle,
                chance_support_handle=chance_support_handle,
                belief_world_fingerprint=belief_fingerprint,
                chance_support_fingerprint=chance_fingerprint,
                scenario_fingerprint=scenario_fingerprint(
                    belief_world_fingerprint=belief_fingerprint,
                    chance_support_fingerprint=chance_fingerprint,
                ),
                weight=weight,
            ),
            hidden_information=hidden_information,
        )


def root_token_bytes(value: bytes | str) -> bytes:
    """Return the canonical bytes accepted by the native state-token ABI."""
    if isinstance(value, bytes):
        token = value
    elif isinstance(value, str):
        try:
            token = value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("state_token string must be ASCII") from exc
    else:
        raise TypeError("state_token must be bytes or str")
    if not token:
        raise ValueError("state_token must not be empty")
    return token


def root_token_fingerprint(state_token: bytes | str) -> str:
    """Return the canonical SHA-256 identity of one engine root token."""
    return _framed_fingerprint(_ROOT_TOKEN_DOMAIN, root_token_bytes(state_token))


def candidate_action_fingerprint(action: Sequence[int]) -> str:
    """Return an order-preserving SHA-256 identity for a complete selection."""
    canonical = tuple(
        strict_int32(value, "candidate action option") for value in action
    )
    payload = struct.pack(">I", len(canonical)) + b"".join(
        struct.pack(">i", value) for value in canonical
    )
    return hashlib.sha256(_CANDIDATE_ACTION_DOMAIN + payload).hexdigest()


def hidden_information_fingerprint(hidden: HiddenInformation) -> str:
    """Return a canonical SHA-256 identity for all six hidden-zone lists."""
    if not isinstance(hidden, HiddenInformation):
        raise TypeError("hidden must be HiddenInformation")
    digest = hashlib.sha256()
    digest.update(_HIDDEN_WORLD_DOMAIN)
    for zone in _hidden_zones(hidden):
        digest.update(struct.pack(">I", len(zone)))
        for card_id in zone:
            parsed = strict_int32(card_id, "hidden-world card ID")
            if parsed <= 0:
                raise ValueError("hidden-world card IDs must be positive")
            digest.update(struct.pack(">i", parsed))
    return digest.hexdigest()


def canonical_chance_support_fingerprint(identity: bytes | str) -> str:
    """Return a canonical identity for an opaque chance-support artifact."""
    return _framed_fingerprint(
        _CHANCE_SUPPORT_DOMAIN,
        _identity_bytes(identity, name="chance_support_identity"),
    )


def root_observation_schema_fingerprint() -> str:
    """Return the identity of the fixed root-visible native JSON schema."""
    return _framed_fingerprint(
        _OBSERVATION_SCHEMA_DOMAIN,
        ROOT_OBSERVATION_ENCODING.encode("ascii"),
    )


def normalize_scenario_support(
    scenarios: Sequence[MaterializedScenario],
    *,
    mode: ScenarioSupportMode,
) -> tuple[ScenarioSupport, tuple[MaterializedScenario, ...]]:
    """Normalize scenario weights while retaining handle/material bindings."""
    _validate_support_mode(mode)
    materialized = tuple(scenarios)
    if not materialized:
        raise ValueError("scenarios must not be empty")
    for index, scenario in enumerate(materialized):
        if not isinstance(scenario, MaterializedScenario):
            raise TypeError(f"scenarios[{index}] must be MaterializedScenario")
        if scenario.handle.belief_world_fingerprint != (
            hidden_information_fingerprint(scenario.hidden_information)
        ):
            raise ValueError(
                f"scenarios[{index}] is not bound to its hidden information"
            )
    try:
        total_weight = math.fsum(item.handle.weight for item in materialized)
    except OverflowError as exc:
        raise ValueError("scenario weight sum must be finite") from exc
    if not math.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("scenario weights must have a positive finite sum")

    normalized_handles = tuple(
        _with_weight(item.handle, item.handle.weight / total_weight)
        for item in materialized
    )
    support = ScenarioSupport(
        mode=mode,
        scenarios=normalized_handles,
        support_fingerprint=scenario_support_fingerprint(mode, normalized_handles),
    )
    normalized_materialized = tuple(
        MaterializedScenario(
            handle=handle,
            hidden_information=item.hidden_information,
        )
        for handle, item in zip(normalized_handles, materialized, strict=True)
    )
    return support, normalized_materialized


def strict_int32(value: Any, name: str) -> int:
    """Return one non-bool integer restricted to the native int32 range."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        parsed = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if parsed < _INT32_MIN or parsed > _INT32_MAX:
        raise ValueError(f"{name} is outside the int32 range")
    return int(parsed)


def _validate_support_mode(mode: ScenarioSupportMode) -> None:
    if not isinstance(mode, ScenarioSupportMode):
        raise TypeError("support mode must be ScenarioSupportMode")
    if mode is ScenarioSupportMode.SAMPLED_BELIEF_MANUAL_COIN_ENUMERATED:
        raise ValueError(
            "one-step native consequences cannot claim enumerated coin support"
        )


def _with_weight(handle: ScenarioHandle, weight: float) -> ScenarioHandle:
    return ScenarioHandle(
        belief_world_handle=handle.belief_world_handle,
        chance_support_handle=handle.chance_support_handle,
        belief_world_fingerprint=handle.belief_world_fingerprint,
        chance_support_fingerprint=handle.chance_support_fingerprint,
        scenario_fingerprint=handle.scenario_fingerprint,
        weight=weight,
    )


def _hidden_zones(hidden: HiddenInformation) -> tuple[tuple[int, ...], ...]:
    return (
        hidden.your_deck,
        hidden.your_prize,
        hidden.opponent_deck,
        hidden.opponent_prize,
        hidden.opponent_hand,
        hidden.opponent_active,
    )


def _identity_bytes(value: bytes | str, *, name: str) -> bytes:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        raise TypeError(f"{name} must be bytes or str")
    if not payload:
        raise ValueError(f"{name} must not be empty")
    return payload


def _framed_fingerprint(domain: bytes, payload: bytes) -> str:
    return hashlib.sha256(
        domain + struct.pack(">Q", len(payload)) + payload
    ).hexdigest()


__all__ = [
    "MaterializedScenario",
    "candidate_action_fingerprint",
    "canonical_chance_support_fingerprint",
    "hidden_information_fingerprint",
    "normalize_scenario_support",
    "root_observation_schema_fingerprint",
    "root_token_bytes",
    "root_token_fingerprint",
    "strict_int32",
]
