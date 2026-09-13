"""Typed semantic identity for one prepared native consequence request."""

from __future__ import annotations

import hashlib
import re
import struct
from collections.abc import Mapping
from typing import Any, Literal, Self

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.engine.compact_consequence import ScenarioSupport, ScenarioSupportMode
from ptcg_rl.engine.consequence_identity import strict_int32

_ROOT_OBSERVATION_DOMAIN = (
    b"ptcg-rl/native-consequence/root-observation-instance/v1\x00"
)
_PREPARED_REQUEST_DOMAIN = b"ptcg-rl/native-consequence/prepared-request/v1\x00"
_UINT64_MAX = (1 << 64) - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class PreparedRequestIdentity(BaseModel):
    """Typed semantic and raw identities for one immutable bridge request."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    root_state_fingerprint: str
    root_observation_fingerprint: str
    root_observation_schema_fingerprint: str
    engine_library_fingerprint: str
    native_abi_fingerprint: str
    candidate_fingerprints: tuple[str, ...]
    scenario_fingerprints: tuple[str, ...]
    scenario_handle_pairs: tuple[tuple[int, int], ...]
    scenario_support_fingerprint: str
    support_mode: ScenarioSupportMode
    legal_action_count: int = Field(ge=1)
    root_player: int = Field(ge=0, le=1)
    manual_coin: Literal[True] = True
    max_cells: int = Field(ge=1)
    max_engine_steps: int = Field(ge=1)
    max_forced_steps: int = Field(ge=0)
    max_observation_bytes: int = Field(ge=1)
    native_request_fingerprint: str
    contract_fingerprint: str

    @field_validator(
        "root_state_fingerprint",
        "root_observation_fingerprint",
        "root_observation_schema_fingerprint",
        "engine_library_fingerprint",
        "native_abi_fingerprint",
        "scenario_support_fingerprint",
        "native_request_fingerprint",
        "contract_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require canonical lowercase SHA-256 identities."""
        return _validate_sha256(value)

    @field_validator("candidate_fingerprints", "scenario_fingerprints")
    @classmethod
    def valid_fingerprint_sequence(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty, duplicate, or malformed ordered identity sets."""
        if not values:
            raise ValueError("prepared request identity sets must not be empty")
        canonical = tuple(_validate_sha256(value) for value in values)
        if len(set(canonical)) != len(canonical):
            raise ValueError("prepared request identity sets must be unique")
        return canonical

    @field_validator("scenario_handle_pairs")
    @classmethod
    def valid_handle_pairs(
        cls,
        values: tuple[tuple[int, int], ...],
    ) -> tuple[tuple[int, int], ...]:
        """Require unique ordered pairs of opaque unsigned handles."""
        if not values:
            raise ValueError("prepared scenario handles must not be empty")
        for pair in values:
            if len(pair) != 2 or any(
                isinstance(value, bool) or not 0 <= value <= _UINT64_MAX
                for value in pair
            ):
                raise ValueError("scenario handles must be uint64 pairs")
        if len(set(values)) != len(values):
            raise ValueError("prepared scenario handle pairs must be unique")
        return values

    @model_validator(mode="after")
    def valid_contract(self) -> Self:
        """Bind every semantic identity and execution cap to one digest."""
        if self.legal_action_count < len(self.candidate_fingerprints):
            raise ValueError("legal action count cannot be below candidate count")
        expected = prepared_request_contract_fingerprint(
            root_state_fingerprint=self.root_state_fingerprint,
            root_observation_fingerprint=self.root_observation_fingerprint,
            root_observation_schema_fingerprint=(
                self.root_observation_schema_fingerprint
            ),
            engine_library_fingerprint=self.engine_library_fingerprint,
            native_abi_fingerprint=self.native_abi_fingerprint,
            candidate_fingerprints=self.candidate_fingerprints,
            scenario_fingerprints=self.scenario_fingerprints,
            scenario_handle_pairs=self.scenario_handle_pairs,
            scenario_support_fingerprint=self.scenario_support_fingerprint,
            support_mode=self.support_mode,
            legal_action_count=self.legal_action_count,
            root_player=self.root_player,
            manual_coin=self.manual_coin,
            max_cells=self.max_cells,
            max_engine_steps=self.max_engine_steps,
            max_forced_steps=self.max_forced_steps,
            max_observation_bytes=self.max_observation_bytes,
        )
        if self.contract_fingerprint != expected:
            raise ValueError("prepared request contract fingerprint does not match")
        return self


def root_observation_fingerprint(observation: Mapping[str, Any]) -> str:
    """Fingerprint one complete JSON-compatible root observation mapping."""
    if not isinstance(observation, Mapping):
        raise TypeError("root observation must be a mapping")
    try:
        canonical = orjson.dumps(observation, option=orjson.OPT_SORT_KEYS)
    except (TypeError, orjson.JSONEncodeError) as exc:
        raise ValueError("root observation must be canonical-JSON compatible") from exc
    payload = _ROOT_OBSERVATION_DOMAIN + struct.pack(">Q", len(canonical)) + canonical
    return hashlib.sha256(payload).hexdigest()


def build_prepared_request_identity(
    *,
    root_state_fingerprint: str,
    root_observation_fingerprint: str,
    root_observation_schema_fingerprint: str,
    engine_library_fingerprint: str,
    native_abi_fingerprint: str,
    candidate_fingerprints: tuple[str, ...],
    scenario_support: ScenarioSupport,
    legal_action_count: int,
    root_player: int,
    manual_coin: Literal[True],
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
    native_request_fingerprint: str,
) -> PreparedRequestIdentity:
    """Build the validated external envelope for a prepared native request."""
    scenario_fingerprints = tuple(
        scenario.scenario_fingerprint for scenario in scenario_support.scenarios
    )
    scenario_handle_pairs = tuple(
        (scenario.belief_world_handle, scenario.chance_support_handle)
        for scenario in scenario_support.scenarios
    )
    contract_fingerprint = prepared_request_contract_fingerprint(
        root_state_fingerprint=root_state_fingerprint,
        root_observation_fingerprint=root_observation_fingerprint,
        root_observation_schema_fingerprint=root_observation_schema_fingerprint,
        engine_library_fingerprint=engine_library_fingerprint,
        native_abi_fingerprint=native_abi_fingerprint,
        candidate_fingerprints=candidate_fingerprints,
        scenario_fingerprints=scenario_fingerprints,
        scenario_handle_pairs=scenario_handle_pairs,
        scenario_support_fingerprint=scenario_support.support_fingerprint,
        support_mode=scenario_support.mode,
        legal_action_count=legal_action_count,
        root_player=root_player,
        manual_coin=manual_coin,
        max_cells=max_cells,
        max_engine_steps=max_engine_steps,
        max_forced_steps=max_forced_steps,
        max_observation_bytes=max_observation_bytes,
    )
    return PreparedRequestIdentity(
        root_state_fingerprint=root_state_fingerprint,
        root_observation_fingerprint=root_observation_fingerprint,
        root_observation_schema_fingerprint=root_observation_schema_fingerprint,
        engine_library_fingerprint=engine_library_fingerprint,
        native_abi_fingerprint=native_abi_fingerprint,
        candidate_fingerprints=candidate_fingerprints,
        scenario_fingerprints=scenario_fingerprints,
        scenario_handle_pairs=scenario_handle_pairs,
        scenario_support_fingerprint=scenario_support.support_fingerprint,
        support_mode=scenario_support.mode,
        legal_action_count=legal_action_count,
        root_player=root_player,
        manual_coin=manual_coin,
        max_cells=max_cells,
        max_engine_steps=max_engine_steps,
        max_forced_steps=max_forced_steps,
        max_observation_bytes=max_observation_bytes,
        native_request_fingerprint=native_request_fingerprint,
        contract_fingerprint=contract_fingerprint,
    )


def prepared_request_contract_fingerprint(
    *,
    root_state_fingerprint: str,
    root_observation_fingerprint: str,
    root_observation_schema_fingerprint: str,
    engine_library_fingerprint: str,
    native_abi_fingerprint: str,
    candidate_fingerprints: tuple[str, ...],
    scenario_fingerprints: tuple[str, ...],
    scenario_handle_pairs: tuple[tuple[int, int], ...],
    scenario_support_fingerprint: str,
    support_mode: ScenarioSupportMode,
    legal_action_count: int,
    root_player: int,
    manual_coin: bool,
    max_cells: int,
    max_engine_steps: int,
    max_forced_steps: int,
    max_observation_bytes: int,
) -> str:
    """Bind semantic identities and execution limits independently of raw ABI."""
    digest = hashlib.sha256()
    digest.update(_PREPARED_REQUEST_DOMAIN)
    for value in (
        root_state_fingerprint,
        root_observation_fingerprint,
        root_observation_schema_fingerprint,
        engine_library_fingerprint,
        native_abi_fingerprint,
        scenario_support_fingerprint,
    ):
        digest.update(bytes.fromhex(_validate_sha256(value)))
    _update_fingerprint_sequence(digest, candidate_fingerprints)
    _update_fingerprint_sequence(digest, scenario_fingerprints)
    digest.update(struct.pack(">I", len(scenario_handle_pairs)))
    for belief_handle, chance_handle in scenario_handle_pairs:
        digest.update(struct.pack(">QQ", belief_handle, chance_handle))
    mode = support_mode.value.encode("ascii")
    digest.update(struct.pack(">I", len(mode)))
    digest.update(mode)
    digest.update(
        struct.pack(
            ">ii?iiii",
            strict_int32(legal_action_count, "legal_action_count"),
            strict_int32(root_player, "root_player"),
            manual_coin,
            strict_int32(max_cells, "max_cells"),
            strict_int32(max_engine_steps, "max_engine_steps"),
            strict_int32(max_forced_steps, "max_forced_steps"),
            strict_int32(max_observation_bytes, "max_observation_bytes"),
        )
    )
    return digest.hexdigest()


def _update_fingerprint_sequence(
    digest: Any,
    values: tuple[str, ...],
) -> None:
    digest.update(struct.pack(">I", len(values)))
    for value in values:
        digest.update(bytes.fromhex(_validate_sha256(value)))


def _validate_sha256(value: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("fingerprints must be lowercase SHA-256 hex strings")
    return value


__all__ = [
    "PreparedRequestIdentity",
    "build_prepared_request_identity",
    "prepared_request_contract_fingerprint",
    "root_observation_fingerprint",
]
