"""Versioned root-visible producer context and observation boundary checks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, Self, cast

import orjson
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.context import (
    DECK_FLOW_FEATURE_SIZE,
    HISTORY_COUNTER_SIZE,
    CardCount,
    ExpectedCardCount,
    GameContextFeatures,
    LastAttackRecord,
)

_CONTEXT_DESCRIPTOR = (
    "root-information-producer/v1;root_player=i8;"
    "context=own_unseen,opponent_revealed,public_belief,history,deck_flow,"
    "last_attacks;observation=native_root_visible_json_v1"
)
ROOT_INFORMATION_PRODUCER_CONTEXT_FINGERPRINT = hashlib.sha256(
    _CONTEXT_DESCRIPTOR.encode("ascii")
).hexdigest()

_FORBIDDEN_NATIVE_KEYS = frozenset(
    {
        "beliefworld",
        "chancesupporthandle",
        "deckorder",
        "godview",
        "hiddeninformation",
        "opponentdeck",
        "opponenthand",
        "rawstate",
        "scenariohandle",
        "searchbegininput",
        "statetoken",
    }
)


class _CardCountRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: int = Field(gt=0)
    count: int = Field(gt=0)


class _ExpectedCardCountRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    card_id: int = Field(gt=0)
    expected_count: float = Field(ge=0.0)

    @field_validator("expected_count")
    @classmethod
    def finite_expected_count(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("expected_count must be finite")
        return value


class _LastAttackRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    serial: int = Field(gt=0)
    attack_id: int = Field(ge=0)


class RootInformationProducerContext(BaseModel):
    """Only request-local state a native engine lane cannot reconstruct."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    root_player: Literal[0, 1]
    own_unseen: tuple[_CardCountRecord, ...] = ()
    opponent_revealed: tuple[_CardCountRecord, ...] = ()
    opponent_belief: tuple[_ExpectedCardCountRecord, ...] = ()
    opponent_belief_entropy: float = 0.0
    opponent_belief_empty: bool = True
    history_counts: tuple[int, ...]
    deck_flow_counts: tuple[int, ...]
    last_attacks: tuple[_LastAttackRecord, ...] = ()

    @field_validator("opponent_belief_entropy")
    @classmethod
    def finite_entropy(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("opponent_belief_entropy must be finite and non-negative")
        return value

    @field_validator("history_counts")
    @classmethod
    def history_width(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != HISTORY_COUNTER_SIZE:
            raise ValueError("history_counts has the wrong width")
        return values

    @field_validator("deck_flow_counts")
    @classmethod
    def deck_flow_width(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if len(values) != DECK_FLOW_FEATURE_SIZE:
            raise ValueError("deck_flow_counts has the wrong width")
        return values

    @model_validator(mode="after")
    def canonical_records(self) -> Self:
        """Reject ambiguous duplicate/order encodings before fingerprinting."""
        for name in ("own_unseen", "opponent_revealed", "opponent_belief"):
            records = getattr(self, name)
            card_ids = tuple(record.card_id for record in records)
            if card_ids != tuple(sorted(set(card_ids))):
                raise ValueError(f"{name} must be uniquely sorted by card_id")
        serials = tuple(record.serial for record in self.last_attacks)
        if serials != tuple(sorted(set(serials))):
            raise ValueError("last_attacks must be uniquely sorted by serial")
        if self.opponent_belief_empty != (len(self.opponent_belief) == 0):
            raise ValueError("opponent_belief_empty contradicts opponent_belief")
        return self

    @classmethod
    def from_features(
        cls,
        features: GameContextFeatures,
        *,
        root_player: int,
    ) -> Self:
        """Freeze public tracker features into the producer wire schema."""
        if root_player not in (0, 1):
            raise ValueError("root_player must be 0 or 1")
        return cls(
            root_player=cast(Literal[0, 1], root_player),
            own_unseen=tuple(
                _CardCountRecord(card_id=item.card_id, count=item.count)
                for item in sorted(features.own_unseen)
            ),
            opponent_revealed=tuple(
                _CardCountRecord(card_id=item.card_id, count=item.count)
                for item in sorted(features.opponent_revealed)
            ),
            opponent_belief=tuple(
                _ExpectedCardCountRecord(
                    card_id=item.card_id,
                    expected_count=item.expected_count,
                )
                for item in sorted(features.opponent_belief)
            ),
            opponent_belief_entropy=features.opponent_belief_entropy,
            opponent_belief_empty=features.opponent_belief_empty,
            history_counts=features.history_counts,
            deck_flow_counts=features.deck_flow_counts,
            last_attacks=tuple(
                _LastAttackRecord(serial=item.serial, attack_id=item.attack_id)
                for item in sorted(features.last_attacks)
            ),
        )

    def to_features(self) -> GameContextFeatures:
        """Reconstruct the existing shared observation-encoder context type."""
        return GameContextFeatures(
            own_unseen=tuple(
                CardCount(card_id=item.card_id, count=item.count)
                for item in self.own_unseen
            ),
            opponent_revealed=tuple(
                CardCount(card_id=item.card_id, count=item.count)
                for item in self.opponent_revealed
            ),
            opponent_belief=tuple(
                ExpectedCardCount(
                    card_id=item.card_id,
                    expected_count=item.expected_count,
                )
                for item in self.opponent_belief
            ),
            opponent_belief_entropy=self.opponent_belief_entropy,
            opponent_belief_empty=self.opponent_belief_empty,
            history_counts=self.history_counts,
            deck_flow_counts=self.deck_flow_counts,
            last_attacks=tuple(
                LastAttackRecord(serial=item.serial, attack_id=item.attack_id)
                for item in self.last_attacks
            ),
        )

    def to_bytes(self) -> bytes:
        """Return canonical compact bytes stored by schema-9 rows."""
        return orjson.dumps(self.model_dump(mode="json"), option=orjson.OPT_SORT_KEYS)

    @classmethod
    def from_bytes(cls, payload: bytes) -> Self:
        """Strictly parse one bounded producer context payload."""
        try:
            decoded = orjson.loads(payload)
        except orjson.JSONDecodeError as exc:
            raise ValueError("producer_context is not valid JSON") from exc
        return cls.model_validate(decoded)


def encode_root_information_producer_context(
    features: GameContextFeatures,
    *,
    root_player: int,
) -> bytes:
    """Encode the shared training/serving producer context."""
    return RootInformationProducerContext.from_features(
        features,
        root_player=root_player,
    ).to_bytes()


def decode_root_information_observation(
    root_observable_state: bytes,
    producer_context: bytes,
) -> Mapping[str, Any]:
    """Decode one native row, reject hidden identities, and attach context."""
    context = RootInformationProducerContext.from_bytes(producer_context)
    observation = decode_root_observable_state(
        root_observable_state,
        root_player=context.root_player,
    )
    observation["gameContext"] = context.to_features().as_observation_dict()
    return observation


def decode_root_observable_state(
    root_observable_state: bytes,
    *,
    root_player: int,
) -> dict[str, Any]:
    """Decode and validate the native-only root-visible observation boundary."""
    if root_player not in (0, 1):
        raise ValueError("root_player must be 0 or 1")
    try:
        decoded = orjson.loads(root_observable_state)
    except orjson.JSONDecodeError as exc:
        raise ValueError("root_observable_state is not valid JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError("root_observable_state must decode to an object")
    observation = cast(dict[str, Any], decoded)
    if set(observation) != {"current", "logs", "select"}:
        raise ValueError("root_observable_state has an unexpected top-level schema")
    _reject_forbidden_keys(observation)
    _validate_root_visible_zones(observation, root_player=root_player)
    return observation


class PublicBeliefFeatureProducer(Protocol):
    """Public-only posterior producer shared by deployed and planned states."""

    @property
    def producer_fingerprint(self) -> str:
        """Return the immutable content identity of posterior semantics."""

    def augment(
        self,
        observation: Any,
        context_features: GameContextFeatures,
    ) -> GameContextFeatures:
        """Return belief fields derived only from root-visible evidence."""


def _reject_forbidden_keys(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(
                character for character in str(key).lower() if character.isalnum()
            )
            if normalized in _FORBIDDEN_NATIVE_KEYS:
                raise ValueError(f"native observation contains forbidden field: {key}")
            _reject_forbidden_keys(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            _reject_forbidden_keys(item)


def _validate_root_visible_zones(
    observation: Mapping[str, Any],
    *,
    root_player: int,
) -> None:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        raise ValueError("native observation current must be an object")
    if int(current.get("yourIndex", -1)) != root_player:
        raise ValueError("native observation perspective differs from producer context")
    players = current.get("players")
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        raise ValueError("native observation players must be a sequence")
    if len(players) != 2 or not all(isinstance(player, Mapping) for player in players):
        raise ValueError("native observation must contain two player objects")
    opponent = cast(Mapping[str, Any], players[1 - root_player])
    if _zone_exposes_card_identity(opponent.get("hand")):
        raise ValueError("native observation exposes sampled opponent hand identity")
    for player in players:
        player_mapping = cast(Mapping[str, Any], player)
        if _zone_exposes_card_identity(player_mapping.get("deck")):
            raise ValueError("native observation exposes ordered deck identity")
        if _zone_exposes_card_identity(player_mapping.get("prize")):
            raise ValueError("native observation exposes hidden prize identity")


def _zone_exposes_card_identity(value: Any) -> bool:
    """Treat a scalar zone size as public while rejecting container identities."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return False
    return _contains_card_identity(value)


def _contains_card_identity(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value > 0
    if isinstance(value, Mapping):
        for key in ("id", "cardId"):
            card_id = value.get(key)
            if (
                isinstance(card_id, int)
                and not isinstance(card_id, bool)
                and card_id > 0
            ):
                return True
        return any(_contains_card_identity(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_contains_card_identity(item) for item in value)
    return False


__all__ = [
    "PublicBeliefFeatureProducer",
    "ROOT_INFORMATION_PRODUCER_CONTEXT_FINGERPRINT",
    "RootInformationProducerContext",
    "decode_root_information_observation",
    "decode_root_observable_state",
    "encode_root_information_producer_context",
]
