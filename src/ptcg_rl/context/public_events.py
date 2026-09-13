"""Versioned, privacy-safe public event deltas for recurrent policies."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from ptcg_rl.engine.constants import AreaType, LogType

PUBLIC_EVENT_SCHEMA_VERSION = 1
PUBLIC_EVENT_WINDOW = 32
PUBLIC_EVENT_ENTITY_FIELDS: tuple[tuple[str, str], ...] = (
    ("cardId", "serial"),
    ("cardIdActive", "serialActive"),
    ("cardIdBench", "serialBench"),
    ("cardIdBefore", "serialBefore"),
    ("cardIdAfter", "serialAfter"),
    ("cardIdTarget", "serialTarget"),
)
PUBLIC_EVENT_CATEGORICAL_FIELDS: tuple[str, ...] = (
    "putDamageCounter",
    "isRecover",
    "head",
    "hasBasicPokemon",
    "result",
    "reason",
)
PUBLIC_EVENT_CATEGORICAL_COUNTS: tuple[int, ...] = (4, 4, 4, 4, 5, 6)
_PUBLIC_EVENT_CATEGORICAL_ENCODINGS: tuple[str, ...] = (
    "missing=0,false=1,true=2,oov=3",
    "missing=0,false=1,true=2,oov=3",
    "missing=0,false=1,true=2,oov=3",
    "missing=0,false=1,true=2,oov=3",
    "missing=0,raw_0_to_2=1_to_3,oov=4",
    "missing=0,raw_1_to_4=1_to_4,oov=5",
)
PUBLIC_EVENT_ENTITY_COUNT = len(PUBLIC_EVENT_ENTITY_FIELDS)
PUBLIC_EVENT_CATEGORICAL_SIZE = len(PUBLIC_EVENT_CATEGORICAL_FIELDS)
PUBLIC_EVENT_TYPE_COUNT = len(LogType) + 2
PUBLIC_EVENT_AREA_COUNT = max(int(value) for value in AreaType) + 2
PUBLIC_EVENT_TYPE_OOV = PUBLIC_EVENT_TYPE_COUNT - 1
PUBLIC_EVENT_AREA_OOV = PUBLIC_EVENT_AREA_COUNT - 1
PUBLIC_EVENT_ACTOR_ROLE_COUNT = 3
KNOWN_LOG_TYPES = frozenset(int(log_type) for log_type in LogType)
COUNTERPARTY_LOG_TYPE_REWRITES: tuple[tuple[int, int], ...] = (
    (int(LogType.DRAW), int(LogType.DRAW_REVERSE)),
    (int(LogType.MOVE_CARD), int(LogType.MOVE_CARD_REVERSE)),
)
COUNTERPARTY_LOG_TYPE_MAP: Mapping[int, int] = MappingProxyType(
    dict(COUNTERPARTY_LOG_TYPE_REWRITES)
)

PUBLIC_LOG_FIELDS: tuple[str, ...] = (
    "type",
    "playerIndex",
    "hasBasicPokemon",
    "cardId",
    "serial",
    "fromArea",
    "toArea",
    "cardIdActive",
    "serialActive",
    "cardIdBench",
    "serialBench",
    "cardIdBefore",
    "serialBefore",
    "cardIdAfter",
    "serialAfter",
    "cardIdTarget",
    "serialTarget",
    "attackId",
    "value",
    "putDamageCounter",
    "isRecover",
    "head",
    "result",
    "reason",
)
PRIVATE_REVERSE_LOG_FIELDS = frozenset(
    {
        "cardId",
        "serial",
        "cardIdActive",
        "serialActive",
        "cardIdBench",
        "serialBench",
        "cardIdBefore",
        "serialBefore",
        "cardIdAfter",
        "serialAfter",
        "cardIdTarget",
        "serialTarget",
        "attackId",
    }
)
PRIVATE_MOVE_LOG_TYPES = frozenset(
    {
        int(LogType.DRAW),
        int(LogType.DRAW_REVERSE),
        int(LogType.MOVE_CARD),
        int(LogType.MOVE_CARD_REVERSE),
    }
)
_HIDDEN_IDENTITY_LOG_TYPES = frozenset(
    {int(LogType.DRAW_REVERSE), int(LogType.MOVE_CARD_REVERSE)}
)
_KNOWN_AREAS = frozenset(int(area) for area in AreaType)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_INT32_MAX = int(np.iinfo(np.int32).max)


class PublicEventActorRole(IntEnum):
    """Actor identity relative to the policy seat."""

    UNKNOWN = 0
    SELF = 1
    OPPONENT = 2


@dataclass(frozen=True)
class PublicEvent:
    """One canonical public engine log without protected card identity."""

    event_type: int
    actor_role: PublicEventActorRole
    from_area: int = 0
    to_area: int = 0
    card_ids: tuple[int, ...] = (0,) * PUBLIC_EVENT_ENTITY_COUNT
    serials: tuple[int, ...] = (0,) * PUBLIC_EVENT_ENTITY_COUNT
    entity_mask: tuple[bool, ...] = (False,) * PUBLIC_EVENT_ENTITY_COUNT
    attack_id: int = 0
    attack_id_present: bool = False
    value: float = 0.0
    value_present: bool = False
    categorical_values: tuple[int, ...] = (0,) * PUBLIC_EVENT_CATEGORICAL_SIZE

    def __post_init__(self) -> None:
        """Reject malformed event rows before they enter trajectory storage."""
        if not 1 <= self.event_type < PUBLIC_EVENT_TYPE_COUNT:
            raise ValueError("public event type is outside the embedding domain")
        if not 0 <= int(self.actor_role) < PUBLIC_EVENT_ACTOR_ROLE_COUNT:
            raise ValueError("public event actor role is invalid")
        if not 0 <= self.from_area < PUBLIC_EVENT_AREA_COUNT:
            raise ValueError("public event from_area is outside the embedding domain")
        if not 0 <= self.to_area < PUBLIC_EVENT_AREA_COUNT:
            raise ValueError("public event to_area is outside the embedding domain")
        if len(self.card_ids) != PUBLIC_EVENT_ENTITY_COUNT:
            raise ValueError("public event card_ids has invalid width")
        if len(self.serials) != PUBLIC_EVENT_ENTITY_COUNT:
            raise ValueError("public event serials has invalid width")
        if len(self.entity_mask) != PUBLIC_EVENT_ENTITY_COUNT:
            raise ValueError("public event entity_mask has invalid width")
        if len(self.categorical_values) != PUBLIC_EVENT_CATEGORICAL_SIZE:
            raise ValueError("public event categorical_values has invalid width")
        if any(value < 0 or value > np.iinfo(np.uint16).max for value in self.card_ids):
            raise ValueError("public event card ID exceeds uint16 storage")
        if any(
            value < np.iinfo(np.int32).min or value > np.iinfo(np.int32).max
            for value in (*self.serials, self.attack_id)
        ):
            raise ValueError("public event integer identity exceeds int32 storage")
        if any(
            not present and (card_id != 0 or serial != 0)
            for card_id, serial, present in zip(
                self.card_ids,
                self.serials,
                self.entity_mask,
                strict=True,
            )
        ):
            raise ValueError("masked public event identity must be canonical zero")
        if not self.attack_id_present and self.attack_id != 0:
            raise ValueError("masked public event attack ID must be canonical zero")
        if not self.value_present and (
            self.value != 0.0 or bool(np.signbit(self.value))
        ):
            raise ValueError("masked public event value must be canonical zero")
        if not np.isfinite(self.value):
            raise ValueError("public event value must be finite")
        if any(
            value < 0 or value >= count
            for value, count in zip(
                self.categorical_values,
                PUBLIC_EVENT_CATEGORICAL_COUNTS,
                strict=True,
            )
        ):
            raise ValueError("public event categorical value is outside its domain")


@dataclass(frozen=True, order=True)
class PublicEventOverflowCount:
    """Order-free type/actor count for an event evicted from the short window."""

    event_type: int
    actor_role: PublicEventActorRole
    count: int

    def __post_init__(self) -> None:
        """Validate one canonical sparse overflow entry."""
        if not 1 <= self.event_type < PUBLIC_EVENT_TYPE_COUNT:
            raise ValueError("overflow event type is outside the embedding domain")
        if not 0 <= int(self.actor_role) < PUBLIC_EVENT_ACTOR_ROLE_COUNT:
            raise ValueError("overflow actor role is invalid")
        if not 1 <= self.count <= _INT32_MAX:
            raise ValueError("overflow event count is outside int32 storage")


@dataclass(frozen=True)
class PublicEventDelta:
    """Events observed since the seat's previous committed policy decision."""

    events: tuple[PublicEvent, ...] = ()
    overflow: tuple[PublicEventOverflowCount, ...] = ()

    def __post_init__(self) -> None:
        """Keep retained events bounded and overflow entries canonical."""
        if len(self.events) > PUBLIC_EVENT_WINDOW:
            raise ValueError("public event delta exceeds the bounded window")
        keys = tuple((item.event_type, int(item.actor_role)) for item in self.overflow)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("public event overflow entries must be sorted and unique")
        if self.overflow and len(self.events) != PUBLIC_EVENT_WINDOW:
            raise ValueError(
                "public event overflow requires a full retained event window"
            )

    @property
    def dropped_count(self) -> int:
        """Return the exact number of events summarized outside the short window."""
        return sum(item.count for item in self.overflow)


@dataclass(frozen=True)
class PublicEventDecisionToken:
    """Prepare/commit token preventing stale or duplicate event consumption."""

    generation: int
    delta: PublicEventDelta


PUBLIC_EVENT_DECISION_CLOCK = "one_committed_non_forced_policy_action_per_seat"


_PUBLIC_EVENT_SCHEMA_DESCRIPTOR = {
    "name": "public_event_delta",
    "version": PUBLIC_EVENT_SCHEMA_VERSION,
    "decision_clock": PUBLIC_EVENT_DECISION_CLOCK,
    "ordering": "actor_visible_engine_callback_order",
    "consumption_source": "authoritative_engine_callback_for_target_seat",
    "retained_window": PUBLIC_EVENT_WINDOW,
    "overflow": "sorted_sparse_exact_type_actor_counts_for_evicted_prefix",
    "source_fields": PUBLIC_LOG_FIELDS,
    "event_type": {
        "padding": 0,
        "known_offset": 1,
        "oov": PUBLIC_EVENT_TYPE_OOV,
        "known": tuple((item.name, int(item)) for item in LogType),
    },
    "actor_role": tuple(role.name.lower() for role in PublicEventActorRole),
    "areas": {
        "missing": 0,
        "oov": PUBLIC_EVENT_AREA_OOV,
        "known": tuple((item.name, int(item)) for item in AreaType),
    },
    "entities": PUBLIC_EVENT_ENTITY_FIELDS,
    "entity_identity": ("uint16_card_id", "int32_match_serial", "presence"),
    "attack_identity": ("int32_attack_id", "presence"),
    "continuous": ("value", "presence"),
    "categorical": tuple(
        zip(
            PUBLIC_EVENT_CATEGORICAL_FIELDS,
            PUBLIC_EVENT_CATEGORICAL_COUNTS,
            _PUBLIC_EVENT_CATEGORICAL_ENCODINGS,
            strict=True,
        )
    ),
    "privacy": {
        "reverse_log_identity": "removed_before_encoding",
        "reverse_private_fields": tuple(sorted(PRIVATE_REVERSE_LOG_FIELDS)),
        "private_move_log_types": tuple(sorted(PRIVATE_MOVE_LOG_TYPES)),
        "counterparty_type_rewrites": COUNTERPARTY_LOG_TYPE_REWRITES,
        "unknown_counterparty_payload": "only_type_and_actor_retained_fail_closed",
    },
}


def _schema_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(b"ptcg-rl/public-event-schema/v1\x00" + encoded).hexdigest()


PUBLIC_EVENT_SCHEMA_FINGERPRINT = _schema_fingerprint(_PUBLIC_EVENT_SCHEMA_DESCRIPTOR)
PUBLIC_EVENT_DECISION_CLOCK_FINGERPRINT = hashlib.sha256(
    b"ptcg-rl/public-event-decision-clock/v1\x00"
    + PUBLIC_EVENT_DECISION_CLOCK.encode("ascii")
).hexdigest()


def public_event_schema_metadata() -> dict[str, int | str]:
    """Return portable metadata for the event reconstruction contract."""
    return {
        "version": PUBLIC_EVENT_SCHEMA_VERSION,
        "fingerprint": PUBLIC_EVENT_SCHEMA_FINGERPRINT,
    }


def validate_public_event_schema_metadata(value: Any) -> None:
    """Reject missing or incompatible public-event metadata."""
    if not isinstance(value, Mapping):
        raise ValueError("public event schema metadata is missing")
    if value.get("version") != PUBLIC_EVENT_SCHEMA_VERSION:
        raise ValueError("public event schema version mismatch")
    fingerprint = value.get("fingerprint")
    if (
        not isinstance(fingerprint, str)
        or _SHA256_PATTERN.fullmatch(fingerprint) is None
    ):
        raise ValueError("public event schema fingerprint is invalid")
    if fingerprint != PUBLIC_EVENT_SCHEMA_FINGERPRINT:
        raise ValueError("public event schema fingerprint mismatch")


def public_event_from_log(log: Any, *, player_index: int | None) -> PublicEvent:
    """Project one actor-visible engine log into a deterministic event token."""
    raw_type = _optional_int(_field(log, "type"))
    event_type = (
        raw_type + 1
        if raw_type is not None and 0 <= raw_type < len(LogType)
        else PUBLIC_EVENT_TYPE_OOV
    )
    actor_role = _actor_role(
        _optional_int(_field(log, "playerIndex")),
        player_index=player_index,
    )
    hidden_identity = raw_type in _HIDDEN_IDENTITY_LOG_TYPES
    card_ids: list[int] = []
    serials: list[int] = []
    entity_mask: list[bool] = []
    for card_field, serial_field in PUBLIC_EVENT_ENTITY_FIELDS:
        card_id = None if hidden_identity else _optional_int(_field(log, card_field))
        serial = None if hidden_identity else _optional_int(_field(log, serial_field))
        card_ids.append(_uint16_or_zero(card_id))
        serials.append(_int32_or_zero(serial))
        entity_mask.append(card_id is not None or serial is not None)

    raw_attack_id = None if hidden_identity else _optional_int(_field(log, "attackId"))
    value, value_present = _finite_float(_field(log, "value"))
    return PublicEvent(
        event_type=event_type,
        actor_role=actor_role,
        from_area=_area_index(_optional_int(_field(log, "fromArea"))),
        to_area=_area_index(_optional_int(_field(log, "toArea"))),
        card_ids=tuple(card_ids),
        serials=tuple(serials),
        entity_mask=tuple(entity_mask),
        attack_id=_int32_or_zero(raw_attack_id),
        attack_id_present=raw_attack_id is not None,
        value=value,
        value_present=value_present,
        categorical_values=(
            _boolean_category(_field(log, "putDamageCounter")),
            _boolean_category(_field(log, "isRecover")),
            _boolean_category(_field(log, "head")),
            _boolean_category(_field(log, "hasBasicPokemon")),
            _bounded_category(_field(log, "result"), minimum=0, maximum=2),
            _bounded_category(_field(log, "reason"), minimum=1, maximum=4),
        ),
    )


def append_public_events(
    delta: PublicEventDelta,
    logs: Sequence[Any],
    *,
    player_index: int | None,
) -> PublicEventDelta:
    """Append one accepted log batch and summarize any evicted prefix."""
    appended = [*delta.events]
    appended.extend(
        public_event_from_log(log, player_index=player_index) for log in logs
    )
    overflow_size = max(0, len(appended) - PUBLIC_EVENT_WINDOW)
    overflow_counts: Counter[tuple[int, int]] = Counter(
        {(item.event_type, int(item.actor_role)): item.count for item in delta.overflow}
    )
    for event in appended[:overflow_size]:
        key = (event.event_type, int(event.actor_role))
        overflow_counts[key] += 1
        if overflow_counts[key] > _INT32_MAX:
            raise ValueError("public event overflow count exceeds int32 storage")
    return PublicEventDelta(
        events=tuple(appended[overflow_size:]),
        overflow=tuple(
            PublicEventOverflowCount(
                event_type=event_type,
                actor_role=PublicEventActorRole(actor_role),
                count=count,
            )
            for (event_type, actor_role), count in sorted(overflow_counts.items())
        ),
    )


def _actor_role(
    actor_index: int | None,
    *,
    player_index: int | None,
) -> PublicEventActorRole:
    if actor_index not in (0, 1) or player_index not in (0, 1):
        return PublicEventActorRole.UNKNOWN
    if actor_index == player_index:
        return PublicEventActorRole.SELF
    return PublicEventActorRole.OPPONENT


def _area_index(value: int | None) -> int:
    if value is None:
        return 0
    return value if value in _KNOWN_AREAS else PUBLIC_EVENT_AREA_OOV


def _uint16_or_zero(value: int | None) -> int:
    if value is None:
        return 0
    if value < 0 or value > np.iinfo(np.uint16).max:
        raise ValueError("public event card ID exceeds uint16 storage")
    return value


def _int32_or_zero(value: int | None) -> int:
    if value is None:
        return 0
    if value < np.iinfo(np.int32).min or value > _INT32_MAX:
        raise ValueError("public event integer identity exceeds int32 storage")
    return value


def _finite_float(value: Any) -> tuple[float, bool]:
    if value is None:
        return (0.0, False)
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return (0.0, False)
    if not np.isfinite(converted):
        return (0.0, False)
    return (converted, True)


def _boolean_category(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (bool, np.bool_)):
        return 2 if bool(value) else 1
    return 3


def _bounded_category(value: Any, *, minimum: int, maximum: int) -> int:
    if value is None:
        return 0
    converted = _optional_int(value)
    if converted is None or converted < minimum or converted > maximum:
        return maximum - minimum + 2
    return converted - minimum + 1


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


__all__ = [
    "COUNTERPARTY_LOG_TYPE_MAP",
    "COUNTERPARTY_LOG_TYPE_REWRITES",
    "PRIVATE_MOVE_LOG_TYPES",
    "PRIVATE_REVERSE_LOG_FIELDS",
    "KNOWN_LOG_TYPES",
    "PUBLIC_EVENT_ACTOR_ROLE_COUNT",
    "PUBLIC_EVENT_AREA_COUNT",
    "PUBLIC_EVENT_CATEGORICAL_COUNTS",
    "PUBLIC_EVENT_CATEGORICAL_SIZE",
    "PUBLIC_EVENT_DECISION_CLOCK",
    "PUBLIC_EVENT_DECISION_CLOCK_FINGERPRINT",
    "PUBLIC_EVENT_ENTITY_COUNT",
    "PUBLIC_EVENT_SCHEMA_FINGERPRINT",
    "PUBLIC_EVENT_SCHEMA_VERSION",
    "PUBLIC_EVENT_TYPE_COUNT",
    "PUBLIC_EVENT_TYPE_OOV",
    "PUBLIC_EVENT_WINDOW",
    "PUBLIC_LOG_FIELDS",
    "PublicEvent",
    "PublicEventActorRole",
    "PublicEventDecisionToken",
    "PublicEventDelta",
    "PublicEventOverflowCount",
    "append_public_events",
    "public_event_from_log",
    "public_event_schema_metadata",
    "validate_public_event_schema_metadata",
]
