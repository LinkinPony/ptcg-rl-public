"""Strict public-event reconstruction from native training-arena columns."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from ptcg_rl.context.public_events import (
    PUBLIC_EVENT_AREA_COUNT,
    PUBLIC_EVENT_ENTITY_FIELDS,
    PUBLIC_EVENT_WINDOW,
    PublicEvent,
    PublicEventActorRole,
    PublicEventDelta,
    PublicEventOverflowCount,
)
from ptcg_rl.engine.constants import AreaType, LogType
from ptcg_rl.engine.native_public_history import (
    EXPECTED_LOG_PARAM_COUNTS,
    LOG_PARAM_WIDTH,
    SUCCESS_STATUSES,
)

if TYPE_CHECKING:
    from ptcg_rl.context.public_event_arrays import PublicEventBatch
    from ptcg_rl.engine.native_training import NativeTrainingBatchView

_LOG_FIELDS: tuple[tuple[str, ...], ...] = (
    ("playerIndex",),
    ("playerIndex", "hasBasicPokemon"),
    ("playerIndex",),
    ("playerIndex",),
    ("playerIndex", "cardId", "serial"),
    ("playerIndex",),
    ("playerIndex", "cardId", "serial", "fromArea", "toArea"),
    ("playerIndex", "fromArea", "toArea"),
    (
        "playerIndex",
        "cardIdActive",
        "serialActive",
        "cardIdBench",
        "serialBench",
    ),
    (
        "playerIndex",
        "cardIdBefore",
        "serialBefore",
        "cardIdAfter",
        "serialAfter",
    ),
    ("playerIndex", "cardId", "serial"),
    ("playerIndex", "cardId", "serial", "cardIdTarget", "serialTarget"),
    ("playerIndex", "cardId", "serial", "cardIdTarget", "serialTarget"),
    ("playerIndex", "cardId", "serial", "cardIdTarget", "serialTarget"),
    (
        "playerIndex",
        "cardId",
        "serial",
        "cardIdBefore",
        "serialBefore",
        "cardIdAfter",
        "serialAfter",
    ),
    ("playerIndex", "cardId", "serial", "attackId"),
    ("playerIndex", "cardId", "serial", "value", "putDamageCounter"),
    ("playerIndex", "isRecover", "cardId", "serial"),
    ("playerIndex", "isRecover", "cardId", "serial"),
    ("playerIndex", "isRecover", "cardId", "serial"),
    ("playerIndex", "isRecover", "cardId", "serial"),
    ("playerIndex", "isRecover", "cardId", "serial"),
    ("playerIndex", "head"),
    ("result", "reason"),
)
_LOG_FIELD_INDEXES: tuple[dict[str, int], ...] = tuple(
    {field: parameter for parameter, field in enumerate(fields)}
    for fields in _LOG_FIELDS
)
_FIELD_PARAMETERS: dict[str, tuple[int, ...]] = {
    field: tuple(indexes.get(field, -1) for indexes in _LOG_FIELD_INDEXES)
    for field in {field for fields in _LOG_FIELDS for field in fields}
}
_HIDDEN_IDENTITY_LOG_TYPES = frozenset(
    (int(LogType.DRAW_REVERSE), int(LogType.MOVE_CARD_REVERSE))
)
_KNOWN_AREAS = frozenset(int(area) for area in AreaType)
_PUBLIC_EVENT_AREA_OOV = PUBLIC_EVENT_AREA_COUNT - 1
_INT32_MAX = int(np.iinfo(np.int32).max)


class NativePublicEventError(ValueError):
    """Raised when native public-log columns violate the event ABI."""


def native_public_event_deltas(
    batch: NativeTrainingBatchView,
    *,
    rows: npt.ArrayLike | None = None,
    previous: Sequence[PublicEventDelta] | None = None,
) -> tuple[PublicEventDelta, ...]:
    """Convert every acting-seat native log row into a bounded event delta.

    The native ABI has already privacy-projected each log for ``select_player``.
    This adapter preserves the Python callback schema exactly and validates the
    fixed-column ABI before exposing any event to the recurrent policy.
    """
    _validate_native_public_logs(batch)
    selected = (
        np.arange(batch.batch_size, dtype=np.int64)
        if rows is None
        else _selected_rows(rows, size=batch.batch_size)
    )
    row_count = int(selected.size)
    if previous is None:
        prior = (PublicEventDelta(),) * row_count
    else:
        prior = tuple(previous)
        if len(prior) != row_count:
            raise NativePublicEventError(
                "previous public-event deltas do not align with native rows"
            )

    deltas: list[PublicEventDelta] = []
    for output_row, source_row in enumerate(selected):
        row = int(source_row)
        start = int(batch.log_offsets[row])
        stop = int(batch.log_offsets[row + 1])
        deltas.append(
            _append_native_public_events(
                prior[output_row],
                batch,
                start=start,
                stop=stop,
                player_index=int(batch.select_player[row]),
            )
        )
    return tuple(deltas)


def native_public_event_batch(
    batch: NativeTrainingBatchView,
    *,
    rows: npt.ArrayLike | None = None,
    pin_memory: bool = False,
) -> PublicEventBatch:
    """Encode native logs directly through the array-oriented implementation."""
    from ptcg_rl.engine.native_public_event_batch import (
        native_public_event_batch as encode,
    )

    return encode(batch, rows=rows, pin_memory=pin_memory)


def _selected_rows(values: npt.ArrayLike, *, size: int) -> npt.NDArray[np.int64]:
    rows = np.asarray(values)
    if rows.ndim != 1 or rows.size <= 0:
        raise NativePublicEventError("native public-event rows are invalid")
    if not np.issubdtype(rows.dtype, np.integer):
        raise NativePublicEventError("native public-event rows are not integers")
    selected = rows.astype(np.int64, copy=False)
    if (
        np.any(selected < 0)
        or np.any(selected >= size)
        or np.unique(selected).size != selected.size
    ):
        raise NativePublicEventError("native public-event rows are invalid")
    return selected


def _append_native_public_events(
    delta: PublicEventDelta,
    batch: NativeTrainingBatchView,
    *,
    start: int,
    stop: int,
    player_index: int,
) -> PublicEventDelta:
    """Append fixed-column native logs without generic mapping round-trips."""
    if start == stop:
        return delta
    new_events = tuple(
        _native_public_event(
            batch,
            log_index=log_index,
            player_index=player_index,
        )
        for log_index in range(start, stop)
    )
    if (
        not delta.events
        and not delta.overflow
        and len(new_events) <= PUBLIC_EVENT_WINDOW
    ):
        return PublicEventDelta(events=new_events)

    appended = (*delta.events, *new_events)
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


def _native_public_event(
    batch: NativeTrainingBatchView,
    *,
    log_index: int,
    player_index: int,
) -> PublicEvent:
    """Project one validated fixed-column log directly into its public token."""
    raw_type = int(batch.log_type[log_index])
    fields = _LOG_FIELD_INDEXES[raw_type]
    hidden_identity = raw_type in _HIDDEN_IDENTITY_LOG_TYPES

    card_ids: list[int] = []
    serials: list[int] = []
    entity_mask: list[bool] = []
    for card_field, serial_field in PUBLIC_EVENT_ENTITY_FIELDS:
        card_parameter = None if hidden_identity else fields.get(card_field)
        serial_parameter = None if hidden_identity else fields.get(serial_field)
        card_ids.append(
            0
            if card_parameter is None
            else int(batch.log_params[card_parameter][log_index])
        )
        serials.append(
            0
            if serial_parameter is None
            else int(batch.log_params[serial_parameter][log_index])
        )
        entity_mask.append(card_parameter is not None or serial_parameter is not None)

    attack_parameter = None if hidden_identity else fields.get("attackId")
    value_parameter = fields.get("value")
    return PublicEvent(
        event_type=raw_type + 1,
        actor_role=_native_actor_role(
            _native_parameter(batch, fields, log_index, "playerIndex"),
            player_index=player_index,
        ),
        from_area=_native_area(_native_parameter(batch, fields, log_index, "fromArea")),
        to_area=_native_area(_native_parameter(batch, fields, log_index, "toArea")),
        card_ids=tuple(card_ids),
        serials=tuple(serials),
        entity_mask=tuple(entity_mask),
        attack_id=(
            0
            if attack_parameter is None
            else int(batch.log_params[attack_parameter][log_index])
        ),
        attack_id_present=attack_parameter is not None,
        value=(
            0.0
            if value_parameter is None
            else float(batch.log_params[value_parameter][log_index])
        ),
        value_present=value_parameter is not None,
        categorical_values=(
            _native_boolean_category(batch, fields, log_index, "putDamageCounter"),
            _native_boolean_category(batch, fields, log_index, "isRecover"),
            _native_boolean_category(batch, fields, log_index, "head"),
            _native_boolean_category(batch, fields, log_index, "hasBasicPokemon"),
            _native_bounded_category(
                batch,
                fields,
                log_index,
                "result",
                minimum=0,
                maximum=2,
            ),
            _native_bounded_category(
                batch,
                fields,
                log_index,
                "reason",
                minimum=1,
                maximum=4,
            ),
        ),
    )


def _native_parameter(
    batch: NativeTrainingBatchView,
    fields: dict[str, int],
    log_index: int,
    field: str,
) -> int | None:
    parameter = fields.get(field)
    return None if parameter is None else int(batch.log_params[parameter][log_index])


def _native_actor_role(
    actor_index: int | None,
    *,
    player_index: int,
) -> PublicEventActorRole:
    if actor_index not in (0, 1) or player_index not in (0, 1):
        return PublicEventActorRole.UNKNOWN
    if actor_index == player_index:
        return PublicEventActorRole.SELF
    return PublicEventActorRole.OPPONENT


def _native_area(value: int | None) -> int:
    if value is None:
        return 0
    return value if value in _KNOWN_AREAS else _PUBLIC_EVENT_AREA_OOV


def _native_boolean_category(
    batch: NativeTrainingBatchView,
    fields: dict[str, int],
    log_index: int,
    field: str,
) -> int:
    value = _native_parameter(batch, fields, log_index, field)
    if value is None:
        return 0
    return 2 if bool(value) else 1


def _native_bounded_category(
    batch: NativeTrainingBatchView,
    fields: dict[str, int],
    log_index: int,
    field: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    value = _native_parameter(batch, fields, log_index, field)
    if value is None:
        return 0
    if value < minimum or value > maximum:
        return maximum - minimum + 2
    return value - minimum + 1


def _validate_native_public_logs(batch: NativeTrainingBatchView) -> None:
    rows = batch.batch_size
    offsets = batch.log_offsets.astype(np.int64, copy=False)
    if (
        offsets.shape != (rows + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != batch.log_count
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise NativePublicEventError("native public-log offsets are invalid")
    if batch.select_player.shape != (rows,):
        raise NativePublicEventError("native acting-seat column does not align")
    if batch.log_type.shape != (batch.log_count,) or batch.log_param_count.shape != (
        batch.log_count,
    ):
        raise NativePublicEventError("native public-log columns do not align")
    if len(batch.log_params) != LOG_PARAM_WIDTH or any(
        column.shape != (batch.log_count,) for column in batch.log_params
    ):
        raise NativePublicEventError("native public-log parameter columns do not align")

    lengths = np.diff(offsets)
    successful = np.isin(batch.status, tuple(SUCCESS_STATUSES))
    if np.any((~successful) & (lengths != 0)):
        raise NativePublicEventError("native slot-error row exposed public logs")
    if np.any(successful & ~np.isin(batch.select_player, (0, 1))):
        raise NativePublicEventError("successful native row has no acting seat")

    log_types = batch.log_type.astype(np.int64, copy=False)
    if np.any((log_types < 0) | (log_types >= len(LogType))):
        raise NativePublicEventError("native public log type is unsupported")
    expected = np.asarray(EXPECTED_LOG_PARAM_COUNTS, dtype=np.int64)[log_types]
    if np.any(batch.log_param_count.astype(np.int64, copy=False) != expected):
        raise NativePublicEventError(
            "native public log parameter count is not canonical"
        )
    for parameter, column in enumerate(batch.log_params):
        if np.any((expected <= parameter) & (column.astype(np.int64, copy=False) != 0)):
            raise NativePublicEventError(
                "native public log has nonzero private/unused parameters"
            )


__all__ = [
    "NativePublicEventError",
    "native_public_event_batch",
    "native_public_event_deltas",
]
