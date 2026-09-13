"""Second-pass replay loading for retained consequence-audit roots."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.engine.session import HiddenInformation
from ptcg_rl.evaluation.consequence_audit_sampling import CaseLocator, RootShape

_CASE_COLUMNS = (
    "date",
    "episode_id",
    "step_index",
    "player_index",
    "search_begin_input",
    "action",
    "select_context",
    "select_min_count",
    "select_max_count",
    "select_option_count",
    "select_deck_present",
    "own_unseen_ids",
    "own_unseen_counts",
    "opponent_deck_ids",
    "godview_opp_hand_ids",
    "player0_deck_count",
    "player0_hand_count",
    "player0_prize_ids",
    "player0_active_ids",
    "player1_deck_count",
    "player1_hand_count",
    "player1_prize_ids",
    "player1_active_ids",
)


@dataclass(frozen=True, slots=True)
class ConsequenceAuditCase:
    """One bounded retained replay root with transient engine material."""

    case_id: str
    date: str
    episode_id: int
    step_index: int
    player_index: int
    state_token: str
    observed_action: tuple[int, ...]
    hidden: HiddenInformation
    labels: frozenset[str]


def load_audit_cases(
    locators: Sequence[CaseLocator],
    *,
    batch_size: int,
    fallback_card_id: int,
    fallback_basic_pokemon_id: int,
) -> tuple[ConsequenceAuditCase, ...]:
    """Load only retained replay rows and construct count-correct worlds."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if fallback_card_id <= 0 or fallback_basic_pokemon_id <= 0:
        raise ValueError("fallback card IDs must be positive")
    by_path: dict[Path, dict[int, CaseLocator]] = defaultdict(dict)
    for locator in locators:
        rows = by_path[locator.path]
        if locator.row_index in rows:
            raise ValueError("duplicate retained replay row locator")
        rows[locator.row_index] = locator

    cases: list[ConsequenceAuditCase] = []
    for path in sorted(by_path):
        targets = by_path[path]
        parquet_file = pq.ParquetFile(path)
        _require_columns(parquet_file.schema_arrow.names, path)
        absolute_start = 0
        remaining = set(targets)
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(_CASE_COLUMNS),
            use_threads=True,
        ):
            stop = absolute_start + batch.num_rows
            selected = sorted(
                index for index in remaining if absolute_start <= index < stop
            )
            if selected:
                local_indices = pa.array(
                    [index - absolute_start for index in selected],
                    type=pa.int64(),
                )
                for row_index, raw in zip(
                    selected,
                    batch.take(local_indices).to_pylist(),
                    strict=True,
                ):
                    cases.append(
                        _audit_case(
                            cast(Mapping[str, Any], raw),
                            targets[row_index],
                            fallback_card_id=fallback_card_id,
                            fallback_basic_pokemon_id=fallback_basic_pokemon_id,
                        )
                    )
                    remaining.remove(row_index)
            absolute_start = stop
        if remaining:
            raise ValueError(f"retained row indices exceed Parquet file: {path}")
    return tuple(sorted(cases, key=lambda item: item.case_id))


def _audit_case(
    row: Mapping[str, Any],
    locator: CaseLocator,
    *,
    fallback_card_id: int,
    fallback_basic_pokemon_id: int,
) -> ConsequenceAuditCase:
    shape = _root_shape(row)
    if shape != locator.shape:
        raise ValueError("retained replay row changed between scan and load")
    token = row.get("search_begin_input")
    if not isinstance(token, str) or not token:
        raise ValueError("retained replay row has no search_begin_input")
    player = shape.player_index
    if player not in (0, 1):
        raise ValueError("retained replay row has an invalid player index")
    opponent = 1 - player
    own_pool = _expanded_counts(row.get("own_unseen_ids"), row.get("own_unseen_counts"))
    opponent_pool = _positive_ints(row.get("opponent_deck_ids"))
    godview_hand = _positive_ints(row.get("godview_opp_hand_ids"))
    opponent_active_ids = tuple(
        int(value) for value in _sequence(row.get(f"player{opponent}_active_ids"))
    )
    hidden_active = tuple(
        fallback_basic_pokemon_id for card_id in opponent_active_ids if card_id <= 0
    )

    def count(seat: int, zone: str) -> int:
        return _zone_count(row, seat, zone)

    return ConsequenceAuditCase(
        case_id=locator.case_id,
        date=shape.date,
        episode_id=shape.episode_id,
        step_index=shape.step_index,
        player_index=player,
        state_token=token,
        observed_action=tuple(int(value) for value in _sequence(row.get("action"))),
        hidden=HiddenInformation.from_sequences(
            your_deck=(
                ()
                if bool(row.get("select_deck_present"))
                else _fill_cards(
                    own_pool,
                    count(player, "deck"),
                    fallback=fallback_card_id,
                )
            ),
            your_prize=_fill_cards(
                own_pool, count(player, "prize"), fallback=fallback_card_id
            ),
            opponent_deck=_fill_cards(
                opponent_pool, count(opponent, "deck"), fallback=fallback_card_id
            ),
            opponent_prize=_fill_cards(
                opponent_pool, count(opponent, "prize"), fallback=fallback_card_id
            ),
            opponent_hand=_fill_cards(
                godview_hand or opponent_pool,
                count(opponent, "hand"),
                fallback=fallback_card_id,
            ),
            opponent_active=hidden_active,
        ),
        labels=locator.labels,
    )


def _root_shape(row: Mapping[str, Any]) -> RootShape:
    return RootShape(
        date=str(row["date"]),
        episode_id=int(row["episode_id"]),
        step_index=int(row["step_index"]),
        player_index=int(row["player_index"]),
        select_context=int(row["select_context"]),
        select_min_count=int(row["select_min_count"]),
        select_max_count=int(row["select_max_count"]),
        select_option_count=int(row["select_option_count"]),
    )


def _zone_count(row: Mapping[str, Any], seat: int, zone: str) -> int:
    if zone == "prize":
        return len(_sequence(row.get(f"player{seat}_prize_ids")))
    return int(row[f"player{seat}_{zone}_count"])


def _expanded_counts(ids: Any, counts: Any) -> tuple[int, ...]:
    card_ids = tuple(int(value) for value in _sequence(ids))
    card_counts = tuple(int(value) for value in _sequence(counts))
    if len(card_ids) != len(card_counts):
        raise ValueError("own unseen ID/count columns have different lengths")
    return tuple(
        card_id
        for card_id, count in zip(card_ids, card_counts, strict=True)
        if card_id > 0 and count > 0
        for _ in range(count)
    )


def _positive_ints(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in _sequence(value) if int(item) > 0)


def _fill_cards(source: Sequence[int], count: int, *, fallback: int) -> tuple[int, ...]:
    if count < 0:
        raise ValueError("hidden-zone count must be non-negative")
    pool = tuple(int(card_id) for card_id in source if int(card_id) > 0)
    pool = pool or (fallback,)
    return tuple(pool[index % len(pool)] for index in range(count))


def _sequence(value: Any) -> Sequence[Any]:
    return (
        value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
        else ()
    )


def _require_columns(names: Sequence[str], path: Path) -> None:
    missing = sorted(set(_CASE_COLUMNS).difference(names))
    if missing:
        raise ValueError(f"Parquet input {path} is missing columns: {missing}")


__all__ = ["ConsequenceAuditCase", "load_audit_cases"]
