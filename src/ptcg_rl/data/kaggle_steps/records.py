"""Streaming records for step-level Kaggle replay behavior cloning data."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO, cast

import pyarrow as pa

from ptcg_rl.actions.selection import (
    is_forced,
    is_legal_action,
    normalize_action_order,
)
from ptcg_rl.context import (
    DECK_FLOW_FEATURE_SIZE,
    HISTORY_COUNTER_SIZE,
    GameContext,
    GameContextFeatures,
)
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.engine.effects import parse_effect_logs
from ptcg_rl.engine.feature_vectors import (
    DYNAMIC_EFFECT_FEATURE_SIZE,
    make_dynamic_effect_feature_row,
)

Row = dict[str, Any]

STEPS_KEY = b'"steps"'
DEFAULT_CHUNK_SIZE = 1 << 20
ENERGY_TYPE_COUNT = 12
STEP_ROW_SCHEMA_VERSION = 6

OPTION_FIELDS: tuple[tuple[str, str], ...] = (
    ("type", "type"),
    ("number", "number"),
    ("area", "area"),
    ("index", "index"),
    ("playerIndex", "player_index"),
    ("toolIndex", "tool_index"),
    ("energyIndex", "energy_index"),
    ("count", "count"),
    ("inPlayArea", "in_play_area"),
    ("inPlayIndex", "in_play_index"),
    ("attackId", "attack_id"),
    ("cardId", "card_id"),
    ("serial", "serial"),
    ("specialConditionType", "special_condition_type"),
)
LOG_FIELDS: tuple[str, ...] = (
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


@dataclass(frozen=True)
class PendingObservation:
    """One ACTIVE observation waiting for the next replay action."""

    step_index: int
    player_index: int
    side: Mapping[str, Any]
    context_features: GameContextFeatures
    godview_opp_hand_ids: tuple[int, ...] = ()
    godview_opp_hand_available: bool = False
    godview_opp_hand_count_match: bool = False


@dataclass(frozen=True)
class StepRowMetadata:
    """Metadata required to build one BC-compatible step row."""

    date: str = ""
    episode_id: int = 0
    replay_path: str = ""
    step_index: int = 0
    player_index: int = 0
    team_name: str = ""
    opponent_team_name: str = ""
    reward: float | None = None
    status: str = ""
    result: str = ""
    deck_signature: str = ""
    opponent_deck_signature: str = ""
    opponent_deck_ids: tuple[int, ...] = ()
    godview_opp_hand_ids: tuple[int, ...] = ()
    godview_opp_hand_available: bool = False
    godview_opp_hand_count_match: bool = False


def row_from_observation(
    observation: Mapping[str, Any],
    action: Sequence[int],
    *,
    metadata: StepRowMetadata,
    context_features: GameContextFeatures,
    before_side: Mapping[str, Any] | None = None,
    after_side: Mapping[str, Any] | None = None,
    forced: bool | None = None,
) -> Row:
    """Build a BC-compatible step row from one observation/action pair."""
    select = _mapping(observation.get("select"))
    normalized_action = tuple(int(index) for index in action)
    row: Row = {}
    _add_episode_fields(
        row,
        observation=observation,
        metadata=metadata,
        action=normalized_action,
        forced=is_forced(select) if forced is None else forced,
    )
    _add_select_fields(row, select)
    _add_current_fields(row, _mapping(observation.get("current")))
    _add_context_fields(row, context_features)
    if before_side is not None and after_side is not None:
        _add_chosen_effect_fields(
            row,
            before_side=before_side,
            after_side=after_side,
            action=normalized_action,
        )
    else:
        row["chosen_effect_features"] = [0.0] * DYNAMIC_EFFECT_FEATURE_SIZE
        row["chosen_effect_mask"] = False
    return row


def extract_replay_rows(
    replay_path: Path,
    *,
    drop_forced_actions: bool = True,
    normalize_unordered_actions: bool = True,
    fast_prefix_bytes: int = 65_536,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[list[Row], Counter[str]]:
    """Extract compact step rows from one replay without loading all steps."""
    counters: Counter[str] = Counter()
    metadata = side_metadata(replay_path, fast_prefix_bytes=fast_prefix_bytes)
    contexts = _initial_contexts(metadata)
    pending: dict[int, PendingObservation] = {}
    rows: list[Row] = []
    last_sides: list[Mapping[str, Any]] = []

    for step_index, sides in iter_replay_steps(replay_path, chunk_size=chunk_size):
        counters["steps_seen"] += 1
        last_sides = sides
        for player_index, side in enumerate(sides):
            pending_observation = pending.pop(player_index, None)
            if pending_observation is None:
                continue
            row = _row_from_pending(
                replay_path,
                pending_observation,
                action_side=side,
                metadata=metadata,
                drop_forced_actions=drop_forced_actions,
                normalize_unordered_actions=normalize_unordered_actions,
                counters=counters,
            )
            if row is not None:
                rows.append(row)

        for player_index, side in enumerate(sides):
            if _has_active_select(side):
                context_features = contexts[player_index].update(
                    _mapping(side.get("observation"))
                )
                pending[player_index] = PendingObservation(
                    step_index=step_index,
                    player_index=player_index,
                    side=side,
                    context_features=context_features,
                    **_godview_opp_hand_label(sides, player_index=player_index),
                )
                counters["active_select_observations"] += 1

    terminal_prize_diffs = _terminal_prize_diffs(last_sides)
    for row in rows:
        row["terminal_prize_diff"] = terminal_prize_diffs.get(
            int(row["player_index"])
        )
    counters["rows"] += len(rows)
    counters["dangling_observations"] += len(pending)
    return rows, counters


def side_metadata(
    replay_path: Path,
    *,
    fast_prefix_bytes: int,
) -> dict[int, Row]:
    """Return per-player episode metadata and deck signatures when available."""
    try:
        side_rows = deck_records.fast_episode_side_rows(
            replay_path=replay_path,
            card_meta={},
            known_decks={},
            prefix_bytes=fast_prefix_bytes,
            include_step_count=False,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        side_rows = None
    if side_rows:
        return {int(row["player_index"]): row for row in side_rows}

    replay = replay_stub(replay_path, chunk_size=fast_prefix_bytes)
    rewards = deck_records.rewards_from_replay(replay)
    statuses = deck_records.statuses_from_replay(replay)
    team_names = deck_records.team_names_from_replay(replay)
    episode_id = deck_records.episode_id_from_replay(replay, replay_path)
    metadata: dict[int, Row] = {}
    for player_index in range(2):
        metadata[player_index] = {
            "date": replay_path.parent.name,
            "episode_id": episode_id,
            "player_index": player_index,
            "team_name": deck_records.list_str(team_names, player_index),
            "reward": deck_records.list_float(rewards, player_index),
            "status": deck_records.list_str(statuses, player_index),
            "result": deck_records.result_label(
                deck_records.list_float(rewards, player_index),
                deck_records.list_str(statuses, player_index),
            ),
            "deck_signature": "",
            "deck_ids": [],
            "opponent_player_index": 1 - player_index,
            "opponent_deck_signature": "",
            "opponent_deck_ids": [],
        }
    return metadata


def replay_stub(replay_path: Path, *, chunk_size: int) -> Row:
    """Parse top-level replay metadata while replacing the large steps array."""
    prefix = _read_prefix_through_steps_key(replay_path, chunk_size=chunk_size)
    return cast(Row, json.loads(prefix + b'"steps": []}'))


def iter_replay_steps(
    replay_path: Path,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> Iterable[tuple[int, list[Mapping[str, Any]]]]:
    """Yield top-level replay step arrays one at a time."""
    array_start = _steps_array_start(replay_path, chunk_size=chunk_size)
    with replay_path.open("r", encoding="utf-8") as file_obj:
        file_obj.seek(array_start + 1)
        for step_index, item in enumerate(_iter_json_array_items(file_obj, chunk_size)):
            if isinstance(item, list):
                sides = [
                    cast(Mapping[str, Any], side)
                    for side in item
                    if isinstance(side, Mapping)
                ]
                yield step_index, sides


def step_row_schema() -> pa.Schema:
    """Return the Parquet schema for compact step rows."""
    fields = [
        pa.field("date", pa.string()),
        pa.field("episode_id", pa.int64()),
        pa.field("replay_path", pa.string()),
        pa.field("schema_version", pa.int16()),
        pa.field("step_index", pa.int32()),
        pa.field("player_index", pa.int8()),
        pa.field("team_name", pa.string()),
        pa.field("opponent_team_name", pa.string()),
        pa.field("reward", pa.float32()),
        pa.field("status", pa.string()),
        pa.field("result", pa.string()),
        pa.field("deck_signature", pa.string()),
        pa.field("opponent_deck_signature", pa.string()),
        pa.field("remaining_overage_time", pa.float32()),
        pa.field("search_begin_input", pa.string()),
        pa.field("action", pa.list_(pa.int16())),
        pa.field("action_length", pa.int16()),
        pa.field("is_forced", pa.bool_()),
        pa.field("terminal_prize_diff", pa.int16()),
        pa.field("current_prize_diff", pa.int16()),
        pa.field("select_type", pa.int16()),
        pa.field("select_context", pa.int16()),
        pa.field("select_min_count", pa.int16()),
        pa.field("select_max_count", pa.int16()),
        pa.field("select_option_count", pa.int16()),
        pa.field("remain_damage_counter", pa.int16()),
        pa.field("remain_energy_cost", pa.int16()),
        pa.field("context_card_id", pa.int32()),
        pa.field("context_card_serial", pa.int32()),
        pa.field("effect_card_id", pa.int32()),
        pa.field("effect_card_serial", pa.int32()),
        pa.field("select_deck_present", pa.bool_()),
        pa.field("select_deck_ids", pa.list_(pa.int32())),
        pa.field("own_unseen_ids", pa.list_(pa.int32())),
        pa.field("own_unseen_counts", pa.list_(pa.int16())),
        pa.field("opp_revealed_ids", pa.list_(pa.int32())),
        pa.field("opp_revealed_counts", pa.list_(pa.int16())),
        pa.field("history_counts", pa.list_(pa.int16())),
        pa.field("deck_flow_counts", pa.list_(pa.int32())),
        pa.field("last_attack_serials", pa.list_(pa.int32())),
        pa.field("last_attack_ids", pa.list_(pa.int32())),
        pa.field("opponent_deck_ids", pa.list_(pa.int32())),
        pa.field("godview_opp_hand_ids", pa.list_(pa.int32())),
        pa.field("godview_opp_hand_available", pa.bool_()),
        pa.field("godview_opp_hand_count_match", pa.bool_()),
        pa.field("chosen_effect_features", pa.list_(pa.float32())),
        pa.field("chosen_effect_mask", pa.bool_()),
        pa.field("looking_ids", pa.list_(pa.int32())),
        pa.field("stadium_ids", pa.list_(pa.int32())),
        pa.field("turn", pa.int16()),
        pa.field("turn_action_count", pa.int16()),
        pa.field("your_index", pa.int8()),
        pa.field("first_player", pa.int8()),
        pa.field("state_result", pa.int8()),
        pa.field("supporter_played", pa.bool_()),
        pa.field("stadium_played", pa.bool_()),
        pa.field("energy_attached", pa.bool_()),
        pa.field("retreated", pa.bool_()),
    ]
    for player_index in range(2):
        fields.extend(_player_schema_fields(player_index))
    for _, column_name in OPTION_FIELDS:
        fields.append(pa.field(f"option_{column_name}", pa.list_(pa.int32())))
    return pa.schema(fields)


def _row_from_pending(
    replay_path: Path,
    pending_observation: PendingObservation,
    *,
    action_side: Mapping[str, Any],
    metadata: dict[int, Row],
    drop_forced_actions: bool,
    normalize_unordered_actions: bool,
    counters: Counter[str],
) -> Row | None:
    observation = _mapping(pending_observation.side.get("observation"))
    select = _mapping(observation.get("select"))
    action = _int_sequence(action_side.get("action"))
    if action is None:
        counters["invalid_action_type"] += 1
        return None
    if not is_legal_action(select, action):
        counters["invalid_option_actions"] += 1
        return None
    forced = is_forced(select)
    if forced and drop_forced_actions:
        counters["forced_actions_dropped"] += 1
        return None
    if normalize_unordered_actions:
        action = normalize_action_order(select, action)

    return row_from_observation(
        observation,
        action,
        metadata=_step_row_metadata(
            replay_path,
            pending_observation=pending_observation,
            metadata=metadata,
        ),
        context_features=pending_observation.context_features,
        before_side=pending_observation.side,
        after_side=action_side,
        forced=forced,
    )


def _add_episode_fields(
    row: Row,
    *,
    observation: Mapping[str, Any],
    metadata: StepRowMetadata,
    action: Sequence[int],
    forced: bool,
) -> None:
    row.update(
        {
            "date": metadata.date,
            "episode_id": metadata.episode_id,
            "replay_path": metadata.replay_path,
            "schema_version": STEP_ROW_SCHEMA_VERSION,
            "step_index": metadata.step_index,
            "player_index": metadata.player_index,
            "team_name": metadata.team_name,
            "opponent_team_name": metadata.opponent_team_name,
            "reward": metadata.reward,
            "status": metadata.status,
            "result": metadata.result,
            "deck_signature": metadata.deck_signature,
            "opponent_deck_signature": metadata.opponent_deck_signature,
            "opponent_deck_ids": [int(card_id) for card_id in metadata.opponent_deck_ids],
            "godview_opp_hand_ids": [
                int(card_id) for card_id in metadata.godview_opp_hand_ids
            ],
            "godview_opp_hand_available": metadata.godview_opp_hand_available,
            "godview_opp_hand_count_match": metadata.godview_opp_hand_count_match,
            "remaining_overage_time": _float_value(
                observation.get("remainingOverageTime")
            ),
            "search_begin_input": _optional_string(
                observation.get("search_begin_input")
            ),
            "action": [int(index) for index in action],
            "action_length": len(action),
            "is_forced": forced,
            "terminal_prize_diff": None,
        }
    )


def _step_row_metadata(
    replay_path: Path,
    *,
    pending_observation: PendingObservation,
    metadata: dict[int, Row],
) -> StepRowMetadata:
    player_index = pending_observation.player_index
    player_meta = metadata.get(player_index, {})
    opponent_meta = metadata.get(1 - player_index, {})
    return StepRowMetadata(
        date=str(player_meta.get("date", replay_path.parent.name)),
        episode_id=int(player_meta.get("episode_id", replay_path.stem)),
        replay_path=deck_records.display_path(replay_path),
        step_index=pending_observation.step_index,
        player_index=player_index,
        team_name=str(player_meta.get("team_name", "")),
        opponent_team_name=str(opponent_meta.get("team_name", "")),
        reward=_float_value(player_meta.get("reward")),
        status=str(player_meta.get("status", "")),
        result=str(player_meta.get("result", "")),
        deck_signature=str(player_meta.get("deck_signature", "")),
        opponent_deck_signature=str(player_meta.get("opponent_deck_signature", "")),
        opponent_deck_ids=tuple(
            int(card_id) for card_id in _sequence(player_meta.get("opponent_deck_ids"))
        ),
        godview_opp_hand_ids=tuple(
            int(card_id) for card_id in pending_observation.godview_opp_hand_ids
        ),
        godview_opp_hand_available=pending_observation.godview_opp_hand_available,
        godview_opp_hand_count_match=pending_observation.godview_opp_hand_count_match,
    )


def _add_select_fields(row: Row, select: Mapping[str, Any]) -> None:
    options = _sequence(select.get("option"))
    context_card = _mapping_or_none(select.get("contextCard"))
    effect_card = _mapping_or_none(select.get("effect"))
    row.update(
        {
            "select_type": _int_value(select.get("type")),
            "select_context": _int_value(select.get("context")),
            "select_min_count": _int_value(select.get("minCount")),
            "select_max_count": _int_value(select.get("maxCount")),
            "select_option_count": len(options),
            "remain_damage_counter": _int_value(select.get("remainDamageCounter")),
            "remain_energy_cost": _int_value(select.get("remainEnergyCost")),
            "context_card_id": _card_id(context_card),
            "context_card_serial": _card_serial(context_card),
            "effect_card_id": _card_id(effect_card),
            "effect_card_serial": _card_serial(effect_card),
            "select_deck_present": _is_sequence(select.get("deck")),
            "select_deck_ids": _card_ids(_sequence(select.get("deck"))),
        }
    )
    for source_name, column_name in OPTION_FIELDS:
        row[f"option_{column_name}"] = [
            _optional_int(_mapping(option).get(source_name)) for option in options
        ]


def _add_current_fields(row: Row, current: Mapping[str, Any]) -> None:
    players = [_mapping(player) for player in _sequence(current.get("players"))]
    prize_counts = [len(_sequence(player.get("prize"))) for player in players[:2]]
    your_index = _int_value(current.get("yourIndex"))
    opponent_index = 1 - your_index if your_index in (0, 1) else -1
    row.update(
        {
            "current_prize_diff": (
                prize_counts[opponent_index] - prize_counts[your_index]
                if your_index in (0, 1) and len(prize_counts) >= 2
                else None
            ),
            "looking_ids": _card_ids(_sequence(current.get("looking"))),
            "stadium_ids": _card_ids(_sequence(current.get("stadium"))),
            "turn": _int_value(current.get("turn")),
            "turn_action_count": _int_value(current.get("turnActionCount")),
            "your_index": your_index,
            "first_player": _int_value(current.get("firstPlayer")),
            "state_result": _int_value(current.get("result")),
            "supporter_played": bool(current.get("supporterPlayed", False)),
            "stadium_played": bool(current.get("stadiumPlayed", False)),
            "energy_attached": bool(current.get("energyAttached", False)),
            "retreated": bool(current.get("retreated", False)),
        }
    )
    for player_index in range(2):
        player = players[player_index] if player_index < len(players) else {}
        _add_player_fields(row, player_index, player)


def _add_player_fields(row: Row, player_index: int, player: Mapping[str, Any]) -> None:
    prefix = f"player{player_index}"
    active = [_mapping(pokemon) for pokemon in _sequence(player.get("active"))]
    bench = [_mapping(pokemon) for pokemon in _sequence(player.get("bench"))]
    row.update(
        {
            f"{prefix}_deck_count": _int_value(player.get("deckCount")),
            f"{prefix}_hand_count": _int_value(player.get("handCount")),
            f"{prefix}_bench_max": _int_value(player.get("benchMax")),
            f"{prefix}_poisoned": bool(player.get("poisoned", False)),
            f"{prefix}_burned": bool(player.get("burned", False)),
            f"{prefix}_asleep": bool(player.get("asleep", False)),
            f"{prefix}_paralyzed": bool(player.get("paralyzed", False)),
            f"{prefix}_confused": bool(player.get("confused", False)),
            f"{prefix}_hand_ids": _card_ids(_sequence(player.get("hand"))),
            f"{prefix}_discard_ids": _card_ids(_sequence(player.get("discard"))),
            f"{prefix}_prize_ids": _card_ids(_sequence(player.get("prize"))),
        }
    )
    _add_pokemon_zone_fields(row, f"{prefix}_active", active)
    _add_pokemon_zone_fields(row, f"{prefix}_bench", bench)


def _add_pokemon_zone_fields(
    row: Row,
    prefix: str,
    pokemon: Sequence[Mapping[str, Any]],
) -> None:
    row[f"{prefix}_ids"] = [_int_value(card.get("id")) for card in pokemon]
    row[f"{prefix}_serials"] = [_int_value(card.get("serial")) for card in pokemon]
    row[f"{prefix}_hp"] = [_int_value(card.get("hp")) for card in pokemon]
    row[f"{prefix}_max_hp"] = [_int_value(card.get("maxHp")) for card in pokemon]
    row[f"{prefix}_appear_this_turn"] = [
        bool(card.get("appearThisTurn", False)) for card in pokemon
    ]
    row[f"{prefix}_energy_counts"] = [
        _energy_counts(_sequence(card.get("energies"))) for card in pokemon
    ]
    row[f"{prefix}_tool_counts"] = [
        len(_sequence(card.get("tools"))) for card in pokemon
    ]
    row[f"{prefix}_evolution_depths"] = [
        len(_sequence(card.get("preEvolution"))) for card in pokemon
    ]


def _player_schema_fields(player_index: int) -> list[pa.Field]:
    prefix = f"player{player_index}"
    fields = [
        pa.field(f"{prefix}_deck_count", pa.int16()),
        pa.field(f"{prefix}_hand_count", pa.int16()),
        pa.field(f"{prefix}_bench_max", pa.int16()),
        pa.field(f"{prefix}_poisoned", pa.bool_()),
        pa.field(f"{prefix}_burned", pa.bool_()),
        pa.field(f"{prefix}_asleep", pa.bool_()),
        pa.field(f"{prefix}_paralyzed", pa.bool_()),
        pa.field(f"{prefix}_confused", pa.bool_()),
        pa.field(f"{prefix}_hand_ids", pa.list_(pa.int32())),
        pa.field(f"{prefix}_discard_ids", pa.list_(pa.int32())),
        pa.field(f"{prefix}_prize_ids", pa.list_(pa.int32())),
    ]
    for zone in ("active", "bench"):
        zone_prefix = f"{prefix}_{zone}"
        fields.extend(
            [
                pa.field(f"{zone_prefix}_ids", pa.list_(pa.int32())),
                pa.field(f"{zone_prefix}_serials", pa.list_(pa.int32())),
                pa.field(f"{zone_prefix}_hp", pa.list_(pa.int16())),
                pa.field(f"{zone_prefix}_max_hp", pa.list_(pa.int16())),
                pa.field(f"{zone_prefix}_appear_this_turn", pa.list_(pa.bool_())),
                pa.field(
                    f"{zone_prefix}_energy_counts",
                    pa.list_(pa.list_(pa.int16())),
                ),
                pa.field(f"{zone_prefix}_tool_counts", pa.list_(pa.int16())),
                pa.field(f"{zone_prefix}_evolution_depths", pa.list_(pa.int16())),
            ]
        )
    return fields


def _initial_contexts(metadata: Mapping[int, Row]) -> dict[int, GameContext]:
    contexts: dict[int, GameContext] = {}
    for player_index in range(2):
        context = GameContext(player_index=player_index)
        context.set_own_deck(
            [
                int(card_id)
                for card_id in _sequence(metadata.get(player_index, {}).get("deck_ids"))
            ]
        )
        contexts[player_index] = context
    return contexts


def _add_context_fields(row: Row, context_features: GameContextFeatures) -> None:
    fields = context_features.row_fields()
    if len(fields["history_counts"]) != HISTORY_COUNTER_SIZE:
        raise ValueError("history context width mismatch")
    if len(fields["deck_flow_counts"]) != DECK_FLOW_FEATURE_SIZE:
        raise ValueError("deck-flow context width mismatch")
    row.update(fields)


def _add_chosen_effect_fields(
    row: Row,
    *,
    before_side: Mapping[str, Any],
    after_side: Mapping[str, Any],
    action: Sequence[int],
) -> None:
    before_observation = _mapping(before_side.get("observation"))
    after_observation = _mapping(after_side.get("observation"))
    logs = _sequence(after_observation.get("logs"))
    if not logs:
        row["chosen_effect_features"] = [0.0] * DYNAMIC_EFFECT_FEATURE_SIZE
        row["chosen_effect_mask"] = False
        return

    before_state = _namespace_tree(_mapping(before_observation.get("current")))
    after_state = _namespace_tree(_mapping(after_observation.get("current")))
    summary = parse_effect_logs(
        [_log_namespace(log) for log in logs],
        before_state=before_state,
        after_state=after_state,
    )
    feature_row = make_dynamic_effect_feature_row(
        select=tuple(int(index) for index in action),
        option_types=_selected_option_values(row, "option_type", action),
        attack_ids=_selected_option_values(row, "option_attack_id", action),
        card_ids=_selected_option_values(row, "option_card_id", action),
        summary=summary,
        before_state=before_state,
        after_state=after_state,
        perspective_player=_int_value(row.get("player_index")),
    )
    row["chosen_effect_features"] = [float(value) for value in feature_row.vector]
    row["chosen_effect_mask"] = True


def _selected_option_values(
    row: Mapping[str, Any],
    field_name: str,
    action: Sequence[int],
) -> tuple[int, ...]:
    values = _sequence(row.get(field_name))
    selected: list[int] = []
    for option_index in action:
        if 0 <= int(option_index) < len(values):
            value = values[int(option_index)]
            selected.append(0 if value is None else int(value))
    return tuple(selected)


def _namespace_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{str(key): _namespace_tree(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace_tree(item) for item in value]
    return value


def _log_namespace(log: Any) -> SimpleNamespace:
    raw = _mapping(log)
    values = {field_name: raw.get(field_name) for field_name in LOG_FIELDS}
    return SimpleNamespace(**values)


def _has_active_select(side: Mapping[str, Any]) -> bool:
    if str(side.get("status", "")) != "ACTIVE":
        return False
    observation = _mapping(side.get("observation"))
    return isinstance(observation.get("select"), Mapping)


def _godview_opp_hand_label(
    sides: Sequence[Mapping[str, Any]],
    *,
    player_index: int,
) -> dict[str, Any]:
    opponent_index = 1 - player_index
    default = {
        "godview_opp_hand_ids": (),
        "godview_opp_hand_available": False,
        "godview_opp_hand_count_match": False,
    }
    if player_index < 0 or player_index >= len(sides):
        return default
    if opponent_index < 0 or opponent_index >= len(sides):
        return default

    active_observation = _mapping(sides[player_index].get("observation"))
    active_current = _mapping(active_observation.get("current"))
    expected_count = _player_hand_count(active_current, opponent_index)

    opponent_observation = _mapping(sides[opponent_index].get("observation"))
    opponent_current = _mapping(opponent_observation.get("current"))
    opponent_player = _player_state(opponent_current, opponent_index)
    hand = opponent_player.get("hand")
    if not _is_sequence(hand):
        return default
    hand_cards = _sequence(hand)
    hand_ids = tuple(_card_ids(hand_cards))
    if any(card_id <= 0 for card_id in hand_ids):
        return default
    return {
        "godview_opp_hand_ids": hand_ids,
        "godview_opp_hand_available": True,
        "godview_opp_hand_count_match": len(hand_ids) == expected_count,
    }


def _player_hand_count(current: Mapping[str, Any], player_index: int) -> int:
    return _int_value(_player_state(current, player_index).get("handCount"))


def _player_state(current: Mapping[str, Any], player_index: int) -> Mapping[str, Any]:
    players = _sequence(current.get("players"))
    if 0 <= player_index < len(players):
        return _mapping(players[player_index])
    return {}


def _terminal_prize_diffs(sides: Sequence[Mapping[str, Any]]) -> dict[int, int | None]:
    diffs: dict[int, int | None] = {}
    for player_index, side in enumerate(sides):
        current = _mapping(_mapping(side.get("observation")).get("current"))
        players = [_mapping(player) for player in _sequence(current.get("players"))]
        if len(players) < 2:
            diffs[player_index] = None
            continue
        prize_counts = [len(_sequence(player.get("prize"))) for player in players[:2]]
        diffs[player_index] = prize_counts[1 - player_index] - prize_counts[player_index]
    return diffs


def _iter_json_array_items(file_obj: TextIO, chunk_size: int) -> Iterable[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    while True:
        buffer = _read_until_meaningful(file_obj, buffer, chunk_size)
        stripped = buffer.lstrip()
        if stripped.startswith("]"):
            return
        buffer = stripped[1:].lstrip() if stripped.startswith(",") else stripped
        while True:
            try:
                item, end = decoder.raw_decode(buffer)
                yield item
                buffer = buffer[end:]
                break
            except json.JSONDecodeError:
                chunk = file_obj.read(chunk_size)
                if not chunk:
                    raise
                buffer = (buffer + chunk).lstrip()


def _read_until_meaningful(file_obj: TextIO, buffer: str, chunk_size: int) -> str:
    while not buffer.strip():
        chunk = file_obj.read(chunk_size)
        if not chunk:
            return buffer
        buffer += chunk
    return buffer


def _read_prefix_through_steps_key(replay_path: Path, *, chunk_size: int) -> bytes:
    with replay_path.open("rb") as file_obj:
        buffer = bytearray()
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                raise ValueError(f"steps key not found in replay: {replay_path}")
            buffer.extend(chunk)
            index = buffer.find(STEPS_KEY)
            if index >= 0:
                return bytes(buffer[:index])


def _steps_array_start(replay_path: Path, *, chunk_size: int) -> int:
    absolute_base = 0
    buffer = b""
    key_position: int | None = None
    with replay_path.open("rb") as file_obj:
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                raise ValueError(f"steps array not found in replay: {replay_path}")
            buffer += chunk
            if key_position is None:
                local_key = buffer.find(STEPS_KEY)
                if local_key >= 0:
                    key_position = absolute_base + local_key
            if key_position is not None:
                key_local = key_position - absolute_base
                colon = buffer.find(b":", key_local + len(STEPS_KEY))
                if colon >= 0:
                    bracket = buffer.find(b"[", colon + 1)
                    if bracket >= 0:
                        return absolute_base + bracket
            if len(buffer) <= chunk_size:
                continue
            trim = len(buffer) - chunk_size
            if key_position is not None:
                trim = min(trim, max(0, key_position - absolute_base))
            absolute_base += trim
            buffer = buffer[trim:]


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str)


def _int_sequence(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, Sequence) or isinstance(value, str):
        return None
    if not all(type(item) is int for item in value):
        return None
    return tuple(int(item) for item in value)


def _card_ids(cards: Sequence[Any]) -> list[int]:
    return [_card_id(_mapping_or_none(card)) for card in cards]


def _card_id(card: Mapping[str, Any] | None) -> int:
    if card is None:
        return 0
    return _int_value(card.get("id"))


def _card_serial(card: Mapping[str, Any] | None) -> int:
    if card is None:
        return 0
    return _int_value(card.get("serial"))


def _energy_counts(energies: Sequence[Any]) -> list[int]:
    counts = [0] * ENERGY_TYPE_COUNT
    for energy in energies:
        index = _int_value(energy)
        if 0 <= index < ENERGY_TYPE_COUNT:
            counts[index] += 1
    return counts


def _int_value(value: Any) -> int:
    return int(value) if value is not None else 0


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None


def _float_value(value: Any) -> float | None:
    return float(value) if value is not None else None
