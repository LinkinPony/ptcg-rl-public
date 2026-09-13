"""Replay and deck-record helpers for Kaggle deck environment analysis."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from ptcg_rl.decks.identity import canonical_signature, parse_canonical_signature

try:
    import orjson
except ImportError:  # pragma: no cover - depends on local analysis environment.
    orjson = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class CardMeta:
    """Small card metadata record keyed by competition card id."""

    card_id: int
    name: str
    expansion: str
    collection_no: str
    stage_or_type: str
    rule: str
    category: str
    card_type: str

    @property
    def is_pokemon(self) -> bool:
        """Return whether the card appears to be a Pokemon card."""
        value = self.stage_or_type.lower()
        return (
            "basic pok" in value
            or "stage 1" in value
            or "stage 2" in value
            or "mega" in value
        )


def episode_side_rows(
    *,
    replay_path: Path,
    replay: dict[str, Any],
    card_meta: dict[int, CardMeta],
    known_decks: dict[str, str],
) -> list[dict[str, Any]]:
    """Return per-player deck observations from one replay."""
    decks = registered_decks(replay)
    return _side_rows_from_parts(
        replay_path=replay_path,
        replay=replay,
        decks=decks,
        card_meta=card_meta,
        known_decks=known_decks,
    )


def episode_side_rows_from_data(
    *,
    replay_path: Path,
    replay: dict[str, Any],
    size_bytes: int,
    card_meta: dict[int, CardMeta],
    known_decks: dict[str, str],
) -> list[dict[str, Any]]:
    """Return side rows for a replay loaded from a compressed member."""
    return _side_rows_from_parts(
        replay_path=replay_path,
        replay=replay,
        decks=registered_decks(replay),
        card_meta=card_meta,
        known_decks=known_decks,
        size_bytes=size_bytes,
    )


def fast_episode_side_rows(
    *,
    replay_path: Path,
    card_meta: dict[int, CardMeta],
    known_decks: dict[str, str],
    prefix_bytes: int,
    include_step_count: bool,
    require_first_player: bool = False,
) -> list[dict[str, Any]] | None:
    """Return side rows from a replay prefix, or None if a full parse is needed."""
    data = _read_prefix(replay_path, prefix_bytes)
    return fast_episode_side_rows_from_bytes(
        data=data,
        replay_path=replay_path,
        size_bytes=replay_path.stat().st_size,
        card_meta=card_meta,
        known_decks=known_decks,
        include_step_count=include_step_count,
        require_first_player=require_first_player,
    )


def fast_episode_side_rows_from_bytes(
    *,
    data: bytes,
    replay_path: Path,
    size_bytes: int,
    card_meta: dict[int, CardMeta],
    known_decks: dict[str, str],
    include_step_count: bool,
    require_first_player: bool = False,
) -> list[dict[str, Any]] | None:
    """Return side rows from in-memory replay prefix bytes.

    The source path is an identity hint and need not exist.  This lets compact
    corpus builders inspect ZIP/TAR members without extracting them first.
    """
    decks = _fast_registered_decks(data)
    if len(decks) < 2:
        return None
    rewards = _fast_array(data, b'"rewards"')
    statuses = _fast_array(data, b'"statuses"')
    team_names = _fast_team_names(data)
    episode_id = _fast_episode_id(data)
    first_player = _fast_first_player(data)
    if rewards is None or statuses is None or team_names is None or episode_id is None:
        return None
    if require_first_player and first_player is None:
        return None
    replay_stub = {
        "info": {"EpisodeId": episode_id, "TeamNames": team_names},
        "rewards": rewards,
        "statuses": statuses,
        "steps": [None] * _fast_step_count(data, include_step_count),
    }
    return _side_rows_from_parts(
        replay_path=replay_path,
        replay=replay_stub,
        decks=decks,
        card_meta=card_meta,
        known_decks=known_decks,
        first_player=first_player,
        size_bytes=size_bytes,
    )


def fast_episode_identity_and_decks(
    data: bytes,
) -> tuple[int, dict[int, list[int]]] | None:
    """Extract one episode ID and both exact registered decks from a prefix.

    This deliberately exposes only the stable, small subset needed by
    streaming catalog builders. Callers may retry with a larger prefix when
    the initial replay prefix does not contain both registration actions.
    """
    episode_id = _fast_episode_id(data)
    decks = _fast_registered_decks(data)
    if episode_id is None or len(decks) < 2:
        return None
    return episode_id, decks


def registered_decks(replay: dict[str, Any]) -> dict[int, list[int]]:
    """Extract first 60-card registration action for each player."""
    decks: dict[int, list[int]] = {}
    for step in replay.get("steps") or []:
        if not isinstance(step, list):
            continue
        for player_index, side in enumerate(step):
            if player_index in decks or not isinstance(side, dict):
                continue
            action = side.get("action")
            if is_deck_registration(action):
                decks[player_index] = cast(list[int], action)
        if len(decks) >= 2:
            break
    return decks


def _side_rows_from_parts(
    *,
    replay_path: Path,
    replay: dict[str, Any],
    decks: dict[int, list[int]],
    card_meta: dict[int, CardMeta],
    known_decks: dict[str, str],
    first_player: int | None = None,
    size_bytes: int | None = None,
) -> list[dict[str, Any]]:
    date = replay_path.parent.name
    episode_id = episode_id_from_replay(replay, replay_path)
    rewards = rewards_from_replay(replay)
    statuses = statuses_from_replay(replay)
    team_names = team_names_from_replay(replay)
    resolved_first_player = (
        first_player if first_player is not None else first_player_from_replay(replay)
    )
    rows: list[dict[str, Any]] = []

    for player_index in sorted(decks):
        deck_ids = decks[player_index]
        signature = deck_signature(deck_ids)
        opponent_index = 1 - player_index
        opponent_ids = decks.get(opponent_index)
        opponent_signature = deck_signature(opponent_ids) if opponent_ids else ""
        label = deck_label(signature, deck_ids, card_meta, known_decks)
        opponent_label = (
            deck_label(opponent_signature, opponent_ids, card_meta, known_decks)
            if opponent_ids
            else ""
        )
        reward = list_float(rewards, player_index)
        status = list_str(statuses, player_index)
        rows.append(
            {
                "date": date,
                "episode_id": episode_id,
                "player_index": player_index,
                "first_player_index": resolved_first_player,
                "went_first": (
                    None
                    if resolved_first_player is None
                    else player_index == resolved_first_player
                ),
                "team_name": list_str(team_names, player_index),
                "reward": reward,
                "status": status,
                "result": result_label(reward, status),
                "deck_signature": signature,
                "deck_ids": list(deck_ids),
                "deck_hash": signature_hash(signature),
                "deck_label": label,
                "known_deck": known_decks.get(signature, ""),
                "opponent_player_index": opponent_index,
                "opponent_deck_signature": opponent_signature,
                "opponent_deck_ids": list(opponent_ids or ()),
                "opponent_deck_hash": (
                    signature_hash(opponent_signature) if opponent_signature else ""
                ),
                "opponent_deck_label": opponent_label,
                "unique_card_ids": len(Counter(deck_ids)),
                "total_cards": len(deck_ids),
                "pokemon_summary": pokemon_summary(Counter(deck_ids), card_meta),
                "top_cards": top_cards(Counter(deck_ids), card_meta, limit=8),
                "step_count": len(replay.get("steps") or []),
                "size_bytes": (
                    replay_path.stat().st_size if size_bytes is None else size_bytes
                ),
                "path": display_path(replay_path),
            }
        )
    return rows


def _read_prefix(path: Path, prefix_bytes: int) -> bytes:
    with path.open("rb") as file_obj:
        return file_obj.read(prefix_bytes)


def _fast_episode_id(data: bytes) -> int | None:
    match = re.search(rb'"EpisodeId"\s*:\s*(\d+)', data)
    if match is None:
        return None
    return int(match.group(1))


def _fast_team_names(data: bytes) -> list[str] | None:
    value = _fast_array(data, b'"TeamNames"')
    if not isinstance(value, list):
        return None
    return [str(item) for item in value]


def _fast_first_player(data: bytes) -> int | None:
    """Return the first resolved engine first-player value in a replay prefix."""
    for match in re.finditer(rb'"firstPlayer"\s*:\s*(-?\d+)', data):
        value = int(match.group(1))
        if value in (0, 1):
            return value
    return None


def _fast_array(data: bytes, key: bytes) -> list[Any] | None:
    start = _value_start(data, key)
    if start is None or start >= len(data) or data[start] != ord("["):
        return None
    end = _matching_bracket(data, start)
    if end is None:
        return None
    value = _json_loads(data[start : end + 1])
    return value if isinstance(value, list) else None


def _fast_registered_decks(data: bytes) -> dict[int, list[int]]:
    decks: dict[int, list[int]] = {}
    position = 0
    while len(decks) < 2:
        key_index = data.find(b'"action"', position)
        if key_index < 0:
            break
        position = key_index + len(b'"action"')
        start = _value_start_after_index(data, position)
        if start is None or start >= len(data) or data[start] != ord("["):
            continue
        end = _matching_bracket(data, start)
        if end is None:
            break
        position = end + 1
        value = _json_loads(data[start : end + 1])
        _add_fast_deck_value(decks, value)
    return decks


def _add_fast_deck_value(decks: dict[int, list[int]], value: Any) -> None:
    if _is_int_deck(value):
        decks.setdefault(len(decks), list(value))
        return
    if not isinstance(value, list):
        return
    for nested in value:
        if len(decks) >= 2:
            break
        if _is_int_deck(nested):
            decks.setdefault(len(decks), list(nested))


def _is_int_deck(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 60
        and all(type(item) is int for item in value)
    )


def _value_start(data: bytes, key: bytes) -> int | None:
    key_index = data.find(key)
    if key_index < 0:
        return None
    return _value_start_after_index(data, key_index + len(key))


def _value_start_after_index(data: bytes, index: int) -> int | None:
    colon_index = data.find(b":", index)
    if colon_index < 0:
        return None
    position = colon_index + 1
    while position < len(data) and data[position] in b" \n\r\t":
        position += 1
    return position


def _matching_bracket(data: bytes, start: int) -> int | None:
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(data)):
        char = data[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == ord("\\"):
                escaped = True
            elif char == ord('"'):
                in_string = False
            continue
        if char == ord('"'):
            in_string = True
        elif char == ord("["):
            depth += 1
        elif char == ord("]"):
            depth -= 1
            if depth == 0:
                return index
    return None


def _json_loads(data: bytes) -> Any:
    if orjson is not None:
        return orjson.loads(data)
    return json.loads(data.decode("utf-8"))


def _fast_step_count(data: bytes, include_step_count: bool) -> int:
    if not include_step_count:
        return 0
    return data.count(b'"step"')


def is_deck_registration(action: Any) -> bool:
    """Return whether an action is a competition deck registration."""
    return (
        isinstance(action, list)
        and len(action) == 60
        and all(type(value) is int for value in action)
    )


def deck_signature(card_ids: list[int]) -> str:
    """Return a stable sorted card-count signature for a 60-card deck."""
    return canonical_signature(card_ids)


def deck_label(
    signature: str,
    card_ids: list[int],
    card_meta: dict[int, CardMeta],
    known_decks: dict[str, str],
) -> str:
    """Return an exact known-deck label or a readable Pokemon summary label."""
    known_name = known_decks.get(signature)
    if known_name:
        return known_name
    counts = Counter(card_ids)
    pokemon = [
        (count, card_meta.get(card_id, missing_card(card_id)).name)
        for card_id, count in counts.items()
        if card_meta.get(card_id, missing_card(card_id)).is_pokemon
    ]
    if not pokemon:
        return f"unknown_{signature_hash(signature)}"
    pokemon.sort(key=lambda item: (-item[0], item[1]))
    return " / ".join(name for _, name in pokemon[:3])


def signature_counts(signature: str) -> Counter[int]:
    """Parse a deck signature into card counts."""
    if not signature:
        return Counter()
    return Counter(parse_canonical_signature(signature).card_ids)


def pokemon_summary(counts: Counter[int], card_meta: dict[int, CardMeta]) -> str:
    """Return a compact Pokemon-only card summary for one deck."""
    pokemon = [
        (count, card_meta.get(card_id, missing_card(card_id)).name)
        for card_id, count in counts.items()
        if card_meta.get(card_id, missing_card(card_id)).is_pokemon
    ]
    pokemon.sort(key=lambda item: (-item[0], item[1]))
    return "; ".join(f"{name} x{count}" for count, name in pokemon[:8])


def top_cards(
    counts: Counter[int],
    card_meta: dict[int, CardMeta],
    *,
    limit: int,
) -> str:
    """Return the most common cards in a signature."""
    cards = [
        (count, card_id, card_meta.get(card_id, missing_card(card_id)).name)
        for card_id, count in counts.items()
    ]
    cards.sort(key=lambda item: (-item[0], item[2], item[1]))
    return "; ".join(f"{name} x{count}" for count, _, name in cards[:limit])


def load_known_decks(
    known_deck_paths: dict[str, Path],
    known_deck_dirs: list[Path],
) -> dict[str, str]:
    """Load exact known deck signatures from configured files and directories."""
    known: dict[str, str] = {}
    for name, raw_path in sorted(known_deck_paths.items()):
        path = repo_path(raw_path)
        if path.exists():
            known[deck_signature(read_deck(path))] = name

    for raw_dir in known_deck_dirs:
        directory = repo_path(raw_dir)
        if not directory.exists():
            continue
        for deck_path in sorted(directory.glob("*/deck.csv")):
            known.setdefault(
                deck_signature(read_deck(deck_path)), deck_path.parent.name
            )
    return known


def read_deck(path: Path) -> list[int]:
    """Read a one-card-id-per-line competition deck file."""
    values: list[int] = []
    with path.open(encoding="utf-8", newline="") as file_obj:
        for raw_line in file_obj:
            value = raw_line.strip().strip(",")
            if value:
                values.append(int(value))
    if len(values) != 60:
        raise ValueError(f"deck must contain 60 card ids, got {len(values)}: {path}")
    return values


def load_card_meta(path: Path) -> dict[int, CardMeta]:
    """Load first card-metadata row for each card id."""
    cards: dict[int, CardMeta] = {}
    with path.open(encoding="utf-8-sig", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            card_id = int(row["Card ID"])
            if card_id in cards:
                continue
            cards[card_id] = CardMeta(
                card_id=card_id,
                name=row["Card Name"],
                expansion=row["Expansion"],
                collection_no=row["Collection No."],
                stage_or_type=stage_or_type(row),
                rule=row["Rule"],
                category=row["Category"],
                card_type=row["Type"],
            )
    return cards


def stage_or_type(row: dict[str, str]) -> str:
    """Read the stage/type column without embedding non-ASCII source text."""
    for key, value in row.items():
        if key.startswith("Stage ("):
            return value
    return ""


def missing_card(card_id: int) -> CardMeta:
    """Return placeholder metadata for an unknown card id."""
    return CardMeta(
        card_id=card_id,
        name=f"Card {card_id}",
        expansion="",
        collection_no="",
        stage_or_type="",
        rule="",
        category="",
        card_type="",
    )


def iter_replay_paths(replay_root: Path, selected_dates: set[str]) -> Iterable[Path]:
    """Yield replay JSON paths from selected date directories."""
    if not replay_root.exists():
        raise FileNotFoundError(f"replay_root does not exist: {replay_root}")
    for date_dir in sorted(path for path in replay_root.iterdir() if path.is_dir()):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date_dir.name):
            continue
        if selected_dates and date_dir.name not in selected_dates:
            continue
        yield from sorted(date_dir.glob("*.json"))


def episode_id_from_replay(replay: dict[str, Any], replay_path: Path) -> int:
    """Return the Kaggle episode id from replay info or filename."""
    info = replay.get("info") or {}
    raw_episode_id = info.get("EpisodeId")
    if raw_episode_id is not None:
        return int(raw_episode_id)
    return int(replay_path.stem)


def team_names_from_replay(replay: dict[str, Any]) -> list[str]:
    """Return team names from replay info."""
    info = replay.get("info") or {}
    team_names = info.get("TeamNames") or []
    return [str(team_name) for team_name in team_names]


def rewards_from_replay(replay: dict[str, Any]) -> list[Any]:
    """Return final rewards from replay-level fields or the last step."""
    rewards = replay.get("rewards")
    if isinstance(rewards, list):
        return rewards
    steps = replay.get("steps") or []
    if not steps:
        return []
    last_step = steps[-1]
    if not isinstance(last_step, list):
        return []
    return [
        side.get("reward") if isinstance(side, dict) else None for side in last_step
    ]


def statuses_from_replay(replay: dict[str, Any]) -> list[str]:
    """Return final statuses from replay-level fields or the last step."""
    statuses = replay.get("statuses")
    if isinstance(statuses, list):
        return [str(status) for status in statuses]
    steps = replay.get("steps") or []
    if not steps or not isinstance(steps[-1], list):
        return []
    return [str(side.get("status", "")) for side in steps[-1] if isinstance(side, dict)]


def first_player_from_replay(replay: dict[str, Any]) -> int | None:
    """Return the first resolved engine first-player index from replay steps."""
    for step in replay.get("steps") or []:
        if not isinstance(step, list):
            continue
        for side in step:
            if not isinstance(side, dict):
                continue
            observation = side.get("observation")
            if not isinstance(observation, dict):
                continue
            current = observation.get("current")
            if not isinstance(current, dict):
                continue
            value = current.get("firstPlayer")
            if type(value) is int and value in (0, 1):
                return value
    return None


def list_float(values: list[Any], index: int) -> float | None:
    """Read one optional float from a positional list."""
    if index >= len(values):
        return None
    value = values[index]
    if value is None or value == "":
        return None
    return float(value)


def list_str(values: list[Any], index: int) -> str:
    """Read one string from a positional list."""
    if index >= len(values):
        return ""
    return str(values[index])


def result_label(reward: float | None, status: str) -> str:
    """Map reward and terminal status to a compact result label."""
    if reward is None:
        return "other"
    if status and status != "DONE":
        return "other"
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def signature_hash(signature: str) -> str:
    """Return a short stable hash for a deck signature."""
    return hashlib.sha1(signature.encode("utf-8")).hexdigest()[:12]


def repo_path(path: Path) -> Path:
    """Resolve a path relative to the repository root."""
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def display_path(path: Path) -> str:
    """Return a repository-relative path when possible."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
