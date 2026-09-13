"""BC-compatible trajectory row buffering for vectorized rollouts."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.context import context_features_from_observation
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records
from ptcg_rl.engine.vector_battle import DeckPair, FinishedGame
from ptcg_rl.rl.rollout import RolloutDecision

RL_STEP_ROW_SCHEMA_VERSION = step_records.STEP_ROW_SCHEMA_VERSION + 1
RL_EXTENSION_FIELDS: tuple[pa.Field, ...] = (
    pa.field("game_id", pa.string()),
    pa.field("decision_index", pa.int32()),
    pa.field("action_logprob", pa.float32()),
    pa.field("value_pred", pa.float32()),
    pa.field("sampling_temperature", pa.float32()),
    pa.field("policy_version", pa.string()),
    pa.field("opponent_name", pa.string()),
    pa.field("opponent_tier", pa.int8()),
    pa.field("rollout_seat", pa.int8()),
    pa.field("episode_length", pa.int32()),
)
RL_EXTENSION_FIELD_NAMES = tuple(field.name for field in RL_EXTENSION_FIELDS)
GAME_ROW_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string()),
        pa.field("episode_id", pa.int64()),
        pa.field("winner_index", pa.int8()),
        pa.field("steps", pa.int32()),
        pa.field("rows", pa.int32()),
        pa.field("seat0_decisions", pa.int32()),
        pa.field("seat1_decisions", pa.int32()),
        pa.field("policy_version", pa.string()),
        pa.field("opponent_name", pa.string()),
        pa.field("opponent_tier", pa.int8()),
        pa.field("deck0_signature", pa.string()),
        pa.field("deck1_signature", pa.string()),
        pa.field("deck0_hash", pa.string()),
        pa.field("deck1_hash", pa.string()),
    ]
)


@dataclass(frozen=True)
class CompletedTrajectory:
    """One finalized game worth of trajectory rows."""

    game_id: str
    rows: tuple[step_records.Row, ...]
    game_row: step_records.Row


@dataclass(frozen=True)
class TrajectoryWriteResult:
    """Files and manifest emitted by ``TrajectoryShardWriter.close``."""

    manifest_path: Path
    games_path: Path
    manifest: Mapping[str, Any]


@dataclass
class _GameBuffer:
    episode_id: int
    deck_pair: DeckPair
    policy_version: str
    opponent_name: str
    opponent_tier: int
    rows: list[step_records.Row] = field(default_factory=list)
    decision_counts: list[int] = field(default_factory=lambda: [0, 0])
    hand_snapshots: dict[int, tuple[int, ...]] = field(default_factory=dict)


class TrajectoryRecorder:
    """Buffer rollout decisions by game and finalize BC-compatible rows."""

    def __init__(
        self,
        *,
        date: str | None = None,
        policy_name: str = "policy",
        policy_version: str = "",
        opponent_name: str = "self_play",
        opponent_tier: int = -1,
        initial_episode_id: int = 0,
    ) -> None:
        """Initialize recorder metadata shared by emitted rows."""
        if not -128 <= int(opponent_tier) <= 127:
            raise ValueError("opponent_tier must fit in int8")
        if initial_episode_id < 0:
            raise ValueError("initial_episode_id must be non-negative")
        self._date = date or datetime.now(UTC).date().isoformat()
        self._policy_name = str(policy_name)
        self._policy_version = str(policy_version)
        self._opponent_name = str(opponent_name)
        self._opponent_tier = int(opponent_tier)
        self._next_episode_id = int(initial_episode_id)
        self._buffers: dict[str, _GameBuffer] = {}
        self._completed: list[CompletedTrajectory] = []
        self._counters: Counter[str] = Counter()

    @property
    def counters(self) -> Counter[str]:
        """Return recorder counters."""
        return Counter(self._counters)

    @property
    def pending_game_count(self) -> int:
        """Return games currently buffered but not finalized."""
        return len(self._buffers)

    @property
    def completed_game_count(self) -> int:
        """Return finalized games waiting to be consumed."""
        return len(self._completed)

    def record(self, decision: RolloutDecision) -> None:
        """Record one non-forced policy decision."""
        if decision.public_event_delta is not None:
            raise ValueError(
                "BC-compatible Parquet rows cannot store recurrent public events"
            )
        if decision.seat not in (0, 1):
            raise ValueError(f"invalid rollout seat: {decision.seat}")
        buffer = self._buffer_for(decision)
        _update_hand_snapshots(buffer, decision.observation)
        decision_index = buffer.decision_counts[decision.seat]
        metadata = self._row_metadata(decision, buffer, decision_index=decision_index)
        row = step_records.row_from_observation(
            decision.observation,
            decision.action,
            metadata=metadata,
            context_features=context_features_from_observation(decision.observation),
            forced=False,
        )
        row.update(
            {
                "schema_version": RL_STEP_ROW_SCHEMA_VERSION,
                "game_id": decision.game_id,
                "decision_index": decision_index,
                "action_logprob": float(decision.action_logprob),
                "value_pred": float(decision.value_pred),
                "sampling_temperature": float(decision.sampling_temperature),
                "policy_version": buffer.policy_version,
                "opponent_name": buffer.opponent_name,
                "opponent_tier": buffer.opponent_tier,
                "rollout_seat": decision.seat,
                "episode_length": None,
            }
        )
        buffer.rows.append(row)
        buffer.decision_counts[decision.seat] += 1
        self._counters["recorded_rows"] += 1

    def finalize(self, finished: FinishedGame) -> None:
        """Finalize a completed game, backfilling terminal row fields."""
        buffer = self._buffers.pop(finished.game_id, None)
        if buffer is None:
            self._counters["finished_without_rows"] += 1
            return
        if finished.winner_index not in (0, 1, 2):
            self._discard_buffer(buffer, reason="invalid_winner")
            return

        terminal_prize_diffs = _terminal_prize_diffs(finished.observation)
        finalized_rows: list[step_records.Row] = []
        for row in buffer.rows:
            seat = int(row["player_index"])
            reward = _reward_for_seat(seat, finished.winner_index)
            row["reward"] = reward
            row["status"] = "DONE"
            row["result"] = _result_label(reward)
            row["terminal_prize_diff"] = terminal_prize_diffs.get(seat)
            row["episode_length"] = buffer.decision_counts[seat]
            finalized_rows.append(dict(row))

        completed = CompletedTrajectory(
            game_id=finished.game_id,
            rows=tuple(finalized_rows),
            game_row=_game_row(
                finished,
                buffer,
                rows=len(finalized_rows),
                policy_version=buffer.policy_version,
                opponent_name=buffer.opponent_name,
                opponent_tier=buffer.opponent_tier,
            ),
        )
        self._completed.append(completed)
        self._counters["finalized_games"] += 1
        self._counters["finalized_rows"] += len(finalized_rows)

    def discard(self, game_id: str, *, reason: str = "discarded") -> None:
        """Drop a buffered game without emitting rows."""
        buffer = self._buffers.pop(game_id, None)
        if buffer is None:
            self._counters["discard_missing_games"] += 1
            return
        self._discard_buffer(buffer, reason=reason)

    def pop_completed(self) -> tuple[CompletedTrajectory, ...]:
        """Return and clear completed trajectories."""
        completed = tuple(self._completed)
        self._completed.clear()
        return completed

    def _buffer_for(self, decision: RolloutDecision) -> _GameBuffer:
        normalized_deck_pair = _normalize_deck_pair(decision.deck_pair)
        buffer = self._buffers.get(decision.game_id)
        if buffer is not None:
            if buffer.deck_pair != normalized_deck_pair:
                raise ValueError(f"deck_pair changed for game_id={decision.game_id}")
            return buffer
        buffer = _GameBuffer(
            episode_id=self._next_episode_id,
            deck_pair=normalized_deck_pair,
            policy_version=self._policy_version,
            opponent_name=_decision_opponent_name(decision, self._opponent_name),
            opponent_tier=_decision_opponent_tier(decision, self._opponent_tier),
        )
        self._next_episode_id += 1
        self._buffers[decision.game_id] = buffer
        self._counters["started_games"] += 1
        return buffer

    def _row_metadata(
        self,
        decision: RolloutDecision,
        buffer: _GameBuffer,
        *,
        decision_index: int,
    ) -> step_records.StepRowMetadata:
        seat = decision.seat
        opponent_seat = 1 - seat
        hand_ids, hand_available, hand_count_match = _opponent_hand_label(
            buffer,
            decision.observation,
            player_index=seat,
        )
        return step_records.StepRowMetadata(
            date=self._date,
            episode_id=buffer.episode_id,
            replay_path="",
            step_index=_observation_step(
                decision.observation,
                fallback=decision_index,
            ),
            player_index=seat,
            team_name=self._team_name(decision.policy_role, buffer.policy_version),
            opponent_team_name=buffer.opponent_name,
            reward=None,
            status="",
            result="",
            deck_signature=_deck_signature(buffer.deck_pair[seat]),
            opponent_deck_signature=_deck_signature(buffer.deck_pair[opponent_seat]),
            opponent_deck_ids=buffer.deck_pair[opponent_seat],
            godview_opp_hand_ids=hand_ids,
            godview_opp_hand_available=hand_available,
            godview_opp_hand_count_match=hand_count_match,
        )

    def _team_name(self, policy_role: str, policy_version: str) -> str:
        label = self._policy_name if policy_role == "candidate" else policy_role
        return f"{label}@{policy_version}" if policy_version else label

    def _discard_buffer(self, buffer: _GameBuffer, *, reason: str) -> None:
        self._counters["discarded_games"] += 1
        self._counters["discarded_rows"] += len(buffer.rows)
        self._counters[f"discarded_{reason}"] += 1


class TrajectoryShardWriter:
    """Buffered Parquet writer for completed rollout trajectories."""

    def __init__(
        self,
        *,
        output_dir: Path,
        rows_per_shard: int = 65_536,
        compression: str = "zstd",
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a rollout trajectory writer."""
        if rows_per_shard <= 0:
            raise ValueError("rows_per_shard must be positive")
        self.output_dir = Path(output_dir)
        self.rows_per_shard = int(rows_per_shard)
        self.compression = str(compression)
        self.config = dict(config or {})
        self.metadata = dict(metadata or {})
        self.shard_dir = self.output_dir / "shards"
        self.manifest_path = self.output_dir / "manifest.json"
        self.games_path = self.output_dir / "games.parquet"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self._row_buffer: list[step_records.Row] = []
        self._game_rows: list[step_records.Row] = []
        self._shards: list[dict[str, Any]] = []
        self._closed = False
        self._row_schema = trajectory_row_schema()

    @property
    def shards(self) -> tuple[Mapping[str, Any], ...]:
        """Return already flushed row shard manifest entries."""
        return tuple(dict(shard) for shard in self._shards)

    def add_completed(
        self,
        trajectories: CompletedTrajectory | Sequence[CompletedTrajectory],
    ) -> None:
        """Add finalized trajectories, flushing row shards as needed."""
        self._raise_if_closed()
        if isinstance(trajectories, CompletedTrajectory):
            self._add_trajectory(trajectories)
            return
        for trajectory in trajectories:
            self._add_trajectory(trajectory)

    def close(self) -> TrajectoryWriteResult:
        """Flush all buffered rows, write games.parquet, and write manifest."""
        self._raise_if_closed()
        self._flush_rows(len(self._row_buffer))
        games_table = pa.Table.from_pylist(self._game_rows, schema=GAME_ROW_SCHEMA)
        _write_parquet_atomic(
            self.games_path,
            games_table,
            compression=self.compression,
        )
        manifest = self._manifest()
        _write_json_atomic(self.manifest_path, manifest)
        self._closed = True
        return TrajectoryWriteResult(
            manifest_path=self.manifest_path,
            games_path=self.games_path,
            manifest=manifest,
        )

    def _add_trajectory(self, trajectory: CompletedTrajectory) -> None:
        rows = [dict(row) for row in trajectory.rows]
        if self._row_buffer and len(self._row_buffer) + len(rows) > self.rows_per_shard:
            self._flush_rows(len(self._row_buffer))
        self._row_buffer.extend(rows)
        self._game_rows.append(dict(trajectory.game_row))
        while len(self._row_buffer) >= self.rows_per_shard:
            self._flush_rows(self.rows_per_shard)

    def _flush_rows(self, count: int) -> None:
        if count <= 0:
            return
        rows = self._row_buffer[:count]
        del self._row_buffer[:count]
        shard_index = len(self._shards)
        shard_path = self.shard_dir / f"steps-{shard_index:05d}.parquet"
        table = pa.Table.from_pylist(rows, schema=self._row_schema)
        _write_parquet_atomic(shard_path, table, compression=self.compression)
        self._shards.append(
            {
                "path": deck_records.display_path(shard_path),
                "rows": len(rows),
                "bytes": shard_path.stat().st_size,
            }
        )

    def _manifest(self) -> dict[str, Any]:
        row_count = sum(int(shard["rows"]) for shard in self._shards)
        shard_bytes = sum(int(shard["bytes"]) for shard in self._shards)
        game_bytes = self.games_path.stat().st_size if self.games_path.exists() else 0
        return {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "config": self.config,
            "metadata": self.metadata,
            "schema": self._row_schema.to_string(),
            "schema_version": RL_STEP_ROW_SCHEMA_VERSION,
            "game_schema": GAME_ROW_SCHEMA.to_string(),
            "summary": {
                "games": len(self._game_rows),
                "rows": row_count,
                "shards": len(self._shards),
                "bytes": shard_bytes + game_bytes,
                "row_bytes": shard_bytes,
                "game_bytes": game_bytes,
            },
            "shards": [dict(shard) for shard in self._shards],
            "games": {
                "path": deck_records.display_path(self.games_path),
                "rows": len(self._game_rows),
                "bytes": game_bytes,
            },
            "output_dir": deck_records.display_path(self.output_dir),
        }

    def _raise_if_closed(self) -> None:
        if self._closed:
            raise RuntimeError("TrajectoryShardWriter is closed")


def trajectory_row_schema() -> pa.Schema:
    """Return the BC step-row schema with RL rollout extension columns."""
    return pa.schema([*step_records.step_row_schema(), *RL_EXTENSION_FIELDS])


def _game_row(
    finished: FinishedGame,
    buffer: _GameBuffer,
    *,
    rows: int,
    policy_version: str,
    opponent_name: str,
    opponent_tier: int,
) -> step_records.Row:
    deck_signatures = tuple(_deck_signature(deck) for deck in buffer.deck_pair)
    return {
        "game_id": finished.game_id,
        "episode_id": buffer.episode_id,
        "winner_index": int(finished.winner_index),
        "steps": int(finished.steps),
        "rows": rows,
        "seat0_decisions": buffer.decision_counts[0],
        "seat1_decisions": buffer.decision_counts[1],
        "policy_version": policy_version,
        "opponent_name": opponent_name,
        "opponent_tier": opponent_tier,
        "deck0_signature": deck_signatures[0],
        "deck1_signature": deck_signatures[1],
        "deck0_hash": deck_records.signature_hash(deck_signatures[0]),
        "deck1_hash": deck_records.signature_hash(deck_signatures[1]),
    }


def _normalize_deck_pair(deck_pair: DeckPair) -> DeckPair:
    return (
        tuple(int(card_id) for card_id in deck_pair[0]),
        tuple(int(card_id) for card_id in deck_pair[1]),
    )


def _decision_opponent_name(decision: RolloutDecision, fallback: str) -> str:
    raw_value = getattr(decision, "opponent_name", "")
    value = str(raw_value).strip()
    return value or fallback


def _decision_opponent_tier(decision: RolloutDecision, fallback: int) -> int:
    raw_value = getattr(decision, "opponent_tier", fallback)
    value = int(raw_value)
    if not -128 <= value <= 127:
        raise ValueError("opponent_tier must fit in int8")
    return value


def _deck_signature(deck: Sequence[int]) -> str:
    return deck_records.deck_signature([int(card_id) for card_id in deck])


def _update_hand_snapshots(
    buffer: _GameBuffer,
    observation: Mapping[str, Any],
) -> None:
    current = _mapping(_field(observation, "current"))
    for player_index in (0, 1):
        hand_ids = _visible_hand_ids(current, player_index)
        if hand_ids is not None:
            buffer.hand_snapshots[player_index] = hand_ids


def _opponent_hand_label(
    buffer: _GameBuffer,
    observation: Mapping[str, Any],
    *,
    player_index: int,
) -> tuple[tuple[int, ...], bool, bool]:
    opponent_index = 1 - player_index
    hand_ids = buffer.hand_snapshots.get(opponent_index)
    if hand_ids is None:
        return ((), False, False)
    current = _mapping(_field(observation, "current"))
    expected_count = _hand_count(current, opponent_index)
    return (hand_ids, True, len(hand_ids) == expected_count)


def _visible_hand_ids(
    current: Mapping[str, Any],
    player_index: int,
) -> tuple[int, ...] | None:
    player = _player_state(current, player_index)
    raw_hand = player.get("hand")
    if raw_hand is None:
        return None
    hand_ids: list[int] = []
    for card in _sequence(raw_hand):
        card_id = _optional_int(_field(card, "id"))
        if card_id is None or card_id <= 0:
            return None
        hand_ids.append(card_id)
    return tuple(hand_ids)


def _terminal_prize_diffs(observation: Mapping[str, Any]) -> dict[int, int | None]:
    current = _mapping(_field(observation, "current"))
    players = [_mapping(player) for player in _sequence(current.get("players"))]
    if len(players) < 2:
        return {0: None, 1: None}
    prize_counts = [len(_sequence(player.get("prize"))) for player in players[:2]]
    return {
        0: prize_counts[1] - prize_counts[0],
        1: prize_counts[0] - prize_counts[1],
    }


def _reward_for_seat(seat: int, winner_index: int) -> float:
    if winner_index == 2:
        return 0.0
    return 1.0 if winner_index == seat else -1.0


def _result_label(reward: float) -> str:
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _hand_count(current: Mapping[str, Any], player_index: int) -> int:
    return _int_value(_player_state(current, player_index).get("handCount"))


def _player_state(current: Mapping[str, Any], player_index: int) -> Mapping[str, Any]:
    players = _sequence(current.get("players"))
    if 0 <= player_index < len(players):
        return _mapping(players[player_index])
    return {}


def _observation_step(observation: Mapping[str, Any], *, fallback: int) -> int:
    value = _field(observation, "step")
    return int(value) if value is not None else fallback


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return value
    return ()


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _int_value(value: Any) -> int:
    return int(value) if value is not None else 0


def _write_parquet_atomic(
    path: Path,
    table: pa.Table,
    *,
    compression: str,
) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    pq.write_table(table, tmp_path, compression=compression)
    tmp_path.replace(path)


def _write_json_atomic(path: Path, data: Mapping[str, Any]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)
