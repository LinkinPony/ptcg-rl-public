"""Durable single-writer asset, match, lease, result, and rating ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from ptcg_rl.evaluation.continuous_league.asset_ledger import LeagueAssetLedgerMixin
from ptcg_rl.evaluation.continuous_league.database import LeagueDatabase
from ptcg_rl.evaluation.continuous_league.models import (
    BundleIdentity,
    ComponentRating,
    ContinuousLeagueConfig,
    LeaseRequest,
    MatchLease,
    MatchResult,
    WorkerHeartbeat,
)
from ptcg_rl.evaluation.continuous_league.rating import ComponentTrueSkill


class LeagueLedger(LeagueAssetLedgerMixin):
    """Authoritative SQLite ledger used only by the coordinator for writes."""

    def __init__(self, config: ContinuousLeagueConfig, *, repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root.resolve()
        self.database = LeagueDatabase(self._repo_path(config.database_path))
        self.skill = ComponentTrueSkill(config.rating)
        try:
            self._bind_season_contract()
        except BaseException:
            self.database.close()
            raise

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self.database.close()

    def enqueue_match(
        self,
        side_a_bundle_id: str,
        side_b_bundle_id: str,
        *,
        priority: float,
        reason: str,
    ) -> str:
        """Queue one exact game; seat ordering is intentional and unmirrored."""
        if side_a_bundle_id == side_b_bundle_id:
            raise ValueError("cannot schedule a bundle against itself")
        match_id = uuid.uuid4().hex
        with self.database.transaction() as connection:
            sides = connection.execute(
                """SELECT bundle_id, controller_id FROM bundles
                   WHERE bundle_id IN (?, ?) AND active = 1""",
                (side_a_bundle_id, side_b_bundle_id),
            ).fetchall()
            if len(sides) != 2:
                raise ValueError("both scheduled bundles must exist and be active")
            controller_ids = [str(row["controller_id"]) for row in sides]
            requires_cuda = connection.execute(
                """SELECT MAX(requires_cuda) AS required FROM controllers
                   WHERE controller_id IN (?, ?)""",
                tuple(controller_ids),
            ).fetchone()["required"]
            connection.execute(
                """INSERT INTO matches(
                       match_id, side_a_bundle_id, side_b_bundle_id, state,
                       priority, schedule_reason, requires_cuda, created_at
                   ) VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)""",
                (
                    match_id,
                    side_a_bundle_id,
                    side_b_bundle_id,
                    priority,
                    reason,
                    int(requires_cuda),
                    _utc_now(),
                ),
            )
        return match_id

    def heartbeat(self, heartbeat: WorkerHeartbeat) -> None:
        """Upsert one worker heartbeat without mutating match ownership."""
        with self.database.transaction() as connection:
            self._upsert_heartbeat(connection, heartbeat)

    def lease_match(self, request: LeaseRequest) -> MatchLease | None:
        """Atomically expire old leases and grant one resource-compatible task."""
        heartbeat = request.heartbeat
        if not heartbeat.resources.quiet:
            self.heartbeat(heartbeat)
            return None
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=self.config.scheduling.lease_seconds)
        with self.database.transaction() as connection:
            self._upsert_heartbeat(connection, heartbeat)
            connection.execute(
                """UPDATE matches SET state = 'queued', leased_by = NULL,
                       lease_expires_at = NULL
                   WHERE state = 'leased' AND lease_expires_at < ?""",
                (now.isoformat(),),
            )
            row = connection.execute(
                """SELECT * FROM matches
                   WHERE state = 'queued'
                     AND (requires_cuda = 0 OR ? = 1)
                   ORDER BY priority DESC, created_at, match_id LIMIT 1""",
                (int(heartbeat.resources.cuda_available),),
            ).fetchone()
            if row is None:
                return None
            changed = connection.execute(
                """UPDATE matches SET state = 'leased', leased_by = ?,
                       lease_expires_at = ?, attempts = attempts + 1
                   WHERE match_id = ? AND state = 'queued'""",
                (heartbeat.worker_id, expires.isoformat(), row["match_id"]),
            ).rowcount
            if changed != 1:
                return None
            return self._match_lease(connection, str(row["match_id"]), expires)

    def submit_result(self, result: MatchResult) -> bool:
        """Idempotently commit one result and at most one ordered rating event."""
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM results WHERE match_id = ?", (result.match_id,)
            ).fetchone()
            if existing is not None:
                if not _same_result(existing, result):
                    raise ValueError("duplicate match result payload differs")
                return False
            match = connection.execute(
                "SELECT * FROM matches WHERE match_id = ?", (result.match_id,)
            ).fetchone()
            if match is None:
                raise KeyError(f"unknown league match: {result.match_id}")
            if str(match["state"]) == "cancelled":
                raise ValueError("cancelled league match cannot accept a result")
            connection.execute(
                """INSERT INTO results(
                       match_id, worker_id, outcome, terminal_reason, started_at,
                       finished_at, steps, duration_seconds, telemetry_msgpack,
                       telemetry_sha256
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.match_id,
                    result.worker_id,
                    result.outcome,
                    result.terminal_reason,
                    result.started_at,
                    result.finished_at,
                    result.steps,
                    result.duration_seconds,
                    result.telemetry_msgpack,
                    hashlib.sha256(result.telemetry_msgpack).hexdigest(),
                ),
            )
            connection.execute(
                """UPDATE matches SET state = 'completed', completed_at = ?,
                       lease_expires_at = NULL WHERE match_id = ?""",
                (result.finished_at, result.match_id),
            )
            if result.outcome != "unresolved":
                event_seq = self._rate_result(connection, match, result)
                connection.execute(
                    "UPDATE results SET event_seq = ? WHERE match_id = ?",
                    (event_seq, result.match_id),
                )
                self._increment_candidate_games(connection, match)
            return True

    def rebuild_ratings(self) -> int:
        """Replay immutable rating events in event_seq order into current state."""
        with self.database.transaction() as connection:
            components = connection.execute(
                "SELECT component_id, component_kind FROM component_ratings"
            ).fetchall()
            state = {
                str(row["component_id"]): self.skill.initial(
                    str(row["component_id"]),
                    cast(
                        Literal["controller", "deck"],
                        str(row["component_kind"]),
                    ),
                )
                for row in components
            }
            events = connection.execute(
                """SELECT e.outcome, m.side_a_bundle_id, m.side_b_bundle_id
                   FROM rating_events e JOIN matches m USING(match_id)
                   ORDER BY e.event_seq"""
            ).fetchall()
            for event in events:
                side_a = self._bundle_components_from_state(
                    connection, str(event["side_a_bundle_id"]), state
                )
                side_b = self._bundle_components_from_state(
                    connection, str(event["side_b_bundle_id"]), state
                )
                state.update(
                    self.skill.rate(side_a, side_b, cast(Any, event["outcome"]))
                )
            connection.executemany(
                """UPDATE component_ratings SET mu = ?, sigma = ?, games = ?
                   WHERE component_id = ?""",
                [
                    (rating.mu, rating.sigma, rating.games, component_id)
                    for component_id, rating in sorted(state.items())
                ],
            )
            return len(events)

    def metadata(self, key: str) -> str | None:
        """Read one coordinator metadata value."""
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT value FROM league_meta WHERE key = ?", (key,)
            ).fetchone()
            return None if row is None else str(row["value"])

    def set_metadata(self, key: str, value: str) -> None:
        """Atomically upsert one coordinator metadata value."""
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO league_meta(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (key, value),
            )

    def bind_source_revision(
        self,
        source_commit: str,
        *,
        allow_revision: bool,
    ) -> None:
        """Bind an immutable daemon revision or explicitly append an upgrade."""
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM league_meta WHERE key = 'season_source_commit'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO league_meta(key, value) VALUES (?, ?)",
                    ("season_source_commit", source_commit),
                )
                revisions: list[dict[str, object]] = [
                    {
                        "commit": source_commit,
                        "adopted_at": _utc_now(),
                        "reason": "initial",
                    }
                ]
            else:
                bound_commit = str(row["value"])
                history_row = connection.execute(
                    "SELECT value FROM league_meta WHERE key = 'source_revisions'"
                ).fetchone()
                revisions = (
                    cast(
                        list[dict[str, object]],
                        json.loads(str(history_row["value"])),
                    )
                    if history_row is not None
                    else [
                        {
                            "commit": bound_commit,
                            "adopted_at": None,
                            "reason": "legacy_initial",
                        }
                    ]
                )
                if bound_commit != source_commit:
                    if not allow_revision:
                        raise RuntimeError(
                            "league source commit differs; explicit revision "
                            "adoption is required"
                        )
                    active_lease = connection.execute(
                        """SELECT 1 FROM matches
                           WHERE state = 'leased' AND lease_expires_at > ? LIMIT 1""",
                        (_utc_now(),),
                    ).fetchone()
                    if active_lease is not None:
                        raise RuntimeError(
                            "cannot adopt a league source revision with active leases"
                        )
                    revisions.append(
                        {
                            "commit": source_commit,
                            "adopted_at": _utc_now(),
                            "reason": "explicit_operational_upgrade",
                        }
                    )
                    connection.execute(
                        "UPDATE league_meta SET value = ? WHERE key = ?",
                        (source_commit, "season_source_commit"),
                    )
            connection.execute(
                """INSERT INTO league_meta(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (
                    "source_revisions",
                    json.dumps(revisions, sort_keys=True, separators=(",", ":")),
                ),
            )

    def advance_telemetry_cursor(self, result_seq: int) -> None:
        """Commit an exported prefix and compact its SQLite telemetry blobs."""
        if result_seq <= 0:
            raise ValueError("telemetry result cursor must be positive")
        with self.database.transaction() as connection:
            current_row = connection.execute(
                "SELECT value FROM league_meta WHERE key = 'telemetry_result_seq'"
            ).fetchone()
            current = 0 if current_row is None else int(current_row["value"])
            cursor = max(current, result_seq)
            connection.execute(
                """INSERT INTO league_meta(key, value) VALUES (
                       'telemetry_result_seq', ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (str(cursor),),
            )
            connection.execute(
                """UPDATE results SET telemetry_msgpack = X''
                   WHERE result_seq <= ? AND length(telemetry_msgpack) > 0""",
                (cursor,),
            )

    def _bind_season_contract(self) -> None:
        """Bind one database forever to its rating and ActTime identities."""
        contract = json.dumps(
            {
                "rating": self.config.rating.model_dump(mode="json"),
                "runtime_fingerprint": self.config.runtime_fingerprint,
                "belief_fingerprint": self.config.belief_fingerprint,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.database.transaction() as connection:
            row = connection.execute(
                "SELECT value FROM league_meta WHERE key = 'season_contract'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO league_meta(key, value) VALUES ('season_contract', ?)",
                    (contract,),
                )
            elif str(row["value"]) != contract:
                raise ValueError(
                    "continuous league database belongs to another season contract"
                )

    def queue_depth(self) -> int:
        """Return immediately leaseable and currently leased work count."""
        with self.database.read() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM matches WHERE state IN ('queued', 'leased')"
                ).fetchone()[0]
            )

    def controller_exists(self, controller_id: str) -> bool:
        """Return whether one immutable controller identity is registered."""
        with self.database.read() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM controllers WHERE controller_id = ?",
                    (controller_id,),
                ).fetchone()
                is not None
            )

    def discovery_initialized(self, asset_type: str) -> bool:
        """Return whether the initial no-backfill cursor was established."""
        return self.metadata(f"discovery_initialized:{asset_type}") == "1"

    def discovery_seen(self, asset_type: str, path: Path) -> bool:
        """Return whether a path has crossed its discovery cursor."""
        with self.database.read() as connection:
            return (
                connection.execute(
                    "SELECT 1 FROM discovery_seen WHERE asset_type = ? AND path = ?",
                    (asset_type, str(path.resolve())),
                ).fetchone()
                is not None
            )

    def mark_discovery_seen(
        self,
        asset_type: str,
        path: Path,
        *,
        fingerprint: str,
    ) -> None:
        """Advance one durable path cursor after inspection or initial skip."""
        resolved = path.resolve()
        with self.database.transaction() as connection:
            connection.execute(
                """INSERT INTO discovery_seen(
                       asset_type, path, mtime_ns, fingerprint, discovered_at
                   ) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(asset_type, path) DO NOTHING""",
                (
                    asset_type,
                    str(resolved),
                    resolved.stat().st_mtime_ns,
                    fingerprint,
                    _utc_now(),
                ),
            )

    def _match_lease(
        self, connection: sqlite3.Connection, match_id: str, expires: datetime
    ) -> MatchLease:
        row = connection.execute(
            """SELECT m.*, ba.controller_id AS a_controller,
                      ba.deck_digest AS a_deck, bb.controller_id AS b_controller,
                      bb.deck_digest AS b_deck,
                      ca.kind AS a_kind, ca.asset_path AS a_path,
                      cb.kind AS b_kind, cb.asset_path AS b_path,
                      da.path AS a_deck_path, db.path AS b_deck_path
               FROM matches m
               JOIN bundles ba ON ba.bundle_id = m.side_a_bundle_id
               JOIN bundles bb ON bb.bundle_id = m.side_b_bundle_id
               JOIN controllers ca ON ca.controller_id = ba.controller_id
               JOIN controllers cb ON cb.controller_id = bb.controller_id
               JOIN decks da ON da.deck_digest = ba.deck_digest
               JOIN decks db ON db.deck_digest = bb.deck_digest
               WHERE m.match_id = ?""",
            (match_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("leased match disappeared")
        return MatchLease(
            match_id=match_id,
            side_a=BundleIdentity(
                bundle_id=str(row["side_a_bundle_id"]),
                controller_id=str(row["a_controller"]),
                deck_digest=str(row["a_deck"]),
            ),
            side_b=BundleIdentity(
                bundle_id=str(row["side_b_bundle_id"]),
                controller_id=str(row["b_controller"]),
                deck_digest=str(row["b_deck"]),
            ),
            side_a_controller_kind=cast(Any, row["a_kind"]),
            side_b_controller_kind=cast(Any, row["b_kind"]),
            side_a_controller_path=_optional_path(row["a_path"]),
            side_b_controller_path=_optional_path(row["b_path"]),
            side_a_deck_path=Path(str(row["a_deck_path"])),
            side_b_deck_path=Path(str(row["b_deck_path"])),
            requires_cuda=bool(row["requires_cuda"]),
            runtime_fingerprint=self.config.runtime_fingerprint,
            belief_fingerprint=self.config.belief_fingerprint,
            lease_expires_at=expires.isoformat(),
        )

    def _rate_result(
        self,
        connection: sqlite3.Connection,
        match: sqlite3.Row,
        result: MatchResult,
    ) -> int:
        side_a = self._bundle_components(connection, str(match["side_a_bundle_id"]))
        side_b = self._bundle_components(connection, str(match["side_b_bundle_id"]))
        before = {item.component_id: item for item in (*side_a, *side_b)}
        after = self.skill.rate(side_a, side_b, result.outcome)
        cursor = connection.execute(
            """INSERT INTO rating_events(
                   match_id, outcome, before_json, after_json, created_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                result.match_id,
                result.outcome,
                _ratings_json(before),
                _ratings_json(after),
                _utc_now(),
            ),
        )
        if cursor.lastrowid is None:
            raise RuntimeError("SQLite did not assign a rating event sequence")
        event_seq = int(cursor.lastrowid)
        connection.executemany(
            """UPDATE component_ratings SET mu = ?, sigma = ?, games = ?
               WHERE component_id = ?""",
            [
                (rating.mu, rating.sigma, rating.games, component_id)
                for component_id, rating in sorted(after.items())
            ],
        )
        return event_seq

    def _bundle_components(
        self, connection: sqlite3.Connection, bundle_id: str
    ) -> tuple[ComponentRating, ComponentRating]:
        rows = connection.execute(
            """SELECT r.* FROM bundles b JOIN component_ratings r
                   ON r.component_id IN (b.controller_id, b.deck_digest)
               WHERE b.bundle_id = ?
               ORDER BY CASE r.component_kind WHEN 'controller' THEN 0 ELSE 1 END""",
            (bundle_id,),
        ).fetchall()
        if len(rows) != 2:
            raise RuntimeError("bundle component ratings are incomplete")
        return cast(
            tuple[ComponentRating, ComponentRating],
            tuple(_component_rating(row) for row in rows),
        )

    def _bundle_components_from_state(
        self,
        connection: sqlite3.Connection,
        bundle_id: str,
        state: Mapping[str, ComponentRating],
    ) -> tuple[ComponentRating, ComponentRating]:
        row = connection.execute(
            "SELECT controller_id, deck_digest FROM bundles WHERE bundle_id = ?",
            (bundle_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError("rating event references a missing bundle")
        return state[str(row["controller_id"])], state[str(row["deck_digest"])]

    @staticmethod
    def _increment_candidate_games(
        connection: sqlite3.Connection, match: sqlite3.Row
    ) -> None:
        connection.execute(
            """UPDATE controllers SET decided_games = decided_games + 1
               WHERE candidate_state = 'candidate' AND controller_id IN (
                   SELECT controller_id FROM bundles WHERE bundle_id IN (?, ?)
               )""",
            (match["side_a_bundle_id"], match["side_b_bundle_id"]),
        )

    @staticmethod
    def _upsert_heartbeat(
        connection: sqlite3.Connection, heartbeat: WorkerHeartbeat
    ) -> None:
        observed_at = _utc_now()
        connection.execute(
            """INSERT INTO worker_heartbeats(
                   worker_id, hostname, source_commit, runtime_fingerprint,
                   belief_fingerprint, resources_json, current_match_id,
                   games_completed, errors, first_seen_at, last_seen_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(worker_id) DO UPDATE SET
                   hostname = excluded.hostname,
                   source_commit = excluded.source_commit,
                   runtime_fingerprint = excluded.runtime_fingerprint,
                   belief_fingerprint = excluded.belief_fingerprint,
                   resources_json = excluded.resources_json,
                   current_match_id = excluded.current_match_id,
                   games_completed = excluded.games_completed,
                   errors = excluded.errors,
                   last_seen_at = excluded.last_seen_at""",
            (
                heartbeat.worker_id,
                heartbeat.hostname,
                heartbeat.source_commit,
                heartbeat.runtime_fingerprint,
                heartbeat.belief_fingerprint,
                heartbeat.resources.model_dump_json(),
                heartbeat.current_match_id,
                heartbeat.games_completed,
                heartbeat.errors,
                observed_at,
                observed_at,
            ),
        )

    def _repo_path(self, path: Path) -> Path:
        return path if path.is_absolute() else self.repo_root / path


def _component_rating(row: sqlite3.Row) -> ComponentRating:
    return ComponentRating(
        component_id=str(row["component_id"]),
        component_kind=cast(Any, row["component_kind"]),
        mu=float(row["mu"]),
        sigma=float(row["sigma"]),
        games=int(row["games"]),
    )


def _ratings_json(ratings: Mapping[str, ComponentRating]) -> str:
    return json.dumps(
        {key: value.model_dump(mode="json") for key, value in sorted(ratings.items())},
        sort_keys=True,
        separators=(",", ":"),
    )


def _same_result(row: sqlite3.Row, result: MatchResult) -> bool:
    return (
        str(row["worker_id"]) == result.worker_id
        and str(row["outcome"]) == result.outcome
        and str(row["terminal_reason"]) == result.terminal_reason
        and str(row["started_at"]) == result.started_at
        and str(row["finished_at"]) == result.finished_at
        and int(row["steps"]) == result.steps
        and float(row["duration_seconds"]) == result.duration_seconds
        and str(row["telemetry_sha256"])
        == hashlib.sha256(result.telemetry_msgpack).hexdigest()
    )


def _optional_path(value: object) -> Path | None:
    return None if value is None else Path(str(value))


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["LeagueLedger"]
