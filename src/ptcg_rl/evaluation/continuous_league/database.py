"""SQLite schema and transaction helpers for the single-writer league ledger."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterator
from pathlib import Path

_SCHEMA_VERSION = 4

_SCHEMA = """
CREATE TABLE IF NOT EXISTS league_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS controllers (
    controller_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL CHECK (kind IN ('checkpoint', 'script')),
    label TEXT NOT NULL,
    asset_path TEXT,
    asset_sha256 TEXT NOT NULL,
    pair_manifest_path TEXT,
    candidate_kind TEXT NOT NULL
        CHECK (candidate_kind IN ('automatic', 'manual', 'anchor')),
    candidate_state TEXT NOT NULL
        CHECK (candidate_state IN ('candidate', 'incumbent', 'rejected', 'anchor')),
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    requires_cuda INTEGER NOT NULL CHECK (requires_cuda IN (0, 1)),
    runtime_fingerprint TEXT NOT NULL,
    belief_fingerprint TEXT NOT NULL,
    introduced_at TEXT NOT NULL,
    decided_games INTEGER NOT NULL DEFAULT 0,
    p_top20 REAL,
    decision_reason TEXT
);

CREATE TABLE IF NOT EXISTS decks (
    deck_digest TEXT PRIMARY KEY,
    signature TEXT NOT NULL,
    label TEXT NOT NULL,
    path TEXT NOT NULL,
    file_sha256 TEXT NOT NULL,
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    introduced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS controller_deck_compatibility (
    controller_id TEXT NOT NULL REFERENCES controllers(controller_id),
    deck_digest TEXT NOT NULL,
    PRIMARY KEY (controller_id, deck_digest)
);

CREATE TABLE IF NOT EXISTS bundles (
    bundle_id TEXT PRIMARY KEY,
    controller_id TEXT NOT NULL REFERENCES controllers(controller_id),
    deck_digest TEXT NOT NULL REFERENCES decks(deck_digest),
    active INTEGER NOT NULL CHECK (active IN (0, 1)),
    introduced_at TEXT NOT NULL,
    UNIQUE (controller_id, deck_digest)
);

CREATE TABLE IF NOT EXISTS submission_aliases (
    alias TEXT PRIMARY KEY,
    controller_id TEXT NOT NULL REFERENCES controllers(controller_id),
    release_manifest_path TEXT NOT NULL,
    release_manifest_sha256 TEXT NOT NULL,
    introduced_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS component_ratings (
    component_id TEXT PRIMARY KEY,
    component_kind TEXT NOT NULL CHECK (component_kind IN ('controller', 'deck')),
    mu REAL NOT NULL,
    sigma REAL NOT NULL,
    games INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS matches (
    match_id TEXT PRIMARY KEY,
    side_a_bundle_id TEXT NOT NULL REFERENCES bundles(bundle_id),
    side_b_bundle_id TEXT NOT NULL REFERENCES bundles(bundle_id),
    state TEXT NOT NULL CHECK (state IN ('queued', 'leased', 'completed', 'cancelled')),
    priority REAL NOT NULL,
    schedule_reason TEXT NOT NULL,
    requires_cuda INTEGER NOT NULL CHECK (requires_cuda IN (0, 1)),
    created_at TEXT NOT NULL,
    leased_by TEXT,
    lease_expires_at TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    completed_at TEXT
);

CREATE INDEX IF NOT EXISTS matches_queue_idx
    ON matches(state, requires_cuda, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS matches_pair_idx
    ON matches(side_a_bundle_id, side_b_bundle_id, state);

CREATE TABLE IF NOT EXISTS results (
    result_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL UNIQUE REFERENCES matches(match_id),
    worker_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    terminal_reason TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    steps INTEGER NOT NULL,
    duration_seconds REAL NOT NULL,
    telemetry_msgpack BLOB NOT NULL,
    telemetry_sha256 TEXT NOT NULL,
    event_seq INTEGER UNIQUE
);

CREATE TABLE IF NOT EXISTS rating_events (
    event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id TEXT NOT NULL UNIQUE REFERENCES matches(match_id),
    outcome TEXT NOT NULL,
    before_json TEXT NOT NULL,
    after_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS worker_heartbeats (
    worker_id TEXT PRIMARY KEY,
    hostname TEXT NOT NULL,
    source_commit TEXT NOT NULL,
    runtime_fingerprint TEXT NOT NULL,
    belief_fingerprint TEXT NOT NULL,
    resources_json TEXT NOT NULL,
    current_match_id TEXT,
    games_completed INTEGER NOT NULL,
    errors INTEGER NOT NULL,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS discovery_seen (
    asset_type TEXT NOT NULL,
    path TEXT NOT NULL,
    mtime_ns INTEGER NOT NULL,
    fingerprint TEXT NOT NULL,
    discovered_at TEXT NOT NULL,
    PRIMARY KEY (asset_type, path)
);
"""


class LeagueDatabase:
    """Thread-safe connection owned by one coordinator process."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=FULL")
            self.connection.execute("PRAGMA foreign_keys=ON")
            self.connection.executescript(_SCHEMA)
            row = self.connection.execute(
                "SELECT value FROM league_meta WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                self.connection.execute(
                    "INSERT INTO league_meta(key, value) VALUES (?, ?)",
                    ("schema_version", str(_SCHEMA_VERSION)),
                )
            elif int(row["value"]) != _SCHEMA_VERSION:
                raise ValueError("unsupported continuous league database schema")

    @contextlib.contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """Run one immediate single-writer transaction."""
        with self._lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield self.connection
            except BaseException:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")

    @contextlib.contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        """Serialize reads against this process's one shared connection."""
        with self._lock:
            yield self.connection

    def close(self) -> None:
        """Checkpoint WAL best-effort and close the coordinator connection."""
        with self._lock:
            with contextlib.suppress(sqlite3.OperationalError):
                self.connection.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self.connection.close()


__all__ = ["LeagueDatabase"]
