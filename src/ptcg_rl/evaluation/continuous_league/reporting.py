"""Read-mostly projections used by CLI status and the dashboard API."""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict

from ptcg_rl.evaluation.continuous_league.models import ComponentRating
from ptcg_rl.evaluation.continuous_league.rating import compose_bundle_rating

_LEGACY_DECK_HASH_PATTERN = re.compile(r"(?:^|_)([0-9a-f]{12})(?=_|$)")


class StandingRow(BaseModel):
    """One checkpoint, exact deck, or composed bundle table row."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    identity: str
    label: str
    kind: Literal["checkpoint", "script", "deck", "bundle"]
    controller_id: str | None = None
    controller_label: str | None = None
    deck_digest: str | None = None
    deck_hash: str | None = None
    deck_label: str | None = None
    mu: float
    sigma: float
    conservative: float
    percentile: float
    p_top20: float | None = None
    games: int
    wins: int
    draws: int
    losses: int
    unresolved: int = 0
    bundle_count: int = 0
    rank_eligible: bool = True
    active: bool
    candidate_kind: str | None = None
    candidate_state: str | None = None
    decided_games: int | None = None
    decision_reason: str | None = None
    aliases: tuple[str, ...] = ()


class TrendPoint(BaseModel):
    """One component posterior after an ordered rating event."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_seq: int
    mu: float
    sigma: float
    conservative: float


class MatchupRow(BaseModel):
    """Bundle-pair outcome aggregate from side A's perspective."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    side_a_bundle_id: str
    side_b_bundle_id: str
    games: int
    wins: int
    draws: int
    losses: int
    unresolved: int


class WorkerRow(BaseModel):
    """Latest worker resource, liveness, and throughput projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    worker_id: str
    hostname: str
    source_commit: str
    runtime_fingerprint: str
    belief_fingerprint: str
    current_match_id: str | None
    games_completed: int
    games_per_hour: float
    errors: int
    first_seen_at: str
    last_seen_at: str
    resources: dict[str, Any]


class LeagueSummary(BaseModel):
    """Small operational overview for the league page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    available: bool
    controllers: int = 0
    decks: int = 0
    bundles: int = 0
    candidates: int = 0
    incumbents: int = 0
    queued: int = 0
    leased: int = 0
    completed: int = 0
    unresolved: int = 0
    rating_events: int = 0
    workers: int = 0


class LeagueRepository:
    """Open short-lived read-only SQLite connections for dashboard readers."""

    def __init__(
        self,
        database_path: Path,
        *,
        deck_aliases: Mapping[str, str] | None = None,
    ) -> None:
        self.database_path = database_path.resolve()
        self.deck_aliases = dict(deck_aliases or {})

    def summary(self) -> LeagueSummary:
        """Return empty availability instead of creating a missing database."""
        if not self.database_path.is_file():
            return LeagueSummary(available=False)
        with self._read() as connection:
            controller_counts = _counts(
                connection,
                "SELECT candidate_state, COUNT(*) AS count FROM controllers GROUP BY candidate_state",
            )
            match_counts = _counts(
                connection,
                "SELECT state, COUNT(*) AS count FROM matches GROUP BY state",
            )
            return LeagueSummary(
                available=True,
                controllers=_scalar(connection, "SELECT COUNT(*) FROM controllers"),
                decks=_scalar(connection, "SELECT COUNT(*) FROM decks"),
                bundles=_scalar(connection, "SELECT COUNT(*) FROM bundles"),
                candidates=controller_counts.get("candidate", 0),
                incumbents=controller_counts.get("incumbent", 0),
                queued=match_counts.get("queued", 0),
                leased=match_counts.get("leased", 0),
                completed=match_counts.get("completed", 0),
                unresolved=_scalar(
                    connection,
                    "SELECT COUNT(*) FROM results WHERE outcome = 'unresolved'",
                ),
                rating_events=_scalar(connection, "SELECT COUNT(*) FROM rating_events"),
                workers=_scalar(connection, "SELECT COUNT(*) FROM worker_heartbeats"),
            )

    def standings(
        self, kind: Literal["controllers", "decks", "bundles"]
    ) -> tuple[StandingRow, ...]:
        """Build one global component or composed-bundle table."""
        if not self.database_path.is_file():
            return ()
        with self._read() as connection:
            if kind == "controllers":
                rows = self._controller_standings(connection)
            elif kind == "decks":
                rows = self._deck_standings(connection)
            else:
                rows = self._bundle_standings(connection)
        ordered = sorted(
            rows,
            key=lambda item: (
                not item.rank_eligible,
                -item.conservative,
                item.identity,
            ),
        )
        percentiles = _standing_percentiles(ordered, kind=kind)
        probability_rows = ordered
        if kind == "controllers":
            probability_rows = [
                item
                for item in ordered
                if item.kind == "checkpoint"
                and item.active
                and item.candidate_state == "incumbent"
            ]
        else:
            probability_rows = [item for item in ordered if item.rank_eligible]
        probabilities = _top_fraction_probabilities(probability_rows)
        return tuple(
            item.model_copy(
                update={
                    "percentile": percentiles[item.identity],
                    "p_top20": (
                        item.p_top20
                        if item.p_top20 is not None
                        else probabilities.get(item.identity)
                    ),
                }
            )
            for item in ordered
        )

    def trend(self, component_id: str, *, limit: int = 512) -> tuple[TrendPoint, ...]:
        """Read bounded ordered posterior history for one component."""
        if not self.database_path.is_file():
            return ()
        with self._read() as connection:
            rows = connection.execute(
                """SELECT event_seq, after_json FROM rating_events
                   WHERE after_json LIKE ? ORDER BY event_seq DESC LIMIT ?""",
                (f'%"{component_id}"%', limit),
            ).fetchall()
        output: list[TrendPoint] = []
        for row in reversed(rows):
            payload = json.loads(str(row["after_json"])).get(component_id)
            if not isinstance(payload, dict):
                continue
            mu = float(payload["mu"])
            sigma = float(payload["sigma"])
            output.append(
                TrendPoint(
                    event_seq=int(row["event_seq"]),
                    mu=mu,
                    sigma=sigma,
                    conservative=mu - 3.0 * sigma,
                )
            )
        return tuple(output)

    def matchups(self, *, limit: int = 500) -> tuple[MatchupRow, ...]:
        """Aggregate exact ordered bundle matchups."""
        if not self.database_path.is_file():
            return ()
        with self._read() as connection:
            rows = connection.execute(
                """SELECT m.side_a_bundle_id, m.side_b_bundle_id,
                     COUNT(*) AS games,
                     SUM(r.outcome = 'side_a_win') AS wins,
                     SUM(r.outcome = 'draw') AS draws,
                     SUM(r.outcome = 'side_b_win') AS losses,
                     SUM(r.outcome = 'unresolved') AS unresolved
                   FROM results r JOIN matches m USING(match_id)
                   GROUP BY m.side_a_bundle_id, m.side_b_bundle_id
                   ORDER BY games DESC, m.side_a_bundle_id, m.side_b_bundle_id
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return tuple(
            MatchupRow(
                side_a_bundle_id=str(row["side_a_bundle_id"]),
                side_b_bundle_id=str(row["side_b_bundle_id"]),
                games=int(row["games"]),
                wins=int(row["wins"]),
                draws=int(row["draws"]),
                losses=int(row["losses"]),
                unresolved=int(row["unresolved"]),
            )
            for row in rows
        )

    def workers(self) -> tuple[WorkerRow, ...]:
        """Return all last-known worker heartbeat projections."""
        if not self.database_path.is_file():
            return ()
        with self._read() as connection:
            rows = connection.execute(
                "SELECT * FROM worker_heartbeats ORDER BY worker_id"
            ).fetchall()
        now = datetime.now(UTC)
        output: list[WorkerRow] = []
        for row in rows:
            first_seen = str(row["first_seen_at"])
            elapsed_hours = max(
                (now - datetime.fromisoformat(first_seen)).total_seconds() / 3600.0,
                1.0 / 3600.0,
            )
            completed = int(row["games_completed"])
            output.append(
                WorkerRow(
                    worker_id=str(row["worker_id"]),
                    hostname=str(row["hostname"]),
                    source_commit=str(row["source_commit"]),
                    runtime_fingerprint=str(row["runtime_fingerprint"]),
                    belief_fingerprint=str(row["belief_fingerprint"]),
                    current_match_id=(
                        None
                        if row["current_match_id"] is None
                        else str(row["current_match_id"])
                    ),
                    games_completed=completed,
                    games_per_hour=completed / elapsed_hours,
                    errors=int(row["errors"]),
                    first_seen_at=first_seen,
                    last_seen_at=str(row["last_seen_at"]),
                    resources=json.loads(str(row["resources_json"])),
                )
            )
        return tuple(output)

    def _controller_standings(
        self, connection: sqlite3.Connection
    ) -> list[StandingRow]:
        rows = connection.execute(
            """SELECT c.*, r.mu, r.sigma, r.games,
                 COALESCE(GROUP_CONCAT(a.alias, '\n'), '') AS aliases
               FROM controllers c JOIN component_ratings r
                 ON r.component_id = c.controller_id
               LEFT JOIN submission_aliases a USING(controller_id)
               GROUP BY c.controller_id ORDER BY c.controller_id"""
        ).fetchall()
        outcomes_by_id = _component_outcomes(
            connection,
            component_kind="controller",
        )
        bundle_counts = _bundle_counts(
            connection,
            component_kind="controller",
        )
        output: list[StandingRow] = []
        for row in rows:
            controller_id = str(row["controller_id"])
            outcomes = outcomes_by_id.get(controller_id, _empty_outcomes())
            output.append(
                StandingRow(
                    identity=controller_id,
                    label=str(row["label"]),
                    kind=cast(Literal["checkpoint", "script"], str(row["kind"])),
                    controller_id=str(row["controller_id"]),
                    mu=float(row["mu"]),
                    sigma=float(row["sigma"]),
                    conservative=float(row["mu"]) - 3.0 * float(row["sigma"]),
                    percentile=0.0,
                    p_top20=(None if row["p_top20"] is None else float(row["p_top20"])),
                    games=int(row["games"]),
                    wins=outcomes["wins"],
                    draws=outcomes["draws"],
                    losses=outcomes["losses"],
                    unresolved=outcomes["unresolved"],
                    bundle_count=bundle_counts.get(controller_id, 0),
                    rank_eligible=(
                        bool(row["active"])
                        and str(row["kind"]) == "checkpoint"
                    ),
                    active=bool(row["active"]),
                    candidate_kind=str(row["candidate_kind"]),
                    candidate_state=str(row["candidate_state"]),
                    decided_games=int(row["decided_games"]),
                    decision_reason=(
                        None
                        if row["decision_reason"] is None
                        else str(row["decision_reason"])
                    ),
                    aliases=tuple(filter(None, str(row["aliases"]).split("\n"))),
                )
            )
        return output

    def _deck_standings(self, connection: sqlite3.Connection) -> list[StandingRow]:
        rows = connection.execute(
            """SELECT d.*, r.mu, r.sigma, r.games FROM decks d
               JOIN component_ratings r ON r.component_id = d.deck_digest
               ORDER BY d.deck_digest"""
        ).fetchall()
        outcomes_by_id = _component_outcomes(connection, component_kind="deck")
        bundle_counts = _bundle_counts(connection, component_kind="deck")
        output: list[StandingRow] = []
        for row in rows:
            deck_digest = str(row["deck_digest"])
            label = self._deck_label(str(row["label"]), deck_digest)
            outcomes = outcomes_by_id.get(deck_digest, _empty_outcomes())
            bundle_count = bundle_counts.get(deck_digest, 0)
            output.append(
                StandingRow(
                    identity=deck_digest,
                    label=label,
                    kind="deck",
                    deck_digest=deck_digest,
                    deck_hash=_legacy_deck_hash(
                        str(row["label"]),
                        Path(str(row["path"])).stem,
                    ),
                    deck_label=label,
                    mu=float(row["mu"]),
                    sigma=float(row["sigma"]),
                    conservative=float(row["mu"]) - 3.0 * float(row["sigma"]),
                    percentile=0.0,
                    games=int(row["games"]),
                    wins=outcomes["wins"],
                    draws=outcomes["draws"],
                    losses=outcomes["losses"],
                    unresolved=outcomes["unresolved"],
                    bundle_count=bundle_count,
                    rank_eligible=bool(row["active"]) and bundle_count > 0,
                    active=bool(row["active"]),
                )
            )
        return output

    def _bundle_standings(self, connection: sqlite3.Connection) -> list[StandingRow]:
        rows = connection.execute(
            """SELECT b.*, c.label AS controller_label, d.label AS deck_label,
                      d.path AS deck_path,
                      c.candidate_kind,
                      c.candidate_state,
                      c.decision_reason, cr.mu AS controller_mu,
                      c.decided_games,
                      cr.sigma AS controller_sigma, dr.mu AS deck_mu,
                      dr.sigma AS deck_sigma
               FROM bundles b JOIN controllers c USING(controller_id)
               JOIN decks d USING(deck_digest)
               JOIN component_ratings cr ON cr.component_id = b.controller_id
               JOIN component_ratings dr ON dr.component_id = b.deck_digest
               ORDER BY b.bundle_id"""
        ).fetchall()
        outcomes_by_id = _bundle_outcomes(connection)
        output: list[StandingRow] = []
        for row in rows:
            controller = ComponentRating(
                component_id=str(row["controller_id"]),
                component_kind="controller",
                mu=float(row["controller_mu"]),
                sigma=float(row["controller_sigma"]),
            )
            deck = ComponentRating(
                component_id=str(row["deck_digest"]),
                component_kind="deck",
                mu=float(row["deck_mu"]),
                sigma=float(row["deck_sigma"]),
            )
            composed = compose_bundle_rating(controller, deck)
            outcomes = outcomes_by_id.get(
                str(row["bundle_id"]),
                _empty_outcomes(),
            )
            deck_label = self._deck_label(
                str(row["deck_label"]),
                deck.component_id,
            )
            label = f"{row['controller_label']} / {deck_label}"
            output.append(
                StandingRow(
                    identity=str(row["bundle_id"]),
                    label=label,
                    kind="bundle",
                    controller_id=controller.component_id,
                    controller_label=str(row["controller_label"]),
                    deck_digest=deck.component_id,
                    deck_hash=_legacy_deck_hash(
                        str(row["deck_label"]),
                        Path(str(row["deck_path"])).stem,
                    ),
                    deck_label=deck_label,
                    mu=composed.mu,
                    sigma=composed.sigma,
                    conservative=composed.conservative,
                    percentile=0.0,
                    games=sum(outcomes[name] for name in ("wins", "draws", "losses")),
                    wins=outcomes["wins"],
                    draws=outcomes["draws"],
                    losses=outcomes["losses"],
                    unresolved=outcomes["unresolved"],
                    bundle_count=1,
                    rank_eligible=bool(row["active"]),
                    active=bool(row["active"]),
                    candidate_kind=str(row["candidate_kind"]),
                    candidate_state=str(row["candidate_state"]),
                    decided_games=int(row["decided_games"]),
                    decision_reason=(
                        None
                        if row["decision_reason"] is None
                        else str(row["decision_reason"])
                    ),
                )
            )
        return output

    def _deck_label(self, stored_label: str, deck_digest: str) -> str:
        for key in (deck_digest, deck_digest[:12], stored_label):
            if alias := self.deck_aliases.get(key):
                return alias
        return stored_label

    @contextmanager
    def _read(self) -> Iterator[sqlite3.Connection]:
        uri = f"file:{self.database_path}?mode=ro"
        connection = sqlite3.connect(uri, uri=True, timeout=5.0)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()


def _legacy_deck_hash(*sources: str) -> str | None:
    """Read a persisted compact deck identifier without recomputing it."""
    for source in sources:
        if match := _LEGACY_DECK_HASH_PATTERN.search(source.lower()):
            return match.group(1)
    return None


def _bundle_outcomes(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    rows = connection.execute(
        """SELECT m.side_a_bundle_id, m.side_b_bundle_id, r.outcome
           FROM results r JOIN matches m USING(match_id)"""
    ).fetchall()
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        side_a = str(row["side_a_bundle_id"])
        side_b = str(row["side_b_bundle_id"])
        outcome = str(row["outcome"])
        _record_outcome(output.setdefault(side_a, _empty_outcomes()), outcome, True)
        _record_outcome(output.setdefault(side_b, _empty_outcomes()), outcome, False)
    return output


def _component_outcomes(
    connection: sqlite3.Connection,
    *,
    component_kind: Literal["controller", "deck"],
) -> dict[str, dict[str, int]]:
    """Count each result once from one component's evidence perspective."""
    column = "controller_id" if component_kind == "controller" else "deck_digest"
    rows = connection.execute(
        f"""SELECT ba.{column} AS side_a_component,
                   bb.{column} AS side_b_component, r.outcome
            FROM results r JOIN matches m USING(match_id)
            JOIN bundles ba ON ba.bundle_id = m.side_a_bundle_id
            JOIN bundles bb ON bb.bundle_id = m.side_b_bundle_id
            """,  # noqa: S608
    ).fetchall()
    output: dict[str, dict[str, int]] = {}
    for row in rows:
        side_a = str(row["side_a_component"])
        side_b = str(row["side_b_component"])
        outcome = str(row["outcome"])
        if side_a == side_b:
            counts = output.setdefault(side_a, _empty_outcomes())
            counts["unresolved" if outcome == "unresolved" else "draws"] += 1
        else:
            _record_outcome(
                output.setdefault(side_a, _empty_outcomes()),
                outcome,
                True,
            )
            _record_outcome(
                output.setdefault(side_b, _empty_outcomes()),
                outcome,
                False,
            )
    return output


def _bundle_counts(
    connection: sqlite3.Connection,
    *,
    component_kind: Literal["controller", "deck"],
) -> dict[str, int]:
    column = "controller_id" if component_kind == "controller" else "deck_digest"
    rows = connection.execute(
        f"SELECT {column}, COUNT(*) AS count FROM bundles "
        f"WHERE active = 1 GROUP BY {column}"  # noqa: S608
    ).fetchall()
    return {str(row[0]): int(row[1]) for row in rows}


def _record_outcome(
    counts: dict[str, int],
    outcome: str,
    side_a: bool,
) -> None:
    if outcome == "unresolved":
        counts["unresolved"] += 1
    elif outcome == "draw":
        counts["draws"] += 1
    elif side_a == (outcome == "side_a_win"):
        counts["wins"] += 1
    else:
        counts["losses"] += 1


def _empty_outcomes() -> dict[str, int]:
    return {"wins": 0, "draws": 0, "losses": 0, "unresolved": 0}


def _counts(connection: sqlite3.Connection, query: str) -> dict[str, int]:
    return {str(row[0]): int(row[1]) for row in connection.execute(query)}


def _scalar(connection: sqlite3.Connection, query: str) -> int:
    return int(connection.execute(query).fetchone()[0])


def _top_fraction_probabilities(rows: list[StandingRow]) -> dict[str, float]:
    if not rows:
        return {}
    samples = 5_000
    generator = np.random.default_rng(20260804)
    values = np.column_stack(
        [generator.normal(row.mu, row.sigma, samples) for row in rows]
    )
    top_count = max(1, math.ceil(0.20 * len(rows)))
    threshold = np.partition(values, -top_count, axis=1)[:, -top_count]
    probabilities = np.mean(values >= threshold[:, None], axis=0)
    return {
        row.identity: float(probability)
        for row, probability in zip(rows, probabilities, strict=True)
    }


def _standing_percentiles(
    rows: list[StandingRow],
    *,
    kind: Literal["controllers", "decks", "bundles"],
) -> dict[str, float]:
    groups: tuple[list[StandingRow], ...]
    if kind == "controllers":
        groups = ([row for row in rows if row.rank_eligible],)
    else:
        groups = ([row for row in rows if row.rank_eligible],)
    output: dict[str, float] = {row.identity: 0.0 for row in rows}
    for group in groups:
        count = len(group)
        output.update(
            {
                row.identity: 1.0 - index / max(1, count)
                for index, row in enumerate(group)
            }
        )
    return output


__all__ = [
    "LeagueRepository",
    "LeagueSummary",
    "MatchupRow",
    "StandingRow",
    "TrendPoint",
    "WorkerRow",
]
