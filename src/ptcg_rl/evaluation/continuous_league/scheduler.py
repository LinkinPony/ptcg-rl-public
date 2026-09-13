"""Challenge-ladder scheduling and automatic checkpoint promotion gates."""

from __future__ import annotations

import hashlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.evaluation.continuous_league.ledger import LeagueLedger
from ptcg_rl.evaluation.continuous_league.models import ComponentRating
from ptcg_rl.evaluation.continuous_league.rating import (
    compose_bundle_rating,
    probability_top_fraction,
    promotion_decision,
)


@dataclass(frozen=True, slots=True)
class ScheduledBundle:
    """Compact scheduler projection for one active combination."""

    bundle_id: str
    controller_id: str
    deck_digest: str
    candidate_state: str
    decided_games: int
    bundle_games: int
    controller_conservative: float
    conservative: float


class ChallengeScheduler:
    """Keep a bounded queue populated even when no candidate exists."""

    def __init__(self, ledger: LeagueLedger) -> None:
        self.ledger = ledger

    def fill_queue(self) -> int:
        """Evaluate gates and fill the queue to its configured target."""
        self.evaluate_candidates()
        self._make_room_for_candidates()
        target = self.ledger.config.scheduling.queue_target
        added = 0
        while self.ledger.queue_depth() < target:
            scheduled = self._schedule_one()
            if not scheduled:
                break
            added += 1
        return added

    def _make_room_for_candidates(self) -> None:
        """Do not let a pre-existing background backlog delay a new candidate."""
        with self.ledger.database.transaction() as connection:
            candidate_exists = connection.execute(
                """SELECT 1 FROM controllers WHERE candidate_state = 'candidate'
                   AND active = 1 LIMIT 1"""
            ).fetchone()
            if candidate_exists is None:
                return
            connection.execute(
                """UPDATE matches SET state = 'cancelled'
                   WHERE state = 'queued'
                     AND schedule_reason = 'background_adjacent_or_stale'"""
            )

    def evaluate_candidates(self) -> int:
        """Apply deterministic top-20 gates to every automatic candidate."""
        changed = 0
        with self.ledger.database.transaction() as connection:
            candidates = connection.execute(
                """SELECT c.*, r.mu, r.sigma, r.games
                   FROM controllers c JOIN component_ratings r
                     ON r.component_id = c.controller_id
                   WHERE c.candidate_kind = 'automatic'
                     AND c.candidate_state = 'candidate' AND c.active = 1
                   ORDER BY c.introduced_at, c.controller_id"""
            ).fetchall()
            incumbents = tuple(
                _controller_rating(row)
                for row in connection.execute(
                    """SELECT c.controller_id, r.mu, r.sigma, r.games
                       FROM controllers c JOIN component_ratings r
                         ON r.component_id = c.controller_id
                       WHERE c.kind = 'checkpoint'
                         AND c.candidate_state = 'incumbent' AND c.active = 1
                       ORDER BY c.controller_id"""
                ).fetchall()
            )
            for row in candidates:
                candidate = _controller_rating(row)
                decided_games = int(row["decided_games"])
                probability = probability_top_fraction(
                    candidate,
                    incumbents,
                    self.ledger.config.promotion,
                    evidence_games=decided_games,
                )
                anchor_games = _anchor_games(connection, str(row["controller_id"]))
                decision = promotion_decision(
                    probability_top=probability,
                    decided_games=decided_games,
                    anchor_games=anchor_games,
                    incumbent_count=len(incumbents),
                    candidate_kind=str(row["candidate_kind"]),
                    config=self.ledger.config.promotion,
                )
                connection.execute(
                    "UPDATE controllers SET p_top20 = ? WHERE controller_id = ?",
                    (probability, row["controller_id"]),
                )
                if not decision.decided:
                    continue
                state = "incumbent" if decision.accepted else "rejected"
                active = int(decision.accepted)
                connection.execute(
                    """UPDATE controllers SET candidate_state = ?, active = ?,
                           decision_reason = ? WHERE controller_id = ?""",
                    (state, active, decision.reason, row["controller_id"]),
                )
                connection.execute(
                    "UPDATE bundles SET active = ? WHERE controller_id = ?",
                    (active, row["controller_id"]),
                )
                if not decision.accepted:
                    connection.execute(
                        """UPDATE matches SET state = 'cancelled'
                           WHERE state = 'queued' AND (
                               side_a_bundle_id IN (
                                   SELECT bundle_id FROM bundles WHERE controller_id = ?
                               ) OR side_b_bundle_id IN (
                                   SELECT bundle_id FROM bundles WHERE controller_id = ?
                               )
                           )""",
                        (row["controller_id"], row["controller_id"]),
                    )
                changed += 1
        return changed

    def _schedule_one(self) -> bool:
        with self.ledger.database.read() as connection:
            bundles = _active_bundles(connection)
            if len(bundles) < 2:
                return False
            candidates = [
                item for item in bundles if item.candidate_state == "candidate"
            ]
            if candidates:
                side = min(
                    candidates,
                    key=lambda item: (
                        item.bundle_games,
                        item.decided_games,
                        item.controller_id,
                        item.deck_digest,
                    ),
                )
                opponents = [
                    item
                    for item in bundles
                    if item.controller_id != side.controller_id
                    and item.candidate_state != "candidate"
                ]
                if not opponents:
                    return False
                opponent = min(
                    opponents,
                    key=lambda item: self._candidate_opponent_key(
                        connection,
                        side,
                        item,
                        boundary=_incumbent_top20_boundary(connection),
                    ),
                )
                candidate_first = _side_a_assignment(
                    side.bundle_id,
                    opponent.bundle_id,
                    _controller_scheduled_games(connection, side.controller_id),
                )
                side_a, side_b = (
                    (side, opponent) if candidate_first else (opponent, side)
                )
                reason = "candidate_ladder"
                if opponent.candidate_state == "anchor":
                    reason = "candidate_anchor_calibration"
                elif opponent.deck_digest == side.deck_digest:
                    reason = "candidate_cross_checkpoint_decomposition"
                priority = 1000.0 - min(999.0, float(side.decided_games))
            else:
                pair = self._background_pair(connection, bundles)
                if pair is None:
                    return False
                side_a, side_b = pair
                pair_games = _pair_games(connection, side_a.bundle_id, side_b.bundle_id)
                if not _side_a_assignment(
                    side_a.bundle_id, side_b.bundle_id, pair_games
                ):
                    side_a, side_b = side_b, side_a
                reason = "background_adjacent_or_stale"
                priority = 10.0
        self.ledger.enqueue_match(
            side_a.bundle_id,
            side_b.bundle_id,
            priority=priority,
            reason=reason,
        )
        return True

    def _candidate_opponent_key(
        self,
        connection: sqlite3.Connection,
        candidate: ScheduledBundle,
        opponent: ScheduledBundle,
        *,
        boundary: float | None,
    ) -> tuple[float, float, float, float, str]:
        games = _pair_games(connection, candidate.bundle_id, opponent.bundle_id)
        cross_evidence = 0.0 if candidate.deck_digest == opponent.deck_digest else 0.25
        scheduled_games = _controller_scheduled_games(
            connection, candidate.controller_id
        )
        anchor_due = (
            scheduled_games % self.ledger.config.scheduling.anchor_frequency == 0
        )
        anchor_penalty = (
            0.0
            if (anchor_due and opponent.candidate_state == "anchor")
            else (
                2.0
                if anchor_due
                else (1.0 if opponent.candidate_state == "anchor" else 0.0)
            )
        )
        distance = abs(candidate.conservative - opponent.conservative)
        boundary_distance = (
            distance
            if boundary is None
            else abs(opponent.controller_conservative - boundary)
        )
        return (
            anchor_penalty,
            games / self.ledger.config.scheduling.stale_pair_games + cross_evidence,
            boundary_distance,
            distance,
            opponent.bundle_id,
        )

    def _background_pair(
        self,
        connection: sqlite3.Connection,
        bundles: Sequence[ScheduledBundle],
    ) -> tuple[ScheduledBundle, ScheduledBundle] | None:
        best: (
            tuple[tuple[float, float, str, str], ScheduledBundle, ScheduledBundle]
            | None
        ) = None
        for index, side_a in enumerate(bundles):
            for side_b in bundles[index + 1 :]:
                if side_a.controller_id == side_b.controller_id and (
                    side_a.deck_digest == side_b.deck_digest
                ):
                    continue
                games = _pair_games(connection, side_a.bundle_id, side_b.bundle_id)
                distance = abs(side_a.conservative - side_b.conservative)
                anchor_penalty = (
                    0.0
                    if "anchor" in {side_a.candidate_state, side_b.candidate_state}
                    else 0.5
                )
                key = (
                    games / self.ledger.config.scheduling.stale_pair_games
                    + anchor_penalty,
                    distance,
                    side_a.bundle_id,
                    side_b.bundle_id,
                )
                if best is None or key < best[0]:
                    best = (key, side_a, side_b)
        if best is None:
            return None
        return best[1], best[2]


def _active_bundles(connection: sqlite3.Connection) -> tuple[ScheduledBundle, ...]:
    rows = connection.execute(
        """SELECT b.bundle_id, b.controller_id, b.deck_digest,
                  c.candidate_state, c.decided_games,
                  cr.mu AS controller_mu, cr.sigma AS controller_sigma,
                  cr.games AS controller_games,
                  dr.mu AS deck_mu, dr.sigma AS deck_sigma,
                  dr.games AS deck_games,
                  (SELECT COUNT(*) FROM matches m
                   WHERE m.state != 'cancelled' AND (
                     m.side_a_bundle_id = b.bundle_id OR
                     m.side_b_bundle_id = b.bundle_id
                   )) AS bundle_games
           FROM bundles b JOIN controllers c USING(controller_id)
           JOIN component_ratings cr ON cr.component_id = b.controller_id
           JOIN component_ratings dr ON dr.component_id = b.deck_digest
           WHERE b.active = 1 AND c.active = 1
           ORDER BY b.bundle_id"""
    ).fetchall()
    output: list[ScheduledBundle] = []
    for row in rows:
        controller = ComponentRating(
            component_id=str(row["controller_id"]),
            component_kind="controller",
            mu=float(row["controller_mu"]),
            sigma=float(row["controller_sigma"]),
            games=int(row["controller_games"]),
        )
        deck = ComponentRating(
            component_id=str(row["deck_digest"]),
            component_kind="deck",
            mu=float(row["deck_mu"]),
            sigma=float(row["deck_sigma"]),
            games=int(row["deck_games"]),
        )
        output.append(
            ScheduledBundle(
                bundle_id=str(row["bundle_id"]),
                controller_id=controller.component_id,
                deck_digest=deck.component_id,
                candidate_state=str(row["candidate_state"]),
                decided_games=int(row["decided_games"]),
                bundle_games=int(row["bundle_games"]),
                controller_conservative=(controller.mu - 3.0 * controller.sigma),
                conservative=compose_bundle_rating(controller, deck).conservative,
            )
        )
    return tuple(output)


def _controller_rating(row: Mapping[str, Any]) -> ComponentRating:
    return ComponentRating(
        component_id=str(row["controller_id"]),
        component_kind="controller",
        mu=float(row["mu"]),
        sigma=float(row["sigma"]),
        games=int(row["games"]),
    )


def _anchor_games(connection: sqlite3.Connection, controller_id: str) -> int:
    return int(
        connection.execute(
            """SELECT COUNT(*) FROM results r JOIN matches m USING(match_id)
               JOIN bundles ba ON ba.bundle_id = m.side_a_bundle_id
               JOIN bundles bb ON bb.bundle_id = m.side_b_bundle_id
               JOIN controllers ca ON ca.controller_id = ba.controller_id
               JOIN controllers cb ON cb.controller_id = bb.controller_id
               WHERE r.outcome != 'unresolved' AND (
                 (ba.controller_id = ? AND cb.candidate_state = 'anchor') OR
                 (bb.controller_id = ? AND ca.candidate_state = 'anchor')
               )""",
            (controller_id, controller_id),
        ).fetchone()[0]
    )


def _pair_games(
    connection: sqlite3.Connection, side_a_bundle_id: str, side_b_bundle_id: str
) -> int:
    return int(
        connection.execute(
            """SELECT COUNT(*) FROM matches WHERE state != 'cancelled' AND (
                 (side_a_bundle_id = ? AND side_b_bundle_id = ?) OR
                 (side_a_bundle_id = ? AND side_b_bundle_id = ?)
               )""",
            (
                side_a_bundle_id,
                side_b_bundle_id,
                side_b_bundle_id,
                side_a_bundle_id,
            ),
        ).fetchone()[0]
    )


def _controller_scheduled_games(
    connection: sqlite3.Connection, controller_id: str
) -> int:
    return int(
        connection.execute(
            """SELECT COUNT(*) FROM matches m
               JOIN bundles ba ON ba.bundle_id = m.side_a_bundle_id
               JOIN bundles bb ON bb.bundle_id = m.side_b_bundle_id
               WHERE m.state != 'cancelled' AND
                 (ba.controller_id = ? OR bb.controller_id = ?)""",
            (controller_id, controller_id),
        ).fetchone()[0]
    )


def _incumbent_top20_boundary(connection: sqlite3.Connection) -> float | None:
    values = sorted(
        (
            float(row["mu"]) - 3.0 * float(row["sigma"])
            for row in connection.execute(
                """SELECT r.mu, r.sigma FROM controllers c
                   JOIN component_ratings r ON r.component_id = c.controller_id
                   WHERE c.kind = 'checkpoint' AND c.active = 1
                     AND c.candidate_state = 'incumbent'"""
            )
        ),
        reverse=True,
    )
    if not values:
        return None
    index = max(0, min(len(values) - 1, (len(values) + 4) // 5 - 1))
    return values[index]


def _side_a_assignment(first_id: str, second_id: str, scheduled_games: int) -> bool:
    """Choose a reproducible seat without imposing a mirror or balance pair."""
    payload = f"{first_id}\0{second_id}\0{scheduled_games}".encode()
    return bool(hashlib.sha256(payload).digest()[0] & 1)


__all__ = ["ChallengeScheduler", "ScheduledBundle"]
