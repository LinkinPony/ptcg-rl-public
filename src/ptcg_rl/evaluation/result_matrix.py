"""Streaming adjudication of exact evaluation-bundle game results."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from ptcg_rl.evaluation.posterior import MatchupOutcome

ResolvedResult = Literal["win", "draw", "loss"]


class BundleMetadata(BaseModel):
    """Stable identity fields attached to an evaluated bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    pilot_id: str
    archetype: str
    deck_signature: str
    deck_hash: str
    deck_label: str
    variant_weight: float = 1.0


@dataclass(frozen=True)
class AdjudicatedMatrix:
    """Aggregated outcomes and diagnostics on a shared bundle support."""

    outcomes: tuple[MatchupOutcome, ...]
    candidates: Mapping[str, BundleMetadata]
    opponents: Mapping[str, BundleMetadata]
    cell_rows: tuple[Mapping[str, Any], ...]
    total_games: int
    resolved_games: int
    unresolved_games: int
    candidate_error_losses: int
    opponent_error_wins: int
    infrastructure_errors: int


class BundleResultAccumulator:
    """Incrementally aggregate raw game rows without retaining every game."""

    def __init__(self, *, require_balanced_seats: bool = True) -> None:
        self._require_balanced_seats = require_balanced_seats
        self._candidates: dict[str, BundleMetadata] = {}
        self._opponents: dict[str, BundleMetadata] = {}
        self._cells: dict[tuple[str, str], dict[str, int]] = {}
        self._total_games = 0
        self._candidate_error_losses = 0
        self._opponent_error_wins = 0
        self._infrastructure_errors = 0

    def observe(self, row: Mapping[str, Any]) -> None:
        """Adjudicate and aggregate one raw bundle-gauntlet row."""
        candidate = _metadata(row, prefix="candidate")
        opponent = _metadata(row, prefix="opponent")
        _register_metadata(self._candidates, candidate)
        _register_metadata(self._opponents, opponent)
        key = (candidate.bundle_id, opponent.bundle_id)
        cell = self._cells.setdefault(key, _empty_cell())
        self._total_games += 1
        seat = int(row.get("candidate_seat", -1))
        if seat in (0, 1):
            cell[f"seat_{seat}_games"] += 1

        result, reason = adjudicate_game(row)
        if result is None:
            cell["unresolved"] += 1
            if reason == "infrastructure_error":
                self._infrastructure_errors += 1
            return
        result_field = {"win": "wins", "draw": "draws", "loss": "losses"}[result]
        cell[result_field] += 1
        if seat in (0, 1):
            cell[f"seat_{seat}_resolved"] += 1
        if reason == "candidate_error":
            cell["candidate_errors"] += 1
            self._candidate_error_losses += 1
        elif reason == "opponent_error":
            cell["opponent_errors"] += 1
            self._opponent_error_wins += 1

    def finish(self) -> AdjudicatedMatrix:
        """Validate strata and return an immutable aggregated matrix."""
        if not self._cells:
            raise ValueError("bundle evaluation contains no game rows")
        cell_rows: list[Mapping[str, Any]] = []
        outcomes: list[MatchupOutcome] = []
        unresolved = 0
        resolved = 0
        for (candidate_id, opponent_id), counts in sorted(self._cells.items()):
            if (
                self._require_balanced_seats
                and counts["seat_0_games"] != counts["seat_1_games"]
            ):
                raise ValueError(
                    "candidate seats are not balanced for cell "
                    f"{candidate_id!r} vs {opponent_id!r}: "
                    f"{counts['seat_0_games']} != {counts['seat_1_games']}"
                )
            games = counts["wins"] + counts["draws"] + counts["losses"]
            resolved += games
            unresolved += counts["unresolved"]
            if games > 0:
                outcomes.append(
                    MatchupOutcome(
                        candidate_id=candidate_id,
                        opponent_id=opponent_id,
                        wins=counts["wins"],
                        draws=counts["draws"],
                        losses=counts["losses"],
                    )
                )
            cell_rows.append(
                {
                    "candidate_id": candidate_id,
                    "opponent_id": opponent_id,
                    "resolved_games": games,
                    **counts,
                }
            )
        return AdjudicatedMatrix(
            outcomes=tuple(outcomes),
            candidates=dict(self._candidates),
            opponents=dict(self._opponents),
            cell_rows=tuple(cell_rows),
            total_games=self._total_games,
            resolved_games=resolved,
            unresolved_games=unresolved,
            candidate_error_losses=self._candidate_error_losses,
            opponent_error_wins=self._opponent_error_wins,
            infrastructure_errors=self._infrastructure_errors,
        )


def adjudicate_game(
    row: Mapping[str, Any],
) -> tuple[ResolvedResult | None, str]:
    """Apply evaluation error semantics to one engine game result."""
    terminal_reason = str(row.get("terminal_reason", ""))
    result = str(row.get("candidate_result", ""))
    if terminal_reason == "finished" and result in {"win", "draw", "loss"}:
        return (cast(ResolvedResult, result), "engine_result")
    if terminal_reason == "agent_error":
        actor = str(row.get("error_actor", ""))
        if actor == "candidate":
            return ("loss", "candidate_error")
        if actor == "opponent":
            return ("win", "opponent_error")
        return (None, "infrastructure_error")
    if terminal_reason == "infrastructure_error":
        return (None, "infrastructure_error")
    return (None, "unresolved")


def _metadata(row: Mapping[str, Any], *, prefix: str) -> BundleMetadata:
    return BundleMetadata(
        bundle_id=_required_text(row, f"{prefix}_bundle_id"),
        pilot_id=_required_text(row, f"{prefix}_pilot_id"),
        archetype=_required_text(row, f"{prefix}_archetype"),
        deck_signature=_required_text(row, f"{prefix}_deck_signature"),
        deck_hash=str(row.get(f"{prefix}_deck_hash", "")),
        deck_label=str(row.get(f"{prefix}_deck_label", "")),
        variant_weight=float(row.get(f"{prefix}_variant_weight", 1.0)),
    )


def _register_metadata(
    registry: dict[str, BundleMetadata], metadata: BundleMetadata
) -> None:
    previous = registry.setdefault(metadata.bundle_id, metadata)
    if previous != metadata:
        raise ValueError(f"inconsistent metadata for bundle {metadata.bundle_id!r}")


def _required_text(row: Mapping[str, Any], field: str) -> str:
    value = str(row.get(field, "")).strip()
    if not value:
        raise ValueError(f"bundle game row is missing {field!r}")
    return value


def _empty_cell() -> dict[str, int]:
    return {
        "wins": 0,
        "draws": 0,
        "losses": 0,
        "unresolved": 0,
        "candidate_errors": 0,
        "opponent_errors": 0,
        "seat_0_games": 0,
        "seat_1_games": 0,
        "seat_0_resolved": 0,
        "seat_1_resolved": 0,
    }
