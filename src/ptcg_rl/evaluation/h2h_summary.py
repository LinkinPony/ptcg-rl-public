"""Streaming W/D/L summaries for exact bundle head-to-head results."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.result_matrix import ResolvedResult, adjudicate_game

_RESULTS: tuple[ResolvedResult, ...] = ("win", "draw", "loss")
_POSTERIOR_SAMPLES = 65_536
_SUMMARY_COLUMNS = (
    "candidate_bundle_id",
    "opponent_bundle_id",
    "candidate_seat",
    "candidate_result",
    "terminal_reason",
    "error_actor",
    "candidate_illegal_actions",
    "opponent_illegal_actions",
    "steps",
    "candidate_decisions",
    "opponent_decisions",
    "candidate_action_seconds",
    "opponent_action_seconds",
)


@dataclass
class _OutcomeAggregate:
    """Constant-memory aggregate for one candidate or matchup."""

    games: int = 0
    engine_finished: int = 0
    unresolved: int = 0
    candidate_errors: int = 0
    opponent_errors: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    seat_outcomes: dict[int, Counter[str]] = field(
        default_factory=lambda: {0: Counter(), 1: Counter()}
    )
    candidate_illegal_actions: int = 0
    opponent_illegal_actions: int = 0
    steps: int = 0
    candidate_decisions: int = 0
    opponent_decisions: int = 0
    candidate_action_seconds: float = 0.0
    opponent_action_seconds: float = 0.0

    def observe(self, row: Mapping[str, Any]) -> None:
        """Consume one raw bundle-gauntlet result row."""
        self.games += 1
        self.engine_finished += int(str(row["terminal_reason"]) == "finished")
        result, reason = adjudicate_game(row)
        if result is None:
            self.unresolved += 1
        else:
            self.outcomes[result] += 1
            seat = int(row["candidate_seat"])
            if seat in self.seat_outcomes:
                self.seat_outcomes[seat][result] += 1
        self.candidate_errors += int(reason == "candidate_error")
        self.opponent_errors += int(reason == "opponent_error")
        self.candidate_illegal_actions += int(row["candidate_illegal_actions"])
        self.opponent_illegal_actions += int(row["opponent_illegal_actions"])
        self.steps += int(row["steps"])
        self.candidate_decisions += int(row["candidate_decisions"])
        self.opponent_decisions += int(row["opponent_decisions"])
        self.candidate_action_seconds += float(row["candidate_action_seconds"])
        self.opponent_action_seconds += float(row["opponent_action_seconds"])

    def row(self, *, identity: str) -> dict[str, Any]:
        """Return stable scalar fields suitable for JSON and Parquet."""
        wins = self.outcomes["win"]
        draws = self.outcomes["draw"]
        losses = self.outcomes["loss"]
        resolved = wins + draws + losses
        posterior = _score_posterior(self.seat_outcomes, identity=identity)
        return {
            "games": self.games,
            "engine_finished": self.engine_finished,
            "resolved": resolved,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "unresolved": self.unresolved,
            "score_rate": ((wins + 0.5 * draws) / resolved if resolved > 0 else None),
            "score_posterior_mean": (
                float(np.mean(posterior)) if posterior is not None else None
            ),
            "score_ci95_lower": (
                float(np.quantile(posterior, 0.025)) if posterior is not None else None
            ),
            "score_ci95_upper": (
                float(np.quantile(posterior, 0.975)) if posterior is not None else None
            ),
            "seat_0_wins": self.seat_outcomes[0]["win"],
            "seat_0_draws": self.seat_outcomes[0]["draw"],
            "seat_0_losses": self.seat_outcomes[0]["loss"],
            "seat_1_wins": self.seat_outcomes[1]["win"],
            "seat_1_draws": self.seat_outcomes[1]["draw"],
            "seat_1_losses": self.seat_outcomes[1]["loss"],
            "candidate_illegal_actions": self.candidate_illegal_actions,
            "opponent_illegal_actions": self.opponent_illegal_actions,
            "candidate_errors": self.candidate_errors,
            "opponent_errors": self.opponent_errors,
            "mean_steps": self.steps / self.games if self.games > 0 else None,
            "candidate_mean_act_ms": _mean_milliseconds(
                self.candidate_action_seconds,
                self.candidate_decisions,
            ),
            "opponent_mean_act_ms": _mean_milliseconds(
                self.opponent_action_seconds,
                self.opponent_decisions,
            ),
        }


def summarize_bundle_h2h(
    games_path: Path,
    *,
    output_dir: Path,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Stream one games Parquet into candidate and matchup score artifacts."""
    candidates: dict[str, _OutcomeAggregate] = {}
    matchups: dict[tuple[str, str], _OutcomeAggregate] = {}
    total = _OutcomeAggregate()
    parquet = pq.ParquetFile(games_path)
    missing = set(_SUMMARY_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"bundle games are missing summary columns: {sorted(missing)}")

    for batch in parquet.iter_batches(
        columns=list(_SUMMARY_COLUMNS), batch_size=65_536
    ):
        for row in batch.to_pylist():
            candidate_id = str(row["candidate_bundle_id"])
            opponent_id = str(row["opponent_bundle_id"])
            candidates.setdefault(candidate_id, _OutcomeAggregate()).observe(row)
            matchups.setdefault(
                (candidate_id, opponent_id),
                _OutcomeAggregate(),
            ).observe(row)
            total.observe(row)
    if total.games <= 0:
        raise ValueError("cannot summarize an empty bundle games artifact")

    candidate_rows = [
        {
            "candidate_bundle_id": candidate_id,
            **aggregate.row(identity=f"candidate:{candidate_id}"),
        }
        for candidate_id, aggregate in sorted(candidates.items())
    ]
    matchup_rows = [
        {
            "candidate_bundle_id": candidate_id,
            "opponent_bundle_id": opponent_id,
            **aggregate.row(identity=f"matchup:{candidate_id}:{opponent_id}"),
        }
        for (candidate_id, opponent_id), aggregate in sorted(matchups.items())
    ]
    candidates_path = output_dir / "candidate_summary.parquet"
    matchups_path = output_dir / "matchups.parquet"
    _write_rows(candidates_path, candidate_rows, compression=compression)
    _write_rows(matchups_path, matchup_rows, compression=compression)
    return {
        "protocol": "bundle-h2h-summary-v1",
        "posterior_samples": _POSTERIOR_SAMPLES,
        "posterior": "Jeffreys Dirichlet per seat; equal-seat score",
        "games": total.games,
        "candidate_bundles": len(candidate_rows),
        "matchups": len(matchup_rows),
        "quality": total.row(identity="all-games"),
        "single_candidate": candidate_rows[0] if len(candidate_rows) == 1 else None,
        "candidates_path": records.display_path(candidates_path),
        "matchups_path": records.display_path(matchups_path),
    }


def _score_posterior(
    seat_outcomes: Mapping[int, Counter[str]],
    *,
    identity: str,
) -> np.ndarray[Any, np.dtype[np.float64]] | None:
    rng = np.random.default_rng(_stable_seed(identity))
    seat_scores: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    for seat in (0, 1):
        counts = seat_outcomes[seat]
        resolved = sum(counts[result] for result in _RESULTS)
        if resolved <= 0:
            continue
        posterior = rng.dirichlet(
            np.asarray(
                [counts[result] + 0.5 for result in _RESULTS],
                dtype=np.float64,
            ),
            size=_POSTERIOR_SAMPLES,
        )
        seat_scores.append(posterior[:, 0] + 0.5 * posterior[:, 1])
    if not seat_scores:
        return None
    return cast(
        np.ndarray[Any, np.dtype[np.float64]],
        np.mean(np.stack(seat_scores), axis=0),
    )


def _stable_seed(identity: str) -> int:
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _mean_milliseconds(seconds: float, decisions: int) -> float | None:
    return 1000.0 * seconds / decisions if decisions > 0 else None


def _write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    compression: str,
) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty H2H summary: {path}")
    pq.write_table(pa.Table.from_pylist(rows), path, compression=compression)


__all__ = ["summarize_bundle_h2h"]
