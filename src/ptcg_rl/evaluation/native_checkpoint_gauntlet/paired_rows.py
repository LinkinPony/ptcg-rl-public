"""Row-level integrity checks for paired gauntlet artifacts."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq

from ptcg_rl.evaluation.continuous_league.native_match import match_seeds
from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_provenance import participant

_UINT32_MAX = 2**32 - 1
_STATIC_FIELDS = (
    "format",
    "game_index",
    "candidate_deck_id",
    "candidate_deck_hash",
    "candidate_deck_label",
    "candidate_deck_signature",
    "candidate_deck_source",
    "baseline_deck_id",
    "baseline_deck_hash",
    "baseline_deck_label",
    "baseline_deck_signature",
    "baseline_deck_source",
    "candidate_seat",
    "baseline_seat",
    "baseline_checkpoint",
)
_SEED_FIELDS = ("engine_seed", "candidate_agent_seed", "baseline_agent_seed")
_FIRST_PLAYER_FIELDS = ("first_player", "candidate_went_first")
_REQUIRED_COLUMNS = (
    "campaign_fingerprint",
    "match_id",
    "candidate_checkpoint",
    *_SEED_FIELDS,
    "candidate_result",
    "baseline_result",
    "candidate_score",
    "baseline_score",
    "outcome",
    "terminal_reason",
    "steps",
    *_STATIC_FIELDS,
    *_FIRST_PLAYER_FIELDS,
)


def games_path(path: Path) -> Path:
    """Resolve either an artifact directory or its games parquet."""
    resolved = path.resolve()
    games = resolved / "games.parquet" if resolved.is_dir() else resolved
    if not games.is_file():
        raise ValueError(f"games parquet does not exist: {games}")
    return games


def read_rows(path: Path) -> list[dict[str, Any]]:
    """Read only columns needed for paired analysis."""
    try:
        columns = list(dict.fromkeys(_REQUIRED_COLUMNS))
        table = pq.read_table(path, columns=columns)
    except Exception as error:
        raise ValueError(
            f"cannot read required games columns from {path}: {error}"
        ) from error
    return cast(list[dict[str, Any]], table.to_pylist())


def validate_artifact(
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    arm: str,
    allow_unresolved: bool = False,
    maximum_engine_steps: int | None = None,
) -> None:
    """Validate completion, score coding, provenance binding, and seed domain."""
    if not rows:
        raise ValueError(f"{arm} games artifact is empty")
    if _summary_count(summary, "games") != len(rows):
        raise ValueError(f"{arm} summary game count differs from games parquet")
    fingerprint = summary.get("campaign_fingerprint")
    candidate_label = participant(summary, "candidate").get("label")
    baseline_label = participant(summary, "baseline").get("label")
    if not all(
        isinstance(value, str) and value
        for value in (fingerprint, candidate_label, baseline_label)
    ):
        raise ValueError(f"{arm} summary labels or campaign fingerprint are invalid")
    game_indices: list[int] = []
    unresolved_games = 0
    for row in rows:
        match_id = row.get("match_id")
        if not isinstance(match_id, str) or not match_id:
            raise ValueError(f"{arm} artifact contains an invalid match_id")
        if row.get("format") != "native_checkpoint_gauntlet_games_v1":
            raise ValueError(f"{arm} artifact has an unsupported row format")
        if row.get("campaign_fingerprint") != fingerprint:
            raise ValueError(f"{arm} row campaign fingerprint differs from summary")
        if row.get("candidate_checkpoint") != candidate_label:
            raise ValueError(f"{arm} candidate label differs from summary")
        if row.get("baseline_checkpoint") != baseline_label:
            raise ValueError(f"{arm} baseline label differs from summary")
        _validate_seats_and_first_player(row, arm=arm)
        resolved = _validate_outcome(
            row,
            arm=arm,
            allow_unresolved=allow_unresolved,
            maximum_engine_steps=maximum_engine_steps,
        )
        unresolved_games += int(not resolved)
        for field in _SEED_FIELDS:
            _validate_uint32(row.get(field), label=f"{arm} {field}")
        _validate_seed_derivation(row, arm=arm)
        game_index = row.get("game_index")
        if isinstance(game_index, bool) or not isinstance(game_index, int):
            raise ValueError(f"{arm} artifact contains an invalid game_index")
        game_indices.append(game_index)
    if sorted(game_indices) != list(range(len(rows))):
        raise ValueError(f"{arm} game_index values are not contiguous and unique")
    if _summary_count(summary, "resolved_games") != len(rows) - unresolved_games:
        raise ValueError(f"{arm} summary resolved count differs from games parquet")
    if _summary_count(summary, "unresolved_games") != unresolved_games:
        raise ValueError(f"{arm} summary unresolved count differs from games parquet")


def validate_rosters(
    control_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    control_summary: dict[str, Any],
    treatment_summary: dict[str, Any],
    control_manifest: dict[str, Any],
    treatment_manifest: dict[str, Any],
) -> None:
    """Require summary counts and exact deck identities to agree in both arms."""
    for role in ("candidate", "baseline"):
        control_map = _deck_map(control_rows, role=role)
        treatment_map = _deck_map(treatment_rows, role=role)
        if control_map != treatment_map:
            raise ValueError(f"{role} exact roster differs between paired arms")
        for arm, manifest in (
            ("control", control_manifest),
            ("treatment", treatment_manifest),
        ):
            if _manifest_deck_map(manifest, role=role) != control_map:
                raise ValueError(f"{arm} {role} rows differ from campaign manifest")
        actual_count = len(control_map)
        for arm, summary in (
            ("control", control_summary),
            ("treatment", treatment_summary),
        ):
            if _summary_count(summary, f"{role}_decks") != actual_count:
                raise ValueError(f"{arm} summary {role} deck count differs from rows")
            roster_count = participant(summary, role).get("exact_roster_decks")
            if roster_count != actual_count:
                raise ValueError(
                    f"{arm} participant {role} roster count differs from rows"
                )


def index_matches(rows: list[dict[str, Any]], *, arm: str) -> dict[str, dict[str, Any]]:
    """Build a strict unique match index."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        match_id = str(row["match_id"])
        if match_id in indexed:
            raise ValueError(f"{arm} artifact contains duplicate match_id {match_id}")
        indexed[match_id] = row
    return indexed


def validate_pair_fields(
    control: dict[str, Any], treatment: dict[str, Any], *, match_id: str
) -> None:
    """Require all schedule/static fields and role-aligned seeds to match."""
    for field in _STATIC_FIELDS:
        if control.get(field) != treatment.get(field):
            raise ValueError(f"paired match {match_id} differs in static field {field}")
    for field in _SEED_FIELDS:
        if control.get(field) != treatment.get(field):
            raise ValueError(
                f"paired match {match_id} differs in role-aligned seed {field}"
            )


def score(row: dict[str, Any], *, arm: str) -> float:
    """Return a validated W/D/L candidate score."""
    result = row.get("candidate_score")
    if isinstance(result, bool) or not isinstance(result, (int, float)):
        raise ValueError(f"{arm} candidate score is not numeric")
    value = float(result)
    if not math.isfinite(value) or value not in (0.0, 0.5, 1.0):
        raise ValueError(f"{arm} candidate score is not a W/D/L score")
    return value


def candidate_went_first(row: dict[str, Any]) -> int:
    """Return actual first-player status as -1/0/1 after row validation."""
    value = row["candidate_went_first"]
    return -1 if value is None else int(value)


def validate_balance(
    deck_ids: np.ndarray,
    seats: np.ndarray,
    *,
    expected_decks: int | None,
    expected_repeats: int | None,
) -> dict[str, Any]:
    """Require a complete baseline-deck by candidate-seat rectangular design."""
    _validate_optional_count(expected_decks, "expected_decks")
    _validate_optional_count(expected_repeats, "expected_repeats")
    decks = sorted(str(value) for value in np.unique(deck_ids))
    seat_values = {int(value) for value in np.unique(seats)}
    if seat_values != {0, 1}:
        raise ValueError("paired design must contain both candidate seats")
    counts = Counter(
        (str(deck_id), int(seat)) for deck_id, seat in zip(deck_ids, seats, strict=True)
    )
    expected_keys = {(deck_id, seat) for deck_id in decks for seat in (0, 1)}
    if set(counts) != expected_keys or len(set(counts.values())) != 1:
        raise ValueError("baseline_deck_id by candidate_seat strata are not balanced")
    repeats = next(iter(counts.values()))
    if repeats < 2:
        raise ValueError(
            "paired sandwich inference requires at least two repeats per stratum"
        )
    if expected_decks is not None and len(decks) != expected_decks:
        raise ValueError("observed baseline deck count differs from expected_decks")
    if expected_repeats is not None and repeats != expected_repeats:
        raise ValueError("observed stratum repeats differ from expected_repeats")
    return {
        "baseline_exact_decks": len(decks),
        "candidate_seats": 2,
        "repeats_per_deck_seat": repeats,
        "balanced_strata": len(counts),
        "expected_decks": expected_decks,
        "expected_repeats": expected_repeats,
        "balance_expectation_mode": (
            "explicit"
            if expected_decks is not None or expected_repeats is not None
            else "inferred"
        ),
    }


def is_resolved(row: dict[str, Any]) -> bool:
    """Return whether a row has a scored terminal outcome."""
    return row.get("candidate_result") != "unresolved"


def _validate_outcome(
    row: dict[str, Any],
    *,
    arm: str,
    allow_unresolved: bool,
    maximum_engine_steps: int | None,
) -> bool:
    reason = row.get("terminal_reason")
    if reason in ("infrastructure_error", "both_sides_error"):
        raise ValueError(f"{arm} artifact contains an infrastructure error")
    candidate = row.get("candidate_result")
    if candidate == "unresolved":
        if not allow_unresolved:
            raise ValueError(f"{arm} artifact contains an unresolved game")
        if (
            row.get("baseline_result") != "unresolved"
            or row.get("outcome") != "unresolved"
            or not _is_nan(row.get("candidate_score"))
            or not _is_nan(row.get("baseline_score"))
        ):
            raise ValueError(f"{arm} unresolved outcome fields are inconsistent")
        steps = row.get("steps")
        if (
            reason != "max_steps"
            or isinstance(maximum_engine_steps, bool)
            or not isinstance(maximum_engine_steps, int)
            or maximum_engine_steps <= 0
            or isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < maximum_engine_steps
        ):
            raise ValueError(
                f"{arm} unresolved game is not a verified engine step-cap outcome"
            )
        return False
    if candidate not in ("win", "loss", "draw") or row.get("outcome") == "unresolved":
        raise ValueError(f"{arm} result is invalid")
    outcome = row.get("outcome")
    resolved_reason_outcome = {
        "normal": None,
        "side_a_timeout": "side_b_win",
        "side_b_timeout": "side_a_win",
        "side_a_act_error": "side_b_win",
        "side_b_act_error": "side_a_win",
        "side_a_illegal_action": "side_b_win",
        "side_b_illegal_action": "side_a_win",
    }
    if reason not in resolved_reason_outcome:
        raise ValueError(f"{arm} resolved terminal reason is invalid")
    expected_fault_outcome = resolved_reason_outcome[str(reason)]
    if expected_fault_outcome is not None and outcome != expected_fault_outcome:
        raise ValueError(f"{arm} participant fault outcome is inconsistent")
    inverse = {"win": "loss", "loss": "win", "draw": "draw"}[str(candidate)]
    if row.get("baseline_result") != inverse:
        raise ValueError(f"{arm} candidate and baseline results are inconsistent")
    value = score(row, arm=arm)
    expected = {"win": 1.0, "loss": 0.0, "draw": 0.5}[str(candidate)]
    baseline = row.get("baseline_score")
    if (
        value != expected
        or not isinstance(baseline, (int, float))
        or isinstance(baseline, bool)
    ):
        raise ValueError(f"{arm} result and score coding are inconsistent")
    if not math.isclose(float(baseline), 1.0 - value, abs_tol=1e-12):
        raise ValueError(f"{arm} candidate and baseline scores are inconsistent")
    candidate_seat = int(row["candidate_seat"])
    if candidate == "draw":
        expected_outcome = "draw"
    else:
        winner = candidate_seat if candidate == "win" else 1 - candidate_seat
        expected_outcome = "side_a_win" if winner == 0 else "side_b_win"
    if outcome != expected_outcome:
        raise ValueError(f"{arm} native outcome is inconsistent with candidate result")
    return True


def _is_nan(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isnan(float(value))
    )


def _validate_seats_and_first_player(row: dict[str, Any], *, arm: str) -> None:
    candidate_seat = row.get("candidate_seat")
    baseline_seat = row.get("baseline_seat")
    first_player = row.get("first_player")
    went_first = row.get("candidate_went_first")
    if any(isinstance(value, bool) for value in (candidate_seat, baseline_seat)):
        raise ValueError(f"{arm} seat or first-player field is invalid")
    if candidate_seat not in (0, 1) or baseline_seat != 1 - int(candidate_seat):
        raise ValueError(f"{arm} logical seats are invalid")
    if first_player is None and went_first is None:
        return
    if isinstance(first_player, bool):
        raise ValueError(f"{arm} actual first-player field is invalid")
    if first_player not in (0, 1) or not isinstance(went_first, bool):
        raise ValueError(f"{arm} actual first-player field is invalid")
    if went_first != (first_player == candidate_seat):
        raise ValueError(f"{arm} candidate_went_first is inconsistent")


def _validate_uint32(value: Any, *, label: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _UINT32_MAX
    ):
        raise ValueError(f"{label} is not a valid uint32 seed")


def _validate_seed_derivation(row: dict[str, Any], *, arm: str) -> None:
    expected_engine, seat_seeds = match_seeds(str(row["match_id"]))
    candidate_seat = int(row["candidate_seat"])
    expected = {
        "engine_seed": expected_engine,
        "candidate_agent_seed": seat_seeds[candidate_seat],
        "baseline_agent_seed": seat_seeds[1 - candidate_seat],
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{arm} {field} does not match its match_id and role")


def _summary_count(summary: dict[str, Any], field: str) -> int:
    value = summary.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"summary {field} is not a non-negative integer")
    return value


def _deck_map(
    rows: list[dict[str, Any]], *, role: str
) -> dict[str, tuple[str, str, str, str]]:
    output: dict[str, tuple[str, str, str, str]] = {}
    for row in rows:
        deck_id = _nonempty_string(row[f"{role}_deck_id"], f"{role} deck id")
        metadata = (
            _nonempty_string(row[f"{role}_deck_hash"], f"{role} deck hash"),
            _nonempty_string(row[f"{role}_deck_label"], f"{role} deck label"),
            _nonempty_string(row[f"{role}_deck_signature"], f"{role} deck signature"),
            _nonempty_string(row[f"{role}_deck_source"], f"{role} deck source"),
        )
        if deck_id in output and output[deck_id] != metadata:
            raise ValueError(f"{role} deck metadata is inconsistent for {deck_id}")
        output[deck_id] = metadata
    return output


def _manifest_deck_map(
    manifest: dict[str, Any], *, role: str
) -> dict[str, tuple[str, str, str, str]]:
    participant_value = manifest.get(role)
    if not isinstance(participant_value, dict):
        raise ValueError(f"manifest {role} participant is missing")
    decks = participant_value.get("decks")
    if not isinstance(decks, list) or not decks:
        raise ValueError(f"manifest {role} deck roster is missing")
    output: dict[str, tuple[str, str, str, str]] = {}
    for raw in decks:
        if not isinstance(raw, dict):
            raise ValueError(f"manifest {role} deck entry is invalid")
        deck_id = _nonempty_string(raw.get("deck_digest"), f"manifest {role} digest")
        metadata = (
            _nonempty_string(raw.get("deck_hash"), f"manifest {role} deck hash"),
            _nonempty_string(raw.get("label"), f"manifest {role} deck label"),
            _nonempty_string(
                raw.get("deck_signature"), f"manifest {role} deck signature"
            ),
            _nonempty_string(raw.get("path"), f"manifest {role} deck path"),
        )
        if deck_id in output:
            raise ValueError(f"manifest {role} contains duplicate deck digest")
        output[deck_id] = metadata
    return output


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is missing or invalid")
    return value


def _validate_optional_count(value: int | None, label: str) -> None:
    if value is not None and (isinstance(value, bool) or value <= 0):
        raise ValueError(f"{label} must be positive when provided")
