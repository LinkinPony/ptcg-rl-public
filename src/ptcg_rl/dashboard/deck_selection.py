"""Typed deck-selection projections over runtime ladder artifacts."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from ptcg_rl.dashboard.deck_selection_models import (
    DeckSelectionCell,
    DeckSelectionPayload,
    DeckSelectionQuality,
    DeckSelectionStanding,
)
from ptcg_rl.dashboard.task_models import TaskReceipt

_POSTERIOR_SAMPLES = 4096


def build_deck_selection_payload(
    receipt: TaskReceipt,
    *,
    spec: Mapping[str, Any],
    summary: Mapping[str, Any],
    matchup_rows: Sequence[Mapping[str, Any]],
    result_fingerprint: str | None,
) -> DeckSelectionPayload:
    """Build equal-opponent standings and a directed matchup matrix."""
    request = _mapping(spec.get("request"))
    if request.get("kind") != "runtime_elo":
        raise ValueError("deck selection requires a runtime Elo task")
    resolved = _mapping(spec.get("resolved_inputs"))
    checkpoint = _mapping(resolved.get("checkpoint"))
    resolved_decks = [_mapping(item) for item in _sequence(resolved.get("decks"))]
    runtime_deck_ids = tuple(
        str(item.get("runtime_deck_id"))
        for item in resolved_decks
        if item.get("runtime_deck_id")
    )
    requested_deck_ids = tuple(
        str(item)
        for item in _sequence(request.get("deck_ids"))
        if isinstance(item, str)
    )
    expected_runtime_ids = (
        runtime_deck_ids
        if len(runtime_deck_ids) == len(requested_deck_ids)
        else requested_deck_ids
    )
    rng = np.random.default_rng(_stable_seed(receipt.spec_fingerprint))
    cells, posterior_samples = _directed_cells(matchup_rows, rng=rng)
    standings = _standings(cells, posterior_samples=posterior_samples)
    observed_decks = len(standings)
    expected_decks = len(requested_deck_ids)
    expected_matchups = expected_decks * max(0, expected_decks - 1) // 2
    expected_games_per_matchup = max(
        0,
        _int_value(request.get("games_per_pair")),
    )
    expected_games = expected_matchups * expected_games_per_matchup
    observed_games = sum(max(0, _int_value(row.get("games"))) for row in matchup_rows)
    pair_keys = {
        tuple(sorted((str(row.get("deck_a_id")), str(row.get("deck_b_id")))))
        for row in matchup_rows
    }
    expected_pair_keys = set(combinations(sorted(expected_runtime_ids), 2))
    observed_matchups = len(pair_keys)
    matchup_workload_complete = all(
        _int_value(row.get("games")) == expected_games_per_matchup
        for row in matchup_rows
    )
    agent_errors = sum(
        max(0, _int_value(row.get("agent_error_games"))) for row in matchup_rows
    )
    truncated = sum(max(0, _int_value(row.get("truncated"))) for row in matchup_rows)
    seat_balanced = bool(cells) and all(
        cell.seat_0_games == cell.seat_1_games and cell.seat_0_games > 0
        for cell in cells
    )
    warnings = _quality_warnings(
        receipt=receipt,
        expected_decks=expected_decks,
        observed_decks=observed_decks,
        deck_set_matches={cell.candidate_id for cell in cells}
        == set(expected_runtime_ids),
        expected_matchups=expected_matchups,
        observed_matchups=observed_matchups,
        matchup_set_matches=pair_keys == expected_pair_keys,
        matchup_rows=len(matchup_rows),
        expected_games=expected_games,
        observed_games=observed_games,
        matchup_workload_complete=matchup_workload_complete,
        seat_balanced=seat_balanced,
        agent_errors=agent_errors,
        truncated=truncated,
        summary=summary,
    )
    quality = DeckSelectionQuality(
        ready=not warnings,
        expected_decks=expected_decks,
        observed_decks=observed_decks,
        expected_matchups=expected_matchups,
        observed_matchups=observed_matchups,
        expected_games_per_matchup=expected_games_per_matchup,
        expected_games=expected_games,
        observed_games=observed_games,
        seat_balanced=seat_balanced,
        agent_error_games=agent_errors,
        truncated_games=truncated,
        warnings=warnings,
    )
    finalized = tuple(
        row.model_copy(
            update={
                "evidence_state": (
                    "ready"
                    if quality.ready and row.evidence_state == "ready"
                    else "missing"
                    if row.evidence_state == "missing"
                    else "incomplete"
                )
            }
        )
        for row in standings
    )
    return DeckSelectionPayload(
        task_id=receipt.task_id,
        task_state=receipt.state,
        semantics="same_checkpoint_equal_opponent_deck_selection_v1",
        checkpoint_id=str(request.get("checkpoint_id", "")),
        checkpoint_fingerprint=_optional_text(checkpoint.get("policy_sha256")),
        spec_fingerprint=receipt.spec_fingerprint,
        result_fingerprint=result_fingerprint,
        quality=quality,
        standings=finalized,
        cells=cells,
    )


def _directed_cells(
    matchup_rows: Sequence[Mapping[str, Any]],
    *,
    rng: np.random.Generator,
) -> tuple[
    tuple[DeckSelectionCell, ...],
    dict[tuple[str, str], NDArray[np.float64]],
]:
    cells: list[DeckSelectionCell] = []
    samples: dict[tuple[str, str], NDArray[np.float64]] = {}
    ordered_rows = sorted(
        matchup_rows,
        key=lambda row: (str(row.get("deck_a_id")), str(row.get("deck_b_id"))),
    )
    for row in ordered_rows:
        wins = max(0, _int_value(row.get("deck_a_wins")))
        draws = max(0, _int_value(row.get("draws")))
        losses = max(0, _int_value(row.get("deck_b_wins")))
        resolved = wins + draws + losses
        posterior_a: NDArray[np.float64] | None = None
        if resolved > 0:
            posterior_a = rng.beta(
                1.0 + wins + 0.5 * draws,
                1.0 + losses + 0.5 * draws,
                size=_POSTERIOR_SAMPLES,
            )
        cell_a = _cell(
            row,
            reverse=False,
            wins=wins,
            draws=draws,
            losses=losses,
            posterior_samples=posterior_a,
        )
        cell_b = _cell(
            row,
            reverse=True,
            wins=losses,
            draws=draws,
            losses=wins,
            posterior_samples=None if posterior_a is None else 1.0 - posterior_a,
        )
        cells.extend((cell_a, cell_b))
        if posterior_a is not None:
            samples[(cell_a.candidate_id, cell_a.opponent_id)] = posterior_a
            samples[(cell_b.candidate_id, cell_b.opponent_id)] = 1.0 - posterior_a
    cells.sort(key=lambda cell: (cell.candidate_label, cell.opponent_label))
    return tuple(cells), samples


def _cell(
    row: Mapping[str, Any],
    *,
    reverse: bool,
    wins: int,
    draws: int,
    losses: int,
    posterior_samples: NDArray[np.float64] | None,
) -> DeckSelectionCell:
    candidate_prefix = "deck_b" if reverse else "deck_a"
    opponent_prefix = "deck_a" if reverse else "deck_b"
    truncated = max(0, _int_value(row.get("truncated")))
    errors = max(0, _int_value(row.get("agent_error_games")))
    resolved = wins + draws + losses
    seat_0_games = max(
        0,
        _int_value(row.get(f"{candidate_prefix}_seat_0_games")),
    )
    seat_1_games = max(
        0,
        _int_value(row.get(f"{candidate_prefix}_seat_1_games")),
    )
    seat_ready = seat_0_games > 0 and seat_0_games == seat_1_games
    state: Literal["ready", "incomplete", "missing"]
    if resolved == 0:
        state = "missing"
    elif truncated or errors or not seat_ready:
        state = "incomplete"
    else:
        state = "ready"
    return DeckSelectionCell(
        candidate_id=str(row.get(f"{candidate_prefix}_id", "")),
        candidate_hash=str(row.get(f"{candidate_prefix}_hash", "")),
        opponent_id=str(row.get(f"{opponent_prefix}_id", "")),
        opponent_hash=str(row.get(f"{opponent_prefix}_hash", "")),
        candidate_label=str(row.get(f"{candidate_prefix}_label", "")),
        opponent_label=str(row.get(f"{opponent_prefix}_label", "")),
        games=resolved,
        wins=wins,
        draws=draws,
        losses=losses,
        truncated=truncated,
        agent_error_games=errors,
        seat_0_games=seat_0_games,
        seat_1_games=seat_1_games,
        score_rate=(None if resolved == 0 else (wins + 0.5 * draws) / float(resolved)),
        posterior_mean=(
            None if posterior_samples is None else float(np.mean(posterior_samples))
        ),
        credible_low=_quantile(posterior_samples, 0.025),
        credible_high=_quantile(posterior_samples, 0.975),
        evidence_state=state,
    )


def _standings(
    cells: Sequence[DeckSelectionCell],
    *,
    posterior_samples: Mapping[tuple[str, str], NDArray[np.float64]],
) -> tuple[DeckSelectionStanding, ...]:
    grouped: dict[str, list[DeckSelectionCell]] = defaultdict(list)
    for cell in cells:
        grouped[cell.candidate_id].append(cell)
    rows: list[DeckSelectionStanding] = []
    for deck_id, deck_cells in grouped.items():
        first = deck_cells[0]
        scored = [cell for cell in deck_cells if cell.score_rate is not None]
        scores = [
            float(cell.score_rate) for cell in scored if cell.score_rate is not None
        ]
        sample_arrays = [
            posterior_samples[(cell.candidate_id, cell.opponent_id)]
            for cell in scored
            if (cell.candidate_id, cell.opponent_id) in posterior_samples
        ]
        aggregate_samples = (
            None
            if not sample_arrays
            else np.mean(np.stack(sample_arrays, axis=0), axis=0)
        )
        worst = min(
            scored,
            key=lambda cell: (
                float(cell.score_rate if cell.score_rate is not None else 1.0),
                cell.opponent_label,
            ),
            default=None,
        )
        best = max(
            scored,
            key=lambda cell: (
                float(cell.score_rate if cell.score_rate is not None else -1.0),
                cell.opponent_label,
            ),
            default=None,
        )
        state: Literal["ready", "incomplete", "missing"]
        if not scored:
            state = "missing"
        elif any(cell.evidence_state != "ready" for cell in deck_cells):
            state = "incomplete"
        else:
            state = "ready"
        rows.append(
            DeckSelectionStanding(
                rank=1,
                deck_id=deck_id,
                deck_hash=first.candidate_hash,
                deck_label=first.candidate_label,
                games=sum(cell.games for cell in deck_cells),
                wins=sum(cell.wins for cell in deck_cells),
                draws=sum(cell.draws for cell in deck_cells),
                losses=sum(cell.losses for cell in deck_cells),
                truncated=sum(cell.truncated for cell in deck_cells),
                opponent_count=len(scored),
                equal_score_rate=None if not scores else sum(scores) / len(scores),
                posterior_mean=(
                    None
                    if aggregate_samples is None
                    else float(np.mean(aggregate_samples))
                ),
                credible_low=_quantile(aggregate_samples, 0.025),
                credible_high=_quantile(aggregate_samples, 0.975),
                worst_opponent_id=None if worst is None else worst.opponent_id,
                worst_opponent_label=None if worst is None else worst.opponent_label,
                worst_matchup_score=None if worst is None else worst.score_rate,
                best_opponent_id=None if best is None else best.opponent_id,
                best_opponent_label=None if best is None else best.opponent_label,
                best_matchup_score=None if best is None else best.score_rate,
                evidence_state=state,
            )
        )
    rows.sort(
        key=lambda row: (
            -float(row.equal_score_rate if row.equal_score_rate is not None else -1.0),
            row.deck_label,
        )
    )
    return tuple(
        row.model_copy(update={"rank": rank}) for rank, row in enumerate(rows, start=1)
    )


def _quality_warnings(
    *,
    receipt: TaskReceipt,
    expected_decks: int,
    observed_decks: int,
    deck_set_matches: bool,
    expected_matchups: int,
    observed_matchups: int,
    matchup_set_matches: bool,
    matchup_rows: int,
    expected_games: int,
    observed_games: int,
    matchup_workload_complete: bool,
    seat_balanced: bool,
    agent_errors: int,
    truncated: int,
    summary: Mapping[str, Any],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if receipt.state != "succeeded":
        warnings.append("评测尚未成功完成")
    if expected_decks < 2:
        warnings.append("至少需要两个卡组")
    if observed_decks != expected_decks:
        warnings.append(f"卡组结果不完整：期望 {expected_decks}，实际 {observed_decks}")
    elif not deck_set_matches:
        warnings.append("结果中的卡组身份与任务请求不一致")
    if observed_matchups != expected_matchups:
        warnings.append(
            f"Matchup 不完整：期望 {expected_matchups}，实际 {observed_matchups}"
        )
    elif not matchup_set_matches:
        warnings.append("结果中的 matchup 集合与完整 round robin 不一致")
    if matchup_rows != observed_matchups:
        warnings.append("存在重复的卡组 matchup 结果")
    if observed_games != expected_games or not matchup_workload_complete:
        warnings.append(
            f"对局工作量不完整：期望 {expected_games}，实际 {observed_games}"
        )
    if observed_matchups > 0 and not seat_balanced:
        warnings.append("存在未镜像的先后手证据")
    if agent_errors:
        warnings.append(f"存在 {agent_errors} 局 agent error")
    if truncated:
        warnings.append(f"存在 {truncated} 局未决或超步数对局")
    if _int_value(summary.get("agent_error_games")) != agent_errors:
        warnings.append("汇总与 matchup 的 agent error 计数不一致")
    return tuple(warnings)


def _stable_seed(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big", signed=False)


def _quantile(
    values: NDArray[np.float64] | None,
    quantile: float,
) -> float | None:
    return None if values is None else float(np.quantile(values, quantile))


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> Sequence[object]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_value(value: object) -> int:
    try:
        if isinstance(value, (int, float, str)):
            return int(value)
    except (TypeError, ValueError):
        pass
    return 0


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
