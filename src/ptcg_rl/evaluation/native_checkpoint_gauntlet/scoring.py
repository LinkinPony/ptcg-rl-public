"""Aggregation and reporting for native cross-checkpoint gauntlets."""

from __future__ import annotations

import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    NativeCheckpointGauntletConfig,
)
from ptcg_rl.evaluation.native_deck_elo.storage import (
    safe_rate,
    write_json_atomic,
    write_text_atomic,
)


def write_final_artifacts(
    config: NativeCheckpointGauntletConfig,
    *,
    rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
    campaign_fingerprint: str,
    elapsed_seconds: float,
    resumed_games: int,
    execution_telemetry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish compact results and a finite summary after full completion."""
    ordered = sorted(rows, key=lambda row: int(row["game_index"]))
    games_path = output_dir / "games.parquet"
    _write_parquet(games_path, ordered, compression=config.compression)
    candidate_rows = _group_rows(
        ordered,
        id_key="candidate_deck_id",
        role="candidate",
    )
    baseline_rows = _group_rows(
        ordered,
        id_key="baseline_deck_id",
        role="baseline",
    )
    cell_rows = _cell_rows(ordered)
    same_deck_rows = [
        row for row in cell_rows if row["candidate_deck_id"] == row["baseline_deck_id"]
    ]
    candidate_path = output_dir / "candidate_decks.parquet"
    baseline_path = output_dir / "baseline_decks.parquet"
    cells_path = output_dir / "cells.parquet"
    same_decks_path = output_dir / "same_decks.parquet"
    _write_parquet(candidate_path, candidate_rows, compression=config.compression)
    _write_parquet(baseline_path, baseline_rows, compression=config.compression)
    _write_parquet(cells_path, cell_rows, compression=config.compression)
    _write_parquet(same_decks_path, same_deck_rows, compression=config.compression)

    resolved = [row for row in ordered if row["candidate_result"] != "unresolved"]
    candidate_score = sum(float(row["candidate_score"]) for row in resolved)
    terminal_counts = Counter(str(row["terminal_reason"]) for row in ordered)
    candidate_decisions = sum(int(row["candidate_policy_decisions"]) for row in ordered)
    baseline_decisions = sum(int(row["baseline_policy_decisions"]) for row in ordered)
    if execution_telemetry is not None:
        candidate_decisions = int(execution_telemetry["candidate_policy_rows"])
        baseline_decisions = int(execution_telemetry["baseline_policy_rows"])
    candidate_batches = (
        0
        if execution_telemetry is None
        else int(execution_telemetry["candidate_policy_batches"])
    )
    baseline_batches = (
        0
        if execution_telemetry is None
        else int(execution_telemetry["baseline_policy_batches"])
    )
    summary: dict[str, Any] = {
        "format": "native_checkpoint_gauntlet_summary_v1",
        "campaign_fingerprint": campaign_fingerprint,
        "candidate": _participant_summary(config, role="candidate"),
        "baseline": _participant_summary(config, role="baseline"),
        "games": len(ordered),
        "resolved_games": len(resolved),
        "unresolved_games": len(ordered) - len(resolved),
        "resumed_games": resumed_games,
        "candidate_score_rate": safe_rate(candidate_score, len(resolved)),
        "baseline_score_rate": safe_rate(
            len(resolved) - candidate_score, len(resolved)
        ),
        "candidate_wins": sum(row["candidate_result"] == "win" for row in ordered),
        "candidate_losses": sum(row["candidate_result"] == "loss" for row in ordered),
        "draws": sum(row["candidate_result"] == "draw" for row in ordered),
        "terminal_reason_counts": dict(sorted(terminal_counts.items())),
        "cross_roster_cells": len(cell_rows),
        "same_exact_deck_cells": len(same_deck_rows),
        "candidate_decks": len(candidate_rows),
        "baseline_decks": len(baseline_rows),
        "candidate_deck_results": candidate_rows,
        "baseline_deck_results": baseline_rows,
        "same_deck_results": same_deck_rows,
        "elapsed_seconds": elapsed_seconds,
        "session_games_per_second": safe_rate(
            len(ordered) - resumed_games, elapsed_seconds
        ),
        "candidate_effective_policy_batch_size": safe_rate(
            (
                candidate_decisions
                if execution_telemetry is not None
                else sum(int(row["candidate_policy_batch_size_sum"]) for row in ordered)
            ),
            candidate_batches
            if execution_telemetry is not None
            else candidate_decisions,
        ),
        "baseline_effective_policy_batch_size": safe_rate(
            (
                baseline_decisions
                if execution_telemetry is not None
                else sum(int(row["baseline_policy_batch_size_sum"]) for row in ordered)
            ),
            baseline_batches if execution_telemetry is not None else baseline_decisions,
        ),
        "candidate_policy_decisions": candidate_decisions,
        "baseline_policy_decisions": baseline_decisions,
        "candidate_mean_action_seconds": safe_rate(
            (
                float(execution_telemetry["candidate_policy_seconds"])
                if execution_telemetry is not None
                else sum(float(row["candidate_action_seconds"]) for row in ordered)
            ),
            (
                candidate_decisions
                if execution_telemetry is not None
                else sum(int(row["candidate_decisions"]) for row in ordered)
            ),
        ),
        "baseline_mean_action_seconds": safe_rate(
            (
                float(execution_telemetry["baseline_policy_seconds"])
                if execution_telemetry is not None
                else sum(float(row["baseline_action_seconds"]) for row in ordered)
            ),
            (
                baseline_decisions
                if execution_telemetry is not None
                else sum(int(row["baseline_decisions"]) for row in ordered)
            ),
        ),
        "concurrency": config.concurrency,
        "native_worker_replicas": config.native_worker_replicas,
        "native_match_total_concurrency": (
            config.concurrency * config.native_worker_replicas
            if config.backend == "native_match"
            else None
        ),
        "policy_batch_max_rows": config.policy_batch_max_rows,
        "policy_batch_wait_ms": config.policy_batch_wait_ms,
        "policy_batch_coalesce_temperatures": (
            config.policy_batch_coalesce_temperatures
        ),
        "checkpoint_resident_precision": config.checkpoint_resident_precision,
        "checkpoint_rollout_inductor": config.checkpoint_rollout_inductor,
        "runner_source_commit": config.runner_source_commit,
        "match_seed_namespace": config.match_seed_namespace,
        "games_path": records.display_path(games_path),
        "candidate_decks_path": records.display_path(candidate_path),
        "baseline_decks_path": records.display_path(baseline_path),
        "cells_path": records.display_path(cells_path),
        "same_decks_path": records.display_path(same_decks_path),
    }
    if execution_telemetry is not None:
        collection_seconds = float(execution_telemetry["collection_elapsed_seconds"])
        summary["native_collection"] = dict(execution_telemetry)
        summary["native_collection_games_per_second"] = safe_rate(
            len(ordered), collection_seconds
        )
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.md"
    summary["summary_path"] = records.display_path(summary_path)
    summary["report_path"] = records.display_path(report_path)
    write_json_atomic(summary_path, summary)
    write_text_atomic(report_path, render_report(summary))
    return summary


def _participant_summary(
    config: NativeCheckpointGauntletConfig,
    *,
    role: str,
) -> dict[str, Any]:
    participant = config.candidate if role == "candidate" else config.baseline
    summary: dict[str, Any] = {
        "label": participant.label,
        "checkpoint_path": records.display_path(participant.checkpoint_path),
        "checkpoint_sha256": participant.expected_checkpoint_sha256,
        "checkpoint_source_commit": participant.checkpoint_source_commit,
        "public_catalog_manifest_path": records.display_path(
            participant.public_catalog_manifest_path
        ),
        "public_catalog_manifest_sha256": (
            participant.expected_public_catalog_manifest_sha256
        ),
        "provenance_fingerprint": participant.provenance_fingerprint,
        "policy_temperature": participant.policy_temperature,
        "exact_roster_decks": len(participant.decks),
    }
    if participant.source_identity_path is not None:
        summary.update(
            source_identity_path=records.display_path(participant.source_identity_path),
            source_identity_sha256=participant.expected_source_identity_sha256,
        )
    if participant.resolved_config_path is not None:
        summary.update(
            resolved_config_path=records.display_path(participant.resolved_config_path),
            resolved_config_sha256=participant.expected_resolved_config_sha256,
        )
    if participant.pair_manifest_path is not None:
        summary.update(
            pair_manifest_path=records.display_path(participant.pair_manifest_path),
            pair_manifest_sha256=participant.expected_pair_manifest_sha256,
        )
    else:
        if participant.policy_evaluation_binding_path is None:
            raise AssertionError("validated evaluation binding path is missing")
        summary.update(
            policy_evaluation_binding_path=records.display_path(
                participant.policy_evaluation_binding_path
            ),
            policy_evaluation_binding_sha256=(
                participant.expected_policy_evaluation_binding_sha256
            ),
        )
    return summary


def _group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    id_key: str,
    role: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row[id_key]), []).append(row)
    output: list[dict[str, Any]] = []
    for values in grouped.values():
        first = values[0]
        resolved = [row for row in values if row["candidate_result"] != "unresolved"]
        candidate_score = sum(float(row["candidate_score"]) for row in resolved)
        role_result_key = f"{role}_result"
        result_counts = Counter(str(row[role_result_key]) for row in values)
        score = (
            candidate_score if role == "candidate" else len(resolved) - candidate_score
        )
        lower, upper = _wilson_interval(score, len(resolved))
        output.append(
            {
                "deck_id": first[f"{role}_deck_id"],
                "deck_hash": first[f"{role}_deck_hash"],
                "deck_label": first[f"{role}_deck_label"],
                "deck_signature": first[f"{role}_deck_signature"],
                "deck_source": first[f"{role}_deck_source"],
                "scheduled_games": len(values),
                "resolved_games": len(resolved),
                "wins": result_counts.get("win", 0),
                "losses": result_counts.get("loss", 0),
                "draws": result_counts.get("draw", 0),
                "unresolved": result_counts.get("unresolved", 0),
                "score_rate": safe_rate(score, len(resolved)),
                "score_ci95_low": lower,
                "score_ci95_high": upper,
                "seat0_games": sum(int(row[f"{role}_seat"]) == 0 for row in values),
                "seat1_games": sum(int(row[f"{role}_seat"]) == 1 for row in values),
            }
        )
    output.sort(key=lambda row: (float(row["score_rate"]), str(row["deck_label"])))
    for rank, row in enumerate(output, start=1):
        row["ascending_score_rank"] = rank
    return output


def _cell_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (str(row["candidate_deck_id"]), str(row["baseline_deck_id"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for values in grouped.values():
        first = values[0]
        resolved = [row for row in values if row["candidate_result"] != "unresolved"]
        score = sum(float(row["candidate_score"]) for row in resolved)
        counts = Counter(str(row["candidate_result"]) for row in values)
        lower, upper = _wilson_interval(score, len(resolved))
        output.append(
            {
                "candidate_deck_id": first["candidate_deck_id"],
                "candidate_deck_hash": first["candidate_deck_hash"],
                "candidate_deck_label": first["candidate_deck_label"],
                "baseline_deck_id": first["baseline_deck_id"],
                "baseline_deck_hash": first["baseline_deck_hash"],
                "baseline_deck_label": first["baseline_deck_label"],
                "scheduled_games": len(values),
                "resolved_games": len(resolved),
                "candidate_wins": counts.get("win", 0),
                "baseline_wins": counts.get("loss", 0),
                "draws": counts.get("draw", 0),
                "unresolved": counts.get("unresolved", 0),
                "candidate_score_rate": safe_rate(score, len(resolved)),
                "candidate_score_ci95_low": lower,
                "candidate_score_ci95_high": upper,
                "candidate_seat0_games": sum(
                    int(row["candidate_seat"]) == 0 for row in values
                ),
                "candidate_seat1_games": sum(
                    int(row["candidate_seat"]) == 1 for row in values
                ),
            }
        )
    output.sort(
        key=lambda row: (
            str(row["candidate_deck_label"]),
            str(row["baseline_deck_label"]),
        )
    )
    return output


def _wilson_interval(score: float, games: int) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 1.0
    z = 1.959963984540054
    proportion = score / games
    denominator = 1.0 + z * z / games
    center = (proportion + z * z / (2.0 * games)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / games + z * z / (4.0 * games * games)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _write_parquet(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression: str,
) -> None:
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in rows]),
        temporary,
        compression=compression,
    )
    temporary.replace(path)


def render_report(summary: Mapping[str, Any]) -> str:
    """Render a compact human-facing diagnostic report."""
    candidate = cast(Mapping[str, Any], summary["candidate"])
    baseline = cast(Mapping[str, Any], summary["baseline"])
    lines = [
        "# Native Checkpoint Gauntlet Report",
        "",
        f"Candidate `{candidate['label']}` uses its own "
        f"{candidate['exact_roster_decks']}-deck exact roster at "
        f"T={float(candidate['policy_temperature']):g}; baseline "
        f"`{baseline['label']}` uses its own "
        f"{baseline['exact_roster_decks']}-deck exact roster at "
        f"T={float(baseline['policy_temperature']):g}.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Games | {summary['games']} |",
        f"| Resolved | {summary['resolved_games']} |",
        f"| Candidate score | {float(summary['candidate_score_rate']):.2%} |",
        f"| Cross-roster cells | {summary['cross_roster_cells']} |",
        f"| Throughput | {float(summary['session_games_per_second']):.3f} games/s |",
        f"| Candidate effective batch | "
        f"{float(summary['candidate_effective_policy_batch_size']):.2f} |",
        f"| Baseline effective batch | "
        f"{float(summary['baseline_effective_policy_batch_size']):.2f} |",
        "",
        "## Candidate decks, weakest first",
        "",
        "| Deck | deck_hash | W-L-D-U | Score | 95% CI |",
        "|---|---|---:|---:|---:|",
    ]
    for raw_row in cast(Sequence[Mapping[str, Any]], summary["candidate_deck_results"]):
        lines.append(
            "| {label} | `{deck_hash}` | {wins}-{losses}-{draws}-{unresolved} | "
            "{score:.2%} | {low:.2%}–{high:.2%} |".format(
                label=raw_row["deck_label"],
                deck_hash=raw_row["deck_hash"],
                wins=raw_row["wins"],
                losses=raw_row["losses"],
                draws=raw_row["draws"],
                unresolved=raw_row["unresolved"],
                score=float(raw_row["score_rate"]),
                low=float(raw_row["score_ci95_low"]),
                high=float(raw_row["score_ci95_high"]),
            )
        )
    lines.extend(
        [
            "",
            "Each cross-roster cell is mirrored by logical seat. The native engine "
            "and policy controllers receive deterministic seeds derived from the "
            "match ID. Aligned campaigns that share a non-null match-seed namespace "
            "therefore form common-random-number pairs up to policy-induced RNG "
            "stream divergence after their actions first differ.",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["render_report", "write_final_artifacts"]
