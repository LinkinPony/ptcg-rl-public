"""Stream campaign shards into scorer-compatible exact-bundle game rows."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_collection_campaign.artifact_io import (
    file_sha256,
    read_json,
    write_json_atomic,
)
from ptcg_rl.evaluation.native_collection_campaign.models import (
    CampaignSelection,
    CampaignTask,
    NativeCollectionCampaignPlan,
    PlannedBundle,
)
from ptcg_rl.evaluation.native_collection_campaign.planner import (
    load_campaign_plan,
)


@dataclass
class _ScreeningTally:
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    unresolved: int = 0
    seat_games: list[int] = field(default_factory=lambda: [0, 0])
    opponent_cells: set[str] = field(default_factory=set)


def merge_campaign_results(
    plan_path: Path,
    *,
    output_path: Path | None = None,
    read_batch_size: int = 16_384,
    compression: str = "zstd",
    root: Path | None = None,
) -> dict[str, Any]:
    """Validate all task outputs and atomically stream one scoring Parquet."""
    if read_batch_size <= 0:
        raise ValueError("campaign merge batch size must be positive")
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    plan = load_campaign_plan(plan_path, root=repo_root)
    target = _resolve_path(
        output_path or (plan.output_dir / "games_for_deck_strength.parquet"),
        root=repo_root,
    )
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    if target.exists() or manifest_path.exists():
        return _validate_existing_merge(
            plan,
            target=target,
            manifest_path=manifest_path,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    bundles = {
        bundle.bundle_id: bundle for bundle in (*plan.candidates, *plan.opponents)
    }
    writer: pq.ParquetWriter | None = None
    output_schema: pa.Schema | None = None
    total_rows = 0
    try:
        for task in sorted(plan.tasks, key=lambda item: item.task_id):
            games_path = _complete_task_games(task, root=repo_root)
            candidate_by_deck = _task_bundle_by_deck(
                task.candidate_bundle_ids,
                bundles=bundles,
            )
            opponent_by_deck = _task_bundle_by_deck(
                task.opponent_bundle_ids,
                bundles=bundles,
            )
            task_rows = 0
            for batch in pq.ParquetFile(games_path).iter_batches(
                batch_size=read_batch_size
            ):
                raw_rows = cast(list[dict[str, Any]], batch.to_pylist())
                output_rows = []
                for raw in raw_rows:
                    source_index = int(raw["game_index"])
                    if source_index != task_rows:
                        raise ValueError(
                            f"task {task.task_id} game indexes are not contiguous"
                        )
                    output_rows.append(
                        _scorer_row(
                            raw,
                            task=task,
                            plan_fingerprint=plan.plan_fingerprint,
                            candidate_by_deck=candidate_by_deck,
                            opponent_by_deck=opponent_by_deck,
                            merged_game_index=total_rows,
                        )
                    )
                    task_rows += 1
                    total_rows += 1
                table = pa.Table.from_pylist(output_rows)
                if writer is None:
                    output_schema = table.schema
                    writer = pq.ParquetWriter(
                        temporary,
                        output_schema,
                        compression=compression,
                    )
                elif output_schema is not None and table.schema != output_schema:
                    table = table.cast(output_schema)
                writer.write_table(table)
            if task_rows != task.gauntlet.total_games:
                raise ValueError(
                    f"task {task.task_id} has {task_rows} games, expected "
                    f"{task.gauntlet.total_games}"
                )
        if writer is None:
            raise ValueError("campaign contains no result rows")
    except BaseException:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise
    writer.close()
    temporary.replace(target)
    manifest = _merge_manifest(plan, target=target, games_total=total_rows)
    write_json_atomic(manifest_path, manifest)
    return manifest


def rank_screening_bundles(
    plan_path: Path,
    *,
    games_path: Path | None = None,
    output_path: Path | None = None,
    prior_alpha: float = 1.0,
    prior_beta: float = 1.0,
    read_batch_size: int = 16_384,
    root: Path | None = None,
) -> dict[str, Any]:
    """Publish a small conjugate-posterior ranking for adaptive screening."""
    if (
        not math.isfinite(prior_alpha)
        or not math.isfinite(prior_beta)
        or prior_alpha <= 0.0
        or prior_beta <= 0.0
    ):
        raise ValueError("screening beta prior must be finite and positive")
    if read_batch_size <= 0:
        raise ValueError("screening ranking batch size must be positive")
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    plan = load_campaign_plan(plan_path, root=repo_root)
    source = _resolve_path(
        games_path or (plan.output_dir / "games_for_deck_strength.parquet"),
        root=repo_root,
    )
    if not source.is_file():
        raise FileNotFoundError(source)
    target = _resolve_path(
        output_path or (plan.output_dir / "screening_bundle_standings.parquet"),
        root=repo_root,
    )
    manifest_path = target.with_suffix(target.suffix + ".manifest.json")
    source_sha256 = file_sha256(source)
    if target.exists() or manifest_path.exists():
        manifest = read_json(manifest_path)
        if (
            not target.is_file()
            or manifest.get("source_plan_fingerprint") != plan.plan_fingerprint
            or manifest.get("games_sha256") != source_sha256
            or manifest.get("standings_sha256") != file_sha256(target)
        ):
            raise ValueError("screening ranking artifacts are incomplete or stale")
        return manifest
    candidates = {item.bundle_id: item for item in plan.candidates}
    tallies = {bundle_id: _ScreeningTally() for bundle_id in candidates}
    columns = {
        "candidate_bundle_id",
        "opponent_bundle_id",
        "candidate_result",
        "candidate_seat",
    }
    parquet = pq.ParquetFile(source)
    if not columns <= set(parquet.schema_arrow.names):
        raise ValueError("merged campaign games lack screening ranking columns")
    for batch in parquet.iter_batches(
        batch_size=read_batch_size,
        columns=sorted(columns),
    ):
        for row in cast(list[dict[str, Any]], batch.to_pylist()):
            bundle_id = str(row["candidate_bundle_id"])
            try:
                tally = tallies[bundle_id]
            except KeyError as error:
                raise ValueError(
                    "screening games contain an unknown candidate"
                ) from error
            result = str(row["candidate_result"])
            seat = int(row["candidate_seat"])
            if seat not in (0, 1):
                raise ValueError("screening game has an invalid candidate seat")
            tally.games += 1
            tally.seat_games[seat] += 1
            tally.opponent_cells.add(str(row["opponent_bundle_id"]))
            if result == "win":
                tally.wins += 1
            elif result == "draw":
                tally.draws += 1
            elif result == "loss":
                tally.losses += 1
            elif result == "unresolved":
                tally.unresolved += 1
            else:
                raise ValueError(f"unknown screening result: {result}")
    rows: list[dict[str, Any]] = []
    for bundle_id, bundle in candidates.items():
        tally = tallies[bundle_id]
        resolved = tally.wins + tally.draws + tally.losses
        if tally.games == 0 or resolved == 0 or min(tally.seat_games) == 0:
            raise ValueError(
                f"screening bundle lacks balanced resolved evidence: {bundle_id}"
            )
        alpha = prior_alpha + tally.wins + 0.5 * tally.draws
        beta = prior_beta + tally.losses + 0.5 * tally.draws
        posterior_mean = alpha / (alpha + beta)
        posterior_variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1.0))
        rows.append(
            {
                "bundle_id": bundle_id,
                "checkpoint_id": bundle.checkpoint_id,
                "model_fingerprint": bundle.model_fingerprint,
                "deck_digest": bundle.deck_digest,
                "deck_hash": bundle.deck_hash,
                "deck_label": bundle.deck_label,
                "deck_signature": bundle.deck_signature,
                "games": tally.games,
                "resolved": resolved,
                "wins": tally.wins,
                "draws": tally.draws,
                "losses": tally.losses,
                "unresolved": tally.unresolved,
                "score": (tally.wins + 0.5 * tally.draws) / resolved,
                "posterior_alpha": alpha,
                "posterior_beta": beta,
                "posterior_mean": posterior_mean,
                "posterior_sd": math.sqrt(posterior_variance),
                "conservative_score": (
                    posterior_mean - 1.96 * math.sqrt(posterior_variance)
                ),
                "opponent_cells": len(tally.opponent_cells),
                "seat0_games": tally.seat_games[0],
                "seat1_games": tally.seat_games[1],
                "within_prior_mass_reference": False,
            }
        )
    rows.sort(key=lambda row: (-float(row["conservative_score"]), row["bundle_id"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
    temporary.replace(target)
    manifest = {
        "format": "native_collection_screening_ranking_v1",
        "source_plan_fingerprint": plan.plan_fingerprint,
        "games_path": str(_display_path(source, root=repo_root)),
        "games_sha256": source_sha256,
        "standings_path": str(_display_path(target, root=repo_root)),
        "standings_sha256": file_sha256(target),
        "bundles": len(rows),
        "prior_alpha": prior_alpha,
        "prior_beta": prior_beta,
        "score_column": "conservative_score",
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def select_promoted_bundles(
    plan_path: Path,
    standings_path: Path,
    *,
    output_path: Path,
    keep_count: int,
    score_column: str = "deploy_mean",
    require_within_prior_mass: bool = True,
    maximum_per_deck: int | None = None,
    root: Path | None = None,
) -> CampaignSelection:
    """Freeze a deterministic posterior-ranking promotion for the next stage."""
    if keep_count <= 0:
        raise ValueError("promotion keep count must be positive")
    if maximum_per_deck is not None and maximum_per_deck <= 0:
        raise ValueError("maximum_per_deck must be positive")
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    plan = load_campaign_plan(plan_path, root=repo_root)
    standings = _resolve_path(standings_path, root=repo_root)
    table = pq.read_table(standings)
    required = {"bundle_id", "deck_signature", score_column}
    missing = required - set(table.column_names)
    if missing:
        raise ValueError(
            "bundle standings are missing promotion columns: "
            + ", ".join(sorted(missing))
        )
    candidate_ids = {bundle.bundle_id for bundle in plan.candidates}
    rows = cast(list[dict[str, Any]], table.to_pylist())
    eligible = [
        row
        for row in rows
        if str(row["bundle_id"]) in candidate_ids
        and (
            not require_within_prior_mass
            or bool(row.get("within_prior_mass_reference", False))
        )
    ]
    eligible.sort(key=lambda row: (-float(row[score_column]), str(row["bundle_id"])))
    selected: list[str] = []
    per_deck: dict[str, int] = {}
    for row in eligible:
        signature = str(row["deck_signature"])
        if maximum_per_deck is not None and per_deck.get(signature, 0) >= (
            maximum_per_deck
        ):
            continue
        selected.append(str(row["bundle_id"]))
        per_deck[signature] = per_deck.get(signature, 0) + 1
        if len(selected) == keep_count:
            break
    if len(selected) != keep_count:
        raise ValueError(
            f"promotion retained {len(selected)} eligible bundles, expected {keep_count}"
        )
    selection = CampaignSelection(
        source_plan_fingerprint=plan.plan_fingerprint,
        standings_sha256=file_sha256(standings),
        selected_bundle_ids=tuple(selected),
        score_column=score_column,
    )
    target = _resolve_path(output_path, root=repo_root)
    if target.exists():
        existing = CampaignSelection.model_validate(read_json(target))
        if existing != selection:
            raise ValueError("promotion path already binds a different decision")
    else:
        write_json_atomic(target, selection)
    return selection


def _scorer_row(
    raw: dict[str, Any],
    *,
    task: CampaignTask,
    plan_fingerprint: str,
    candidate_by_deck: dict[str, PlannedBundle],
    opponent_by_deck: dict[str, PlannedBundle],
    merged_game_index: int,
) -> dict[str, Any]:
    candidate_digest = str(raw["candidate_deck_id"])
    opponent_digest = str(raw["baseline_deck_id"])
    try:
        candidate = candidate_by_deck[candidate_digest]
        opponent = opponent_by_deck[opponent_digest]
    except KeyError as error:
        raise ValueError("native task returned an unplanned exact deck") from error
    if (
        str(raw.get("candidate_deck_hash", "")) != candidate.deck_hash
        or str(raw.get("baseline_deck_hash", "")) != opponent.deck_hash
    ):
        raise ValueError("native task changed an authoritative deck_hash")
    outcome = str(raw.get("outcome", raw.get("terminal_reason", "")))
    terminal_reason = "finished" if outcome == "engine_terminal" else outcome
    candidate_result = (
        str(raw.get("candidate_result", "unresolved"))
        if terminal_reason == "finished"
        else "unresolved"
    )
    error_actor = "infrastructure" if outcome == "infrastructure_error" else ""
    return {
        "format": "native_collection_campaign_scorer_game_v1",
        "source_plan_fingerprint": plan_fingerprint,
        "game_index": merged_game_index,
        "source_task_id": task.task_id,
        "source_game_index": int(raw["game_index"]),
        "source_match_id": str(raw["match_id"]),
        "candidate_bundle_id": candidate.bundle_id,
        "candidate_pilot_id": candidate.checkpoint_id,
        "candidate_model_fingerprint": candidate.model_fingerprint,
        "candidate_archetype": candidate.deck_label,
        "candidate_deck_id": candidate.deck_digest,
        "candidate_deck_hash": candidate.deck_hash,
        "candidate_deck_label": candidate.deck_label,
        "candidate_deck_signature": candidate.deck_signature,
        "candidate_variant_weight": 1.0,
        "opponent_bundle_id": opponent.bundle_id,
        "opponent_pilot_id": opponent.checkpoint_id,
        "opponent_model_fingerprint": opponent.model_fingerprint,
        "opponent_archetype": opponent.deck_label,
        "opponent_deck_id": opponent.deck_digest,
        "opponent_deck_hash": opponent.deck_hash,
        "opponent_deck_label": opponent.deck_label,
        "opponent_deck_signature": opponent.deck_signature,
        "opponent_variant_weight": 1.0,
        "candidate_seat": int(raw["candidate_seat"]),
        "candidate_result": candidate_result,
        "terminal_reason": terminal_reason,
        "error_actor": error_actor,
        "source_outcome": outcome,
        "candidate_score": _optional_float(raw.get("candidate_score")),
        "candidate_decisions": int(raw.get("candidate_decisions", 0)),
        "native_engine_steps": int(raw.get("native_engine_steps", 0) or 0),
    }


def _task_bundle_by_deck(
    bundle_ids: tuple[str, ...],
    *,
    bundles: dict[str, PlannedBundle],
) -> dict[str, PlannedBundle]:
    output = {
        bundles[bundle_id].deck_digest: bundles[bundle_id] for bundle_id in bundle_ids
    }
    if len(output) != len(bundle_ids):
        raise ValueError("campaign task repeats an exact deck route")
    return output


def _complete_task_games(task: CampaignTask, *, root: Path) -> Path:
    output_dir = _resolve_path(task.gauntlet.output_dir, root=root)
    progress = read_json(output_dir / "progress.json")
    games_path = output_dir / "games.parquet"
    if progress.get("complete") is not True or not games_path.is_file():
        raise ValueError(f"campaign task {task.task_id} is incomplete")
    return games_path


def _validate_existing_merge(
    plan: NativeCollectionCampaignPlan,
    *,
    target: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if not target.is_file():
        raise ValueError("campaign merge manifest exists without its Parquet")
    if not manifest_path.is_file():
        games_total = _validate_unmanifested_merge(plan, target=target)
        manifest = _merge_manifest(
            plan,
            target=target,
            games_total=games_total,
        )
        write_json_atomic(manifest_path, manifest)
        return manifest
    manifest = read_json(manifest_path)
    expected_total = sum(task.gauntlet.total_games for task in plan.tasks)
    if (
        manifest.get("plan_fingerprint") != plan.plan_fingerprint
        or manifest.get("games_sha256") != file_sha256(target)
        or int(manifest.get("games_total", -1)) != expected_total
        or int(manifest.get("tasks_total", -1)) != len(plan.tasks)
    ):
        raise ValueError("existing campaign merge differs from the immutable plan")
    return manifest


def _validate_unmanifested_merge(
    plan: NativeCollectionCampaignPlan,
    *,
    target: Path,
) -> int:
    """Reconcile a crash after Parquet publication but before its manifest."""
    expected = {task.task_id: task for task in plan.tasks}
    next_indexes = dict.fromkeys(expected, 0)
    total_rows = 0
    required = {
        "game_index",
        "source_plan_fingerprint",
        "source_task_id",
        "source_game_index",
        "candidate_bundle_id",
        "opponent_bundle_id",
    }
    parquet = pq.ParquetFile(target)
    if missing := required - set(parquet.schema_arrow.names):
        raise ValueError(
            "unmanifested campaign merge is missing identity columns: "
            + ", ".join(sorted(missing))
        )
    for batch in parquet.iter_batches(batch_size=16_384, columns=sorted(required)):
        for row in cast(list[dict[str, Any]], batch.to_pylist()):
            if row["source_plan_fingerprint"] != plan.plan_fingerprint:
                raise ValueError("unmanifested merge binds a different campaign plan")
            task_id = str(row["source_task_id"])
            try:
                task = expected[task_id]
            except KeyError as error:
                raise ValueError(
                    "unmanifested merge contains an unknown task"
                ) from error
            source_index = int(row["source_game_index"])
            if source_index != next_indexes[task_id]:
                raise ValueError("unmanifested merge task indexes are not contiguous")
            if int(row["game_index"]) != total_rows:
                raise ValueError("unmanifested merge game indexes are not contiguous")
            if (
                row["candidate_bundle_id"] not in task.candidate_bundle_ids
                or row["opponent_bundle_id"] not in task.opponent_bundle_ids
            ):
                raise ValueError("unmanifested merge contains an unplanned bundle")
            next_indexes[task_id] += 1
            total_rows += 1
    if any(
        next_indexes[task.task_id] != task.gauntlet.total_games for task in plan.tasks
    ):
        raise ValueError("unmanifested merge does not contain every planned game")
    return total_rows


def _merge_manifest(
    plan: NativeCollectionCampaignPlan,
    *,
    target: Path,
    games_total: int,
) -> dict[str, Any]:
    return {
        "format": "native_collection_campaign_merged_games_v1",
        "plan_fingerprint": plan.plan_fingerprint,
        "games_path": str(target),
        "games_sha256": file_sha256(target),
        "games_total": games_total,
        "tasks_total": len(plan.tasks),
        "candidate_bundles": len(plan.candidates),
        "opponent_bundles": len(plan.opponents),
    }


def _resolve_path(path: Path, *, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _display_path(path: Path, *, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError:
        return resolved


def _optional_float(value: object) -> float:
    if value is None:
        return float("nan")
    if not isinstance(value, (int, float, str)):
        raise TypeError("campaign score must be numeric or null")
    return float(value)


__all__ = [
    "merge_campaign_results",
    "rank_screening_bundles",
    "select_promoted_bundles",
]
