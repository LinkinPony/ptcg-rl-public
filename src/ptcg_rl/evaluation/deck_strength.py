"""Post-process bundle gauntlets into decision-oriented deck standings.

The workflow keeps the target metagame, observed game matrix, and posterior
decision scores separate.  Raw games are streamed from Parquet, adjudicated by
``BundleResultAccumulator``, and scored on one shared opponent denominator.
"""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.deck_strength_models import (
    DeckStrengthConfig,
    MetaAllocation,
)
from ptcg_rl.evaluation.deck_strength_report import write_deck_strength_outputs
from ptcg_rl.evaluation.meta_distribution import (
    MetaDistribution,
    build_meta_distribution,
)
from ptcg_rl.evaluation.posterior import (
    BundlePosteriorSummary,
    evaluate_bundle_posteriors,
)
from ptcg_rl.evaluation.result_matrix import (
    AdjudicatedMatrix,
    BundleMetadata,
    BundleResultAccumulator,
)

_REQUIRED_GAME_COLUMNS = frozenset(
    {
        "candidate_bundle_id",
        "candidate_pilot_id",
        "candidate_archetype",
        "candidate_deck_signature",
        "opponent_bundle_id",
        "opponent_pilot_id",
        "opponent_archetype",
        "opponent_deck_signature",
        "candidate_seat",
        "candidate_result",
        "terminal_reason",
        "error_actor",
    }
)


def score_deck_strength(config: DeckStrengthConfig) -> dict[str, Any]:
    """Score one or more exact bundle-gauntlet game shards and write artifacts."""
    matrix = _load_game_matrix(config)
    raw_bundle_rows = _raw_bundle_standing_rows(matrix)
    raw_pilot_rows = _raw_pilot_standing_rows(raw_bundle_rows)
    raw_deck_rows = _raw_deck_standing_rows(raw_bundle_rows)
    archetype_conflicts = _signature_archetype_conflicts(matrix.opponents)
    tracked_signatures = {
        metadata.deck_signature for metadata in matrix.opponents.values()
    }
    meta = build_meta_distribution(
        records.repo_path(config.side_observations_path),
        config=config.meta,
        tracked_signatures=tracked_signatures,
    )
    allocation = _allocate_bundle_meta(
        matrix.opponents,
        meta=meta,
        pilot_weights=config.pilot_weights,
    )
    reserved_unknown = config.posterior.unknown_opponent_id
    if reserved_unknown in matrix.opponents:
        raise ValueError(
            f"opponent bundle id {reserved_unknown!r} is reserved for unknown meta"
        )

    candidate_ids = tuple(matrix.candidates)
    deploy_support = {
        bundle_id: weight
        for bundle_id, weight in allocation.weights.items()
        if weight > 0.0
    }
    deploy_outcomes = tuple(
        outcome for outcome in matrix.outcomes if outcome.opponent_id in deploy_support
    )
    deploy = evaluate_bundle_posteriors(
        candidate_ids=candidate_ids,
        outcomes=deploy_outcomes,
        meta_weights=deploy_support,
        unknown_mass=meta.other_mass,
        config=config.posterior,
        self_opponents=_self_opponents(
            matrix.candidates,
            matrix.opponents,
            support=set(deploy_support),
        ),
    )

    robust_weights = {
        opponent_id: 1.0 / len(matrix.opponents) for opponent_id in matrix.opponents
    }
    robust_config = config.posterior.model_copy(
        update={
            "meta_concentration": config.robust_meta_concentration,
            "seed": config.posterior.seed + 104_729,
        }
    )
    robust = evaluate_bundle_posteriors(
        candidate_ids=candidate_ids,
        outcomes=matrix.outcomes,
        meta_weights=robust_weights,
        unknown_mass=0.0,
        config=robust_config,
        self_opponents=_self_opponents(
            matrix.candidates,
            matrix.opponents,
            support=set(matrix.opponents),
        ),
    )

    bundle_rows = _bundle_standing_rows(
        matrix,
        deploy.candidates,
        robust.candidates,
        max_prior_only_meta_mass=config.max_prior_only_meta_mass,
    )
    deck_rows = _deck_standing_rows(bundle_rows)
    contrast_rows = [contrast.model_dump() for contrast in deploy.contrasts]
    cell_rows = _cell_output_rows(
        matrix,
        meta_weights=allocation.weights,
        robust_weights=robust_weights,
    )
    summary = _summary(
        config,
        matrix=matrix,
        meta=meta,
        allocation=allocation,
        raw_bundle_rows=raw_bundle_rows,
        raw_pilot_rows=raw_pilot_rows,
        raw_deck_rows=raw_deck_rows,
        bundle_rows=bundle_rows,
        deck_rows=deck_rows,
        opponent_signature_archetype_conflicts=archetype_conflicts,
    )
    write_deck_strength_outputs(
        config,
        cell_rows=cell_rows,
        meta_rows=allocation.rows,
        raw_bundle_rows=raw_bundle_rows,
        raw_pilot_rows=raw_pilot_rows,
        raw_deck_rows=raw_deck_rows,
        bundle_rows=bundle_rows,
        deck_rows=deck_rows,
        contrast_rows=contrast_rows,
        summary=summary,
    )
    return summary


def _raw_bundle_standing_rows(
    matrix: AdjudicatedMatrix,
) -> list[dict[str, Any]]:
    """Rank bundles using only observed results on the common opponent panel.

    Every opponent bundle contributes equally. Unresolved games are excluded
    from that cell's denominator, while observed same-deck and self-bundle
    games remain evidence rather than being replaced with a neutral score.
    """
    cells_by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for cell in matrix.cell_rows:
        cells_by_candidate[str(cell["candidate_id"])].append(cell)

    rows: list[dict[str, Any]] = []
    for candidate_id, metadata in matrix.candidates.items():
        cells = cells_by_candidate[candidate_id]
        wins = sum(int(cell["wins"]) for cell in cells)
        draws = sum(int(cell["draws"]) for cell in cells)
        losses = sum(int(cell["losses"]) for cell in cells)
        unresolved = sum(int(cell["unresolved"]) for cell in cells)
        resolved = wins + draws + losses
        cell_scores = [
            (int(cell["wins"]) + 0.5 * int(cell["draws"])) / int(cell["resolved_games"])
            for cell in cells
            if int(cell["resolved_games"]) > 0
        ]
        same_deck_cells = [
            cell
            for cell in cells
            if matrix.opponents[str(cell["opponent_id"])].deck_signature
            == metadata.deck_signature
        ]
        same_deck_resolved = sum(
            int(cell["resolved_games"]) for cell in same_deck_cells
        )
        same_deck_score = (
            sum(
                int(cell["wins"]) + 0.5 * int(cell["draws"]) for cell in same_deck_cells
            )
            / same_deck_resolved
            if same_deck_resolved > 0
            else None
        )
        rows.append(
            {
                **metadata.model_dump(),
                "opponent_cells": len(cells),
                "resolved_opponent_cells": len(cell_scores),
                "unresolved_only_opponent_cells": len(cells) - len(cell_scores),
                "scheduled_games": resolved + unresolved,
                "resolved_games": resolved,
                "unresolved_games": unresolved,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "raw_equal_opponent_score": (
                    statistics.fmean(cell_scores) if cell_scores else None
                ),
                "raw_pooled_score": (
                    (wins + 0.5 * draws) / resolved if resolved > 0 else None
                ),
                "same_deck_resolved_games": same_deck_resolved,
                "same_deck_score": same_deck_score,
                "seat_0_games": sum(int(cell["seat_0_games"]) for cell in cells),
                "seat_1_games": sum(int(cell["seat_1_games"]) for cell in cells),
                "seat_0_resolved": sum(int(cell["seat_0_resolved"]) for cell in cells),
                "seat_1_resolved": sum(int(cell["seat_1_resolved"]) for cell in cells),
            }
        )

    rows.sort(key=_raw_bundle_sort_key)
    previous_score: float | None = None
    previous_rank: int | None = None
    for position, row in enumerate(rows, start=1):
        score = cast(float | None, row["raw_equal_opponent_score"])
        if score is None:
            row["raw_rank"] = None
            continue
        if previous_score is None or score != previous_score:
            previous_rank = position
            previous_score = score
        row["raw_rank"] = previous_rank
    return rows


def _raw_bundle_sort_key(row: Mapping[str, Any]) -> tuple[float, float, str]:
    panel_score = cast(float | None, row["raw_equal_opponent_score"])
    pooled_score = cast(float | None, row["raw_pooled_score"])
    return (
        -(panel_score if panel_score is not None else -math.inf),
        -(pooled_score if pooled_score is not None else -math.inf),
        str(row["bundle_id"]),
    )


def _raw_pilot_standing_rows(
    raw_bundle_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate every available candidate bundle for each evaluated pilot."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_bundle_rows:
        grouped[str(row["pilot_id"])].append(row)
    output = []
    for pilot_id, rows in grouped.items():
        output.append(
            {
                "pilot_id": pilot_id,
                "candidate_decks": len({str(row["deck_signature"]) for row in rows}),
                **_raw_aggregate_values(rows),
            }
        )
    _rank_raw_aggregate_rows(output)
    return output


def _raw_deck_standing_rows(
    raw_bundle_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate an exact deck across every pilot for which it was evaluated."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in raw_bundle_rows:
        grouped[str(row["deck_signature"])].append(row)
    output = []
    for deck_signature, rows in grouped.items():
        deck_hashes = {str(row["deck_hash"]) for row in rows}
        if len(deck_hashes) != 1:
            raise ValueError(
                "one exact candidate deck signature resolved to multiple deck_hash "
                f"values: {sorted(deck_hashes)}"
            )
        output.append(
            {
                "deck_signature": deck_signature,
                "deck_hash": next(iter(deck_hashes)),
                "deck_label": str(rows[0]["deck_label"]),
                "pilots": len({str(row["pilot_id"]) for row in rows}),
                **_raw_aggregate_values(rows),
            }
        )
    _rank_raw_aggregate_rows(output)
    return output


def _raw_aggregate_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["raw_equal_opponent_score"])
        for row in rows
        if row["raw_equal_opponent_score"] is not None
    ]
    wins = sum(int(row["wins"]) for row in rows)
    draws = sum(int(row["draws"]) for row in rows)
    losses = sum(int(row["losses"]) for row in rows)
    unresolved = sum(int(row["unresolved_games"]) for row in rows)
    resolved = wins + draws + losses
    return {
        "candidate_bundles": len(rows),
        "scored_candidate_bundles": len(scores),
        "unresolved_only_candidate_bundles": len(rows) - len(scores),
        "opponent_cells": sum(int(row["opponent_cells"]) for row in rows),
        "scheduled_games": resolved + unresolved,
        "resolved_games": resolved,
        "unresolved_games": unresolved,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "raw_equal_bundle_score": statistics.fmean(scores) if scores else None,
        "raw_pooled_score": ((wins + 0.5 * draws) / resolved if resolved > 0 else None),
    }


def _rank_raw_aggregate_rows(rows: list[dict[str, Any]]) -> None:
    rows.sort(
        key=lambda row: (
            -cast(float, row["raw_equal_bundle_score"])
            if row["raw_equal_bundle_score"] is not None
            else math.inf,
            -cast(float, row["raw_pooled_score"])
            if row["raw_pooled_score"] is not None
            else math.inf,
            str(row.get("pilot_id", row.get("deck_signature", ""))),
        )
    )
    previous_score: float | None = None
    previous_rank: int | None = None
    for position, row in enumerate(rows, start=1):
        score = cast(float | None, row["raw_equal_bundle_score"])
        if score is None:
            row["raw_rank"] = None
            continue
        if previous_score is None or score != previous_score:
            previous_score = score
            previous_rank = position
        row["raw_rank"] = previous_rank


def _load_game_matrix(config: DeckStrengthConfig) -> AdjudicatedMatrix:
    accumulator = BundleResultAccumulator(
        require_balanced_seats=config.require_balanced_seats
    )
    for configured_path in config.games_paths:
        path = records.repo_path(configured_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"bundle game shard not found: {path}")
        parquet_file = pq.ParquetFile(path)
        missing = _REQUIRED_GAME_COLUMNS - set(parquet_file.schema_arrow.names)
        if missing:
            raise ValueError(
                f"bundle game shard {path} is missing columns: "
                + ", ".join(sorted(missing))
            )
        for batch in parquet_file.iter_batches(batch_size=config.read_batch_size):
            rows = cast(list[dict[str, Any]], batch.to_pylist())
            for row in rows:
                accumulator.observe(row)
    return accumulator.finish()


def _signature_archetype_conflicts(
    opponents: Mapping[str, BundleMetadata],
) -> dict[str, tuple[str, ...]]:
    labels: dict[str, set[str]] = defaultdict(set)
    for metadata in opponents.values():
        labels[metadata.deck_signature].add(metadata.archetype)
    return {
        signature: tuple(sorted(archetypes))
        for signature, archetypes in sorted(labels.items())
        if len(archetypes) > 1
    }


def _allocate_bundle_meta(
    opponents: Mapping[str, BundleMetadata],
    *,
    meta: MetaDistribution,
    pilot_weights: Mapping[str, float],
) -> MetaAllocation:
    by_signature: dict[str, list[BundleMetadata]] = defaultdict(list)
    for metadata in opponents.values():
        by_signature[metadata.deck_signature].append(metadata)
    exact_weights = meta.exact_variant_weights()
    archetype_weights = {
        archetype.archetype: archetype.meta_weight for archetype in meta.archetypes
    }
    signature_archetypes = _meta_signature_archetypes(meta)
    weights: dict[str, float] = {}
    rows: list[Mapping[str, Any]] = []
    default_signatures: list[str] = []
    implicit_multi_pilot_signatures: list[str] = []
    for signature, bundles in sorted(by_signature.items()):
        bundles.sort(key=lambda item: item.bundle_id)
        explicit_count = sum(metadata.pilot_id in pilot_weights for metadata in bundles)
        if explicit_count == 0:
            default_signatures.append(signature)
            split_mode = "default_unit_pilot_weights"
        elif explicit_count == len(bundles):
            split_mode = "configured_pilot_weights"
        else:
            split_mode = "mixed_configured_and_unit_default"
        if len(bundles) > 1 and explicit_count != len(bundles):
            implicit_multi_pilot_signatures.append(signature)
        numerators: list[float] = []
        for metadata in bundles:
            controller_weight = float(metadata.variant_weight)
            if not math.isfinite(controller_weight) or controller_weight < 0.0:
                raise ValueError(
                    "opponent controller allocation weights must be finite and "
                    f"non-negative: {metadata.bundle_id!r}"
                )
            pilot_weight = pilot_weights.get(metadata.pilot_id, 1.0)
            numerators.append(pilot_weight * controller_weight)
        denominator = sum(numerators)
        if denominator <= 0.0:
            raise ValueError(
                "opponent allocation has zero total weight for exact signature "
                f"{signature!r}"
            )
        signature_mass = exact_weights.get(signature, 0.0)
        canonical_archetype = signature_archetypes.get(
            signature,
            min(metadata.archetype for metadata in bundles),
        )
        for metadata, numerator in zip(bundles, numerators, strict=True):
            share = numerator / denominator
            bundle_mass = signature_mass * share
            weights[metadata.bundle_id] = bundle_mass
            rows.append(
                {
                    "opponent_bundle_id": metadata.bundle_id,
                    "opponent_pilot_id": metadata.pilot_id,
                    "archetype": canonical_archetype,
                    "deck_signature": signature,
                    "deck_hash": metadata.deck_hash,
                    "deck_label": metadata.deck_label,
                    "archetype_meta_weight": archetype_weights.get(
                        canonical_archetype, 0.0
                    ),
                    "signature_meta_weight": signature_mass,
                    "configured_pilot_weight": pilot_weights.get(
                        metadata.pilot_id, 1.0
                    ),
                    "pilot_weight_explicit": metadata.pilot_id in pilot_weights,
                    "controller_allocation_weight": metadata.variant_weight,
                    "bundle_share_within_signature": share,
                    "bundle_meta_weight": bundle_mass,
                    "pilot_split_mode": split_mode,
                    "is_unknown_tail": False,
                    "row_kind": "covered_bundle",
                }
            )
    other_archetype = next(
        archetype
        for archetype in meta.archetypes
        if archetype.archetype == meta.other_name
    )
    for variant in other_archetype.variants:
        rows.append(
            {
                "opponent_bundle_id": "",
                "opponent_pilot_id": "",
                "archetype": meta.other_name,
                "deck_signature": variant.signature,
                "deck_hash": "",
                "deck_label": "uncovered exact variant",
                "archetype_meta_weight": meta.other_mass,
                "signature_meta_weight": variant.meta_weight,
                "configured_pilot_weight": 0.0,
                "pilot_weight_explicit": False,
                "controller_allocation_weight": 0.0,
                "bundle_share_within_signature": 0.0,
                "bundle_meta_weight": variant.meta_weight,
                "pilot_split_mode": "uncovered_exact_variant",
                "is_unknown_tail": True,
                "row_kind": "uncovered_variant",
                "raw_observations": variant.raw_observations,
                "effective_observations": variant.effective_observations,
                "conditional_unknown_weight": variant.conditional_weight,
            }
        )
    known_mass = sum(weights.values())
    if not math.isclose(known_mass + meta.other_mass, 1.0, abs_tol=1e-6):
        raise ValueError(
            "allocated known bundle mass and unknown tail do not sum to one"
        )
    rows.sort(
        key=lambda row: (
            -float(row["bundle_meta_weight"]),
            str(row["opponent_bundle_id"]),
        )
    )
    return MetaAllocation(
        weights=dict(sorted(weights.items())),
        rows=tuple(rows),
        default_pilot_signatures=tuple(sorted(default_signatures)),
        implicit_multi_pilot_signatures=tuple(sorted(implicit_multi_pilot_signatures)),
    )


def _meta_signature_archetypes(meta: MetaDistribution) -> dict[str, str]:
    """Choose the dominant public-meta label for every exact signature."""
    candidates: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for archetype in meta.archetypes:
        if archetype.archetype == meta.other_name:
            continue
        for variant in archetype.variants:
            candidates[variant.signature].append(
                (variant.meta_weight, archetype.archetype)
            )
    return {
        signature: max(values, key=lambda item: (item[0], item[1]))[1]
        for signature, values in candidates.items()
    }


def _self_opponents(
    candidates: Mapping[str, BundleMetadata],
    opponents: Mapping[str, BundleMetadata],
    *,
    support: set[str],
) -> dict[str, str]:
    output: dict[str, str] = {}
    for candidate_id, candidate in candidates.items():
        matches = [
            opponent_id
            for opponent_id, opponent in opponents.items()
            if opponent_id in support
            and opponent.pilot_id == candidate.pilot_id
            and opponent.deck_signature == candidate.deck_signature
        ]
        if candidate_id in matches:
            output[candidate_id] = candidate_id
        elif len(matches) == 1:
            output[candidate_id] = matches[0]
    return output


def _bundle_standing_rows(
    matrix: AdjudicatedMatrix,
    deploy: Sequence[BundlePosteriorSummary],
    robust: Sequence[BundlePosteriorSummary],
    *,
    max_prior_only_meta_mass: float,
) -> list[dict[str, Any]]:
    robust_by_id = {summary.candidate_id: summary for summary in robust}
    robust_order = sorted(
        robust,
        key=lambda summary: (-summary.matchup_cvar_mean, summary.candidate_id),
    )
    robust_ranks = {
        summary.candidate_id: rank for rank, summary in enumerate(robust_order, start=1)
    }
    rows: list[dict[str, Any]] = []
    for summary in deploy:
        metadata = matrix.candidates[summary.candidate_id]
        robust_summary = robust_by_id[summary.candidate_id]
        values = summary.model_dump()
        values.pop("candidate_id")
        values["deploy_rank"] = values.pop("rank")
        rows.append(
            {
                **metadata.model_dump(),
                **values,
                "robust_rank": robust_ranks[summary.candidate_id],
                "robust_score": robust_summary.matchup_cvar_mean,
                "robust_credible_low": (robust_summary.matchup_cvar_credible_low),
                "robust_credible_high": (robust_summary.matchup_cvar_credible_high),
                "robust_prior_only_matchups": robust_summary.prior_only_matchups,
                "within_prior_mass_reference": (
                    summary.prior_only_meta_mass <= max_prior_only_meta_mass
                ),
            }
        )
    return rows


def _deck_standing_rows(
    bundle_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in bundle_rows:
        grouped[str(row["deck_signature"])].append(row)
    output: list[dict[str, Any]] = []
    for signature, rows in grouped.items():
        pilot_scores: dict[str, list[float]] = defaultdict(list)
        pilot_robust: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            pilot = str(row["pilot_id"])
            pilot_scores[pilot].append(float(row["deploy_mean"]))
            pilot_robust[pilot].append(float(row["robust_score"]))
        deploy_by_pilot = {
            pilot: statistics.fmean(scores) for pilot, scores in pilot_scores.items()
        }
        robust_by_pilot = {
            pilot: statistics.fmean(scores) for pilot, scores in pilot_robust.items()
        }
        portable_score = statistics.fmean(deploy_by_pilot.values())
        portable_robust = statistics.fmean(robust_by_pilot.values())
        best = max(
            rows,
            key=lambda row: (float(row["deploy_mean"]), str(row["bundle_id"])),
        )
        pilot_values = tuple(deploy_by_pilot.values())
        output.append(
            {
                "deck_signature": signature,
                "deck_hash": str(rows[0]["deck_hash"]),
                "deck_label": str(rows[0]["deck_label"]),
                "archetype": str(rows[0]["archetype"]),
                "bundle_count": len(rows),
                "pilot_count": len(deploy_by_pilot),
                "portable_identified": len(deploy_by_pilot) >= 2,
                "portable_score": portable_score,
                "portable_robust_score": portable_robust,
                "best_bundle_id": str(best["bundle_id"]),
                "best_pilot_id": str(best["pilot_id"]),
                "best_bundle_deploy_mean": float(best["deploy_mean"]),
                "competence_gap": max(pilot_values) - portable_score,
                "pilot_score_range": max(pilot_values) - min(pilot_values),
                "pilot_score_standard_deviation": (
                    statistics.pstdev(pilot_values) if len(pilot_values) >= 2 else 0.0
                ),
                "games": sum(int(row["games"]) for row in rows),
                "max_prior_only_meta_mass": max(
                    float(row["prior_only_meta_mass"]) for row in rows
                ),
                "within_prior_mass_reference": all(
                    bool(row["within_prior_mass_reference"]) for row in rows
                ),
            }
        )
    output.sort(
        key=lambda row: (
            -float(row["portable_score"]),
            -float(row["portable_robust_score"]),
            str(row["deck_signature"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["portable_rank"] = rank
    return output


def _cell_output_rows(
    matrix: AdjudicatedMatrix,
    *,
    meta_weights: Mapping[str, float],
    robust_weights: Mapping[str, float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for raw_row in matrix.cell_rows:
        row = dict(raw_row)
        candidate_id = str(row["candidate_id"])
        opponent_id = str(row["opponent_id"])
        games = int(row["resolved_games"])
        score = (
            (int(row["wins"]) + 0.5 * int(row["draws"])) / games if games > 0 else None
        )
        output.append(
            {
                **row,
                "candidate_pilot_id": matrix.candidates[candidate_id].pilot_id,
                "candidate_deck_signature": (
                    matrix.candidates[candidate_id].deck_signature
                ),
                "opponent_pilot_id": matrix.opponents[opponent_id].pilot_id,
                "opponent_deck_signature": (
                    matrix.opponents[opponent_id].deck_signature
                ),
                "empirical_score": score,
                "opponent_meta_weight": meta_weights.get(opponent_id, 0.0),
                "robust_support_weight": robust_weights[opponent_id],
            }
        )
    return output


def _summary(
    config: DeckStrengthConfig,
    *,
    matrix: AdjudicatedMatrix,
    meta: MetaDistribution,
    allocation: MetaAllocation,
    raw_bundle_rows: Sequence[Mapping[str, Any]],
    raw_pilot_rows: Sequence[Mapping[str, Any]],
    raw_deck_rows: Sequence[Mapping[str, Any]],
    bundle_rows: Sequence[Mapping[str, Any]],
    deck_rows: Sequence[Mapping[str, Any]],
    opponent_signature_archetype_conflicts: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    unresolved_fraction = matrix.unresolved_games / matrix.total_games
    max_prior_mass = max(float(row["prior_only_meta_mass"]) for row in bundle_rows)
    checks = {
        "effective_meta_coverage": (
            meta.diagnostics.effective_coverage >= config.min_effective_meta_coverage
        ),
        "unresolved_fraction": (unresolved_fraction <= config.max_unresolved_fraction),
        "prior_only_meta_mass": (max_prior_mass <= config.max_prior_only_meta_mass),
        "explicit_multi_pilot_weights": (
            not config.require_explicit_multi_pilot_weights
            or not allocation.implicit_multi_pilot_signatures
        ),
    }
    warnings = [name for name, observed in checks.items() if not observed]
    portable_unidentified = sum(
        not bool(row["portable_identified"]) for row in deck_rows
    )
    return {
        "method": "hierarchical_bundle_posterior_v1",
        "decision_role": "single_direct_comparison",
        "quality_warnings": warnings,
        "quality_checks": {
            "effective_meta_coverage": meta.diagnostics.effective_coverage,
            "min_effective_meta_coverage": config.min_effective_meta_coverage,
            "unresolved_fraction": unresolved_fraction,
            "max_unresolved_fraction": config.max_unresolved_fraction,
            "max_candidate_prior_only_meta_mass": max_prior_mass,
            "max_prior_only_meta_mass": config.max_prior_only_meta_mass,
            "implicit_multi_pilot_signature_count": len(
                allocation.implicit_multi_pilot_signatures
            ),
            "require_explicit_multi_pilot_weights": (
                config.require_explicit_multi_pilot_weights
            ),
        },
        "inputs": {
            "games_paths": [
                records.display_path(records.repo_path(path).resolve())
                for path in config.games_paths
            ],
            "side_observations_path": records.display_path(
                records.repo_path(config.side_observations_path).resolve()
            ),
        },
        "output_dir": records.display_path(records.repo_path(config.output_dir)),
        "artifacts": {
            "cells": records.display_path(
                records.repo_path(config.output_dir) / "cells.parquet"
            ),
            "meta": records.display_path(
                records.repo_path(config.output_dir) / "meta.parquet"
            ),
            "bundle_standings": records.display_path(
                records.repo_path(config.output_dir) / "bundle_standings.parquet"
            ),
            "raw_bundle_standings": records.display_path(
                records.repo_path(config.output_dir) / "raw_bundle_standings.parquet"
            ),
            "raw_pilot_standings": records.display_path(
                records.repo_path(config.output_dir) / "raw_pilot_standings.parquet"
            ),
            "raw_deck_standings": records.display_path(
                records.repo_path(config.output_dir) / "raw_deck_standings.parquet"
            ),
            "deck_standings": records.display_path(
                records.repo_path(config.output_dir) / "deck_standings.parquet"
            ),
            "contrasts": records.display_path(
                records.repo_path(config.output_dir) / "contrasts.parquet"
            ),
            "report": records.display_path(
                records.repo_path(config.output_dir) / "report.md"
            ),
            "summary": records.display_path(
                records.repo_path(config.output_dir) / "summary.json"
            ),
        },
        "games": {
            "total": matrix.total_games,
            "resolved": matrix.resolved_games,
            "unresolved": matrix.unresolved_games,
            "candidate_error_losses": matrix.candidate_error_losses,
            "opponent_error_wins": matrix.opponent_error_wins,
            "infrastructure_errors": matrix.infrastructure_errors,
            "cells": len(matrix.cell_rows),
        },
        "meta": {
            "reference_date": (
                meta.reference_date.isoformat() if meta.reference_date else None
            ),
            "known_mass": 1.0 - meta.other_mass,
            "unknown_mass": meta.other_mass,
            "total_effective_observations": meta.total_effective_observations,
            "diagnostics": meta.diagnostics.model_dump(mode="json"),
            "default_unit_pilot_weight_signature_count": len(
                allocation.default_pilot_signatures
            ),
            "default_unit_pilot_weight_signatures": list(
                allocation.default_pilot_signatures
            ),
            "implicit_multi_pilot_weight_signatures": list(
                allocation.implicit_multi_pilot_signatures
            ),
            "opponent_signature_archetype_conflict_count": len(
                opponent_signature_archetype_conflicts
            ),
            "opponent_signature_archetype_conflicts": {
                signature: list(labels)
                for signature, labels in opponent_signature_archetype_conflicts.items()
            },
            "pilot_weight_note": (
                "pilot_weights and per-bundle controller_allocation_weight only "
                "split an exact signature's mass; they never change that "
                "signature's target meta mass"
            ),
        },
        "standings": {
            "candidate_bundles": len(bundle_rows),
            "candidate_decks": len(deck_rows),
            "opponent_bundles": len(matrix.opponents),
            "portable_unidentified_decks": portable_unidentified,
            "top_raw_bundle": dict(raw_bundle_rows[0]),
            "top_raw_pilot": dict(raw_pilot_rows[0]),
            "top_raw_deck": dict(raw_deck_rows[0]),
            "top_bundle": dict(bundle_rows[0]),
            "top_portable_deck": dict(deck_rows[0]),
        },
        "config": config.model_dump(mode="json"),
    }


__all__ = ["DeckStrengthConfig", "score_deck_strength"]
