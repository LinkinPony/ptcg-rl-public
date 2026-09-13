"""Artifact writing and Markdown reporting for deck-strength evaluation."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.deck_strength_models import DeckStrengthConfig


def write_deck_strength_outputs(
    config: DeckStrengthConfig,
    *,
    cell_rows: Sequence[Mapping[str, Any]],
    meta_rows: Sequence[Mapping[str, Any]],
    raw_bundle_rows: Sequence[Mapping[str, Any]],
    raw_pilot_rows: Sequence[Mapping[str, Any]],
    raw_deck_rows: Sequence[Mapping[str, Any]],
    bundle_rows: Sequence[Mapping[str, Any]],
    deck_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Atomically write compact tables and human-readable summaries."""
    output_dir = records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (
        ("cells.parquet", cell_rows),
        ("meta.parquet", meta_rows),
        ("raw_bundle_standings.parquet", raw_bundle_rows),
        ("raw_pilot_standings.parquet", raw_pilot_rows),
        ("raw_deck_standings.parquet", raw_deck_rows),
        ("bundle_standings.parquet", bundle_rows),
        ("deck_standings.parquet", deck_rows),
    ):
        _write_parquet_atomic(
            output_dir / name,
            rows,
            compression=config.compression,
        )
    _write_parquet_atomic(
        output_dir / "contrasts.parquet",
        contrast_rows,
        compression=config.compression,
        schema=_contrast_schema(),
    )
    _write_text_atomic(
        output_dir / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(
        output_dir / "report.md",
        render_deck_strength_report(
            summary,
            raw_bundle_rows,
            raw_pilot_rows,
            raw_deck_rows,
            bundle_rows,
            deck_rows,
            contrast_rows,
        ),
    )


def render_deck_strength_report(
    summary: Mapping[str, Any],
    raw_bundle_rows: Sequence[Mapping[str, Any]],
    raw_pilot_rows: Sequence[Mapping[str, Any]],
    raw_deck_rows: Sequence[Mapping[str, Any]],
    bundle_rows: Sequence[Mapping[str, Any]],
    deck_rows: Sequence[Mapping[str, Any]],
    contrast_rows: Sequence[Mapping[str, Any]],
) -> str:
    """Render evidence warnings plus deployment and portability rankings."""
    quality = cast(Mapping[str, Any], summary["quality_checks"])
    meta = cast(Mapping[str, Any], summary["meta"])
    warnings = cast(Sequence[str], summary["quality_warnings"])
    lines = [
        "# Deck strength evaluation",
        "",
        "Decision role: **single direct comparison**",
        "",
        "Evidence diagnostics:",
        "",
        f"- effective meta coverage: {float(quality['effective_meta_coverage']):.1%} "
        f"(minimum {float(quality['min_effective_meta_coverage']):.1%})",
        f"- unresolved games: {float(quality['unresolved_fraction']):.1%} "
        f"(maximum {float(quality['max_unresolved_fraction']):.1%})",
        "- maximum candidate prior-only meta mass: "
        f"{float(quality['max_candidate_prior_only_meta_mass']):.1%} "
        f"(maximum {float(quality['max_prior_only_meta_mass']):.1%})",
        "- exact signatures with implicit multi-pilot weights: "
        f"{int(quality['implicit_multi_pilot_signature_count'])}",
        "- exact signatures with conflicting bundle archetype labels: "
        f"{int(meta['opponent_signature_archetype_conflict_count'])} "
        "(canonicalized from public meta)",
        f"- warnings: {', '.join(warnings) if warnings else 'none'}",
        "",
        "## Raw common-opponent results",
        "",
        "This ranking uses no target-meta weights and no Bayesian prior. Every "
        "observed opponent bundle has equal weight, unresolved games are omitted "
        "from their cell denominator, and actual same-deck games are retained.",
        "",
        "| rank | deck_hash | deck | pilot | equal-opponent score | pooled score | W-D-L | unresolved |",
        "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in raw_bundle_rows[:25]:
        label = str(row["deck_label"]).replace("|", "/")
        pilot = str(row["pilot_id"]).replace("|", "/")
        rank = row["raw_rank"] if row["raw_rank"] is not None else "n/a"
        equal_score = row["raw_equal_opponent_score"]
        pooled_score = row["raw_pooled_score"]
        lines.append(
            f"| {rank} | {row['deck_hash']} | {label} | {pilot} "
            f"| {_format_optional_score(equal_score)} "
            f"| {_format_optional_score(pooled_score)} "
            f"| {int(row['wins'])}-{int(row['draws'])}-{int(row['losses'])} "
            f"| {int(row['unresolved_games'])} |"
        )
    lines.extend(
        [
            "",
            "### Raw pilots across every available candidate deck",
            "",
            "No common-roster filter is applied. Candidate deck count is shown "
            "so changing historical roster coverage remains visible.",
            "",
            "| rank | pilot | candidate decks | equal-bundle score | pooled score | W-D-L | unresolved |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in raw_pilot_rows:
        lines.append(
            f"| {row['raw_rank']} | {row['pilot_id']} "
            f"| {int(row['candidate_decks'])} "
            f"| {_format_optional_score(row['raw_equal_bundle_score'])} "
            f"| {_format_optional_score(row['raw_pooled_score'])} "
            f"| {int(row['wins'])}-{int(row['draws'])}-{int(row['losses'])} "
            f"| {int(row['unresolved_games'])} |"
        )
    lines.extend(
        [
            "",
            "### Raw exact decks across every available pilot",
            "",
            "No checkpoint-range filter is applied. Pilot coverage is shown "
            "rather than using it to discard games.",
            "",
            "| rank | deck_hash | pilots | equal-bundle score | pooled score | W-D-L | unresolved |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in raw_deck_rows:
        lines.append(
            f"| {row['raw_rank']} | {row['deck_hash']} | {int(row['pilots'])} "
            f"| {_format_optional_score(row['raw_equal_bundle_score'])} "
            f"| {_format_optional_score(row['raw_pooled_score'])} "
            f"| {int(row['wins'])}-{int(row['draws'])}-{int(row['losses'])} "
            f"| {int(row['unresolved_games'])} |"
        )
    lines.extend(
        [
            "",
            "## Deployment bundles",
            "",
            "| rank | bundle | deck | pilot | deploy | 95% CI | P(best) | P(top-k) | regret | robust |",
            "| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in bundle_rows[:25]:
        bundle = str(row["bundle_id"]).replace("|", "/")
        label = str(row["deck_label"]).replace("|", "/")
        pilot = str(row["pilot_id"]).replace("|", "/")
        lines.append(
            f"| {int(row['deploy_rank'])} | {bundle} | {label} | {pilot} "
            f"| {float(row['deploy_mean']):.1%} "
            f"| {float(row['deploy_credible_low']):.1%}–{float(row['deploy_credible_high']):.1%} "
            f"| {float(row['probability_best']):.1%} "
            f"| {float(row['probability_top_k']):.1%} "
            f"| {float(row['expected_regret']):.1%} "
            f"| {float(row['robust_score']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Portable deck view",
            "",
            "Portable score is the equal-pilot mean deploy score for one exact deck. "
            "Competence gap is the best pilot score minus that portable score. A "
            "one-pilot row is conditional and is not a portable estimate.",
            "",
            "| rank | deck | pilots | portable | robust | gap | prior-only |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in deck_rows[:25]:
        label = str(row["deck_label"]).replace("|", "/")
        lines.append(
            f"| {int(row['portable_rank'])} | {label} | {int(row['pilot_count'])} "
            f"| {float(row['portable_score']):.1%} "
            f"| {float(row['portable_robust_score']):.1%} "
            f"| {float(row['competence_gap']):.1%} "
            f"| {float(row['max_prior_only_meta_mass']):.1%} |"
        )
    lines.extend(
        [
            "",
            "## Bundle contrasts",
            "",
            "Differences use shared posterior meta draws; positive values favor the candidate.",
            "",
            "| candidate | reference | delta | 95% CI | P(delta>0) | P(delta>-2pp) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in contrast_rows[:50]:
        candidate = str(row["candidate_id"]).replace("|", "/")
        reference = str(row["reference_id"]).replace("|", "/")
        lines.append(
            f"| {candidate} | {reference} | {float(row['mean_delta']):+.1%} "
            f"| {float(row['credible_low']):+.1%}–{float(row['credible_high']):+.1%} "
            f"| {float(row['probability_above_zero']):.1%} "
            f"| {float(row['probability_noninferior_2pp']):.1%} |"
        )
    return "\n".join(lines) + "\n"


def _format_optional_score(value: Any) -> str:
    return f"{float(value):.1%}" if value is not None else "n/a"


def _write_parquet_atomic(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    compression: str,
    schema: pa.Schema | None = None,
) -> None:
    if not rows and schema is None:
        raise ValueError(f"cannot write empty standings artifact: {path}")
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in rows], schema=schema),
        temporary,
        compression=compression,
    )
    temporary.replace(path)


def _contrast_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("candidate_id", pa.string(), nullable=False),
            pa.field("reference_id", pa.string(), nullable=False),
            pa.field("mean_delta", pa.float64(), nullable=False),
            pa.field("standard_deviation", pa.float64(), nullable=False),
            pa.field("credible_low", pa.float64(), nullable=False),
            pa.field("credible_high", pa.float64(), nullable=False),
            pa.field("probability_above_zero", pa.float64(), nullable=False),
            pa.field("probability_noninferior_2pp", pa.float64(), nullable=False),
        ]
    )


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


__all__ = ["render_deck_strength_report", "write_deck_strength_outputs"]
