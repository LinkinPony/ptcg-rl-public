"""Terminal rendering for shared RL performance statistics."""

from __future__ import annotations

import json
from typing import Literal

from ptcg_rl.dashboard.models import PerformanceTable, ScopeName, WindowName
from ptcg_rl.dashboard.repository import DashboardRepository

OutputFormat = Literal["table", "json", "markdown"]


def render_performance(
    repository: DashboardRepository,
    run: str,
    *,
    window: WindowName,
    scope: ScopeName,
    output_format: OutputFormat,
) -> str:
    """Render one performance table without duplicating aggregation logic."""
    table = repository.performance_table(run, window=window, scope=scope)
    if output_format == "json":
        return json.dumps(table.model_dump(mode="json"), indent=2, ensure_ascii=False)
    if output_format == "markdown":
        return _markdown(table)
    return _plain_table(table)


def _markdown(table: PerformanceTable) -> str:
    lines = [
        f"Run: `{table.run_id}` ({table.scope}, {table.window})",
        "",
        "| Deck | Games | W-D-L | Score | Self-play | Sentinel | Adaptive | Scripted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        table.decks,
        key=lambda item: item.slices["all"].score_rate or -1.0,
        reverse=True,
    ):
        all_stats = row.slices["all"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{row.display_name} (`{row.deck_label}`)",
                    str(all_stats.games),
                    f"{all_stats.wins}-{all_stats.draws}-{all_stats.losses}",
                    _percent(all_stats.score_rate),
                    _percent(row.slices["self_play"].score_rate),
                    _percent(row.slices["sentinel"].score_rate),
                    _percent(row.slices["adaptive_history"].score_rate),
                    _percent(row.slices["scripted"].score_rate),
                )
            )
            + " |"
        )
    return "\n".join(lines)


def _plain_table(table: PerformanceTable) -> str:
    rows = [
        (
            row.display_name,
            str(row.slices["all"].games),
            f"{row.slices['all'].wins}-{row.slices['all'].draws}-{row.slices['all'].losses}",
            _percent(row.slices["all"].score_rate),
            _percent(row.slices["self_play"].score_rate),
            _percent(row.slices["sentinel"].score_rate),
            _percent(row.slices["adaptive_history"].score_rate),
            _percent(row.slices["scripted"].score_rate),
        )
        for row in sorted(
            table.decks,
            key=lambda item: item.slices["all"].score_rate or -1.0,
            reverse=True,
        )
    ]
    headers = (
        "deck",
        "games",
        "W-D-L",
        "score",
        "self",
        "sentinel",
        "adaptive",
        "scripted",
    )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    lines = [
        f"run: {table.run_id} scope={table.scope} window={table.window}",
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(headers)),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(lines)


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{100.0 * value:.2f}%"
