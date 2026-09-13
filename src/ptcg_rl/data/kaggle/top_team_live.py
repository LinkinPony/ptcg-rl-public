"""Reusable fixed-window pipeline for live Kaggle top-team deck analysis."""

from __future__ import annotations

from typing import Any

from ptcg_rl.data.kaggle.top_team_live_analysis import analyze_replays
from ptcg_rl.data.kaggle.top_team_live_models import TopTeamLiveConfig
from ptcg_rl.data.kaggle.top_team_live_sync import (
    discover_daily_replays,
    fetch_inventory,
    materialize_inventory,
)

__all__ = ["TopTeamLiveConfig", "run"]


def run(config: TopTeamLiveConfig) -> dict[str, Any]:
    """Fetch inventory, materialize replays, and report every target exact deck."""
    daily_replays = discover_daily_replays(config)
    inventory, query_counts = fetch_inventory(config)
    manifest = materialize_inventory(config, inventory, daily_replays)
    return analyze_replays(
        config,
        inventory,
        manifest,
        daily_replays,
        query_counts,
    )
