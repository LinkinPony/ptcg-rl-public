"""Reusable paired analysis for completed native checkpoint gauntlets."""

from __future__ import annotations

import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_statistics import (
    build_statistics,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_validation import (
    load_validated_pair,
)

FORMAT = "native_checkpoint_gauntlet_paired_analysis_v2"


def analyze_paired_gauntlets(
    control_path: Path,
    treatment_path: Path,
    *,
    control_temperature: float | None = None,
    treatment_temperature: float | None = None,
    baseline_temperature: float = 0.0,
    expected_decks: int | None = None,
    expected_repeats: int | None = None,
    maximum_incomplete_pairs: int = 0,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 20_260_813,
) -> dict[str, Any]:
    """Validate two artifacts and return their paired temperature contrast."""
    pair = load_validated_pair(
        control_path,
        treatment_path,
        control_temperature=control_temperature,
        treatment_temperature=treatment_temperature,
        baseline_temperature=baseline_temperature,
        expected_decks=expected_decks,
        expected_repeats=expected_repeats,
        maximum_incomplete_pairs=maximum_incomplete_pairs,
    )
    statistics = build_statistics(
        pair,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "format": FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "control_games_path": str(pair.control_path),
            "treatment_games_path": str(pair.treatment_path),
        },
        "provenance": pair.provenance,
        "validation": pair.validation,
        **statistics,
    }


def write_analysis_json(path: Path, result: dict[str, Any]) -> None:
    """Atomically write standards-compliant JSON to an explicit output path."""
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


__all__ = ["FORMAT", "analyze_paired_gauntlets", "write_analysis_json"]
