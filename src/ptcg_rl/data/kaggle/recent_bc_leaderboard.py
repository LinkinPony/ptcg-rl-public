"""Immutable leaderboard cohort gates for recent Kaggle BC data."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from ptcg_rl.data.kaggle.recent_bc_artifacts import (
    display_path,
    repo_path,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class LeaderboardCohort:
    """One fixed leaderboard snapshot and its admitted team identities."""

    teams_by_pilot: Mapping[str, Mapping[str, Any]]
    identity: Mapping[str, Any]


def load_leaderboard_cohort(
    path: Path,
    *,
    top_fraction: float,
    max_rank: int,
    min_score: float,
) -> LeaderboardCohort:
    """Admit teams satisfying percentile, absolute-rank, or score gates."""
    resolved = repo_path(path)
    rows, member = _read_rows(resolved)
    total_teams = len(rows)
    percentile_rank = max(1, math.ceil(total_teams * top_fraction))
    admitted: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        rank = int(row["Rank"])
        score = float(row["Score"])
        criteria = tuple(
            criterion
            for criterion, passes in (
                ("top_fraction", rank <= percentile_rank),
                ("top_rank", rank <= max_rank),
                ("minimum_score", score >= min_score),
            )
            if passes
        )
        if not criteria:
            continue
        team_name = str(row["TeamName"]).strip()
        pilot_key = normalized_pilot_key(team_name)
        if pilot_key in admitted:
            raise ValueError("leaderboard team names collide after normalization")
        admitted[pilot_key] = MappingProxyType(
            {
                "pilot_key": pilot_key,
                "team_id": str(row["TeamId"]).strip(),
                "team_name": team_name,
                "rank": rank,
                "score": score,
                "criteria": criteria,
            }
        )
    if not admitted:
        raise ValueError("leaderboard gates admitted no teams")
    identity = MappingProxyType(
        {
            "path": display_path(resolved),
            "sha256": sha256_file(resolved),
            "csv_member": member,
            "total_teams": total_teams,
            "top_fraction": top_fraction,
            "top_fraction_rank_cutoff": percentile_rank,
            "max_rank": max_rank,
            "min_score": min_score,
            "gate_semantics": "top_fraction_or_top_rank_or_minimum_score",
            "admitted_teams": len(admitted),
        }
    )
    return LeaderboardCohort(
        teams_by_pilot=MappingProxyType(admitted),
        identity=identity,
    )


def normalized_pilot_key(team_name: str) -> str:
    """Match the compact Daily-ingest pilot identity exactly."""
    normalized = " ".join(team_name.strip().casefold().split())
    if not normalized:
        raise ValueError("leaderboard team name is empty")
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def _read_rows(path: Path) -> tuple[list[dict[str, str]], str | None]:
    if path.suffix.casefold() == ".zip":
        with zipfile.ZipFile(path) as archive:
            members = sorted(
                name for name in archive.namelist() if name.casefold().endswith(".csv")
            )
            if len(members) != 1:
                raise ValueError("leaderboard zip must contain exactly one CSV")
            member = members[0]
            text = archive.read(member).decode("utf-8-sig")
    else:
        member = None
        text = path.read_text(encoding="utf-8-sig")
    rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    required = {"Rank", "TeamId", "TeamName", "Score"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("leaderboard snapshot is empty or missing required columns")
    ranks = [int(row["Rank"]) for row in rows]
    if len(ranks) != len(set(ranks)) or min(ranks) != 1:
        raise ValueError("leaderboard ranks are incomplete or duplicated")
    rows.sort(key=lambda row: int(row["Rank"]))
    return rows, member


__all__ = [
    "LeaderboardCohort",
    "load_leaderboard_cohort",
    "normalized_pilot_key",
]
