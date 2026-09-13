"""Analyze deck results for current public leaderboard top teams."""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

import ptcg_rl.data.kaggle_deck.records as records
import ptcg_rl.data.kaggle_deck.reports as reports


class TopLeaderboardEnvironmentConfig(BaseModel):
    """Hydra-backed config for top-leaderboard deck analysis."""

    model_config = ConfigDict(extra="forbid")

    side_observations_parquet: Path = Path(
        "outputs/kaggle_deck_environment/latest/side_observations.parquet",
    )
    leaderboard_path: Path = Path(
        "data/external/kaggle_public_leaderboard/latest/pokemon-tcg-ai-battle.zip",
    )
    output_dir: Path = Path("outputs/kaggle_deck_environment/top_leaderboard/latest")
    card_data_csv: Path = Path("data/EN_Card_Data.csv")
    top_n: int = 100
    min_games_for_win_rate: int = 20
    write_side_observations: bool = True

    @field_validator("top_n", "min_games_for_win_rate")
    @classmethod
    def positive_limit(cls, value: int) -> int:
        """Reject non-positive limits."""
        if value <= 0:
            raise ValueError("limits must be positive")
        return value


@dataclass(frozen=True)
class LeaderboardTeam:
    """One public leaderboard team row."""

    rank: int
    team_id: str
    team_name: str
    last_submission_date: str
    score: float
    submission_count: int
    team_member_user_names: str


def run(config: TopLeaderboardEnvironmentConfig) -> dict[str, Any]:
    """Build top-leaderboard-only deck environment reports."""
    output_dir = records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    card_meta = records.load_card_meta(records.repo_path(config.card_data_csv))
    side_rows = _load_side_rows(records.repo_path(config.side_observations_parquet))
    leaderboard_rows = _load_leaderboard(records.repo_path(config.leaderboard_path))
    top_teams = leaderboard_rows[: config.top_n]
    top_team_map = {team.team_name: team for team in top_teams}
    enriched_rows = _enrich_rows(side_rows, top_team_map)
    top_side_rows = [row for row in enriched_rows if row["is_top_team"]]
    top_vs_top_rows = [row for row in top_side_rows if row["opponent_is_top_team"]]

    leaderboard_output_rows = [_leaderboard_output(team) for team in top_teams]
    reports.write_csv(
        output_dir / "leaderboard_top.csv",
        leaderboard_output_rows,
        _leaderboard_fields(),
    )
    _write_environment_outputs(
        output_dir,
        prefix="top_side",
        side_rows=top_side_rows,
        card_meta=card_meta,
        top_team_map=top_team_map,
        min_games_for_win_rate=config.min_games_for_win_rate,
        write_side_observations=config.write_side_observations,
    )
    _write_environment_outputs(
        output_dir,
        prefix="top_vs_top",
        side_rows=top_vs_top_rows,
        card_meta=card_meta,
        top_team_map=top_team_map,
        min_games_for_win_rate=config.min_games_for_win_rate,
        write_side_observations=config.write_side_observations,
    )

    report = _summary_report(
        config,
        output_dir,
        leaderboard_rows=leaderboard_rows,
        top_teams=top_teams,
        all_side_rows=side_rows,
        top_side_rows=top_side_rows,
        top_vs_top_rows=top_vs_top_rows,
        top_side_decks=reports.deck_summaries(
            top_side_rows,
            card_meta,
            min_games_for_win_rate=config.min_games_for_win_rate,
        ),
        top_vs_top_decks=reports.deck_summaries(
            top_vs_top_rows,
            card_meta,
            min_games_for_win_rate=config.min_games_for_win_rate,
        ),
    )
    reports.write_json(output_dir / "summary.json", report)
    print(json.dumps(_console_summary(output_dir, report), indent=2, sort_keys=True))
    return report


def _load_side_rows(path: Path) -> list[dict[str, Any]]:
    """Read side observations from Parquet."""
    import pyarrow.parquet as pq

    if not path.exists():
        raise FileNotFoundError(f"side observations parquet not found: {path}")
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def _load_leaderboard(path: Path) -> list[LeaderboardTeam]:
    """Read a Kaggle public leaderboard CSV or downloaded zip."""
    if not path.exists():
        raise FileNotFoundError(f"leaderboard file not found: {path}")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zip_file:
            csv_names = [
                name for name in zip_file.namelist() if name.lower().endswith(".csv")
            ]
            if not csv_names:
                raise ValueError(f"leaderboard zip has no CSV: {path}")
            with zip_file.open(csv_names[0]) as file_obj:
                data = file_obj.read().decode("utf-8-sig")
    else:
        data = path.read_text(encoding="utf-8-sig")
    rows = [_leaderboard_team(row) for row in csv.DictReader(io.StringIO(data))]
    rows.sort(key=lambda row: row.rank)
    return rows


def _leaderboard_team(row: dict[str, str]) -> LeaderboardTeam:
    return LeaderboardTeam(
        rank=_int_value(row.get("Rank")),
        team_id=row.get("TeamId", ""),
        team_name=row.get("TeamName", ""),
        last_submission_date=row.get("LastSubmissionDate", ""),
        score=_float_value(row.get("Score")),
        submission_count=_int_value(row.get("SubmissionCount")),
        team_member_user_names=row.get("TeamMemberUserNames", ""),
    )


def _enrich_rows(
    side_rows: list[dict[str, Any]],
    top_team_map: dict[str, LeaderboardTeam],
) -> list[dict[str, Any]]:
    by_side = {(row["episode_id"], row["player_index"]): row for row in side_rows}
    enriched_rows: list[dict[str, Any]] = []
    for row in side_rows:
        enriched = dict(row)
        opponent = by_side.get((row["episode_id"], row["opponent_player_index"]))
        opponent_team_name = opponent["team_name"] if opponent else ""
        team = top_team_map.get(str(row["team_name"]))
        opponent_team = top_team_map.get(str(opponent_team_name))
        enriched.update(
            {
                "is_top_team": team is not None,
                "leaderboard_rank": team.rank if team else None,
                "leaderboard_score": team.score if team else None,
                "leaderboard_last_submission_date": (
                    team.last_submission_date if team else None
                ),
                "leaderboard_team_id": team.team_id if team else None,
                "leaderboard_submission_count": (
                    team.submission_count if team else None
                ),
                "opponent_team_name": opponent_team_name,
                "opponent_is_top_team": opponent_team is not None,
                "opponent_leaderboard_rank": (
                    opponent_team.rank if opponent_team else None
                ),
                "opponent_leaderboard_score": (
                    opponent_team.score if opponent_team else None
                ),
            }
        )
        enriched_rows.append(enriched)
    return enriched_rows


def _write_environment_outputs(
    output_dir: Path,
    *,
    prefix: str,
    side_rows: list[dict[str, Any]],
    card_meta: dict[int, records.CardMeta],
    top_team_map: dict[str, LeaderboardTeam],
    min_games_for_win_rate: int,
    write_side_observations: bool,
) -> None:
    deck_rows = reports.deck_summaries(
        side_rows,
        card_meta,
        min_games_for_win_rate=min_games_for_win_rate,
    )
    daily_rows = reports.daily_deck_summaries(
        side_rows,
        card_meta,
        min_games_for_win_rate=min_games_for_win_rate,
    )
    team_rows = _with_team_leaderboard_fields(
        reports.team_deck_summaries(
            side_rows,
            min_games_for_win_rate=min_games_for_win_rate,
        ),
        top_team_map,
    )
    matchup_rows = reports.matchup_matrix(
        side_rows,
        min_games_for_win_rate=min_games_for_win_rate,
    )
    reports.write_csv(
        output_dir / f"{prefix}_deck_signature_summary.csv",
        deck_rows,
        reports.deck_fields(),
    )
    reports.write_csv(
        output_dir / f"{prefix}_daily_deck_summary.csv",
        daily_rows,
        reports.daily_fields(),
    )
    reports.write_csv(
        output_dir / f"{prefix}_team_deck_summary.csv",
        team_rows,
        _team_leaderboard_fields(),
    )
    reports.write_csv(
        output_dir / f"{prefix}_matchup_matrix.csv",
        matchup_rows,
        reports.matchup_fields(),
    )
    if write_side_observations:
        reports.write_parquet(
            output_dir / f"{prefix}_side_observations.parquet", side_rows
        )


def _with_team_leaderboard_fields(
    team_rows: list[dict[str, Any]],
    top_team_map: dict[str, LeaderboardTeam],
) -> list[dict[str, Any]]:
    for row in team_rows:
        team = top_team_map.get(str(row["team_name"]))
        row["leaderboard_rank"] = team.rank if team else None
        row["leaderboard_score"] = team.score if team else None
        row["leaderboard_last_submission_date"] = (
            team.last_submission_date if team else None
        )
        row["leaderboard_team_id"] = team.team_id if team else None
    team_rows.sort(
        key=lambda row: (
            _empty_rank_last(row["leaderboard_rank"]),
            -int(row["games"]),
            -float(row["win_rate"]),
        )
    )
    return team_rows


def _summary_report(
    config: TopLeaderboardEnvironmentConfig,
    output_dir: Path,
    *,
    leaderboard_rows: list[LeaderboardTeam],
    top_teams: list[LeaderboardTeam],
    all_side_rows: list[dict[str, Any]],
    top_side_rows: list[dict[str, Any]],
    top_vs_top_rows: list[dict[str, Any]],
    top_side_decks: list[dict[str, Any]],
    top_vs_top_decks: list[dict[str, Any]],
) -> dict[str, Any]:
    observed_top_team_names = {str(row["team_name"]) for row in top_side_rows}
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": {
            "all_side_observations": len(all_side_rows),
            "leaderboard_rows": len(leaderboard_rows),
            "top_n": len(top_teams),
            "observed_top_teams": len(observed_top_team_names),
            "unobserved_top_teams": len(top_teams) - len(observed_top_team_names),
            "top_side_observations": len(top_side_rows),
            "top_side_episodes": _unique_episode_count(top_side_rows),
            "top_vs_top_side_observations": len(top_vs_top_rows),
            "top_vs_top_games": _unique_episode_count(top_vs_top_rows),
        },
        "unobserved_top_teams": [
            _leaderboard_output(team)
            for team in top_teams
            if team.team_name not in observed_top_team_names
        ],
        "top_side_top_decks": top_side_decks[:10],
        "top_vs_top_top_decks": top_vs_top_decks[:10],
        "outputs": _output_paths(output_dir),
    }


def _console_summary(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": records.display_path(output_dir),
        "summary": report["summary"],
        "top_side_top_decks": report["top_side_top_decks"][:5],
        "top_vs_top_top_decks": report["top_vs_top_top_decks"][:5],
    }


def _unique_episode_count(rows: list[dict[str, Any]]) -> int:
    return len({row["episode_id"] for row in rows})


def _output_paths(output_dir: Path) -> dict[str, str]:
    prefixes = ("top_side", "top_vs_top")
    paths = {
        "leaderboard_top": records.display_path(output_dir / "leaderboard_top.csv")
    }
    for prefix in prefixes:
        paths[f"{prefix}_deck_signature_summary"] = records.display_path(
            output_dir / f"{prefix}_deck_signature_summary.csv",
        )
        paths[f"{prefix}_daily_deck_summary"] = records.display_path(
            output_dir / f"{prefix}_daily_deck_summary.csv",
        )
        paths[f"{prefix}_team_deck_summary"] = records.display_path(
            output_dir / f"{prefix}_team_deck_summary.csv",
        )
        paths[f"{prefix}_matchup_matrix"] = records.display_path(
            output_dir / f"{prefix}_matchup_matrix.csv",
        )
        paths[f"{prefix}_side_observations"] = records.display_path(
            output_dir / f"{prefix}_side_observations.parquet",
        )
    return paths


def _leaderboard_output(team: LeaderboardTeam) -> dict[str, Any]:
    return {
        "rank": team.rank,
        "team_id": team.team_id,
        "team_name": team.team_name,
        "last_submission_date": team.last_submission_date,
        "score": team.score,
        "submission_count": team.submission_count,
        "team_member_user_names": team.team_member_user_names,
    }


def _leaderboard_fields() -> list[str]:
    return [
        "rank",
        "team_id",
        "team_name",
        "last_submission_date",
        "score",
        "submission_count",
        "team_member_user_names",
    ]


def _team_leaderboard_fields() -> list[str]:
    return [
        "leaderboard_rank",
        "leaderboard_score",
        "leaderboard_last_submission_date",
        "leaderboard_team_id",
        *reports.team_fields(),
    ]


def _empty_rank_last(value: Any) -> int:
    if value in (None, ""):
        return 1_000_000
    return int(value)


def _int_value(value: str | None) -> int:
    if value is None or value == "":
        return 0
    return int(value)


def _float_value(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_top_leaderboard_environment",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = TopLeaderboardEnvironmentConfig.model_validate(
        cast(dict[str, Any], raw_config),
    )
    run(config)


if __name__ == "__main__":
    main()
