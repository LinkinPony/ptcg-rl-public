"""Analyze Kaggle top-episode deck environment and observed win rates."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, field_validator

import ptcg_rl.data.kaggle_deck.records as records
import ptcg_rl.data.kaggle_deck.reports as reports


class KaggleDeckEnvironmentConfig(BaseModel):
    """Hydra-backed config for Kaggle replay deck analysis."""

    model_config = ConfigDict(extra="forbid")

    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    dates: list[str] = Field(default_factory=list)
    output_dir: Path = Path("outputs/kaggle_deck_environment/latest")
    card_data_csv: Path = Path("data/EN_Card_Data.csv")
    public_web_rankings_csv: Path | None = Path(
        "outputs/public_deck_ev/play_limitless_por/rankings.csv",
    )
    known_deck_paths: dict[str, Path] = Field(default_factory=dict)
    known_deck_dirs: list[Path] = Field(default_factory=list)
    max_episodes: int | None = None
    min_games_for_win_rate: int = 20
    top_n_decks: int = 50
    min_public_match_score: float = 0.20
    write_side_observations: bool = True
    parser_mode: Literal["fast", "full"] = "fast"
    fast_prefix_bytes: int = 65_536
    include_step_count: bool = False

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings early."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("max_episodes")
    @classmethod
    def positive_optional_limit(cls, value: int | None) -> int | None:
        """Reject non-positive episode limits."""
        if value is not None and value <= 0:
            raise ValueError("max_episodes must be positive when set")
        return value

    @field_validator("min_games_for_win_rate", "top_n_decks")
    @classmethod
    def positive_limit(cls, value: int) -> int:
        """Reject non-positive limits."""
        if value <= 0:
            raise ValueError("limits must be positive")
        return value

    @field_validator("fast_prefix_bytes")
    @classmethod
    def positive_prefix_bytes(cls, value: int) -> int:
        """Reject non-positive fast-parser prefix sizes."""
        if value <= 0:
            raise ValueError("fast_prefix_bytes must be positive")
        return value

    @field_validator("min_public_match_score")
    @classmethod
    def valid_match_score(cls, value: float) -> float:
        """Reject invalid match-score thresholds."""
        if value < 0.0 or value > 1.0:
            raise ValueError("min_public_match_score must be in [0, 1]")
        return value


def run(config: KaggleDeckEnvironmentConfig) -> dict[str, Any]:
    """Analyze local Kaggle replay JSON files and write reusable summaries."""
    output_dir = records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    card_meta = records.load_card_meta(records.repo_path(config.card_data_csv))
    known_decks = records.load_known_decks(
        config.known_deck_paths,
        config.known_deck_dirs,
    )
    replay_paths = records.iter_replay_paths(
        records.repo_path(config.replay_root),
        set(config.dates),
    )

    side_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    by_date: dict[str, Counter[str]] = defaultdict(Counter)
    for episode_number, replay_path in enumerate(replay_paths, start=1):
        if config.max_episodes is not None and episode_number > config.max_episodes:
            counters["max_episodes_reached"] += 1
            break
        _scan_replay(
            replay_path,
            card_meta=card_meta,
            known_decks=known_decks,
            config=config,
            side_rows=side_rows,
            counters=counters,
            by_date=by_date,
        )

    deck_rows = reports.deck_summaries(
        side_rows,
        card_meta,
        min_games_for_win_rate=config.min_games_for_win_rate,
    )
    daily_rows = reports.daily_deck_summaries(
        side_rows,
        card_meta,
        min_games_for_win_rate=config.min_games_for_win_rate,
    )
    team_rows = reports.team_deck_summaries(
        side_rows,
        min_games_for_win_rate=config.min_games_for_win_rate,
    )
    matchup_rows = reports.matchup_matrix(
        side_rows,
        min_games_for_win_rate=config.min_games_for_win_rate,
    )
    card_rows = reports.card_usage(side_rows, card_meta)
    comparison_rows = reports.public_web_comparison(
        deck_rows,
        public_web_rankings_csv=config.public_web_rankings_csv,
        min_public_match_score=config.min_public_match_score,
        top_n_decks=config.top_n_decks,
    )
    reports.write_outputs(
        output_dir,
        side_rows=side_rows,
        deck_rows=deck_rows,
        daily_rows=daily_rows,
        team_rows=team_rows,
        matchup_rows=matchup_rows,
        card_rows=card_rows,
        comparison_rows=comparison_rows,
        write_side_observations=config.write_side_observations,
    )

    report = _summary_report(
        config,
        output_dir,
        counters=counters,
        by_date=by_date,
        known_deck_count=len(known_decks),
        top_decks=deck_rows[: config.top_n_decks],
    )
    reports.write_json(output_dir / "summary.json", report)
    print(json.dumps(_console_summary(output_dir, report), indent=2, sort_keys=True))
    return report


def _scan_replay(
    replay_path: Path,
    *,
    card_meta: dict[int, records.CardMeta],
    known_decks: dict[str, str],
    config: KaggleDeckEnvironmentConfig,
    side_rows: list[dict[str, Any]],
    counters: Counter[str],
    by_date: dict[str, Counter[str]],
) -> None:
    counters["episode_json_files"] += 1
    date = replay_path.parent.name
    by_date[date]["episode_json_files"] += 1
    rows = None
    if config.parser_mode == "fast":
        try:
            rows = records.fast_episode_side_rows(
                replay_path=replay_path,
                card_meta=card_meta,
                known_decks=known_decks,
                prefix_bytes=config.fast_prefix_bytes,
                include_step_count=config.include_step_count,
            )
        except (OSError, ValueError, json.JSONDecodeError):
            rows = None
        if rows is None:
            counters["fast_parse_fallbacks"] += 1
            by_date[date]["fast_parse_fallbacks"] += 1

    if rows is None:
        try:
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            counters["scan_errors"] += 1
            by_date[date]["scan_errors"] += 1
            return
        rows = records.episode_side_rows(
            replay_path=replay_path,
            replay=replay,
            card_meta=card_meta,
            known_decks=known_decks,
        )

    side_rows.extend(rows)
    counters["side_observations"] += len(rows)
    by_date[date]["side_observations"] += len(rows)
    if len(rows) < 2:
        counters["episodes_with_missing_deck_registration"] += 1
        by_date[date]["episodes_with_missing_deck_registration"] += 1


def _summary_report(
    config: KaggleDeckEnvironmentConfig,
    output_dir: Path,
    *,
    counters: Counter[str],
    by_date: dict[str, Counter[str]],
    known_deck_count: int,
    top_decks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": dict(sorted(counters.items())),
        "by_date": {
            date: dict(sorted(date_counts.items()))
            for date, date_counts in sorted(by_date.items())
        },
        "known_deck_count": known_deck_count,
        "top_decks": top_decks,
        "outputs": reports.output_paths(output_dir),
    }


def _console_summary(output_dir: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_dir": records.display_path(output_dir),
        "summary": report["summary"],
        "top_decks": report["top_decks"][:5],
    }


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_deck_environment",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = KaggleDeckEnvironmentConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    run(config)


if __name__ == "__main__":
    main()
