"""Build lightweight meta-gauntlet reports from mined deck summaries."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator


class MetaGauntletConfig(BaseModel):
    """Config for constructing a meta opponent pool and matchup report."""

    model_config = ConfigDict(extra="forbid")

    deck_signature_summary_path: Path
    matchup_matrix_path: Path | None = None
    output_dir: Path = Path("outputs/belief/meta_gauntlet")
    top_n: int = 20
    min_games: int = 1
    candidate_signature: str | None = None

    @field_validator("top_n")
    @classmethod
    def valid_top_n(cls, value: int) -> int:
        """Reject non-positive pool sizes."""
        if value <= 0:
            raise ValueError("top_n must be positive")
        return value

    @field_validator("min_games")
    @classmethod
    def valid_min_games(cls, value: int) -> int:
        """Reject negative game thresholds."""
        if value < 0:
            raise ValueError("min_games must be non-negative")
        return value


@dataclass(frozen=True)
class GauntletDeck:
    """One opponent deck in the meta gauntlet."""

    rank: int
    deck_hash: str
    label: str
    games: int
    win_rate: float
    signature: str


def build_meta_gauntlet(config: MetaGauntletConfig) -> dict[str, object]:
    """Write gauntlet CSV artifacts and return a small summary."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    decks = load_gauntlet_decks(
        config.deck_signature_summary_path,
        top_n=config.top_n,
        min_games=config.min_games,
    )
    pool_path = config.output_dir / "meta_gauntlet.csv"
    _write_gauntlet_decks(pool_path, decks)

    matchup_path: Path | None = None
    matchup_count = 0
    if config.candidate_signature and config.matchup_matrix_path is not None:
        matchup_rows = candidate_matchup_rows(
            config.matchup_matrix_path,
            candidate_signature=config.candidate_signature,
            opponents=decks,
        )
        matchup_path = config.output_dir / "candidate_matchups.csv"
        _write_dict_rows(matchup_path, matchup_rows)
        matchup_count = len(matchup_rows)

    summary: dict[str, object] = {
        "deck_count": len(decks),
        "pool_path": str(pool_path),
        "matchup_count": matchup_count,
        "matchup_path": str(matchup_path) if matchup_path is not None else "",
    }
    (config.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def load_gauntlet_decks(
    deck_signature_summary_path: Path,
    *,
    top_n: int,
    min_games: int,
) -> tuple[GauntletDeck, ...]:
    """Load top supported deck signatures for offline evaluation."""
    decks: list[GauntletDeck] = []
    with deck_signature_summary_path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            games = _int_field(row, "games")
            signature = row.get("deck_signature", "")
            if games < min_games or not signature:
                continue
            decks.append(
                GauntletDeck(
                    rank=_int_field(row, "rank"),
                    deck_hash=row.get("deck_hash", ""),
                    label=row.get("deck_label", ""),
                    games=games,
                    win_rate=_float_field(row, "win_rate"),
                    signature=signature,
                )
            )
    decks.sort(key=lambda deck: (deck.games, deck.win_rate), reverse=True)
    return tuple(decks[:top_n])


def candidate_matchup_rows(
    matchup_matrix_path: Path,
    *,
    candidate_signature: str,
    opponents: tuple[GauntletDeck, ...],
) -> list[dict[str, object]]:
    """Return empirical candidate-vs-gauntlet matchup rows when observed."""
    rows_by_opponent: dict[str, dict[str, object]] = {}
    with matchup_matrix_path.open(encoding="utf-8", newline="") as file_obj:
        for row in csv.DictReader(file_obj):
            if row.get("candidate_deck_signature") != candidate_signature:
                continue
            rows_by_opponent[row.get("opponent_deck_signature", "")] = {
                "opponent_signature": row.get("opponent_deck_signature", ""),
                "games": _int_field(row, "games"),
                "wins": _int_field(row, "wins"),
                "losses": _int_field(row, "losses"),
                "draws": _int_field(row, "draws"),
                "win_rate": _float_field(row, "win_rate"),
            }
    output: list[dict[str, object]] = []
    for opponent in opponents:
        observed = rows_by_opponent.get(opponent.signature, {})
        output.append(
            {
                "opponent_rank": opponent.rank,
                "opponent_hash": opponent.deck_hash,
                "opponent_label": opponent.label,
                "opponent_signature": opponent.signature,
                "opponent_pool_games": opponent.games,
                "observed_games": observed.get("games", 0),
                "observed_wins": observed.get("wins", 0),
                "observed_losses": observed.get("losses", 0),
                "observed_draws": observed.get("draws", 0),
                "observed_win_rate": observed.get("win_rate", ""),
            }
        )
    return output


def _write_gauntlet_decks(path: Path, decks: tuple[GauntletDeck, ...]) -> None:
    rows = [
        {
            "rank": deck.rank,
            "deck_hash": deck.deck_hash,
            "deck_label": deck.label,
            "games": deck.games,
            "win_rate": deck.win_rate,
            "deck_signature": deck.signature,
        }
        for deck in decks
    ]
    _write_dict_rows(path, rows)


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _int_field(row: dict[str, str], field: str) -> int:
    value = row.get(field, "")
    return int(value) if value else 0


def _float_field(row: dict[str, str], field: str) -> float:
    value = row.get(field, "")
    return float(value) if value else 0.0
