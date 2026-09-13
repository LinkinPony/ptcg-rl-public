"""Build a deck signature summary CSV (belief prior) from deck files.

The output matches the ``deck_signature_summary.csv`` schema consumed by
``ArchetypePrior.from_deck_signature_summary`` (belief tokens and probe
sampler priors): ``deck_label,deck_signature,games,win_rate``.

Decks are deduplicated by signature; the first label wins.

Run with:
    PYTHONPATH=src python src/tools/build_deck_signature_summary.py \
      --output docs/experiments/<pool>/deck_signature_summary.csv \
      docs/experiments/<pool_a>/decks/*.csv docs/experiments/<pool_b>/*/deck.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ptcg_rl.data.kaggle_deck import records


def build_summary_rows(
    deck_paths: list[Path],
    *,
    games: int,
    win_rate: float,
) -> list[dict[str, object]]:
    """Return signature-deduplicated summary rows for the given deck files."""
    rows: list[dict[str, object]] = []
    seen_signatures: set[str] = set()
    for deck_path in deck_paths:
        resolved = records.repo_path(deck_path)
        deck = records.read_deck(resolved)
        signature = records.deck_signature(list(deck))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        rows.append(
            {
                "deck_label": _deck_label(resolved),
                "deck_signature": signature,
                "games": games,
                "win_rate": win_rate,
            }
        )
    return rows


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    """Write summary rows as a small CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=("deck_label", "deck_signature", "games", "win_rate"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _deck_label(path: Path) -> str:
    if path.stem != "deck":
        return path.stem
    return path.parent.name


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deck_paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--win-rate", type=float, default=0.5)
    args = parser.parse_args()

    rows = build_summary_rows(
        list(args.deck_paths),
        games=args.games,
        win_rate=args.win_rate,
    )
    write_summary(records.repo_path(args.output), rows)
    print(f"wrote {len(rows)} unique deck signatures to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
