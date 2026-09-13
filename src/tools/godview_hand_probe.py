"""Probe replay-side opponent-hand recovery for godview belief labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from glob import glob
from pathlib import Path
from typing import Any

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps.records import extract_replay_rows


def main() -> None:
    """Run the probe from a simple CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "replay_glob",
        help="Glob for Kaggle replay JSON files.",
    )
    parser.add_argument("--max-replays", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=1 << 20)
    args = parser.parse_args()
    print(
        json.dumps(
            run_probe(
                args.replay_glob,
                max_replays=args.max_replays,
                chunk_size=args.chunk_size,
            ),
            indent=2,
            sort_keys=True,
        )
    )


def run_probe(
    replay_glob: str,
    *,
    max_replays: int,
    chunk_size: int,
) -> dict[str, Any]:
    """Return recovery counters for sampled replay rows."""
    counters: Counter[str] = Counter()
    replay_pattern = Path(replay_glob)
    resolved_glob = (
        replay_glob
        if replay_pattern.is_absolute()
        else str(deck_records.repo_path(replay_pattern))
    )
    replay_paths = sorted(Path(path) for path in glob(resolved_glob, recursive=True))
    for replay_path in replay_paths[:max_replays]:
        counters["replays"] += 1
        rows, extraction_counters = extract_replay_rows(
            replay_path,
            drop_forced_actions=False,
            chunk_size=chunk_size,
        )
        counters.update({f"extract_{key}": value for key, value in extraction_counters.items()})
        for row in rows:
            counters["rows"] += 1
            if bool(row.get("godview_opp_hand_available", False)):
                counters["hand_available_rows"] += 1
            else:
                counters["hand_missing_rows"] += 1
            if bool(row.get("godview_opp_hand_count_match", False)):
                counters["hand_count_match_rows"] += 1
            elif bool(row.get("godview_opp_hand_available", False)):
                counters["hand_count_mismatch_rows"] += 1
    row_count = counters["rows"]
    available = counters["hand_available_rows"]
    return {
        "counters": dict(counters),
        "hand_available_rate": _rate(available, row_count),
        "hand_count_match_rate": _rate(counters["hand_count_match_rows"], available),
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    main()
