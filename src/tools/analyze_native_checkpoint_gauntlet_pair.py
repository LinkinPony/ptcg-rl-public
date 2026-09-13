"""CLI for paired common-random-number native gauntlet analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ptcg_rl.evaluation.native_checkpoint_gauntlet.paired_analysis import (
    analyze_paired_gauntlets,
    write_analysis_json,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "control", type=Path, help="control games.parquet or artifact dir"
    )
    parser.add_argument(
        "treatment", type=Path, help="treatment games.parquet or artifact dir"
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--control-temperature",
        type=float,
        default=None,
        help="expected candidate temperature; omitted means T=0",
    )
    parser.add_argument(
        "--treatment-temperature",
        type=float,
        default=None,
        help="expected candidate temperature; omitted means T=1",
    )
    parser.add_argument(
        "--baseline-temperature",
        type=float,
        default=0.0,
        help="expected fixed baseline temperature; defaults to T=0",
    )
    parser.add_argument(
        "--expected-decks",
        type=_positive_int,
        required=True,
        help="required baseline exact-deck count",
    )
    parser.add_argument(
        "--expected-repeats",
        type=_positive_int,
        required=True,
        help="required repeats per baseline-deck/seat stratum",
    )
    parser.add_argument(
        "--maximum-incomplete-pairs",
        type=_nonnegative_int,
        default=0,
        help=(
            "explicit cap for verified max-steps pairs; defaults to zero and "
            "therefore fails closed"
        ),
    )
    parser.add_argument("--bootstrap-replicates", type=_positive_int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=_nonnegative_int, default=20_260_813)
    return parser


def main() -> None:
    """Validate, analyze, publish JSON, and print a concise result pointer."""
    parser = _parser()
    args = parser.parse_args()
    try:
        result = analyze_paired_gauntlets(
            args.control,
            args.treatment,
            control_temperature=args.control_temperature,
            treatment_temperature=args.treatment_temperature,
            baseline_temperature=args.baseline_temperature,
            expected_decks=args.expected_decks,
            expected_repeats=args.expected_repeats,
            maximum_incomplete_pairs=args.maximum_incomplete_pairs,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )
        write_analysis_json(args.output_json, result)
    except (OSError, ValueError) as error:
        parser.exit(2, f"paired analysis refused: {error}\n")
    main_effect = _mapping(result["main_exact_deck_macro"])
    sandwich = _mapping(main_effect["stratified_sandwich"])
    print(
        json.dumps(
            {
                "output_json": str(args.output_json.resolve()),
                "paired_difference": main_effect["paired_difference"],
                "ci95": [sandwich["ci95_low"], sandwich["ci95_high"]],
                "p_value": sandwich["two_sided_p_value"],
            },
            sort_keys=True,
        )
    )


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("analysis result mapping is invalid")
    return value


if __name__ == "__main__":
    main()
