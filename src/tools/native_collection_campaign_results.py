"""Merge campaign shards or freeze a posterior-ranking promotion."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ptcg_rl.evaluation.native_collection_campaign.results import (
    merge_campaign_results,
    rank_screening_bundles,
    select_promoted_bundles,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    merge = commands.add_parser("merge")
    merge.add_argument("plan", type=Path)
    merge.add_argument("--output", type=Path)
    merge.add_argument("--read-batch-size", type=int, default=16_384)
    rank = commands.add_parser("rank")
    rank.add_argument("plan", type=Path)
    rank.add_argument("--games", type=Path)
    rank.add_argument("--output", type=Path)
    rank.add_argument("--prior-alpha", type=float, default=1.0)
    rank.add_argument("--prior-beta", type=float, default=1.0)
    rank.add_argument("--read-batch-size", type=int, default=16_384)
    promote = commands.add_parser("promote")
    promote.add_argument("plan", type=Path)
    promote.add_argument("standings", type=Path)
    promote.add_argument("--output", type=Path, required=True)
    promote.add_argument("--keep-count", type=int, required=True)
    promote.add_argument("--score-column", default="deploy_mean")
    promote.add_argument("--maximum-per-deck", type=int)
    promote.add_argument(
        "--allow-outside-prior-mass-reference",
        action="store_true",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Execute one bounded artifact transformation."""
    arguments = _parser().parse_args(argv)
    if arguments.command == "merge":
        result = merge_campaign_results(
            arguments.plan,
            output_path=arguments.output,
            read_batch_size=arguments.read_batch_size,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if arguments.command == "rank":
        result = rank_screening_bundles(
            arguments.plan,
            games_path=arguments.games,
            output_path=arguments.output,
            prior_alpha=arguments.prior_alpha,
            prior_beta=arguments.prior_beta,
            read_batch_size=arguments.read_batch_size,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    selection = select_promoted_bundles(
        arguments.plan,
        arguments.standings,
        output_path=arguments.output,
        keep_count=arguments.keep_count,
        score_column=arguments.score_column,
        require_within_prior_mass=(not arguments.allow_outside_prior_mass_reference),
        maximum_per_deck=arguments.maximum_per_deck,
    )
    print(json.dumps(selection.model_dump(mode="json"), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
