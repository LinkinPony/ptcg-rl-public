"""Inspect or explicitly execute one campaign device shard."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ptcg_rl.evaluation.native_collection_campaign.runner import (
    run_campaign_device,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--device-index", type=int, required=True)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="run pending tasks; without this flag the command is read-only",
    )
    parser.add_argument(
        "--reconcile-stale-lock",
        action="store_true",
        help="remove a same-host lock only after proving its PID is absent",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the selected operation and print its durable status snapshot."""
    arguments = _parser().parse_args(argv)
    status = run_campaign_device(
        arguments.plan,
        device_index=arguments.device_index,
        execute=arguments.execute,
        reconcile_stale_lock=arguments.reconcile_stale_lock,
    )
    print(json.dumps(status, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
