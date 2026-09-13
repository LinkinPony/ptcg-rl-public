"""One-command, idempotent Kaggle release submission.

Example:
    PYTHONPATH=data/sample_submission:src \
      python src/tools/submission/submit_release.py --profile <profile>

The command builds and validates the package, publishes a durable dispatch
receipt, uploads at most once, polls the existing submission, and records the
Kaggle reference. Re-running the same profile is reconcile-only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ptcg_rl.submission import protocol


def main(argv: Sequence[str] | None = None) -> int:
    """Run the protected release path with explicit upload authorization."""
    parser = argparse.ArgumentParser(
        description=(
            "Build, validate, and idempotently submit one immutable release. "
            "Re-running the same profile only reconciles existing state."
        )
    )
    parser.add_argument("--profile", required=True)
    parser.add_argument("--message")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--wait-seconds", type=float)
    args = parser.parse_args(argv)

    profile = protocol.load_submission_profile(str(args.profile))
    if profile.release_asset is None:
        raise SystemExit(
            "protected release submission requires a release_asset profile"
        )

    protocol_args = [
        "--profile",
        str(args.profile),
        "--submit",
        "--allow-draft",
    ]
    if args.message is not None:
        protocol_args.extend(("--message", str(args.message)))
    if args.output is not None:
        protocol_args.extend(("--output", str(args.output)))
    if args.wait_seconds is not None:
        protocol_args.extend(("--wait-seconds", str(args.wait_seconds)))
    return protocol.main(protocol_args)


if __name__ == "__main__":
    raise SystemExit(main())
