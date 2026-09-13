"""Internal worker entry point for one distributed release H2H shard."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptcg_rl.evaluation.distributed_release_h2h import (
    ReleaseH2HShardConfig,
    run_release_h2h_shard,
)


def main() -> None:
    """Validate a frozen shard envelope and execute only its assigned games."""
    args = _parse_args()
    config = ReleaseH2HShardConfig.model_validate_json(
        args.shard_config.read_text(encoding="utf-8")
    )
    result = run_release_h2h_shard(
        config,
        status_path=args.status_path,
        result_path=args.result_path,
    )
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-config", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
