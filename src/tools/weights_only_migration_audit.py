"""Write a raw-tensor and zero-output weights-only migration manifest."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ptcg_rl.model.weights_only_migration_audit import (
    WeightsOnlyMigrationAuditConfig,
    run_weights_only_migration_audit,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Audit one immutable checkpoint and publish its compact JSON manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initialization-seed", type=int, default=0)
    args = parser.parse_args(argv)
    config = WeightsOnlyMigrationAuditConfig(
        checkpoint_path=args.checkpoint,
        expected_checkpoint_sha256=args.expected_sha256,
        output_path=args.output,
        initialization_seed=args.initialization_seed,
    )
    manifest = run_weights_only_migration_audit(config)
    print(
        json.dumps(
            {
                "checkpoint_sha256": manifest["checkpoint"]["sha256"],
                "output": str(config.output_path),
                "summary": manifest["summary"],
                "valid": manifest["valid"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
