"""Publish one source-bound recurrent PPO-only policy checkpoint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ptcg_rl.model.recurrent_migration import (
    RecurrentPolicyMigrationConfig,
    migrate_recurrent_policy_checkpoint,
)
from ptcg_rl.rl.training import (
    RLTrainConfig,
    _load_hydra_config,
    _resolved_target_model_config,
)
from ptcg_rl.training.host_policy import require_cuda_training_host


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the target profile and execute its immutable migration."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--source-policy-sha256", required=True)
    parser.add_argument("--source-training-state", type=Path, required=True)
    parser.add_argument("--source-training-state-sha256", required=True)
    parser.add_argument("--source-pair-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    require_cuda_training_host()
    raw = _load_hydra_config(args.config_name)
    config = RLTrainConfig.model_validate(raw)
    report = migrate_recurrent_policy_checkpoint(
        _resolved_target_model_config(config),
        RecurrentPolicyMigrationConfig(
            source_policy_path=args.source_policy,
            source_policy_sha256=args.source_policy_sha256,
            source_training_state_path=args.source_training_state,
            source_training_state_sha256=args.source_training_state_sha256,
            source_pair_manifest_path=args.source_pair_manifest,
            output_dir=args.output_dir,
        ),
    )
    print(
        json.dumps(
            {
                "manifest_sha256": report["manifest_sha256"],
                "source_pair": report["source_pair"],
                "summary": report["summary"],
                "target_policy": report["target_policy"],
                "valid": report["valid"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
