"""Publish a source-bound dense-v3 to DCCR-v4 checkpoint pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ptcg_rl.rl.compositional_transition_pair import (
    CompositionalPairTransitionConfig,
    publish_compositional_transition_pair,
)
from ptcg_rl.rl.training import (
    RLTrainConfig,
    _load_hydra_config,
    _resolved_target_model_config,
)
from ptcg_rl.training.host_policy import require_cuda_training_host


def main() -> int:
    """Resolve one target profile and execute its immutable topology transition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--source-policy", type=Path, required=True)
    parser.add_argument("--source-policy-sha256", required=True)
    parser.add_argument("--source-training-state", type=Path, required=True)
    parser.add_argument("--source-training-state-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--factorization-device", default="cuda")
    parser.add_argument("--factorization-seed", type=int)
    parser.add_argument("--svd-oversampling", type=int, default=8)
    parser.add_argument("--svd-power-iterations", type=int, default=2)
    args = parser.parse_args()

    require_cuda_training_host()
    raw = _load_hydra_config(args.config_name)
    config = RLTrainConfig.model_validate(raw)
    config = config.model_copy(update={"model": _resolved_target_model_config(config)})
    _pair, receipt = publish_compositional_transition_pair(
        config,
        CompositionalPairTransitionConfig(
            source_policy_path=args.source_policy,
            source_policy_sha256=args.source_policy_sha256,
            source_training_state_path=args.source_training_state,
            source_training_state_sha256=args.source_training_state_sha256,
            output_dir=args.output_dir,
            factorization_device=args.factorization_device,
            factorization_seed=(
                config.seed
                if args.factorization_seed is None
                else args.factorization_seed
            ),
            svd_oversampling=args.svd_oversampling,
            svd_power_iterations=args.svd_power_iterations,
        ),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
