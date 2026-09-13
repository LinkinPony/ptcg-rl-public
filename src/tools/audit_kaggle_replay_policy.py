"""Hydra entry point for matched-state Kaggle pilot policy auditing."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.kaggle_replay_policy_audit import (
    KaggleReplayPolicyAuditConfig,
    run,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/kaggle_replay_policy_audit",
)
def main(hydra_config: DictConfig) -> None:
    """Run one immutable replay-to-checkpoint audit."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = KaggleReplayPolicyAuditConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    summary = run(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
