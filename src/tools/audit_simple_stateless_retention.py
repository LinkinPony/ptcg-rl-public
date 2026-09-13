"""Hydra entry point for full-model routed retention auditing."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.simple_stateless_retention_audit import (
    SimpleStatelessRetentionAuditConfig,
    run,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/simple_stateless_retention_audit",
)
def main(hydra_config: DictConfig) -> None:
    """Run one immutable routed retention comparison."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = SimpleStatelessRetentionAuditConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    summary = run(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
