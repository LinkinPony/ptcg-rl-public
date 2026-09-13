"""Hydra entry point for RL sprint health diagnostics."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.experiment_health import (
    RLSprintHealthConfig,
    score_rl_sprint_health,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/rl_sprint1/health",
)
def main(hydra_config: DictConfig) -> None:
    """Score all completed arms without reading deployment outcomes."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = RLSprintHealthConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(score_rl_sprint_health(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
