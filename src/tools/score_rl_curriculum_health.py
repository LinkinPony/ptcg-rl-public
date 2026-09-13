"""Hydra entry point for specialist curriculum sampling diagnostics."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.curriculum_experiment import (
    RLCurriculumHealthConfig,
    score_rl_curriculum_health,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/rl_curriculum/stage_a_health",
)
def main(hydra_config: DictConfig) -> None:
    """Score configured and observed curriculum distributions."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = RLCurriculumHealthConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(score_rl_curriculum_health(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
