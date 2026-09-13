"""Hydra entry point for cumulative exact-resume health recovery."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.resume_health import (
    RLResumeHealthConfig,
    recover_resume_health_summary,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/rl_sprint1/arm_b_resume_health",
)
def main(hydra_config: DictConfig) -> None:
    """Recover and publish one cumulative health-only run summary."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = RLResumeHealthConfig.model_validate(cast(dict[str, Any], raw))
    print(
        json.dumps(
            recover_resume_health_summary(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
