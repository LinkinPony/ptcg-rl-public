"""Hydra entry point for immutable public-pilot teacher preparation."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.teacher_pipeline import (
    PublicPilotTeacherPipelineConfig,
    run_public_pilot_teacher_pipeline,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/public_pilot_teacher/yushin_a3cd_20260710",
)
def main(hydra_config: DictConfig) -> None:
    """Prepare the frozen public-pilot replay and BC teacher artifacts."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = PublicPilotTeacherPipelineConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    summary = run_public_pilot_teacher_pipeline(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
