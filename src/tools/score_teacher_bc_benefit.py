"""Hydra entry point for held-out public-teacher BC selection."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.teacher_bc_selection import (
    TeacherBCSelectionConfig,
    score_teacher_bc_benefit,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/teacher_bc/selection",
)
def main(hydra_config: DictConfig) -> None:
    """Score all frozen teacher BC arms against initial held-out metrics."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = TeacherBCSelectionConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(score_teacher_bc_benefit(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
