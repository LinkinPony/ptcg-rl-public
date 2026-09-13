"""Hydra entry point for public-teacher engine sibling evaluation."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.teacher_sibling import run_teacher_sibling_evaluation
from ptcg_rl.evaluation.teacher_sibling_config import TeacherSiblingConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/teacher_sibling/base",
)
def main(hydra_config: DictConfig) -> None:
    """Run the frozen engine sibling campaign and print its summary."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = TeacherSiblingConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(run_teacher_sibling_evaluation(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
