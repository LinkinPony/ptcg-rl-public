"""Hydra entry point for step-level behavior cloning.

Run with:
    PYTHONPATH=third_party/pytorch-lightning/src:data/sample_submission:src \
        python src/tools/train_bc.py
"""

from __future__ import annotations

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.behavior_cloning import (
    BehaviorCloningConfig,
    run_behavior_cloning,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/bc/versions/dev",
)
def main(hydra_config: DictConfig) -> None:
    """Run BC training from extracted Kaggle step shards."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = BehaviorCloningConfig.model_validate(cast(dict[str, Any], raw_config))
    print(run_behavior_cloning(config))


if __name__ == "__main__":
    main()
