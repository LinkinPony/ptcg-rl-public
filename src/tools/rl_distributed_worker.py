"""Hydra entry point for a distributed actor+inference RL worker.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/rl_distributed_worker.py
"""

from __future__ import annotations

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.training import RLTrainConfig, run_distributed_actor_worker


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/train/base",
)
def main(hydra_config: DictConfig) -> None:
    """Run a distributed actor+inference worker."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = RLTrainConfig.model_validate(cast(dict[str, Any], raw_config))
    print(run_distributed_actor_worker(config))


if __name__ == "__main__":
    main()
