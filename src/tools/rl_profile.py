"""Hydra entry point for RL rollout and learner profiling.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/rl_profile.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.profile import RLProfileConfig, run_rl_profile


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/profile",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured RL throughput profiling."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = RLProfileConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_rl_profile(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
