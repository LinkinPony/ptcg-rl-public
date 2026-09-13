"""Hydra entry point for local Battle API arena evaluation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/arena_battle.py
"""

from __future__ import annotations

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.arena import ArenaConfig, run_arena


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/arena/versions/dev",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured arena games."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = ArenaConfig.model_validate(cast(dict[str, Any], raw_config))
    print(run_arena(config))


if __name__ == "__main__":
    main()
