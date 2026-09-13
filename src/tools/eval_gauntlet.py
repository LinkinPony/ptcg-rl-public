"""Hydra entry point for tiered checkpoint gauntlet evaluation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/eval_gauntlet.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.gauntlet import GauntletConfig, run_gauntlet


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/arena/versions/gauntlet_smoke",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured tiered gauntlet evaluation."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = GauntletConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_gauntlet(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
