"""Hydra entry point for opponent-pool smoke checks.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/opponent_smoke.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.opponents.smoke import OpponentSmokeConfig, run_opponent_smoke


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="opponents/smoke",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured opponent-pool smoke checks."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = OpponentSmokeConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_opponent_smoke(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
