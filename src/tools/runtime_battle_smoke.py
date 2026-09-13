"""Hydra entry point for runtime Battle API smoke tests.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/runtime_battle_smoke.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.agent.runtime_smoke import (
    RuntimeBattleSmokeConfig,
    run_runtime_battle_smoke,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="agent/runtime_smoke",
)
def main(hydra_config: DictConfig) -> None:
    """Run one local Battle API smoke game through the Kaggle runtime."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = RuntimeBattleSmokeConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(run_runtime_battle_smoke(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
