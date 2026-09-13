"""Hydra entry point for Stage-0 rollout advantage weights.

Run with:
    PYTHONPATH=src python src/tools/rl_advantage_weights.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.advantage_weights import (
    AdvantageWeightRewriteConfig,
    rewrite_advantage_weight_manifest,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/advantage_weights/stage0",
)
def main(hydra_config: DictConfig) -> None:
    """Append advantage weights to configured rollout shards."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = AdvantageWeightRewriteConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(rewrite_advantage_weight_manifest(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
