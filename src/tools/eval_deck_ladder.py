"""Hydra entry point for default Kaggle-like runtime deck ladder evaluation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/eval_deck_ladder.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.runtime_deck_ladder import (
    RuntimeDeckLadderConfig,
    run_runtime_deck_ladder,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/deck_ladder/versions/runtime_dynamic37",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured runtime deck ladder evaluation."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = RuntimeDeckLadderConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_runtime_deck_ladder(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
