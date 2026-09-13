"""Hydra entry point for building single-checkpoint parameter ensembles."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.checkpoint_ensemble import (
    CheckpointEnsembleConfig,
    build_checkpoint_ensembles,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation/checkpoint_ensemble",
    config_name="base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve, validate, and build one ensemble campaign."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping")
    config = CheckpointEnsembleConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(build_checkpoint_ensembles(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
