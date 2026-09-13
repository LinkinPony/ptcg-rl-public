"""Hydra entry point for bounded native cross-checkpoint roster evaluation."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.native_checkpoint_gauntlet import (
    NativeCheckpointGauntletConfig,
    run_native_checkpoint_gauntlet,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation/native_checkpoint_gauntlet",
    config_name="base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve and execute one immutable cross-checkpoint campaign."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping")
    config = NativeCheckpointGauntletConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(run_native_checkpoint_gauntlet(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
