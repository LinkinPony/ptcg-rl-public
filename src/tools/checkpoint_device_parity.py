"""Hydra entry point for CPU-to-GPU checkpoint action parity."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.checkpoint_device_parity import (
    CheckpointDeviceParityConfig,
    run_checkpoint_device_parity,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/checkpoint_selection/device_parity",
)
def main(hydra_config: DictConfig) -> None:
    """Verify that GPU acceleration preserves CPU checkpoint actions."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = CheckpointDeviceParityConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(
        json.dumps(
            run_checkpoint_device_parity(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
