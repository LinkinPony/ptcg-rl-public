"""Hydra entry point for opponent-belief calibration reports."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.context.calibration import (
    BeliefCalibrationConfig,
    run_belief_calibration,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="context/belief_calibration",
)
def main(hydra_config: DictConfig) -> None:
    """Run belief calibration and print a JSON summary."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = BeliefCalibrationConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_belief_calibration(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
