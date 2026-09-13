"""Hydra entry point for rollout artifact validation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/rl_rollout_validate.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl import RolloutArtifactValidationConfig, validate_rollout_artifacts


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/rollout/validation",
)
def main(hydra_config: DictConfig) -> None:
    """Validate configured rollout artifacts."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = RolloutArtifactValidationConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(validate_rollout_artifacts(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
