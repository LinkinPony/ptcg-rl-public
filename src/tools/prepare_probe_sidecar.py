"""Hydra entry point for dynamic-effect probe sidecar extraction."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle_steps.probe_sidecar import (
    ProbeSidecarConfig,
    run_probe_sidecar,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="data/kaggle_steps_probe_sidecar",
)
def main(hydra_config: DictConfig) -> None:
    """Run probe sidecar extraction and print a JSON summary."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = ProbeSidecarConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_probe_sidecar(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
