"""Hydra entry point for deployment-native release archive H2H."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.release_h2h import ReleaseH2HConfig, run_release_h2h


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/release_h2h/base",
)
def main(hydra_config: DictConfig) -> None:
    """Validate the resolved campaign and run or resume its games."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = ReleaseH2HConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_release_h2h(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
