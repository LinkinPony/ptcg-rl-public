"""Hydra entry point for exact bundle-vs-bundle evaluation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/eval_bundle_gauntlet.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.bundle_gauntlet import (
    BundleGauntletConfig,
    run_bundle_gauntlet,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/bundle_gauntlet/base",
)
def main(hydra_config: DictConfig) -> None:
    """Run the configured exact-bundle gauntlet."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = BundleGauntletConfig.model_validate(
        cast(dict[str, Any], raw_config),
    )
    print(json.dumps(run_bundle_gauntlet(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
