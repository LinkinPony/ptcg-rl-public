"""Hydra entry point for the exact-runtime P0 shadow integration smoke."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.search_shadow import (
    SearchShadowSmokeConfig,
    run_search_shadow_smoke,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/inference_time_search/shadow_smoke",
)
def main(hydra_config: DictConfig) -> None:
    """Run disabled-vs-shadow action-equivalence and lifecycle checks."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = SearchShadowSmokeConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_search_shadow_smoke(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
