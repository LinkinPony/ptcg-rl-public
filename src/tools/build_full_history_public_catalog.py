"""Hydra entry point for the immutable full-history public deck catalog."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle_deck.catalog_history import (
    FullHistoryPublicCatalogConfig,
    run,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="data/kaggle_public_catalog_full_history_20260807",
)
def main(hydra_config: DictConfig) -> None:
    """Build and report one immutable public catalog asset."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    report = run(FullHistoryPublicCatalogConfig.model_validate(cast(dict[str, Any], raw_config)))
    print(json.dumps(report["result"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
