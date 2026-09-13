"""Hydra entry point for the native collection campaign WebUI."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_collection_campaign.webui import (
    CampaignWebUiConfig,
    run_campaign_webui,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation/native_collection_campaign_webui",
    config_name="base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve, validate, and serve one immutable campaign plan."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping")
    config = CampaignWebUiConfig.model_validate(cast(dict[str, Any], raw))
    run_campaign_webui(config, root=records.repo_path(Path(".")))


if __name__ == "__main__":
    main()
