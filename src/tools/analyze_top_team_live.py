"""Hydra entry point for fixed-window Kaggle top-team deck analysis."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle.top_team_live import TopTeamLiveConfig, run


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="data/kaggle_top_team_live",
)
def main(hydra_config: DictConfig) -> None:
    """Run the validated live top-team replay pipeline."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = TopTeamLiveConfig.model_validate(cast(dict[str, Any], raw_config))
    report = run(config)
    print(
        json.dumps(
            {
                "analysis": report["analysis"],
                "inventory": report["inventory"],
                "outputs": report["outputs"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
