"""Hydra entry point for qualified replay-window collection."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.qualified_replay_collection import (
    QualifiedReplayCollectionRequestConfig,
    collect_qualified_replays,
    resolve_qualified_replay_collection,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="data/kaggle_1100plus_exact_deck_bc_20260808",
)
def main(hydra_config: DictConfig) -> None:
    """Collect bounded public episodes for the frozen qualified submissions."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    request = QualifiedReplayCollectionRequestConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    config = resolve_qualified_replay_collection(request)
    report = collect_qualified_replays(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
