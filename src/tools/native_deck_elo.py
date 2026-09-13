"""Hydra entry point for bounded native batched deck Elo evaluation."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.native_deck_elo import (
    NativeDeckEloConfig,
    run_native_deck_elo,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation/native_deck_elo",
    config_name="base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve one immutable campaign and run it to completion."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping")
    config = NativeDeckEloConfig.model_validate(cast(dict[str, Any], raw))
    print(json.dumps(run_native_deck_elo(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
