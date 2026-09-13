"""Hydra entry point for posterior deck-strength post-processing."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.deck_strength import DeckStrengthConfig, score_deck_strength


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/deck_strength/base",
)
def main(hydra_config: DictConfig) -> None:
    """Build hierarchical meta weights and score bundle gauntlet artifacts."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = DeckStrengthConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(score_deck_strength(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
