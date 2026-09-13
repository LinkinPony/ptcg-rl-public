"""Export and publish the preregistered deck-conditioning comparison inputs."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.deck_conditioning_comparison import (
    DeckConditioningComparisonConfig,
    prepare_deck_conditioning_comparison,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/deck_conditioning_comparison/base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve the immutable profile and publish final-precision runner JSON."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("deck conditioning comparison config must be a mapping")
    config = DeckConditioningComparisonConfig.model_validate(cast(dict[str, Any], raw))
    print(
        json.dumps(
            prepare_deck_conditioning_comparison(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
