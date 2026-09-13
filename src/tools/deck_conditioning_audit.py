"""Run the Hydra-backed deck-conditioned policy verification campaign."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.deck_conditioning_audit import (
    DeckConditioningAuditConfig,
    run_deck_conditioning_audit,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/deck_conditioning_audit/base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve config, execute the audit, and print its immutable summary."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("deck conditioning audit config must be a mapping")
    config = DeckConditioningAuditConfig.model_validate(cast(dict[str, Any], raw))
    print(
        json.dumps(
            run_deck_conditioning_audit(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
