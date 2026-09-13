"""Hydra entry point for historical exact-route checkpoint discovery."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.native_collection_campaign.inventory import (
    InventoryBuildConfig,
    build_historical_inventory,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation/native_collection_inventory",
    config_name="base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve discovery inputs and publish one immutable inventory."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping")
    config = InventoryBuildConfig.model_validate(cast(dict[str, Any], raw))
    inventory = build_historical_inventory(config)
    print(
        json.dumps(
            {
                "inventory_fingerprint": inventory.inventory_fingerprint,
                "checkpoints": len(inventory.checkpoints),
                "model_states": len(
                    {item.model_fingerprint for item in inventory.checkpoints}
                ),
                "discovered_policy_checkpoints": len(
                    {
                        path
                        for item in inventory.checkpoints
                        for path in item.discovered_checkpoint_paths
                    }
                ),
                "policy_only_contexts": sum(
                    item.evaluation_binding_path is not None
                    for item in inventory.checkpoints
                ),
                "bundles": sum(len(item.decks) for item in inventory.checkpoints),
                "exclusions": len(inventory.exclusions),
                "output_path": str(config.output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
