"""Hydra entry point for immutable multi-device campaign planning."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.native_collection_campaign.planner import (
    CampaignBuildConfig,
    build_campaign_plan,
)


@hydra.main(
    version_base=None,
    config_path="../../configs/evaluation/native_collection_campaign",
    config_name="base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve one stage and publish its immutable device work graph."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a mapping")
    config = CampaignBuildConfig.model_validate(cast(dict[str, Any], raw))
    plan = build_campaign_plan(config)
    print(
        json.dumps(
            {
                "plan_fingerprint": plan.plan_fingerprint,
                "stage_id": plan.stage_id,
                "candidate_bundles": len(plan.candidates),
                "opponent_bundles": len(plan.opponents),
                "tasks": len(plan.tasks),
                "games": sum(task.gauntlet.total_games for task in plan.tasks),
                "plan_path": str(config.plan_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
