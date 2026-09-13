"""Hydra entry point for the P0 replay ActTime audit."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.search_timing import (
    SearchTimingAuditConfig,
    run_search_timing_audit,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/inference_time_search/replay_timing",
)
def main(hydra_config: DictConfig) -> None:
    """Run the streaming decision-level timing audit."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = SearchTimingAuditConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_search_timing_audit(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
