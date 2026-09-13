"""Hydra entry point for controlled S2 runtime fault injection."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.search_fault_stress import run_search_fault_stress
from ptcg_rl.evaluation.search_stress_config import SearchTraceStressConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/inference_time_search/fault_stress",
)
def main(hydra_config: DictConfig) -> None:
    """Run search exception, partial coverage, and pool-pressure diagnostics."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = SearchTraceStressConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_search_fault_stress(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
