"""Hydra entry point for S2 virtual-clock frozen-trace stress."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.search_stress import run_search_trace_stress
from ptcg_rl.evaluation.search_stress_config import SearchTraceStressConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/inference_time_search/trace_stress",
)
def main(hydra_config: DictConfig) -> None:
    """Run S2 trace, slowdown, and low-overage boundary cells."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = SearchTraceStressConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_search_trace_stress(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
