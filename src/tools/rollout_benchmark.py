"""Hydra entry point for vectorized rollout throughput benchmarks.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/rollout_benchmark.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.benchmark import RolloutBenchmarkConfig, run_rollout_benchmark


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/rollout_benchmark",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured rollout benchmark."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = RolloutBenchmarkConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_rollout_benchmark(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
