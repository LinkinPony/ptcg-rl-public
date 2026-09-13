"""Hydra entry point for step-extraction throughput benchmarks.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/kaggle_steps_benchmark.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle_steps.benchmark import (
    KaggleStepBenchmarkConfig,
    run_kaggle_step_benchmark,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="data/kaggle_steps_benchmark",
)
def main(hydra_config: DictConfig) -> None:
    """Run the step-extraction benchmark from Hydra config."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = KaggleStepBenchmarkConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(run_kaggle_step_benchmark(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
