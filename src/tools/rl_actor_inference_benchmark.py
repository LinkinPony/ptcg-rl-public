"""Hydra entry point for learner-empty async actor/inference benchmarking.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/rl_actor_inference_benchmark.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.actor_inference_benchmark import (
    ActorInferenceBenchmarkConfig,
    run_actor_inference_benchmark,
)
from ptcg_rl.rl.training import RLTrainConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/train/base",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured learner-empty actor/inference benchmark."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    raw_benchmark = raw_config.pop("benchmark", {})
    if not isinstance(raw_benchmark, dict):
        raise ValueError("benchmark config must resolve to a dictionary.")
    train_config = RLTrainConfig.model_validate(cast(dict[str, Any], raw_config))
    benchmark_config = ActorInferenceBenchmarkConfig.model_validate(raw_benchmark)
    print(
        json.dumps(
            run_actor_inference_benchmark(train_config, benchmark_config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
