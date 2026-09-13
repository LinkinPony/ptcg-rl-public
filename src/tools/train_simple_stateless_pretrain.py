"""Hydra entry point for full-corpus simple-stateless replay pretraining."""

# ruff: noqa: E402

from __future__ import annotations

from typing import Any, cast

from ptcg_rl.training.cuda_allocator import configure_cuda_allocator

configure_cuda_allocator()

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.simple_stateless_pretrain import (
    run_simple_stateless_pretraining,
)
from ptcg_rl.training.simple_stateless_pretrain_config import (
    SimpleStatelessPretrainingConfig,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/simple_stateless_pretrain/top30_full_20260723",
)
def main(hydra_config: DictConfig) -> None:
    """Extract the configured corpus and run supervised pretraining."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = SimpleStatelessPretrainingConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(run_simple_stateless_pretraining(config))


if __name__ == "__main__":
    main()
