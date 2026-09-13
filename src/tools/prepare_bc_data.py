"""Hydra entry point for full BC data preparation.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/prepare_bc_data.py
"""

from __future__ import annotations

from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle_steps.full_pipeline import (
    PrepareBCDataConfig,
    run,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="data/prepare_bc_data",
)
def main(hydra_config: DictConfig) -> None:
    """Prepare full BC step shards from Kaggle daily replay datasets."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = PrepareBCDataConfig.model_validate(cast(dict[str, Any], raw_config))
    run(config)


if __name__ == "__main__":
    main()
