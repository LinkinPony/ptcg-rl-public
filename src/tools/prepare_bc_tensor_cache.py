"""Build tensor-ready behavior-cloning batch cache shards.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/prepare_bc_tensor_cache.py \
        --config-name training/bc/versions/first_100step
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.training.bc_tensor_cache import write_bc_tensor_cache
from ptcg_rl.training.behavior_cloning import BehaviorCloningConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="training/bc/versions/first_100step",
)
def main(hydra_config: DictConfig) -> None:
    """Build a tensor cache from the configured Parquet BC data source."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = BehaviorCloningConfig.model_validate(cast(dict[str, Any], raw_config))
    output_dir = config.data.tensor_cache_dir or Path(
        "outputs/training/bc_tensor_cache"
    ) / f"{config.run.version}_bs{config.data.batch_size}"
    source_data = config.data.model_copy(update={"tensor_cache_dir": None})
    manifest = write_bc_tensor_cache(
        source_data,
        output_dir=output_dir,
        seed=config.seed,
        max_train_batches=config.max_train_batches_per_epoch,
        max_validation_batches=config.max_validation_batches,
    )
    split_summary = {
        split: {
            "batch_count": data["batch_count"],
            "samples": data["samples"],
        }
        for split, data in manifest["splits"].items()
    }
    print({"output_dir": str(output_dir), "splits": split_summary})


if __name__ == "__main__":
    main()
