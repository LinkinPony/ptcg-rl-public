"""Hydra coordinator entry point for multi-host immutable release H2H."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.distributed_runner import (
    DistributedReleaseLaunchConfig,
    run_distributed_release_h2h,
)
from ptcg_rl.evaluation.release_h2h import ReleaseH2HConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/release_h2h/distributed_base",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve one campaign, then coordinate its immutable remote shards."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    payload = cast(dict[str, Any], raw)
    distributed_raw = payload.pop("distributed", None)
    if not isinstance(distributed_raw, dict):
        raise ValueError("distributed release config is missing")
    release = ReleaseH2HConfig.model_validate(payload)
    launch = DistributedReleaseLaunchConfig.model_validate(distributed_raw)
    print(
        json.dumps(
            run_distributed_release_h2h(release, launch),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
