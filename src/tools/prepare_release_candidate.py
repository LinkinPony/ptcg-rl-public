"""Hydra entry point for immutable release-candidate assets."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.submission.release_assets.builder import (
    ReleaseCandidateAssetConfig,
    prepare_release_candidate_asset,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="submission/release_asset_build",
)
def main(hydra_config: DictConfig) -> None:
    """Export and validate one fixed checkpoint/deployment bundle."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = ReleaseCandidateAssetConfig.model_validate(cast(dict[str, Any], raw))
    print(
        json.dumps(
            prepare_release_candidate_asset(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
