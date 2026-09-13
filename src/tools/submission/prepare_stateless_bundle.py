"""Hydra entry point for immutable fixed-deck stateless deployment bundles."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.submission.release_assets.stateless_builder import (
    StatelessReleaseBundleConfig,
    prepare_stateless_release_bundle,
)


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="submission/stateless_release_bundle",
)
def main(hydra_config: DictConfig) -> None:
    """Export and freeze one fixed-deck deployment bundle."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = StatelessReleaseBundleConfig.model_validate(cast(dict[str, Any], raw))
    print(
        json.dumps(
            prepare_stateless_release_bundle(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
