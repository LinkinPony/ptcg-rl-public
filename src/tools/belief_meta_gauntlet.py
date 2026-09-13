"""Hydra entry point for building a belief-model meta gauntlet."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.belief.gauntlet import MetaGauntletConfig, build_meta_gauntlet


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="belief/meta_gauntlet",
)
def main(config: DictConfig) -> None:
    """Build configured gauntlet artifacts."""
    resolved = OmegaConf.to_container(config, resolve=True)
    gauntlet_config = MetaGauntletConfig.model_validate(resolved)
    summary = build_meta_gauntlet(gauntlet_config)
    print(summary)


if __name__ == "__main__":
    main()
