"""Hydra entry point for game-context log increment characterization.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/game_context_log_probe.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.context.log_probe import (
    GameContextLogProbeConfig,
    run_game_context_log_probe,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="context/log_probe",
)
def main(hydra_config: DictConfig) -> None:
    """Run the log increment probe."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = GameContextLogProbeConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(run_game_context_log_probe(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
