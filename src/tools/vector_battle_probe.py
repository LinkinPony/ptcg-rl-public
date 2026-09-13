"""Hydra entry point for direct pointer Battle API probes.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/vector_battle_probe.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.engine.vector_probe import (
    VectorBattleProbeConfig,
    run_vector_battle_probe,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/vector_probe",
)
def main(hydra_config: DictConfig) -> None:
    """Run configured direct-pointer Battle API probes."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = VectorBattleProbeConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_vector_battle_probe(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
