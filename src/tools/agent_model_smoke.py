"""Hydra entry point for real-replay agent model smoke tests.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/agent_model_smoke.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.model.smoke import AgentModelSmokeConfig, run_agent_model_smoke


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="model/smoke",
)
def main(hydra_config: DictConfig) -> None:
    """Run model smoke checks from extracted replay steps."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = AgentModelSmokeConfig.model_validate(cast(dict[str, Any], raw_config))
    print(json.dumps(run_agent_model_smoke(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
