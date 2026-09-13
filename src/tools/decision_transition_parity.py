"""Hydra entry point for bounded exact decision-transition parity evidence.

Run with:
    PYTHONPATH=data/sample_submission:src \
      python src/tools/decision_transition_parity.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.consequence_parity_audit import (
    run_decision_transition_parity_audit,
)
from ptcg_rl.evaluation.consequence_parity_config import (
    DecisionTransitionParityConfig,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/decision_transition_parity",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve, validate, and run the audit configuration."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = DecisionTransitionParityConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    summary = run_decision_transition_parity_audit(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
