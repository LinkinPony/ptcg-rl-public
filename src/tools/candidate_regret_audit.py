"""Hydra entry point for cross-deck/prompt candidate-regret evidence.

Run with:
    PYTHONPATH=data/sample_submission:src \
      python src/tools/candidate_regret_audit.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.candidate_regret_audit import run_candidate_regret_audit
from ptcg_rl.evaluation.candidate_regret_config import CandidateRegretAuditConfig


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/candidate_regret_audit",
)
def main(hydra_config: DictConfig) -> None:
    """Resolve, validate, and run the diagnostic campaign."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = CandidateRegretAuditConfig.model_validate(cast(dict[str, Any], raw))
    summary = run_candidate_regret_audit(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
