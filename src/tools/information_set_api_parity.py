"""Offline engine-parity probe for information-set policy iteration.

Run with:
    PYTHONPATH=data/sample_submission:src \
      python src/tools/information_set_api_parity.py
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
    config_name="engine/information_set_api_parity",
)
def main(hydra_config: DictConfig) -> None:
    """Validate and execute the independent Search/native grid comparison."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = DecisionTransitionParityConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    summary = run_decision_transition_parity_audit(config)
    _require_valid_parity(summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _require_valid_parity(summary: dict[str, Any]) -> None:
    """Reject evidence that cannot establish the native transition contract."""
    coverage = cast(dict[str, Any], summary["coverage"])
    failures = cast(dict[str, Any], summary["failures"])
    invalid = {
        "native_parity_failure_cells": int(
            coverage["native_parity_failure_cells"]
        ),
        "unmatched_isolated_cells": int(coverage["interpretable_cells"])
        - int(coverage["native_isolated_match_cells"]),
        "root_failures": sum(
            int(value)
            for value in cast(dict[str, Any], failures["root_failure_counts"]).values()
        ),
        "reference_failures": sum(
            int(value)
            for value in cast(
                dict[str, Any],
                failures["reference_failure_counts"],
            ).values()
        ),
    }
    if any(invalid.values()):
        raise RuntimeError(f"information-set engine parity failed: {invalid}")


if __name__ == "__main__":
    main()
