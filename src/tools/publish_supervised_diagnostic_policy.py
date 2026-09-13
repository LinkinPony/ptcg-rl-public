"""Hydra entry point for diagnostic supervised-to-routed policy publication."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.simple_stateless_diagnostic_policy import (
    DiagnosticRoutedPolicyConfig,
    publish_diagnostic_routed_policy,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/publish_supervised_diagnostic_policy",
)
def main(hydra_config: DictConfig) -> None:
    """Publish or verify one diagnostic-only routed policy."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = DiagnosticRoutedPolicyConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(
        json.dumps(
            publish_diagnostic_routed_policy(config),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
