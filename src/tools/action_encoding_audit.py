"""Hydra entry point for replay option-encoding coverage audits.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/action_encoding_audit.py
"""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.actions.alignment import (
    OptionEncodingAuditConfig,
    run_option_encoding_audit,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="actions/encoding_audit",
)
def main(hydra_config: DictConfig) -> None:
    """Run the option-encoding audit from Hydra config."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary.")
    config = OptionEncodingAuditConfig.model_validate(
        cast(dict[str, Any], raw_config)
    )
    print(json.dumps(run_option_encoding_audit(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
