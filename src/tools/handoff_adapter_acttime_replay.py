"""Hydra entry point for the packaged handoff adapter ActTime replay."""

from __future__ import annotations

import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.evaluation.handoff_adapter_replay import (
    run_handoff_adapter_act_time_replay,
)
from ptcg_rl.evaluation.handoff_adapter_replay_config import (
    HandoffAdapterActTimeReplayConfig,
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/handoff_adapter_acttime_replay",
)
def main(hydra_config: DictConfig) -> None:
    """Validate immutable inputs, execute the direct A/B, and publish evidence."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = HandoffAdapterActTimeReplayConfig.model_validate(cast(dict[str, Any], raw))
    summary = run_handoff_adapter_act_time_replay(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
