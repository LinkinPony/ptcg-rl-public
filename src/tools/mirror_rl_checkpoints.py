"""Hydra entry point for a remote RL checkpoint/resume mirror."""

from __future__ import annotations

import json
import sys
import time
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.rl.distributed.checkpoint_mirror import (
    CheckpointMirror,
    CheckpointMirrorConfig,
    MirrorResult,
)


def _report(result: MirrorResult) -> None:
    print(json.dumps(result.to_dict(), sort_keys=True), flush=True)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="rl/checkpoint_mirror",
)
def main(hydra_config: DictConfig) -> None:
    """Validate the Hydra config and run the mirror once or continuously."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw_config, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = CheckpointMirrorConfig.model_validate(cast(dict[str, Any], raw_config))
    mirror = CheckpointMirror(config)
    if config.watch:
        while True:
            try:
                _report(mirror.mirror_once())
            except Exception as error:  # Keep a long-running safety mirror alive.
                print(
                    json.dumps({"status": "error", "detail": str(error)}),
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(config.poll_interval_seconds)
    else:
        _report(mirror.mirror_once())


if __name__ == "__main__":
    main()
