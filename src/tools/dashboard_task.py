"""Fixed JSON runners used by the dashboard task workbench."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ptcg_rl.evaluation.distributed_runner import (
    DistributedReleaseLaunchConfig,
    run_distributed_release_h2h,
)
from ptcg_rl.evaluation.release_h2h import ReleaseH2HConfig, run_release_h2h
from ptcg_rl.training.runtime_deck_ladder import (
    RuntimeDeckLadderConfig,
    run_runtime_deck_ladder,
)


def main() -> int:
    """Execute one allowlisted runner from a server-generated JSON config."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind",
        required=True,
        choices=("release_h2h", "runtime_elo"),
    )
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    payload = _read_object(args.config)
    if args.kind == "runtime_elo":
        config = RuntimeDeckLadderConfig.model_validate(payload)
        result = run_runtime_deck_ladder(config)
    else:
        release_raw = payload.get("release")
        if not isinstance(release_raw, dict):
            raise ValueError("release H2H wrapper is missing release config")
        release = ReleaseH2HConfig.model_validate(release_raw)
        distributed_raw = payload.get("distributed")
        if distributed_raw is None:
            result = run_release_h2h(release)
        else:
            result = run_distributed_release_h2h(
                release,
                DistributedReleaseLaunchConfig.model_validate(distributed_raw),
            )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dashboard task config must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
