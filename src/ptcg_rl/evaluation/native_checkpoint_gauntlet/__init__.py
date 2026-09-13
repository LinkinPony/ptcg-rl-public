"""Native cross-checkpoint evaluation over checkpoint-native exact rosters."""

from ptcg_rl.evaluation.native_checkpoint_gauntlet.collection_runner import (
    run_native_collection_checkpoint_gauntlet,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    NativeCheckpointGauntletConfig,
)
from ptcg_rl.evaluation.native_checkpoint_gauntlet.runner import (
    run_native_checkpoint_gauntlet as run_native_match_checkpoint_gauntlet,
)


def run_native_checkpoint_gauntlet(
    config: NativeCheckpointGauntletConfig,
) -> dict[str, object]:
    """Dispatch to the reusable training-path backend or the diagnostic lane."""
    if config.backend == "native_collection":
        return run_native_collection_checkpoint_gauntlet(config)
    return run_native_match_checkpoint_gauntlet(config)


__all__ = ["NativeCheckpointGauntletConfig", "run_native_checkpoint_gauntlet"]
