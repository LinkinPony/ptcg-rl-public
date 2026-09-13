"""Hydra entry point for the deployment-aligned integrated planner profile."""

from __future__ import annotations

import importlib
import json
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.engine.native_planning_session_payload import (
    NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR,
    native_planning_session_abi_fingerprint,
    native_planning_session_schema_fingerprint,
)
from ptcg_rl.evaluation.planner_profile import (
    run_integrated_planner_profile,
    validate_profile_act_time_replays,
    validate_profile_belief_runtime,
)
from ptcg_rl.evaluation.planner_profile_config import (
    REQUIRED_PLANNER_PROFILE_SHAPES,
    IntegratedPlannerProfileConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_builder import (
    build_planner_profile_corpus,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusReader,
)
from ptcg_rl.evaluation.planner_profile_manifest import (
    validate_planner_profile_corpus_manifest,
)
from ptcg_rl.evaluation.planner_profile_package_assets import (
    build_planner_profile_package_assets,
    validate_planner_profile_package_assets,
)
from ptcg_rl.rl.model_fingerprint import constructed_checkpoint_fingerprint


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="evaluation/planner_integrated_profile",
)
def main(hydra_config: DictConfig) -> None:
    """Build the immutable corpus, lint the campaign, or execute it once."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to a dictionary")
    config = IntegratedPlannerProfileConfig.model_validate(cast(dict[str, Any], raw))
    if config.operation == "build_corpus":
        result = build_planner_profile_corpus(config.corpus_build).as_dict()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if config.operation == "build_package_assets":
        result = build_planner_profile_package_assets(config).as_dict()
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    if config.operation == "validate":
        result = _validation_summary(config)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    summary = run_integrated_planner_profile(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _validation_summary(config: IntegratedPlannerProfileConfig) -> dict[str, Any]:
    """Resolve static identities and verify immutable file contracts."""
    from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256

    checkpoint_sha256 = file_sha256(config.checkpoint_path)
    native_library_sha256 = file_sha256(config.native_library_path)
    if checkpoint_sha256 != config.expected_checkpoint_sha256:
        raise ValueError("profile checkpoint fingerprint differs from config")
    model_fingerprint, checkpoint_policy_version = constructed_checkpoint_fingerprint(
        config.checkpoint_path,
        migration_seed=config.model_migration_seed,
    )
    if model_fingerprint != config.expected_model_fingerprint:
        raise ValueError("profile constructed model fingerprint differs from config")
    if checkpoint_policy_version != config.policy_version:
        raise ValueError("profile checkpoint publication version differs from config")
    if native_library_sha256 != config.expected_native_library_sha256:
        raise ValueError("profile native library fingerprint differs from config")
    native_abi_fingerprint = native_planning_session_abi_fingerprint(
        NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR
    )
    if native_abi_fingerprint != config.expected_native_abi_fingerprint:
        raise ValueError("profile native ABI fingerprint differs from config")
    native_schema_fingerprint = native_planning_session_schema_fingerprint()
    if native_schema_fingerprint != config.expected_native_schema_fingerprint:
        raise ValueError("profile native schema fingerprint differs from config")
    prior_sha256, belief_fingerprint = validate_profile_belief_runtime(config)
    act_time_assets = validate_profile_act_time_replays(config)
    corpus = PlannerProfileCorpusReader(
        config.decision_corpus_path,
        expected_sha256=config.expected_decision_corpus_sha256,
    )
    shape_counts = corpus.shape_counts()
    missing_shapes = sorted(REQUIRED_PLANNER_PROFILE_SHAPES.difference(shape_counts))
    if missing_shapes:
        raise ValueError(f"planner corpus is missing required shapes: {missing_shapes}")
    corpus_validation = validate_planner_profile_corpus_manifest(
        corpus,
        config.corpus_build.manifest_path,
        expected_sha256=config.expected_decision_corpus_manifest_sha256,
        shape_counts=shape_counts,
    )
    package_validation = validate_planner_profile_package_assets(config)
    identities: dict[str, dict[str, str]] = {}
    for runtime_id, runtime in sorted(config.runtime_profiles.items()):
        point = next(point for point in config.points if point.runtime_id == runtime_id)
        model_identity = config.model_identity_for(point)
        resolved = runtime.planner.resolve_for_lease(
            model_fingerprint=model_identity.model_fingerprint,
            policy_version=model_identity.policy_version,
            proposal_version=model_identity.proposal_version,
        )
        static = resolved.static
        identities[runtime_id] = {
            "runtime_fingerprint": resolved.runtime_fingerprint,
            "planner_fingerprint": static.planner_fingerprint,
            "controller_fingerprint": static.controller.controller_fingerprint,
            "constructor_fingerprint": static.constructor_fingerprint,
            "scorer_fingerprint": static.scorer_fingerprint,
            "tensorizer_fingerprint": static.tensorizer_fingerprint,
            "continuation_semantics_fingerprint": (
                static.continuation_semantics_fingerprint
            ),
        }
    factory_paths = {config.production_backend_factory}
    for runtime in config.runtime_profiles.values():
        factory_paths.add(runtime.belief.factory)
        if runtime.learner is not None:
            factory_paths.add(runtime.learner.factory)
    for path in sorted(factory_paths):
        _require_factory(path)
    return {
        "campaign_id": config.campaign_id,
        "operation": config.operation,
        "checkpoint_sha256": checkpoint_sha256,
        "model_fingerprint": model_fingerprint,
        "model_migration_seed": config.model_migration_seed,
        "native_library_sha256": native_library_sha256,
        "native_abi_fingerprint": native_abi_fingerprint,
        "native_schema_fingerprint": native_schema_fingerprint,
        "belief_prior_sha256": prior_sha256,
        "belief_runtime_fingerprint": belief_fingerprint,
        "act_time_replay_assets": act_time_assets,
        "decision_corpus_sha256": corpus.sha256,
        "decision_corpus_manifest_sha256": corpus_validation.manifest_sha256,
        "decision_corpus_rows": corpus.rows,
        "validated_corpus_rows": corpus_validation.validated_rows,
        "decision_corpus_provenance": corpus_validation.as_dict(),
        "decision_corpus_shape_counts": shape_counts,
        "package_assets": package_validation.as_dict(),
        "runtime_identities": identities,
        "points": len(config.points),
        "factory_paths": sorted(factory_paths),
        "status": "valid",
    }


def _require_factory(path: str) -> None:
    """Reject a preregistered production surface that is not importable."""
    module_name, factory_name = path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name, None)
    if not callable(factory):
        raise TypeError(f"profile factory is not callable: {path}")


if __name__ == "__main__":
    main()
