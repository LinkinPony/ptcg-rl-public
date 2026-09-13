"""Actor-local model and planner runtimes for integrated profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ptcg_rl.agent.packaged_planner import PackagedPlannerConfig
from ptcg_rl.agent.planner_select_policy import PlannerSelectPolicy
from ptcg_rl.agent.runtime import CheckpointPolicy, PolicyRuntimeAgent
from ptcg_rl.belief.runtime_identity import belief_runtime_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfileModelIdentity,
    PlannerProfilePointConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusRecord,
)
from ptcg_rl.evaluation.planner_profile_package_models import (
    PlannerPackageAssetManifestEntry,
)
from ptcg_rl.model.network import SampleDecodeTrace
from ptcg_rl.rl.collection import ModelRolloutPolicy
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.planner_profile_archive import (
    ExtractedPlannerPackage,
    extract_validated_planner_package,
)
from ptcg_rl.rl.planner_profile_backend_factories import (
    ProfileBeliefRuntime,
    create_profile_belief_sampler,
)
from ptcg_rl.rl.planner_profile_inference import ProfileInferenceServer
from ptcg_rl.rl.planner_profile_inputs import CollatedProfileRoots
from ptcg_rl.rl.planner_runtime_factory import (
    PlannerBehaviorRuntime,
    create_planner_behavior_runtime,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity


@dataclass(slots=True)
class ProfileActor:
    """One policy client and actor-local native planner lane."""

    actor_id: str
    policy: Any
    planner_runtime: PlannerBehaviorRuntime
    belief: ProfileBeliefRuntime


@dataclass(frozen=True, slots=True)
class ProfileActorWave:
    """One decoded actor-local root wave awaiting shared planning."""

    actor: ProfileActor
    records: tuple[PlannerProfileCorpusRecord, ...]
    collated: CollatedProfileRoots
    trace: SampleDecodeTrace
    actor_policy_wait_ms: float


@dataclass(frozen=True, slots=True)
class H200ActorResources:
    """One inference server and its fixed actor-local services."""

    inference_server: ProfileInferenceServer
    actors: tuple[ProfileActor, ...]


@dataclass(frozen=True, slots=True)
class PackagedActorResources:
    """Local stateless service plus the separate ActTime agent surface."""

    actor: ProfileActor
    agent: PolicyRuntimeAgent
    planner_policy: PlannerSelectPolicy
    workspace: ExtractedPlannerPackage
    deployment_deck: tuple[int, ...]


def create_h200_actor_resources(
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    identity: PlannerProfileModelIdentity,
) -> H200ActorResources:
    """Create one spawned CUDA server and fixed actor-local clients."""
    actor_count = runtime.planner.batching.actor_count
    actor_ids = tuple(f"profile-actor-{index}" for index in range(actor_count))
    server = ProfileInferenceServer.start(
        checkpoint_path=identity.checkpoint_path,
        expected_model_fingerprint=identity.model_fingerprint,
        policy_version=identity.policy_version,
        proposal_version=identity.proposal_version,
        runtime=runtime.planner,
        actor_purposes=dict.fromkeys(actor_ids, "planner_behavior"),
    )
    actors: list[ProfileActor] = []
    try:
        for actor_id in actor_ids:
            belief = create_profile_belief_sampler(runtime.belief)
            planner = create_planner_behavior_runtime(
                runtime_config=runtime.planner,
                sampler_config=runtime.belief.sampler,
                belief_config=runtime.belief.producer,
                stochastic_seed=runtime.belief.stochastic_seed,
                native_library_path=config.native_library_path,
            )
            actors.append(
                ProfileActor(
                    actor_id=actor_id,
                    policy=server.clients[actor_id],
                    planner_runtime=planner,
                    belief=belief,
                )
            )
    except Exception:
        for actor in actors:
            actor.planner_runtime.close()
        server.close()
        raise
    return H200ActorResources(
        inference_server=server,
        actors=tuple(actors),
    )


def create_packaged_actor_resources(
    *,
    config: IntegratedPlannerProfileConfig,
    point: PlannerProfilePointConfig,
    runtime: PlannerProfileRuntimeConfig,
    identity: PlannerProfileModelIdentity,
    resolved: ResolvedPlannerRuntimeIdentity,
    asset: PlannerPackageAssetManifestEntry,
) -> PackagedActorResources:
    """Create the exact FP16 CPU service from one prebuilt package archive."""
    packaged = runtime.packaged
    if packaged is None:
        raise ValueError("packaged profile has no packaged workload")
    workspace = extract_validated_planner_package(asset)
    planner = None
    planner_policy = None
    agent = None
    try:
        spec = PackagedPlannerConfig.from_file(
            workspace.agent_dir / "planner_runtime.json"
        )
        _validate_persisted_spec(
            spec,
            config=config,
            runtime=runtime,
            identity=identity,
            resolved=resolved,
            asset=asset,
        )
        checkpoint_path = workspace.agent_dir / "agent_checkpoint.pt"
        deployment_deck = tuple(
            int(line.strip())
            for line in (workspace.agent_dir / "deck.csv")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        )
        if len(deployment_deck) != 60:
            raise ValueError("packaged profile deployment deck is not 60 cards")
        checkpoint = CheckpointPolicy(checkpoint_path, device="cpu")
        if checkpoint.checkpoint_sha256 != identity.checkpoint_sha256:
            raise ValueError("packaged profile loaded another checkpoint")
        if checkpoint.policy_version != identity.policy_version:
            raise ValueError("packaged profile loaded another policy version")
        if canonical_model_state_fingerprint(checkpoint.planner_model) != (
            identity.model_fingerprint
        ):
            raise ValueError("packaged profile loaded another canonical model")
        sampler = BeliefSampler(config=spec.belief_sampler)
        producer = OpponentBeliefFeatureProducer.from_config(spec.belief_producer)
        belief = ProfileBeliefRuntime(
            sampler=sampler,
            producer=producer,
            fingerprint=belief_runtime_fingerprint(sampler, producer),
        )
        if belief.fingerprint != spec.belief_runtime_fingerprint:
            raise ValueError("extracted package belief runtime differs from spec")
        planner = create_planner_behavior_runtime(
            runtime_config=spec.planner,
            sampler_config=spec.belief_sampler,
            belief_config=spec.belief_producer,
            stochastic_seed=spec.stochastic_seed,
            native_library_path=spec.native_library_path,
        )
        planner_policy = PlannerSelectPolicy(
            model=checkpoint.planner_model,
            device=checkpoint.planner_device,
            runtime_config=spec.planner,
            service=planner.service,
            policy_version=spec.policy_version,
            proposal_version=spec.proposal_version,
            verified_model_fingerprint=spec.model_fingerprint,
            base_policy=checkpoint,
            temperature=0.0,
        )
        agent = PolicyRuntimeAgent(
            config=packaged.act_time.model_copy(
                update={
                    "checkpoint_path": checkpoint_path,
                    "planner": spec if point.planner_enabled else None,
                }
            ),
            policy=planner_policy if point.planner_enabled else checkpoint,
        )
        direct_policy = ModelRolloutPolicy(
            checkpoint.planner_model,
            policy_version=spec.policy_version,
            planner_context_capacity=spec.planner.contexts.retained_root_rows,
            verified_model_fingerprint=spec.model_fingerprint,
            proposal_version=spec.proposal_version,
        )
        return PackagedActorResources(
            actor=ProfileActor(
                actor_id="packaged-profile-actor",
                policy=direct_policy,
                planner_runtime=planner,
                belief=belief,
            ),
            agent=agent,
            planner_policy=planner_policy,
            workspace=workspace,
            deployment_deck=deployment_deck,
        )
    except Exception:
        if agent is not None:
            agent.close()
        elif planner_policy is not None:
            planner_policy.close()
        if planner is not None:
            planner.close()
        workspace.close()
        raise


def _validate_persisted_spec(
    spec: PackagedPlannerConfig,
    *,
    config: IntegratedPlannerProfileConfig,
    runtime: PlannerProfileRuntimeConfig,
    identity: PlannerProfileModelIdentity,
    resolved: ResolvedPlannerRuntimeIdentity,
    asset: PlannerPackageAssetManifestEntry,
) -> None:
    packaged = runtime.packaged
    if packaged is None:
        raise ValueError("persisted package spec requires a packaged workload")
    expected = (
        identity.source_checkpoint_sha256,
        identity.checkpoint_sha256,
        identity.model_fingerprint,
        identity.policy_version,
        identity.proposal_version,
        resolved.static.planner_fingerprint,
        resolved.runtime_fingerprint,
        config.expected_native_library_sha256,
        config.expected_native_abi_fingerprint,
        config.expected_native_schema_fingerprint,
        runtime.belief.stochastic_seed,
    )
    actual = (
        spec.source_checkpoint_sha256,
        spec.checkpoint_sha256,
        spec.model_fingerprint,
        spec.policy_version,
        spec.proposal_version,
        spec.expected_planner_fingerprint,
        spec.expected_runtime_fingerprint,
        spec.native_library_sha256,
        spec.native_abi_fingerprint,
        spec.native_schema_fingerprint,
        spec.stochastic_seed,
    )
    if actual != expected:
        raise ValueError("persisted package spec differs from selected point identity")
    if spec.planner_enabled_by_default != packaged.planner_enabled_by_default:
        raise ValueError("persisted package spec changes the immutable planner default")
    if spec.planner != runtime.planner:
        raise ValueError("persisted package spec changes planner semantics")
    if (
        asset.planner_fingerprint != spec.expected_planner_fingerprint
        or asset.runtime_fingerprint != spec.expected_runtime_fingerprint
        or asset.planner_enabled_by_default != spec.planner_enabled_by_default
    ):
        raise ValueError("package manifest differs from its persisted planner spec")


__all__ = [
    "H200ActorResources",
    "PackagedActorResources",
    "ProfileActor",
    "ProfileActorWave",
    "create_h200_actor_resources",
    "create_packaged_actor_resources",
]
