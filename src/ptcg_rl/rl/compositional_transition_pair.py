"""Durable source-bound checkpoint-pair transition into routed DCCR-v4."""

from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.evaluation.search_identity import fingerprint_payload
from ptcg_rl.model import build_agent_policy_value_net
from ptcg_rl.rl import compositional_transition_pair_io as pair_io
from ptcg_rl.rl.checkpoint_pair import (
    AsyncCheckpointPairPublisher,
    PublishedCheckpointPair,
)
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload
from ptcg_rl.rl.compositional_factorization import TruncatedSvdConfig
from ptcg_rl.rl.compositional_transition import (
    migrate_dense_v3_to_compositional_v4,
)
from ptcg_rl.rl.learner import WeightPublisher, WeightPublisherConfig
from ptcg_rl.rl.model_publication import prepare_model_state
from ptcg_rl.rl.training import (
    RLTrainConfig,
    _build_lr_scheduler,
    _build_ppo_optimizer,
    _learner_checkpoint_fields,
    _optimizer_parameter_names,
    _planned_lr_scheduler_steps,
    _resume_relevant_config_sha256,
)

_SHA256_LENGTH = 64


class CompositionalPairTransitionConfig(BaseModel):
    """Immutable source pair, target profile, and publication destination."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_policy_path: Path
    source_policy_sha256: str
    source_training_state_path: Path
    source_training_state_sha256: str
    output_dir: Path
    factorization_device: str = "cuda"
    factorization_seed: int
    svd_oversampling: int = 8
    svd_power_iterations: int = 2

    @field_validator("source_policy_sha256", "source_training_state_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require explicit canonical source fingerprints."""
        normalized = value.strip().lower()
        if len(normalized) != _SHA256_LENGTH or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("transition source fingerprints must be SHA256")
        return normalized

    @field_validator("factorization_device")
    @classmethod
    def nonempty_device(cls, value: str) -> str:
        """Reject an empty factorization device."""
        if not value.strip():
            raise ValueError("factorization_device must be non-empty")
        return value

    @field_validator(
        "factorization_seed",
        "svd_oversampling",
        "svd_power_iterations",
    )
    @classmethod
    def nonnegative_factorization_integer(cls, value: int) -> int:
        """Keep the formal randomized factorization recipe bounded and valid."""
        if isinstance(value, bool) or value < 0:
            raise ValueError("factorization integers must be non-negative")
        return value


def publish_compositional_transition_pair(
    target_config: RLTrainConfig,
    transition: CompositionalPairTransitionConfig,
) -> tuple[PublishedCheckpointPair, Mapping[str, Any]]:
    """Convert online/target states separately and publish a fresh exact pair."""
    source_policy_path = pair_io.resolved_file(
        transition.source_policy_path,
        label="source policy checkpoint",
    )
    source_state_path = pair_io.resolved_file(
        transition.source_training_state_path,
        label="source learner state",
    )
    pair_io.require_file_sha256(
        source_policy_path,
        transition.source_policy_sha256,
        label="source policy checkpoint",
    )
    pair_io.require_file_sha256(
        source_state_path,
        transition.source_training_state_sha256,
        label="source learner state",
    )
    source_version = pair_io.checkpoint_version(source_policy_path)
    factorization_config = TruncatedSvdConfig(
        seed=transition.factorization_seed,
        oversampling=transition.svd_oversampling,
        power_iterations=transition.svd_power_iterations,
    )
    output_dir = deck_records.repo_path(transition.output_dir).resolve()
    pair_io.validate_target_profile(
        target_config,
        source_policy_path=source_policy_path,
        output_dir=output_dir,
        policy_version=source_version,
    )

    source_checkpoint = torch.load(
        source_policy_path,
        map_location="cpu",
        weights_only=False,
    )
    source_config = pair_io.checkpoint_model_config(source_checkpoint)
    source_online_state = pair_io.checkpoint_state_dict(source_checkpoint)
    source_sidecar = torch.load(
        source_state_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(source_sidecar, dict):
        raise ValueError("source learner state must be a mapping")
    pair_io.validate_source_sidecar(
        source_sidecar,
        source_policy_path=source_policy_path,
        source_policy_sha256=transition.source_policy_sha256,
        policy_version=source_version,
    )
    source_auxiliary = source_sidecar.get("auxiliary_state")
    if not isinstance(source_auxiliary, Mapping):
        raise ValueError("source learner state has no auxiliary target state")
    source_policy_iteration = source_auxiliary.get("amortized_policy_iteration")
    if not isinstance(source_policy_iteration, Mapping):
        raise ValueError("source learner state has no policy-iteration target")
    source_target_state = source_policy_iteration.get("target_model_state_dict")
    if not isinstance(source_target_state, Mapping):
        raise ValueError("source policy-iteration target model is missing")
    source_target_state = pair_io.tensor_state_dict(
        source_target_state,
        label="source target model",
    )

    torch.manual_seed(target_config.seed)
    online_model = build_agent_policy_value_net(target_config.model).cpu()
    initial_target_state = {
        name: value.detach().cpu().clone()
        for name, value in online_model.state_dict().items()
    }
    online_result = migrate_dense_v3_to_compositional_v4(
        online_model,
        source_online_state,
        source_config=source_config,
        factorization_device=transition.factorization_device,
        factorization_config=factorization_config,
        source_policy_sha256=transition.source_policy_sha256,
    )
    del source_checkpoint, source_online_state
    gc.collect()

    target_model = build_agent_policy_value_net(target_config.model).cpu()
    target_model.load_state_dict(initial_target_state, strict=True)
    del initial_target_state
    target_result = migrate_dense_v3_to_compositional_v4(
        target_model,
        source_target_state,
        source_config=source_config,
        factorization_device=transition.factorization_device,
        factorization_config=factorization_config,
        source_policy_sha256=transition.source_policy_sha256,
    )
    del source_target_state, target_model
    gc.collect()

    auxiliary_state, auxiliary_decisions = pair_io.converted_auxiliary_state(
        source_auxiliary,
        target_state=target_result.state_dict,
    )
    target_manifest = target_result.manifest
    source_counters = {
        "completed_iterations": int(source_sidecar["completed_iterations"]),
        "total_optimizer_updates": int(source_sidecar["total_optimizer_updates"]),
        "planned_scheduler_steps": int(source_sidecar["planned_scheduler_steps"]),
        "policy_iteration_optimizer_updates": int(
            source_policy_iteration["optimizer_updates"]
        ),
    }
    del source_sidecar, source_auxiliary, source_policy_iteration, target_result
    gc.collect()

    manifest = {
        "schema": "dense-v3-to-dccr-v4-checkpoint-pair-v1",
        "lossy_topology_transition": True,
        "source_pair": {
            "policy_path": deck_records.display_path(source_policy_path),
            "policy_sha256": transition.source_policy_sha256,
            "training_state_path": deck_records.display_path(source_state_path),
            "training_state_sha256": transition.source_training_state_sha256,
            "policy_version": source_version,
        },
        "target": {
            "run_version": target_config.run.version,
            "output_dir": deck_records.display_path(output_dir),
            "model_config_sha256": fingerprint_payload(
                target_config.model.model_dump(mode="json")
            ),
            "resume_config_sha256": _resume_relevant_config_sha256(target_config),
        },
        "factorization_recipe": factorization_config.manifest(),
        "online_conversion": online_result.manifest,
        "target_conversion": target_manifest,
        "state_decisions": {
            "optimizer": "reset_fresh_for_changed_topology",
            "scheduler": "restart_fixed_target_budget",
            "completed_iterations": "reset_zero_new_branch",
            "total_optimizer_updates": "reset_zero_new_branch",
            "ppo_optimizer_updates": auxiliary_decisions["ppo_optimizer_updates"],
            "serving_shadow": "republish_from_converted_online",
            "target_model": "converted_separately_preserving_source_lag",
            "policy_iteration_optimizer_clock": "preserved",
            "pending_targets": auxiliary_decisions["pending_targets"],
            "other_auxiliary": auxiliary_decisions["other_auxiliary"],
            "replay": "new_run_empty_store",
            "curriculum": "profile_declared_bootstrap",
        },
        "source_counters": source_counters,
        "fixed_budgets": {
            "distillation_optimizer_updates": (
                target_config.ppo.transition_distillation.optimizer_updates
            ),
            "total_training_iterations": (target_config.collection.training_iterations),
            "scheduler_total_updates": _planned_lr_scheduler_steps(target_config),
        },
    }
    manifest_sha256 = fingerprint_payload(manifest)
    manifest = {**manifest, "manifest_sha256": manifest_sha256}
    manifest_path = output_dir / "transition" / "transition_manifest.json"
    atomic_write_bytes(
        manifest_path,
        json_payload(manifest),
        overwrite=False,
    )

    online_model.load_state_dict(online_result.state_dict, strict=True)
    optimizer = _build_ppo_optimizer(
        target_config,
        model=online_model,
        device=torch.device("cpu"),
    )
    planned_steps = _planned_lr_scheduler_steps(target_config)
    scheduler = _build_lr_scheduler(
        optimizer,
        config=target_config,
        planned_steps=planned_steps,
    )
    metadata = {
        "architecture_transition": {
            "kind": "dense-v3-to-dccr-v4-lossy",
            "manifest_path": deck_records.display_path(manifest_path),
            "manifest_sha256": manifest_sha256,
            "source_policy_sha256": transition.source_policy_sha256,
            "source_training_state_sha256": (transition.source_training_state_sha256),
        }
    }
    prepared_online = prepare_model_state(online_result.state_dict)
    publisher_config = WeightPublisherConfig(
        keep_last=target_config.learner.disk_checkpoint_keep_last,
        retain_every_versions=(
            target_config.learner.disk_checkpoint_retain_every_versions
        ),
    )
    weight_publisher = WeightPublisher(output_dir / "weights", publisher_config)
    with AsyncCheckpointPairPublisher(
        weight_publisher,
        output_dir / "resume",
        keep_last=publisher_config.keep_last,
        retain_every_versions=publisher_config.retain_every_versions,
    ) as pair_publisher:
        pair = pair_publisher.publish(
            prepared_online,
            version=source_version,
            metadata=metadata,
            checkpoint_fields=_learner_checkpoint_fields(
                target_config,
                metadata=metadata,
            ),
            completed_iterations=0,
            total_optimizer_updates=0,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            planned_scheduler_steps=planned_steps,
            resume_config_sha256=_resume_relevant_config_sha256(target_config),
            optimizer_parameter_names=_optimizer_parameter_names(
                online_model,
                optimizer,
            ),
            auxiliary_state=auxiliary_state,
            take_prepared_ownership=True,
            take_auxiliary_ownership=True,
        )

    del prepared_online, auxiliary_state, online_result
    pair_io.verify_published_pair(
        target_config,
        pair=pair,
        policy_version=source_version,
    )
    receipt = {
        "schema": "dense-v3-to-dccr-v4-transition-receipt-v1",
        "manifest_path": deck_records.display_path(manifest_path),
        "manifest_sha256": manifest_sha256,
        "policy_path": deck_records.display_path(pair.policy.path),
        "policy_sha256": pair.policy_sha256,
        "policy_size_bytes": pair.policy_size_bytes,
        "training_state_path": deck_records.display_path(pair.training_state_path),
        "training_state_sha256": pair.training_state_sha256,
        "training_state_size_bytes": pair.training_state_size_bytes,
        "pair_manifest_path": deck_records.display_path(pair.pair_manifest_path),
        "policy_version": pair.version,
        "strict_resume_verified": True,
    }
    receipt_path = output_dir / "transition" / "transition_receipt.json"
    atomic_write_bytes(receipt_path, json_payload(receipt), overwrite=False)
    return pair, receipt


__all__ = [
    "CompositionalPairTransitionConfig",
    "publish_compositional_transition_pair",
]
