"""Full-corpus supervised pretraining for the clean simple-stateless policy."""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from ptcg_rl.context.public_event_arrays import collate_public_event_deltas
from ptcg_rl.model.sequence.action import collate_accepted_actions
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    build_sparse_belief_targets,
    normalized_sparse_belief_row_losses,
    simple_count_first_rows,
    simple_stateless_parameter_report,
    two_hot_wdl_targets,
    uses_wdl_critic,
    validate_simple_stateless_parameter_report,
    wdl_value_from_logits,
)
from ptcg_rl.model.simple_stateless.backbone import (
    select_simple_stateless_backbone_rows,
)
from ptcg_rl.rl.checkpoint_pair_io import (
    atomic_write_bytes,
    json_payload,
    publish_torch_file,
)
from ptcg_rl.rl.model_compatibility import model_config_fingerprint
from ptcg_rl.rl.model_fingerprint import canonical_model_state_fingerprint
from ptcg_rl.rl.policy_inputs import (
    collate_simple_stateless_actor_rows,
    collate_simple_stateless_observation_rows,
)
from ptcg_rl.rl.stateless_checkpoint import load_stateless_checkpoint_pair
from ptcg_rl.rl.stateless_family_private_transition import (
    family_private_transition_fingerprint,
    migrate_family_private_from_pair,
)
from ptcg_rl.rl.stateless_training import (
    resolve_simple_stateless_training_resources,
)
from ptcg_rl.rl.transition_distillation import (
    policy_forward_kl_losses,
    visited_option_prefix_mask,
)
from ptcg_rl.training.host_policy import require_cuda_training_host
from ptcg_rl.training.run_config import resolve_training_output_dir
from ptcg_rl.training.simple_stateless_pretrain_artifact import (
    SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA,
    SupervisedPolicyArtifactManifest,
    SupervisedPolicyBaselineAnchorRecord,
    SupervisedPolicyEvaluationMetrics,
    SupervisedPolicyFinalEpochSelectionRecord,
    SupervisedPolicyInitializationRecord,
    SupervisedPolicyOutcomeWeightingRecord,
    SupervisedPolicySelectionRecord,
    SupervisedPolicyTopologyTransitionRecord,
    SupervisedPolicyTrainMonitorSelectionRecord,
    load_supervised_policy_artifact,
    supervised_outcome_weighting_record,
    supervised_policy_payload,
)
from ptcg_rl.training.simple_stateless_pretrain_config import (
    ReplayPretrainingOptimizationConfig,
    SimpleStatelessPretrainingConfig,
)
from ptcg_rl.training.simple_stateless_pretrain_data import (
    REPLAY_SPLITS,
    TEMPORAL_PRETRAINING_SHARD_SCHEMA,
    PretrainingPartRecord,
    ReplayPretrainingDatasetManifest,
    ReplayPretrainingExample,
    ReplaySplit,
    file_sha256,
    is_temporal_pretraining_format,
    iter_pretraining_rows,
    load_pretraining_manifest,
    load_pretraining_metadata_columns,
    load_pretraining_part,
    load_temporal_pretraining_geometry,
    pretraining_split_indices,
)
from ptcg_rl.training.simple_stateless_pretrain_evaluation import (
    evaluate_pretraining_policy,
    evaluate_temporal_pretraining_monitor,
)
from ptcg_rl.training.simple_stateless_pretrain_extract import (
    build_replay_source_index,
    engine_fact_producer_fingerprint,
    extract_replay_dataset,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    TrainableParameterScope as _TrainableParameterScope,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    assert_frozen_state_unchanged as _assert_frozen_state_unchanged,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    private_parameter_state as _private_parameter_state,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    private_training_records as _residual_training_records,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    resolve_pretraining_batch_routes as _resolve_batch_routes,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    resolve_trainable_parameter_scope as _resolve_trainable_parameter_scope,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    state_subset_fingerprint as _state_subset_fingerprint,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    trainable_scope_audit as _trainable_scope_audit,
)
from ptcg_rl.training.simple_stateless_pretrain_scope import (
    validate_target_dataset as _validate_target_dataset,
)
from ptcg_rl.training.simple_stateless_pretrain_weighting import (
    ReplayPretrainingOutcomeWeightsConfig,
)
from ptcg_rl.training.simple_stateless_temporal_pretrain import (
    TemporalMonitorPlan,
    TemporalPretrainingBatchPlan,
    build_temporal_monitor_plan,
    temporal_epoch_batches,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_RUN_CONFIG_DOMAIN = b"ptcg-rl/simple-stateless-pretraining-run/v1\x00"
_EPISODE_WEIGHTING_SEMANTICS = "global-outcome-weighted-episode-seat-mean-fixed-step-v3"
_EPISODE_STATS_DOMAIN = (
    b"ptcg-rl/simple-stateless-pretraining-episode-seat-stats/v3\x00"
)
_EpisodeSeatKey = tuple[int, int]


@dataclass(frozen=True)
class _TrainingCursor:
    epoch: int
    part_position: int
    batch_position: int
    global_step: int


@dataclass(frozen=True)
class _BatchMetrics:
    loss: float
    policy_loss: float
    baseline_policy_forward_kl_loss: float
    root_value_loss: float
    prefix_value_loss: float
    belief_loss: float
    root_value_mae: float
    gradient_norm: float
    learning_rate: float
    examples: int
    context_examples: int
    decode_tokens: int
    routed_examples: int


@dataclass(frozen=True)
class _EpisodeWeightingPlan:
    """Fixed-scale estimator of the outcome-weighted episode objective."""

    decision_counts: Mapping[_EpisodeSeatKey, int]
    episode_outcomes: Mapping[_EpisodeSeatKey, float]
    train_examples: int
    train_episodes: int
    optimizer_steps_per_epoch: int
    batch_loss_scale: float
    episode_stats_fingerprint: str
    outcome_weighting: SupervisedPolicyOutcomeWeightingRecord


@dataclass(frozen=True)
class _SelectionState:
    """Durable best-validation checkpoint pointer."""

    evaluations: int = 0
    selected_optimizer_step: int | None = None
    selected_epoch: int | None = None
    checkpoint_filename: str | None = None
    checkpoint_size_bytes: int | None = None
    checkpoint_sha256: str | None = None
    selected_model_state_fingerprint: str | None = None
    validation: SupervisedPolicyEvaluationMetrics | None = None
    route_examples: tuple[tuple[str, int], ...] = ()
    initial_monitor: SupervisedPolicyEvaluationMetrics | None = None
    final_monitor: SupervisedPolicyEvaluationMetrics | None = None
    significant_best_nll: float | None = None
    stale_epochs: int = 0
    stop_reason: str | None = None
    monitor_fingerprint: str | None = None


def run_simple_stateless_pretraining(
    config: SimpleStatelessPretrainingConfig,
) -> dict[str, Any]:
    """Extract the complete local corpus and pretrain one RL-compatible policy."""
    resources = resolve_simple_stateless_training_resources(config)
    output_dir = resolve_training_output_dir(
        task_name="pretrain",
        run=config.run,
        output_dir=config.output_dir,
    )
    output_dir = _path(output_dir)
    dataset_dir = (
        output_dir / "dataset"
        if config.data.dataset_dir is None
        else _path(config.data.dataset_dir)
    )
    source_records, source_identity = build_replay_source_index(
        config.data,
        repo_root=_REPO_ROOT,
    )
    sequence_config = (
        resources.model_config.sequence if config.data.temporal_sequence else None
    )
    engine_fact_config = (
        None if sequence_config is None else sequence_config.engine_facts
    )
    expected_engine_fact_fingerprint = engine_fact_producer_fingerprint(
        engine_fact_config
    )
    validation = {
        "trainer": config.trainer,
        "stage": config.stage,
        "run_version": config.run.version,
        "source_replays": len(source_records),
        "source_manifest_sha256": source_identity.source_manifest_sha256,
        "replay_manifest_sha256": source_identity.replay_manifest_sha256,
        "top_teams_sha256": source_identity.top_teams_sha256,
        "episode_teams_sha256": source_identity.episode_teams_sha256,
        "source_selection": source_identity.source_selection,
        "model_config_fingerprint": model_config_fingerprint(resources.model_config),
        "exact_registry_fingerprint": (resources.model_config.resolved_registry_sha256),
        "public_catalog_fingerprint": resources.catalog.fingerprint,
        "input_contract_fingerprint": resources.input_contract.fingerprint,
        "engine_fact_producer_fingerprint": expected_engine_fact_fingerprint,
        "output_dir": str(output_dir),
        "dataset_dir": str(dataset_dir),
        "uses_validation_split": False,
        "uses_test_split": (
            config.stage == "evaluate" and config.evaluation.split == "test"
        ),
        "private_residual_training": True,
        "trainable_scope": config.trainable_scope.model_dump(mode="json"),
        "expected_target_deck_digest": config.data.expected_target_deck_digest,
        "outcome_weighting_declaration": (
            config.optimization.outcome_weights.model_dump(mode="json")
        ),
        "initialization": {
            "mode": config.initialization.mode,
            "pair_manifest_path": (
                None
                if config.initialization.pair_manifest_path is None
                else str(_initialization_pair_path(config))
            ),
            "pair_manifest_sha256": _initialization_pair_manifest_sha256(config),
            "family_private_transition_fingerprint": (
                None
                if config.initialization.family_private_transition is None
                else family_private_transition_fingerprint(
                    config.initialization.family_private_transition
                )
            ),
        },
    }
    if config.stage == "validate":
        return validation

    dataset: ReplayPretrainingDatasetManifest | None = None
    if config.stage in {"pipeline", "extract"}:
        dataset = extract_replay_dataset(
            config.data,
            repo_root=_REPO_ROOT,
            catalog_manifest_path=_path(config.public_deck_catalog.manifest_path),
            input_contract=resources.input_contract,
            public_catalog_fingerprint=resources.catalog.fingerprint,
            output_dir=dataset_dir,
            target_deck_digest=(
                config.data.expected_target_deck_digest
                or config.trainable_scope.target_deck_digest
            ),
            model_config_fingerprint=model_config_fingerprint(resources.model_config),
            exact_registry_fingerprint=(
                resources.model_config.resolved_registry_sha256
            ),
            event_contract_fingerprint=(
                resources.policy_identity.public_context_fingerprint
            ),
            sequence_contract_fingerprint=(
                resources.policy_identity.sequence_contract_fingerprint
            ),
            engine_fact_config=engine_fact_config,
            engine_fact_fingerprint=expected_engine_fact_fingerprint,
            route_expert_ids={
                route.deck_digest: route.expert_id
                for route in resources.model_config.exact_routes
            },
        )
    if config.stage == "extract":
        if dataset is None:
            raise RuntimeError("extract stage did not return a dataset")
        validation["uses_validation_split"] = dataset.split_examples("validation") > 0
        _prepare_run_directory(output_dir, config=config, validation=validation)
        return {
            **validation,
            "dataset_manifest": str(dataset_dir / "manifest.json"),
            "dataset_fingerprint": dataset.fingerprint,
            "examples": dataset.examples_committed,
            "parts": len(dataset.parts),
        }
    if dataset is None:
        dataset = load_pretraining_manifest(dataset_dir / "manifest.json")
    _validate_dataset_for_training(
        dataset,
        public_catalog_fingerprint=resources.catalog.fingerprint,
        input_contract_fingerprint=resources.input_contract.fingerprint,
        expected_target_deck_digest=config.data.expected_target_deck_digest,
        expected_model_config_fingerprint=model_config_fingerprint(
            resources.model_config
        ),
        expected_exact_registry_fingerprint=(
            resources.model_config.resolved_registry_sha256
        ),
        expected_event_contract_fingerprint=(
            resources.policy_identity.public_context_fingerprint
        ),
        expected_sequence_contract_fingerprint=(
            resources.policy_identity.sequence_contract_fingerprint
        ),
        expected_engine_fact_producer_fingerprint=(expected_engine_fact_fingerprint),
        require_validation=(
            config.evaluation.require_validation
            or (
                config.trainable_scope.mode == "exact_actor_private_v2"
                and not config.evaluation.train_only
            )
        ),
        require_train_only=config.evaluation.train_only,
    )
    validation["uses_validation_split"] = dataset.split_examples("validation") > 0
    if config.stage == "finalize":
        require_cuda_training_host()
        return _finalize_latest_training_checkpoint(
            config,
            resources=resources,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            validation=validation,
        )
    if config.stage != "evaluate":
        _prepare_run_directory(output_dir, config=config, validation=validation)
    require_cuda_training_host()
    if config.stage == "evaluate":
        return _evaluate_supervised_artifact(
            config,
            resources=resources,
            dataset=dataset,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            validation=validation,
        )
    return _train(
        config,
        resources=resources,
        dataset=dataset,
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        validation=validation,
    )


def _finalize_latest_training_checkpoint(
    config: SimpleStatelessPretrainingConfig,
    *,
    resources: Any,
    dataset: ReplayPretrainingDatasetManifest,
    dataset_dir: Path,
    output_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish an artifact from the latest durable temporal checkpoint."""
    if not is_temporal_pretraining_format(dataset.format):
        raise ValueError("partial-checkpoint finalization requires temporal data")
    sequence_config = resources.model_config.sequence
    if sequence_config is None:
        raise RuntimeError("temporal finalization has no sequence model")
    optimization = config.optimization
    _seed_everything(config.collection.seed)
    model, initialization = _initialize_policy(config, resources=resources)
    baseline_anchor = _baseline_policy_anchor_record(
        model,
        initialization=initialization,
        optimization=optimization,
    )
    validate_simple_stateless_parameter_report(
        simple_stateless_parameter_report(model, resources.model_config)
    )
    trainable_scope = _resolve_trainable_parameter_scope(
        model,
        config.trainable_scope,
    )
    _validate_target_dataset(
        dataset,
        dataset_dir=dataset_dir,
        model=model,
        trainable_scope=trainable_scope,
    )
    initial_private_state = _private_parameter_state(model)
    initial_temporal_fingerprints = _temporal_parameter_fingerprints(model)
    frozen_initial_fingerprint = _state_subset_fingerprint(
        model.state_dict(),
        trainable_scope.frozen_state_names,
    )
    optimizer = torch.optim.AdamW(
        trainable_scope.parameters,
        lr=optimization.learning_rate,
        betas=(optimization.adam_beta1, optimization.adam_beta2),
        eps=optimization.adam_epsilon,
        weight_decay=optimization.weight_decay,
    )
    config_fingerprint = _finalization_source_config_fingerprint(
        config,
        output_dir=output_dir,
    )
    cursor, route_examples, selection = _restore_training_state(
        output_dir,
        model=model,
        optimizer=optimizer,
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config_fingerprint,
        resume=optimization.resume,
    )
    if cursor.global_step <= 0:
        raise ValueError("partial-checkpoint finalization requires trained weights")
    held_out = dataset.split_examples("validation") > 0
    if held_out:
        _validate_held_out_partial_finalization_checkpoint(cursor, selection)
    elif not config.evaluation.train_only and cursor.epoch != 0:
        raise ValueError(
            "partial-checkpoint finalization is reserved for an unfinished first epoch"
        )
    _assert_frozen_state_unchanged(
        model,
        trainable_scope=trainable_scope,
        initial_fingerprint=frozen_initial_fingerprint,
    )
    steps_per_epoch = _steps_per_epoch(
        dataset,
        optimization,
        dataset_dir=dataset_dir,
        temporal_max_context_blocks=sequence_config.max_context_blocks,
    )
    episode_weighting = _build_episode_weighting_plan(
        dataset,
        dataset_dir=dataset_dir,
        optimization=optimization,
        optimizer_steps_per_epoch=steps_per_epoch,
    )
    if episode_weighting is None:
        raise ValueError(
            "temporal partial-checkpoint finalization requires episode weighting"
        )
    if config.evaluation.train_only:
        _validate_train_only_finalization_checkpoint(
            cursor,
            optimizer_steps_per_epoch=steps_per_epoch,
        )
        if trainable_scope.mode == "exact_actor_private_v2" and not route_examples:
            raise RuntimeError("final exact-private checkpoint has no routed rows")
        train_only_selection = SupervisedPolicyFinalEpochSelectionRecord(
            selected_optimizer_step=cursor.global_step,
            selected_epoch=cursor.epoch,
            final_optimizer_steps=cursor.global_step,
            selected_model_state_fingerprint=(
                canonical_model_state_fingerprint(model)
            ),
        )
        _require_temporal_parameter_updates(
            model,
            initial=initial_temporal_fingerprints,
        )
        artifact_manifest = _publish_final_artifact(
            output_dir,
            model=model,
            initial_private_state=initial_private_state,
            route_examples=route_examples,
            dataset=dataset,
            dataset_manifest_path=dataset_dir / "manifest.json",
            optimizer_steps=cursor.global_step,
            completed_epochs=cursor.epoch,
            input_contract_fingerprint=resources.input_contract.fingerprint,
            initialization=initialization,
            baseline_policy_anchor=baseline_anchor,
            trainable_scope=trainable_scope,
            frozen_initial_fingerprint=frozen_initial_fingerprint,
            selection=train_only_selection,
            outcome_weighting=episode_weighting.outcome_weighting,
            event_contract_fingerprint=dataset.event_contract_fingerprint,
            sequence_contract_fingerprint=dataset.sequence_contract_fingerprint,
        )
        return {
            **validation,
            "complete": True,
            "operation": "completed_epoch_train_only_finalization",
            "optimizer_steps": cursor.global_step,
            "completed_epochs": cursor.epoch,
            "artifact_manifest": str(
                output_dir / "artifacts" / "supervised_policy_manifest.json"
            ),
            "model_state_fingerprint": artifact_manifest.model_state_fingerprint,
        }
    selection_record: (
        SupervisedPolicySelectionRecord | SupervisedPolicyTrainMonitorSelectionRecord
    )
    if held_out:
        held_out_monitor = build_temporal_monitor_plan(
            dataset,
            dataset_dir=dataset_dir,
            maximum_targets=optimization.monitor_max_targets,
            target_chunk_decisions=optimization.target_chunk_decisions,
            split="validation",
        )
        if selection.monitor_fingerprint != held_out_monitor.fingerprint:
            raise ValueError("partial checkpoint held-out monitor changed")
        device = torch.device(config.device)
        model.to(device)
        monitor_started = time.perf_counter()
        print(
            "pretraining_finalize phase=validation_started "
            f"step={cursor.global_step} targets={held_out_monitor.target_count}",
            flush=True,
        )
        held_out_validation = evaluate_temporal_pretraining_monitor(
            model,
            dataset=dataset,
            dataset_dir=dataset_dir,
            monitor=held_out_monitor,
            batch_size=optimization.batch_size,
            target_chunk_decisions=optimization.target_chunk_decisions,
            max_context_blocks=sequence_config.max_context_blocks,
            device=device,
            trainable_scope=trainable_scope,
            maximum_batch_context_blocks=(
                optimization.maximum_batch_context_blocks
            ),
            maximum_batch_context_state_tokens=(
                optimization.maximum_batch_context_state_tokens
            ),
            maximum_batch_target_options=(
                optimization.maximum_batch_target_options
            ),
        )
        print(
            "pretraining_finalize phase=validation_complete "
            f"step={cursor.global_step} "
            "episode_nll="
            f"{held_out_validation.episode_normalized_policy_nll:.6f} "
            f"elapsed_seconds={time.perf_counter() - monitor_started:.2f}",
            flush=True,
        )
        selection = _update_best_checkpoint(
            output_dir,
            model=model,
            current=selection,
            cursor=cursor,
            completed_epoch=cursor.epoch,
            validation=held_out_validation,
            route_examples=route_examples,
            require_routed_examples=False,
            dataset_fingerprint=dataset.fingerprint,
            config_fingerprint=config_fingerprint,
        )
        if selection.selected_optimizer_step != cursor.global_step:
            raise RuntimeError(
                "requested latest partial checkpoint did not improve on the "
                "held-out selection"
            )
        selection = replace(selection, final_monitor=held_out_validation)
        _report_validation(
            output_dir,
            selection=selection,
            validation=held_out_validation,
        )
        selection_record = _complete_selection_record(
            selection,
            dataset=dataset,
            final_optimizer_steps=cursor.global_step,
        )
        artifact_route_examples = _load_selected_model(
            output_dir,
            model=model,
            selection=selection,
            dataset_fingerprint=dataset.fingerprint,
            config_fingerprint=config_fingerprint,
        )
        _require_temporal_parameter_updates(
            model,
            initial=initial_temporal_fingerprints,
        )
        artifact_manifest = _publish_final_artifact(
            output_dir,
            model=model,
            initial_private_state=initial_private_state,
            route_examples=artifact_route_examples,
            dataset=dataset,
            dataset_manifest_path=dataset_dir / "manifest.json",
            optimizer_steps=cursor.global_step,
            completed_epochs=cursor.epoch,
            input_contract_fingerprint=resources.input_contract.fingerprint,
            initialization=initialization,
            baseline_policy_anchor=baseline_anchor,
            trainable_scope=trainable_scope,
            frozen_initial_fingerprint=frozen_initial_fingerprint,
            selection=selection_record,
            outcome_weighting=episode_weighting.outcome_weighting,
            event_contract_fingerprint=dataset.event_contract_fingerprint,
            sequence_contract_fingerprint=dataset.sequence_contract_fingerprint,
        )
        _report_validation(
            output_dir,
            selection=selection,
            validation=held_out_validation,
        )
        return {
            **validation,
            "complete": True,
            "operation": "partial_checkpoint_held_out_finalization",
            "optimizer_steps": cursor.global_step,
            "completed_epochs": cursor.epoch,
            "validation": held_out_validation.model_dump(mode="json"),
            "artifact_manifest": str(
                output_dir / "artifacts" / "supervised_policy_manifest.json"
            ),
            "model_state_fingerprint": (artifact_manifest.model_state_fingerprint),
        }
    temporal_monitor = build_temporal_monitor_plan(
        dataset,
        dataset_dir=dataset_dir,
        maximum_targets=optimization.monitor_max_targets,
        target_chunk_decisions=optimization.target_chunk_decisions,
        split="train",
    )
    if (
        selection.initial_monitor is None
        or selection.monitor_fingerprint != temporal_monitor.fingerprint
    ):
        raise ValueError("partial checkpoint has no matching initial train monitor")
    device = torch.device(config.device)
    model.to(device)
    monitor_started = time.perf_counter()
    print(
        "pretraining_finalize phase=monitor_started "
        f"step={cursor.global_step} targets={temporal_monitor.target_count}",
        flush=True,
    )
    monitor_metrics = evaluate_temporal_pretraining_monitor(
        model,
        dataset=dataset,
        dataset_dir=dataset_dir,
        monitor=temporal_monitor,
        batch_size=optimization.batch_size,
        target_chunk_decisions=optimization.target_chunk_decisions,
        max_context_blocks=sequence_config.max_context_blocks,
        device=device,
        trainable_scope=trainable_scope,
        maximum_batch_context_blocks=optimization.maximum_batch_context_blocks,
        maximum_batch_context_state_tokens=(
            optimization.maximum_batch_context_state_tokens
        ),
        maximum_batch_target_options=(optimization.maximum_batch_target_options),
    )
    print(
        "pretraining_finalize phase=monitor_complete "
        f"step={cursor.global_step} "
        f"episode_nll={monitor_metrics.episode_normalized_policy_nll:.6f} "
        f"elapsed_seconds={time.perf_counter() - monitor_started:.2f}",
        flush=True,
    )
    selection = _update_best_checkpoint(
        output_dir,
        model=model,
        current=selection,
        cursor=cursor,
        completed_epoch=cursor.epoch,
        validation=monitor_metrics,
        route_examples=route_examples,
        require_routed_examples=False,
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config_fingerprint,
    )
    if selection.selected_optimizer_step != cursor.global_step:
        raise RuntimeError(
            "requested partial checkpoint did not improve on the initial monitor"
        )
    selection = replace(
        selection,
        final_monitor=monitor_metrics,
        stop_reason="maximum_steps",
    )
    selection_record = _complete_train_monitor_selection_record(
        selection,
        final_optimizer_steps=cursor.global_step,
    )
    artifact_route_examples = _load_selected_model(
        output_dir,
        model=model,
        selection=selection,
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config_fingerprint,
    )
    _require_temporal_parameter_updates(
        model,
        initial=initial_temporal_fingerprints,
    )
    artifact_manifest = _publish_final_artifact(
        output_dir,
        model=model,
        initial_private_state=initial_private_state,
        route_examples=artifact_route_examples,
        dataset=dataset,
        dataset_manifest_path=dataset_dir / "manifest.json",
        optimizer_steps=cursor.global_step,
        completed_epochs=cursor.epoch,
        input_contract_fingerprint=resources.input_contract.fingerprint,
        initialization=initialization,
        baseline_policy_anchor=baseline_anchor,
        trainable_scope=trainable_scope,
        frozen_initial_fingerprint=frozen_initial_fingerprint,
        selection=selection_record,
        outcome_weighting=episode_weighting.outcome_weighting,
        event_contract_fingerprint=dataset.event_contract_fingerprint,
        sequence_contract_fingerprint=dataset.sequence_contract_fingerprint,
    )
    _report_validation(
        output_dir,
        selection=selection,
        validation=monitor_metrics,
    )
    return {
        **validation,
        "complete": True,
        "operation": "partial_checkpoint_finalization",
        "optimizer_steps": cursor.global_step,
        "completed_epochs": cursor.epoch,
        "monitor": monitor_metrics.model_dump(mode="json"),
        "artifact_manifest": str(
            output_dir / "artifacts" / "supervised_policy_manifest.json"
        ),
        "model_state_fingerprint": artifact_manifest.model_state_fingerprint,
    }


def _validate_held_out_finalization_checkpoint(
    cursor: _TrainingCursor,
    selection: _SelectionState,
) -> None:
    """Require an epoch-boundary state with a durable held-out selection."""
    if cursor.epoch <= 0 or cursor.part_position != 0 or cursor.batch_position != 0:
        raise ValueError(
            "held-out temporal finalization requires a completed epoch checkpoint"
        )
    if (
        selection.initial_monitor is None
        or selection.initial_monitor.split != "validation"
        or selection.validation is None
        or selection.validation.split != "validation"
        or selection.selected_epoch is None
        or selection.selected_epoch > cursor.epoch
    ):
        raise ValueError("completed epoch checkpoint has no valid held-out selection")


def _validate_held_out_partial_finalization_checkpoint(
    cursor: _TrainingCursor,
    selection: _SelectionState,
) -> None:
    """Require a durable earlier selection before validating the latest state."""
    if cursor.epoch <= 0 or cursor.global_step <= 0:
        raise ValueError(
            "held-out partial finalization requires at least one completed epoch"
        )
    if (
        selection.initial_monitor is None
        or selection.initial_monitor.split != "validation"
        or selection.validation is None
        or selection.validation.split != "validation"
        or selection.selected_optimizer_step is None
        or selection.selected_optimizer_step > cursor.global_step
        or selection.selected_epoch is None
        or selection.selected_epoch > cursor.epoch
        or selection.monitor_fingerprint is None
    ):
        raise ValueError(
            "partial checkpoint has no valid earlier held-out selection"
        )


def _validate_train_only_finalization_checkpoint(
    cursor: _TrainingCursor,
    *,
    optimizer_steps_per_epoch: int,
) -> None:
    """Require an exact completed-epoch state for unvalidated final selection."""
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("train-only finalization has no optimizer steps per epoch")
    if (
        cursor.epoch <= 0
        or cursor.part_position != 0
        or cursor.batch_position != 0
        or cursor.global_step != cursor.epoch * optimizer_steps_per_epoch
    ):
        raise ValueError(
            "train-only finalization requires an exact completed epoch checkpoint"
        )


def _train(
    config: SimpleStatelessPretrainingConfig,
    *,
    resources: Any,
    dataset: ReplayPretrainingDatasetManifest,
    dataset_dir: Path,
    output_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    optimization = config.optimization
    _seed_everything(config.collection.seed)
    model, initialization = _initialize_policy(config, resources=resources)
    baseline_model, baseline_anchor = _copy_baseline_policy_anchor(
        model,
        initialization=initialization,
        optimization=optimization,
    )
    validate_simple_stateless_parameter_report(
        simple_stateless_parameter_report(model, resources.model_config)
    )
    trainable_scope = _resolve_trainable_parameter_scope(
        model,
        config.trainable_scope,
    )
    _validate_target_dataset(
        dataset,
        dataset_dir=dataset_dir,
        model=model,
        trainable_scope=trainable_scope,
    )
    initial_private_state = _private_parameter_state(model)
    initial_temporal_fingerprints = (
        _temporal_parameter_fingerprints(model)
        if is_temporal_pretraining_format(dataset.format)
        else {}
    )
    frozen_initial_fingerprint = _state_subset_fingerprint(
        model.state_dict(),
        trainable_scope.frozen_state_names,
    )
    device = torch.device(config.device)
    model.to(device)
    if baseline_model is not None:
        baseline_model.to(device)
    optimizer = torch.optim.AdamW(
        trainable_scope.parameters,
        lr=optimization.learning_rate,
        betas=(optimization.adam_beta1, optimization.adam_beta2),
        eps=optimization.adam_epsilon,
        weight_decay=optimization.weight_decay,
    )
    config_fingerprint = _config_fingerprint(config)
    cursor, route_examples, selection = _restore_training_state(
        output_dir,
        model=model,
        optimizer=optimizer,
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config_fingerprint,
        resume=optimization.resume,
    )
    _assert_frozen_state_unchanged(
        model,
        trainable_scope=trainable_scope,
        initial_fingerprint=frozen_initial_fingerprint,
    )
    sequence_config = resources.model_config.sequence
    steps_per_epoch = _steps_per_epoch(
        dataset,
        optimization,
        dataset_dir=dataset_dir,
        temporal_max_context_blocks=(
            None if sequence_config is None else sequence_config.max_context_blocks
        ),
    )
    phase_started = time.perf_counter()
    episode_weighting = _build_episode_weighting_plan(
        dataset,
        dataset_dir=dataset_dir,
        optimization=optimization,
        optimizer_steps_per_epoch=steps_per_epoch,
    )
    _publish_episode_weighting_audit(
        output_dir,
        plan=episode_weighting,
        dataset_fingerprint=dataset.fingerprint,
    )
    print(
        "pretraining_startup phase=episode_weighting_complete "
        f"elapsed_seconds={time.perf_counter() - phase_started:.2f}",
        flush=True,
    )
    uses_validation_selection = dataset.split_examples("validation") > 0
    temporal_monitor: TemporalMonitorPlan | None = None
    if (
        is_temporal_pretraining_format(dataset.format)
        and not config.evaluation.train_only
    ):
        if sequence_config is None:
            raise RuntimeError("temporal dataset has no sequence model")
        monitor_split: ReplaySplit = (
            "validation" if uses_validation_selection else "train"
        )
        phase_started = time.perf_counter()
        temporal_monitor = build_temporal_monitor_plan(
            dataset,
            dataset_dir=dataset_dir,
            maximum_targets=optimization.monitor_max_targets,
            target_chunk_decisions=optimization.target_chunk_decisions,
            split=monitor_split,
        )
        print(
            "pretraining_startup phase=monitor_plan_complete "
            f"targets={temporal_monitor.target_count} "
            "parts="
            f"{sum(bool(rows) for rows in temporal_monitor.rows_by_part)} "
            f"elapsed_seconds={time.perf_counter() - phase_started:.2f}",
            flush=True,
        )
        if (
            selection.monitor_fingerprint is not None
            and selection.monitor_fingerprint != temporal_monitor.fingerprint
        ):
            raise ValueError("resumed temporal split monitor changed")
        if selection.initial_monitor is None:
            phase_started = time.perf_counter()
            print(
                "pretraining_startup phase=initial_monitor_started "
                f"targets={temporal_monitor.target_count}",
                flush=True,
            )
            initial_monitor = evaluate_temporal_pretraining_monitor(
                model,
                dataset=dataset,
                dataset_dir=dataset_dir,
                monitor=temporal_monitor,
                batch_size=optimization.batch_size,
                target_chunk_decisions=(optimization.target_chunk_decisions),
                max_context_blocks=sequence_config.max_context_blocks,
                device=device,
                trainable_scope=trainable_scope,
                maximum_batch_context_blocks=(
                    optimization.maximum_batch_context_blocks
                ),
                maximum_batch_context_state_tokens=(
                    optimization.maximum_batch_context_state_tokens
                ),
                maximum_batch_target_options=(
                    optimization.maximum_batch_target_options
                ),
            )
            print(
                "pretraining_startup phase=initial_monitor_complete "
                f"episode_nll={initial_monitor.episode_normalized_policy_nll:.6f} "
                f"elapsed_seconds={time.perf_counter() - phase_started:.2f}",
                flush=True,
            )
            selection = _update_best_checkpoint(
                output_dir,
                model=model,
                current=selection,
                cursor=cursor,
                completed_epoch=cursor.epoch,
                validation=initial_monitor,
                route_examples=route_examples,
                require_routed_examples=False,
                dataset_fingerprint=dataset.fingerprint,
                config_fingerprint=config_fingerprint,
            )
            selection = replace(
                selection,
                initial_monitor=initial_monitor,
                final_monitor=initial_monitor,
                significant_best_nll=(initial_monitor.episode_normalized_policy_nll),
                monitor_fingerprint=temporal_monitor.fingerprint,
            )
            _publish_training_state(
                output_dir,
                model=model,
                optimizer=optimizer,
                cursor=cursor,
                route_examples=route_examples,
                selection=selection,
                dataset_fingerprint=dataset.fingerprint,
                config_fingerprint=config_fingerprint,
            )
    total_steps = _total_steps(
        optimizer_steps_per_epoch=steps_per_epoch,
        optimization=optimization,
    )
    started = time.perf_counter()
    rolling: list[_BatchMetrics] = []
    rolling_input_seconds: list[float] = []
    rolling_optimization_seconds: list[float] = []
    final_cursor = cursor
    stop_requested = False
    early_stopped = False
    train_part_indices = tuple(
        index
        for index, record in enumerate(dataset.parts)
        if _part_split_examples(dataset, record, "train") > 0
    )
    if not train_part_indices:
        raise ValueError("pretraining dataset has no train parts")
    for epoch in range(cursor.epoch, optimization.epochs):
        part_order = _part_order(
            len(train_part_indices),
            seed=optimization.shuffle_seed + epoch,
        )
        first_part = cursor.part_position if epoch == cursor.epoch else 0
        for part_position in range(first_part, len(part_order)):
            part_input_started = time.perf_counter()
            dataset_part_index = train_part_indices[part_order[part_position]]
            record = dataset.parts[dataset_part_index]
            part = load_pretraining_part(dataset_dir / "parts" / record.filename)
            split_indices = pretraining_split_indices(part, "train")
            batch_seed = (
                optimization.shuffle_seed
                + epoch * max(1, len(train_part_indices))
                + dataset_part_index
            )
            temporal_plans = (
                temporal_epoch_batches(
                    part,
                    target_decisions=optimization.batch_size,
                    target_chunk_decisions=(optimization.target_chunk_decisions),
                    max_context_blocks=(
                        resources.model_config.sequence.max_context_blocks
                        if resources.model_config.sequence is not None
                        else 0
                    ),
                    seed=batch_seed,
                    maximum_batch_context_blocks=(
                        optimization.maximum_batch_context_blocks
                    ),
                    maximum_batch_context_state_tokens=(
                        optimization.maximum_batch_context_state_tokens
                    ),
                    maximum_batch_target_options=(
                        optimization.maximum_batch_target_options
                    ),
                    selected_targets=tuple(int(index) for index in split_indices),
                )
                if is_temporal_pretraining_format(dataset.format)
                else ()
            )
            row_order = (
                np.asarray([], dtype=np.int64)
                if temporal_plans
                else split_indices[
                    _row_order(
                        int(split_indices.size),
                        seed=batch_seed,
                    )
                ]
            )
            batch_count = (
                len(temporal_plans)
                if temporal_plans
                else math.ceil(int(split_indices.size) / optimization.batch_size)
            )
            if batch_count <= 0:
                raise RuntimeError("train part contains no training rows")
            materialized_rows = (
                tuple(
                    iter_pretraining_rows(
                        part,
                        range(part.example_count),
                        catalog_fingerprint=dataset.public_catalog_fingerprint,
                        input_contract_fingerprint=(dataset.input_contract_fingerprint),
                    )
                )
                if temporal_plans
                else ()
            )
            pending_part_input_seconds = time.perf_counter() - part_input_started
            first_batch = (
                cursor.batch_position
                if epoch == cursor.epoch and part_position == first_part
                else 0
            )
            for batch_position in range(first_batch, batch_count):
                if (
                    optimization.maximum_steps is not None
                    and final_cursor.global_step >= optimization.maximum_steps
                ):
                    stop_requested = True
                    break
                temporal_plan = (
                    temporal_plans[batch_position] if temporal_plans else None
                )
                start = batch_position * optimization.batch_size
                indices = (
                    np.asarray(temporal_plan.target_indices, dtype=np.int64)
                    if temporal_plan is not None
                    else row_order[start : start + optimization.batch_size]
                )
                input_started = time.perf_counter()
                examples = (
                    tuple(materialized_rows[int(index)] for index in indices)
                    if temporal_plan is not None
                    else tuple(
                        iter_pretraining_rows(
                            part,
                            indices.tolist(),
                            catalog_fingerprint=dataset.public_catalog_fingerprint,
                            input_contract_fingerprint=(
                                dataset.input_contract_fingerprint
                            ),
                        )
                    )
                )
                context_examples = (
                    ()
                    if temporal_plan is None
                    else tuple(
                        materialized_rows[index]
                        for index in temporal_plan.context_indices
                    )
                )
                input_seconds = (
                    pending_part_input_seconds + time.perf_counter() - input_started
                )
                pending_part_input_seconds = 0.0
                learning_rate = _learning_rate(
                    optimization,
                    step=final_cursor.global_step,
                    total_steps=total_steps,
                )
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                optimization_started = time.perf_counter()
                metrics, matched = _train_batch(
                    model,
                    optimizer,
                    examples,
                    optimization=optimization,
                    device=device,
                    learning_rate=learning_rate,
                    trainable_scope=trainable_scope,
                    episode_weighting=episode_weighting,
                    baseline_model=baseline_model,
                    temporal_plan=temporal_plan,
                    context_examples=context_examples,
                )
                optimization_seconds = time.perf_counter() - optimization_started
                route_examples.update(matched)
                rolling.append(metrics)
                rolling_input_seconds.append(input_seconds)
                rolling_optimization_seconds.append(optimization_seconds)
                next_cursor = _next_cursor(
                    epoch=epoch,
                    part_position=part_position,
                    batch_position=batch_position,
                    batch_count=batch_count,
                    part_count=len(part_order),
                    global_step=final_cursor.global_step + 1,
                )
                final_cursor = next_cursor
                epoch_finished = final_cursor.epoch > epoch
                if (
                    final_cursor.global_step % optimization.log_interval_steps == 0
                    or final_cursor.global_step == 1
                ):
                    _report_status(
                        output_dir,
                        cursor=final_cursor,
                        metrics=rolling,
                        total_steps=total_steps,
                        dataset=dataset,
                        started=started,
                        episode_weighting=episode_weighting,
                        route_examples=route_examples,
                        input_seconds=rolling_input_seconds,
                        optimization_seconds=rolling_optimization_seconds,
                    )
                    rolling.clear()
                    rolling_input_seconds.clear()
                    rolling_optimization_seconds.clear()
                if (
                    uses_validation_selection
                    and temporal_monitor is None
                    and epoch_finished
                    and _should_validate_epoch(
                        completed_epoch=epoch + 1,
                        optimization=optimization,
                    )
                ):
                    validation_metrics = evaluate_pretraining_policy(
                        model,
                        dataset=dataset,
                        dataset_dir=dataset_dir,
                        split="validation",
                        batch_size=optimization.batch_size,
                        device=device,
                        trainable_scope=trainable_scope,
                        baseline_model=baseline_model,
                        outcome_weights=(
                            optimization.outcome_weights
                            if optimization.episode_normalized_weighting
                            else None
                        ),
                    )
                    selection = _update_best_checkpoint(
                        output_dir,
                        model=model,
                        current=selection,
                        cursor=final_cursor,
                        completed_epoch=epoch + 1,
                        validation=validation_metrics,
                        route_examples=route_examples,
                        require_routed_examples=(
                            trainable_scope.mode == "exact_actor_private_v2"
                        ),
                        dataset_fingerprint=dataset.fingerprint,
                        config_fingerprint=config_fingerprint,
                    )
                    _report_validation(
                        output_dir,
                        selection=selection,
                        validation=validation_metrics,
                    )
                if temporal_monitor is not None and epoch_finished:
                    sequence_config = resources.model_config.sequence
                    if sequence_config is None:
                        raise RuntimeError("temporal monitor lost sequence config")
                    monitor_metrics = evaluate_temporal_pretraining_monitor(
                        model,
                        dataset=dataset,
                        dataset_dir=dataset_dir,
                        monitor=temporal_monitor,
                        batch_size=optimization.batch_size,
                        target_chunk_decisions=(optimization.target_chunk_decisions),
                        max_context_blocks=sequence_config.max_context_blocks,
                        device=device,
                        trainable_scope=trainable_scope,
                        maximum_batch_context_blocks=(
                            optimization.maximum_batch_context_blocks
                        ),
                        maximum_batch_context_state_tokens=(
                            optimization.maximum_batch_context_state_tokens
                        ),
                        maximum_batch_target_options=(
                            optimization.maximum_batch_target_options
                        ),
                    )
                    selection = _update_best_checkpoint(
                        output_dir,
                        model=model,
                        current=selection,
                        cursor=final_cursor,
                        completed_epoch=epoch + 1,
                        validation=monitor_metrics,
                        route_examples=route_examples,
                        require_routed_examples=False,
                        dataset_fingerprint=dataset.fingerprint,
                        config_fingerprint=config_fingerprint,
                    )
                    selection, early_stopped = _advance_temporal_early_stopping(
                        selection,
                        monitor=monitor_metrics,
                        completed_epoch=epoch + 1,
                        optimization=optimization,
                    )
                    _report_validation(
                        output_dir,
                        selection=selection,
                        validation=monitor_metrics,
                    )
                if (
                    final_cursor.global_step % optimization.checkpoint_interval_steps
                    == 0
                    or epoch_finished
                ):
                    _publish_training_state(
                        output_dir,
                        model=model,
                        optimizer=optimizer,
                        cursor=final_cursor,
                        route_examples=route_examples,
                        selection=selection,
                        dataset_fingerprint=dataset.fingerprint,
                        config_fingerprint=config_fingerprint,
                    )
                if early_stopped:
                    stop_requested = True
                    break
            if stop_requested:
                break
        if stop_requested:
            break
    _publish_training_state(
        output_dir,
        model=model,
        optimizer=optimizer,
        cursor=final_cursor,
        route_examples=route_examples,
        selection=selection,
        dataset_fingerprint=dataset.fingerprint,
        config_fingerprint=config_fingerprint,
    )
    if (stop_requested and not early_stopped) or (
        not early_stopped and final_cursor.epoch < optimization.epochs
    ):
        return {
            **validation,
            "complete": False,
            "optimizer_steps": final_cursor.global_step,
            "total_steps": total_steps,
            "status_path": str(output_dir / "status.json"),
            "outcome_weighting": (
                None
                if episode_weighting is None
                else episode_weighting.outcome_weighting.model_dump(mode="json")
            ),
        }
    selection_record: (
        SupervisedPolicySelectionRecord
        | SupervisedPolicyTrainMonitorSelectionRecord
        | SupervisedPolicyFinalEpochSelectionRecord
        | None
    ) = None
    artifact_route_examples = route_examples
    if uses_validation_selection:
        selection_record = _complete_selection_record(
            selection,
            dataset=dataset,
            final_optimizer_steps=final_cursor.global_step,
        )
        artifact_route_examples = _load_selected_model(
            output_dir,
            model=model,
            selection=selection,
            dataset_fingerprint=dataset.fingerprint,
            config_fingerprint=config_fingerprint,
        )
        if (
            trainable_scope.mode == "exact_actor_private_v2"
            and not artifact_route_examples
        ):
            raise RuntimeError("selected exact-private checkpoint has no routed rows")
    elif config.evaluation.train_only:
        if (
            trainable_scope.mode == "exact_actor_private_v2"
            and not artifact_route_examples
        ):
            raise RuntimeError("final exact-private checkpoint has no routed rows")
        selection_record = SupervisedPolicyFinalEpochSelectionRecord(
            selected_optimizer_step=final_cursor.global_step,
            selected_epoch=final_cursor.epoch,
            final_optimizer_steps=final_cursor.global_step,
            selected_model_state_fingerprint=(canonical_model_state_fingerprint(model)),
        )
    elif temporal_monitor is not None:
        selection = replace(
            selection,
            stop_reason=(
                selection.stop_reason
                or (
                    "maximum_steps"
                    if optimization.maximum_steps is not None
                    and final_cursor.global_step >= optimization.maximum_steps
                    else "maximum_epochs"
                )
            ),
        )
        selection_record = _complete_train_monitor_selection_record(
            selection,
            final_optimizer_steps=final_cursor.global_step,
        )
        artifact_route_examples = _load_selected_model(
            output_dir,
            model=model,
            selection=selection,
            dataset_fingerprint=dataset.fingerprint,
            config_fingerprint=config_fingerprint,
        )
    if (baseline_model is None) != (baseline_anchor is None):
        raise RuntimeError("baseline policy anchor state is incomplete")
    if baseline_model is not None and baseline_anchor is not None:
        _freeze_baseline_policy(
            baseline_model,
            expected_fingerprint=(baseline_anchor.source_policy_model_fingerprint),
        )
    if initial_temporal_fingerprints:
        _require_temporal_parameter_updates(
            model,
            initial=initial_temporal_fingerprints,
        )
    artifact_manifest = _publish_final_artifact(
        output_dir,
        model=model,
        initial_private_state=initial_private_state,
        route_examples=artifact_route_examples,
        dataset=dataset,
        dataset_manifest_path=dataset_dir / "manifest.json",
        optimizer_steps=final_cursor.global_step,
        completed_epochs=final_cursor.epoch,
        input_contract_fingerprint=resources.input_contract.fingerprint,
        initialization=initialization,
        baseline_policy_anchor=baseline_anchor,
        trainable_scope=trainable_scope,
        frozen_initial_fingerprint=frozen_initial_fingerprint,
        selection=selection_record,
        outcome_weighting=(
            None if episode_weighting is None else episode_weighting.outcome_weighting
        ),
        event_contract_fingerprint=dataset.event_contract_fingerprint,
        sequence_contract_fingerprint=dataset.sequence_contract_fingerprint,
    )
    return {
        **validation,
        "complete": True,
        "optimizer_steps": final_cursor.global_step,
        "dataset_examples": dataset.examples_committed,
        "dataset_fingerprint": dataset.fingerprint,
        "artifact_manifest": str(
            output_dir / "artifacts" / "supervised_policy_manifest.json"
        ),
        "model_state_fingerprint": artifact_manifest.model_state_fingerprint,
        "outcome_weighting": (
            None
            if episode_weighting is None
            else episode_weighting.outcome_weighting.model_dump(mode="json")
        ),
        "selected_optimizer_step": (
            None
            if selection_record is None
            else selection_record.selected_optimizer_step
        ),
        "validation": (
            None
            if selection_record is None
            or isinstance(
                selection_record,
                SupervisedPolicyFinalEpochSelectionRecord,
            )
            else (
                selection_record.validation.model_dump(mode="json")
                if isinstance(
                    selection_record,
                    SupervisedPolicySelectionRecord,
                )
                else selection_record.best.model_dump(mode="json")
            )
        ),
    }


def _train_batch(
    model: SimpleStatelessPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    examples: tuple[ReplayPretrainingExample, ...],
    *,
    optimization: ReplayPretrainingOptimizationConfig,
    device: torch.device,
    learning_rate: float,
    trainable_scope: _TrainableParameterScope,
    episode_weighting: _EpisodeWeightingPlan | None,
    baseline_model: SimpleStatelessPolicyValueNet | None,
    temporal_plan: TemporalPretrainingBatchPlan | None = None,
    context_examples: tuple[ReplayPretrainingExample, ...] = (),
) -> tuple[_BatchMetrics, Counter[str]]:
    if not examples:
        raise ValueError("pretraining batch cannot be empty")
    batch = collate_simple_stateless_actor_rows(
        tuple(example.actor_row for example in examples),
        device=device,
        deduplicate_belief=True,
    )
    weights = _batch_loss_weights(
        examples,
        episode_normalized=optimization.episode_normalized_weighting,
        plan=episode_weighting,
        device=device,
    )
    routes = _resolve_batch_routes(
        batch.deck_signatures,
        model.config,
        device=device,
        trainable_scope=trainable_scope,
    )
    if (temporal_plan is None) != (not context_examples):
        raise ValueError("temporal pretraining context plan is incomplete")
    if temporal_plan is not None and baseline_model is not None:
        raise ValueError("temporal pretraining does not support a baseline anchor")
    evaluate_decode = (
        optimization.policy_loss_weight > 0.0
        or optimization.prefix_value_loss_weight > 0.0
    )
    evaluate_root = optimization.root_value_loss_weight > 0.0
    evaluate_belief = optimization.belief_loss_weight > 0.0
    evaluate_outcomes = evaluate_root or optimization.prefix_value_loss_weight > 0.0
    outcomes = (
        torch.tensor(
            [example.outcome for example in examples],
            dtype=torch.float32,
            device=device,
        )
        if evaluate_outcomes
        else None
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if temporal_plan is None:
            state = model.encode_observation_state(
                state=batch.states,
                unique_deck_card_ids=batch.unique_deck_card_ids,
                deck_counts=batch.deck_counts,
                deck_valid_mask=batch.deck_valid_mask,
                belief_summary=batch.belief_summary,
                route_plan=routes,
                allow_unrouted_rows=routes.allow_unrouted_rows,
            )
        else:
            context_batch = collate_simple_stateless_observation_rows(
                tuple(example.actor_row for example in context_examples),
                device=device,
                deduplicate_belief=True,
            )
            context_routes = _resolve_batch_routes(
                context_batch.deck_signatures,
                model.config,
                device=device,
                trainable_scope=trainable_scope,
            )
            event_deltas = tuple(
                example.actor_row.public_event_delta for example in context_examples
            )
            accepted_actions = tuple(
                example.accepted_action for example in context_examples
            )
            if any(delta is None for delta in event_deltas) or any(
                action is None for action in accepted_actions
            ):
                raise ValueError("temporal pretraining context payload is incomplete")
            snapshots = model.encode_observation_state(
                state=context_batch.states,
                unique_deck_card_ids=context_batch.unique_deck_card_ids,
                deck_counts=context_batch.deck_counts,
                deck_valid_mask=context_batch.deck_valid_mask,
                belief_summary=context_batch.belief_summary,
                route_plan=context_routes,
                allow_unrouted_rows=context_routes.allow_unrouted_rows,
            )
            temporal_contexts = model.replay_sequence(
                snapshots,
                collate_public_event_deltas(
                    cast(tuple[Any, ...], event_deltas),
                    device=device,
                ),
                collate_accepted_actions(
                    cast(tuple[Any, ...], accepted_actions),
                    device=device,
                ),
                sequence_offsets=temporal_plan.sequence_offsets,
                block_indices=torch.tensor(
                    temporal_plan.block_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
            conditioned = model.condition_sequence(
                snapshots,
                temporal_contexts,
            )
            state = select_simple_stateless_backbone_rows(
                conditioned,
                torch.tensor(
                    temporal_plan.target_row_indices,
                    dtype=torch.long,
                    device=device,
                ),
            )
        evaluation = None
        if evaluate_decode:
            option_embeddings = model.encode_legal_options(
                state,
                batch.options,
                route_plan=routes,
                allow_unrouted_rows=routes.allow_unrouted_rows,
            )
            evaluation = model.heads.teacher_forced(
                state.policy,
                state.opponent_belief,
                option_embeddings,
                batch.options,
                tuple(example.action for example in examples),
                route_plan=routes,
                evaluate_prefix_values=(optimization.prefix_value_loss_weight > 0.0),
            )
        baseline_evaluation = None
        if baseline_model is not None:
            with torch.no_grad():
                baseline_state = baseline_model.encode_observation_state(
                    state=batch.states,
                    unique_deck_card_ids=batch.unique_deck_card_ids,
                    deck_counts=batch.deck_counts,
                    deck_valid_mask=batch.deck_valid_mask,
                    belief_summary=batch.belief_summary,
                    route_plan=routes,
                    allow_unrouted_rows=routes.allow_unrouted_rows,
                )
                baseline_option_embeddings = baseline_model.encode_legal_options(
                    baseline_state,
                    batch.options,
                    route_plan=routes,
                    allow_unrouted_rows=routes.allow_unrouted_rows,
                )
                baseline_evaluation = baseline_model.heads.teacher_forced(
                    baseline_state.policy,
                    baseline_state.opponent_belief,
                    baseline_option_embeddings,
                    batch.options,
                    tuple(example.action for example in examples),
                    route_plan=routes,
                    evaluate_prefix_values=False,
                )
        root_wdl_logits = (
            model.heads.root_wdl_logits(
                state.value,
                state.opponent_belief,
                route_plan=routes,
            )
            if evaluate_root and uses_wdl_critic(model.config)
            else None
        )
        root_values = (
            wdl_value_from_logits(root_wdl_logits)
            if root_wdl_logits is not None
            else (
                model.heads.root_value(
                    state.value,
                    state.opponent_belief,
                    route_plan=routes,
                )
                if evaluate_root
                else None
            )
        )
        belief_logits = model.belief_logits(state) if evaluate_belief else None
    zero = torch.zeros((), dtype=torch.float32, device=device)
    if optimization.policy_loss_weight > 0.0:
        if evaluation is None:
            raise RuntimeError("policy objective did not evaluate the decoder")
        policy_rows = -evaluation.action_logprobs.float()
        policy_loss = (policy_rows * weights).sum()
    else:
        policy_loss = zero
    if evaluate_root:
        if root_values is None or outcomes is None:
            raise RuntimeError("root-value objective is missing its tensors")
        if root_wdl_logits is None:
            root_rows = torch.square(root_values.float() - outcomes)
        else:
            root_targets = two_hot_wdl_targets(outcomes)
            root_rows = -(
                root_targets * torch.log_softmax(root_wdl_logits.float(), dim=-1)
            ).sum(dim=-1)
        root_loss = (root_rows * weights).sum()
        root_value_mae = (torch.abs(root_values.float() - outcomes) * weights).sum()
    else:
        root_loss = zero
        root_value_mae = zero
    if optimization.prefix_value_loss_weight > 0.0:
        if evaluation is None or outcomes is None:
            raise RuntimeError("prefix-value objective is missing its tensors")
        prefix_targets = outcomes.unsqueeze(1).expand_as(evaluation.prefix_values)
        prefix_squared = torch.square(evaluation.prefix_values.float() - prefix_targets)
        token_counts = evaluation.token_mask.sum(dim=1).clamp_min(1)
        prefix_rows = (prefix_squared * evaluation.token_mask).sum(dim=1) / token_counts
        prefix_loss = (prefix_rows * weights).sum()
    else:
        prefix_loss = zero
    if evaluate_belief:
        if belief_logits is None:
            raise RuntimeError("belief objective is missing its logits")
        sparse_belief = build_sparse_belief_targets(
            tuple(example.opponent_deck for example in examples),
            tuple(Counter(dict(example.known_opponent_counts)) for example in examples),
            card_vocab_size=(model.backbone.input_encoder.card_encoder.num_card_ids),
            device=device,
        )
        belief_rows, belief_valid = normalized_sparse_belief_row_losses(
            belief_logits,
            sparse_belief,
        )
        valid_weights = weights * belief_valid
        weighted_belief_sum = (belief_rows * valid_weights).sum()
        belief_loss = (
            weighted_belief_sum
            if optimization.episode_normalized_weighting
            else weighted_belief_sum / valid_weights.sum().clamp_min(1.0e-8)
        )
    else:
        belief_loss = zero
    if optimization.baseline_policy_forward_kl_weight > 0.0:
        if evaluation is None or baseline_evaluation is None:
            raise RuntimeError("baseline policy KL is missing decoder evidence")
        count_mask = simple_count_first_rows(batch.options)
        option_prefix_mask = visited_option_prefix_mask(
            tuple(example.action for example in examples),
            max_counts=batch.options.max_counts,
            count_mask=count_mask,
        )
        anchor_kl = policy_forward_kl_losses(
            student_step_logits=evaluation.step_logits,
            teacher_step_logits=baseline_evaluation.step_logits,
            student_count_logits=evaluation.count_logits,
            teacher_count_logits=baseline_evaluation.count_logits,
            option_prefix_mask=option_prefix_mask,
            count_mask=count_mask,
            reference=evaluation.action_logprobs,
            rows=len(examples),
        )
        baseline_policy_forward_kl_loss = (anchor_kl.row_kl * weights).sum()
    else:
        if baseline_model is not None or baseline_evaluation is not None:
            raise RuntimeError("disabled baseline policy KL constructed an anchor")
        baseline_policy_forward_kl_loss = zero
    loss = (
        optimization.policy_loss_weight * policy_loss
        + optimization.baseline_policy_forward_kl_weight
        * baseline_policy_forward_kl_loss
        + optimization.root_value_loss_weight * root_loss
        + optimization.prefix_value_loss_weight * prefix_loss
        + optimization.belief_loss_weight * belief_loss
    )
    loss.backward()  # type: ignore[no-untyped-call]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        trainable_scope.parameters,
        optimization.maximum_gradient_norm,
    )
    if not bool(torch.isfinite(gradient_norm)):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("pretraining gradient norm is not finite")
    optimizer.step()
    matched = Counter[str]()
    by_module = {
        route.module_key: route.deck_digest for route in model.config.exact_routes
    }
    for group in routes.groups:
        matched[by_module[group.module_key]] += int(group.row_indices.numel())
    routed = sum(matched.values())
    metrics_tensor = torch.stack(
        (
            loss.detach().float(),
            policy_loss.detach().float(),
            baseline_policy_forward_kl_loss.detach().float(),
            root_loss.detach().float(),
            prefix_loss.detach().float(),
            belief_loss.detach().float(),
            root_value_mae.detach().float(),
            gradient_norm.detach().float(),
        )
    ).cpu()
    values = metrics_tensor.tolist()
    return (
        _BatchMetrics(
            loss=values[0],
            policy_loss=values[1],
            baseline_policy_forward_kl_loss=values[2],
            root_value_loss=values[3],
            prefix_value_loss=values[4],
            belief_loss=values[5],
            root_value_mae=values[6],
            gradient_norm=values[7],
            learning_rate=learning_rate,
            examples=len(examples),
            context_examples=len(context_examples),
            decode_tokens=(
                0 if evaluation is None else int(evaluation.token_mask.sum().item())
            ),
            routed_examples=routed,
        ),
        matched,
    )


def _publish_final_artifact(
    output_dir: Path,
    *,
    model: SimpleStatelessPolicyValueNet,
    initial_private_state: Mapping[str, Tensor],
    route_examples: Counter[str],
    dataset: ReplayPretrainingDatasetManifest,
    dataset_manifest_path: Path,
    optimizer_steps: int,
    completed_epochs: int,
    input_contract_fingerprint: str,
    initialization: SupervisedPolicyInitializationRecord,
    baseline_policy_anchor: SupervisedPolicyBaselineAnchorRecord | None,
    trainable_scope: _TrainableParameterScope,
    frozen_initial_fingerprint: str,
    selection: (
        SupervisedPolicySelectionRecord
        | SupervisedPolicyTrainMonitorSelectionRecord
        | SupervisedPolicyFinalEpochSelectionRecord
        | None
    ),
    outcome_weighting: SupervisedPolicyOutcomeWeightingRecord | None,
    event_contract_fingerprint: str | None,
    sequence_contract_fingerprint: str | None,
) -> SupervisedPolicyArtifactManifest:
    artifact_dir = output_dir / "artifacts"
    policy_path = artifact_dir / "policy_final.pt"
    manifest_path = artifact_dir / "supervised_policy_manifest.json"
    artifact_format: Literal[
        "simple-stateless-supervised-policy-v2",
        "simple-stateless-supervised-policy-v3",
        "simple-stateless-supervised-policy-v4",
    ] = SUPERVISED_POLICY_ARTIFACT_SCHEMA
    if is_temporal_pretraining_format(dataset.format):
        artifact_format = (
            WARMSTART_TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA
            if initialization.mode == "rl_pair"
            else TEMPORAL_SUPERVISED_POLICY_ARTIFACT_SCHEMA
        )
    state_fingerprint = canonical_model_state_fingerprint(model)
    scope_audit = _trainable_scope_audit(
        model,
        trainable_scope=trainable_scope,
        frozen_initial_fingerprint=frozen_initial_fingerprint,
    )
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            existing = SupervisedPolicyArtifactManifest.model_validate(
                json.load(handle)
            )
        if (
            existing.format != artifact_format
            or existing.model_state_fingerprint != state_fingerprint
            or existing.dataset_fingerprint != dataset.fingerprint
            or existing.initialization != initialization
            or existing.baseline_policy_anchor != baseline_policy_anchor
            or existing.outcome_weighting != outcome_weighting
            or existing.trainable_scope != scope_audit
            or existing.selection != selection
            or existing.event_contract_fingerprint != event_contract_fingerprint
            or existing.sequence_contract_fingerprint != sequence_contract_fingerprint
            or not policy_path.is_file()
            or policy_path.stat().st_size != existing.policy_size_bytes
            or file_sha256(policy_path) != existing.policy_sha256
        ):
            raise ValueError("existing supervised artifact differs from final state")
        return existing
    if policy_path.is_file():
        payload = torch.load(policy_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("model_state"), Mapping
        ):
            raise TypeError("orphan supervised policy payload is invalid")
        orphan_state = cast(Mapping[str, Tensor], payload["model_state"])
        if canonical_model_state_fingerprint(orphan_state) != state_fingerprint:
            raise ValueError("orphan supervised policy differs from final state")
        size_bytes = policy_path.stat().st_size
        policy_sha256 = file_sha256(policy_path)
    else:
        size_bytes, policy_sha256 = publish_torch_file(
            policy_path,
            supervised_policy_payload(
                model,
                artifact_format=artifact_format,
            ),
        )
    residual_training = _residual_training_records(
        model,
        initial_private_state=initial_private_state,
        route_examples=route_examples,
        trainable_parameter_names=frozenset(trainable_scope.parameter_names),
    )
    if route_examples and not any(
        record.policy_parameter_delta_l2 > 0.0 or record.value_parameter_delta_l2 > 0.0
        for record in residual_training
        if record.matched_examples > 0
    ):
        raise RuntimeError("matched private residuals received no parameter update")
    exact_registry = model.config.resolved_registry_sha256
    public_catalog = model.config.public_deck_catalog_fingerprint
    if exact_registry is None or public_catalog is None:
        raise RuntimeError("pretrained model is missing resolved identities")
    manifest = SupervisedPolicyArtifactManifest(
        format=artifact_format,
        policy_filename=policy_path.name,
        policy_size_bytes=size_bytes,
        policy_sha256=policy_sha256,
        model_state_fingerprint=state_fingerprint,
        model_config_fingerprint=model_config_fingerprint(model.config),
        exact_registry_fingerprint=exact_registry,
        public_catalog_fingerprint=public_catalog,
        input_contract_fingerprint=input_contract_fingerprint,
        event_contract_fingerprint=event_contract_fingerprint,
        sequence_contract_fingerprint=sequence_contract_fingerprint,
        dataset_manifest_sha256=file_sha256(dataset_manifest_path),
        dataset_fingerprint=dataset.fingerprint,
        optimizer_steps=optimizer_steps,
        completed_epochs=completed_epochs,
        initialization=initialization,
        baseline_policy_anchor=baseline_policy_anchor,
        outcome_weighting=outcome_weighting,
        residual_training=residual_training,
        trainable_scope=scope_audit,
        selection=selection,
    )
    atomic_write_bytes(
        manifest_path,
        json_payload(manifest.model_dump(mode="json")),
        overwrite=False,
    )
    return manifest


def _should_validate_epoch(
    *,
    completed_epoch: int,
    optimization: ReplayPretrainingOptimizationConfig,
) -> bool:
    return (
        completed_epoch % optimization.validation_interval_epochs == 0
        or completed_epoch == optimization.epochs
    )


def _update_best_checkpoint(
    output_dir: Path,
    *,
    model: SimpleStatelessPolicyValueNet,
    current: _SelectionState,
    cursor: _TrainingCursor,
    completed_epoch: int,
    validation: SupervisedPolicyEvaluationMetrics,
    route_examples: Counter[str],
    require_routed_examples: bool,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> _SelectionState:
    """Publish the immutable model when objective-aligned validation NLL improves."""
    candidate_nll = (
        validation.episode_normalized_policy_nll
        if validation.outcome_weighted_episode_policy_nll is None
        else validation.outcome_weighted_episode_policy_nll
    )
    current_nll = (
        None
        if current.validation is None
        else (
            current.validation.episode_normalized_policy_nll
            if current.validation.outcome_weighted_episode_policy_nll is None
            else current.validation.outcome_weighted_episode_policy_nll
        )
    )
    better = current_nll is None or candidate_nll < current_nll
    if not better:
        return replace(current, evaluations=current.evaluations + 1)
    selected_route_examples = _canonical_route_examples(route_examples)
    if require_routed_examples and not selected_route_examples:
        raise RuntimeError("best validation checkpoint has no routed examples")
    state_dir = output_dir / "training_state"
    filename = f"best_step_{cursor.global_step:08d}.pt"
    path = state_dir / filename
    model_state = {
        name: value.detach().cpu() for name, value in model.state_dict().items()
    }
    model_state_fingerprint = canonical_model_state_fingerprint(model_state)
    payload = {
        "format": "simple-stateless-pretraining-best-v2",
        "model_state": model_state,
        "model_state_fingerprint": model_state_fingerprint,
        "optimizer_step": cursor.global_step,
        "completed_epoch": completed_epoch,
        "validation": validation.model_dump(mode="json"),
        "route_examples": dict(selected_route_examples),
        "dataset_fingerprint": dataset_fingerprint,
        "config_fingerprint": config_fingerprint,
    }
    if path.is_file():
        existing = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(existing, Mapping)
            or existing.get("model_state_fingerprint")
            != payload["model_state_fingerprint"]
            or existing.get("optimizer_step") != cursor.global_step
            or existing.get("completed_epoch") != completed_epoch
            or existing.get("validation") != payload["validation"]
            or existing.get("route_examples") != payload["route_examples"]
            or existing.get("dataset_fingerprint") != dataset_fingerprint
            or existing.get("config_fingerprint") != config_fingerprint
        ):
            raise ValueError("existing best validation checkpoint changed")
        size_bytes = path.stat().st_size
        sha256 = file_sha256(path)
    else:
        size_bytes, sha256 = publish_torch_file(path, payload)
    return replace(
        current,
        evaluations=current.evaluations + 1,
        selected_optimizer_step=cursor.global_step,
        selected_epoch=completed_epoch,
        checkpoint_filename=filename,
        checkpoint_size_bytes=size_bytes,
        checkpoint_sha256=sha256,
        selected_model_state_fingerprint=model_state_fingerprint,
        validation=validation,
        route_examples=selected_route_examples,
    )


def _selection_payload(selection: _SelectionState) -> dict[str, Any]:
    return {
        "evaluations": selection.evaluations,
        "selected_optimizer_step": selection.selected_optimizer_step,
        "selected_epoch": selection.selected_epoch,
        "checkpoint_filename": selection.checkpoint_filename,
        "checkpoint_size_bytes": selection.checkpoint_size_bytes,
        "checkpoint_sha256": selection.checkpoint_sha256,
        "selected_model_state_fingerprint": (
            selection.selected_model_state_fingerprint
        ),
        "validation": (
            None
            if selection.validation is None
            else selection.validation.model_dump(mode="json")
        ),
        "route_examples": dict(selection.route_examples),
        "initial_monitor": (
            None
            if selection.initial_monitor is None
            else selection.initial_monitor.model_dump(mode="json")
        ),
        "final_monitor": (
            None
            if selection.final_monitor is None
            else selection.final_monitor.model_dump(mode="json")
        ),
        "significant_best_nll": selection.significant_best_nll,
        "stale_epochs": selection.stale_epochs,
        "stop_reason": selection.stop_reason,
        "monitor_fingerprint": selection.monitor_fingerprint,
    }


def _selection_from_payload(value: object) -> _SelectionState:
    if value is None:
        return _SelectionState()
    if not isinstance(value, Mapping):
        raise TypeError("pretraining selection state is invalid")
    raw_validation = value.get("validation")
    validation = (
        None
        if raw_validation is None
        else SupervisedPolicyEvaluationMetrics.model_validate(raw_validation)
    )
    raw_initial_monitor = value.get("initial_monitor")
    initial_monitor = (
        None
        if raw_initial_monitor is None
        else SupervisedPolicyEvaluationMetrics.model_validate(raw_initial_monitor)
    )
    raw_final_monitor = value.get("final_monitor")
    final_monitor = (
        None
        if raw_final_monitor is None
        else SupervisedPolicyEvaluationMetrics.model_validate(raw_final_monitor)
    )
    selection = _SelectionState(
        evaluations=int(value.get("evaluations", 0)),
        selected_optimizer_step=_optional_int(value.get("selected_optimizer_step")),
        selected_epoch=_optional_int(value.get("selected_epoch")),
        checkpoint_filename=_optional_str(value.get("checkpoint_filename")),
        checkpoint_size_bytes=_optional_int(value.get("checkpoint_size_bytes")),
        checkpoint_sha256=_optional_str(value.get("checkpoint_sha256")),
        selected_model_state_fingerprint=_optional_str(
            value.get("selected_model_state_fingerprint")
        ),
        validation=validation,
        route_examples=_route_examples_from_payload(value.get("route_examples", {})),
        initial_monitor=initial_monitor,
        final_monitor=final_monitor,
        significant_best_nll=_optional_float(value.get("significant_best_nll")),
        stale_epochs=int(value.get("stale_epochs", 0)),
        stop_reason=_optional_str(value.get("stop_reason")),
        monitor_fingerprint=_optional_str(value.get("monitor_fingerprint")),
    )
    populated = (
        selection.selected_optimizer_step,
        selection.selected_epoch,
        selection.checkpoint_filename,
        selection.checkpoint_size_bytes,
        selection.checkpoint_sha256,
        selection.selected_model_state_fingerprint,
        selection.validation,
    )
    if (
        selection.evaluations < 0
        or (
            selection.evaluations == 0
            and (
                any(item is not None for item in populated) or selection.route_examples
            )
        )
        or (selection.evaluations > 0 and any(item is None for item in populated))
        or selection.stale_epochs < 0
    ):
        raise ValueError("pretraining selection state is incomplete")
    return selection


def _validate_selection_checkpoint(
    state_dir: Path,
    *,
    selection: _SelectionState,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> None:
    if selection.evaluations == 0:
        return
    if selection.validation is None:
        raise ValueError("best validation checkpoint has no metrics")
    path = state_dir / str(selection.checkpoint_filename)
    if (
        Path(str(selection.checkpoint_filename)).name != selection.checkpoint_filename
        or not path.is_file()
        or path.stat().st_size != selection.checkpoint_size_bytes
        or file_sha256(path) != selection.checkpoint_sha256
    ):
        raise ValueError("best validation checkpoint failed fingerprint checks")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(payload, Mapping)
        or payload.get("format") != "simple-stateless-pretraining-best-v2"
        or payload.get("dataset_fingerprint") != dataset_fingerprint
        or payload.get("config_fingerprint") != config_fingerprint
        or payload.get("optimizer_step") != selection.selected_optimizer_step
        or payload.get("completed_epoch") != selection.selected_epoch
        or payload.get("model_state_fingerprint")
        != selection.selected_model_state_fingerprint
        or payload.get("validation") != selection.validation.model_dump(mode="json")
        or payload.get("route_examples") != dict(selection.route_examples)
    ):
        raise ValueError("best validation checkpoint identity changed")


def _load_selected_model(
    output_dir: Path,
    *,
    model: SimpleStatelessPolicyValueNet,
    selection: _SelectionState,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> Counter[str]:
    _validate_selection_checkpoint(
        output_dir / "training_state",
        selection=selection,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    path = output_dir / "training_state" / str(selection.checkpoint_filename)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("model_state"),
        Mapping,
    ):
        raise TypeError("best validation checkpoint has no model state")
    state = cast(Mapping[str, Tensor], payload["model_state"])
    if canonical_model_state_fingerprint(state) != payload.get(
        "model_state_fingerprint"
    ):
        raise ValueError("best validation model fingerprint changed")
    model.load_state_dict(state, strict=True)
    return Counter(dict(selection.route_examples))


def _complete_selection_record(
    selection: _SelectionState,
    *,
    dataset: ReplayPretrainingDatasetManifest,
    final_optimizer_steps: int,
) -> SupervisedPolicySelectionRecord:
    if (
        selection.selected_optimizer_step is None
        or selection.selected_epoch is None
        or selection.selected_model_state_fingerprint is None
        or selection.checkpoint_sha256 is None
        or selection.checkpoint_size_bytes is None
        or selection.validation is None
        or dataset.split_assignment_fingerprint is None
    ):
        raise RuntimeError("completed pretraining has no best validation checkpoint")
    return SupervisedPolicySelectionRecord(
        split_assignment_fingerprint=dataset.split_assignment_fingerprint,
        selected_optimizer_step=selection.selected_optimizer_step,
        selected_epoch=selection.selected_epoch,
        final_optimizer_steps=final_optimizer_steps,
        selected_model_state_fingerprint=(selection.selected_model_state_fingerprint),
        selected_checkpoint_sha256=selection.checkpoint_sha256,
        selected_checkpoint_size_bytes=selection.checkpoint_size_bytes,
        validation=selection.validation,
    )


def _advance_temporal_early_stopping(
    selection: _SelectionState,
    *,
    monitor: SupervisedPolicyEvaluationMetrics,
    completed_epoch: int,
    optimization: ReplayPretrainingOptimizationConfig,
) -> tuple[_SelectionState, bool]:
    """Apply the relative-improvement/patience contract after one epoch."""
    reference = selection.significant_best_nll
    if reference is None or reference <= 0.0:
        raise ValueError("temporal early stopping has no positive reference NLL")
    candidate = monitor.episode_normalized_policy_nll
    relative_improvement = (reference - candidate) / reference
    significant = relative_improvement >= optimization.monitor_relative_improvement
    stale_epochs = 0 if significant else selection.stale_epochs
    significant_best = candidate if significant else reference
    if completed_epoch >= optimization.minimum_epochs and not significant:
        stale_epochs += 1
    should_stop = (
        completed_epoch >= optimization.minimum_epochs
        and stale_epochs >= optimization.early_stopping_patience
    )
    return (
        replace(
            selection,
            final_monitor=monitor,
            significant_best_nll=significant_best,
            stale_epochs=stale_epochs,
            stop_reason="patience_exhausted" if should_stop else None,
        ),
        should_stop,
    )


def _complete_train_monitor_selection_record(
    selection: _SelectionState,
    *,
    final_optimizer_steps: int,
) -> SupervisedPolicyTrainMonitorSelectionRecord:
    """Bind the selected absolute minimum to the full monitor history."""
    if (
        selection.selected_optimizer_step is None
        or selection.selected_epoch is None
        or selection.selected_model_state_fingerprint is None
        or selection.checkpoint_sha256 is None
        or selection.checkpoint_size_bytes is None
        or selection.validation is None
        or selection.initial_monitor is None
        or selection.final_monitor is None
        or selection.monitor_fingerprint is None
        or selection.stop_reason
        not in {"patience_exhausted", "maximum_epochs", "maximum_steps"}
    ):
        raise RuntimeError("completed temporal pretraining has no selected monitor")
    return SupervisedPolicyTrainMonitorSelectionRecord(
        monitor_fingerprint=selection.monitor_fingerprint,
        selected_optimizer_step=selection.selected_optimizer_step,
        selected_epoch=selection.selected_epoch,
        final_optimizer_steps=final_optimizer_steps,
        selected_model_state_fingerprint=(selection.selected_model_state_fingerprint),
        selected_checkpoint_sha256=selection.checkpoint_sha256,
        selected_checkpoint_size_bytes=selection.checkpoint_size_bytes,
        initial=selection.initial_monitor,
        best=selection.validation,
        final=selection.final_monitor,
        stop_reason=selection.stop_reason,  # type: ignore[arg-type]
    )


def _report_validation(
    output_dir: Path,
    *,
    selection: _SelectionState,
    validation: SupervisedPolicyEvaluationMetrics,
) -> None:
    payload = {
        "phase": "validation",
        "evaluation_index": selection.evaluations,
        "selected_optimizer_step": selection.selected_optimizer_step,
        "selected_epoch": selection.selected_epoch,
        "metrics": validation.model_dump(mode="json"),
    }
    atomic_write_bytes(
        output_dir / "validation_latest.json",
        json_payload(payload),
        overwrite=True,
    )
    print(
        "validation "
        f"nll={validation.episode_normalized_policy_nll:.6f} "
        "outcome_weighted_nll="
        f"{validation.outcome_weighted_episode_policy_nll} "
        "anchor_kl="
        f"{validation.episode_normalized_baseline_policy_forward_kl} "
        f"exact={validation.exact_action_sequence_accuracy:.4f} "
        f"examples={validation.examples} "
        f"selected_step={selection.selected_optimizer_step}",
        flush=True,
    )


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("pretraining selection integer is invalid")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("pretraining selection float is invalid")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError("pretraining selection float is not finite")
    return normalized


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("pretraining selection string is invalid")
    return value


def _canonical_route_examples(
    route_examples: Mapping[str, int],
) -> tuple[tuple[str, int], ...]:
    """Freeze positive routed-example counts in deterministic order."""
    return tuple(
        sorted(
            (str(deck_digest), int(count))
            for deck_digest, count in route_examples.items()
            if count > 0
        )
    )


def _route_examples_from_payload(
    value: object,
) -> tuple[tuple[str, int], ...]:
    """Parse one fail-closed routed-example snapshot."""
    if not isinstance(value, Mapping):
        raise TypeError("pretraining selected route counts are invalid")
    parsed: dict[str, int] = {}
    for raw_digest, raw_count in value.items():
        if (
            not isinstance(raw_digest, str)
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count <= 0
        ):
            raise TypeError("pretraining selected route count is invalid")
        parsed[raw_digest] = raw_count
    return _canonical_route_examples(parsed)


def _publish_training_state(
    output_dir: Path,
    *,
    model: SimpleStatelessPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    cursor: _TrainingCursor,
    route_examples: Counter[str],
    selection: _SelectionState,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> None:
    state_dir = output_dir / "training_state"
    filename = f"state_step_{cursor.global_step:08d}.pt"
    path = state_dir / filename
    if path.exists():
        return
    latest_path = state_dir / "latest.json"
    previous_filename: str | None = None
    if latest_path.is_file():
        with latest_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        raw_previous = previous.get("filename")
        if isinstance(raw_previous, str):
            previous_filename = raw_previous
    size_bytes, sha256 = publish_torch_file(
        path,
        {
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "optimizer_state": optimizer.state_dict(),
            "cursor": cursor.__dict__,
            "route_examples": dict(route_examples),
            "selection": _selection_payload(selection),
            "dataset_fingerprint": dataset_fingerprint,
            "config_fingerprint": config_fingerprint,
        },
    )
    atomic_write_bytes(
        latest_path,
        json_payload(
            {
                "filename": filename,
                "size_bytes": size_bytes,
                "sha256": sha256,
                "cursor": cursor.__dict__,
                "dataset_fingerprint": dataset_fingerprint,
                "config_fingerprint": config_fingerprint,
                "previous_filename": previous_filename,
            }
        ),
        overwrite=True,
    )
    retained = {filename}
    if previous_filename is not None:
        retained.add(previous_filename)
    for stale in state_dir.glob("state_step_*.pt"):
        if stale.name not in retained:
            stale.unlink()
    selected_best = selection.checkpoint_filename
    for stale in state_dir.glob("best_step_*.pt"):
        if stale.name != selected_best:
            stale.unlink()


def _restore_training_state(
    output_dir: Path,
    *,
    model: SimpleStatelessPolicyValueNet,
    optimizer: torch.optim.Optimizer,
    dataset_fingerprint: str,
    config_fingerprint: str,
    resume: bool,
) -> tuple[_TrainingCursor, Counter[str], _SelectionState]:
    latest_path = output_dir / "training_state" / "latest.json"
    if not latest_path.is_file():
        return (_TrainingCursor(0, 0, 0, 0), Counter(), _SelectionState())
    if not resume:
        raise FileExistsError("pretraining state exists but resume is disabled")
    with latest_path.open(encoding="utf-8") as handle:
        latest = json.load(handle)
    if (
        latest.get("dataset_fingerprint") != dataset_fingerprint
        or latest.get("config_fingerprint") != config_fingerprint
    ):
        raise ValueError("pretraining resume identity changed")
    state_path = latest_path.parent / str(latest["filename"])
    if (
        not state_path.is_file()
        or state_path.stat().st_size != int(latest["size_bytes"])
        or file_sha256(state_path) != str(latest["sha256"])
    ):
        raise ValueError("pretraining checkpoint failed fingerprint checks")
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("pretraining checkpoint payload is invalid")
    model.load_state_dict(cast(Mapping[str, Tensor], payload["model_state"]))
    optimizer.load_state_dict(cast(dict[str, Any], payload["optimizer_state"]))
    cursor = _TrainingCursor(**cast(dict[str, int], payload["cursor"]))
    route_examples = Counter(
        {
            str(key): int(value)
            for key, value in cast(
                Mapping[str, int],
                payload.get("route_examples", {}),
            ).items()
        }
    )
    selection = _selection_from_payload(payload.get("selection"))
    _validate_selection_checkpoint(
        latest_path.parent,
        selection=selection,
        dataset_fingerprint=dataset_fingerprint,
        config_fingerprint=config_fingerprint,
    )
    return (cursor, route_examples, selection)


def _report_status(
    output_dir: Path,
    *,
    cursor: _TrainingCursor,
    metrics: Sequence[_BatchMetrics],
    total_steps: int,
    dataset: ReplayPretrainingDatasetManifest,
    started: float,
    episode_weighting: _EpisodeWeightingPlan | None,
    route_examples: Mapping[str, int],
    input_seconds: Sequence[float],
    optimization_seconds: Sequence[float],
) -> None:
    if not metrics:
        return
    if len(input_seconds) != len(metrics) or len(optimization_seconds) != len(metrics):
        raise ValueError("pretraining status timings are misaligned")
    elapsed = max(time.perf_counter() - started, 1.0e-6)
    progress = cursor.global_step / max(total_steps, 1)
    rate = cursor.global_step / elapsed
    means = {
        field: sum(getattr(metric, field) for metric in metrics) / len(metrics)
        for field in (
            "loss",
            "policy_loss",
            "baseline_policy_forward_kl_loss",
            "root_value_loss",
            "prefix_value_loss",
            "belief_loss",
            "root_value_mae",
            "gradient_norm",
            "learning_rate",
        )
    }
    status = {
        "phase": "training",
        "epoch": cursor.epoch,
        "part_position": cursor.part_position,
        "batch_position": cursor.batch_position,
        "optimizer_steps": cursor.global_step,
        "total_steps": total_steps,
        "progress": progress,
        "steps_per_second": rate,
        "eta_seconds": (
            None if rate <= 0.0 else max(total_steps - cursor.global_step, 0) / rate
        ),
        "examples_committed": dataset.examples_committed,
        "batch_target_decisions": sum(metric.examples for metric in metrics),
        "batch_context_blocks": sum(metric.context_examples for metric in metrics),
        "context_overlap_ratio": (
            sum(metric.context_examples for metric in metrics)
            / max(sum(metric.examples for metric in metrics), 1)
        ),
        "batch_decode_tokens": sum(metric.decode_tokens for metric in metrics),
        "batch_routed_examples": sum(metric.routed_examples for metric in metrics),
        "input_seconds_per_step": sum(input_seconds) / len(input_seconds),
        "optimization_seconds_per_step": (
            sum(optimization_seconds) / len(optimization_seconds)
        ),
        "measured_steps_per_second": (
            len(metrics) / max(sum(input_seconds) + sum(optimization_seconds), 1.0e-6)
        ),
        "route_examples": dict(sorted(route_examples.items())),
        "cuda_memory_allocated_bytes": torch.cuda.memory_allocated(),
        "cuda_memory_reserved_bytes": torch.cuda.memory_reserved(),
        "outcome_weighting": (
            None
            if episode_weighting is None
            else episode_weighting.outcome_weighting.model_dump(mode="json")
        ),
        **means,
    }
    atomic_write_bytes(
        output_dir / "status.json",
        json_payload(status),
        overwrite=True,
    )
    print(
        "training "
        f"step={cursor.global_step}/{total_steps} "
        f"loss={means['loss']:.5f} "
        f"policy={means['policy_loss']:.5f} "
        f"anchor_kl={means['baseline_policy_forward_kl_loss']:.5f} "
        f"value={means['root_value_loss']:.5f} "
        f"belief={means['belief_loss']:.5f} "
        f"routed={status['batch_routed_examples']}/"
        f"{status['batch_target_decisions']} "
        f"context={status['batch_context_blocks']} "
        f"measured_steps_per_second={status['measured_steps_per_second']:.3f} "
        f"input_seconds={status['input_seconds_per_step']:.3f} "
        f"optimization_seconds={status['optimization_seconds_per_step']:.3f}",
        flush=True,
    )


def _evaluate_supervised_artifact(
    config: SimpleStatelessPretrainingConfig,
    *,
    resources: Any,
    dataset: ReplayPretrainingDatasetManifest,
    dataset_dir: Path,
    output_dir: Path,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one selected artifact on an explicitly requested held-out split."""
    raw_manifest_path = config.evaluation.artifact_manifest_path
    if raw_manifest_path is None:
        raise ValueError("supervised evaluation artifact is not configured")
    manifest_path = _path(raw_manifest_path)
    exact_registry = resources.model_config.resolved_registry_sha256
    if exact_registry is None:
        raise ValueError("supervised evaluation requires an exact registry")
    artifact, model_state = load_supervised_policy_artifact(
        manifest_path,
        expected_model_config=resources.model_config,
        expected_exact_registry_fingerprint=exact_registry,
        expected_public_catalog_fingerprint=resources.catalog.fingerprint,
        expected_input_contract_fingerprint=(resources.input_contract.fingerprint),
    )
    if artifact.dataset_fingerprint != dataset.fingerprint:
        raise ValueError("evaluation dataset differs from the selected artifact")
    if (
        not isinstance(artifact.selection, SupervisedPolicySelectionRecord)
        or dataset.split_assignment_fingerprint is None
        or artifact.selection.split_assignment_fingerprint
        != dataset.split_assignment_fingerprint
    ):
        raise ValueError("evaluation split assignment differs from selection")
    scope_audit = artifact.trainable_scope
    if (
        scope_audit is None
        or scope_audit.mode != config.trainable_scope.mode
        or scope_audit.target_deck_digest != config.trainable_scope.target_deck_digest
    ):
        raise ValueError("evaluation artifact differs from configured BC scope")
    baseline_model: SimpleStatelessPolicyValueNet | None = None
    expected_anchor: SupervisedPolicyBaselineAnchorRecord | None = None
    if config.optimization.baseline_policy_forward_kl_weight > 0.0:
        baseline_model, baseline_initialization = _initialize_policy(
            config,
            resources=resources,
        )
        if baseline_initialization != artifact.initialization:
            raise ValueError("evaluation baseline differs from artifact initialization")
        expected_anchor = _baseline_policy_anchor_record(
            baseline_model,
            initialization=baseline_initialization,
            optimization=config.optimization,
        )
        if expected_anchor is None:
            raise RuntimeError("enabled evaluation baseline has no anchor record")
        _freeze_baseline_policy(
            baseline_model,
            expected_fingerprint=(expected_anchor.source_policy_model_fingerprint),
        )
    if expected_anchor != artifact.baseline_policy_anchor:
        raise ValueError("evaluation baseline objective differs from artifact")
    artifact_outcome_weighting = artifact.outcome_weighting
    if artifact_outcome_weighting is None:
        if (
            config.optimization.outcome_weights
            != ReplayPretrainingOutcomeWeightsConfig()
        ):
            raise ValueError(
                "legacy supervised artifact has no outcome-weighting provenance"
            )
    elif (
        not config.optimization.episode_normalized_weighting
        or artifact_outcome_weighting.multipliers != config.optimization.outcome_weights
    ):
        raise ValueError("evaluation outcome weights differ from artifact objective")
    split = config.evaluation.split
    if dataset.split_examples(split) <= 0:
        raise ValueError(f"pretraining dataset has no {split} examples")
    model = SimpleStatelessPolicyValueNet(
        resources.model_config,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(model_state, strict=True)
    trainable_scope = _resolve_trainable_parameter_scope(
        model,
        config.trainable_scope,
    )
    _validate_target_dataset(
        dataset,
        dataset_dir=dataset_dir,
        model=model,
        trainable_scope=trainable_scope,
    )
    device = torch.device(config.device)
    model.to(device)
    if baseline_model is not None:
        baseline_model.to(device)
    if is_temporal_pretraining_format(dataset.format):
        sequence_config = resources.model_config.sequence
        if sequence_config is None:
            raise RuntimeError("temporal evaluation has no sequence model")
        monitor = build_temporal_monitor_plan(
            dataset,
            dataset_dir=dataset_dir,
            maximum_targets=dataset.split_examples(split),
            target_chunk_decisions=(config.optimization.target_chunk_decisions),
            split=split,
        )
        metrics = evaluate_temporal_pretraining_monitor(
            model,
            dataset=dataset,
            dataset_dir=dataset_dir,
            monitor=monitor,
            batch_size=config.optimization.batch_size,
            target_chunk_decisions=(config.optimization.target_chunk_decisions),
            max_context_blocks=sequence_config.max_context_blocks,
            device=device,
            trainable_scope=trainable_scope,
            maximum_batch_context_blocks=(
                config.optimization.maximum_batch_context_blocks
            ),
            maximum_batch_context_state_tokens=(
                config.optimization.maximum_batch_context_state_tokens
            ),
            maximum_batch_target_options=(
                config.optimization.maximum_batch_target_options
            ),
        )
    else:
        metrics = evaluate_pretraining_policy(
            model,
            dataset=dataset,
            dataset_dir=dataset_dir,
            split=split,
            batch_size=config.optimization.batch_size,
            device=device,
            trainable_scope=trainable_scope,
            baseline_model=baseline_model,
            outcome_weights=(
                config.optimization.outcome_weights
                if config.optimization.episode_normalized_weighting
                else None
            ),
        )
    report = {
        "format": "simple-stateless-supervised-evaluation-v1",
        "artifact_manifest_path": str(manifest_path),
        "artifact_manifest_sha256": file_sha256(manifest_path),
        "model_state_fingerprint": artifact.model_state_fingerprint,
        "dataset_fingerprint": dataset.fingerprint,
        "split_assignment_fingerprint": (dataset.split_assignment_fingerprint),
        "selection": artifact.selection.model_dump(mode="json"),
        "metrics": metrics.model_dump(mode="json"),
    }
    report_path = (
        output_dir / "evaluations" / f"{split}-{artifact.model_state_fingerprint}.json"
    )
    if report_path.is_file():
        with report_path.open(encoding="utf-8") as handle:
            if json.load(handle) != report:
                raise ValueError("existing supervised evaluation report changed")
    else:
        atomic_write_bytes(
            report_path,
            json_payload(report),
            overwrite=False,
        )
    return {
        **validation,
        "complete": True,
        "evaluation_report": str(report_path),
        "evaluation": metrics.model_dump(mode="json"),
    }


def _prepare_run_directory(
    output_dir: Path,
    *,
    config: SimpleStatelessPretrainingConfig,
    validation: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "resolved_config.json"
    payload = {
        **validation,
        "config": config.model_dump(mode="json", exclude={"hydra"}),
        "config_fingerprint": _config_fingerprint(config),
    }
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError("pretraining run directory belongs to another config")
        return
    atomic_write_bytes(path, json_payload(payload), overwrite=False)


def _validate_dataset_for_training(
    dataset: ReplayPretrainingDatasetManifest,
    *,
    public_catalog_fingerprint: str,
    input_contract_fingerprint: str,
    expected_target_deck_digest: str | None,
    require_validation: bool,
    expected_model_config_fingerprint: str | None = None,
    expected_exact_registry_fingerprint: str | None = None,
    expected_event_contract_fingerprint: str | None = None,
    expected_sequence_contract_fingerprint: str | None = None,
    expected_engine_fact_producer_fingerprint: str | None = None,
    require_train_only: bool = False,
) -> None:
    if not dataset.complete:
        raise ValueError("pretraining requires a complete compact dataset")
    if (
        dataset.public_catalog_fingerprint != public_catalog_fingerprint
        or dataset.input_contract_fingerprint != input_contract_fingerprint
    ):
        raise ValueError("pretraining dataset input contract changed")
    if dataset.split_examples("train") <= 0:
        raise ValueError("pretraining dataset has no train examples")
    if is_temporal_pretraining_format(getattr(dataset, "format", "")) and (
        dataset.model_config_fingerprint != expected_model_config_fingerprint
        or dataset.exact_registry_fingerprint != expected_exact_registry_fingerprint
        or dataset.event_contract_fingerprint != expected_event_contract_fingerprint
        or dataset.sequence_contract_fingerprint
        != expected_sequence_contract_fingerprint
        or (
            dataset.format == TEMPORAL_PRETRAINING_SHARD_SCHEMA
            and dataset.engine_fact_producer_fingerprint
            != expected_engine_fact_producer_fingerprint
        )
    ):
        raise ValueError("temporal pretraining dataset contract changed")
    if (
        expected_target_deck_digest is not None
        and dataset.target_deck_digest != expected_target_deck_digest
    ):
        raise ValueError("pretraining dataset target deck identity changed")
    if require_validation and dataset.split_examples("validation") <= 0:
        raise ValueError("pretraining requires validation examples")
    if require_train_only and any(
        dataset.split_examples(split) > 0 for split in ("validation", "test")
    ):
        raise ValueError("train-only pretraining dataset contains held-out examples")


def _part_split_examples(
    dataset: ReplayPretrainingDatasetManifest,
    record: PretrainingPartRecord,
    split: ReplaySplit,
) -> int:
    """Return a part split count, treating legacy shards as train-only."""
    if record.split_examples:
        return record.split_examples[split]
    return record.examples if split == "train" else 0


def _build_episode_weighting_plan(
    dataset: ReplayPretrainingDatasetManifest,
    *,
    dataset_dir: Path,
    optimization: ReplayPretrainingOptimizationConfig,
    optimizer_steps_per_epoch: int,
) -> _EpisodeWeightingPlan | None:
    """Scan one shard at a time to derive the immutable train objective."""
    if not optimization.episode_normalized_weighting:
        return None
    counts: Counter[_EpisodeSeatKey] = Counter()
    outcomes: dict[_EpisodeSeatKey, float] = {}
    source_weight_sums: dict[_EpisodeSeatKey, float] = {}
    for record in dataset.parts:
        if _part_split_examples(dataset, record, "train") <= 0:
            continue
        if is_temporal_pretraining_format(dataset.format):
            arrays = load_pretraining_metadata_columns(
                dataset_dir / "parts" / record.filename,
                columns=(
                    "player_indices",
                    "split_codes",
                    "outcomes",
                    "example_weights",
                ),
            )
            indices = np.flatnonzero(
                np.asarray(arrays["split_codes"], dtype=np.int64)
                == REPLAY_SPLITS.index("train")
            )
        else:
            part = load_pretraining_part(dataset_dir / "parts" / record.filename)
            arrays = dict(part.arrays)
            indices = pretraining_split_indices(part, "train")
        episode_ids = np.asarray(arrays["episode_ids"])[indices]
        player_indices = np.asarray(arrays["player_indices"])[indices]
        row_outcomes = np.asarray(arrays["outcomes"])[indices]
        row_weights = np.asarray(arrays["example_weights"])[indices]
        for raw_episode_id, raw_player_index, raw_outcome, raw_weight in zip(
            episode_ids,
            player_indices,
            row_outcomes,
            row_weights,
            strict=True,
        ):
            episode_seat = (int(raw_episode_id), int(raw_player_index))
            outcome = float(raw_outcome)
            weight = float(raw_weight)
            optimization.outcome_weights.multiplier(outcome)
            if not math.isfinite(weight) or weight <= 0.0:
                raise ValueError("pretraining source example weight is invalid")
            previous = outcomes.setdefault(episode_seat, outcome)
            if previous != outcome:
                raise ValueError("one training episode-seat has conflicting outcomes")
            counts[episode_seat] += 1
            source_weight_sums[episode_seat] = (
                source_weight_sums.get(episode_seat, 0.0) + weight
            )
    return _episode_weighting_plan_from_counts(
        counts,
        episode_outcomes=outcomes,
        episode_source_weight_sums=source_weight_sums,
        outcome_weights=optimization.outcome_weights,
        expected_examples=dataset.split_examples("train"),
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
    )


def _episode_weighting_plan_from_counts(
    decision_counts: Mapping[_EpisodeSeatKey, int],
    *,
    episode_outcomes: Mapping[_EpisodeSeatKey, float],
    episode_source_weight_sums: Mapping[_EpisodeSeatKey, float],
    outcome_weights: ReplayPretrainingOutcomeWeightsConfig,
    expected_examples: int,
    optimizer_steps_per_epoch: int,
) -> _EpisodeWeightingPlan:
    """Build a fixed-scale outcome-weighted mean of episode-seat means."""
    canonical = tuple(
        sorted(
            ((int(key[0]), int(key[1])), int(count))
            for key, count in decision_counts.items()
        )
    )
    if (
        not canonical
        or any(
            episode_id < 0 or seat not in (0, 1) or count <= 0
            for (episode_id, seat), count in canonical
        )
        or len({key for key, _count in canonical}) != len(canonical)
    ):
        raise ValueError("train episode-seat decision counts are invalid")
    observed_examples = sum(count for _key, count in canonical)
    if observed_examples != expected_examples:
        raise ValueError(
            "train episode decision counts differ from the dataset manifest: "
            f"{observed_examples}!={expected_examples}"
        )
    if optimizer_steps_per_epoch <= 0:
        raise ValueError("episode weighting requires optimizer steps")
    episode_seats = {key for key, _count in canonical}
    if set(episode_outcomes) != episode_seats:
        raise ValueError("train episode-seat outcomes differ from decision counts")
    if set(episode_source_weight_sums) != episode_seats:
        raise ValueError(
            "train episode-seat source weights differ from decision counts"
        )
    canonical_stats: list[tuple[int, int, int, float, str]] = []
    canonical_outcomes: dict[_EpisodeSeatKey, float] = {}
    source_weight_sum = 0.0
    for episode_seat, count in canonical:
        outcome = float(episode_outcomes[episode_seat])
        outcome_weights.multiplier(outcome)
        source_sum = float(episode_source_weight_sums[episode_seat])
        if not math.isfinite(source_sum) or not math.isclose(
            source_sum,
            1.0,
            rel_tol=1.0e-6,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "source example weights are not normalized within episode-seat"
            )
        canonical_outcomes[episode_seat] = outcome
        source_weight_sum += source_sum
        canonical_stats.append(
            (
                episode_seat[0],
                episode_seat[1],
                count,
                outcome,
                source_sum.hex(),
            )
        )
    episode_count = len(canonical)
    outcome_weighting = supervised_outcome_weighting_record(
        split="train",
        multipliers=outcome_weights,
        episode_outcomes=canonical_outcomes,
        source_example_weight_sum=source_weight_sum,
    )
    stats_payload = json.dumps(
        {
            "episodes": canonical_stats,
            "outcome_weights": outcome_weights.model_dump(mode="json"),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _EpisodeWeightingPlan(
        decision_counts=MappingProxyType(dict(canonical)),
        episode_outcomes=MappingProxyType(canonical_outcomes),
        train_examples=observed_examples,
        train_episodes=episode_count,
        optimizer_steps_per_epoch=optimizer_steps_per_epoch,
        batch_loss_scale=optimizer_steps_per_epoch / episode_count,
        episode_stats_fingerprint=hashlib.sha256(
            _EPISODE_STATS_DOMAIN + stats_payload
        ).hexdigest(),
        outcome_weighting=outcome_weighting,
    )


def _batch_loss_weights(
    examples: Sequence[ReplayPretrainingExample],
    *,
    episode_normalized: bool,
    plan: _EpisodeWeightingPlan | None,
    device: torch.device,
) -> Tensor:
    """Return one fixed-scale loss coefficient per demonstrated decision."""
    if not examples:
        raise ValueError("pretraining batch cannot be empty")
    if not episode_normalized:
        if plan is not None:
            raise ValueError("decision-weighted training received an episode plan")
        return torch.full(
            (len(examples),),
            1.0 / len(examples),
            dtype=torch.float32,
            device=device,
        )
    if plan is None:
        raise ValueError("episode-normalized training has no global weighting plan")
    weights: list[float] = []
    raw_total = sum(plan.outcome_weighting.raw_episode_weight_sums.values())
    mean_multiplier = raw_total / plan.train_episodes
    for example in examples:
        episode_seat = (example.episode_id, example.player_index)
        count = plan.decision_counts.get(episode_seat)
        if count is None:
            raise ValueError(
                f"training row belongs to an uncounted episode-seat: {episode_seat}"
            )
        expected_outcome = plan.episode_outcomes[episode_seat]
        if example.outcome != expected_outcome:
            raise ValueError(
                "training row outcome differs from global episode-seat plan"
            )
        source_weight = float(example.example_weight)
        if not math.isfinite(source_weight) or source_weight <= 0.0:
            raise ValueError("training row source weight is invalid")
        effective_multiplier = (
            plan.outcome_weighting.multipliers.multiplier(example.outcome)
            / mean_multiplier
        )
        weights.append(plan.batch_loss_scale * source_weight * effective_multiplier)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _publish_episode_weighting_audit(
    output_dir: Path,
    *,
    plan: _EpisodeWeightingPlan | None,
    dataset_fingerprint: str,
) -> None:
    """Publish the derived estimator contract without materializing row data."""
    if plan is None:
        return
    payload = {
        "format": "simple-stateless-episode-weighting-audit-v3",
        "semantics": _EPISODE_WEIGHTING_SEMANTICS,
        "dataset_fingerprint": dataset_fingerprint,
        "train_examples": plan.train_examples,
        "train_episodes": plan.train_episodes,
        "optimizer_steps_per_epoch": plan.optimizer_steps_per_epoch,
        "batch_loss_scale": plan.batch_loss_scale,
        "episode_stats_fingerprint": plan.episode_stats_fingerprint,
        "outcome_weighting": plan.outcome_weighting.model_dump(mode="json"),
        "source_example_weight_semantics": (
            "positive_weights_sum_to_one_within_each_episode-seat"
        ),
        "objective": (
            "normalized_outcome_weighted_mean_episode_seat(mean_decision(loss))"
        ),
        "batch_normalization": "fixed_global_scale_no_batch_renormalization",
        "complete_epoch_identity": (
            "mean_step(batch_gradient) == gradient(objective) "
            "when evaluated at one fixed parameter state"
        ),
    }
    path = output_dir / "episode_weighting_audit.json"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            if json.load(handle) != payload:
                raise ValueError("episode weighting audit changed")
        return
    atomic_write_bytes(path, json_payload(payload), overwrite=False)


def _steps_per_epoch(
    dataset: ReplayPretrainingDatasetManifest,
    optimization: ReplayPretrainingOptimizationConfig,
    *,
    dataset_dir: Path,
    temporal_max_context_blocks: int | None,
) -> int:
    if is_temporal_pretraining_format(dataset.format):
        if temporal_max_context_blocks is None:
            raise ValueError("temporal step planning requires sequence geometry")
        steps = 0
        for part_index, record in enumerate(dataset.parts):
            if _part_split_examples(dataset, record, "train") <= 0:
                continue
            part = load_temporal_pretraining_geometry(
                dataset_dir / "parts" / record.filename
            )
            selected_targets = tuple(
                int(index) for index in pretraining_split_indices(part, "train")
            )
            steps += len(
                temporal_epoch_batches(
                    part,
                    target_decisions=optimization.batch_size,
                    target_chunk_decisions=optimization.target_chunk_decisions,
                    max_context_blocks=temporal_max_context_blocks,
                    seed=optimization.shuffle_seed + part_index,
                    selected_targets=selected_targets,
                    maximum_batch_context_blocks=(
                        optimization.maximum_batch_context_blocks
                    ),
                    maximum_batch_context_state_tokens=(
                        optimization.maximum_batch_context_state_tokens
                    ),
                    maximum_batch_target_options=(
                        optimization.maximum_batch_target_options
                    ),
                )
            )
        return steps
    return sum(
        math.ceil(
            _part_split_examples(dataset, record, "train") / optimization.batch_size
        )
        for record in dataset.parts
        if _part_split_examples(dataset, record, "train") > 0
    )


def _total_steps(
    *,
    optimizer_steps_per_epoch: int,
    optimization: ReplayPretrainingOptimizationConfig,
) -> int:
    steps = optimization.epochs * optimizer_steps_per_epoch
    if optimization.maximum_steps is not None:
        return min(steps, optimization.maximum_steps)
    return steps


def _learning_rate(
    config: ReplayPretrainingOptimizationConfig,
    *,
    step: int,
    total_steps: int,
) -> float:
    if config.warmup_steps > 0 and step < config.warmup_steps:
        return config.learning_rate * float(step + 1) / float(config.warmup_steps)
    decay_steps = max(total_steps - config.warmup_steps, 1)
    position = min(max(step - config.warmup_steps, 0), decay_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * position / decay_steps))
    ratio = (
        config.minimum_learning_rate_ratio
        + (1.0 - config.minimum_learning_rate_ratio) * cosine
    )
    return config.learning_rate * ratio


def _next_cursor(
    *,
    epoch: int,
    part_position: int,
    batch_position: int,
    batch_count: int,
    part_count: int,
    global_step: int,
) -> _TrainingCursor:
    if batch_position + 1 < batch_count:
        return _TrainingCursor(
            epoch,
            part_position,
            batch_position + 1,
            global_step,
        )
    if part_position + 1 < part_count:
        return _TrainingCursor(epoch, part_position + 1, 0, global_step)
    return _TrainingCursor(epoch + 1, 0, 0, global_step)


def _part_order(count: int, *, seed: int) -> npt.NDArray[np.int64]:
    generator = np.random.default_rng(seed)
    return generator.permutation(count)


def _row_order(count: int, *, seed: int) -> npt.NDArray[np.int64]:
    generator = np.random.default_rng(seed)
    return generator.permutation(count)


def _config_fingerprint(config: SimpleStatelessPretrainingConfig) -> str:
    payload = {
        "config": config.model_dump(mode="json", exclude={"hydra"}),
        "initialization_pair_manifest_sha256": (
            _initialization_pair_manifest_sha256(config)
        ),
    }
    if config.optimization.episode_normalized_weighting:
        payload["episode_weighting_semantics"] = _EPISODE_WEIGHTING_SEMANTICS
    return hashlib.sha256(
        _RUN_CONFIG_DOMAIN
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _finalization_source_config_fingerprint(
    config: SimpleStatelessPretrainingConfig,
    *,
    output_dir: Path,
) -> str:
    """Recover the exact training identity while allowing only stage=finalize."""
    resolved_path = output_dir / "resolved_config.json"
    if not resolved_path.is_file():
        raise FileNotFoundError(
            "pretraining finalization requires the original resolved config"
        )
    with resolved_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping) or not isinstance(
        payload.get("config"), Mapping
    ):
        raise TypeError("original pretraining resolved config is invalid")
    source_config = SimpleStatelessPretrainingConfig.model_validate(payload["config"])
    expected_finalizer = source_config.model_copy(update={"stage": "finalize"})
    if config.model_dump(mode="json", exclude={"hydra"}) != (
        expected_finalizer.model_dump(mode="json", exclude={"hydra"})
    ):
        raise ValueError("pretraining finalization config differs from the source run")
    fingerprint = payload.get("config_fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != _config_fingerprint(
        source_config
    ):
        raise ValueError("original pretraining config fingerprint changed")
    return fingerprint


def _initialize_policy(
    config: SimpleStatelessPretrainingConfig,
    *,
    resources: Any,
) -> tuple[
    SimpleStatelessPolicyValueNet,
    SupervisedPolicyInitializationRecord,
]:
    """Build the supervised model from random state or a complete RL pair."""
    if config.initialization.mode == "random":
        model = SimpleStatelessPolicyValueNet(resources.model_config)
        return (
            model,
            (
                SupervisedPolicyInitializationRecord(
                    mode="random",
                    random_seed=config.collection.seed,
                    initial_model_state_fingerprint=(
                        canonical_model_state_fingerprint(model)
                    ),
                )
                if config.data.temporal_sequence
                else SupervisedPolicyInitializationRecord(mode="random")
            ),
        )
    pair_path = _initialization_pair_path(config)
    source = load_stateless_checkpoint_pair(pair_path)
    transition = config.initialization.family_private_transition
    if transition is None and source.model_config_value != resources.model_config:
        raise ValueError("pretraining source RL pair changed model topology")
    identity = source.pair.identity
    expected_source_decks = (
        tuple(sorted(clone.deck_digest for clone in transition.route_clones))
        if transition is not None
        else tuple(
            sorted(route.deck_digest for route in resources.model_config.exact_routes)
        )
    )
    expected_identity = (
        (
            transition.source_registry_sha256
            if transition is not None
            else str(resources.model_config.resolved_registry_sha256)
        ),
        resources.catalog.fingerprint,
        resources.input_contract.fingerprint,
        expected_source_decks,
    )
    actual_identity = (
        identity.exact_registry_fingerprint,
        identity.public_deck_catalog_fingerprint,
        identity.input_contract_fingerprint,
        identity.active_exact_deck_digests,
    )
    if actual_identity != expected_identity:
        raise ValueError("pretraining source RL pair changed routed policy identity")
    topology_record: SupervisedPolicyTopologyTransitionRecord | None = None
    if transition is None:
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        model.load_state_dict(source.model_state, strict=True)
        initial_model_state_fingerprint = None
    else:
        model = SimpleStatelessPolicyValueNet(
            resources.model_config,
            load_static_features=False,
            initialize=False,
        )
        family_private = model.backbone.family_private
        if family_private is None:
            raise ValueError("family-private transition target has no family bank")
        family_private.initialize_appended_layers()
        adapters = model.backbone.v2_adapters
        if adapters is None:
            raise ValueError("family-private transition target has no exact adapters")
        target_routes = {
            route.deck_digest: route
            for route in resources.model_config.exact_routes
        }
        initialized_exact_keys = tuple(
            target_routes[item.target_deck_digest].module_key
            for item in transition.route_initializations
            if item.target_deck_digest in target_routes
        )
        if len(initialized_exact_keys) != len(transition.route_initializations):
            raise ValueError(
                "family-private exact initialization is absent from target registry"
            )
        adapters.initialize_exact_routes(initialized_exact_keys)
        inert_names = tuple(
            sorted(
                f"backbone.family_private.{name}"
                for name, _parameter in (
                    family_private.inert_output_named_parameters()
                )
            )
        )
        result = migrate_family_private_from_pair(
            source=source,
            target_model=model,
            target_config=resources.model_config,
            declaration=transition,
            inert_output_tensor_names=inert_names,
        )
        audit = result.audit
        initial_model_state_fingerprint = audit.target_state_fingerprint
        topology_record = SupervisedPolicyTopologyTransitionRecord(
            declaration_fingerprint=audit.declaration_fingerprint,
            source_state_fingerprint=audit.source_state_fingerprint,
            target_state_fingerprint=audit.target_state_fingerprint,
            state_mapping_fingerprint=audit.state_mapping_fingerprint,
            initialized_state_fingerprint=audit.initialized_state_fingerprint,
            source_tensors=audit.source_tensors,
            inherited_target_tensors=audit.inherited_target_tensors,
            cloned_upper_target_tensors=audit.cloned_upper_target_tensors,
            initialized_exact_target_tensors=(
                audit.initialized_exact_target_tensors
            ),
            initialized_appended_target_tensors=(
                audit.initialized_appended_target_tensors
            ),
            initialized_tensors=audit.initialized_tensors,
        )
    initialization = SupervisedPolicyInitializationRecord(
        mode="rl_pair",
        source_pair_manifest_path=str(source.pair.pair_manifest_path),
        source_pair_manifest_sha256=source.pair.pair_manifest_sha256,
        source_pair_version=source.pair.version,
        source_policy_path=str(source.pair.policy_path),
        source_policy_sha256=source.pair.policy_sha256,
        source_policy_model_fingerprint=source.pair.policy_model_fingerprint,
        initial_model_state_fingerprint=initial_model_state_fingerprint,
        topology_transition=topology_record,
    )
    return (model, initialization)


def _copy_baseline_policy_anchor(
    model: SimpleStatelessPolicyValueNet,
    *,
    initialization: SupervisedPolicyInitializationRecord,
    optimization: ReplayPretrainingOptimizationConfig,
) -> tuple[
    SimpleStatelessPolicyValueNet | None,
    SupervisedPolicyBaselineAnchorRecord | None,
]:
    """Copy and freeze the exact initialization policy when KL is enabled."""
    record = _baseline_policy_anchor_record(
        model,
        initialization=initialization,
        optimization=optimization,
    )
    if record is None:
        return (None, None)
    baseline = SimpleStatelessPolicyValueNet(
        model.config,
        load_static_features=False,
        initialize=False,
    )
    baseline.load_state_dict(model.state_dict(), strict=True)
    _freeze_baseline_policy(
        baseline, expected_fingerprint=record.source_policy_model_fingerprint
    )
    return (baseline, record)


def _temporal_parameter_fingerprints(
    model: SimpleStatelessPolicyValueNet,
) -> dict[str, str]:
    """Fingerprint causal path groups without retaining another model copy."""
    groups = {
        "event": "sequence.event_encoder.",
        "state": "sequence.state_encoder.",
        "action": "sequence.action_encoder.",
        "core": "sequence.core.",
        "residual": "sequence.",
    }
    state = model.state_dict()
    fingerprints: dict[str, str] = {}
    for label, prefix in groups.items():
        selected = {
            name: tensor
            for name, tensor in state.items()
            if name.startswith(prefix)
            and (
                label != "residual"
                or any(
                    segment in name
                    for segment in (
                        "policy_residual",
                        "value_residual",
                        "belief_residual",
                    )
                )
            )
        }
        if not selected:
            raise ValueError(f"temporal parameter group is empty: {label}")
        fingerprints[label] = canonical_model_state_fingerprint(selected)
    return fingerprints


def _require_temporal_parameter_updates(
    model: SimpleStatelessPolicyValueNet,
    *,
    initial: Mapping[str, str],
) -> None:
    """Require the selected BC state to have trained every causal path."""
    final = _temporal_parameter_fingerprints(model)
    unchanged = tuple(
        label for label, fingerprint in initial.items() if final[label] == fingerprint
    )
    if unchanged:
        raise RuntimeError(
            "selected temporal BC state left causal paths unchanged: "
            + ", ".join(unchanged)
        )


def _baseline_policy_anchor_record(
    model: SimpleStatelessPolicyValueNet,
    *,
    initialization: SupervisedPolicyInitializationRecord,
    optimization: ReplayPretrainingOptimizationConfig,
) -> SupervisedPolicyBaselineAnchorRecord | None:
    """Bind an enabled forward-KL objective to the loaded RL source bytes."""
    weight = optimization.baseline_policy_forward_kl_weight
    if weight <= 0.0:
        return None
    if (
        initialization.mode != "rl_pair"
        or initialization.source_pair_manifest_sha256 is None
        or initialization.source_policy_sha256 is None
        or initialization.source_policy_model_fingerprint is None
    ):
        raise ValueError("baseline policy KL has no complete RL initialization")
    observed_fingerprint = canonical_model_state_fingerprint(model)
    if observed_fingerprint != initialization.source_policy_model_fingerprint:
        raise ValueError("baseline policy differs from RL initialization")
    return SupervisedPolicyBaselineAnchorRecord(
        weight=weight,
        source_pair_manifest_sha256=(initialization.source_pair_manifest_sha256),
        source_policy_sha256=initialization.source_policy_sha256,
        source_policy_model_fingerprint=(
            initialization.source_policy_model_fingerprint
        ),
    )


def _freeze_baseline_policy(
    model: SimpleStatelessPolicyValueNet,
    *,
    expected_fingerprint: str,
) -> None:
    """Make the anchor immutable and verify its complete initial state."""
    if canonical_model_state_fingerprint(model) != expected_fingerprint:
        raise ValueError("copied baseline policy fingerprint changed")
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("baseline policy anchor has a trainable parameter")


def _initialization_pair_path(
    config: SimpleStatelessPretrainingConfig,
) -> Path:
    path = config.initialization.pair_manifest_path
    if path is None:
        raise ValueError("RL-pair initialization is missing its manifest")
    return _path(path)


def _initialization_pair_manifest_sha256(
    config: SimpleStatelessPretrainingConfig,
) -> str | None:
    if config.initialization.mode == "random":
        return None
    path = _initialization_pair_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"RL initialization pair does not exist: {path}")
    return file_sha256(path)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _path(path: Path) -> Path:
    return path if path.is_absolute() else (_REPO_ROOT / path).resolve()


__all__ = ["run_simple_stateless_pretraining"]
