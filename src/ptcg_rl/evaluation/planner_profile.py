"""Orchestration for one deployment-aligned integrated planner campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from ptcg_rl.agent.search.root_information_tensorizer import (
    ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT,
)
from ptcg_rl.belief.runtime_identity import belief_runtime_fingerprint
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context.belief import OpponentBeliefFeatureProducer
from ptcg_rl.engine.native_planning_session_payload import (
    NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR,
    native_planning_session_abi_fingerprint,
    native_planning_session_schema_fingerprint,
)
from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_config import (
    REQUIRED_PLANNER_PROFILE_SHAPES,
    IntegratedPlannerProfileConfig,
    PlannerProfilePointConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusReader,
)
from ptcg_rl.evaluation.planner_profile_manifest import (
    validate_planner_profile_corpus_manifest,
)
from ptcg_rl.evaluation.planner_profile_metrics import (
    PlannerProfilePointSummary,
    PlannerProfileSink,
    select_profile_budget,
)
from ptcg_rl.evaluation.planner_profile_package_assets import (
    validate_planner_profile_package_assets,
)
from ptcg_rl.evaluation.planner_profile_records import PlannerProfileRunRecord
from ptcg_rl.evaluation.planner_profile_report import (
    build_campaign_summary,
    selected_runtime_artifact,
)
from ptcg_rl.evaluation.planner_profile_runner import (
    ProductionPlannerProfilePointRunner,
)
from ptcg_rl.evaluation.planner_profile_writer import (
    PlannerProfileWriter,
    write_profile_summary,
)
from ptcg_rl.rl.model_fingerprint import constructed_checkpoint_fingerprint

_POINT_ORDER_DOMAIN = b"ptcg-rl/planner-profile-point-order/v1\x00"


def run_integrated_planner_profile(
    config: IntegratedPlannerProfileConfig,
) -> dict[str, Any]:
    """Execute planner off/on and candidate budgets as one direct campaign."""
    _validate_profile_host(config)
    replay_assets = validate_profile_act_time_replays(config)
    checkpoint_sha256 = _require_fingerprint(
        config.checkpoint_path,
        expected=config.expected_checkpoint_sha256,
        label="checkpoint",
    )
    model_fingerprint, checkpoint_policy_version = constructed_checkpoint_fingerprint(
        config.checkpoint_path,
        migration_seed=config.model_migration_seed,
    )
    if model_fingerprint != config.expected_model_fingerprint:
        raise ValueError("profile constructed model fingerprint differs from config")
    if checkpoint_policy_version != config.policy_version:
        raise ValueError("profile checkpoint publication version differs from config")
    native_library_sha256 = _require_fingerprint(
        config.native_library_path,
        expected=config.expected_native_library_sha256,
        label="native library",
    )
    native_abi_fingerprint = native_planning_session_abi_fingerprint(
        NATIVE_PLANNING_SESSION_ABI_DESCRIPTOR
    )
    if native_abi_fingerprint != config.expected_native_abi_fingerprint:
        raise ValueError("profile native ABI fingerprint differs from config")
    native_schema_fingerprint = native_planning_session_schema_fingerprint()
    if native_schema_fingerprint != config.expected_native_schema_fingerprint:
        raise ValueError("profile native schema fingerprint differs from config")
    belief_prior_sha256, belief_fingerprint = validate_profile_belief_runtime(config)
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
    runner = ProductionPlannerProfilePointRunner(
        config,
        package_validation=package_validation,
    )
    oracle_precompute = runner.prepare_oracles()
    point_summaries: list[PlannerProfilePointSummary] = []
    run_records: list[PlannerProfileRunRecord] = []
    scenario_supports: dict[str, Mapping[tuple[int, str], str]] = {}
    oracle_keys: dict[str, frozenset[tuple[int, str]]] = {}
    staging_dir = _prepare_campaign_staging(config.output_dir)
    with PlannerProfileWriter(
        staging_dir,
        rows_per_shard=config.rows_per_shard,
        compression=config.compression,
    ) as writer:
        for point in _measurement_point_order(config):
            sink = PlannerProfileSink(
                campaign_id=config.campaign_id,
                point=point,
                writer=writer,
            )
            run_record = runner.execute(point, sink)
            point_summary = sink.summary()
            _validate_run_record(
                run_record,
                config=config,
                point=point,
                summary=point_summary,
                oracle_elapsed_ms=sink.oracle_elapsed_ms,
                native_library_sha256=native_library_sha256,
            )
            writer.append_run(run_record)
            point_summaries.append(point_summary)
            run_records.append(run_record)
            scenario_supports[point.point_id] = sink.scenario_supports
            oracle_keys[point.point_id] = sink.oracle_keys
        writer.close()
        part_counts = dict(writer.part_counts)
    _validate_campaign_identity(run_records, config=config)
    _validate_paired_decision_evidence(
        config,
        scenario_supports=scenario_supports,
        oracle_keys=oracle_keys,
    )
    selection = select_profile_budget(
        config.resource_contract,
        point_summaries=point_summaries,
        run_records=run_records,
    )
    summary = build_campaign_summary(
        config,
        checkpoint_sha256=checkpoint_sha256,
        model_fingerprint=model_fingerprint,
        native_library_sha256=native_library_sha256,
        native_abi_fingerprint=native_abi_fingerprint,
        native_schema_fingerprint=native_schema_fingerprint,
        belief_prior_sha256=belief_prior_sha256,
        belief_runtime_fingerprint=belief_fingerprint,
        act_time_replay_assets=replay_assets,
        corpus_rows=corpus.rows,
        corpus_shape_counts=shape_counts,
        point_summaries=point_summaries,
        run_records=run_records,
        selection=selection,
        part_counts=part_counts,
    )
    summary["validated_corpus_rows"] = corpus_validation.validated_rows
    summary["decision_corpus_manifest_sha256"] = corpus_validation.manifest_sha256
    summary["decision_corpus_provenance"] = corpus_validation.as_dict()
    summary["package_assets"] = package_validation.as_dict()
    summary["oracle_precompute"] = oracle_precompute
    write_profile_summary(
        staging_dir / "resolved_config.json",
        config.model_dump(mode="json"),
    )
    selected_runtime = selected_runtime_artifact(config, selection=selection)
    if selected_runtime is not None:
        write_profile_summary(
            staging_dir / "selected_runtime.json",
            selected_runtime,
        )
    write_profile_summary(staging_dir / "summary.json", summary)
    _publish_campaign_directory(
        staging_dir,
        output_dir=config.output_dir,
        campaign_id=config.campaign_id,
    )
    return summary


def _measurement_point_order(
    config: IntegratedPlannerProfileConfig,
) -> tuple[PlannerProfilePointConfig, ...]:
    """Return a preregistered nonmonotone order keyed by campaign identity."""

    def order_key(point: PlannerProfilePointConfig) -> bytes:
        digest = hashlib.sha256()
        digest.update(_POINT_ORDER_DOMAIN)
        digest.update(config.campaign_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(point.point_id.encode("utf-8"))
        return digest.digest()

    return tuple(sorted(config.points, key=order_key))


def _validate_run_record(
    record: PlannerProfileRunRecord,
    *,
    config: IntegratedPlannerProfileConfig,
    point: PlannerProfilePointConfig,
    summary: PlannerProfilePointSummary,
    oracle_elapsed_ms: float,
    native_library_sha256: str,
) -> None:
    if (
        record.campaign_id != config.campaign_id
        or record.point_id != point.point_id
        or record.budget_id != point.budget_id
        or record.environment != point.environment
        or record.planner_enabled != point.planner_enabled
    ):
        raise ValueError("profile run record identity differs from its point")
    if record.decisions != summary.decisions or summary.decisions <= 0:
        raise ValueError("profile run record differs from streamed decisions")
    if record.oracle_decisions != summary.oracle_executions:
        raise ValueError("profile run oracle execution count differs from evidence")
    if not math.isclose(
        record.oracle_elapsed_seconds * 1_000.0,
        oracle_elapsed_ms,
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise ValueError("profile run oracle time differs from decision evidence")
    if (
        record.valid_planner_evidence_decisions != summary.planner_decisions
        or record.fallback_decisions != summary.fallback_decisions
        or record.failed_decisions != summary.failed_decisions
        or record.deadline_exceeded_decisions != summary.deadline_exceeded_decisions
    ):
        raise ValueError("profile run counters differ from decision evidence")
    model_identity = config.model_identity_for(point)
    if record.checkpoint_sha256 != model_identity.checkpoint_sha256:
        raise ValueError("profile run used a different serving checkpoint")
    if model_identity.source_checkpoint_sha256 != config.expected_checkpoint_sha256:
        raise ValueError("profile run model has another source checkpoint lineage")
    if record.decision_corpus_sha256 != config.expected_decision_corpus_sha256:
        raise ValueError("profile run used a different decision corpus")
    if record.native_library_fingerprint != native_library_sha256:
        raise ValueError("profile run used a different native engine library")
    if record.native_abi_fingerprint != config.expected_native_abi_fingerprint:
        raise ValueError("profile run used a different native planning ABI")
    if record.native_schema_fingerprint != config.expected_native_schema_fingerprint:
        raise ValueError("profile run used a different native payload schema")
    runtime = config.runtime_for(point).planner
    resolved = runtime.resolve_for_lease(
        model_fingerprint=model_identity.model_fingerprint,
        policy_version=model_identity.policy_version,
        proposal_version=model_identity.proposal_version,
    )
    identity = resolved.runtime_identity
    if (
        record.runtime_fingerprint != resolved.runtime_fingerprint
        or record.planner_fingerprint != identity.planner_fingerprint
        or record.controller_fingerprint != identity.controller_fingerprint
        or record.constructor_fingerprint != identity.constructor_fingerprint
        or record.scorer_fingerprint != identity.scorer_fingerprint
        or record.tensor_schema_fingerprint
        != ROOT_INFORMATION_TENSOR_SCHEMA_FINGERPRINT
    ):
        raise ValueError("profile run differs from its resolved planner identity")
    if (
        record.model_fingerprint != identity.model_fingerprint
        or record.policy_version != identity.policy_version
        or record.proposal_version != identity.proposal_version
    ):
        raise ValueError("profile run differs from its acquired model lease")


def _validate_campaign_identity(
    records: Sequence[PlannerProfileRunRecord],
    *,
    config: IntegratedPlannerProfileConfig,
) -> None:
    if not records:
        raise ValueError("integrated planner campaign has no run records")
    points = {point.point_id: point for point in config.points}
    if {record.point_id for record in records} != set(points):
        raise ValueError("profile run identities differ from the campaign matrix")
    machine_fingerprints = {item.machine_fingerprint for item in records}
    publication_leases = {
        (item.policy_version, item.proposal_version) for item in records
    }
    learner_kernel_fingerprints = {
        item.workload_fingerprint for item in records if item.environment == "h200_mps"
    }
    environment_leases: dict[str, set[tuple[str, str]]] = {}
    source_lineages: set[str] = set()
    for record in records:
        point = points[record.point_id]
        identity = config.model_identity_for(point)
        source_lineages.add(identity.source_checkpoint_sha256)
        environment_leases.setdefault(record.environment, set()).add(
            (record.checkpoint_sha256, record.model_fingerprint)
        )
    native_abis = {item.native_abi_fingerprint for item in records}
    native_schemas = {item.native_schema_fingerprint for item in records}
    if len(machine_fingerprints) != 1:
        raise ValueError("profile points were not measured on one machine state")
    if len(publication_leases) != 1:
        raise ValueError("profile points used different publication leases")
    if len(learner_kernel_fingerprints) != 1:
        raise ValueError("H200 points changed the fixed learner kernel tensors")
    if any(len(leases) != 1 for leases in environment_leases.values()):
        raise ValueError("profile points changed serving model within an environment")
    if source_lineages != {config.expected_checkpoint_sha256}:
        raise ValueError("profile environments have different source lineage")
    if len(native_abis) != 1:
        raise ValueError("profile points used different native ABI identities")
    if len(native_schemas) != 1:
        raise ValueError("profile points used different native schema identities")


def _validate_paired_decision_evidence(
    config: IntegratedPlannerProfileConfig,
    *,
    scenario_supports: Mapping[str, Mapping[tuple[int, str], str]],
    oracle_keys: Mapping[str, frozenset[tuple[int, str]]],
) -> None:
    """Require exact root/repetition support and oracle pairing for every point."""
    expected_point_ids = {point.point_id for point in config.points}
    if set(scenario_supports) != expected_point_ids or set(oracle_keys) != (
        expected_point_ids
    ):
        raise ValueError("paired profile evidence is missing a campaign point")
    first_point = config.points[0].point_id
    reference_support = scenario_supports[first_point]
    reference_oracles = oracle_keys[first_point]
    if not reference_support or not reference_oracles:
        raise ValueError("profile campaign has no paired scenario/oracle evidence")
    for point in config.points[1:]:
        if scenario_supports[point.point_id] != reference_support:
            raise ValueError("profile points used different scenario supports")
        if oracle_keys[point.point_id] != reference_oracles:
            raise ValueError("profile points used different compatible oracle roots")


def validate_profile_act_time_replays(
    config: IntegratedPlannerProfileConfig,
) -> list[dict[str, Any]]:
    """Verify immutable ordered replay structure, callbacks, and deck binding."""
    result: list[dict[str, Any]] = []
    for asset in config.act_time_replay.assets:
        actual_sha256 = _require_fingerprint(
            asset.path,
            expected=asset.sha256,
            label="ActTime replay",
        )
        with asset.path.open("r", encoding="utf-8") as source:
            payload = json.load(source)
        if not isinstance(payload, dict) or not isinstance(payload.get("steps"), list):
            raise ValueError(f"ActTime replay has no ordered steps: {asset.path}")
        steps = payload["steps"]
        callback_counts: list[int] = []
        deck_fingerprints: list[str] = []
        for seat in config.act_time_replay.seats:
            callbacks = 0
            recorded_decks: set[tuple[int, ...]] = set()
            for step in steps:
                if not isinstance(step, list) or len(step) != 2:
                    raise ValueError("ActTime replay step must contain both seats")
                entry = step[seat]
                if not isinstance(entry, dict):
                    raise ValueError("ActTime replay seat entry must be a mapping")
                if entry.get("status") == "ACTIVE":
                    if not isinstance(entry.get("observation"), dict):
                        raise ValueError("active replay callback has no observation")
                    callbacks += 1
                action = entry.get("action")
                if (
                    isinstance(action, list)
                    and len(action) == 60
                    and all(
                        isinstance(card_id, int) and card_id > 0 for card_id in action
                    )
                ):
                    recorded_decks.add(
                        tuple(sorted(int(card_id) for card_id in action))
                    )
            expected_callbacks = asset.active_callbacks_by_seat[seat]
            if callbacks != expected_callbacks:
                raise ValueError("ActTime replay callback count differs from config")
            if len(recorded_decks) != 1:
                raise ValueError("ActTime replay seat lacks one immutable deck")
            callback_counts.append(callbacks)
            deck = next(iter(recorded_decks))
            canonical = json.dumps(deck, separators=(",", ":")).encode("utf-8")
            deck_fingerprints.append(hashlib.sha256(canonical).hexdigest())
        result.append(
            {
                "path": str(asset.path),
                "sha256": actual_sha256,
                "active_callbacks_by_seat": callback_counts,
                "recorded_deck_fingerprints": deck_fingerprints,
            }
        )
    return result


def _prepare_campaign_staging(output_dir: Path) -> Path:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.with_name(f".{output_dir.name}.staging")
    if output_dir.exists() or staging.exists():
        raise FileExistsError("planner profile campaign output is immutable")
    staging.mkdir()
    return staging


def _publish_campaign_directory(
    staging_dir: Path,
    *,
    output_dir: Path,
    campaign_id: str,
) -> None:
    """Hash the complete staging tree, mark it complete, then rename once."""
    artifacts = {
        str(path.relative_to(staging_dir)): file_sha256(path)
        for path in sorted(staging_dir.rglob("*"))
        if path.is_file()
    }
    write_profile_summary(
        staging_dir / "COMPLETED.json",
        {
            "schema_version": 1,
            "campaign_id": campaign_id,
            "artifact_sha256": artifacts,
        },
    )
    if output_dir.exists():
        raise FileExistsError("planner profile output appeared during publication")
    os.replace(staging_dir, output_dir)


def _validate_profile_host(config: IntegratedPlannerProfileConfig) -> None:
    if config.require_h200:
        if not torch.cuda.is_available():
            raise RuntimeError("integrated profile requires a visible NVIDIA H200")
        device_name = torch.cuda.get_device_name(torch.cuda.current_device()).upper()
        if "H200" not in device_name:
            raise RuntimeError(f"integrated profile requires H200, got {device_name}")
    if config.require_cuda_mps and not os.environ.get("CUDA_MPS_PIPE_DIRECTORY"):
        raise RuntimeError("integrated profile requires an active CUDA MPS wrapper")


def _require_fingerprint(path: Path, *, expected: str, label: str) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"profile {label} not found: {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"profile {label} fingerprint does not match config")
    return actual


def validate_profile_belief_runtime(
    config: IntegratedPlannerProfileConfig,
) -> tuple[str, str]:
    """Construct and verify the shared sampler-plus-public-producer identity."""
    workloads = tuple(runtime.belief for runtime in config.runtime_profiles.values())
    if not workloads:
        raise ValueError("integrated profile has no belief workload")
    workload = workloads[0]
    path = workload.sampler.prior_deck_signature_summary_path
    expected = workload.sampler.prior_deck_signature_summary_sha256
    if path is None or expected is None:
        raise ValueError("integrated profile belief prior is not immutable")
    prior_sha256 = _require_fingerprint(path, expected=expected, label="belief prior")
    sampler = BeliefSampler(config=workload.sampler)
    producer = OpponentBeliefFeatureProducer.from_config(workload.producer)
    fingerprint = belief_runtime_fingerprint(sampler, producer)
    configured = {
        runtime.planner.scenario.belief_sampler_fingerprint
        for runtime in config.runtime_profiles.values()
    }
    if configured != {fingerprint}:
        raise ValueError("profile belief runtime fingerprint differs from config")
    return prior_sha256, fingerprint


__all__ = [
    "run_integrated_planner_profile",
    "validate_profile_act_time_replays",
    "validate_profile_belief_runtime",
]
