"""Plan balanced device shards over historical checkpoint-route bundles."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    CheckpointRosterDeckConfig,
    CheckpointRosterParticipantConfig,
    NativeCheckpointGauntletConfig,
)
from ptcg_rl.evaluation.native_collection_campaign.artifact_io import (
    file_sha256,
    read_json,
    write_json_atomic,
)
from ptcg_rl.evaluation.native_collection_campaign.inventory import (
    load_historical_inventory,
)
from ptcg_rl.evaluation.native_collection_campaign.models import (
    PLAN_FORMAT,
    BundleSelector,
    CampaignSelection,
    CampaignTask,
    HistoricalCheckpointInventory,
    HistoricalCheckpointRecord,
    NativeCollectionCampaignPlan,
    PlannedBundle,
    artifact_fingerprint,
)


class CampaignExecutionConfig(BaseModel):
    """Performance knobs shared by every model-pair task in one stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native_worker_replicas: int = Field(default=10, gt=0)
    native_worker_torch_threads: int = Field(default=1, gt=0)
    native_inductor_compile_threads: int | None = Field(default=None, gt=0)
    native_arena_capacity: int = Field(default=512, gt=0)
    native_engine_shards: int = Field(default=8, ge=4, le=8)
    native_engine_fact_workers: int = Field(default=8, gt=0)
    native_policy_cohort_slots: int = Field(default=512, gt=0)
    native_policy_group_bank_limit: int = Field(default=3, ge=1, le=4)
    native_policy_cohort_wait_ms: float = Field(default=0.0, ge=0.0)
    native_frozen_batch_min_rows: int = Field(default=48, gt=0)
    native_frozen_batch_max_wait_waves: int = Field(default=4, gt=0)
    collection_part_games: int = Field(default=1024, gt=0)
    maximum_engine_steps: int = Field(default=10_000, gt=0)
    compression: str = "zstd"
    require_full_wave_geometry: bool = False
    require_aligned_replica_waves: bool = False

    @model_validator(mode="after")
    def coherent_capacities(self) -> CampaignExecutionConfig:
        if self.native_policy_cohort_slots > self.native_arena_capacity:
            raise ValueError("policy cohort slots exceed native arena capacity")
        if self.collection_part_games < self.native_arena_capacity:
            raise ValueError("collection part must be at least one arena")
        if self.require_full_wave_geometry and not (
            self.collection_part_games
            == self.native_arena_capacity
            == self.native_policy_cohort_slots
        ):
            raise ValueError(
                "full-wave geometry requires part, arena, and policy cohort "
                "capacities to match"
            )
        return self


class CampaignBuildConfig(BaseModel):
    """Inputs for one immutable screening or confirmation stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inventory_path: Path
    plan_path: Path
    output_dir: Path
    stage_id: str
    candidate_selector: BundleSelector
    opponent_selector: BundleSelector
    candidate_selection_path: Path | None = None
    device_bindings: tuple[str, ...] = ("0", "1")
    model_pair_placement: Literal["split_across_devices", "exclusive_device"] = (
        "split_across_devices"
    )
    games_per_cell_per_seat: int = Field(default=4, gt=0)
    runner_source_commit: str
    native_library_path: Path = Path("src/native/cg_train/libcg_train.so")
    expected_native_library_sha256: str
    execution: CampaignExecutionConfig = Field(default_factory=CampaignExecutionConfig)

    @field_validator("device_bindings")
    @classmethod
    def unique_devices(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value)
        if not normalized or any(not item for item in normalized):
            raise ValueError("campaign device bindings must be nonempty")
        if len(normalized) != len(set(normalized)):
            raise ValueError("campaign device bindings must be unique")
        return normalized

    @field_validator("stage_id")
    @classmethod
    def nonempty_stage(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("campaign stage ID must be nonempty")
        return normalized

    @model_validator(mode="after")
    def valid_device_repetitions(self) -> CampaignBuildConfig:
        if (
            self.model_pair_placement == "split_across_devices"
            and self.games_per_cell_per_seat < len(self.device_bindings)
        ):
            raise ValueError(
                "games_per_cell_per_seat must give every device every stratum"
            )
        return self


def build_campaign_plan(
    config: CampaignBuildConfig,
    *,
    root: Path | None = None,
) -> NativeCollectionCampaignPlan:
    """Build or validate an immutable, balanced multi-device campaign plan."""
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    inventory = load_historical_inventory(config.inventory_path, root=repo_root)
    library_path = _resolve_path(config.native_library_path, root=repo_root)
    if file_sha256(library_path) != config.expected_native_library_sha256:
        raise ValueError("native evaluation library differs from configured SHA-256")
    all_bundles = _planned_bundles(
        inventory,
        runner_source_commit=config.runner_source_commit,
        native_library_sha256=config.expected_native_library_sha256,
    )
    candidate_selector = _candidate_selector(config, root=repo_root)
    candidates = _select_bundles(all_bundles, candidate_selector, role="candidate")
    opponents = _select_bundles(
        all_bundles,
        config.opponent_selector,
        role="opponent",
    )
    checkpoint_by_id = {item.checkpoint_id: item for item in inventory.checkpoints}
    candidate_groups = _group_by_checkpoint(candidates)
    opponent_groups = _group_by_checkpoint(opponents)
    model_pairs = tuple(
        (
            checkpoint_by_id[candidate_id],
            candidate_bundles,
            checkpoint_by_id[opponent_id],
            opponent_bundles,
        )
        for candidate_id, candidate_bundles in sorted(candidate_groups.items())
        for opponent_id, opponent_bundles in sorted(opponent_groups.items())
    )
    tasks = _build_tasks(
        config,
        inventory=inventory,
        model_pairs=model_pairs,
    )
    inventory_path = _display_path(
        _resolve_path(config.inventory_path, root=repo_root),
        root=repo_root,
    )
    output_dir = _display_path(
        _resolve_path(config.output_dir, root=repo_root),
        root=repo_root,
    )
    candidate_records = tuple(candidates)
    opponent_records = tuple(opponents)
    task_records = tuple(tasks)
    payload = {
        "format": PLAN_FORMAT,
        "stage_id": config.stage_id,
        "inventory_path": str(inventory_path),
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "output_dir": str(output_dir),
        "device_bindings": config.device_bindings,
        "candidates": [item.model_dump(mode="json") for item in candidate_records],
        "opponents": [item.model_dump(mode="json") for item in opponent_records],
        "tasks": [item.model_dump(mode="json") for item in task_records],
    }
    plan = NativeCollectionCampaignPlan(
        plan_fingerprint=artifact_fingerprint(
            "native-collection-campaign-plan/v1",
            payload,
        ),
        stage_id=config.stage_id,
        inventory_path=inventory_path,
        inventory_fingerprint=inventory.inventory_fingerprint,
        output_dir=output_dir,
        device_bindings=config.device_bindings,
        candidates=candidate_records,
        opponents=opponent_records,
        tasks=task_records,
    )
    _publish_or_validate_plan(
        _resolve_path(config.plan_path, root=repo_root),
        plan,
    )
    return plan


def load_campaign_plan(
    path: Path,
    *,
    root: Path | None = None,
) -> NativeCollectionCampaignPlan:
    """Load a plan and reject any payload/fingerprint mismatch."""
    repo_root = (records.repo_path(Path(".")) if root is None else root).resolve()
    raw = read_json(_resolve_path(path, root=repo_root))
    payload = dict(raw)
    payload.pop("plan_fingerprint", None)
    expected = artifact_fingerprint("native-collection-campaign-plan/v1", payload)
    if expected != raw.get("plan_fingerprint"):
        raise ValueError("campaign plan fingerprint differs from its payload")
    plan = NativeCollectionCampaignPlan.model_validate(raw)
    inventory = load_historical_inventory(plan.inventory_path, root=repo_root)
    if inventory.inventory_fingerprint != plan.inventory_fingerprint:
        raise ValueError("campaign plan inventory binding has changed")
    return plan


def _planned_bundles(
    inventory: HistoricalCheckpointInventory,
    *,
    runner_source_commit: str,
    native_library_sha256: str,
) -> tuple[PlannedBundle, ...]:
    output: list[PlannedBundle] = []
    for checkpoint in inventory.checkpoints:
        for deck in checkpoint.decks:
            payload = {
                "checkpoint_id": checkpoint.checkpoint_id,
                "checkpoint_sha256": checkpoint.checkpoint_sha256,
                "model_fingerprint": checkpoint.model_fingerprint,
                "deck_digest": deck.deck_digest,
                "policy_temperature": 0.0,
                "public_catalog_fingerprint": (checkpoint.public_catalog_fingerprint),
                "exact_registry_fingerprint": checkpoint.exact_registry_fingerprint,
                "input_contract_fingerprint": checkpoint.input_contract_fingerprint,
                "runner_source_commit": runner_source_commit,
                "native_library_sha256": native_library_sha256,
                "decode_semantics": "strict-greedy-action-only-v1",
            }
            output.append(
                PlannedBundle(
                    bundle_id=artifact_fingerprint(
                        "native-collection-evaluation-bundle/v1",
                        payload,
                    ),
                    checkpoint_id=checkpoint.checkpoint_id,
                    model_fingerprint=checkpoint.model_fingerprint,
                    deck_digest=deck.deck_digest,
                    deck_hash=deck.deck_hash,
                    deck_label=deck.label,
                    deck_signature=deck.deck_signature,
                    policy_temperature=0.0,
                )
            )
    return tuple(sorted(output, key=lambda item: item.bundle_id))


def _select_bundles(
    bundles: tuple[PlannedBundle, ...],
    selector: BundleSelector,
    *,
    role: str,
) -> tuple[PlannedBundle, ...]:
    available_ids = {bundle.bundle_id for bundle in bundles}
    unknown_ids = set(selector.bundle_ids) - available_ids
    if unknown_ids:
        raise ValueError(f"{role} selector contains unknown bundle IDs")
    if selector.include_all:
        selected = bundles
    else:
        selected = tuple(
            bundle
            for bundle in bundles
            if (
                not selector.checkpoint_ids
                or bundle.checkpoint_id in selector.checkpoint_ids
            )
            and (
                not selector.deck_digests or bundle.deck_digest in selector.deck_digests
            )
            and (not selector.deck_hashes or bundle.deck_hash in selector.deck_hashes)
            and (not selector.bundle_ids or bundle.bundle_id in selector.bundle_ids)
        )
    if not selected:
        raise ValueError(f"{role} selector matched no exact bundles")
    return tuple(sorted(selected, key=lambda item: item.bundle_id))


def _candidate_selector(config: CampaignBuildConfig, *, root: Path) -> BundleSelector:
    if config.candidate_selection_path is None:
        return config.candidate_selector
    selection = CampaignSelection.model_validate(
        read_json(_resolve_path(config.candidate_selection_path, root=root))
    )
    return BundleSelector(bundle_ids=selection.selected_bundle_ids)


def _group_by_checkpoint(
    bundles: tuple[PlannedBundle, ...],
) -> dict[str, tuple[PlannedBundle, ...]]:
    grouped: defaultdict[str, list[PlannedBundle]] = defaultdict(list)
    for bundle in bundles:
        grouped[bundle.checkpoint_id].append(bundle)
    return {
        key: tuple(sorted(values, key=lambda item: (item.deck_digest, item.bundle_id)))
        for key, values in grouped.items()
    }


def _partition_repetitions(total: int, *, devices: int) -> tuple[int, ...]:
    quotient, remainder = divmod(total, devices)
    output = tuple(
        quotient + (1 if index < remainder else 0) for index in range(devices)
    )
    if any(value <= 0 for value in output) or sum(output) != total:
        raise ValueError("device repetition partition is invalid")
    return output


def _build_tasks(
    config: CampaignBuildConfig,
    *,
    inventory: HistoricalCheckpointInventory,
    model_pairs: tuple[
        tuple[
            HistoricalCheckpointRecord,
            tuple[PlannedBundle, ...],
            HistoricalCheckpointRecord,
            tuple[PlannedBundle, ...],
        ],
        ...,
    ],
) -> list[CampaignTask]:
    """Place complete model-pair work without changing cell coverage."""
    tasks: list[CampaignTask] = []
    if config.model_pair_placement == "split_across_devices":
        repetitions = _partition_repetitions(
            config.games_per_cell_per_seat,
            devices=len(config.device_bindings),
        )
        for candidate, candidate_bundles, opponent, opponent_bundles in model_pairs:
            for device_index, (device_binding, device_repetitions) in enumerate(
                zip(config.device_bindings, repetitions, strict=True)
            ):
                tasks.append(
                    _build_task(
                        config,
                        inventory=inventory,
                        candidate=candidate,
                        candidate_bundles=candidate_bundles,
                        opponent=opponent,
                        opponent_bundles=opponent_bundles,
                        device_index=device_index,
                        device_binding=device_binding,
                        repetitions=device_repetitions,
                    )
                )
        return tasks

    device_loads = [0] * len(config.device_bindings)
    weighted_pairs = sorted(
        model_pairs,
        key=lambda pair: (
            -(len(pair[1]) * len(pair[3])),
            pair[0].checkpoint_id,
            pair[2].checkpoint_id,
        ),
    )
    for candidate, candidate_bundles, opponent, opponent_bundles in weighted_pairs:
        device_index = min(
            range(len(config.device_bindings)),
            key=lambda index: (device_loads[index], index),
        )
        task = _build_task(
            config,
            inventory=inventory,
            candidate=candidate,
            candidate_bundles=candidate_bundles,
            opponent=opponent,
            opponent_bundles=opponent_bundles,
            device_index=device_index,
            device_binding=config.device_bindings[device_index],
            repetitions=config.games_per_cell_per_seat,
        )
        tasks.append(task)
        device_loads[device_index] += task.gauntlet.total_games
    return tasks


def _build_task(
    config: CampaignBuildConfig,
    *,
    inventory: HistoricalCheckpointInventory,
    candidate: HistoricalCheckpointRecord,
    candidate_bundles: tuple[PlannedBundle, ...],
    opponent: HistoricalCheckpointRecord,
    opponent_bundles: tuple[PlannedBundle, ...],
    device_index: int,
    device_binding: str,
    repetitions: int,
) -> CampaignTask:
    total_games = len(candidate_bundles) * len(opponent_bundles) * 2 * repetitions
    if config.execution.require_full_wave_geometry:
        games_per_wave = (
            config.execution.native_worker_replicas
            * config.execution.collection_part_games
        )
        if total_games % games_per_wave:
            raise ValueError(
                "campaign task game count does not fill complete replica waves"
            )
    if config.execution.require_aligned_replica_waves:
        part_games = config.execution.collection_part_games
        worker_replicas = config.execution.native_worker_replicas
        minimum_chunks = (total_games + part_games - 1) // part_games
        chunk_count = (
            (minimum_chunks + worker_replicas - 1) // worker_replicas
        ) * worker_replicas
        chunk_count = min(total_games, chunk_count)
        if total_games // chunk_count < config.execution.native_arena_capacity:
            raise ValueError(
                "campaign task aligned replica waves do not fill the configured "
                "native arena"
            )
    identity = {
        "stage_id": config.stage_id,
        "inventory_fingerprint": inventory.inventory_fingerprint,
        "candidate_checkpoint_id": candidate.checkpoint_id,
        "candidate_bundle_ids": [item.bundle_id for item in candidate_bundles],
        "opponent_checkpoint_id": opponent.checkpoint_id,
        "opponent_bundle_ids": [item.bundle_id for item in opponent_bundles],
        "device_index": device_index,
        "games_per_cell_per_seat": repetitions,
        "execution": config.execution.model_dump(mode="json"),
    }
    task_id = artifact_fingerprint("native-collection-campaign-task/v1", identity)
    output_dir = config.output_dir / "tasks" / task_id
    gauntlet = NativeCheckpointGauntletConfig(
        candidate=_participant(
            candidate,
            candidate_bundles,
            label=f"candidate-{candidate.checkpoint_id[:12]}",
        ),
        baseline=_participant(
            opponent,
            opponent_bundles,
            label=f"opponent-{opponent.checkpoint_id[:12]}",
        ),
        runner_source_commit=config.runner_source_commit,
        native_library_path=config.native_library_path,
        expected_native_library_sha256=config.expected_native_library_sha256,
        output_dir=output_dir,
        backend="native_collection",
        total_games=total_games,
        native_worker_replicas=config.execution.native_worker_replicas,
        native_worker_torch_threads=(config.execution.native_worker_torch_threads),
        native_inductor_compile_threads=(
            config.execution.native_inductor_compile_threads
        ),
        native_arena_capacity=config.execution.native_arena_capacity,
        native_engine_shards=config.execution.native_engine_shards,
        native_engine_fact_workers=config.execution.native_engine_fact_workers,
        native_policy_cohort_slots=config.execution.native_policy_cohort_slots,
        native_policy_group_bank_limit=(
            config.execution.native_policy_group_bank_limit
        ),
        native_policy_cohort_wait_ms=(config.execution.native_policy_cohort_wait_ms),
        native_frozen_batch_min_rows=(config.execution.native_frozen_batch_min_rows),
        native_frozen_batch_max_wait_waves=(
            config.execution.native_frozen_batch_max_wait_waves
        ),
        collection_part_games=config.execution.collection_part_games,
        maximum_engine_steps=config.execution.maximum_engine_steps,
        compression=config.execution.compression,
        seed=_matched_block_seed(
            stage_id=config.stage_id,
            inventory_fingerprint=inventory.inventory_fingerprint,
            device_index=device_index,
            repetitions=repetitions,
        ),
        match_seed_namespace=(
            f"{config.stage_id}:device-{device_index}:matched-block-v1"
        ),
        device="cuda",
        evaluation_action_only=True,
    )
    return CampaignTask(
        task_id=task_id,
        device_index=device_index,
        device_binding=device_binding,
        games_per_cell_per_seat=repetitions,
        candidate_bundle_ids=tuple(item.bundle_id for item in candidate_bundles),
        opponent_bundle_ids=tuple(item.bundle_id for item in opponent_bundles),
        gauntlet=gauntlet,
    )


def _participant(
    checkpoint: HistoricalCheckpointRecord,
    bundles: tuple[PlannedBundle, ...],
    *,
    label: str,
) -> CheckpointRosterParticipantConfig:
    deck_by_digest = {deck.deck_digest: deck for deck in checkpoint.decks}
    values: dict[str, object] = {
        "label": label,
        "checkpoint_path": checkpoint.checkpoint_path,
        "expected_checkpoint_sha256": checkpoint.checkpoint_sha256,
        "checkpoint_source_commit": checkpoint.checkpoint_source_commit,
        "source_identity_path": checkpoint.source_identity_path,
        "expected_source_identity_sha256": checkpoint.source_identity_sha256,
        "resolved_config_path": checkpoint.resolved_config_path,
        "expected_resolved_config_sha256": checkpoint.resolved_config_sha256,
        "public_catalog_manifest_path": checkpoint.public_catalog_manifest_path,
        "expected_public_catalog_manifest_sha256": (
            checkpoint.public_catalog_manifest_sha256
        ),
        "provenance_fingerprint": checkpoint.provenance_fingerprint,
        "policy_temperature": 0.0,
        "roster_scope": "subset",
        "decks": tuple(
            CheckpointRosterDeckConfig(
                label=deck_by_digest[bundle.deck_digest].label,
                path=deck_by_digest[bundle.deck_digest].path,
                deck_hash=deck_by_digest[bundle.deck_digest].deck_hash,
            )
            for bundle in bundles
        ),
    }
    if checkpoint.pair_manifest_path is not None:
        values.update(
            pair_manifest_path=checkpoint.pair_manifest_path,
            expected_pair_manifest_sha256=checkpoint.pair_manifest_sha256,
        )
    else:
        values.update(
            policy_evaluation_binding_path=checkpoint.evaluation_binding_path,
            expected_policy_evaluation_binding_sha256=(
                checkpoint.evaluation_binding_sha256
            ),
        )
    return CheckpointRosterParticipantConfig.model_validate(values)


def _matched_block_seed(
    *,
    stage_id: str,
    inventory_fingerprint: str,
    device_index: int,
    repetitions: int,
) -> int:
    identity = f"{stage_id}\0{inventory_fingerprint}\0{device_index}\0{repetitions}"
    digest = hashlib.blake2b(identity.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFFFFFF


def _publish_or_validate_plan(
    path: Path,
    plan: NativeCollectionCampaignPlan,
) -> None:
    if path.exists():
        existing = NativeCollectionCampaignPlan.model_validate(read_json(path))
        if existing != plan:
            raise ValueError("campaign plan path already binds different inputs")
        return
    write_json_atomic(path, plan)


def _resolve_path(path: Path, *, root: Path) -> Path:
    expanded = path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _display_path(path: Path, *, root: Path) -> Path:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError:
        return resolved


__all__ = [
    "CampaignBuildConfig",
    "CampaignExecutionConfig",
    "build_campaign_plan",
    "load_campaign_plan",
]
