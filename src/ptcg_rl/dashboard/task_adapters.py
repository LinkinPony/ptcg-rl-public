"""Typed task requests to fixed argv and generated configuration files."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.dashboard.task_catalog import TaskCatalogService
from ptcg_rl.dashboard.task_models import (
    BundleStrengthTaskRequest,
    ConfigDryRunTaskRequest,
    PackageValidationTaskRequest,
    ReleaseH2HTaskRequest,
    RuntimeEloTaskRequest,
    TaskCreateRequest,
    TaskParticipantRef,
    TaskResourceClass,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.evaluation.bundle_models import (
    BundleAgentConfig,
    BundleGauntletConfig,
    BundleScheduleConfig,
    EvaluationBundleConfig,
)
from ptcg_rl.evaluation.deck_strength_models import DeckStrengthConfig
from ptcg_rl.evaluation.distributed_runner import DistributedReleaseLaunchConfig
from ptcg_rl.evaluation.release_h2h import (
    ReleaseH2HConfig,
    ReleaseH2HDecisionConfig,
    ReleaseParticipantConfig,
)
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.submission.release_assets import (
    load_release_bundle_for_native_execution,
    submission_fingerprint,
)
from ptcg_rl.training.arena_decks import DeckPoolConfig
from ptcg_rl.training.run_config import TrainingRunConfig
from ptcg_rl.training.runtime_deck_ladder import (
    RuntimeDeckLadderConcurrencyConfig,
    RuntimeDeckLadderConfig,
    RuntimeDeckLadderScheduleConfig,
)


@dataclass(frozen=True)
class GeneratedTaskFile:
    """One JSON file atomically published with the task receipt."""

    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class PreparedTask:
    """Complete server-generated execution specification."""

    resource_class: TaskResourceClass
    argv: tuple[str, ...]
    output_dir: Path | None
    status_path: Path | None
    files: tuple[GeneratedTaskFile, ...]
    resolved_inputs: dict[str, Any]


class TaskCommandAdapters:
    """Build allowlisted task commands without browser-controlled paths."""

    def __init__(self, repo_root: Path, catalog: TaskCatalogService) -> None:
        self.repo_root = repo_root.resolve()
        self.catalog = catalog

    def prepare(self, request: TaskCreateRequest, *, task_id: str) -> PreparedTask:
        """Resolve catalog IDs and build one fixed command."""
        if isinstance(request, BundleStrengthTaskRequest):
            return self._bundle_strength(request, task_id=task_id)
        if isinstance(request, ReleaseH2HTaskRequest):
            return self._release_h2h(request, task_id=task_id)
        if isinstance(request, RuntimeEloTaskRequest):
            return self._runtime_elo(request, task_id=task_id)
        if isinstance(request, PackageValidationTaskRequest):
            return self._package_validation(request, task_id=task_id)
        if isinstance(request, ConfigDryRunTaskRequest):
            return self._config_dry_run(request, task_id=task_id)
        raise TypeError(f"unsupported dashboard task request: {type(request).__name__}")

    def _bundle_strength(
        self,
        request: BundleStrengthTaskRequest,
        *,
        task_id: str,
    ) -> PreparedTask:
        output_root = self._evaluation_output(task_id, request.output_label)
        bundle_output = output_root / "bundle"
        score_output = output_root / "strength"
        status_path = output_root / "status.json"
        resolved: dict[str, Any] = {"participants": []}
        candidates = tuple(
            self._bundle_participant(
                participant,
                role="candidate",
                mode=request.mode,
                resolved=resolved,
            )
            for participant in request.candidates
        )
        opponents = tuple(
            self._bundle_participant(
                participant,
                role="opponent",
                mode=request.mode,
                resolved=resolved,
            )
            for participant in request.opponents
        )
        if request.execution == "distributed" and len(candidates) < 2:
            raise ValueError(
                "distributed bundle evaluation requires at least two candidates; "
                "choose remote-only or local for one candidate"
            )
        side = self.catalog.resolve(
            request.side_observations_id,
            kind="side_observations",
        )
        if side.path is None:
            raise ValueError("side observations artifact has no file")
        side_sha = file_sha256(side.path)
        resolved["side_observations"] = {
            "artifact_id": side.item.artifact_id,
            "sha256": side_sha,
        }
        run_name = f"dashboard_{task_id}"
        bundle_config = BundleGauntletConfig(
            candidates=candidates,
            opponents=opponents,
            protocol="dashboard-bundle-gauntlet-v2",
            experiment_id=run_name,
            stage="S4",
            schedule=BundleScheduleConfig(
                games_per_matchup=request.games_per_matchup,
                mirror_sides=True,
            ),
            run=TrainingRunConfig(version=run_name, output_root=Path("outputs")),
            output_dir=bundle_output,
            num_workers=8,
            result_shard_size=16,
            fail_on_error=False,
        )
        score_config = DeckStrengthConfig(
            games_paths=(bundle_output / "games.parquet",),
            side_observations_path=side.path,
            output_dir=score_output,
        )
        argv = [
            sys.executable,
            str(self.repo_root / "src" / "tools" / "run_bundle_evaluation.py"),
            "--bundle-config",
            str(self._task_file(task_id, "bundle_config.json")),
            "--score-config",
            str(self._task_file(task_id, "score_config.json")),
            "--status-path",
            str(status_path),
        ]
        if request.execution == "local":
            argv.append("--local-only")
        elif request.execution == "remote":
            argv.append("--remote-only")
        resource_class: TaskResourceClass
        if request.execution == "remote":
            resource_class = "remote_heavy"
        elif any(
            participant.source == "checkpoint" for participant in request.candidates
        ):
            resource_class = "local_cuda"
        else:
            resource_class = "local_heavy"
        return PreparedTask(
            resource_class=resource_class,
            argv=tuple(argv),
            output_dir=output_root,
            status_path=status_path,
            files=(
                GeneratedTaskFile(
                    "bundle_config.json",
                    bundle_config.model_dump(mode="json"),
                ),
                GeneratedTaskFile(
                    "score_config.json",
                    score_config.model_dump(mode="json"),
                ),
            ),
            resolved_inputs=resolved,
        )

    def _bundle_participant(
        self,
        reference: TaskParticipantRef,
        *,
        role: str,
        mode: str,
        resolved: dict[str, Any],
    ) -> EvaluationBundleConfig:
        if reference.source == "release_bundle":
            selection = self.catalog.resolve(
                reference.artifact_id,
                kind="release_bundle",
            )
            if selection.path is None:
                raise ValueError("release selection has no manifest")
            manifest = load_release_bundle_for_native_execution(selection.path)
            fingerprint = submission_fingerprint(manifest)
            label = reference.label or selection.item.label
            resolved["participants"].append(
                {
                    "role": role,
                    "source": reference.source,
                    "artifact_id": reference.artifact_id,
                    "manifest_sha256": manifest.source_manifest_sha256,
                    "submission_fingerprint": fingerprint,
                }
            )
            return EvaluationBundleConfig(
                bundle_id=f"release_{fingerprint[:24]}",
                pilot_id=label,
                archetype=Path(manifest.deck_path).stem,
                deck_path=manifest.deck_path,
                agent=BundleAgentConfig(
                    kind="release",
                    release_manifest_path=manifest.source_manifest_path,
                ),
            )
        if reference.source == "registered_opponent":
            if role != "opponent":
                raise ValueError("registered opponents cannot be candidates")
            selection = self.catalog.resolve(
                reference.artifact_id,
                kind="registered_opponent",
            )
            if selection.path is None or selection.value is None:
                raise ValueError("registered opponent has no fixed deck")
            registered_deck = canonicalize_deck(records.read_deck(selection.path))
            resolved["participants"].append(
                {
                    "role": role,
                    "source": reference.source,
                    "artifact_id": reference.artifact_id,
                    "deck_digest": registered_deck.deck_digest,
                }
            )
            return EvaluationBundleConfig(
                bundle_id=(
                    f"registered_{selection.value}_{registered_deck.deck_digest[:12]}"
                ),
                pilot_id=selection.value,
                archetype=selection.item.label,
                deck_path=selection.path,
                agent=BundleAgentConfig(
                    kind="registered",
                    registered_name=selection.value,
                ),
            )
        if mode != "diagnostic" or role != "candidate":
            raise ValueError("raw checkpoints are diagnostic candidates only")
        checkpoint = self.catalog.resolve(reference.artifact_id, kind="checkpoint")
        deck_selection = self.catalog.resolve(str(reference.deck_id), kind="deck")
        catalog = self.catalog.resolve(
            str(reference.public_catalog_id),
            kind="public_catalog",
        )
        runtime = self.catalog.resolve(
            str(reference.runtime_template_id),
            kind="runtime_template",
        )
        if (
            checkpoint.path is None
            or deck_selection.path is None
            or catalog.path is None
        ):
            raise ValueError("diagnostic checkpoint inputs are incomplete")
        canonical = canonicalize_deck(records.read_deck(deck_selection.path))
        active = checkpoint.item.metadata.get("active_exact_deck_digests", [])
        if canonical.deck_digest not in active:
            raise ValueError(
                "selected deck has no dedicated route in the checkpoint pair"
            )
        act_time = ActTimeConfig.model_validate_json(str(runtime.value))
        resolved["participants"].append(
            {
                "role": role,
                "source": reference.source,
                "artifact_id": reference.artifact_id,
                "policy_sha256": checkpoint.item.fingerprint,
                "deck_digest": canonical.deck_digest,
                "public_catalog_sha256": catalog.item.fingerprint,
                "runtime_template_fingerprint": runtime.item.fingerprint,
            }
        )
        return EvaluationBundleConfig(
            bundle_id=(
                f"checkpoint_{str(checkpoint.item.fingerprint)[:16]}_"
                f"{canonical.deck_digest[:12]}"
            ),
            pilot_id=reference.label or checkpoint.item.label,
            archetype=deck_selection.item.label,
            deck_path=deck_selection.path,
            agent=BundleAgentConfig(
                kind="simple_stateless_greedy",
                checkpoint_path=checkpoint.path,
                public_catalog_manifest_path=catalog.path,
                device="cuda",
                act_time=act_time,
            ),
        )

    def _release_h2h(
        self,
        request: ReleaseH2HTaskRequest,
        *,
        task_id: str,
    ) -> PreparedTask:
        candidate = self.catalog.resolve(request.candidate_id, kind="release_bundle")
        opponent = self.catalog.resolve(request.opponent_id, kind="release_bundle")
        if candidate.path is None or opponent.path is None:
            raise ValueError("release H2H selection has no manifest")
        candidate_bundle = load_release_bundle_for_native_execution(candidate.path)
        opponent_bundle = load_release_bundle_for_native_execution(opponent.path)
        output_dir = self._evaluation_output(task_id, request.output_label)
        run_name = f"dashboard_{task_id}"
        release = ReleaseH2HConfig(
            run=TrainingRunConfig(version=run_name, output_root=Path("outputs")),
            output_dir=output_dir,
            experiment_id=run_name,
            stage="formal",
            candidate=ReleaseParticipantConfig(
                label=candidate.item.label,
                manifest_path=candidate_bundle.source_manifest_path,
            ),
            opponent=ReleaseParticipantConfig(
                label=opponent.item.label,
                manifest_path=opponent_bundle.source_manifest_path,
            ),
            games=request.games,
            num_workers=4 if request.execution == "local" else 1,
            temp_root=Path("tmp/dashboard_tasks") / task_id / "release_h2h",
            decision=ReleaseH2HDecisionConfig(minimum_games=request.games),
        )
        wrapper = {
            "release": release.model_dump(mode="json"),
            "distributed": (
                None
                if request.execution == "local"
                else DistributedReleaseLaunchConfig().model_dump(mode="json")
            ),
        }
        config_path = self._task_file(task_id, "release_h2h.json")
        return PreparedTask(
            resource_class=(
                "local_heavy" if request.execution == "local" else "remote_heavy"
            ),
            argv=(
                sys.executable,
                str(self.repo_root / "src" / "tools" / "dashboard_task.py"),
                "--kind",
                "release_h2h",
                "--config",
                str(config_path),
            ),
            output_dir=output_dir,
            status_path=output_dir / "status.json",
            files=(GeneratedTaskFile("release_h2h.json", wrapper),),
            resolved_inputs={
                "candidate": {
                    "artifact_id": candidate.item.artifact_id,
                    "manifest_sha256": candidate_bundle.source_manifest_sha256,
                    "submission_fingerprint": submission_fingerprint(candidate_bundle),
                },
                "opponent": {
                    "artifact_id": opponent.item.artifact_id,
                    "manifest_sha256": opponent_bundle.source_manifest_sha256,
                    "submission_fingerprint": submission_fingerprint(opponent_bundle),
                },
            },
        )

    def _runtime_elo(
        self,
        request: RuntimeEloTaskRequest,
        *,
        task_id: str,
    ) -> PreparedTask:
        checkpoint = self.catalog.resolve(request.checkpoint_id, kind="checkpoint")
        catalog = self.catalog.resolve(
            request.public_catalog_id,
            kind="public_catalog",
        )
        runtime = self.catalog.resolve(
            request.runtime_template_id,
            kind="runtime_template",
        )
        decks = tuple(
            self.catalog.resolve(deck_id, kind="deck") for deck_id in request.deck_ids
        )
        if (
            checkpoint.path is None
            or catalog.path is None
            or any(deck.path is None for deck in decks)
        ):
            raise ValueError("runtime Elo inputs are incomplete")
        active = checkpoint.item.metadata.get("active_exact_deck_digests", [])
        resolved_decks: list[dict[str, Any]] = []
        deck_paths: list[Path] = []
        for selection in decks:
            assert selection.path is not None
            canonical = canonicalize_deck(records.read_deck(selection.path))
            if canonical.deck_digest not in active:
                raise ValueError(
                    f"deck {selection.item.label!r} has no checkpoint route"
                )
            deck_paths.append(selection.path)
            resolved_decks.append(
                {
                    "artifact_id": selection.item.artifact_id,
                    "deck_digest": canonical.deck_digest,
                    "runtime_deck_id": (f"path:{records.display_path(selection.path)}"),
                    "deck_hash": records.signature_hash(canonical.signature),
                }
            )
        output_dir = self._evaluation_output(task_id, request.output_label)
        config = RuntimeDeckLadderConfig(
            checkpoint_path=checkpoint.path,
            device="cuda",
            public_catalog_manifest_path=catalog.path,
            deck_pool=DeckPoolConfig(deck_paths=tuple(deck_paths)),
            run=TrainingRunConfig(
                version=f"dashboard_{task_id}",
                output_root=Path("outputs"),
            ),
            output_dir=output_dir,
            schedule=RuntimeDeckLadderScheduleConfig(
                uniform_games_per_pair=request.games_per_pair,
                recent_weighted_extra_games=0,
                mirror_sides=True,
            ),
            concurrency=RuntimeDeckLadderConcurrencyConfig(),
            act_time=ActTimeConfig.model_validate_json(str(runtime.value)),
        )
        config_path = self._task_file(task_id, "runtime_elo.json")
        return PreparedTask(
            resource_class="local_cuda",
            argv=(
                sys.executable,
                str(self.repo_root / "src" / "tools" / "dashboard_task.py"),
                "--kind",
                "runtime_elo",
                "--config",
                str(config_path),
            ),
            output_dir=output_dir,
            status_path=output_dir / "progress.json",
            files=(
                GeneratedTaskFile(
                    "runtime_elo.json",
                    config.model_dump(mode="json"),
                ),
            ),
            resolved_inputs={
                "checkpoint": {
                    "artifact_id": checkpoint.item.artifact_id,
                    "policy_sha256": checkpoint.item.fingerprint,
                },
                "decks": resolved_decks,
                "public_catalog": {
                    "artifact_id": catalog.item.artifact_id,
                    "sha256": catalog.item.fingerprint,
                },
                "runtime_template": {
                    "artifact_id": runtime.item.artifact_id,
                    "fingerprint": runtime.item.fingerprint,
                },
            },
        )

    def _package_validation(
        self,
        request: PackageValidationTaskRequest,
        *,
        task_id: str,
    ) -> PreparedTask:
        profile = self.catalog.resolve(
            request.submission_profile_id,
            kind="submission_profile",
        )
        if profile.value is None:
            raise ValueError("submission profile has no config name")
        label = _safe_output_label(request.output_label)
        output = (
            self.repo_root / "dist" / "dashboard" / f"{label}-{task_id[:12]}.tar.gz"
        )
        return PreparedTask(
            resource_class="local_heavy",
            argv=(
                sys.executable,
                str(self.repo_root / "src" / "tools" / "submission" / "protocol.py"),
                "--profile",
                profile.value,
                "--output",
                str(output),
            ),
            output_dir=output,
            status_path=None,
            files=(),
            resolved_inputs={
                "profile": {
                    "artifact_id": profile.item.artifact_id,
                    "fingerprint": profile.item.fingerprint,
                },
                "archive_path": output.relative_to(self.repo_root).as_posix(),
            },
        )

    def _config_dry_run(
        self,
        request: ConfigDryRunTaskRequest,
        *,
        task_id: str,
    ) -> PreparedTask:
        profile = self.catalog.resolve(
            request.training_profile_id,
            kind="training_profile",
        )
        if profile.value is None:
            raise ValueError("training profile has no Hydra config name")
        return PreparedTask(
            resource_class="light",
            argv=(
                sys.executable,
                str(self.repo_root / "train.py"),
                "--dry-run",
                "--profile",
                profile.value,
                "--run-version",
                f"dashboard_dry_run_{task_id[:12]}",
            ),
            output_dir=None,
            status_path=None,
            files=(),
            resolved_inputs={
                "profile": {
                    "artifact_id": profile.item.artifact_id,
                    "fingerprint": profile.item.fingerprint,
                }
            },
        )

    def _evaluation_output(self, task_id: str, output_label: str) -> Path:
        label = _safe_output_label(output_label)
        return (
            self.repo_root
            / "outputs"
            / "evaluation"
            / "dashboard"
            / f"{label}-{task_id[:12]}"
        )

    def _task_file(self, task_id: str, name: str) -> Path:
        return self.repo_root / "outputs" / "dashboard" / "tasks" / task_id / name


def _safe_output_label(value: str) -> str:
    cleaned = value.strip()
    if (
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", cleaned) is None
        or "latest" in cleaned.lower()
    ):
        raise ValueError(
            "output label must be 1-64 safe characters and cannot contain latest"
        )
    return cleaned


def task_spec_fingerprint(payload: dict[str, Any]) -> str:
    """Hash the complete receipt-owned semantic task specification."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
