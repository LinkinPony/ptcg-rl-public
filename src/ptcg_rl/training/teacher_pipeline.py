"""Orchestration for immutable public-pilot replay teacher preparation."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps.environment import (
    KaggleStepExtractionConfig,
)
from ptcg_rl.data.kaggle_steps.environment import (
    run as run_step_extraction,
)
from ptcg_rl.training.teacher_dataset import (
    PublicPilotTeacherConfig,
    build_public_pilot_teacher,
)
from ptcg_rl.training.teacher_replays import (
    PublicPilotReplaySyncConfig,
    sync_public_pilot_replays,
)


class PublicPilotTeacherPipelineConfig(BaseModel):
    """Validated paths for replay sync, extraction, and teacher filtering."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    sync: PublicPilotReplaySyncConfig
    extraction: KaggleStepExtractionConfig
    teacher: PublicPilotTeacherConfig

    @model_validator(mode="after")
    def aligned_paths(self) -> PublicPilotTeacherPipelineConfig:
        """Require all three stages to consume the exact preceding artifact."""
        sync_output = deck_records.repo_path(self.sync.output_dir)
        extraction_root = deck_records.repo_path(self.extraction.replay_root)
        extraction_output = deck_records.repo_path(self.extraction.output_dir)
        teacher_source = deck_records.repo_path(self.teacher.source_manifest_path)
        teacher_replays = deck_records.repo_path(self.teacher.replay_manifest_path)
        if extraction_root != sync_output / "replays":
            raise ValueError("extraction.replay_root must equal sync.output_dir/replays")
        if teacher_source != extraction_output / "manifest.json":
            raise ValueError(
                "teacher.source_manifest_path must reference extraction manifest"
            )
        if teacher_replays != sync_output / "manifest.json":
            raise ValueError("teacher replay manifest must reference sync manifest")
        if self.teacher.submission_id != self.sync.submission_id:
            raise ValueError("teacher and replay submission ids must match")
        if self.teacher.team_name.casefold() != self.sync.team_name.casefold():
            raise ValueError("teacher and replay team names must match")
        return self


def run_public_pilot_teacher_pipeline(
    config: PublicPilotTeacherPipelineConfig,
) -> dict[str, Any]:
    """Run or verify every immutable teacher data preparation stage."""
    replay_manifest = sync_public_pilot_replays(config.sync)
    replay_manifest_path = deck_records.repo_path(config.sync.output_dir) / "manifest.json"
    extraction_manifest = _extract_or_reuse(
        config.extraction,
        replay_manifest_path=replay_manifest_path,
        replay_manifest=replay_manifest,
    )
    teacher_manifest = build_public_pilot_teacher(config.teacher)
    return {
        "submission_id": config.sync.submission_id,
        "replays": replay_manifest,
        "extraction": extraction_manifest,
        "teacher": teacher_manifest,
    }


def _extract_or_reuse(
    config: KaggleStepExtractionConfig,
    *,
    replay_manifest_path: Path,
    replay_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = deck_records.repo_path(config.output_dir)
    manifest_path = output_dir / "manifest.json"
    identity_path = output_dir / "source_identity.json"
    expected_identity = {
        "replay_manifest_path": deck_records.display_path(replay_manifest_path),
        "replay_manifest_sha256": _sha256(replay_manifest_path),
        "extraction_config": config.model_dump(mode="json"),
    }
    if manifest_path.exists() and identity_path.exists():
        identity = _read_json(identity_path)
        if identity != expected_identity:
            raise ValueError(
                "existing public-pilot extraction identity differs; choose a new "
                "output_dir"
            )
        manifest = _read_json(manifest_path)
        extracted = int(
            _mapping(manifest.get("summary")).get("episode_json_files", -1)
        )
        expected = int(replay_manifest.get("episode_count", -2))
        if extracted != expected:
            raise ValueError(
                f"public-pilot extraction replay count mismatch: {extracted} != {expected}"
            )
        return manifest
    if output_dir.exists():
        shutil.rmtree(output_dir)
    manifest = run_step_extraction(config)
    _atomic_write_json(output_dir / "source_identity.json", expected_identity)
    return manifest


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()
