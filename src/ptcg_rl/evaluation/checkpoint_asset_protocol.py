"""Submission-protocol validation for one checkpoint campaign asset."""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.checkpoint_asset_models import CheckpointAssetAuditConfig
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.submission.protocol import (
    ArchiveEntry,
    SubmissionProfileConfig,
    ValidationConfig,
    build_submission_archive,
    validate_submission_archive,
)


def run_checkpoint_submission_protocol(
    checkpoint_tag: str,
    *,
    asset_path: Path,
    archive_path: Path,
    config: CheckpointAssetAuditConfig,
) -> dict[str, Any]:
    """Build and validate a Kaggle-like archive around one FP16 asset."""
    profile = SubmissionProfileConfig(
        name=f"checkpoint_selection_{checkpoint_tag}",
        draft=True,
        include_paths=(
            ArchiveEntry(source=Path("src/main.py"), target=PurePosixPath("main.py")),
            ArchiveEntry(source=Path("src/ptcg_rl"), target=PurePosixPath("ptcg_rl")),
            ArchiveEntry(
                source=Path("data/sample_submission/cg"),
                target=PurePosixPath("cg"),
            ),
            ArchiveEntry(source=config.deck_path, target=PurePosixPath("deck.csv")),
            ArchiveEntry(
                source=asset_path,
                target=PurePosixPath("agent_checkpoint.pt"),
            ),
            ArchiveEntry(
                source=config.belief_summary_path,
                target=PurePosixPath("belief_prior.csv"),
            ),
            ArchiveEntry(
                source=config.static_features_path,
                target=PurePosixPath(
                    "outputs/cards/static_features/card_static_features.npy"
                ),
            ),
        ),
        required_files=(
            PurePosixPath("main.py"),
            PurePosixPath("deck.csv"),
            PurePosixPath("agent_checkpoint.pt"),
            PurePosixPath("belief_prior.csv"),
            PurePosixPath("cg/api.py"),
            PurePosixPath("outputs/cards/static_features/card_static_features.npy"),
        ),
        validation=ValidationConfig(
            timeout_seconds=300.0,
            expected_action_length=60,
            expected_deck_sha256=file_sha256(records.repo_path(config.deck_path)),
            require_checkpoint_policy=True,
            require_engine=True,
            require_belief_prior=True,
            startup_observation={"select": None},
        ),
    )
    built_path = build_submission_archive(profile, archive_path)
    validation = dict(validate_submission_archive(built_path, profile))
    return {
        "archive_path": records.display_path(built_path),
        "archive_sha256": file_sha256(built_path),
        "validated": True,
        "validation": validation,
    }


__all__ = ["run_checkpoint_submission_protocol"]
