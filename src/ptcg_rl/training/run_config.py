"""Shared training run metadata and output path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


class TrainingRunConfig(BaseModel):
    """Version metadata shared by training entrypoints."""

    model_config = ConfigDict(extra="forbid")

    version: str = "dev"
    output_root: Path = Path("outputs")

    @field_validator("version")
    @classmethod
    def valid_version(cls, value: str) -> str:
        """Reject empty or path-like version names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("training run version must be non-empty")
        path = Path(cleaned)
        if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
            raise ValueError(f"training run version must be a single path part: {value}")
        return cleaned


def resolve_training_output_dir(
    *,
    task_name: str,
    run: TrainingRunConfig,
    output_dir: Path | None,
) -> Path:
    """Return the explicit or version-derived training artifact directory."""
    if output_dir is not None:
        return output_dir
    return run.output_root / "training" / task_name / run.version


def resolved_training_config_dump(
    config: BaseModel,
    *,
    task_name: str,
    run: TrainingRunConfig,
    output_dir: Path | None,
) -> dict[str, Any]:
    """Dump a training config with its resolved artifact directory recorded."""
    data = config.model_dump(mode="json")
    data["output_dir"] = str(
        resolve_training_output_dir(
            task_name=task_name,
            run=run,
            output_dir=output_dir,
        )
    )
    return data
