"""Immutable ITS-EVAL-v1 campaign and stage identity construction."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.agent.search.config import MacroSearchConfig
from ptcg_rl.data.kaggle_deck import records

SearchStage = Literal["S0", "S1", "S2", "S3", "S4", "S5"]


class SearchCampaignIdentityConfig(BaseModel):
    """Frozen inputs used to derive layered search experiment fingerprints."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    stage: SearchStage
    protocol: str = "ITS-EVAL-v1"
    deck_path: Path
    checkpoint_path: Path
    belief_path: Path | None = None
    resolved_search: MacroSearchConfig
    runtime_definition: dict[str, Any] = Field(default_factory=dict)
    panel_definition: tuple[dict[str, Any], ...] = ()
    replay_paths: tuple[Path, ...] = ()
    replay_definition: dict[str, Any] = Field(default_factory=dict)
    meta_paths: tuple[Path, ...] = ()
    meta_definition: dict[str, Any] = Field(default_factory=dict)
    runtime_source_paths: tuple[Path, ...]
    engine_asset_paths: tuple[Path, ...]
    stage_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("experiment_id", "protocol")
    @classmethod
    def non_empty_identifier(cls, value: str) -> str:
        """Reject empty or mutable-looking experiment identifiers."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("campaign identifiers must be non-empty")
        if "latest" in normalized.lower():
            raise ValueError("immutable campaign identifiers cannot contain 'latest'")
        return normalized

    @field_validator("runtime_source_paths", "engine_asset_paths")
    @classmethod
    def require_fingerprint_sources(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        """Require source and engine evidence for every campaign identity."""
        if not value:
            raise ValueError("campaign identity requires fingerprint source paths")
        return value


def build_search_campaign_identity(
    config: SearchCampaignIdentityConfig,
) -> dict[str, Any]:
    """Hash assets, bundle, panel, replay, meta, engine, host, and stage."""
    deck_path = _resolve_file(config.deck_path)
    checkpoint_path = _resolve_file(config.checkpoint_path)
    belief_path = _resolve_optional_file(config.belief_path)
    runtime_sources = tuple(_resolve_file(path) for path in config.runtime_source_paths)
    engine_assets = tuple(_resolve_file(path) for path in config.engine_asset_paths)
    replay_paths = tuple(_resolve_file(path) for path in config.replay_paths)
    meta_paths = tuple(_resolve_file(path) for path in config.meta_paths)

    all_paths = _deduplicate_paths(
        (
            deck_path,
            checkpoint_path,
            *((belief_path,) if belief_path is not None else ()),
            *runtime_sources,
            *engine_assets,
            *replay_paths,
            *meta_paths,
        )
    )
    asset_fingerprints = {
        _display_path(path): file_sha256(path) for path in all_paths
    }
    runtime_source_fingerprints = _select_fingerprints(
        runtime_sources,
        asset_fingerprints,
    )
    engine_asset_fingerprints = _select_fingerprints(
        engine_assets,
        asset_fingerprints,
    )
    replay_asset_fingerprints = _select_fingerprints(
        replay_paths,
        asset_fingerprints,
    )
    meta_asset_fingerprints = _select_fingerprints(
        meta_paths,
        asset_fingerprints,
    )

    engine_fp = fingerprint_payload({"assets": engine_asset_fingerprints})
    bundle_payload = {
        "deck_sha256": asset_fingerprints[_display_path(deck_path)],
        "checkpoint_sha256": asset_fingerprints[_display_path(checkpoint_path)],
        "belief_sha256": (
            asset_fingerprints[_display_path(belief_path)]
            if belief_path is not None
            else None
        ),
        "search": config.resolved_search.model_dump(mode="json"),
        "runtime": config.runtime_definition,
        "runtime_sources": runtime_source_fingerprints,
        "engine_fp": engine_fp,
    }
    bundle_fp = fingerprint_payload(bundle_payload)
    panel_fp = fingerprint_payload({"panel": config.panel_definition})
    replay_fp = fingerprint_payload(
        {
            "assets": replay_asset_fingerprints,
            "definition": config.replay_definition,
        }
    )
    meta_fp = fingerprint_payload(
        {
            "assets": meta_asset_fingerprints,
            "definition": config.meta_definition,
        }
    )
    environment = environment_identity()
    env_fp = fingerprint_payload(environment)
    campaign_payload = {
        "protocol": config.protocol,
        "experiment_id": config.experiment_id,
        "bundle_fp": bundle_fp,
        "panel_fp": panel_fp,
        "replay_fp": replay_fp,
        "meta_fp": meta_fp,
        "engine_fp": engine_fp,
        "env_fp": env_fp,
    }
    campaign_fp = fingerprint_payload(campaign_payload)
    stage_payload = {
        "campaign_fp": campaign_fp,
        "stage": config.stage,
        "parameters": config.stage_parameters,
    }
    stage_fp = fingerprint_payload(stage_payload)
    return {
        "protocol": config.protocol,
        "experiment_id": config.experiment_id,
        "stage": config.stage,
        "asset_fp": asset_fingerprints,
        "bundle_fp": bundle_fp,
        "panel_fp": panel_fp,
        "replay_fp": replay_fp,
        "meta_fp": meta_fp,
        "engine_fp": engine_fp,
        "env_fp": env_fp,
        "campaign_fp": campaign_fp,
        "stage_fp": stage_fp,
        "bundle": bundle_payload,
        "panel": config.panel_definition,
        "replay": {
            "assets": replay_asset_fingerprints,
            "definition": config.replay_definition,
        },
        "meta": {
            "assets": meta_asset_fingerprints,
            "definition": config.meta_definition,
        },
        "engine_assets": engine_asset_fingerprints,
        "environment": environment,
        "stage_parameters": config.stage_parameters,
    }


def file_sha256(path: Path) -> str:
    """Return a streaming SHA256 digest for one immutable asset."""
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_identity_atomic(path: Path, identity: Mapping[str, Any]) -> None:
    """Publish a small human-readable identity manifest atomically."""
    resolved = records.repo_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_suffix(resolved.suffix + ".tmp")
    temporary.write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(resolved)


def environment_identity() -> dict[str, Any]:
    """Capture the host, runtime, thread, CPU, and source revision identity."""
    torch_version: str | None = None
    torch_threads: int | None = None
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_threads = int(torch.get_num_threads())
    except ImportError:
        pass
    affinity: list[int] | None = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(int(cpu) for cpu in os.sched_getaffinity(0))
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "torch": torch_version,
        "torch_threads": torch_threads,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "affinity": affinity,
        "cpu_governor": _cpu_governor(),
        "git_revision": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(_git_output("status", "--porcelain")),
    }


def _cpu_governor() -> str | None:
    path = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor")
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=records.REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def fingerprint_payload(payload: Any) -> str:
    """Hash one JSON-compatible semantic identity payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_optional_file(path: Path | None) -> Path | None:
    return _resolve_file(path) if path is not None else None


def _resolve_file(path: Path) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"campaign identity input does not exist: {resolved}")
    return resolved


def _deduplicate_paths(paths: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(dict.fromkeys(paths))


def _select_fingerprints(
    paths: Sequence[Path],
    fingerprints: Mapping[str, str],
) -> dict[str, str]:
    return {_display_path(path): fingerprints[_display_path(path)] for path in paths}


def _display_path(path: Path) -> str:
    return records.display_path(path)
