"""Replay selection, deterministic seeds, and S1 campaign publication."""

from __future__ import annotations

import glob
import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context import OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_counterfactual_config import (
    SearchCounterfactualConfig,
)
from ptcg_rl.evaluation.search_identity import (
    file_sha256,
    write_identity_atomic,
)


def write_campaign_artifacts(
    output_dir: Path,
    config: SearchCounterfactualConfig,
    identity: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    """Publish config, identity, summary, checksums, and completion manifest."""
    _write_yaml_atomic(
        output_dir / "resolved_config.yaml",
        config.model_dump(mode="json"),
    )
    write_identity_atomic(output_dir / "fingerprints.json", identity)
    write_identity_atomic(output_dir / "environment.json", identity["environment"])
    write_identity_atomic(output_dir / "summary.json", summary)
    output_files = {
        name: {
            "path": records.display_path(path),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in {
            "roots": output_dir / "roots.parquet",
            "candidates": output_dir / "candidates.parquet",
            "evaluations": output_dir / "evaluations.parquet",
        }.items()
    }
    manifest = {
        **identity,
        "runner_complete": summary["runner_complete"],
        "diagnostic_warnings": summary["diagnostic_warnings"],
        "output_files": output_files,
    }
    write_identity_atomic(output_dir / "manifest.json", manifest)


def prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    """Refuse to mix an immutable stage with existing published artifacts."""
    artifacts = (
        "roots.parquet",
        "candidates.parquet",
        "evaluations.parquet",
        "resolved_config.yaml",
        "fingerprints.json",
        "environment.json",
        "summary.json",
        "manifest.json",
    )
    existing = [output_dir / name for name in artifacts if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "immutable S1 output already exists; use a new experiment_id/output_dir: "
            f"{existing[0]}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in existing:
            path.unlink()


def resolve_replay_paths(config: SearchCounterfactualConfig) -> tuple[Path, ...]:
    """Resolve the frozen ordered replay panel."""
    if config.replay_paths:
        paths = tuple(records.repo_path(path) for path in config.replay_paths)
    else:
        paths = tuple(
            Path(path)
            for path in sorted(glob.glob(str(records.repo_path(Path(config.replay_glob)))))
        )
    if config.max_replays is not None:
        paths = paths[: config.max_replays]
    if not paths:
        raise ValueError("no replay paths matched counterfactual audit")
    return paths


def resolve_belief_config(
    config: OpponentBeliefFeatureConfig,
) -> OpponentBeliefFeatureConfig:
    """Resolve an optional belief prior relative to the repository."""
    if config.deck_signature_summary_path is None:
        return config
    return config.model_copy(
        update={
            "deck_signature_summary_path": records.repo_path(
                config.deck_signature_summary_path
            )
        }
    )


def resolve_sampler_config(config: BeliefSamplerConfig) -> BeliefSamplerConfig:
    """Resolve an optional sampling prior relative to the repository."""
    if config.prior_deck_signature_summary_path is None:
        return config
    return config.model_copy(
        update={
            "prior_deck_signature_summary_path": records.repo_path(
                config.prior_deck_signature_summary_path
            )
        }
    )


def belief_distributions(
    policy: CheckpointPolicy,
    observation: Mapping[str, Any],
    sampler: BeliefSampler,
) -> tuple[tuple[float, ...] | None, tuple[float, ...] | None]:
    """Return model belief heads only for model-based determinization."""
    if sampler.config.mode != "model":
        return None, None
    distributions = policy.belief_distributions(observation)
    return distributions if distributions is not None else (None, None)


def resolve_seat(
    metadata: Mapping[str, Any],
    config: SearchCounterfactualConfig,
) -> int:
    """Resolve one exact replay seat from an index or frozen team name."""
    if config.seat_index is not None:
        return config.seat_index
    team_names = sequence(mapping(metadata.get("info")).get("TeamNames"))
    matches = [
        index
        for index, team_name in enumerate(team_names)
        if str(team_name) == str(config.team_name)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one team_name={config.team_name!r} seat, found {matches}"
        )
    return matches[0]


def terminal_value(metadata: Mapping[str, Any], seat: int) -> float | None:
    """Return the resolved replay outcome from one seat's perspective."""
    rewards = sequence(metadata.get("rewards"))
    statuses = sequence(metadata.get("statuses"))
    if seat >= len(rewards) or seat >= len(statuses):
        return None
    if str(statuses[seat]) != "DONE" or rewards[seat] is None:
        return None
    return float(rewards[seat])


def episode_split(
    episode_id: int,
    *,
    seed: int,
    holdout_fraction: float,
) -> str:
    """Assign every root in an episode to one deterministic data split."""
    digest = hashlib.blake2b(
        f"{seed}\0{episode_id}".encode(),
        digest_size=8,
    ).digest()
    unit = int.from_bytes(digest, "little") / float(1 << 64)
    return "holdout" if unit < holdout_fraction else "dev"


def root_seed(seed: int, episode_id: int, step_index: int, purpose: str) -> int:
    """Derive independent reproducible RNG streams for one replay root."""
    digest = hashlib.blake2b(
        f"{seed}\0{episode_id}\0{step_index}\0{purpose}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def phase(observation: Mapping[str, Any], config: SearchCounterfactualConfig) -> str:
    """Classify an early/mid/late stratum from the engine turn counter."""
    turn = int_field(observation.get("current"), "turn", -1)
    if turn <= config.early_turn_max:
        return "early"
    if turn <= config.mid_turn_max:
        return "mid"
    return "late"


def is_own_decision(observation: Mapping[str, Any], seat: int) -> bool:
    """Return whether a replay observation is an actionable root for the seat."""
    return (
        isinstance(observation.get("select"), Mapping)
        and int_field(observation.get("current"), "yourIndex", -1) == seat
    )


def optional_float(value: Any) -> float | None:
    """Normalize an optional numeric value for metrics."""
    return float(value) if value is not None else None


def mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping view or an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def sequence(value: Any) -> Sequence[Any]:
    """Return a non-string sequence or an empty tuple."""
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def int_field(value: Any, name: str, default: int) -> int:
    """Read and normalize an integer field from mapping or object input."""
    if isinstance(value, Mapping):
        item = value.get(name, default)
    else:
        item = getattr(value, name, default)
    return int(item) if item is not None else default


def _write_yaml_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(dict(payload), sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
