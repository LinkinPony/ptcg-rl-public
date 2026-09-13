"""Prepare and validate immutable checkpoint-selection runtime assets."""

from __future__ import annotations

import gc
import glob
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from ptcg_rl.actions.selection import is_forced, is_legal_action
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.context import observation_with_context
from ptcg_rl.agent.search.policy_inputs import canonical_inputs_bitwise_equal
from ptcg_rl.context import (
    GameContext,
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import (
    DEFAULT_CHUNK_SIZE,
    iter_replay_steps,
    replay_stub,
)
from ptcg_rl.evaluation.checkpoint_asset_models import (
    CheckpointAssetAuditConfig,
    CheckpointAssetInput,
)
from ptcg_rl.evaluation.checkpoint_asset_protocol import (
    run_checkpoint_submission_protocol,
)
from ptcg_rl.evaluation.search_identity import (
    environment_identity,
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.submission.checkpoint_assets import (
    RuntimeCheckpointExportConfig,
    export_runtime_checkpoint,
)


def prepare_checkpoint_selection_assets(
    config: CheckpointAssetAuditConfig,
) -> dict[str, Any]:
    """Export assets, audit replay decisions, and validate Kaggle archives."""
    manifest_path = records.repo_path(config.output_manifest_path)
    parquet_path = records.repo_path(config.output_parquet_path)
    _require_absent(manifest_path)
    _require_absent(parquet_path)
    roots, replay_manifest = sample_replay_roots(config)
    if len(roots) != config.max_decisions:
        raise ValueError(
            f"asset parity sampled {len(roots)} decisions, expected {config.max_decisions}"
        )

    all_rows: list[dict[str, Any]] = []
    checkpoint_records: dict[str, dict[str, Any]] = {}
    for checkpoint in config.checkpoints:
        raw_path = records.repo_path(checkpoint.raw_checkpoint_path)
        asset_path = records.repo_path(checkpoint.asset_path)
        archive_path = records.repo_path(checkpoint.protocol_archive_path)
        _require_absent(asset_path)
        _require_absent(archive_path)
        export_summary = export_runtime_checkpoint(
            RuntimeCheckpointExportConfig(
                source_checkpoint=raw_path,
                output_checkpoint=asset_path,
                precision="fp16",
            )
        )
        parity_summary, parity_rows = audit_checkpoint_pair(
            checkpoint.checkpoint_tag,
            raw_path=raw_path,
            asset_path=asset_path,
            roots=roots,
            config=config,
        )
        all_rows.extend(parity_rows)
        protocol_result = run_checkpoint_submission_protocol(
            checkpoint.checkpoint_tag,
            asset_path=asset_path,
            archive_path=archive_path,
            config=config,
        )
        checkpoint_records[checkpoint.checkpoint_tag] = {
            "raw_checkpoint_path": records.display_path(raw_path),
            "raw_checkpoint_sha256": file_sha256(raw_path),
            "asset_path": records.display_path(asset_path),
            "asset_sha256": file_sha256(asset_path),
            "storage_precision": "fp16",
            "compute_precision": "fp32",
            "model_config_fp": _model_config_fingerprint(raw_path),
            "export": dict(export_summary),
            "parity": parity_summary,
            "submission_protocol": protocol_result,
        }
    write_parity_rows(parquet_path, all_rows)
    summary = {
        "protocol": "CHECKPOINT-SELECTION-v1-ASSET-AUDIT",
        "experiment_id": config.experiment_id,
        "environment": environment_identity(),
        "checkpoints": checkpoint_records,
        "shared_assets": {
            "deck_path": records.display_path(records.repo_path(config.deck_path)),
            "deck_sha256": file_sha256(records.repo_path(config.deck_path)),
            "belief_summary_path": records.display_path(
                records.repo_path(config.belief_summary_path)
            ),
            "belief_summary_sha256": file_sha256(
                records.repo_path(config.belief_summary_path)
            ),
            "static_features_path": records.display_path(
                records.repo_path(config.static_features_path)
            ),
            "static_features_sha256": file_sha256(
                records.repo_path(config.static_features_path)
            ),
        },
        "replay_sample": replay_manifest,
        "parity_rows_path": records.display_path(parquet_path),
        "parity_rows_sha256": file_sha256(parquet_path),
        "all_assets_ready": all(
            bool(record["parity"]["passed"])
            and bool(record["submission_protocol"]["validated"])
            for record in checkpoint_records.values()
        ),
    }
    write_identity_atomic(manifest_path, summary)
    return summary


class _ReplayRoot:
    def __init__(
        self,
        *,
        episode_id: int,
        step_index: int,
        seat: int,
        phase: str,
        observation: Mapping[str, Any],
    ) -> None:
        self.episode_id = episode_id
        self.step_index = step_index
        self.seat = seat
        self.phase = phase
        self.observation = observation


def sample_replay_roots(
    config: CheckpointAssetAuditConfig,
) -> tuple[list[_ReplayRoot], dict[str, Any]]:
    replay_paths = tuple(
        Path(path)
        for path in sorted(glob.glob(str(records.repo_path(Path(config.replay_glob)))))
    )
    if not replay_paths:
        raise ValueError("no replay files matched checkpoint asset parity audit")
    belief = OpponentBeliefFeatureProducer.from_config(
        OpponentBeliefFeatureConfig(
            deck_signature_summary_path=records.repo_path(
                config.belief_summary_path
            )
        )
    )
    roots: list[_ReplayRoot] = []
    used_replays: list[Path] = []
    for replay_path in replay_paths:
        if len(roots) >= config.max_decisions:
            break
        metadata = replay_stub(replay_path, chunk_size=DEFAULT_CHUNK_SIZE)
        seat = _team_seat(metadata, config.team_name)
        episode_id = int(_mapping(metadata.get("info")).get("EpisodeId", 0))
        context = GameContext(player_index=seat)
        phase_counts: Counter[str] = Counter()
        replay_used = False
        for step_index, sides in iter_replay_steps(
            replay_path, chunk_size=DEFAULT_CHUNK_SIZE
        ):
            if len(roots) >= config.max_decisions or seat >= len(sides):
                break
            side = sides[seat]
            observation = _mapping(side.get("observation"))
            if (
                str(side.get("status", "")) != "ACTIVE"
                or not _is_own_decision(observation, seat)
            ):
                continue
            context_features = context.update(observation)
            select = _mapping(observation.get("select"))
            if not select or is_forced(select):
                continue
            phase = _phase(observation, config)
            if phase_counts[phase] >= config.decisions_per_phase_per_replay:
                continue
            phase_counts[phase] += 1
            enriched = belief.augment(observation, context_features)
            roots.append(
                _ReplayRoot(
                    episode_id=episode_id,
                    step_index=step_index,
                    seat=seat,
                    phase=phase,
                    observation=_mapping(
                        observation_with_context(observation, enriched)
                    ),
                )
            )
            replay_used = True
        if replay_used:
            used_replays.append(replay_path)
    return roots, {
        "team_name": config.team_name,
        "decisions": len(roots),
        "near_tie_logit_margin": config.near_tie_logit_margin,
        "minimum_top1_match_rate": config.min_top1_match_rate,
        "replays": [
            {
                "path": records.display_path(path),
                "sha256": file_sha256(path),
            }
            for path in used_replays
        ],
    }


def audit_checkpoint_pair(
    checkpoint_tag: str,
    *,
    raw_path: Path,
    asset_path: Path,
    roots: Sequence[_ReplayRoot],
    config: CheckpointAssetAuditConfig,
    raw_device: str = "cpu",
    asset_device: str = "cpu",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    own_deck = records.read_deck(records.repo_path(config.deck_path))
    raw_policy = CheckpointPolicy(
        raw_path,
        device=raw_device,
        own_deck=own_deck,
    )
    asset_policy = CheckpointPolicy(
        asset_path,
        device=asset_device,
        own_deck=own_deck,
    )
    rows: list[dict[str, Any]] = []
    non_tied = 0
    non_tied_matches = 0
    legal_mask_matches = 0
    legal_actions = 0
    max_logit_error = 0.0
    for root_index, root in enumerate(roots):
        observation = root.observation
        raw_input = raw_policy.canonical_input(observation)
        asset_input = asset_policy.canonical_input(observation)
        canonical_match = (
            raw_input is not None
            and asset_input is not None
            and canonical_inputs_bitwise_equal(raw_input, asset_input)
        )
        raw_logits = raw_policy.first_step_logits(observation)
        asset_logits = asset_policy.first_step_logits(observation)
        raw_mask = tuple(math.isfinite(value) for value in raw_logits)
        asset_mask = tuple(math.isfinite(value) for value in asset_logits)
        legal_mask_match = raw_mask == asset_mask and bool(raw_mask)
        legal_mask_matches += int(legal_mask_match)
        finite_errors = [
            abs(raw_value - asset_value)
            for raw_value, asset_value, is_valid in zip(
                raw_logits, asset_logits, raw_mask, strict=True
            )
            if is_valid
        ]
        logit_error = max(finite_errors, default=0.0)
        max_logit_error = max(max_logit_error, logit_error)
        valid_raw_logits = sorted(
            (value for value, valid in zip(raw_logits, raw_mask, strict=True) if valid),
            reverse=True,
        )
        margin = (
            valid_raw_logits[0] - valid_raw_logits[1]
            if len(valid_raw_logits) >= 2
            else math.inf
        )
        tied = margin <= config.near_tie_logit_margin
        raw_action = raw_policy.select_action(observation)
        asset_action = asset_policy.select_action(observation)
        raw_legal = is_legal_action(observation.get("select"), raw_action)
        asset_legal = is_legal_action(observation.get("select"), asset_action)
        legal_actions += int(raw_legal and asset_legal)
        top1_match = raw_action == asset_action
        if not tied:
            non_tied += 1
            non_tied_matches += int(top1_match)
        rows.append(
            {
                "checkpoint_tag": checkpoint_tag,
                "root_index": root_index,
                "episode_id": root.episode_id,
                "step_index": root.step_index,
                "seat": root.seat,
                "phase": root.phase,
                "canonical_input_match": canonical_match,
                "legal_mask_match": legal_mask_match,
                "raw_action_legal": raw_legal,
                "asset_action_legal": asset_legal,
                "near_tie": tied,
                "raw_first_step_margin": margin,
                "top1_match": top1_match,
                "max_logit_abs_error": logit_error,
                "raw_action": list(raw_action),
                "asset_action": list(asset_action),
            }
        )
    match_rate = non_tied_matches / non_tied if non_tied else 0.0
    passed = (
        len(rows) == config.max_decisions
        and legal_mask_matches == len(rows)
        and legal_actions == len(rows)
        and all(bool(row["canonical_input_match"]) for row in rows)
        and non_tied > 0
        and match_rate >= config.min_top1_match_rate
    )
    summary = {
        "decisions": len(rows),
        "non_near_tied_decisions": non_tied,
        "non_near_tied_top1_matches": non_tied_matches,
        "non_near_tied_top1_match_rate": match_rate,
        "legal_mask_matches": legal_mask_matches,
        "both_actions_legal": legal_actions,
        "max_logit_abs_error": max_logit_error,
        "passed": passed,
    }
    del raw_policy, asset_policy
    gc.collect()
    if not passed:
        raise RuntimeError(f"checkpoint asset parity failed for {checkpoint_tag}: {summary}")
    return summary, rows


def write_parity_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(
        pa.Table.from_pylist([dict(row) for row in rows]),
        temporary,
        compression="zstd",
    )
    temporary.replace(path)


def _model_config_fingerprint(path: Path) -> str:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        return fingerprint_payload(None)
    model_config = checkpoint.get("model_config")
    model_dump = getattr(model_config, "model_dump", None)
    if callable(model_dump):
        model_config = model_dump(mode="json")
    return fingerprint_payload(model_config)


def _team_seat(metadata: Mapping[str, Any], team_name: str) -> int:
    names = _sequence(_mapping(metadata.get("info")).get("TeamNames"))
    matches = [index for index, name in enumerate(names) if str(name) == team_name]
    if len(matches) != 1:
        raise ValueError(f"expected one team_name={team_name!r} seat, found {matches}")
    return matches[0]


def _is_own_decision(observation: Mapping[str, Any], seat: int) -> bool:
    return (
        isinstance(observation.get("select"), Mapping)
        and _int_field(observation.get("current"), "yourIndex", -1) == seat
    )


def _phase(
    observation: Mapping[str, Any], config: CheckpointAssetAuditConfig
) -> str:
    turn = _int_field(observation.get("current"), "turn", -1)
    if turn <= config.early_turn_max:
        return "early"
    if turn <= config.mid_turn_max:
        return "mid"
    return "late"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    item = value.get(name, default) if isinstance(value, Mapping) else default
    return int(item) if item is not None else default


def _require_absent(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"immutable checkpoint asset artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


__all__ = [
    "CheckpointAssetAuditConfig",
    "CheckpointAssetInput",
    "audit_checkpoint_pair",
    "prepare_checkpoint_selection_assets",
    "sample_replay_roots",
    "write_parity_rows",
]
