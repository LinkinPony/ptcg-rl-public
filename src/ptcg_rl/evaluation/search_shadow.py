"""Replay-driven exact-runtime P0 shadow-search integration smoke."""

from __future__ import annotations

import glob
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, field_validator

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.runtime import ActTimeConfig, CheckpointPolicy, PolicyRuntimeAgent
from ptcg_rl.agent.search.config import MacroSearchConfig, SearchRuntimeConfig
from ptcg_rl.belief.sampling import BeliefSamplerConfig
from ptcg_rl.context import OpponentBeliefFeatureConfig
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import (
    DEFAULT_CHUNK_SIZE,
    iter_replay_steps,
    replay_stub,
)


class SearchShadowSmokeConfig(BaseModel):
    """Hydra-backed deployment-runtime shadow smoke configuration."""

    model_config = ConfigDict(extra="forbid")

    replay_paths: tuple[Path, ...] = ()
    replay_glob: str = (
        "outputs/kaggle_submission_replays/54498922_comfey_v12395/*.json"
    )
    team_name: str | None = "Marshall Maximizer"
    seat_index: int | None = None
    max_replays: int = 4
    max_shadow_roots: int = 32
    deck_path: Path = Path(
        "docs/experiments/rl_dynamic_deck_pool_20260708/decks/"
        "29_comfey_yveltal_shaymin_4f8e151b4dd0.csv"
    )
    checkpoint_path: Path = Path(
        "outputs/inference_time_search/p0/assets/policy_v12395.pt"
    )
    belief_summary_path: Path = Path(
        "docs/experiments/rl_dynamic_deck_pool_20260708/"
        "deck_signature_summary.csv"
    )
    device: str = "cpu"
    seed: int = 0
    chunk_size: int = DEFAULT_CHUNK_SIZE
    macro: MacroSearchConfig = Field(
        default_factory=lambda: MacroSearchConfig(mode="shadow")
    )
    output_parquet: Path = Path(
        "outputs/inference_time_search/p0/shadow_smoke/roots.parquet"
    )
    output_summary: Path = Path(
        "outputs/inference_time_search/p0/shadow_smoke/summary.json"
    )
    compression: str = "zstd"

    @field_validator("max_replays", "max_shadow_roots", "chunk_size")
    @classmethod
    def positive_limits(cls, value: int) -> int:
        """Reject non-positive smoke bounds."""
        if value <= 0:
            raise ValueError("shadow smoke limits must be positive")
        return value

    @field_validator("seat_index")
    @classmethod
    def valid_seat(cls, value: int | None) -> int | None:
        """Restrict explicit seat selection."""
        if value is not None and value not in (0, 1):
            raise ValueError("seat_index must be 0 or 1")
        return value


def run_search_shadow_smoke(config: SearchShadowSmokeConfig) -> dict[str, Any]:
    """Compare disabled and shadow runtimes on identical replay observations."""
    paths = _resolve_replay_paths(config)
    deck_path = records.repo_path(config.deck_path)
    deck = records.read_deck(deck_path)
    checkpoint_path = records.repo_path(config.checkpoint_path)
    policy = CheckpointPolicy(
        checkpoint_path,
        device=config.device,
        own_deck=deck,
    )
    baseline_config, shadow_config = _runtime_configs(config)
    baseline = PolicyRuntimeAgent(config=baseline_config, policy=policy)
    shadow = PolicyRuntimeAgent(config=shadow_config, policy=policy)
    output_path = records.repo_path(config.output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    writer = pq.ParquetWriter(
        temporary_path,
        _shadow_schema(),
        compression=config.compression,
    )
    writer_open = True
    rows: list[dict[str, Any]] = []
    source_hash = hashlib.sha256()
    replays = 0
    shadow_roots = 0
    try:
        for replay_path in paths:
            if shadow_roots >= config.max_shadow_roots:
                break
            replays += 1
            source_hash.update(replay_path.name.encode("utf-8"))
            source_hash.update(_file_sha256(replay_path).encode("ascii"))
            metadata = replay_stub(replay_path, chunk_size=config.chunk_size)
            seat = _resolve_seat(metadata, config)
            episode_id = int(
                _mapping(metadata.get("info")).get("EpisodeId", replay_path.stem)
            )
            baseline.begin_game(player_index=seat, own_deck=deck)
            shadow.begin_game(player_index=seat, own_deck=deck)
            for step_index, sides in iter_replay_steps(
                replay_path,
                chunk_size=config.chunk_size,
            ):
                if shadow_roots >= config.max_shadow_roots or seat >= len(sides):
                    break
                side = sides[seat]
                observation = _mapping(side.get("observation"))
                if str(side.get("status", "")) != "ACTIVE" or not _is_own_decision(
                    observation,
                    seat,
                ):
                    continue
                baseline_action = tuple(baseline.act(observation))
                shadow_action = tuple(shadow.act(observation))
                telemetry = shadow.last_search_telemetry
                if telemetry.planned_quota_seconds <= 0.0:
                    continue
                shadow_roots += 1
                select = observation.get("select")
                rows.append(
                    {
                        "episode_id": episode_id,
                        "step_index": step_index,
                        "seat": seat,
                        "turn": _int_field(observation.get("current"), "turn", -1),
                        "select_context": _int_field(select, "context", -1),
                        "baseline_action": list(baseline_action),
                        "shadow_action": list(shadow_action),
                        "action_match": (
                            shadow.last_base_action is not None
                            and tuple(shadow.last_base_action) == shadow_action
                        ),
                        "baseline_diagnostic_match": baseline_action == shadow_action,
                        "baseline_legal": is_legal_action(select, baseline_action),
                        "shadow_legal": is_legal_action(select, shadow_action),
                        "stop_reason": telemetry.stop_reason,
                        "planned_quota_seconds": telemetry.planned_quota_seconds,
                        "actual_search_seconds": telemetry.actual_search_seconds,
                        "whole_act_seconds": telemetry.whole_act_seconds,
                        "bank_spent_seconds": telemetry.bank_spent_seconds,
                        "bank_left_seconds": telemetry.bank_left_seconds,
                        "deadline_overshoot_seconds": telemetry.deadline_overshoot_seconds,
                        "candidates": telemetry.candidates,
                        "worlds_requested": telemetry.worlds_requested,
                        "worlds_completed": telemetry.worlds_completed,
                        "transitions": telemetry.transitions,
                        "engine_sessions": telemetry.engine_sessions,
                        "state_pool_peak": telemetry.state_pool_peak,
                        "state_leaks": telemetry.state_leaks,
                        "fallback_available": telemetry.fallback_available,
                    }
                )
                if len(rows) >= 32:
                    writer.write_table(pa.Table.from_pylist(rows, schema=_shadow_schema()))
                    rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=_shadow_schema()))
            rows.clear()
        writer.close()
        writer_open = False
        temporary_path.replace(output_path)
    finally:
        if writer_open:
            writer.close()
        if temporary_path.exists():
            temporary_path.unlink()

    table = pq.read_table(output_path)
    result_rows = table.to_pylist()
    summary = _shadow_summary(
        config,
        result_rows,
        replays=replays,
        source_fingerprint=source_hash.hexdigest(),
        output_path=output_path,
    )
    _write_json_atomic(records.repo_path(config.output_summary), summary)
    return summary


def _runtime_configs(
    config: SearchShadowSmokeConfig,
) -> tuple[ActTimeConfig, ActTimeConfig]:
    belief_path = records.repo_path(config.belief_summary_path)
    belief = OpponentBeliefFeatureConfig(deck_signature_summary_path=belief_path)
    sampler = BeliefSamplerConfig(
        mode="archetype",
        prior_deck_signature_summary_path=belief_path,
    )
    baseline_search = SearchRuntimeConfig(
        sampler=sampler,
        macro=config.macro.model_copy(update={"mode": "disabled"}),
    )
    shadow_search = SearchRuntimeConfig(sampler=sampler, macro=config.macro)
    return (
        ActTimeConfig(
            deck_path=records.repo_path(config.deck_path),
            checkpoint_path=records.repo_path(config.checkpoint_path),
            seed=config.seed,
            prewarm_on_startup=False,
            belief=belief,
            search=baseline_search,
        ),
        ActTimeConfig(
            deck_path=records.repo_path(config.deck_path),
            checkpoint_path=records.repo_path(config.checkpoint_path),
            seed=config.seed,
            prewarm_on_startup=False,
            belief=belief,
            search=shadow_search,
        ),
    )


def _shadow_summary(
    config: SearchShadowSmokeConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    replays: int,
    source_fingerprint: str,
    output_path: Path,
) -> dict[str, Any]:
    roots = len(rows)
    action_matches = sum(bool(row["action_match"]) for row in rows)
    illegal = sum(
        not bool(row["baseline_legal"]) or not bool(row["shadow_legal"])
        for row in rows
    )
    state_leaks = sum(int(row["state_leaks"]) for row in rows)
    fallback_failures = sum(not bool(row["fallback_available"]) for row in rows)
    max_overshoot = max(
        (float(row["deadline_overshoot_seconds"]) for row in rows),
        default=0.0,
    )
    max_bank_spent = max(
        (float(row["bank_spent_seconds"]) for row in rows),
        default=0.0,
    )
    checks = {
        "root_count": roots == config.max_shadow_roots,
        "saved_action_preserved": action_matches == roots,
        "legal_actions": illegal == 0,
        "fallback_available": fallback_failures == 0,
        "deadline_overshoot": max_overshoot == 0.0,
        "search_bank": max_bank_spent <= config.macro.budget.bank_limit_seconds,
        "state_lifecycle": state_leaks == 0,
    }
    return {
        "protocol": "ITS-P0-SHADOW-SMOKE-v1",
        "roots": roots,
        "replays": replays,
        "action_match_rate": action_matches / float(roots) if roots else 0.0,
        "baseline_diagnostic_match_rate": (
            sum(bool(row["baseline_diagnostic_match"]) for row in rows) / float(roots)
            if roots
            else 0.0
        ),
        "illegal_roots": illegal,
        "fallback_failures": fallback_failures,
        "state_leaks": state_leaks,
        "max_state_pool_peak": max(
            (int(row["state_pool_peak"]) for row in rows),
            default=0,
        ),
        "max_deadline_overshoot_seconds": max_overshoot,
        "max_bank_spent_seconds": max_bank_spent,
        "stop_reason_counts": dict(
            sorted(Counter(str(row["stop_reason"]) for row in rows).items())
        ),
        "decision_role": "diagnostic_only",
        "checks": checks,
        "diagnostic_warnings": [
            name for name, observed in checks.items() if not observed
        ],
        "source_fingerprint": source_fingerprint,
        "roots_path": records.display_path(output_path),
        "config": config.model_dump(mode="json"),
    }


def _shadow_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("episode_id", pa.int64()),
            pa.field("step_index", pa.int32()),
            pa.field("seat", pa.int8()),
            pa.field("turn", pa.int16()),
            pa.field("select_context", pa.int16()),
            pa.field("baseline_action", pa.list_(pa.int16())),
            pa.field("shadow_action", pa.list_(pa.int16())),
            pa.field("action_match", pa.bool_()),
            pa.field("baseline_diagnostic_match", pa.bool_()),
            pa.field("baseline_legal", pa.bool_()),
            pa.field("shadow_legal", pa.bool_()),
            pa.field("stop_reason", pa.string()),
            pa.field("planned_quota_seconds", pa.float64()),
            pa.field("actual_search_seconds", pa.float64()),
            pa.field("whole_act_seconds", pa.float64()),
            pa.field("bank_spent_seconds", pa.float64()),
            pa.field("bank_left_seconds", pa.float64()),
            pa.field("deadline_overshoot_seconds", pa.float64()),
            pa.field("candidates", pa.int16()),
            pa.field("worlds_requested", pa.int16()),
            pa.field("worlds_completed", pa.int16()),
            pa.field("transitions", pa.int32()),
            pa.field("engine_sessions", pa.int32()),
            pa.field("state_pool_peak", pa.int16()),
            pa.field("state_leaks", pa.int16()),
            pa.field("fallback_available", pa.bool_()),
        ]
    )


def _resolve_replay_paths(config: SearchShadowSmokeConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        paths = tuple(records.repo_path(path) for path in config.replay_paths)
    else:
        paths = tuple(
            Path(path)
            for path in sorted(glob.glob(str(records.repo_path(Path(config.replay_glob)))))
        )
    paths = paths[: config.max_replays]
    if not paths:
        raise ValueError("no replay paths matched shadow smoke")
    return paths


def _resolve_seat(metadata: Mapping[str, Any], config: SearchShadowSmokeConfig) -> int:
    if config.seat_index is not None:
        return config.seat_index
    team_names = _sequence(_mapping(metadata.get("info")).get("TeamNames"))
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


def _is_own_decision(observation: Mapping[str, Any], seat: int) -> bool:
    return (
        isinstance(observation.get("select"), Mapping)
        and _int_field(observation.get("current"), "yourIndex", -1) == seat
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        while chunk := file_obj.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_field(value: Any, name: str, default: int) -> int:
    if isinstance(value, Mapping):
        item = value.get(name, default)
    else:
        item = getattr(value, name, default)
    return int(item) if item is not None else default
