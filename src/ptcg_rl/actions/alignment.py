"""Replay-level coverage audit for pointer-style option encoding."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.actions.encoding import StateTokenLayout, encode_options
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records
from ptcg_rl.engine.constants import OptionType

KNOWN_OPTION_TYPE_COUNT = len(OptionType)
KNOWN_CONTEXT_COUNT = 49


class OptionEncodingAuditConfig(BaseModel):
    """Config for replay option-encoding coverage audits."""

    model_config = ConfigDict(extra="forbid")

    replay_root: Path = Path("data/external/kaggle_top_episodes_daily")
    dates: list[str] = ["2026-06-16"]
    replay_paths: tuple[Path, ...] = ()
    max_replays: int | None = 150
    max_active_selects: int | None = None
    chunk_size: int = step_records.DEFAULT_CHUNK_SIZE
    max_examples: int = 20
    fail_on_encoding_errors: bool = True
    fail_on_unresolved_pointers: bool = True
    output_path: Path | None = Path("outputs/actions/encoding_audit/summary.json")

    @field_validator("dates")
    @classmethod
    def valid_dates(cls, values: list[str]) -> list[str]:
        """Reject malformed date strings early."""
        for value in values:
            datetime.strptime(value, "%Y-%m-%d")
        return values

    @field_validator("max_replays", "max_active_selects")
    @classmethod
    def valid_optional_positive(cls, value: int | None) -> int | None:
        """Reject non-positive optional limits."""
        if value is not None and value <= 0:
            raise ValueError("optional limits must be positive when set")
        return value

    @field_validator("chunk_size")
    @classmethod
    def valid_chunk_size(cls, value: int) -> int:
        """Reject non-positive parser chunk sizes."""
        if value <= 0:
            raise ValueError("chunk_size must be positive")
        return value

    @field_validator("max_examples")
    @classmethod
    def valid_max_examples(cls, value: int) -> int:
        """Reject negative example limits."""
        if value < 0:
            raise ValueError("max_examples must be non-negative")
        return value


@dataclass
class _AuditStats:
    """Mutable replay audit counters."""

    counters: Counter[str] = field(default_factory=Counter)
    combinations: Counter[tuple[int, int, int]] = field(default_factory=Counter)
    unresolved_combinations: Counter[tuple[int, int, int]] = field(
        default_factory=Counter
    )
    unknown_contexts: Counter[int] = field(default_factory=Counter)
    unknown_option_types: Counter[int] = field(default_factory=Counter)
    issue_examples: list[dict[str, Any]] = field(default_factory=list)
    error_examples: list[dict[str, Any]] = field(default_factory=list)

    def add_example(
        self,
        examples: list[dict[str, Any]],
        example: dict[str, Any],
        *,
        limit: int,
    ) -> None:
        """Append an example row while respecting the configured cap."""
        if len(examples) < limit:
            examples.append(example)


def run_option_encoding_audit(config: OptionEncodingAuditConfig) -> dict[str, Any]:
    """Audit option encoding coverage for replay observations."""
    replay_paths = _resolve_replay_paths(config)
    stats = _AuditStats()
    for replay_path in replay_paths:
        if _limit_reached(stats.counters["replays_seen"], config.max_replays):
            break
        _audit_replay(replay_path, config=config, stats=stats)
        if _limit_reached(
            stats.counters["active_selects"],
            config.max_active_selects,
        ):
            break

    report = _report(config, stats)
    _write_report(config.output_path, report)
    _raise_for_failures(config, stats)
    return report


def audit_replay_paths(
    replay_paths: Iterable[Path],
    *,
    config: OptionEncodingAuditConfig | None = None,
) -> dict[str, Any]:
    """Audit explicit replay paths; useful for tests and focused probes."""
    paths = tuple(replay_paths)
    active_config = config or OptionEncodingAuditConfig(
        replay_paths=paths,
        output_path=None,
    )
    stats = _AuditStats()
    for replay_path in paths:
        if _limit_reached(stats.counters["replays_seen"], active_config.max_replays):
            break
        _audit_replay(replay_path, config=active_config, stats=stats)
    report = _report(active_config, stats)
    _write_report(active_config.output_path, report)
    _raise_for_failures(active_config, stats)
    return report


def _resolve_replay_paths(config: OptionEncodingAuditConfig) -> tuple[Path, ...]:
    if config.replay_paths:
        return tuple(deck_records.repo_path(path) for path in config.replay_paths)
    replay_root = deck_records.repo_path(config.replay_root)
    return tuple(deck_records.iter_replay_paths(replay_root, set(config.dates)))


def _audit_replay(
    replay_path: Path,
    *,
    config: OptionEncodingAuditConfig,
    stats: _AuditStats,
) -> None:
    stats.counters["replays_seen"] += 1
    try:
        steps = step_records.iter_replay_steps(replay_path, chunk_size=config.chunk_size)
        for step_index, sides in steps:
            stats.counters["steps_seen"] += 1
            for player_index, side in enumerate(sides):
                if _limit_reached(
                    stats.counters["active_selects"],
                    config.max_active_selects,
                ):
                    return
                _audit_side(
                    side,
                    replay_path=replay_path,
                    step_index=step_index,
                    player_index=player_index,
                    config=config,
                    stats=stats,
                )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        stats.counters["replay_scan_errors"] += 1
        stats.add_example(
            stats.error_examples,
            {
                "replay_path": deck_records.display_path(replay_path),
                "error": str(exc),
            },
            limit=config.max_examples,
        )


def _audit_side(
    side: Mapping[str, Any],
    *,
    replay_path: Path,
    step_index: int,
    player_index: int,
    config: OptionEncodingAuditConfig,
    stats: _AuditStats,
) -> None:
    if str(side.get("status", "")) != "ACTIVE":
        return
    observation = _mapping(side.get("observation"))
    select = observation.get("select")
    if select is None:
        stats.counters["deck_registration_prompts"] += 1
        return
    if not isinstance(select, Mapping):
        stats.counters["invalid_select_shape"] += 1
        return

    stats.counters["active_selects"] += 1
    try:
        layout = StateTokenLayout.from_observation(observation)
        encoded = encode_options(select, layout)
    except (TypeError, ValueError, AttributeError) as exc:
        stats.counters["encoding_errors"] += 1
        stats.add_example(
            stats.error_examples,
            _prompt_example(
                replay_path=replay_path,
                step_index=step_index,
                player_index=player_index,
                select=select,
                extra={"error": str(exc)},
            ),
            limit=config.max_examples,
        )
        return

    options = _sequence(select.get("option"))
    if len(encoded) != len(options):
        stats.counters["option_count_mismatches"] += 1
        stats.add_example(
            stats.error_examples,
            _prompt_example(
                replay_path=replay_path,
                step_index=step_index,
                player_index=player_index,
                select=select,
                extra={
                    "encoded_options": len(encoded),
                    "raw_options": len(options),
                },
            ),
            limit=config.max_examples,
        )
        return

    select_type = _int_value(select.get("type"), -1)
    select_context = _int_value(select.get("context"), -1)
    if select_context < 0 or select_context >= KNOWN_CONTEXT_COUNT:
        stats.unknown_contexts[select_context] += 1
        stats.counters["oov_context_prompts"] += 1

    for option_index, option in enumerate(options):
        option_mapping = _mapping(option)
        option_type = _int_value(option_mapping.get("type"), -1)
        key = (select_type, select_context, option_type)
        stats.combinations[key] += 1
        stats.counters["options_seen"] += 1
        if option_type < 0 or option_type >= KNOWN_OPTION_TYPE_COUNT:
            stats.unknown_option_types[option_type] += 1
            stats.counters["oov_option_type_options"] += 1
            continue
        required_slots = _required_slot_count(option_mapping)
        resolved_slots = len(encoded[option_index].entity_slots)
        if resolved_slots < required_slots:
            stats.counters["unresolved_pointer_options"] += 1
            stats.unresolved_combinations[key] += 1
            stats.add_example(
                stats.issue_examples,
                _option_example(
                    replay_path=replay_path,
                    step_index=step_index,
                    player_index=player_index,
                    select_type=select_type,
                    select_context=select_context,
                    option_index=option_index,
                    option=option_mapping,
                    required_slots=required_slots,
                    resolved_slots=resolved_slots,
                ),
                limit=config.max_examples,
            )


def _required_slot_count(option: Mapping[str, Any]) -> int:
    option_type = _int_value(option.get("type"), -1)
    if option_type in {
        int(OptionType.PLAY),
        int(OptionType.CARD),
        int(OptionType.TOOL_CARD),
        int(OptionType.ENERGY_CARD),
        int(OptionType.ENERGY),
        int(OptionType.ABILITY),
        int(OptionType.DISCARD),
        int(OptionType.ATTACK),
    }:
        return 1
    if option_type == int(OptionType.SKILL):
        return 1 if _int_value(option.get("cardId")) == 0 else 0
    if option_type in {int(OptionType.ATTACH), int(OptionType.EVOLVE)}:
        return 2
    return 0


def _report(
    config: OptionEncodingAuditConfig,
    stats: _AuditStats,
) -> dict[str, Any]:
    summary = dict(sorted(stats.counters.items()))
    summary["unique_select_contexts"] = len(
        {key[1] for key in stats.combinations}
    )
    summary["unique_option_types"] = len({key[2] for key in stats.combinations})
    summary["unique_select_option_combinations"] = len(stats.combinations)
    return {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config.model_dump(mode="json"),
        "summary": summary,
        "select_option_combinations": _combination_rows(stats.combinations),
        "unresolved_pointer_combinations": _combination_rows(
            stats.unresolved_combinations
        ),
        "unknown_contexts": [
            {"select_context": key, "count": count}
            for key, count in sorted(stats.unknown_contexts.items())
        ],
        "unknown_option_types": [
            {"option_type": key, "count": count}
            for key, count in sorted(stats.unknown_option_types.items())
        ],
        "unresolved_pointer_examples": stats.issue_examples,
        "encoding_error_examples": stats.error_examples,
    }


def _combination_rows(
    counter: Counter[tuple[int, int, int]],
) -> list[dict[str, int]]:
    return [
        {
            "select_type": select_type,
            "select_context": select_context,
            "option_type": option_type,
            "count": count,
        }
        for (select_type, select_context, option_type), count in sorted(
            counter.items()
        )
    ]


def _raise_for_failures(
    config: OptionEncodingAuditConfig,
    stats: _AuditStats,
) -> None:
    if config.fail_on_encoding_errors and (
        stats.counters["encoding_errors"]
        or stats.counters["option_count_mismatches"]
        or stats.counters["replay_scan_errors"]
    ):
        raise ValueError("option encoding audit found encoding or scan errors")
    if config.fail_on_unresolved_pointers and stats.counters[
        "unresolved_pointer_options"
    ]:
        raise ValueError("option encoding audit found unresolved known pointers")


def _write_report(path: Path | None, report: Mapping[str, Any]) -> None:
    if path is None:
        return
    output_path = deck_records.repo_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _prompt_example(
    *,
    replay_path: Path,
    step_index: int,
    player_index: int,
    select: Mapping[str, Any],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "replay_path": deck_records.display_path(replay_path),
        "step_index": step_index,
        "player_index": player_index,
        "select_type": _int_value(select.get("type"), -1),
        "select_context": _int_value(select.get("context"), -1),
        **dict(extra),
    }


def _option_example(
    *,
    replay_path: Path,
    step_index: int,
    player_index: int,
    select_type: int,
    select_context: int,
    option_index: int,
    option: Mapping[str, Any],
    required_slots: int,
    resolved_slots: int,
) -> dict[str, Any]:
    return {
        "replay_path": deck_records.display_path(replay_path),
        "step_index": step_index,
        "player_index": player_index,
        "select_type": select_type,
        "select_context": select_context,
        "option_index": option_index,
        "option_type": _int_value(option.get("type"), -1),
        "required_slots": required_slots,
        "resolved_slots": resolved_slots,
        "area": _optional_int(option.get("area")),
        "index": _optional_int(option.get("index")),
        "player_index_field": _optional_int(option.get("playerIndex")),
        "card_id": _optional_int(option.get("cardId")),
        "serial": _optional_int(option.get("serial")),
    }


def _limit_reached(count: int, limit: int | None) -> bool:
    return limit is not None and count >= limit


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


def _int_value(value: Any, default: int = 0) -> int:
    return int(value) if value is not None else default


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None
