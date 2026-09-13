"""Benchmark exact complete-action consequences through the native engine.

Run with:
    PYTHONPATH=data/sample_submission:src \
      python src/tools/native_exact_consequence_benchmark.py
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import statistics
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar, cast

import hydra
import pyarrow.dataset as pyarrow_dataset
import pyarrow.parquet as parquet
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.engine.native_macro import (
    NATIVE_MACRO_TRAILING_ACTION_ERROR,
    NativeMacro,
    NativeMacroBackend,
)
from ptcg_rl.engine.native_macro_validation import (
    effect_log_signature,
    search_api_macro_parity,
    target_attack_fact_matches,
)
from ptcg_rl.engine.native_planning_session import NativePlanningSessionLane
from ptcg_rl.engine.native_planning_session_request import NativePlanningSessionCaps
from ptcg_rl.engine.native_probe_features import native_transition_feature_vector
from ptcg_rl.engine.native_probe_payload import NativeProbeTransition
from ptcg_rl.engine.session import HiddenInformation

_PLACEHOLDER_CARD_ID = 1
_PLACEHOLDER_BASIC_POKEMON_ID = 463
_MAX_MACRO_STEPS = 32
_ValueT = TypeVar("_ValueT")


class NativeExactConsequenceBenchmarkConfig(BaseModel):
    """Hydra-backed benchmark configuration."""

    model_config = ConfigDict(extra="forbid")

    side_observations_path: Path
    steps_root: Path
    deck_path: Path
    deck_hash: str
    attack_id: int = 1285
    library_path: Path
    observed_case_limit: int = 128
    observed_repetitions: int = 4
    parity_case_limit: int = 16
    world_counts: tuple[int, ...] = (1, 3)
    counterfactual_candidate_limits: tuple[int, ...] = (
        1,
        4,
        8,
        16,
        32,
        64,
        128,
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    )
    target_transition_samples: int = 32768
    min_scale_repetitions: int = 3
    max_scale_repetitions: int = 128
    output_path: Path
    hydra: Mapping[str, Any] | None = None

    @field_validator(
        "observed_case_limit",
        "observed_repetitions",
        "parity_case_limit",
        "target_transition_samples",
        "min_scale_repetitions",
        "max_scale_repetitions",
    )
    @classmethod
    def positive_limits(cls, value: int) -> int:
        """Reject non-positive benchmark limits."""
        if value <= 0:
            raise ValueError("benchmark limits must be positive")
        return value

    @field_validator("world_counts", "counterfactual_candidate_limits")
    @classmethod
    def positive_sequences(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require non-empty positive count sequences."""
        if not value or any(item <= 0 for item in value):
            raise ValueError("benchmark count sequences must be positive")
        return value


@dataclass(frozen=True)
class ReplayMacroCase:
    """One replay-proven Rocket Feathers complete macro."""

    episode_id: int
    step_index: int
    player_index: int
    state_token: str
    hidden: HiddenInformation
    observed_macro: NativeMacro
    root_option_count: int
    root_action: tuple[int, ...]
    discard_option_count: int
    discard_min_count: int
    discard_max_count: int
    discard_action: tuple[int, ...]


@dataclass(frozen=True)
class TimingSample:
    """One end-to-end native exact-consequence timing sample."""

    native_ms: float
    end_to_end_ms: float
    payload_bytes: int
    transitions: int
    resolved: int
    errors: int
    completion_steps: int


def run_benchmark(
    config: NativeExactConsequenceBenchmarkConfig,
) -> dict[str, Any]:
    """Run replay correctness checks and exact-path timing measurements."""
    own_deck = _read_deck(config.deck_path)
    cases, corpus = _load_replay_cases(config, own_deck)
    if not cases:
        raise RuntimeError("no selected target-attack macro was recovered")
    backend = NativeMacroBackend(library_path=config.library_path)
    valid_cases, validation = _validate_replay_cases(
        backend,
        cases,
        attack_id=config.attack_id,
    )
    if not valid_cases:
        raise RuntimeError("no recovered target-attack macro reached its boundary")
    selected_cases = _evenly_spaced(valid_cases, config.observed_case_limit)

    parity_cases = _evenly_spaced(valid_cases, config.parity_case_limit)
    parity = _search_api_parity(
        backend,
        parity_cases,
        attack_id=config.attack_id,
    )
    discard_permutation_parity = _discard_permutation_parity(
        backend,
        selected_cases,
    )
    continuation_parity = _planning_session_parity(
        backend,
        parity_cases,
        library_path=config.library_path,
    )
    observed = _benchmark_observed_macros(backend, selected_cases, config)
    root_options = _benchmark_all_root_options(backend, selected_cases, config)
    counterfactual = _benchmark_counterfactual_scale(backend, valid_cases, config)
    summary = {
        "scope": {
            "deck_hash": config.deck_hash,
            "attack_id": config.attack_id,
            "library_path": str(config.library_path),
            "hidden_world_contract": (
                "fixed count-correct placeholder determinizations; exact for the "
                "tested Rocket Feathers transition, not a belief-quality test"
            ),
            **corpus,
            "replay_macro_validation": validation,
            "timed_observed_macros": len(selected_cases),
        },
        "search_api_parity": parity,
        "unordered_discard_permutation_parity": discard_permutation_parity,
        "planning_session_continuation_parity": continuation_parity,
        "observed_complete_macros": observed,
        "all_root_options_fixed_completion": root_options,
        "rocket_discard_lattice": counterfactual,
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if parity["failure_count"]:
        raise AssertionError("native complete-macro parity failed")
    if not discard_permutation_parity["case_count"]:
        raise AssertionError("discard permutation parity had no eligible cases")
    if discard_permutation_parity["failure_count"]:
        raise AssertionError("discard permutation parity failed")
    if continuation_parity["failure_count"]:
        raise AssertionError("native planning-session continuation parity failed")
    return summary


def _discard_permutation_parity(
    backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
) -> Mapping[str, Any]:
    """Prove reversed order preserves one selected discard set exactly."""
    eligible = tuple(
        case
        for case in cases
        if len(case.discard_action) > 1
        and tuple(reversed(case.discard_action)) != case.discard_action
    )
    failures: list[Mapping[str, Any]] = []
    exact_states = 0
    exact_features = 0
    exact_logs = 0
    failure_count = 0
    for case in eligible:
        permuted_macro = list(case.observed_macro)
        permuted_macro[1] = tuple(reversed(case.discard_action))
        original, permuted = backend.run(
            case.state_token,
            hidden_worlds=[case.hidden],
            macros=[case.observed_macro, tuple(permuted_macro)],
        ).transitions
        states_match = original.after_state == permuted.after_state
        exact_states += int(states_match)
        original_features = native_transition_feature_vector(
            before_state=original.before_state,
            after_state=original.after_state,
            logs=original.logs,
        )
        permuted_features = native_transition_feature_vector(
            before_state=permuted.before_state,
            after_state=permuted.after_state,
            logs=permuted.logs,
        )
        features_match = original_features == permuted_features
        exact_features += int(features_match)
        logs_match = effect_log_signature(original.logs) == effect_log_signature(
            permuted.logs
        )
        exact_logs += int(logs_match)
        if (
            original.error != 0
            or permuted.error != 0
            or not original.resolved
            or not permuted.resolved
            or not states_match
            or not features_match
            or not logs_match
        ):
            failure_count += 1
            if len(failures) < 8:
                failures.append(
                    {
                        "episode_id": case.episode_id,
                        "step_index": case.step_index,
                        "discard_count": len(case.discard_action),
                        "original_error": original.error,
                        "permuted_error": permuted.error,
                        "original_resolved": original.resolved,
                        "permuted_resolved": permuted.resolved,
                        "state_match": states_match,
                        "feature_match": features_match,
                        "log_match": logs_match,
                    }
                )
    return {
        "context": int(SelectContext.DISCARD),
        "case_count": len(eligible),
        "exact_state_count": exact_states,
        "exact_feature_count": exact_features,
        "exact_log_count": exact_logs,
        "failure_count": failure_count,
        "failure_examples": failures,
    }


def _load_replay_cases(
    config: NativeExactConsequenceBenchmarkConfig,
    own_deck: tuple[int, ...],
) -> tuple[list[ReplayMacroCase], dict[str, Any]]:
    side_table = parquet.read_table(
        config.side_observations_path,
        columns=["date", "episode_id", "player_index"],
        filters=[("deck_hash", "=", config.deck_hash)],
    )
    exact_sides: dict[str, set[tuple[int, int]]] = defaultdict(set)
    for row in side_table.to_pylist():
        exact_sides[str(row["date"])].add(
            (int(row["episode_id"]), int(row["player_index"]))
        )

    columns = [
        "date",
        "episode_id",
        "step_index",
        "player_index",
        "search_begin_input",
        "action",
        "select_context",
        "select_min_count",
        "select_max_count",
        "select_option_count",
        "option_attack_id",
        "player0_deck_count",
        "player0_hand_count",
        "player0_prize_ids",
        "player0_active_ids",
        "player1_deck_count",
        "player1_hand_count",
        "player1_prize_ids",
        "player1_active_ids",
        "opponent_deck_ids",
        "godview_opp_hand_ids",
    ]
    episode_rows: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    scanned_dates = 0
    for date, sides in sorted(exact_sides.items()):
        files = sorted((config.steps_root / date).glob("steps-*.parquet"))
        if not files:
            continue
        episode_ids = sorted({episode_id for episode_id, _ in sides})
        episode_field = cast(Any, pyarrow_dataset).field("episode_id")
        table = pyarrow_dataset.dataset(
            [str(path) for path in files], format="parquet"
        ).to_table(
            columns=columns,
            filter=episode_field.isin(episode_ids),
        )
        scanned_dates += 1
        for raw_row in table.to_pylist():
            row = cast(Mapping[str, Any], raw_row)
            episode_rows[(date, int(row["episode_id"]))].append(row)

    cases: list[ReplayMacroCase] = []
    legal_roots = 0
    selected_roots = 0
    for (date, episode_id), rows in episode_rows.items():
        ordered = sorted(rows, key=lambda row: int(row["step_index"]))
        for root_index, row in enumerate(ordered):
            player_index = int(row["player_index"])
            if (episode_id, player_index) not in exact_sides[date]:
                continue
            attack_ids = tuple(row.get("option_attack_id") or ())
            if config.attack_id not in attack_ids:
                continue
            legal_roots += 1
            root_action = tuple(int(index) for index in (row.get("action") or ()))
            if not _selects_attack(root_action, attack_ids, config.attack_id):
                continue
            selected_roots += 1
            continuation = _macro_continuation_rows(
                ordered,
                root_index=root_index,
                root_player=player_index,
            )
            observed_actions = [
                root_action,
                *(
                    tuple(int(index) for index in (successor.get("action") or ()))
                    for successor in continuation
                ),
            ]
            immediate = continuation[0] if continuation else None
            token = row.get("search_begin_input")
            if not isinstance(token, str) or not token:
                continue
            cases.append(
                ReplayMacroCase(
                    episode_id=episode_id,
                    step_index=int(row["step_index"]),
                    player_index=player_index,
                    state_token=token,
                    hidden=_placeholder_hidden(row, own_deck),
                    observed_macro=tuple(observed_actions),
                    root_option_count=int(row.get("select_option_count", 0)),
                    root_action=root_action,
                    discard_option_count=(
                        0
                        if immediate is None
                        else int(immediate.get("select_option_count", 0))
                    ),
                    discard_min_count=(
                        0
                        if immediate is None
                        else int(immediate.get("select_min_count", 0))
                    ),
                    discard_max_count=(
                        0
                        if immediate is None
                        else int(immediate.get("select_max_count", 0))
                    ),
                    discard_action=(
                        ()
                        if immediate is None
                        else tuple(
                            int(index) for index in (immediate.get("action") or ())
                        )
                    ),
                )
            )
    return cases, {
        "exact_deck_sides": sum(len(sides) for sides in exact_sides.values()),
        "dates_with_step_shards": scanned_dates,
        "rocket_legal_roots": legal_roots,
        "rocket_selected_roots": selected_roots,
        "recovered_macro_candidates": len(cases),
    }


def _macro_continuation_rows(
    ordered_rows: Sequence[Mapping[str, Any]],
    *,
    root_index: int,
    root_player: int,
) -> tuple[Mapping[str, Any], ...]:
    """Return same-seat actions before the root macro semantic boundary."""
    continuation: list[Mapping[str, Any]] = []
    for successor in ordered_rows[root_index + 1 :]:
        if int(successor.get("player_index", -1)) != root_player:
            break
        if int(successor.get("select_context", -1)) == int(SelectContext.MAIN):
            break
        continuation.append(successor)
        if len(continuation) >= _MAX_MACRO_STEPS - 1:
            break
    return tuple(continuation)


def _validate_replay_cases(
    backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
    *,
    attack_id: int,
) -> tuple[list[ReplayMacroCase], Mapping[str, Any]]:
    """Keep only replay macros that resolve under the benchmark world."""
    valid: list[ReplayMacroCase] = []
    errors = 0
    trailing_actions = 0
    unresolved = 0
    fact_mismatches = 0
    rejected_examples: list[Mapping[str, Any]] = []
    for case in cases:
        transition = backend.run(
            case.state_token,
            hidden_worlds=[case.hidden],
            macros=[case.observed_macro],
        ).transitions[0]
        reason = ""
        if transition.error != 0:
            if transition.error == NATIVE_MACRO_TRAILING_ACTION_ERROR:
                trailing_actions += 1
                reason = "macro_trailing_action"
            else:
                errors += 1
                reason = f"engine_error_{transition.error}"
        elif not transition.resolved:
            unresolved += 1
            reason = "incomplete_replay_macro"
        elif not target_attack_fact_matches(
            case,
            transition,
            attack_id=attack_id,
        ):
            fact_mismatches += 1
            reason = "effect_fact_mismatch"
        else:
            valid.append(case)
        if reason and len(rejected_examples) < 8:
            rejected_examples.append(
                {
                    "episode_id": case.episode_id,
                    "step_index": case.step_index,
                    "reason": reason,
                }
            )
    return valid, {
        "candidate_count": len(cases),
        "valid_count": len(valid),
        "engine_error_count": errors,
        "trailing_action_error_count": trailing_actions,
        "incomplete_count": unresolved,
        "fact_mismatch_count": fact_mismatches,
        "rejected_examples": rejected_examples,
    }


def _placeholder_hidden(
    row: Mapping[str, Any],
    own_deck: tuple[int, ...],
) -> HiddenInformation:
    seat = int(row["player_index"])
    opponent = 1 - seat
    opponent_deck = tuple(int(card) for card in (row.get("opponent_deck_ids") or ()))
    own_deck_count = int(row[f"player{seat}_deck_count"])
    own_prize_count = len(row.get(f"player{seat}_prize_ids") or ())
    opponent_deck_count = int(row[f"player{opponent}_deck_count"])
    opponent_prize_count = len(row.get(f"player{opponent}_prize_ids") or ())
    opponent_hand_count = int(row[f"player{opponent}_hand_count"])
    godview_hand = tuple(int(card) for card in (row.get("godview_opp_hand_ids") or ()))
    active_ids = tuple(
        int(card) for card in (row.get(f"player{opponent}_active_ids") or ())
    )
    hidden_active = (
        (_PLACEHOLDER_BASIC_POKEMON_ID,)
        if not active_ids or not any(active_ids)
        else ()
    )
    return HiddenInformation.from_sequences(
        your_deck=_fill_cards(own_deck, own_deck_count),
        your_prize=_fill_cards(own_deck, own_prize_count),
        opponent_deck=_fill_cards(opponent_deck, opponent_deck_count),
        opponent_prize=_fill_cards(opponent_deck, opponent_prize_count),
        opponent_hand=_fill_cards(godview_hand or opponent_deck, opponent_hand_count),
        opponent_active=hidden_active,
    )


def _fill_cards(cards: Sequence[int], count: int) -> tuple[int, ...]:
    if count <= 0:
        return ()
    source = tuple(int(card) for card in cards) or (_PLACEHOLDER_CARD_ID,)
    return tuple(source[index % len(source)] for index in range(count))


def _search_api_parity(
    native_backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
    *,
    attack_id: int,
) -> dict[str, Any]:
    return search_api_macro_parity(
        native_backend,
        cases,
        attack_id=attack_id,
    )


def _planning_session_parity(
    macro_backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
    *,
    library_path: Path,
) -> Mapping[str, Any]:
    """Compare v5 continuation chains with exact v2 complete macros."""
    caps = NativePlanningSessionCaps(
        max_engine_steps=256,
        max_forced_steps=32,
        max_observation_bytes=1 << 22,
    )
    contract_fingerprint = hashlib.sha256(
        b"ptcg-rl/native-planning-session-parity/v1\x00"
    ).digest()
    failures: list[Mapping[str, Any]] = []
    exact_logs = 0
    exact_features = 0
    continuation_calls = 0
    failure_count = 0
    native_seconds = 0.0
    end_to_end_started = time.perf_counter()
    with NativePlanningSessionLane(library_path=library_path) as lane:
        for case in cases:
            reference = macro_backend.run(
                case.state_token,
                hidden_worlds=[case.hidden],
                macros=[case.observed_macro],
            ).transitions[0]
            reason = ""
            observed_logs: list[Mapping[str, Any]] = []
            final_observation: Mapping[str, Any] | None = None
            with lane.open_session(
                case.state_token,
                hidden_worlds=[case.hidden],
                candidate_actions=[case.root_action],
                producer_contract_fingerprint=contract_fingerprint,
                root_player=case.player_index,
                max_state_slots=max(64, len(case.observed_macro) + 1),
                caps=caps,
            ) as session:
                payload = session.initial_result.payload
                native_seconds += session.initial_result.timings.native_call_seconds
                final_observation = payload.decode_observation_row(0)
                if final_observation is None:
                    reason = "missing_initial_observation"
                else:
                    observed_logs.extend(
                        cast(Sequence[Mapping[str, Any]], final_observation["logs"])
                    )
                handle = payload.handle_at(0)
                for action in case.observed_macro[1:]:
                    if handle is None:
                        reason = reason or "missing_continuation_handle"
                        break
                    continuation = session.continue_batch(
                        [handle],
                        [action],
                        caps=caps,
                    )
                    continuation_calls += 1
                    native_seconds += continuation.timings.native_call_seconds
                    payload = continuation.payload
                    final_observation = payload.decode_observation_row(0)
                    if final_observation is None:
                        reason = reason or "missing_continuation_observation"
                        break
                    observed_logs.extend(
                        cast(Sequence[Mapping[str, Any]], final_observation["logs"])
                    )
                    handle = payload.handle_at(0)
                if not reason and handle is not None:
                    reason = "unexpected_handle_at_macro_boundary"
            logs_match = tuple(observed_logs) == reference.logs
            exact_logs += int(logs_match)
            features_match = False
            if final_observation is not None:
                session_features = native_transition_feature_vector(
                    before_state=reference.before_state,
                    after_state=cast(Mapping[str, Any], final_observation["current"]),
                    logs=observed_logs,
                )
                reference_features = native_transition_feature_vector(
                    before_state=reference.before_state,
                    after_state=reference.after_state,
                    logs=reference.logs,
                )
                features_match = session_features == reference_features
            exact_features += int(features_match)
            if not reason and not logs_match:
                reason = "root_visible_log_mismatch"
            if not reason and not features_match:
                reason = "consequence_feature_mismatch"
            if reason:
                failure_count += 1
                if len(failures) < 8:
                    failures.append(
                        {
                            "episode_id": case.episode_id,
                            "step_index": case.step_index,
                            "reason": reason,
                        }
                    )
    return {
        "case_count": len(cases),
        "continuation_calls": continuation_calls,
        "exact_log_count": exact_logs,
        "exact_feature_count": exact_features,
        "failure_count": failure_count,
        "failure_examples": failures,
        "native_total_ms": native_seconds * 1000.0,
        "end_to_end_total_ms": (time.perf_counter() - end_to_end_started) * 1000.0,
    }


def _benchmark_observed_macros(
    backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
    config: NativeExactConsequenceBenchmarkConfig,
) -> Mapping[str, Any]:
    reports: dict[str, Any] = {}
    for worlds in config.world_counts:
        _warm_up(backend, cases[0], worlds)
        samples: list[TimingSample] = []
        fact_failures = 0
        for _ in range(config.observed_repetitions):
            for case in cases:
                sample, transitions = _time_batch(
                    backend,
                    case,
                    macros=[case.observed_macro],
                    worlds=worlds,
                    complete_to_boundary=False,
                )
                samples.append(sample)
                fact_failures += sum(
                    not target_attack_fact_matches(
                        case,
                        transition,
                        attack_id=config.attack_id,
                    )
                    for transition in transitions
                )
        reports[str(worlds)] = {
            **_timing_report(samples),
            "fact_failure_count": fact_failures,
            "macro_step_mean": statistics.fmean(
                len(case.observed_macro) for case in cases
            ),
        }
    return reports


def _benchmark_all_root_options(
    backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
    config: NativeExactConsequenceBenchmarkConfig,
) -> Mapping[str, Any]:
    reports: dict[str, Any] = {}
    for worlds in config.world_counts:
        samples: list[TimingSample] = []
        for case in cases:
            macros = [
                ((option_index,),) for option_index in range(case.root_option_count)
            ]
            sample, _ = _time_batch(
                backend,
                case,
                macros=macros,
                worlds=worlds,
                complete_to_boundary=True,
            )
            samples.append(sample)
        reports[str(worlds)] = {
            **_timing_report(samples),
            "root_option_mean": statistics.fmean(
                case.root_option_count for case in cases
            ),
            "root_option_max": max(case.root_option_count for case in cases),
        }
    return reports


def _benchmark_counterfactual_scale(
    backend: NativeMacroBackend,
    cases: Sequence[ReplayMacroCase],
    config: NativeExactConsequenceBenchmarkConfig,
) -> Mapping[str, Any]:
    representative = max(cases, key=lambda case: case.discard_option_count)
    all_subsets = _all_discard_actions(representative)
    reports: dict[str, Any] = {}
    for worlds in config.world_counts:
        world_reports: list[Mapping[str, Any]] = []
        for requested_limit in config.counterfactual_candidate_limits:
            candidate_count = min(requested_limit, len(all_subsets))
            if (
                world_reports
                and candidate_count == world_reports[-1]["candidate_count"]
            ):
                continue
            selected = _evenly_spaced(all_subsets, candidate_count)
            macros = [(representative.root_action, subset) for subset in selected]
            repetitions = max(
                config.min_scale_repetitions,
                min(
                    config.max_scale_repetitions,
                    math.ceil(
                        config.target_transition_samples
                        / max(1, candidate_count * worlds)
                    ),
                ),
            )
            samples: list[TimingSample] = []
            fact_failures = 0
            for _ in range(repetitions):
                sample, transitions = _time_batch(
                    backend,
                    representative,
                    macros=macros,
                    worlds=worlds,
                    complete_to_boundary=True,
                )
                samples.append(sample)
                fact_failures += sum(
                    not target_attack_fact_matches(
                        representative,
                        transition,
                        attack_id=config.attack_id,
                        expected_discard_count=len(macros[index % len(macros)][1]),
                    )
                    for index, transition in enumerate(transitions)
                )
            world_reports.append(
                {
                    "candidate_count": candidate_count,
                    "repetitions": repetitions,
                    "fact_failure_count": fact_failures,
                    **_timing_report(samples),
                }
            )
        reports[str(worlds)] = world_reports
    return {
        "representative_episode_id": representative.episode_id,
        "representative_step_index": representative.step_index,
        "discard_option_count": representative.discard_option_count,
        "exhaustive_candidate_count": len(all_subsets),
        "worlds": reports,
    }


def _time_batch(
    backend: NativeMacroBackend,
    case: ReplayMacroCase,
    *,
    macros: Sequence[Sequence[Sequence[int]]],
    worlds: int,
    complete_to_boundary: bool,
) -> tuple[TimingSample, tuple[NativeProbeTransition, ...]]:
    started = time.perf_counter()
    result = backend.run(
        case.state_token,
        hidden_worlds=[case.hidden] * worlds,
        macros=macros,
        complete_to_boundary=complete_to_boundary,
    )
    for transition in result.transitions:
        if transition.error == 0 and transition.after_state:
            native_transition_feature_vector(
                before_state=transition.before_state,
                after_state=transition.after_state,
                logs=transition.logs,
            )
    end_to_end_ms = (time.perf_counter() - started) * 1000.0
    return (
        TimingSample(
            native_ms=result.native_call_seconds * 1000.0,
            end_to_end_ms=end_to_end_ms,
            payload_bytes=result.payload_bytes,
            transitions=len(result.transitions),
            resolved=sum(transition.resolved for transition in result.transitions),
            errors=sum(transition.error != 0 for transition in result.transitions),
            completion_steps=sum(
                transition.forced_steps for transition in result.transitions
            ),
        ),
        result.transitions,
    )


def _warm_up(
    backend: NativeMacroBackend,
    case: ReplayMacroCase,
    worlds: int,
) -> None:
    for _ in range(8):
        backend.run(
            case.state_token,
            hidden_worlds=[case.hidden] * worlds,
            macros=[case.observed_macro],
        )


def _timing_report(samples: Sequence[TimingSample]) -> Mapping[str, Any]:
    native = [sample.native_ms for sample in samples]
    total = [sample.end_to_end_ms for sample in samples]
    transitions = sum(sample.transitions for sample in samples)
    return {
        "calls": len(samples),
        "transitions": transitions,
        "native_mean_ms": statistics.fmean(native),
        "native_p50_ms": _quantile(native, 0.50),
        "native_p95_ms": _quantile(native, 0.95),
        "end_to_end_mean_ms": statistics.fmean(total),
        "end_to_end_p50_ms": _quantile(total, 0.50),
        "end_to_end_p95_ms": _quantile(total, 0.95),
        "native_us_per_transition": 1000.0 * sum(native) / transitions,
        "end_to_end_us_per_transition": 1000.0 * sum(total) / transitions,
        "mean_payload_bytes": statistics.fmean(
            sample.payload_bytes for sample in samples
        ),
        "resolved_rate": sum(sample.resolved for sample in samples) / transitions,
        "error_rate": sum(sample.errors for sample in samples) / transitions,
        "mean_completion_steps": sum(sample.completion_steps for sample in samples)
        / transitions,
    }


def _all_discard_actions(case: ReplayMacroCase) -> list[tuple[int, ...]]:
    if case.discard_option_count <= 0:
        raise RuntimeError("representative attack has no discard continuation")
    actions: list[tuple[int, ...]] = []
    for count in range(case.discard_min_count, case.discard_max_count + 1):
        actions.extend(itertools.combinations(range(case.discard_option_count), count))
    return actions


def _selects_attack(
    action: Sequence[int],
    attack_ids: Sequence[Any],
    attack_id: int,
) -> bool:
    return (
        len(action) == 1
        and 0 <= action[0] < len(attack_ids)
        and attack_ids[action[0]] == attack_id
    )


def _evenly_spaced(values: Sequence[_ValueT], limit: int) -> list[_ValueT]:
    if len(values) <= limit:
        return list(values)
    if limit == 1:
        return [values[len(values) // 2]]
    return [
        values[round(index * (len(values) - 1) / (limit - 1))] for index in range(limit)
    ]


def _quantile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def _read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(deck_records.read_deck(deck_records.repo_path(path)))
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 cards: {path}")
    return deck


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/native_exact_consequence_benchmark",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw_config = OmegaConf.to_container(hydra_config, resolve=True)
    config = NativeExactConsequenceBenchmarkConfig.model_validate(raw_config)
    print(json.dumps(run_benchmark(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
