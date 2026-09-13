"""Bounded orchestration for generic native consequence parity evidence."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ptcg_rl.agent.search.prompt_actions import build_prompt_action_candidates
from ptcg_rl.engine.native_consequence import NativeConsequenceLane
from ptcg_rl.engine.native_consequence_payload import (
    NATIVE_CONSEQUENCE_PAYLOAD_VERSION,
    NativeConsequenceEndpoint,
)
from ptcg_rl.engine.session import HiddenInformation
from ptcg_rl.evaluation.consequence_audit_corpus import load_audit_cases
from ptcg_rl.evaluation.consequence_audit_sampling import (
    COVERAGE_LABELS,
    SamplingReport,
    sample_replay_roots,
)
from ptcg_rl.evaluation.consequence_parity import (
    PublicSearchDecisionBackend,
    execute_public_decision,
    read_public_search_root,
)
from ptcg_rl.evaluation.consequence_parity_artifact import (
    audit_row,
    file_sha256,
    producer_contract_fingerprint,
    resolve_parquet_paths,
    selected_input_fingerprint,
    validate_interpretable_artifact,
    write_json_atomic,
)
from ptcg_rl.evaluation.consequence_parity_compare import (
    compare_native_reference,
    compare_native_repeat,
)
from ptcg_rl.evaluation.consequence_parity_config import (
    DecisionTransitionParityConfig,
)
from ptcg_rl.evaluation.consequence_parity_io import AtomicAuditShardWriter


@dataclass
class _AuditStats:
    roots_with_native_batch: int = 0
    interpretable_roots: int = 0
    interpretable_cells: int = 0
    candidate_attempts: int = 0
    exhaustive_roots: int = 0
    native_pack_seconds: float = 0.0
    native_call_seconds: float = 0.0
    native_parse_seconds: float = 0.0
    native_isolated_calls: int = 0
    native_isolated_match_cells: int = 0
    prize_defect_exposed_cells: int = 0
    prize_defect_relevant_cells: int = 0
    native_parity_failure_cells: int = 0
    status_counts: Counter[str] = field(default_factory=Counter)
    endpoint_counts: Counter[str] = field(default_factory=Counter)
    effect_counts: Counter[str] = field(default_factory=Counter)
    cell_label_counts: Counter[str] = field(default_factory=Counter)
    root_failure_counts: Counter[str] = field(default_factory=Counter)
    reference_failure_counts: Counter[str] = field(default_factory=Counter)
    mismatch_examples: list[Mapping[str, Any]] = field(default_factory=list)
    root_failure_examples: list[Mapping[str, str]] = field(default_factory=list)


def run_decision_transition_parity_audit(
    config: DecisionTransitionParityConfig,
) -> dict[str, Any]:
    """Run one bounded multi-label native/public-Search parity audit."""
    paths = resolve_parquet_paths(config.steps_globs)
    if not config.library_path.is_file():
        raise FileNotFoundError(
            f"native consequence library does not exist: {config.library_path}"
        )
    summary_path = config.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(
            "audit summary already exists; use a fresh output directory"
        )
    locators, sampling = sample_replay_roots(
        paths,
        capacity_per_label=config.reservoir_capacity_per_label,
        seed=config.reservoir_seed,
        batch_size=config.scan_batch_size,
    )
    if not locators:
        raise RuntimeError("replay corpus yielded no auditable roots")
    cases = load_audit_cases(
        locators,
        batch_size=config.scan_batch_size,
        fallback_card_id=config.fallback_card_id,
        fallback_basic_pokemon_id=config.fallback_basic_pokemon_id,
    )
    selected_fingerprint = selected_input_fingerprint(cases)
    library_fingerprint = file_sha256(config.library_path)
    stats = _AuditStats()
    reference_backend = PublicSearchDecisionBackend(manual_coin=config.manual_coin)
    started = time.perf_counter()

    with (
        NativeConsequenceLane(library_path=config.library_path) as lane,
        AtomicAuditShardWriter(
            config.output_dir,
            output_format=config.output_format,
            shard_rows=config.output_shard_rows,
            max_rows=config.max_output_rows,
            compression=config.compression,
        ) as writer,
    ):
        loaded_library_fingerprint = lane.engine_library_fingerprint
        native_abi_fingerprint = lane.native_abi_fingerprint
        if loaded_library_fingerprint != library_fingerprint:
            raise RuntimeError("native library changed while the audit was starting")
        for case in cases:
            hidden_worlds = tuple(
                _rotated_hidden(case.hidden, offset=index)
                for index in range(config.worlds_per_root)
            )
            try:
                stochastic_seed = _audit_stochastic_seed(
                    config.reservoir_seed,
                    case.case_id,
                )
                root = read_public_search_root(
                    reference_backend,
                    state_token=case.state_token,
                    hidden=case.hidden,
                    root_player=case.player_index,
                )
                candidates = build_prompt_action_candidates(
                    root.get("select"),
                    greedy_action=case.observed_action,
                    exhaustive_action_cap=config.exhaustive_action_cap,
                    beam_width=config.max_candidates_per_root,
                )
                contract_fingerprint = producer_contract_fingerprint(
                    case,
                    candidates.actions,
                    config=config,
                    hidden_worlds=hidden_worlds,
                )
                native = lane.run(
                    case.state_token,
                    hidden_worlds=hidden_worlds,
                    candidate_actions=candidates.actions,
                    producer_contract_fingerprint=contract_fingerprint,
                    root_player=case.player_index,
                    manual_coin=config.manual_coin,
                    stochastic_seed=stochastic_seed,
                    max_cells=config.max_cells,
                    max_engine_steps=config.max_engine_steps,
                    max_forced_steps=config.max_forced_steps,
                    max_observation_bytes=config.max_observation_bytes,
                )
                isolated = tuple(
                    lane.run(
                        case.state_token,
                        hidden_worlds=hidden_worlds,
                        candidate_actions=(action,),
                        producer_contract_fingerprint=contract_fingerprint,
                        root_player=case.player_index,
                        manual_coin=config.manual_coin,
                        stochastic_seed=stochastic_seed,
                        max_cells=config.max_cells,
                        max_engine_steps=config.max_engine_steps,
                        max_forced_steps=config.max_forced_steps,
                        max_observation_bytes=config.max_observation_bytes,
                    )
                    for action in candidates.actions
                )
            except Exception as error:
                _record_root_failure(stats, case.case_id, error)
                continue
            stats.roots_with_native_batch += 1
            stats.exhaustive_roots += int(candidates.exhaustive)
            stats.native_pack_seconds += native.pack_seconds
            stats.native_call_seconds += native.native_call_seconds
            stats.native_parse_seconds += native.parse_seconds
            stats.native_isolated_calls += len(isolated)
            stats.native_pack_seconds += sum(item.pack_seconds for item in isolated)
            stats.native_call_seconds += sum(
                item.native_call_seconds for item in isolated
            )
            stats.native_parse_seconds += sum(
                item.parse_seconds for item in isolated
            )
            root_interpretable = False
            for candidate_index, action in enumerate(candidates.actions):
                for world_index, hidden in enumerate(hidden_worlds):
                    stats.candidate_attempts += 1
                    row_index = native.payload.row_index(
                        world_index,
                        candidate_index,
                    )
                    try:
                        native_repeat = compare_native_repeat(
                            native,
                            batch_row_index=row_index,
                            isolated=isolated[candidate_index],
                            isolated_row_index=isolated[
                                candidate_index
                            ].payload.row_index(world_index, 0),
                        )
                        reference = execute_public_decision(
                            reference_backend,
                            state_token=case.state_token,
                            hidden=hidden,
                            candidate_action=action,
                            root_player=case.player_index,
                            max_forced_steps=config.max_forced_steps,
                        )
                        comparison = compare_native_reference(
                            native,
                            row_index=row_index,
                            reference=reference,
                            candidate_action=action,
                            root_player=case.player_index,
                            effect_atol=config.effect_atol,
                        )
                    except Exception as error:
                        stats.reference_failure_counts[type(error).__name__] += 1
                        continue
                    row = audit_row(
                        case,
                        world_index=world_index,
                        candidate_index=candidate_index,
                        action=action,
                        legal_action_count=candidates.legal_action_count,
                        support_exhaustive=candidates.exhaustive,
                        comparison=comparison,
                        native_repeat=native_repeat,
                        native_result=native,
                    )
                    writer.append(row)
                    root_interpretable = True
                    stats.interpretable_cells += 1
                    stats.status_counts[str(row["status"])] += 1
                    stats.native_isolated_match_cells += int(native_repeat.match)
                    stats.endpoint_counts[
                        NativeConsequenceEndpoint(
                            comparison.native_endpoint
                        ).name.lower()
                    ] += 1
                    for effect_name in (
                        "damage_effect",
                        "healing_effect",
                        "prize_effect",
                        "status_effect",
                        "random_effect",
                    ):
                        stats.effect_counts[effect_name] += int(
                            bool(getattr(comparison, effect_name))
                        )
                    stats.prize_defect_exposed_cells += int(
                        comparison.reference_prize_defect_exposed
                    )
                    stats.prize_defect_relevant_cells += int(
                        comparison.reference_prize_defect_relevant
                    )
                    stats.native_parity_failure_cells += int(
                        bool(row["native_parity_failure"])
                    )
                    for label in COVERAGE_LABELS:
                        stats.cell_label_counts[label] += int(bool(row[label]))
                    if (
                        bool(row["native_parity_failure"])
                        and len(stats.mismatch_examples) < 32
                    ):
                        stats.mismatch_examples.append(
                            {
                                "case_id": case.case_id,
                                "world_index": world_index,
                                "candidate_index": candidate_index,
                                "status": str(row["status"]),
                            }
                        )
            stats.interpretable_roots += int(root_interpretable)
        validate_interpretable_artifact(
            roots=stats.interpretable_roots,
            cells=stats.interpretable_cells,
        )
    elapsed = time.perf_counter() - started
    summary = _summary(
        config,
        sampling=sampling,
        stats=stats,
        elapsed_seconds=elapsed,
        part_count=writer.part_count,
        library_fingerprint=library_fingerprint,
        loaded_library_fingerprint=loaded_library_fingerprint,
        native_abi_fingerprint=native_abi_fingerprint,
        selected_fingerprint=selected_fingerprint,
    )
    write_json_atomic(summary_path, summary)
    return summary


def _record_root_failure(stats: _AuditStats, case_id: str, error: Exception) -> None:
    name = type(error).__name__
    stats.root_failure_counts[name] += 1
    if len(stats.root_failure_examples) < 16:
        # Exception messages may contain engine-private material.  Persist only
        # the exception class and privacy-safe case locator identity.
        stats.root_failure_examples.append({"case_id": case_id, "error_type": name})


def _summary(
    config: DecisionTransitionParityConfig,
    *,
    sampling: SamplingReport,
    stats: _AuditStats,
    elapsed_seconds: float,
    part_count: int,
    library_fingerprint: str,
    loaded_library_fingerprint: str,
    native_abi_fingerprint: str,
    selected_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema": {
            "name": "decision_transition_parity",
            "version": 1,
            "native_payload_version": NATIVE_CONSEQUENCE_PAYLOAD_VERSION,
        },
        "execution": {
            "device": "cpu",
            "cpu_reason": ("the bundled simulator is engine-bound and has no GPU path"),
            "manual_coin": config.manual_coin,
            "primitive": "complete root selection plus only forced prompts",
            "worlds_per_root": config.worlds_per_root,
            "elapsed_seconds": elapsed_seconds,
        },
        "identity": {
            "library_path": str(config.library_path),
            "library_sha256": library_fingerprint,
            "loaded_library_sha256": loaded_library_fingerprint,
            "native_abi_sha256": native_abi_fingerprint,
            "selected_request_set_sha256": selected_fingerprint,
            "corpus_structural_sha256": sampling.corpus_fingerprint,
        },
        "sampling": {
            "method": "per-label deterministic lowest-SHA256 reservoir",
            "labels": list(COVERAGE_LABELS),
            "capacity_per_label": config.reservoir_capacity_per_label,
            "scanned_files": sampling.scanned_files,
            "scanned_rows": sampling.scanned_rows,
            "corpus_label_counts": dict(sampling.corpus_label_counts),
            "retained_label_counts": dict(sampling.retained_label_counts),
            "retained_roots": sampling.retained_roots,
        },
        "coverage": {
            "native_batch_roots": stats.roots_with_native_batch,
            "interpretable_roots": stats.interpretable_roots,
            "candidate_attempts": stats.candidate_attempts,
            "interpretable_cells": stats.interpretable_cells,
            "exhaustive_roots": stats.exhaustive_roots,
            "cell_label_counts": {
                label: int(stats.cell_label_counts[label]) for label in COVERAGE_LABELS
            },
            "native_rng_unsupported_cells": stats.status_counts[
                "native_rng_unsupported"
            ],
            "public_reference_prize_bug_surface_cells": stats.status_counts[
                "public_reference_prize_bug_surface"
            ],
            "public_search_prize_defect_exposed_cells": (
                stats.prize_defect_exposed_cells
            ),
            "public_search_prize_defect_relevant_cells": (
                stats.prize_defect_relevant_cells
            ),
            "native_parity_failure_cells": stats.native_parity_failure_cells,
            "native_isolated_calls": stats.native_isolated_calls,
            "native_isolated_match_cells": stats.native_isolated_match_cells,
            "endpoint_counts": dict(sorted(stats.endpoint_counts.items())),
            "effect_counts": dict(sorted(stats.effect_counts.items())),
        },
        "parity": {
            "status_counts": dict(sorted(stats.status_counts.items())),
            "mismatch_examples": stats.mismatch_examples,
            "effect_atol": config.effect_atol,
        },
        "failures": {
            "root_failure_counts": dict(sorted(stats.root_failure_counts.items())),
            "root_failure_examples": stats.root_failure_examples,
            "reference_failure_counts": dict(
                sorted(stats.reference_failure_counts.items())
            ),
        },
        "timing": {
            "native_pack_seconds": stats.native_pack_seconds,
            "native_call_seconds": stats.native_call_seconds,
            "native_parse_seconds": stats.native_parse_seconds,
            "cells_per_wall_second": stats.interpretable_cells
            / max(elapsed_seconds, 1.0e-12),
        },
        "artifact": {
            "format": config.output_format,
            "part_count": part_count,
            "row_count": stats.interpretable_cells,
            "contains_raw_state_tokens": False,
            "contains_raw_hidden_zone_identities": False,
        },
        "reference_limitations": [
            (
                "a mismatch is classified public_reference_prize_bug_surface "
                "only when the face-up-prize Search defect is exposed and the "
                "transition either has direct prize move/count evidence or "
                "differs only on state surfaces while endpoints, steps, logs, "
                "and independently derived effects agree"
            ),
            (
                "handoff state parity masks private hand/looking identities on "
                "both sides because public Search serializes from the next "
                "actor; public counts, exact effects, logs, and endpoints remain"
                " directly compared"
            ),
            (
                "opaque random outcomes use independently sampled public Search "
                "evidence; exact same-seed equality is enforced against an "
                "isolated native request"
            ),
            "native error 92 is unsupported RNG consumption, not a parity mismatch",
            "count-correct worlds test rule parity, not belief quality",
        ],
    }


def _rotated_hidden(
    hidden: HiddenInformation,
    *,
    offset: int,
) -> HiddenInformation:
    """Create deterministic count-preserving worlds for grid-index parity."""
    your_deck, your_prize = _rotate_partition(
        (hidden.your_deck, hidden.your_prize),
        offset,
    )
    opponent_deck, opponent_prize, opponent_hand = _rotate_partition(
        (hidden.opponent_deck, hidden.opponent_prize, hidden.opponent_hand),
        offset * 3 + 1,
    )
    return HiddenInformation.from_sequences(
        your_deck=your_deck,
        your_prize=your_prize,
        opponent_deck=opponent_deck,
        opponent_prize=opponent_prize,
        opponent_hand=opponent_hand,
        opponent_active=hidden.opponent_active,
    )


def _audit_stochastic_seed(reservoir_seed: str, case_id: str) -> int:
    """Derive a stable root RNG seed without persisting protected state."""
    digest = hashlib.sha256()
    digest.update(b"ptcg-rl/information-set-api-parity/rng/v1\x00")
    digest.update(reservoir_seed.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(case_id.encode("ascii"))
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def _rotate_partition(
    zones: Sequence[Sequence[int]],
    offset: int,
) -> tuple[tuple[int, ...], ...]:
    """Rotate a combined hidden pool without changing any zone count."""
    lengths = tuple(len(zone) for zone in zones)
    pool = tuple(card for zone in zones for card in zone)
    if pool:
        shift = offset % len(pool)
        pool = pool[shift:] + pool[:shift]
    result: list[tuple[int, ...]] = []
    cursor = 0
    for length in lengths:
        result.append(pool[cursor : cursor + length])
        cursor += length
    return tuple(result)


__all__ = ["run_decision_transition_parity_audit"]
