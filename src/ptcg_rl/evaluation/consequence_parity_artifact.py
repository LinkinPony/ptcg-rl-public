"""Privacy-safe rows, identities, and atomic metadata for parity audits."""

from __future__ import annotations

import glob
import hashlib
import json
import os
import struct
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ptcg_rl.engine.native_consequence import NativeConsequenceBatchResult
from ptcg_rl.engine.native_consequence_payload import (
    NativeConsequenceEndpoint,
    NativeConsequenceMetadataColumn,
)
from ptcg_rl.engine.session import HiddenInformation
from ptcg_rl.evaluation.consequence_audit_corpus import ConsequenceAuditCase
from ptcg_rl.evaluation.consequence_audit_sampling import COVERAGE_LABELS
from ptcg_rl.evaluation.consequence_parity_compare import (
    NativeRepeatComparison,
    TransitionParityComparison,
    candidate_fingerprint,
)
from ptcg_rl.evaluation.consequence_parity_config import (
    DecisionTransitionParityConfig,
)

_SELECTED_INPUT_DOMAIN = b"ptcg-rl/decision-transition-audit/selected-input/v1\x00"


def audit_row(
    case: ConsequenceAuditCase,
    *,
    world_index: int,
    candidate_index: int,
    action: Sequence[int],
    legal_action_count: int,
    support_exhaustive: bool,
    comparison: TransitionParityComparison,
    native_repeat: NativeRepeatComparison,
    native_result: NativeConsequenceBatchResult,
) -> dict[str, Any]:
    """Build one schema-complete row without raw engine-private material."""
    root_flags = _root_flags(case.labels)
    return {
        "case_id": case.case_id,
        "world_index": world_index,
        "candidate_index": candidate_index,
        "candidate_action_length": len(action),
        "candidate_fingerprint": candidate_fingerprint(action),
        "legal_action_count": legal_action_count,
        "support_exhaustive": support_exhaustive,
        **root_flags,
        "direct": root_flags["root_direct"],
        "subset": root_flags["root_subset"],
        "ordered": root_flags["root_ordered"],
        "multi_prompt": root_flags["root_multi_prompt"] or comparison.multi_prompt,
        "manual_coin": root_flags["root_manual_coin"] or comparison.manual_coin,
        "handoff": root_flags["root_handoff"] or comparison.handoff,
        "status": (
            comparison.status
            if native_repeat.match
            else "native_batch_isolated_mismatch"
        ),
        "native_error": comparison.native_error,
        "reference_error": comparison.reference_error,
        "native_endpoint": comparison.native_endpoint,
        "reference_endpoint": comparison.reference_endpoint,
        "native_transition_steps": comparison.native_transition_steps,
        "reference_transition_steps": comparison.reference_transition_steps,
        "native_forced_steps": comparison.native_forced_steps,
        "reference_forced_steps": comparison.reference_forced_steps,
        "native_leaf_player": comparison.native_leaf_player,
        "reference_leaf_player": comparison.reference_leaf_player,
        "endpoint_match": comparison.endpoint_match,
        "transition_steps_match": comparison.transition_steps_match,
        "leaf_player_match": comparison.leaf_player_match,
        "leaf_actor_state_match": comparison.leaf_actor_state_match,
        "leaf_actor_log_match": comparison.leaf_actor_log_match,
        "log_match": comparison.log_match,
        "state_match": comparison.state_match,
        "effect_match": comparison.effect_match,
        "parity_match": comparison.parity_match,
        "native_parity_failure": (
            comparison.native_parity_failure or not native_repeat.match
        ),
        "native_isolated_match": native_repeat.match,
        "native_isolated_metadata_match": native_repeat.metadata_match,
        "native_isolated_root_observation_match": (
            native_repeat.root_observation_match
        ),
        "native_isolated_leaf_observation_match": (
            native_repeat.leaf_observation_match
        ),
        "effect_max_abs_diff": comparison.effect_max_abs_diff,
        "native_state_digest": comparison.native_state_digest,
        "reference_state_digest": comparison.reference_state_digest,
        "native_log_digest": comparison.native_log_digest,
        "reference_log_digest": comparison.reference_log_digest,
        "native_rng_unsupported": comparison.native_rng_unsupported,
        "reference_prize_defect_exposed": (comparison.reference_prize_defect_exposed),
        "reference_prize_defect_relevant": (comparison.reference_prize_defect_relevant),
        "damage_effect": comparison.damage_effect,
        "healing_effect": comparison.healing_effect,
        "prize_effect": comparison.prize_effect,
        "status_effect": comparison.status_effect,
        "random_effect": comparison.random_effect,
        **_timing_fields(native_result),
    }


def reference_exception_row(
    case: ConsequenceAuditCase,
    *,
    world_index: int,
    candidate_index: int,
    action: Sequence[int],
    legal_action_count: int,
    support_exhaustive: bool,
    native_result: NativeConsequenceBatchResult,
    row_index: int,
) -> dict[str, Any]:
    """Preserve a failed reference cell without exception/private payloads."""
    metadata = native_result.metadata[row_index]
    root_flags = _root_flags(case.labels)
    endpoint = int(metadata[int(NativeConsequenceMetadataColumn.ENDPOINT)])
    return {
        "case_id": case.case_id,
        "world_index": world_index,
        "candidate_index": candidate_index,
        "candidate_action_length": len(action),
        "candidate_fingerprint": candidate_fingerprint(action),
        "legal_action_count": legal_action_count,
        "support_exhaustive": support_exhaustive,
        **root_flags,
        "direct": root_flags["root_direct"],
        "subset": root_flags["root_subset"],
        "ordered": root_flags["root_ordered"],
        "multi_prompt": root_flags["root_multi_prompt"],
        "manual_coin": root_flags["root_manual_coin"]
        or endpoint == int(NativeConsequenceEndpoint.CHANCE_PROMPT),
        "handoff": root_flags["root_handoff"]
        or endpoint == int(NativeConsequenceEndpoint.TURN_HANDOFF),
        "status": "reference_exception",
        "native_error": int(metadata[int(NativeConsequenceMetadataColumn.ERROR)]),
        "reference_error": -1,
        "native_endpoint": endpoint,
        "reference_endpoint": int(NativeConsequenceEndpoint.INVALID),
        "native_transition_steps": int(
            metadata[int(NativeConsequenceMetadataColumn.TRANSITION_STEPS)]
        ),
        "reference_transition_steps": 0,
        "native_forced_steps": int(
            metadata[int(NativeConsequenceMetadataColumn.FORCED_STEPS)]
        ),
        "reference_forced_steps": 0,
        "native_leaf_player": int(
            metadata[int(NativeConsequenceMetadataColumn.LEAF_PLAYER)]
        ),
        "reference_leaf_player": -1,
        "endpoint_match": None,
        "transition_steps_match": None,
        "leaf_player_match": None,
        "leaf_actor_state_match": None,
        "leaf_actor_log_match": None,
        "log_match": None,
        "state_match": None,
        "effect_match": None,
        "parity_match": False,
        "native_parity_failure": False,
        "native_isolated_match": False,
        "native_isolated_metadata_match": False,
        "native_isolated_root_observation_match": False,
        "native_isolated_leaf_observation_match": False,
        "effect_max_abs_diff": None,
        "native_state_digest": None,
        "reference_state_digest": None,
        "native_log_digest": None,
        "reference_log_digest": None,
        "native_rng_unsupported": False,
        "reference_prize_defect_exposed": False,
        "reference_prize_defect_relevant": False,
        "damage_effect": False,
        "healing_effect": False,
        "prize_effect": False,
        "status_effect": False,
        "random_effect": False,
        **_timing_fields(native_result),
    }


def producer_contract_fingerprint(
    case: ConsequenceAuditCase,
    actions: Sequence[Sequence[int]],
    *,
    config: DecisionTransitionParityConfig,
    hidden_worlds: Sequence[HiddenInformation] | None = None,
) -> bytes:
    """Bind a native request to the audit's external evidence semantics."""
    digest = hashlib.sha256(
        b"ptcg-rl/decision-transition-audit/producer-contract/v1\x00"
    )
    _update_framed(digest, case.case_id.encode("ascii"))
    digest.update(struct.pack(">i", case.player_index))
    _update_framed(
        digest,
        b"single-count-correct-belief-world/manual-coin/root-visible-projection-v1",
    )
    for action in actions:
        _update_framed(digest, bytes.fromhex(candidate_fingerprint(action)))
    worlds = tuple(hidden_worlds) if hidden_worlds is not None else (case.hidden,)
    digest.update(struct.pack(">I", len(worlds)))
    for world in worlds:
        for zone in hidden_zones(world):
            digest.update(struct.pack(">I", len(zone)))
            for card_id in zone:
                digest.update(struct.pack(">i", int(card_id)))
    for bound in (
        config.max_cells,
        config.max_engine_steps,
        config.max_forced_steps,
        config.max_observation_bytes,
    ):
        digest.update(struct.pack(">I", bound))
    return digest.digest()


def selected_input_fingerprint(cases: Sequence[ConsequenceAuditCase]) -> str:
    """Return one aggregate identity; never persist component private values."""
    digest = hashlib.sha256(_SELECTED_INPUT_DOMAIN)
    for case in cases:
        _update_framed(digest, case.case_id.encode("ascii"))
        _update_framed(digest, case.state_token.encode("ascii"))
        for zone in hidden_zones(case.hidden):
            digest.update(struct.pack(">I", len(zone)))
            for card_id in zone:
                digest.update(struct.pack(">i", int(card_id)))
        for option_index in case.observed_action:
            digest.update(struct.pack(">i", option_index))
    return digest.hexdigest()


def resolve_parquet_paths(patterns: Sequence[str]) -> tuple[Path, ...]:
    """Resolve and validate input globs deterministically."""
    paths = {
        Path(match)
        for pattern in patterns
        for match in glob.glob(pattern, recursive=True)
        if Path(match).is_file()
    }
    if not paths:
        raise FileNotFoundError("no replay Parquet files matched steps_globs")
    non_parquet = sorted(path for path in paths if path.suffix != ".parquet")
    if non_parquet:
        raise ValueError(f"replay inputs must be Parquet files: {non_parquet[:4]}")
    return tuple(sorted(paths))


def file_sha256(path: Path) -> str:
    """Stream an immutable library identity."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Publish a small human-readable summary atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_interpretable_artifact(*, roots: int, cells: int) -> None:
    """Reject evidence with no interpretable engine comparison."""
    if roots <= 0 or cells <= 0:
        raise RuntimeError(
            "audit produced zero interpretable roots/cells; no artifact "
            "can be published"
        )


def hidden_zones(hidden: HiddenInformation) -> tuple[tuple[int, ...], ...]:
    """Return transient hidden lists in native ABI order."""
    return (
        hidden.your_deck,
        hidden.your_prize,
        hidden.opponent_deck,
        hidden.opponent_prize,
        hidden.opponent_hand,
        hidden.opponent_active,
    )


def _root_flags(labels: frozenset[str]) -> dict[str, bool]:
    return {f"root_{label}": label in labels for label in COVERAGE_LABELS}


def _timing_fields(result: NativeConsequenceBatchResult) -> dict[str, Any]:
    return {
        "native_pack_ms": result.pack_seconds * 1000.0,
        "native_call_ms": result.native_call_seconds * 1000.0,
        "native_parse_ms": result.parse_seconds * 1000.0,
        "native_payload_bytes": result.payload_bytes,
    }


def _update_framed(digest: Any, value: bytes) -> None:
    digest.update(struct.pack(">Q", len(value)))
    digest.update(value)


__all__ = [
    "audit_row",
    "file_sha256",
    "producer_contract_fingerprint",
    "reference_exception_row",
    "resolve_parquet_paths",
    "selected_input_fingerprint",
    "validate_interpretable_artifact",
    "write_json_atomic",
]
