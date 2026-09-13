"""Exact, privacy-projected native/public-Search consequence comparison."""

from __future__ import annotations

import hashlib
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import orjson

from ptcg_rl.engine.constants import AreaType, LogType
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_NAMES
from ptcg_rl.engine.forward_model import (
    dynamic_effect_feature_from_dict_resolution,
)
from ptcg_rl.engine.native_consequence import NativeConsequenceBatchResult
from ptcg_rl.engine.native_consequence_payload import (
    NativeConsequenceEndpoint,
    NativeConsequenceMetadataColumn,
)
from ptcg_rl.engine.native_macro_validation import effect_log_signature
from ptcg_rl.evaluation.consequence_parity import (
    ReferenceDecisionTransition,
    mapping_logs,
)

NATIVE_UNSUPPORTED_RNG_ERROR = 92

_DIGEST_DOMAIN = b"ptcg-rl/decision-transition-audit/public-digest/v1\x00"
_CANDIDATE_DOMAIN = b"ptcg-rl/decision-transition-audit/candidate/v1\x00"


@dataclass(frozen=True, slots=True)
class TransitionParityComparison:
    """Privacy-safe exact comparison facts for one native/reference cell."""

    status: str
    native_error: int
    reference_error: int
    native_endpoint: int
    reference_endpoint: int
    native_transition_steps: int
    reference_transition_steps: int
    native_forced_steps: int
    reference_forced_steps: int
    native_leaf_player: int
    reference_leaf_player: int
    endpoint_match: bool | None
    transition_steps_match: bool | None
    leaf_player_match: bool | None
    leaf_actor_state_match: bool | None
    leaf_actor_log_match: bool | None
    log_match: bool | None
    state_match: bool | None
    effect_match: bool | None
    parity_match: bool
    native_parity_failure: bool
    effect_max_abs_diff: float | None
    native_state_digest: str | None
    reference_state_digest: str | None
    native_log_digest: str | None
    reference_log_digest: str | None
    manual_coin: bool
    multi_prompt: bool
    handoff: bool
    native_rng_unsupported: bool
    reference_prize_defect_exposed: bool
    reference_prize_defect_relevant: bool
    damage_effect: bool
    healing_effect: bool
    prize_effect: bool
    status_effect: bool
    random_effect: bool


@dataclass(frozen=True, slots=True)
class NativeRepeatComparison:
    """Exact agreement between a grid row and isolated-candidate execution."""

    match: bool
    metadata_match: bool
    root_observation_match: bool
    leaf_observation_match: bool


def compare_native_repeat(
    batch: NativeConsequenceBatchResult,
    *,
    batch_row_index: int,
    isolated: NativeConsequenceBatchResult,
    isolated_row_index: int,
) -> NativeRepeatComparison:
    """Compare one batch cell with the same seeded isolated engine request."""
    semantic_columns = tuple(
        column
        for column in NativeConsequenceMetadataColumn
        if column
        not in {
            NativeConsequenceMetadataColumn.OBSERVATION_OFFSET,
            NativeConsequenceMetadataColumn.LEAF_OBSERVATION_OFFSET,
        }
    )
    batch_metadata = batch.metadata[batch_row_index]
    isolated_metadata = isolated.metadata[isolated_row_index]
    metadata_match = all(
        int(batch_metadata[int(column)]) == int(isolated_metadata[int(column)])
        for column in semantic_columns
    )
    root_observation_match = _optional_bytes_equal(
        batch.payload.observation_bytes_for_row(batch_row_index),
        isolated.payload.observation_bytes_for_row(isolated_row_index),
    )
    leaf_observation_match = _optional_bytes_equal(
        batch.payload.leaf_observation_bytes_for_row(batch_row_index),
        isolated.payload.leaf_observation_bytes_for_row(isolated_row_index),
    )
    return NativeRepeatComparison(
        match=(
            metadata_match
            and root_observation_match
            and leaf_observation_match
        ),
        metadata_match=metadata_match,
        root_observation_match=root_observation_match,
        leaf_observation_match=leaf_observation_match,
    )


def compare_native_reference(
    native_result: NativeConsequenceBatchResult,
    *,
    row_index: int,
    reference: ReferenceDecisionTransition,
    candidate_action: Sequence[int],
    root_player: int,
    effect_atol: float,
) -> TransitionParityComparison:
    """Compare exact endpoints, public logs, state, and effect features."""
    if effect_atol < 0.0 or not math.isfinite(effect_atol):
        raise ValueError("effect_atol must be finite and non-negative")
    metadata = native_result.metadata[row_index]
    native_error = int(metadata[int(NativeConsequenceMetadataColumn.ERROR)])
    native_endpoint = int(metadata[int(NativeConsequenceMetadataColumn.ENDPOINT)])
    native_steps = int(metadata[int(NativeConsequenceMetadataColumn.TRANSITION_STEPS)])
    native_forced = int(metadata[int(NativeConsequenceMetadataColumn.FORCED_STEPS)])
    native_leaf_player = int(
        metadata[int(NativeConsequenceMetadataColumn.LEAF_PLAYER)]
    )
    reference_leaf_player = (
        -1 if reference.leaf_player is None else reference.leaf_player
    )
    reference_endpoint = int(reference.endpoint)
    chance_unsupported = native_error == NATIVE_UNSUPPORTED_RNG_ERROR
    manual_coin, multi_prompt, handoff = _coverage_facts(
        native_endpoint,
        reference_endpoint,
        native_forced,
        reference.forced_steps,
    )
    if native_error != 0 or reference.error_code != 0:
        error_match = native_error == reference.error_code
        status = (
            "native_rng_unsupported"
            if chance_unsupported
            else "matching_bounded_error"
            if error_match
            else "execution_error_mismatch"
        )
        return TransitionParityComparison(
            status=status,
            native_error=native_error,
            reference_error=reference.error_code,
            native_endpoint=native_endpoint,
            reference_endpoint=reference_endpoint,
            native_transition_steps=native_steps,
            reference_transition_steps=reference.transition_steps,
            native_forced_steps=native_forced,
            reference_forced_steps=reference.forced_steps,
            native_leaf_player=native_leaf_player,
            reference_leaf_player=reference_leaf_player,
            endpoint_match=None,
            transition_steps_match=(native_steps == reference.transition_steps),
            leaf_player_match=None,
            leaf_actor_state_match=None,
            leaf_actor_log_match=None,
            log_match=None,
            state_match=None,
            effect_match=None,
            parity_match=error_match and not chance_unsupported,
            native_parity_failure=(not error_match and not chance_unsupported),
            effect_max_abs_diff=None,
            native_state_digest=None,
            reference_state_digest=None,
            native_log_digest=None,
            reference_log_digest=None,
            manual_coin=manual_coin,
            multi_prompt=multi_prompt,
            handoff=handoff,
            native_rng_unsupported=chance_unsupported,
            reference_prize_defect_exposed=(
                reference.public_search_prize_defect_exposed
            ),
            reference_prize_defect_relevant=False,
            damage_effect=False,
            healing_effect=False,
            prize_effect=False,
            status_effect=False,
            random_effect=False,
        )

    native_leaf = native_result.decode_observation_row(row_index)
    reference_leaf = reference.leaf_observation
    if native_leaf is None or reference_leaf is None:
        raise ValueError("successful transition is missing a leaf observation")
    native_logs_raw = mapping_logs(native_leaf)
    reference_logs_raw = mapping_logs(reference_leaf)
    native_logs = _logs_without_null_fields(native_logs_raw)
    reference_logs = _logs_without_null_fields(reference_logs_raw)
    native_signature = effect_log_signature(native_logs)
    reference_signature = effect_log_signature(reference_logs)
    native_state = _state_projection(native_leaf)
    reference_state = _state_projection(reference_leaf)
    native_state_bytes = _canonical_bytes(native_state)
    reference_state_bytes = _canonical_bytes(reference_state)
    native_effect = dynamic_effect_feature_from_dict_resolution(
        select=tuple(int(value) for value in candidate_action),
        before_observation=reference.root_observation,
        after_observation=native_leaf,
        logs=native_logs,
        perspective_player=root_player,
    ).to_numpy()
    reference_effect = dynamic_effect_feature_from_dict_resolution(
        select=tuple(int(value) for value in candidate_action),
        before_observation=reference.root_observation,
        after_observation=reference_leaf,
        logs=reference_logs,
        perspective_player=root_player,
    ).to_numpy()
    max_abs_diff = float(np.max(np.abs(native_effect - reference_effect)))
    native_effect_categories = _effect_categories(native_effect)
    reference_effect_categories = _effect_categories(reference_effect)
    effect_categories = {
        name: native_effect_categories[name] or reference_effect_categories[name]
        for name in native_effect_categories
    }
    endpoint_match = native_endpoint == reference_endpoint
    transition_match = (
        native_steps == reference.transition_steps
        and native_forced == reference.forced_steps
    )
    log_match = native_signature == reference_signature
    state_match = native_state_bytes == reference_state_bytes
    effect_match = max_abs_diff <= effect_atol
    leaf_player_match = native_leaf_player == reference_leaf_player
    native_actor_leaf = native_result.decode_leaf_observation_row(row_index)
    reference_actor_leaf = reference.leaf_actor_observation
    if reference_leaf_player < 0:
        leaf_actor_state_match = (
            native_actor_leaf is None and reference_actor_leaf is None
        )
        leaf_actor_log_match = leaf_actor_state_match
    elif native_actor_leaf is None or reference_actor_leaf is None:
        leaf_actor_state_match = False
        leaf_actor_log_match = False
    else:
        leaf_actor_state_match = _canonical_bytes(
            _actor_state_projection(native_actor_leaf)
        ) == _canonical_bytes(_actor_state_projection(reference_actor_leaf))
        leaf_actor_log_match = effect_log_signature(
            _logs_without_null_fields(mapping_logs(native_actor_leaf))
        ) == effect_log_signature(
            _logs_without_null_fields(mapping_logs(reference_actor_leaf))
        )
    parity_match = all(
        (
            endpoint_match,
            transition_match,
            leaf_player_match,
            leaf_actor_state_match,
            leaf_actor_log_match,
            log_match,
            state_match,
            effect_match,
        )
    )
    prize_defect_relevant = (
        not parity_match
        and reference.public_search_prize_defect_exposed
        and (
            _prize_surface_differs(
                native_leaf,
                reference_leaf,
                native_logs_raw,
                reference_logs_raw,
            )
            or (
                endpoint_match
                and transition_match
                and leaf_player_match
                and leaf_actor_log_match
                and log_match
                and effect_match
                and (not state_match or not leaf_actor_state_match)
            )
        )
    )
    status = "match" if parity_match else "mismatch"
    if prize_defect_relevant:
        # The public Search root is known to materialize prizes face-up. Only
        # suppress a native-failure attribution when the differing transition
        # actually touches that defective prize surface. Mere exposure must
        # not hide an unrelated endpoint, log, state, or feature mismatch.
        status = "public_reference_prize_bug_surface"
    elif not parity_match and effect_categories["random_effect"]:
        # The public Search binding does not expose RNG seeding.  Its realized
        # opaque random branch is therefore an independent distributional
        # reference, not a same-sample equality target.  Seeded batch versus
        # isolated native execution is checked separately by the audit.
        status = "public_reference_unpaired_rng_realization"
    return TransitionParityComparison(
        status=status,
        native_error=native_error,
        reference_error=reference.error_code,
        native_endpoint=native_endpoint,
        reference_endpoint=reference_endpoint,
        native_transition_steps=native_steps,
        reference_transition_steps=reference.transition_steps,
        native_forced_steps=native_forced,
        reference_forced_steps=reference.forced_steps,
        native_leaf_player=native_leaf_player,
        reference_leaf_player=reference_leaf_player,
        endpoint_match=endpoint_match,
        transition_steps_match=transition_match,
        leaf_player_match=leaf_player_match,
        leaf_actor_state_match=leaf_actor_state_match,
        leaf_actor_log_match=leaf_actor_log_match,
        log_match=log_match,
        state_match=state_match,
        effect_match=effect_match,
        parity_match=parity_match,
        native_parity_failure=(status == "mismatch"),
        effect_max_abs_diff=max_abs_diff,
        native_state_digest=_digest(native_state_bytes),
        reference_state_digest=_digest(reference_state_bytes),
        native_log_digest=_digest(_canonical_bytes(native_signature)),
        reference_log_digest=_digest(_canonical_bytes(reference_signature)),
        manual_coin=manual_coin,
        multi_prompt=multi_prompt,
        handoff=handoff,
        native_rng_unsupported=False,
        reference_prize_defect_exposed=reference.public_search_prize_defect_exposed,
        reference_prize_defect_relevant=prize_defect_relevant,
        **effect_categories,
    )


def candidate_fingerprint(action: Sequence[int]) -> str:
    """Return a stable order-preserving identity for public option indices."""
    values = tuple(int(value) for value in action)
    payload = struct.pack(">I", len(values)) + b"".join(
        struct.pack(">i", value) for value in values
    )
    return hashlib.sha256(_CANDIDATE_DOMAIN + payload).hexdigest()


def _coverage_facts(
    native_endpoint: int,
    reference_endpoint: int,
    native_forced: int,
    reference_forced: int,
) -> tuple[bool, bool, bool]:
    chance = int(NativeConsequenceEndpoint.CHANCE_PROMPT)
    strategic = int(NativeConsequenceEndpoint.ROOT_STRATEGIC_PROMPT)
    handoff = int(NativeConsequenceEndpoint.TURN_HANDOFF)
    return (
        native_endpoint == chance or reference_endpoint == chance,
        native_forced > 0
        or reference_forced > 0
        or native_endpoint in {strategic, chance}
        or reference_endpoint in {strategic, chance},
        native_endpoint == handoff or reference_endpoint == handoff,
    )


def _prize_surface_differs(
    native_leaf: Mapping[str, Any],
    reference_leaf: Mapping[str, Any],
    native_logs: Sequence[Mapping[str, Any]],
    reference_logs: Sequence[Mapping[str, Any]],
) -> bool:
    if _prize_counts(native_leaf) != _prize_counts(reference_leaf):
        return True
    return any(_touches_prize(log) for log in (*native_logs, *reference_logs))


def _touches_prize(log: Mapping[str, Any]) -> bool:
    if int(log.get("type", -1)) not in {
        int(LogType.MOVE_CARD),
        int(LogType.MOVE_CARD_REVERSE),
    }:
        return False
    return int(log.get("fromArea", -1)) == int(AreaType.PRIZE) or int(
        log.get("toArea", -1)
    ) == int(AreaType.PRIZE)


def _prize_counts(observation: Mapping[str, Any]) -> tuple[int, ...]:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return ()
    players = current.get("players", ())
    if not isinstance(players, Sequence) or isinstance(players, (str, bytes)):
        return ()
    return tuple(
        len(player.get("prize", ())) if isinstance(player, Mapping) else -1
        for player in players
    )


def _state_projection(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    current = observation.get("current")
    if not isinstance(current, Mapping):
        return {"select": observation.get("select"), "current": current}
    public_current = dict(current)
    looking = public_current.get("looking")
    if isinstance(looking, Sequence) and not isinstance(looking, (str, bytes)):
        public_current["looking"] = [None] * len(looking)
    players = public_current.get("players", ())
    if isinstance(players, Sequence) and not isinstance(players, (str, bytes)):
        public_players: list[Any] = []
        for player in players:
            if not isinstance(player, Mapping):
                public_players.append(player)
                continue
            public_player = dict(player)
            # Public Search has already dropped the prior actor's hand at a
            # handoff. Compare the exact shared state and handCount, never a
            # reconstructed private identity.
            public_player["hand"] = None
            public_players.append(public_player)
        public_current["players"] = public_players
    return cast(
        Mapping[str, Any],
        _without_null_mapping_fields(
            {"select": observation.get("select"), "current": public_current}
        ),
    )


def _actor_state_projection(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep the actual actor's visible hand/select while dropping schema nulls."""
    return cast(
        Mapping[str, Any],
        _without_null_mapping_fields(
            {
                "select": observation.get("select"),
                "current": observation.get("current"),
            }
        ),
    )


def _effect_categories(effect: np.ndarray) -> dict[str, bool]:
    """Describe rule surfaces covered by an independently parsed engine row."""
    by_name = dict(zip(DYNAMIC_EFFECT_FEATURE_NAMES, effect, strict=True))
    damage_names = (
        "opponent_active_damage_norm",
        "opponent_bench_total_damage_norm",
        "self_active_damage_norm",
        "self_bench_total_damage_norm",
    )
    status_names = tuple(
        name for name in DYNAMIC_EFFECT_FEATURE_NAMES if "status_" in name
    )
    return {
        "damage_effect": any(abs(float(by_name[name])) > 0.0 for name in damage_names),
        "healing_effect": abs(float(by_name["self_active_healing_norm"])) > 0.0,
        "prize_effect": abs(float(by_name["prizes_taken_norm"])) > 0.0,
        "status_effect": any(abs(float(by_name[name])) > 0.0 for name in status_names),
        "random_effect": abs(float(by_name["coin_count_norm"])) > 0.0,
    }


def _without_null_mapping_fields(value: Any) -> Any:
    """Drop schema-only null mapping fields while preserving list positions."""
    if isinstance(value, Mapping):
        return {
            name: _without_null_mapping_fields(item)
            for name, item in value.items()
            if item is not None
        }
    if isinstance(value, list):
        return [_without_null_mapping_fields(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_without_null_mapping_fields(item) for item in value)
    return value


def _logs_without_null_fields(
    logs: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {name: value for name, value in log.items() if value is not None}
        for log in logs
    )


def _canonical_bytes(value: Any) -> bytes:
    return orjson.dumps(value, option=orjson.OPT_SORT_KEYS)


def _optional_bytes_equal(
    left: memoryview | None,
    right: memoryview | None,
) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return left == right


def _digest(value: bytes) -> str:
    return hashlib.sha256(_DIGEST_DOMAIN + value).hexdigest()


__all__ = [
    "NATIVE_UNSUPPORTED_RNG_ERROR",
    "NativeRepeatComparison",
    "TransitionParityComparison",
    "candidate_fingerprint",
    "compare_native_reference",
    "compare_native_repeat",
]
