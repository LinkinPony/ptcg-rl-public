"""Array-native GAE and PPO targets over compact stateless fragment parts."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ptcg_rl.actions.selection import ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
from ptcg_rl.decks.identity import DECK_SIZE, canonicalize_deck
from ptcg_rl.rl.stateless_fragment_io import (
    CompactFragmentPart,
    _validate_fragment_arrays,
)
from ptcg_rl.rl.stateless_macro_weights import normalized_present_deck_shares

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_FRAGMENT_CONTRACT_DOMAIN = b"ptcg-rl/stateless-fragment-contract/v1\x00"
_STATIC_CONTRACT_DOMAIN = b"ptcg-rl/stateless-fragment-static-contract/v1\x00"
_FRAGMENT_ID_DOMAIN = b"ptcg-rl/stateless-fragment-id/v1\x00"

_IDENTITY_STRING_FIELDS = (
    ("behavior_policy_fingerprint", "behavior_policy_fingerprints"),
    ("model_config_fingerprint", "model_config_fingerprints"),
    ("action_schema_fingerprint", "action_schema_fingerprints"),
    ("public_context_fingerprint", "public_context_fingerprints"),
    ("card_catalog_fingerprint", "card_catalog_fingerprints"),
    ("public_deck_catalog_fingerprint", "public_deck_catalog_fingerprints"),
    ("exact_registry_fingerprint", "exact_registry_fingerprints"),
    (
        "belief_target_semantics_fingerprint",
        "belief_target_semantics_fingerprints",
    ),
    ("input_contract_fingerprint", "input_contract_fingerprints"),
    ("resolved_config_fingerprint", "resolved_config_fingerprints"),
)
_STATIC_IDENTITY_STRING_FIELDS = _IDENTITY_STRING_FIELDS[1:]

Array = npt.NDArray[np.generic]
SourceArrays = Mapping[str, Array]


@dataclass(frozen=True)
class StatelessArrayOptimizerWindow:
    """Columnar source references and flat targets for one PPO update.

    ``decision_*`` and ``token_*`` source coordinates preserve direct access to
    immutable compact-part columns. No trajectory object graph is reconstructed.
    """

    source_arrays: tuple[SourceArrays, ...]
    static_contract_fingerprint: str
    retained_fragment_part_indices: npt.NDArray[np.int32]
    retained_fragment_rows: npt.NDArray[np.int64]
    fragment_ids: npt.NDArray[np.str_]
    behavior_version_ages: npt.NDArray[np.int64]
    decision_part_indices: npt.NDArray[np.int32]
    decision_rows: npt.NDArray[np.int64]
    decision_fragment_indices: npt.NDArray[np.int64]
    token_part_indices: npt.NDArray[np.int32]
    token_rows: npt.NDArray[np.int64]
    token_offsets: npt.NDArray[np.int64]
    root_values: npt.NDArray[np.float64]
    raw_advantages: npt.NDArray[np.float64]
    normalized_advantages: npt.NDArray[np.float64]
    return_values: npt.NDArray[np.float64]
    token_advantages: npt.NDArray[np.float64]
    token_returns: npt.NDArray[np.float64]
    belief_target_valid: npt.NDArray[np.bool_]
    deck_digests: npt.NDArray[np.str_]
    decision_macro_weights: npt.NDArray[np.float64]
    belief_macro_weights: npt.NDArray[np.float64]
    fragments_seen: int
    fragments_retained: int
    fragments_stale: int
    advantage_mean: float
    advantage_std: float

    def __post_init__(self) -> None:
        """Reject any internal alignment or normalization error."""
        decisions = int(self.decision_rows.shape[0])
        fragments = int(self.retained_fragment_rows.shape[0])
        tokens = int(self.token_rows.shape[0])
        if not self.source_arrays or decisions <= 0 or fragments <= 0:
            raise ValueError("array optimizer window must contain retained decisions")
        if _SHA256_PATTERN.fullmatch(self.static_contract_fingerprint) is None:
            raise ValueError("array optimizer window static contract is invalid")
        if (
            self.retained_fragment_part_indices.shape != (fragments,)
            or self.fragment_ids.shape != (fragments,)
            or self.behavior_version_ages.shape != (fragments,)
            or self.decision_part_indices.shape != (decisions,)
            or self.decision_fragment_indices.shape != (decisions,)
        ):
            raise ValueError("array optimizer window source rows are misaligned")
        decision_fields = (
            self.root_values,
            self.raw_advantages,
            self.normalized_advantages,
            self.return_values,
            self.belief_target_valid,
            self.deck_digests,
            self.decision_macro_weights,
            self.belief_macro_weights,
        )
        if any(field.shape != (decisions,) for field in decision_fields):
            raise ValueError("array optimizer window decision targets are misaligned")
        if (
            self.token_part_indices.shape != (tokens,)
            or self.token_advantages.shape != (tokens,)
            or self.token_returns.shape != (tokens,)
            or self.token_offsets.shape != (decisions + 1,)
            or int(self.token_offsets[0]) != 0
            or int(self.token_offsets[-1]) != tokens
        ):
            raise ValueError("array optimizer window token targets are misaligned")
        if np.any(self.decision_fragment_indices < 0) or np.any(
            self.decision_fragment_indices >= fragments
        ):
            raise ValueError("array optimizer window fragment routing is invalid")
        if (
            self.fragments_seen != self.fragments_retained + self.fragments_stale
            or self.fragments_retained != fragments
        ):
            raise ValueError("array optimizer window fragment counts are inconsistent")
        if np.any(self.behavior_version_ages < 0):
            raise ValueError("array optimizer window behavior age is negative")
        if not np.isclose(
            self.decision_macro_weights.sum(dtype=np.float64),
            1.0,
            rtol=1.0e-6,
            atol=1.0e-6,
        ):
            raise ValueError("decision deck-macro weights must sum to one")
        belief_sum = float(self.belief_macro_weights.sum(dtype=np.float64))
        if np.any(self.belief_target_valid):
            if not math.isclose(belief_sum, 1.0, rel_tol=1.0e-6, abs_tol=1.0e-6):
                raise ValueError("belief deck-macro weights must sum to one")
        elif belief_sum != 0.0:
            raise ValueError("empty belief targets cannot carry macro weight")
        token_sums = _segmented_sums(self.token_advantages, self.token_offsets)
        if not np.allclose(
            token_sums,
            self.normalized_advantages,
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise ValueError("decode-token advantages do not telescope")

    @property
    def decision_count(self) -> int:
        """Return the number of retained policy decisions."""
        return int(self.decision_rows.shape[0])

    @property
    def token_count(self) -> int:
        """Return the number of retained active decode tokens."""
        return int(self.token_rows.shape[0])

    @property
    def route_deck_digests(self) -> tuple[str, ...]:
        """Expose route identities to the existing microbatch scheduler."""
        return tuple(str(value) for value in self.deck_digests)


@dataclass(frozen=True)
class _ContiguousSourceGroup:
    """One contiguous run of retained coordinates from a compact source part."""

    part_index: int
    start: int
    stop: int


def prepare_stateless_array_optimizer_window(
    parts: Sequence[CompactFragmentPart],
    *,
    current_policy_version: int,
    maximum_version_age: int,
    gamma: float,
    gae_lambda: float,
    normalize_epsilon: float,
    deck_target_shares: Mapping[str, float] | None = None,
) -> StatelessArrayOptimizerWindow:
    """Validate compact columns and build targets without trajectory objects."""
    _validate_gae_settings(gamma, gae_lambda, normalize_epsilon)
    if current_policy_version < 0:
        raise ValueError("current policy version must be non-negative")
    if maximum_version_age < 0:
        raise ValueError("maximum version age must be non-negative")
    compact_parts = tuple(parts)
    if not compact_parts:
        raise ValueError("array optimizer window requires at least one part")
    sources = tuple(part.arrays for part in compact_parts)

    static_fingerprints: list[str] = []
    fragment_ids_seen: set[str] = set()
    all_behavior_versions: list[npt.NDArray[np.int64]] = []
    retained_rows_by_part: list[npt.NDArray[np.int64]] = []
    fragments_seen = 0
    for arrays in sources:
        static_fingerprints.extend(_validate_source_semantics(arrays))
        ids = tuple(str(value) for value in arrays["fragment_ids"])
        if len(ids) != len(set(ids)) or fragment_ids_seen.intersection(ids):
            raise ValueError("array optimizer window contains duplicate fragment IDs")
        fragment_ids_seen.update(ids)
        behavior_versions = np.asarray(
            arrays["behavior_policy_versions"],
            dtype=np.int64,
        )
        all_behavior_versions.append(behavior_versions)
        fragments_seen += int(behavior_versions.shape[0])

    if len(set(static_fingerprints)) != 1:
        raise ValueError("array optimizer window mixes static fragment contracts")
    if any(
        np.any(version > current_policy_version) for version in all_behavior_versions
    ):
        raise ValueError("current policy version cannot trail behavior")
    for versions in all_behavior_versions:
        age = current_policy_version - versions
        retained_rows_by_part.append(
            np.flatnonzero(age <= maximum_version_age).astype(
                np.int64,
                copy=False,
            )
        )
    fragments_retained = sum(int(rows.shape[0]) for rows in retained_rows_by_part)
    if fragments_retained <= 0:
        raise ValueError("array optimizer window contains no trainable fragments")

    (
        fragment_part_indices,
        fragment_rows,
        fragment_ids,
        fragment_lengths,
        fragment_bootstraps,
        decision_part_indices,
        decision_rows,
        decision_fragment_indices,
    ) = _retained_coordinates(sources, retained_rows_by_part)
    behavior_version_ages = np.concatenate(
        tuple(
            current_policy_version - versions[retained_rows]
            for versions, retained_rows in zip(
                all_behavior_versions,
                retained_rows_by_part,
                strict=True,
            )
            if retained_rows.size > 0
        )
    ).astype(np.int64, copy=False)
    decision_groups = _contiguous_source_groups(
        decision_part_indices,
        source_count=len(sources),
    )
    fragment_groups = _contiguous_source_groups(
        fragment_part_indices,
        source_count=len(sources),
    )
    root_values = _gather_decision_float64(
        sources,
        decision_rows,
        decision_groups,
        "root_values",
    )
    rewards = _gather_decision_float64(
        sources,
        decision_rows,
        decision_groups,
        "rewards",
    )
    raw_advantages, return_values = _array_gae(
        root_values,
        rewards,
        decision_fragment_indices,
        fragment_lengths,
        fragment_bootstraps,
        horizon=int(np.asarray(sources[0]["horizons"])[0]),
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    advantage_mean = float(raw_advantages.mean())
    advantage_std = float(raw_advantages.std())
    scale = advantage_std if advantage_std > normalize_epsilon else math.inf
    if math.isinf(scale):
        normalized_advantages = np.zeros_like(raw_advantages)
    else:
        normalized_advantages = (raw_advantages - advantage_mean) / scale

    (
        token_part_indices,
        token_rows,
        token_offsets,
        prefix_values,
    ) = _retained_tokens(sources, decision_rows, decision_groups)
    token_advantages, token_returns = _flat_token_targets(
        prefix_values,
        token_offsets,
        raw_advantages,
        return_values,
        normalization_mean=advantage_mean,
        normalization_scale=scale,
    )
    known_totals = _retained_known_totals(
        sources,
        decision_rows,
        decision_groups,
    )
    belief_target_valid = known_totals < DECK_SIZE
    fragment_decks = _gather_fragment_strings(
        sources,
        fragment_rows,
        fragment_groups,
        "own_deck_digests",
    )
    deck_digests = fragment_decks[decision_fragment_indices]
    decision_macro_weights = _deck_macro_weights(
        deck_digests,
        np.ones(root_values.shape[0], dtype=np.bool_),
        target_shares=deck_target_shares,
    )
    belief_macro_weights = _deck_macro_weights(
        deck_digests,
        belief_target_valid,
        target_shares=deck_target_shares,
    )
    return StatelessArrayOptimizerWindow(
        source_arrays=sources,
        static_contract_fingerprint=static_fingerprints[0],
        retained_fragment_part_indices=fragment_part_indices,
        retained_fragment_rows=fragment_rows,
        fragment_ids=fragment_ids,
        behavior_version_ages=behavior_version_ages,
        decision_part_indices=decision_part_indices,
        decision_rows=decision_rows,
        decision_fragment_indices=decision_fragment_indices,
        token_part_indices=token_part_indices,
        token_rows=token_rows,
        token_offsets=token_offsets,
        root_values=root_values,
        raw_advantages=raw_advantages,
        normalized_advantages=normalized_advantages,
        return_values=return_values,
        token_advantages=token_advantages,
        token_returns=token_returns,
        belief_target_valid=belief_target_valid,
        deck_digests=deck_digests,
        decision_macro_weights=decision_macro_weights,
        belief_macro_weights=belief_macro_weights,
        fragments_seen=fragments_seen,
        fragments_retained=fragments_retained,
        fragments_stale=fragments_seen - fragments_retained,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
    )


def _validate_source_semantics(arrays: SourceArrays) -> tuple[str, ...]:
    """Validate semantics not covered by the compact structural schema."""
    _validate_fragment_arrays(arrays)
    fragments = int(arrays["fragment_ids"].shape[0])
    decisions = int(arrays["decision_indices"].shape[0])
    fragment_offsets = np.asarray(
        arrays["fragment_decision_offsets"],
        dtype=np.int64,
    )
    lengths = np.diff(fragment_offsets)
    expected_mapping = np.repeat(
        np.arange(fragments, dtype=np.int32),
        lengths,
    )
    if not np.array_equal(
        np.asarray(arrays["decision_fragment_indices"], dtype=np.int32),
        expected_mapping,
    ):
        raise ValueError("decision-to-fragment index is corrupt")

    horizons = np.asarray(arrays["horizons"], dtype=np.int64)
    behavior_versions = np.asarray(
        arrays["behavior_policy_versions"],
        dtype=np.int64,
    )
    starts = np.asarray(arrays["start_decision_indices"], dtype=np.int64)
    generations = np.asarray(arrays["curriculum_generations"], dtype=np.int64)
    seats = np.asarray(arrays["seats"], dtype=np.int64)
    if (
        np.any(horizons <= 0)
        or np.any(behavior_versions < 0)
        or np.any(starts < 0)
        or np.any(generations < 0)
        or np.any((seats != 0) & (seats != 1))
    ):
        raise ValueError("fragment identity or context integer is invalid")
    for field in (
        *(array_field for _name, array_field in _IDENTITY_STRING_FIELDS),
        "own_deck_digests",
        "opponent_deck_digests",
        "opponent_artifact_fingerprints",
    ):
        _validate_sha256_values(arrays[field], field=field)
    if "sequence_contract_fingerprints" in arrays:
        _validate_sha256_values(
            arrays["sequence_contract_fingerprints"],
            field="sequence_contract_fingerprints",
        )
    for field in ("game_ids", "assignment_ids"):
        values = tuple(str(value) for value in arrays[field])
        if any(not value or value.strip() != value for value in values):
            raise ValueError(f"{field} contains invalid text")

    for row in range(fragments):
        own = canonicalize_deck(np.asarray(arrays["own_decks"])[row])
        opponent = canonicalize_deck(np.asarray(arrays["opponent_decks"])[row])
        if own.deck_digest != str(arrays["own_deck_digests"][row]):
            raise ValueError("fragment own deck fingerprint mismatch")
        if opponent.deck_digest != str(arrays["opponent_deck_digests"][row]):
            raise ValueError("fragment opponent deck fingerprint mismatch")

    decision_positions = np.arange(decisions, dtype=np.int64) - np.repeat(
        fragment_offsets[:-1],
        lengths,
    )
    expected_decision_indices = starts[expected_mapping] + decision_positions
    if not np.array_equal(
        np.asarray(arrays["decision_indices"], dtype=np.int64),
        expected_decision_indices,
    ):
        raise ValueError("fragment decision identities are not contiguous")
    _validate_endpoints(arrays, expected_mapping, fragment_offsets)
    _validate_action_traces(arrays)
    _validate_known_counts(arrays, fragment_offsets)
    _validate_numeric_model_inputs(arrays)

    static_fingerprints: list[str] = []
    for row in range(fragments):
        identity_payload = _identity_payload(arrays, row)
        static_payload = {
            key: value
            for key, value in identity_payload.items()
            if key not in {"behavior_policy_version", "behavior_policy_fingerprint"}
        }
        static_fingerprint = _fingerprint(
            _STATIC_CONTRACT_DOMAIN,
            static_payload,
        )
        static_fingerprints.append(static_fingerprint)
        identity_fingerprint = _fingerprint(
            _FRAGMENT_CONTRACT_DOMAIN,
            identity_payload,
        )
        expected_fragment_id = _fingerprint(
            _FRAGMENT_ID_DOMAIN,
            {
                "identity": identity_fingerprint,
                "game_id": str(arrays["game_ids"][row]),
                "seat": int(arrays["seats"][row]),
                "start_decision_index": int(arrays["start_decision_indices"][row]),
                "decisions": int(lengths[row]),
                "terminal": bool(arrays["terminal"][row]),
                "truncated": bool(arrays["truncated"][row]),
            },
        )
        if expected_fragment_id != str(arrays["fragment_ids"][row]):
            raise ValueError("stored fragment ID differs from array content")
    return tuple(static_fingerprints)


def _validate_endpoints(
    arrays: SourceArrays,
    decision_fragments: npt.NDArray[np.int32],
    fragment_offsets: npt.NDArray[np.int64],
) -> None:
    terminal = np.asarray(arrays["terminal"], dtype=np.bool_)
    truncated = np.asarray(arrays["truncated"], dtype=np.bool_)
    bootstraps = np.asarray(arrays["bootstrap_values"], dtype=np.float64)
    terminal_rewards = np.asarray(arrays["terminal_rewards"], dtype=np.float64)
    rewards = np.asarray(arrays["rewards"], dtype=np.float64)
    root_values = np.asarray(arrays["root_values"], dtype=np.float64)
    if (
        np.any(~np.isfinite(root_values))
        or np.any(np.abs(root_values) > 1.0)
        or np.any(~np.isfinite(rewards))
        or np.any(~np.isfinite(bootstraps))
        or np.any(~np.isfinite(terminal_rewards))
    ):
        raise ValueError("fragment values or rewards are invalid")
    if np.any(terminal & (bootstraps != 0.0)):
        raise ValueError("terminal fragment contains a bootstrap")
    if np.any(terminal & ~np.isin(terminal_rewards, np.asarray((-1.0, 0.0, 1.0)))):
        raise ValueError("terminal reward must be engine win/loss/draw")
    final_rows = fragment_offsets[1:] - 1
    if np.any(terminal & (rewards[final_rows] != terminal_rewards)):
        raise ValueError("terminal reward is not assigned to the final decision")
    if np.any(truncated & (terminal_rewards != 0.0)):
        raise ValueError("truncated fragment carries a terminal reward")
    if np.any(truncated & (np.abs(bootstraps) > 1.0)):
        raise ValueError("truncated fragment bootstrap is outside [-1, 1]")
    if np.any(truncated[decision_fragments] & (rewards != 0.0)):
        raise ValueError("non-terminal fragment rewards must remain zero")


def _validate_action_traces(arrays: SourceArrays) -> None:
    decisions = int(arrays["decision_indices"].shape[0])
    option_offsets = np.asarray(arrays["option_offsets"], dtype=np.int64)
    action_offsets = np.asarray(arrays["action_offsets"], dtype=np.int64)
    token_offsets = np.asarray(arrays["token_offsets"], dtype=np.int64)
    option_lengths = np.diff(option_offsets)
    action_lengths = np.diff(action_offsets)
    token_lengths = np.diff(token_offsets)
    minimum = np.asarray(arrays["min_counts"], dtype=np.int64)
    maximum = np.asarray(arrays["max_counts"], dtype=np.int64)
    if (
        np.any(minimum < 0)
        or np.any(minimum > maximum)
        or np.any(maximum > option_lengths)
    ):
        raise ValueError("fragment selection bounds are invalid")
    if np.any(action_lengths < minimum) or np.any(action_lengths > maximum):
        raise ValueError("fragment action violates selection bounds")

    action_rows = np.repeat(np.arange(decisions, dtype=np.int64), action_lengths)
    choices = np.asarray(arrays["action_choices"], dtype=np.int64)
    if np.any(choices < 0) or np.any(choices >= option_lengths[action_rows]):
        raise ValueError("fragment action references an illegal option")
    if choices.size:
        order = np.lexsort((choices, action_rows))
        ordered_rows = action_rows[order]
        ordered_choices = choices[order]
        if np.any(
            (ordered_rows[1:] == ordered_rows[:-1])
            & (ordered_choices[1:] == ordered_choices[:-1])
        ):
            raise ValueError("fragment action contains duplicate options")

    option_rows = np.repeat(np.arange(decisions, dtype=np.int64), option_lengths)
    unordered = np.zeros(decisions, dtype=np.bool_)
    contexts = np.asarray(arrays["option_contexts"], dtype=np.int64)
    if contexts.size:
        np.logical_or.at(
            unordered,
            option_rows,
            np.isin(
                contexts,
                np.asarray(tuple(ENGINE_PROVEN_UNORDERED_SET_CONTEXTS)),
            ),
        )
    count_first = (minimum < maximum) & unordered
    if choices.size > 1:
        same_action = action_rows[1:] == action_rows[:-1]
        noncanonical = (
            same_action & count_first[action_rows[1:]] & (choices[1:] < choices[:-1])
        )
        if np.any(noncanonical):
            raise ValueError("count-first fragment action is not canonical")
    stop_sampled = np.asarray(arrays["stop_sampled"], dtype=np.bool_)
    expected_stop = (~count_first) & (action_lengths < maximum)
    if not np.array_equal(stop_sampled, expected_stop):
        raise ValueError("fragment STOP flag differs from action termination")
    expected_token_lengths = (
        action_lengths + count_first.astype(np.int64) + stop_sampled.astype(np.int64)
    )
    if np.any(token_lengths <= 0) or not np.array_equal(
        token_lengths,
        expected_token_lengths,
    ):
        raise ValueError("decode-token trace differs from action termination")

    token_logprobs = np.asarray(arrays["token_logprobs"], dtype=np.float64)
    prefix_values = np.asarray(arrays["prefix_values"], dtype=np.float64)
    action_logprobs = np.asarray(arrays["action_logprobs"], dtype=np.float64)
    if (
        np.any(~np.isfinite(token_logprobs))
        or np.any(~np.isfinite(prefix_values))
        or np.any(np.abs(prefix_values) > 1.0)
        or np.any(~np.isfinite(action_logprobs))
        or not np.allclose(
            _segmented_sums(token_logprobs, token_offsets),
            action_logprobs,
            rtol=1.0e-5,
            atol=1.0e-6,
        )
    ):
        raise ValueError("fragment decode-token values are invalid")


def _validate_known_counts(
    arrays: SourceArrays,
    fragment_offsets: npt.NDArray[np.int64],
) -> None:
    decisions = int(arrays["decision_indices"].shape[0])
    known_offsets = np.asarray(arrays["known_offsets"], dtype=np.int64)
    known_lengths = np.diff(known_offsets)
    known_rows = np.repeat(np.arange(decisions, dtype=np.int64), known_lengths)
    card_ids = np.asarray(arrays["known_card_ids"], dtype=np.int64)
    counts = np.asarray(arrays["known_counts"], dtype=np.int64)
    if np.any(card_ids <= 0) or np.any(counts <= 0):
        raise ValueError("known opponent counts must be positive")
    if card_ids.size > 1 and np.any(
        (known_rows[1:] == known_rows[:-1]) & (card_ids[1:] <= card_ids[:-1])
    ):
        raise ValueError("known opponent card IDs must be unique and sorted")
    if np.any(_segmented_sums(counts, known_offsets) > DECK_SIZE):
        raise ValueError("public evidence exceeds exact opponent deck")

    opponent_decks = np.asarray(arrays["opponent_decks"], dtype=np.int64)
    for fragment, (decision_start, decision_stop) in enumerate(
        zip(fragment_offsets[:-1], fragment_offsets[1:], strict=True)
    ):
        exact_ids, exact_counts = np.unique(
            opponent_decks[fragment],
            return_counts=True,
        )
        known_start = int(known_offsets[int(decision_start)])
        known_stop = int(known_offsets[int(decision_stop)])
        fragment_ids = card_ids[known_start:known_stop]
        fragment_counts = counts[known_start:known_stop]
        locations = np.searchsorted(exact_ids, fragment_ids)
        in_bounds = locations < exact_ids.shape[0]
        present = np.zeros(fragment_ids.shape[0], dtype=np.bool_)
        present[in_bounds] = exact_ids[locations[in_bounds]] == fragment_ids[in_bounds]
        covered = np.zeros(fragment_ids.shape[0], dtype=np.bool_)
        covered[present] = exact_counts[locations[present]] >= fragment_counts[present]
        if not np.all(covered):
            raise ValueError(
                "public evidence cannot be subtracted from learner opponent deck"
            )


def _validate_numeric_model_inputs(arrays: SourceArrays) -> None:
    belief_ids = np.asarray(arrays["belief_card_ids"], dtype=np.int64)
    belief_counts = np.asarray(arrays["belief_expected_counts"], dtype=np.float64)
    belief_scalars = np.asarray(arrays["belief_scalars"], dtype=np.float64)
    if (
        np.any(belief_ids <= 0)
        or np.any(~np.isfinite(belief_counts))
        or np.any(belief_counts < 0.0)
        or np.any(~np.isfinite(belief_scalars))
        or np.any(belief_scalars[:, 0] < 0.0)
        or np.any(belief_scalars[:, 1:3] < 0.0)
        or np.any((belief_scalars[:, 3] < 0.0) | (belief_scalars[:, 3] > 1.0))
    ):
        raise ValueError("public belief summary contains invalid values")
    for field in (
        "state_scalars",
        "option_scalars",
        "option_dynamic_effect_features",
    ):
        if np.any(~np.isfinite(np.asarray(arrays[field], dtype=np.float64))):
            raise ValueError(f"{field} contains non-finite values")


def _identity_payload(arrays: SourceArrays, row: int) -> dict[str, object]:
    sequence = "fragment_schema_versions" in arrays
    payload: dict[str, object] = {
        "schema_version": (
            int(arrays["fragment_schema_versions"][row]) if sequence else 1
        ),
        "horizon": int(arrays["horizons"][row]),
        "behavior_policy_version": int(arrays["behavior_policy_versions"][row]),
    }
    payload.update(
        {name: str(arrays[field][row]) for name, field in _IDENTITY_STRING_FIELDS}
    )
    payload["sequence_contract_fingerprint"] = (
        str(arrays["sequence_contract_fingerprints"][row])
        if sequence
        else None
    )
    return payload


def _retained_coordinates(
    sources: tuple[SourceArrays, ...],
    retained_rows_by_part: Sequence[npt.NDArray[np.int64]],
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.str_],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
]:
    fragment_parts: list[npt.NDArray[np.int32]] = []
    fragment_rows: list[npt.NDArray[np.int64]] = []
    fragment_ids: list[npt.NDArray[np.str_]] = []
    fragment_lengths: list[npt.NDArray[np.int64]] = []
    fragment_bootstraps: list[npt.NDArray[np.float64]] = []
    decision_parts: list[npt.NDArray[np.int32]] = []
    decision_rows: list[npt.NDArray[np.int64]] = []
    decision_fragments: list[npt.NDArray[np.int64]] = []
    fragment_base = 0
    for part_index, (arrays, retained_rows) in enumerate(
        zip(sources, retained_rows_by_part, strict=True)
    ):
        if retained_rows.size == 0:
            continue
        offsets = np.asarray(arrays["fragment_decision_offsets"], dtype=np.int64)
        starts = offsets[retained_rows]
        stops = offsets[retained_rows + 1]
        lengths = stops - starts
        local_decision_rows = _expand_ranges(starts, lengths)
        fragment_parts.append(np.full(retained_rows.shape, part_index, dtype=np.int32))
        fragment_rows.append(retained_rows)
        fragment_ids.append(
            np.asarray(arrays["fragment_ids"][retained_rows], dtype=np.str_)
        )
        fragment_lengths.append(lengths)
        fragment_bootstraps.append(
            np.asarray(arrays["bootstrap_values"][retained_rows], dtype=np.float64)
        )
        decision_parts.append(
            np.full(local_decision_rows.shape, part_index, dtype=np.int32)
        )
        decision_rows.append(local_decision_rows)
        decision_fragments.append(
            np.repeat(
                np.arange(
                    fragment_base,
                    fragment_base + retained_rows.shape[0],
                    dtype=np.int64,
                ),
                lengths,
            )
        )
        fragment_base += int(retained_rows.shape[0])
    return (
        np.concatenate(fragment_parts),
        np.concatenate(fragment_rows),
        np.concatenate(fragment_ids),
        np.concatenate(fragment_lengths),
        np.concatenate(fragment_bootstraps),
        np.concatenate(decision_parts),
        np.concatenate(decision_rows),
        np.concatenate(decision_fragments),
    )


def _array_gae(
    root_values: npt.NDArray[np.float64],
    rewards: npt.NDArray[np.float64],
    decision_fragments: npt.NDArray[np.int64],
    fragment_lengths: npt.NDArray[np.int64],
    fragment_bootstraps: npt.NDArray[np.float64],
    *,
    horizon: int,
    gamma: float,
    gae_lambda: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    fragments = int(fragment_lengths.shape[0])
    values = np.zeros((fragments, horizon), dtype=np.float64)
    reward_matrix = np.zeros_like(values)
    fragment_starts = np.zeros(fragments, dtype=np.int64)
    fragment_starts[1:] = np.cumsum(fragment_lengths[:-1])
    positions = np.arange(root_values.shape[0], dtype=np.int64) - np.repeat(
        fragment_starts,
        fragment_lengths,
    )
    values[decision_fragments, positions] = root_values
    reward_matrix[decision_fragments, positions] = rewards
    advantages = np.zeros_like(values)
    running = np.zeros(fragments, dtype=np.float64)
    discount = gamma * gae_lambda
    for timestep in range(horizon - 1, -1, -1):
        active = fragment_lengths > timestep
        next_values = np.where(
            fragment_lengths > timestep + 1,
            values[:, min(timestep + 1, horizon - 1)],
            fragment_bootstraps,
        )
        delta = reward_matrix[:, timestep] + gamma * next_values - values[:, timestep]
        running = np.where(active, delta + discount * running, 0.0)
        advantages[:, timestep] = running
    flattened = advantages[decision_fragments, positions]
    return flattened, flattened + root_values


def _retained_tokens(
    sources: tuple[SourceArrays, ...],
    decision_rows: npt.NDArray[np.int64],
    groups: tuple[_ContiguousSourceGroup, ...],
) -> tuple[
    npt.NDArray[np.int32],
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.float64],
]:
    token_parts: list[npt.NDArray[np.int32]] = []
    token_rows: list[npt.NDArray[np.int64]] = []
    prefix_values: list[npt.NDArray[np.float64]] = []
    token_lengths: list[npt.NDArray[np.int64]] = []
    for group in groups:
        arrays = sources[group.part_index]
        selected = decision_rows[group.start : group.stop]
        offsets = np.asarray(arrays["token_offsets"], dtype=np.int64)
        starts = offsets[selected]
        lengths = offsets[selected + 1] - starts
        local_token_rows = _expand_ranges(starts, lengths)
        token_parts.append(
            np.full(local_token_rows.shape, group.part_index, dtype=np.int32)
        )
        token_rows.append(local_token_rows)
        prefix_values.append(
            np.asarray(
                arrays["prefix_values"][local_token_rows],
                dtype=np.float64,
            )
        )
        token_lengths.append(lengths)
    lengths = np.concatenate(token_lengths)
    offsets = np.zeros(decision_rows.shape[0] + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths)
    return (
        np.concatenate(token_parts),
        np.concatenate(token_rows),
        offsets,
        np.concatenate(prefix_values),
    )


def _flat_token_targets(
    prefix_values: npt.NDArray[np.float64],
    token_offsets: npt.NDArray[np.int64],
    raw_advantages: npt.NDArray[np.float64],
    return_values: npt.NDArray[np.float64],
    *,
    normalization_mean: float,
    normalization_scale: float,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lengths = np.diff(token_offsets)
    decision_rows = np.repeat(
        np.arange(raw_advantages.shape[0], dtype=np.int64),
        lengths,
    )
    next_values = np.empty_like(prefix_values)
    if prefix_values.shape[0] > 1:
        next_values[:-1] = prefix_values[1:]
    ends = token_offsets[1:] - 1
    next_values[ends] = return_values
    raw_token_advantages = next_values - prefix_values
    starts = token_offsets[:-1]
    raw_token_advantages[starts] += raw_advantages - _segmented_sums(
        raw_token_advantages,
        token_offsets,
    )
    if math.isinf(normalization_scale):
        normalized = np.zeros_like(raw_token_advantages)
    else:
        normalized = raw_token_advantages / normalization_scale
        normalized[starts] -= normalization_mean / normalization_scale
    return normalized, return_values[decision_rows]


def _retained_known_totals(
    sources: tuple[SourceArrays, ...],
    decision_rows: npt.NDArray[np.int64],
    groups: tuple[_ContiguousSourceGroup, ...],
) -> npt.NDArray[np.int64]:
    totals: list[npt.NDArray[np.int64]] = []
    for group in groups:
        arrays = sources[group.part_index]
        selected = decision_rows[group.start : group.stop]
        offsets = np.asarray(arrays["known_offsets"], dtype=np.int64)
        cumulative = np.zeros(
            int(arrays["known_counts"].shape[0]) + 1,
            dtype=np.int64,
        )
        cumulative[1:] = np.cumsum(np.asarray(arrays["known_counts"], dtype=np.int64))
        totals.append(cumulative[offsets[selected + 1]] - cumulative[offsets[selected]])
    return np.concatenate(totals)


def _gather_decision_float64(
    sources: tuple[SourceArrays, ...],
    decision_rows: npt.NDArray[np.int64],
    groups: tuple[_ContiguousSourceGroup, ...],
    field: str,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        np.concatenate(
            [
                np.asarray(
                    sources[group.part_index][field][
                        decision_rows[group.start : group.stop]
                    ],
                    dtype=np.float64,
                )
                for group in groups
            ]
        ),
        dtype=np.float64,
    )


def _gather_fragment_strings(
    sources: tuple[SourceArrays, ...],
    fragment_rows: npt.NDArray[np.int64],
    groups: tuple[_ContiguousSourceGroup, ...],
    field: str,
) -> npt.NDArray[np.str_]:
    return np.concatenate(
        [
            np.asarray(
                sources[group.part_index][field][
                    fragment_rows[group.start : group.stop]
                ],
                dtype=np.str_,
            )
            for group in groups
        ]
    )


def _contiguous_source_groups(
    part_indices: npt.NDArray[np.int32],
    *,
    source_count: int,
) -> tuple[_ContiguousSourceGroup, ...]:
    """Group part-major coordinates once without full-window masks per part."""
    if part_indices.ndim != 1 or part_indices.size == 0:
        raise RuntimeError("retained source coordinates are empty or malformed")
    if (
        np.any(part_indices < 0)
        or np.any(part_indices >= source_count)
        or np.any(part_indices[1:] < part_indices[:-1])
    ):
        raise RuntimeError("retained source coordinates are not part-major")
    starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(part_indices[1:] != part_indices[:-1]).astype(
                np.int64,
                copy=False,
            )
            + 1,
        )
    )
    stops = np.concatenate(
        (starts[1:], np.asarray([part_indices.shape[0]], dtype=np.int64))
    )
    return tuple(
        _ContiguousSourceGroup(
            part_index=int(part_indices[int(start)]),
            start=int(start),
            stop=int(stop),
        )
        for start, stop in zip(starts, stops, strict=True)
    )


def _deck_macro_weights(
    deck_digests: npt.NDArray[np.str_],
    eligible: npt.NDArray[np.bool_],
    *,
    target_shares: Mapping[str, float] | None = None,
) -> npt.NDArray[np.float64]:
    weights = np.zeros(deck_digests.shape[0], dtype=np.float64)
    selected = np.flatnonzero(eligible)
    if selected.size == 0:
        return weights
    decks, inverse, counts = np.unique(
        deck_digests[selected],
        return_inverse=True,
        return_counts=True,
    )
    shares = normalized_present_deck_shares(
        (str(deck) for deck in decks),
        target_shares,
    )
    macro = np.asarray([shares[str(deck)] for deck in decks], dtype=np.float64)
    weights[selected] = macro[inverse] / counts[inverse].astype(np.float64)
    return weights


def _expand_ranges(
    starts: npt.NDArray[np.int64],
    lengths: npt.NDArray[np.int64],
) -> npt.NDArray[np.int64]:
    output_offsets = np.zeros(lengths.shape[0], dtype=np.int64)
    if lengths.shape[0] > 1:
        output_offsets[1:] = np.cumsum(lengths[:-1])
    return (
        np.repeat(starts, lengths)
        + np.arange(int(lengths.sum()), dtype=np.int64)
        - np.repeat(output_offsets, lengths)
    )


def _segmented_sums(
    values: npt.NDArray[np.generic],
    offsets: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    normalized = np.asarray(values, dtype=np.float64)
    cumulative = np.zeros(values.shape[0] + 1, dtype=np.float64)
    cumulative[1:] = np.cumsum(normalized)
    return cumulative[offsets[1:]] - cumulative[offsets[:-1]]


def _validate_sha256_values(values: Array, *, field: str) -> None:
    if any(_SHA256_PATTERN.fullmatch(str(value)) is None for value in values):
        raise ValueError(f"{field} contains an invalid SHA-256 fingerprint")


def _fingerprint(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _validate_gae_settings(
    gamma: float,
    gae_lambda: float,
    normalize_epsilon: float,
) -> None:
    if (
        not math.isfinite(gamma)
        or not 0.0 <= gamma <= 1.0
        or not math.isfinite(gae_lambda)
        or not 0.0 <= gae_lambda <= 1.0
    ):
        raise ValueError("GAE gamma and lambda must lie in [0, 1]")
    if not math.isfinite(normalize_epsilon) or normalize_epsilon <= 0.0:
        raise ValueError("advantage normalization epsilon must be positive")


__all__ = [
    "StatelessArrayOptimizerWindow",
    "prepare_stateless_array_optimizer_window",
]
