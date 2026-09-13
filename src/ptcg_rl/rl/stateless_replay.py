"""Reconstruct and target compact stateless fragments for one PPO window."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import numpy.typing as npt

from ptcg_rl.actions.encoding import EncodedOptionArrayFeatures
from ptcg_rl.belief.public_catalog import (
    ExpectedRemainingCard,
    PublicDeckPosterior,
)
from ptcg_rl.context.public_event_arrays import PublicEventArrayBlock
from ptcg_rl.context.public_events import PublicEventDelta
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.model.sequence.action import (
    ACCEPTED_ACTION_SCHEMA_VERSION,
    AcceptedActionRecord,
)
from ptcg_rl.model.state_encoder import StateTokenArrayFeatures
from ptcg_rl.rl.policy_inputs import SimpleStatelessActorRow
from ptcg_rl.rl.sequence_types import SequenceDecisionIdentity
from ptcg_rl.rl.stateless_fragment import (
    StatelessFragment,
    StatelessFragmentContext,
    StatelessFragmentDecision,
    StatelessFragmentIdentity,
    fragment_is_trainable,
)
from ptcg_rl.rl.stateless_fragment_io import CompactFragmentPart
from ptcg_rl.rl.stateless_macro_weights import normalized_present_deck_shares
from ptcg_rl.rl.token_credit import compute_prompt_token_credit


@dataclass(frozen=True)
class StatelessPpoTarget:
    """One decision and its root/prefix PPO targets."""

    fragment_id: str
    decision: StatelessFragmentDecision
    opponent_deck: tuple[int, ...]
    deck_digest: str
    raw_advantage: float
    normalized_advantage: float
    return_value: float
    token_advantages: tuple[float, ...]
    token_returns: tuple[float, ...]
    belief_target_valid: bool
    game_id: str = ""
    seat: Literal[0, 1] = 0


@dataclass(frozen=True)
class SequenceContextDecision:
    """Raw complete block available as current-weight learner context."""

    game_id: str
    seat: Literal[0, 1]
    decision: StatelessFragmentDecision
    loss_bearing: bool


@dataclass(frozen=True)
class StatelessOptimizerWindow:
    """All retained decisions normalized together for one optimizer update."""

    targets: tuple[StatelessPpoTarget, ...]
    fragments_seen: int
    fragments_retained: int
    fragments_stale: int
    advantage_mean: float
    advantage_std: float
    decision_macro_weights: tuple[float, ...]
    belief_macro_weights: tuple[float, ...]
    sequence_context: tuple[SequenceContextDecision, ...] = ()

    def __post_init__(self) -> None:
        """Require complete global and deck-macro alignment."""
        size = len(self.targets)
        if size <= 0:
            raise ValueError("optimizer window must contain retained decisions")
        if (
            len(self.decision_macro_weights) != size
            or len(self.belief_macro_weights) != size
        ):
            raise ValueError("optimizer window weights are misaligned")
        if not math.isclose(
            sum(self.decision_macro_weights),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("decision deck-macro weights must sum to one")
        valid_belief = any(target.belief_target_valid for target in self.targets)
        belief_sum = sum(self.belief_macro_weights)
        if valid_belief and not math.isclose(
            belief_sum,
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError("belief deck-macro weights must sum to one")
        if not valid_belief and belief_sum != 0.0:
            raise ValueError("empty belief targets cannot carry macro weight")
        if any(
            not math.isclose(
                sum(target.token_advantages),
                target.normalized_advantage,
                rel_tol=1e-5,
                abs_tol=1e-5,
            )
            for target in self.targets
        ):
            raise ValueError("decode-token advantages do not telescope")
        if self.sequence_context:
            keys = tuple(
                (row.game_id, row.seat, row.decision.decision_index)
                for row in self.sequence_context
            )
            if len(keys) != len(set(keys)):
                raise ValueError("sequence context contains duplicate decisions")
            grouped: dict[tuple[str, int], list[int]] = {}
            for game_id, seat, decision_index in keys:
                grouped.setdefault((game_id, seat), []).append(decision_index)
            if any(
                indices != list(range(indices[0], indices[-1] + 1))
                for indices in grouped.values()
            ):
                raise ValueError("sequence context has a discontinuous game-seat tape")

    @property
    def deck_digests(self) -> tuple[str, ...]:
        """Return target route identities in decision order."""
        return tuple(target.deck_digest for target in self.targets)


def reconstruct_stateless_fragment_part(
    part: CompactFragmentPart,
) -> tuple[StatelessFragment, ...]:
    """Rebuild immutable fragments and rerun their semantic validators."""
    arrays = part.arrays
    fragment_offsets = _integers(arrays["fragment_decision_offsets"])
    fragments: list[StatelessFragment] = []
    for fragment_index in range(part.fragment_count):
        identity = _fragment_identity(arrays, fragment_index)
        context = _fragment_context(arrays, fragment_index)
        start = fragment_offsets[fragment_index]
        stop = fragment_offsets[fragment_index + 1]
        decisions = tuple(
            _decision(
                arrays,
                decision_index,
                identity=identity,
                context=context,
            )
            for decision_index in range(start, stop)
        )
        fragment = StatelessFragment(
            identity=identity,
            context=context,
            decisions=decisions,
            terminal=bool(arrays["terminal"][fragment_index]),
            truncated=bool(arrays["truncated"][fragment_index]),
            bootstrap_value=float(arrays["bootstrap_values"][fragment_index]),
            terminal_reward=float(arrays["terminal_rewards"][fragment_index]),
        )
        if fragment.fragment_id != str(arrays["fragment_ids"][fragment_index]):
            raise ValueError("stored fragment ID differs from reconstructed content")
        if any(
            int(arrays["decision_fragment_indices"][row]) != fragment_index
            for row in range(start, stop)
        ):
            raise ValueError("decision-to-fragment index is corrupt")
        fragments.append(fragment)
    return tuple(fragments)


def prepare_stateless_optimizer_window(
    fragments: Sequence[StatelessFragment],
    *,
    current_policy_version: int,
    maximum_version_age: int,
    gamma: float,
    gae_lambda: float,
    normalize_epsilon: float,
    deck_target_shares: Mapping[str, float] | None = None,
) -> StatelessOptimizerWindow:
    """Filter whole fragments, run bootstrapped GAE, then normalize once."""
    _validate_gae_settings(gamma, gae_lambda, normalize_epsilon)
    if not fragments:
        raise ValueError("optimizer window requires at least one fragment")
    static_contract = fragments[0].identity.static_contract_fingerprint
    if any(
        fragment.identity.static_contract_fingerprint != static_contract
        for fragment in fragments
    ):
        raise ValueError("optimizer window mixes static fragment contracts")
    retained = tuple(
        fragment
        for fragment in fragments
        if fragment_is_trainable(
            fragment,
            current_policy_version=current_policy_version,
            maximum_version_age=maximum_version_age,
        )
    )
    if not retained:
        raise ValueError("optimizer window contains no trainable fragments")
    retained_ids = {fragment.fragment_id for fragment in retained}
    sequence_enabled = fragments[0].identity.schema_version == 2
    sequence_rows: list[SequenceContextDecision] = []
    if sequence_enabled:
        ordered_fragments = sorted(
            fragments,
            key=lambda item: (
                item.context.game_id,
                item.context.seat,
                item.context.start_decision_index,
            ),
        )
        for fragment in ordered_fragments:
            sequence_rows.extend(
                SequenceContextDecision(
                    game_id=fragment.context.game_id,
                    seat=fragment.context.seat,
                    decision=decision,
                    loss_bearing=fragment.fragment_id in retained_ids,
                )
                for decision in fragment.decisions
            )
    flattened: list[
        tuple[
            StatelessFragment,
            StatelessFragmentDecision,
            float,
            float,
        ]
    ] = []
    for fragment in retained:
        advantages, returns = _fragment_gae(
            fragment,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        flattened.extend(
            (fragment, decision, advantage, return_value)
            for decision, advantage, return_value in zip(
                fragment.decisions,
                advantages,
                returns,
                strict=True,
            )
        )
    raw_advantages = np.asarray(
        [item[2] for item in flattened],
        dtype=np.float64,
    )
    mean = float(raw_advantages.mean())
    std = float(raw_advantages.std())
    scale = std if std > normalize_epsilon else math.inf
    targets: list[StatelessPpoTarget] = []
    for fragment, decision, advantage, return_value in flattened:
        normalized = (advantage - mean) / scale
        token_advantages = _normalized_token_credit(
            decision.prefix_values,
            downstream_return=return_value,
            root_advantage=advantage,
            normalization_mean=mean,
            normalization_scale=scale,
        )
        targets.append(
            StatelessPpoTarget(
                fragment_id=fragment.fragment_id,
                decision=decision,
                opponent_deck=fragment.context.opponent_deck,
                deck_digest=fragment.context.own_deck_digest,
                raw_advantage=advantage,
                normalized_advantage=normalized,
                return_value=return_value,
                token_advantages=token_advantages,
                token_returns=(return_value,) * len(token_advantages),
                belief_target_valid=(
                    sum(
                        count
                        for _card_id, count in decision.known_opponent_counts
                    )
                    < len(fragment.context.opponent_deck)
                ),
                game_id=fragment.context.game_id,
                seat=fragment.context.seat,
            )
        )
    target_tuple = tuple(targets)
    return StatelessOptimizerWindow(
        targets=target_tuple,
        fragments_seen=len(fragments),
        fragments_retained=len(retained),
        fragments_stale=len(fragments) - len(retained),
        advantage_mean=mean,
        advantage_std=std,
        decision_macro_weights=_deck_macro_weights(
            target_tuple,
            belief_only=False,
            target_shares=deck_target_shares,
        ),
        belief_macro_weights=_deck_macro_weights(
            target_tuple,
            belief_only=True,
            target_shares=deck_target_shares,
        ),
        sequence_context=tuple(sequence_rows),
    )


def _fragment_identity(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    index: int,
) -> StatelessFragmentIdentity:
    return StatelessFragmentIdentity(
        schema_version=cast(
            Literal[1, 2],
            (
            int(arrays["fragment_schema_versions"][index])
            if "fragment_schema_versions" in arrays
            else 1
            ),
        ),
        horizon=int(arrays["horizons"][index]),
        behavior_policy_version=int(arrays["behavior_policy_versions"][index]),
        behavior_policy_fingerprint=str(
            arrays["behavior_policy_fingerprints"][index]
        ),
        model_config_fingerprint=str(arrays["model_config_fingerprints"][index]),
        action_schema_fingerprint=str(arrays["action_schema_fingerprints"][index]),
        public_context_fingerprint=str(
            arrays["public_context_fingerprints"][index]
        ),
        card_catalog_fingerprint=str(arrays["card_catalog_fingerprints"][index]),
        public_deck_catalog_fingerprint=str(
            arrays["public_deck_catalog_fingerprints"][index]
        ),
        exact_registry_fingerprint=str(
            arrays["exact_registry_fingerprints"][index]
        ),
        belief_target_semantics_fingerprint=str(
            arrays["belief_target_semantics_fingerprints"][index]
        ),
        input_contract_fingerprint=str(
            arrays["input_contract_fingerprints"][index]
        ),
        resolved_config_fingerprint=str(
            arrays["resolved_config_fingerprints"][index]
        ),
        sequence_contract_fingerprint=(
            str(arrays["sequence_contract_fingerprints"][index])
            if "sequence_contract_fingerprints" in arrays
            else None
        ),
    )


def _fragment_context(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    index: int,
) -> StatelessFragmentContext:
    own_deck = tuple(int(value) for value in arrays["own_decks"][index])
    opponent_deck = tuple(
        int(value) for value in arrays["opponent_decks"][index]
    )
    return StatelessFragmentContext(
        game_id=str(arrays["game_ids"][index]),
        seat=int(arrays["seats"][index]),  # type: ignore[arg-type]
        start_decision_index=int(arrays["start_decision_indices"][index]),
        own_deck=own_deck,
        own_deck_digest=str(arrays["own_deck_digests"][index]),
        opponent_deck=opponent_deck,
        opponent_deck_digest=str(arrays["opponent_deck_digests"][index]),
        curriculum_generation=int(arrays["curriculum_generations"][index]),
        assignment_id=str(arrays["assignment_ids"][index]),
        opponent_artifact_fingerprint=str(
            arrays["opponent_artifact_fingerprints"][index]
        ),
    )


def _decision(
    arrays: Mapping[str, npt.NDArray[np.generic]],
    row: int,
    *,
    identity: StatelessFragmentIdentity,
    context: StatelessFragmentContext,
) -> StatelessFragmentDecision:
    values = arrays
    state_start, state_stop = _row_bounds(values["state_offsets"], row)
    attachment_start, attachment_stop = _row_bounds(
        values["attachment_offsets"],
        row,
    )
    option_start, option_stop = _row_bounds(values["option_offsets"], row)
    belief_start, belief_stop = _row_bounds(values["belief_offsets"], row)
    known_start, known_stop = _row_bounds(values["known_offsets"], row)
    action_start, action_stop = _row_bounds(values["action_offsets"], row)
    token_start, token_stop = _row_bounds(values["token_offsets"], row)
    state = StateTokenArrayFeatures(
        card_ids=values["state_card_ids"][state_start:state_stop],
        areas=values["state_areas"][state_start:state_stop],
        owner_roles=values["state_owner_roles"][state_start:state_stop],
        token_kinds=values["state_token_kinds"][state_start:state_stop],
        scalars=values["state_scalars"][state_start:state_stop],
        last_attack_ids=values["state_last_attack_ids"][state_start:state_stop],
        attachment_card_ids=values["attachment_card_ids"][
            attachment_start:attachment_stop
        ],
        attachment_parent_indices=values["attachment_parent_indices"][
            attachment_start:attachment_stop
        ],
        attachment_kinds=values["attachment_kinds"][
            attachment_start:attachment_stop
        ],
        entity_slots=values["state_entity_slots"][state_start:state_stop],
    )
    options = EncodedOptionArrayFeatures(
        option_types=values["option_types"][option_start:option_stop],
        contexts=values["option_contexts"][option_start:option_stop],
        entity_slots=values["option_entity_slots"][option_start:option_stop],
        entity_slot_mask=values["option_entity_slot_mask"][
            option_start:option_stop
        ],
        attack_ids=values["option_attack_ids"][option_start:option_stop],
        card_ids=values["option_card_ids"][option_start:option_stop],
        scalars=values["option_scalars"][option_start:option_stop],
        dynamic_effect_features=values["option_dynamic_effect_features"][
            option_start:option_stop
        ],
        dynamic_effect_masks=values["option_dynamic_effect_masks"][
            option_start:option_stop
        ],
    )
    belief_scalars = values["belief_scalars"][row]
    posterior = PublicDeckPosterior(
        entries=(),
        unknown_probability=float(belief_scalars[3]),
        expected_remaining=tuple(
            ExpectedRemainingCard(
                card_id=int(values["belief_card_ids"][index]),
                expected_count=float(values["belief_expected_counts"][index]),
            )
            for index in range(belief_start, belief_stop)
        ),
        entropy=float(belief_scalars[0]),
        compatible_deck_count=int(belief_scalars[1]),
        public_evidence_count=int(belief_scalars[2]),
    )
    own_deck = canonicalize_deck(context.own_deck)
    sequence_identity = (
        SequenceDecisionIdentity(
            game_id=context.game_id,
            seat=context.seat,
            decision_index=int(values["decision_indices"][row]),
            request_id=str(values["sequence_request_ids"][row]),
        )
        if identity.schema_version == 2
        else None
    )
    producer_fingerprint = (
        str(values["engine_fact_producer_fingerprints"][row])
        if identity.schema_version == 2
        else ""
    )
    actor_row = SimpleStatelessActorRow(
        state=state,
        options=options,
        min_count=int(values["min_counts"][row]),
        max_count=int(values["max_counts"][row]),
        own_deck=own_deck,
        belief_summary=posterior,
        catalog_fingerprint=identity.public_deck_catalog_fingerprint,
        input_contract_fingerprint=identity.input_contract_fingerprint,
        public_event_delta=(
            _public_event_array_block(values).delta_at(row)
            if identity.schema_version == 2
            else PublicEventDelta()
        ),
        engine_fact_producer_fingerprint=(
            producer_fingerprint or None
        ),
        sequence_identity=sequence_identity,
    )
    accepted_action = (
        _accepted_action(values, row, action_start, action_stop)
        if identity.schema_version == 2
        else None
    )
    return StatelessFragmentDecision(
        decision_index=int(values["decision_indices"][row]),
        actor_row=actor_row,
        action=tuple(
            int(value)
            for value in values["action_choices"][action_start:action_stop]
        ),
        action_logprob=float(values["action_logprobs"][row]),
        token_logprobs=tuple(
            float(value)
            for value in values["token_logprobs"][token_start:token_stop]
        ),
        prefix_values=tuple(
            float(value)
            for value in values["prefix_values"][token_start:token_stop]
        ),
        root_value=float(values["root_values"][row]),
        stop_sampled=bool(values["stop_sampled"][row]),
        known_opponent_counts=tuple(
            (
                int(values["known_card_ids"][index]),
                int(values["known_counts"][index]),
            )
            for index in range(known_start, known_stop)
        ),
        public_event_delta=(
            actor_row.public_event_delta
            if identity.schema_version == 2
            else None
        ),
        accepted_action=accepted_action,
        reward=float(values["rewards"][row]),
    )


def _public_event_array_block(
    values: Mapping[str, npt.NDArray[np.generic]],
) -> PublicEventArrayBlock:
    """View persisted V2 EVENT columns through the canonical CSR contract."""
    return PublicEventArrayBlock(
        event_offsets=values["event_offsets"],
        event_types=values["event_types"],
        actor_roles=values["event_actor_roles"],
        from_areas=values["event_from_areas"],
        to_areas=values["event_to_areas"],
        card_ids=values["event_card_ids"],
        serials=values["event_serials"],
        entity_mask=values["event_entity_mask"],
        attack_ids=values["event_attack_ids"],
        attack_id_mask=values["event_attack_id_mask"],
        values=values["event_values"],
        value_mask=values["event_value_mask"],
        categorical_values=values["event_categorical_values"],
        overflow_offsets=values["event_overflow_offsets"],
        overflow_event_types=values["event_overflow_types"],
        overflow_actor_roles=values["event_overflow_actor_roles"],
        overflow_counts=values["event_overflow_counts"],
    )


def _accepted_action(
    values: Mapping[str, npt.NDArray[np.generic]],
    row: int,
    start: int,
    stop: int,
) -> AcceptedActionRecord:
    """Rebuild one stable accepted complete action without transient indices."""
    return AcceptedActionRecord(
        schema_version=ACCEPTED_ACTION_SCHEMA_VERSION,
        stable_identity=str(values["accepted_action_stable_ids"][row]),
        prompt_context=int(values["accepted_action_prompt_contexts"][row]),
        option_types=tuple(
            int(value)
            for value in values["accepted_action_option_types"][start:stop]
        ),
        option_contexts=tuple(
            int(value)
            for value in values["accepted_action_option_contexts"][start:stop]
        ),
        card_ids=tuple(
            int(value)
            for value in values["accepted_action_card_ids"][start:stop]
        ),
        attack_ids=tuple(
            int(value)
            for value in values["accepted_action_attack_ids"][start:stop]
        ),
        option_scalars=tuple(
            tuple(float(value) for value in item)
            for item in values["accepted_action_option_scalars"][start:stop]
        ),
        entity_card_ids=tuple(
            tuple(int(value) for value in item)
            for item in values["accepted_action_entity_card_ids"][start:stop]
        ),
        entity_areas=tuple(
            tuple(int(value) for value in item)
            for item in values["accepted_action_entity_areas"][start:stop]
        ),
        entity_owner_roles=tuple(
            tuple(int(value) for value in item)
            for item in values[
                "accepted_action_entity_owner_roles"
            ][start:stop]
        ),
        entity_token_kinds=tuple(
            tuple(int(value) for value in item)
            for item in values[
                "accepted_action_entity_token_kinds"
            ][start:stop]
        ),
        entity_scalars=tuple(
            tuple(float(value) for value in item.reshape(-1))
            for item in values["accepted_action_entity_scalars"][start:stop]
        ),
        ordered=bool(values["accepted_action_ordered"][row]),
        stop_sampled=bool(values["stop_sampled"][row]),
        accepted=True,
        fallback=bool(values["accepted_action_fallback"][row]),
    )


def _fragment_gae(
    fragment: StatelessFragment,
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    values = np.asarray(
        [decision.root_value for decision in fragment.decisions],
        dtype=np.float64,
    )
    rewards = np.asarray(
        [decision.reward for decision in fragment.decisions],
        dtype=np.float64,
    )
    next_values = np.empty_like(values)
    if len(values) > 1:
        next_values[:-1] = values[1:]
    next_values[-1] = fragment.bootstrap_value
    deltas = rewards + gamma * next_values - values
    advantages = np.empty_like(deltas)
    running = 0.0
    for index in range(len(deltas) - 1, -1, -1):
        running = float(deltas[index]) + gamma * gae_lambda * running
        advantages[index] = running
    returns = advantages + values
    return (
        tuple(float(value) for value in advantages),
        tuple(float(value) for value in returns),
    )


def _normalized_token_credit(
    prefix_values: tuple[float, ...],
    *,
    downstream_return: float,
    root_advantage: float,
    normalization_mean: float,
    normalization_scale: float,
) -> tuple[float, ...]:
    credit = compute_prompt_token_credit(
        prefix_values,
        downstream_return=downstream_return,
        gamma=1.0,
        gae_lambda=0.0,
    )
    raw = list(credit.advantages)
    raw[0] += root_advantage - sum(raw)
    if math.isinf(normalization_scale):
        return (0.0,) * len(raw)
    normalized = [value / normalization_scale for value in raw]
    normalized[0] -= normalization_mean / normalization_scale
    return tuple(normalized)


def _deck_macro_weights(
    targets: tuple[StatelessPpoTarget, ...],
    *,
    belief_only: bool,
    target_shares: Mapping[str, float] | None = None,
) -> tuple[float, ...]:
    eligible = tuple(
        index
        for index, target in enumerate(targets)
        if not belief_only or target.belief_target_valid
    )
    shares = normalized_present_deck_shares(
        (targets[index].deck_digest for index in eligible),
        target_shares,
    )
    if not shares:
        return (0.0,) * len(targets)
    counts = Counter(targets[index].deck_digest for index in eligible)
    weights = [0.0] * len(targets)
    for index in eligible:
        deck = targets[index].deck_digest
        weights[index] = shares[deck] / float(counts[deck])
    return tuple(weights)


def _row_bounds(
    offsets: npt.NDArray[np.generic],
    row: int,
) -> tuple[int, int]:
    return (int(offsets[row]), int(offsets[row + 1]))


def _integers(values: npt.NDArray[np.generic]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


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
    "StatelessOptimizerWindow",
    "StatelessPpoTarget",
    "prepare_stateless_optimizer_window",
    "reconstruct_stateless_fragment_part",
]
