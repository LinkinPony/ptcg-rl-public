"""Known-density, card-agnostic complete-action candidate proposals."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations, permutations

from ptcg_rl.rl.amortized_policy_iteration.contracts import (
    CandidateProposalConfig,
    ProposalSource,
)


@dataclass(frozen=True, slots=True)
class CompleteActionSpace:
    """Exact cardinality and ordering contract for one legal prompt."""

    option_count: int
    min_count: int
    max_count: int
    ordered: bool

    def __post_init__(self) -> None:
        if self.option_count < 0:
            raise ValueError("option_count must be non-negative")
        if not 0 <= self.min_count <= self.max_count <= self.option_count:
            raise ValueError("complete-action count bounds are invalid")

    @property
    def legal_action_count(self) -> int:
        """Return the full legal sequence/set count without materializing it."""
        counter = math.perm if self.ordered else math.comb
        return sum(
            counter(self.option_count, count)
            for count in range(self.min_count, self.max_count + 1)
        )

    @property
    def cardinality_count(self) -> int:
        """Return the number of legal selection cardinalities."""
        return self.max_count - self.min_count + 1


@dataclass(frozen=True, slots=True)
class BehaviorCandidate:
    """One frozen-policy draw and its exact complete-action probability."""

    action: tuple[int, ...]
    probability: float


@dataclass(frozen=True, slots=True)
class ProposedCandidate:
    """One canonical retained action with its single-draw mixture density."""

    action: tuple[int, ...]
    q_prop: float
    behavior_probability: float
    sources: tuple[ProposalSource, ...]


@dataclass(frozen=True, slots=True)
class CandidateProposal:
    """Complete retained support and its exhaustive-versus-sampled marker."""

    candidates: tuple[ProposedCandidate, ...]
    legal_action_count: int
    exhaustive: bool
    probability_semantics: str = "single_draw_marginal"

    def __post_init__(self) -> None:
        if self.legal_action_count <= 0:
            raise ValueError("legal_action_count must be positive")
        if not self.candidates:
            raise ValueError("candidate proposal must retain at least one action")
        actions = tuple(candidate.action for candidate in self.candidates)
        if len(set(actions)) != len(actions):
            raise ValueError("candidate proposal contains duplicate actions")
        if self.exhaustive and len(actions) != self.legal_action_count:
            raise ValueError("exhaustive proposal does not cover the legal space")
        if not self.exhaustive and len(actions) >= self.legal_action_count:
            raise ValueError("sampled proposal must not claim full legal coverage")
        if any(
            not math.isfinite(candidate.q_prop)
            or candidate.q_prop <= 0.0
            or candidate.q_prop > 1.0
            for candidate in self.candidates
        ):
            raise ValueError("every retained candidate needs q_prop in (0, 1]")


def build_candidate_proposal(
    space: CompleteActionSpace,
    *,
    behavior_candidates: Sequence[BehaviorCandidate],
    behavior_probability: Callable[[tuple[int, ...]], float],
    config: CandidateProposalConfig,
    rng: random.Random,
) -> CandidateProposal:
    """Construct exhaustive support or a deduplicated mixture sample.

    For sampled support, ``q_prop`` is the exact marginal probability of one
    draw from the declared mixture. Sampling components may collide; retained
    actions are deduplicated by engine-equivalent complete-action identity.
    """
    legal_count = space.legal_action_count
    if legal_count <= 0:
        raise ValueError("complete-action space is empty")
    if legal_count <= config.exhaustive_action_cap:
        actions = enumerate_complete_actions(space)
        uniform = 1.0 / float(legal_count)
        return CandidateProposal(
            candidates=tuple(
                ProposedCandidate(
                    action=action,
                    q_prop=uniform,
                    behavior_probability=_behavior_probability(
                        behavior_probability,
                        action,
                    ),
                    sources=("exhaustive",),
                )
                for action in actions
            ),
            legal_action_count=legal_count,
            exhaustive=True,
        )

    retained: dict[tuple[int, ...], list[ProposalSource]] = {}

    def add(action: Sequence[int], source: ProposalSource) -> None:
        canonical = canonical_complete_action(space, action)
        _validate_legal_action(space, canonical)
        if canonical not in retained and len(retained) >= config.max_candidates:
            return
        labels = retained.setdefault(canonical, [])
        if source not in labels:
            labels.append(source)

    behavior_retained = 0
    for candidate in behavior_candidates:
        expected = _behavior_probability(behavior_probability, candidate.action)
        if not math.isclose(
            expected,
            float(candidate.probability),
            rel_tol=1.0e-5,
            abs_tol=1.0e-8,
        ):
            raise ValueError("frozen behavior candidate probability differs")
        before = len(retained)
        add(candidate.action, "behavior")
        behavior_retained += int(len(retained) > before)
        if behavior_retained >= config.behavior_samples:
            break

    _sample_component_without_replacement(
        space,
        requested=config.structural_samples,
        draw=lambda: sample_structural_action(space, rng=rng),
        source="structural",
        add=add,
        retained=retained,
        max_candidates=config.max_candidates,
    )
    _sample_component_without_replacement(
        space,
        requested=config.exploration_samples,
        draw=lambda: action_from_rank(
            space,
            rng.randrange(space.legal_action_count),
        ),
        source="exploration",
        add=add,
        retained=retained,
        max_candidates=config.max_candidates,
    )
    if not retained:
        add(action_from_rank(space, rng.randrange(legal_count)), "exploration")

    candidates = tuple(
        ProposedCandidate(
            action=action,
            q_prop=mixture_proposal_probability(
                space,
                action,
                behavior_probability=_behavior_probability(
                    behavior_probability,
                    action,
                ),
                config=config,
            ),
            behavior_probability=_behavior_probability(
                behavior_probability,
                action,
            ),
            sources=tuple(labels),
        )
        for action, labels in retained.items()
    )
    return CandidateProposal(
        candidates=candidates,
        legal_action_count=legal_count,
        exhaustive=False,
    )


def mixture_proposal_probability(
    space: CompleteActionSpace,
    action: Sequence[int],
    *,
    behavior_probability: float,
    config: CandidateProposalConfig,
) -> float:
    """Return the exact declared one-draw mixture density for an action."""
    canonical = canonical_complete_action(space, action)
    _validate_legal_action(space, canonical)
    if (
        not math.isfinite(behavior_probability)
        or behavior_probability < 0.0
        or behavior_probability > 1.0
    ):
        raise ValueError("behavior probability must be in [0, 1]")
    structural = structural_action_probability(space, canonical)
    exploration = 1.0 / float(space.legal_action_count)
    probability = (
        config.behavior_weight * behavior_probability
        + config.structural_weight * structural
        + config.exploration_weight * exploration
    )
    if not math.isfinite(probability) or probability <= 0.0 or probability > 1.0:
        raise ValueError("mixture proposal probability is invalid")
    return probability


def reweight_candidate_proposal(
    proposal: CandidateProposal,
    space: CompleteActionSpace,
    *,
    behavior_probabilities: Sequence[float],
    config: CandidateProposalConfig,
) -> CandidateProposal:
    """Attach exact frozen-policy probabilities after one batched model pass."""
    if len(behavior_probabilities) != len(proposal.candidates):
        raise ValueError("behavior probabilities must align with retained candidates")
    if proposal.exhaustive:
        q_values = (1.0 / float(proposal.legal_action_count),) * len(
            proposal.candidates
        )
    else:
        q_values = tuple(
            mixture_proposal_probability(
                space,
                candidate.action,
                behavior_probability=float(probability),
                config=config,
            )
            for candidate, probability in zip(
                proposal.candidates,
                behavior_probabilities,
                strict=True,
            )
        )
    return CandidateProposal(
        candidates=tuple(
            ProposedCandidate(
                action=candidate.action,
                q_prop=q_prop,
                behavior_probability=float(probability),
                sources=candidate.sources,
            )
            for candidate, probability, q_prop in zip(
                proposal.candidates,
                behavior_probabilities,
                q_values,
                strict=True,
            )
        ),
        legal_action_count=proposal.legal_action_count,
        exhaustive=proposal.exhaustive,
        probability_semantics=proposal.probability_semantics,
    )


def structural_action_probability(
    space: CompleteActionSpace,
    action: Sequence[int],
) -> float:
    """Return count-uniform, within-count-uniform generic proposal density."""
    canonical = canonical_complete_action(space, action)
    _validate_legal_action(space, canonical)
    counter = math.perm if space.ordered else math.comb
    within_count = counter(space.option_count, len(canonical))
    return 1.0 / float(space.cardinality_count * within_count)


def sample_structural_action(
    space: CompleteActionSpace,
    *,
    rng: random.Random,
) -> tuple[int, ...]:
    """Sample count first, then a legal sequence/set uniformly at that count."""
    count = rng.randint(space.min_count, space.max_count)
    counter = math.perm if space.ordered else math.comb
    within_count = counter(space.option_count, count)
    preceding = sum(
        counter(space.option_count, previous)
        for previous in range(space.min_count, count)
    )
    return action_from_rank(space, preceding + rng.randrange(within_count))


def enumerate_complete_actions(
    space: CompleteActionSpace,
) -> tuple[tuple[int, ...], ...]:
    """Materialize a small complete legal action space."""
    factory = permutations if space.ordered else combinations
    return tuple(
        tuple(action)
        for count in range(space.min_count, space.max_count + 1)
        for action in factory(range(space.option_count), count)
    )


def action_from_rank(
    space: CompleteActionSpace,
    rank: int,
) -> tuple[int, ...]:
    """Unrank one complete action without enumerating the legal space."""
    if rank < 0 or rank >= space.legal_action_count:
        raise ValueError("action rank is outside the legal space")
    remaining = rank
    counter = math.perm if space.ordered else math.comb
    for count in range(space.min_count, space.max_count + 1):
        block_size = counter(space.option_count, count)
        if remaining < block_size:
            if space.ordered:
                return _unrank_permutation(space.option_count, count, remaining)
            return _unrank_combination(space.option_count, count, remaining)
        remaining -= block_size
    raise RuntimeError("complete-action unranking exhausted its legal space")


def canonical_complete_action(
    space: CompleteActionSpace,
    action: Sequence[int],
) -> tuple[int, ...]:
    """Canonicalize only action spaces whose engine semantics are unordered."""
    values = tuple(int(index) for index in action)
    return values if space.ordered else tuple(sorted(values))


def _sample_component_without_replacement(
    space: CompleteActionSpace,
    *,
    requested: int,
    draw: Callable[[], tuple[int, ...]],
    source: ProposalSource,
    add: Callable[[Sequence[int], ProposalSource], None],
    retained: dict[tuple[int, ...], list[ProposalSource]],
    max_candidates: int,
) -> None:
    target = min(max_candidates, len(retained) + requested)
    attempts = 0
    attempt_cap = max(32, requested * 16)
    while len(retained) < target and attempts < attempt_cap:
        add(draw(), source)
        attempts += 1
        if len(retained) >= space.legal_action_count:
            break


def _behavior_probability(
    probability: Callable[[tuple[int, ...]], float],
    action: Sequence[int],
) -> float:
    value = float(probability(tuple(int(index) for index in action)))
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("behavior probability must be in [0, 1]")
    return value


def _validate_legal_action(
    space: CompleteActionSpace,
    action: tuple[int, ...],
) -> None:
    if not space.min_count <= len(action) <= space.max_count:
        raise ValueError("action cardinality is outside the legal bounds")
    if len(set(action)) != len(action):
        raise ValueError("complete actions cannot repeat option indices")
    if any(index < 0 or index >= space.option_count for index in action):
        raise ValueError("complete action contains an invalid option index")
    if not space.ordered and action != tuple(sorted(action)):
        raise ValueError("unordered complete action is not canonical")


def _unrank_combination(
    option_count: int,
    count: int,
    rank: int,
) -> tuple[int, ...]:
    action: list[int] = []
    next_index = 0
    remaining = rank
    for remaining_slots in range(count, 0, -1):
        for index in range(next_index, option_count):
            suffixes = math.comb(option_count - index - 1, remaining_slots - 1)
            if remaining < suffixes:
                action.append(index)
                next_index = index + 1
                break
            remaining -= suffixes
    return tuple(action)


def _unrank_permutation(
    option_count: int,
    count: int,
    rank: int,
) -> tuple[int, ...]:
    available = list(range(option_count))
    action: list[int] = []
    remaining = rank
    for position in range(count):
        suffixes = math.perm(len(available) - 1, count - position - 1)
        choice, remaining = divmod(remaining, suffixes)
        action.append(available.pop(choice))
    return tuple(action)


__all__ = [
    "BehaviorCandidate",
    "CandidateProposal",
    "CompleteActionSpace",
    "ProposedCandidate",
    "action_from_rank",
    "build_candidate_proposal",
    "canonical_complete_action",
    "enumerate_complete_actions",
    "mixture_proposal_probability",
    "reweight_candidate_proposal",
    "sample_structural_action",
    "structural_action_probability",
]
