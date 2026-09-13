"""Identity-free candidate source streams and constructor telemetry."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ptcg_rl.actions.selection import random_legal_action
from ptcg_rl.agent.search.candidate_budget import CandidateBudgetPlan
from ptcg_rl.agent.search.candidates import (
    CandidateConstructionResult,
    CandidateExpansionInputs,
    CandidateSourceInputs,
    MultiSourceCandidateConstructor,
)
from ptcg_rl.agent.search.mutations import legal_local_mutations

Action = tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CandidateSourceTelemetry:
    """Compact source, quota, refill, dedup, and provenance diagnostics."""

    offered_by_source: Mapping[str, int]
    configured_seed_quotas: Mapping[str, int]
    used_seed_quotas: Mapping[str, int]
    configured_expansion_quotas: Mapping[str, int]
    used_expansion_quotas: Mapping[str, int]
    refill_slots: int
    duplicate_offers: int
    multi_source_candidates: int
    provenance_counts: Mapping[str, int]


def build_seed_source_inputs(
    actions: Sequence[Action],
    priors: Mapping[Action, float],
    *,
    seed: int,
    stream_limit: int,
) -> CandidateSourceInputs:
    """Build generic ranked source streams without card/deck identity rules."""
    if stream_limit <= 0:
        raise ValueError("candidate source stream limit must be positive")
    canonical = tuple(dict.fromkeys(tuple(action) for action in actions))
    if not canonical:
        raise ValueError("candidate source audit requires legal actions")
    if set(canonical) != set(priors):
        raise ValueError("candidate priors must cover the exhaustive action set")
    if any(
        not math.isfinite(float(priors[action])) or float(priors[action]) < 0.0
        for action in canonical
    ):
        raise ValueError("candidate priors must be finite and non-negative")
    if math.fsum(float(priors[action]) for action in canonical) <= 0.0:
        raise ValueError("candidate priors must carry positive mass")
    ranked = tuple(
        sorted(canonical, key=lambda action: (-float(priors[action]), action))
    )
    base = ranked[:stream_limit]
    cardinality = _cardinality_round_robin(ranked, priors)[:stream_limit]
    structural = _structural_order(canonical)[:stream_limit]
    stochastic = _gumbel_order(canonical, priors, seed=seed)[:stream_limit]
    return CandidateSourceInputs(
        base=base,
        # The immutable migration checkpoint has a zero proposal residual.
        # Its proposal ordering is therefore exactly the shared base ordering.
        proposal=base,
        cardinality=cardinality,
        stochastic=stochastic,
        structural=structural,
        # ``MultiSourceCandidateConstructor`` owns the bounded random stream.
        # Leaving this empty avoids feeding the offline exhaustive listing back
        # into the deployed constructor path.
        random=(),
    )


def build_engine_guided_expansion_inputs(
    select: Any,
    seed_result: CandidateConstructionResult,
    *,
    robust_scores: Mapping[Action, float],
    ordered: bool,
) -> CandidateExpansionInputs:
    """Expand scored seed parents through generic legal local mutations."""
    if not seed_result.valid:
        return CandidateExpansionInputs()
    retained = set(seed_result.candidates.actions)
    if set(robust_scores) != retained:
        raise ValueError(
            "expansion scores must cover exactly the retained seed support"
        )
    parents = sorted(
        seed_result.candidates.actions,
        key=lambda action: (-float(robust_scores[action]), action),
    )
    mutations: list[Action] = []
    seen = set(seed_result.candidates.actions)
    for parent in parents:
        for mutation in legal_local_mutations(select, parent, ordered=ordered):
            if mutation.action in seen:
                continue
            seen.add(mutation.action)
            mutations.append(mutation.action)
    return CandidateExpansionInputs(mutation=tuple(mutations))


def construct_with_expansion(
    constructor: MultiSourceCandidateConstructor,
    select: Any,
    *,
    greedy_action: Action,
    budget: CandidateBudgetPlan,
    seed_inputs: CandidateSourceInputs,
    expansion_inputs: CandidateExpansionInputs,
    ordered: bool,
    stochastic_seed: int,
) -> CandidateConstructionResult:
    """Run the same constructor contract used by training and serving."""
    return constructor.construct(
        select,
        greedy_action=greedy_action,
        budget=budget,
        seed_inputs=seed_inputs,
        expansion_inputs=expansion_inputs,
        ordered=ordered,
        stochastic_seed=stochastic_seed,
    )


def source_telemetry(
    *,
    source_inputs: CandidateSourceInputs,
    expansion_inputs: CandidateExpansionInputs,
    budget: CandidateBudgetPlan,
    result: CandidateConstructionResult,
    select: Any,
    ordered: bool,
    stochastic_seed: int,
) -> CandidateSourceTelemetry:
    """Derive diagnostic quota/refill/dedup facts from immutable inputs."""
    offered_streams: dict[str, tuple[Action, ...]] = {}
    for source, actions in source_inputs.as_dict().items():
        offered_streams[str(source)] = actions
    for expansion_source, actions in expansion_inputs.as_dict().items():
        offered_streams[str(expansion_source)] = actions
    internal_random = (
        ()
        if result.candidates.exhaustive
        else _constructor_random_actions(
            select,
            ordered=ordered,
            seed=stochastic_seed,
            attempts=max(16, budget.k_seed * 16),
        )
    )
    offered_streams["random"] = (
        *offered_streams.get("random", ()),
        *internal_random,
    )
    offered_by_source = {
        source: len(tuple(dict.fromkeys(actions)))
        for source, actions in offered_streams.items()
    }
    anchor_count = 0 if result.candidates.exhaustive else 1
    total_offers = (
        sum(len(actions) for actions in offered_streams.values()) + anchor_count
    )
    unique_offers = len(
        {action for actions in offered_streams.values() for action in actions}
        | (
            {result.candidates.actions[0]}
            if anchor_count and result.candidates.actions
            else set()
        )
    )
    used_seed = dict(result.seed_source_usage)
    used_expansion = dict(result.expansion_source_usage)
    configured_seed: dict[str, int] = {
        str(source): count for source, count in budget.seed_quotas.as_dict().items()
    }
    configured_expansion: dict[str, int] = {
        str(source): count
        for source, count in budget.expansion_quotas.as_dict().items()
    }
    refill = sum(
        max(0, used_seed.get(source, 0) - configured_seed[source])
        for source in configured_seed
    ) + sum(
        max(0, used_expansion.get(source, 0) - configured_expansion[source])
        for source in configured_expansion
    )
    provenance: Counter[str] = Counter()
    for labels in result.candidates.sources:
        for label in labels:
            provenance[label] += 1
    return CandidateSourceTelemetry(
        offered_by_source=dict(sorted(offered_by_source.items())),
        configured_seed_quotas=configured_seed,
        used_seed_quotas=used_seed,
        configured_expansion_quotas=configured_expansion,
        used_expansion_quotas=used_expansion,
        refill_slots=refill,
        duplicate_offers=max(0, total_offers - unique_offers),
        multi_source_candidates=sum(
            len(labels) > 1 for labels in result.candidates.sources
        ),
        provenance_counts=dict(sorted(provenance.items())),
    )


def deterministic_root_seed(case_id: str, campaign_seed: str) -> int:
    """Return a stable stochastic-source seed independent of Python hashing."""
    digest = hashlib.sha256(
        b"ptcg-rl/candidate-regret/source-seed/v1\x00"
        + campaign_seed.encode("utf-8")
        + bytes.fromhex(case_id)
    ).digest()
    return int.from_bytes(digest[:8], "big")


def _cardinality_round_robin(
    actions: Sequence[Action],
    priors: Mapping[Action, float],
) -> tuple[Action, ...]:
    groups: dict[int, list[Action]] = defaultdict(list)
    for action in actions:
        groups[len(action)].append(action)
    for group in groups.values():
        group.sort(key=lambda action: (-float(priors[action]), action))
    output: list[Action] = []
    depth = 0
    while True:
        emitted = False
        for cardinality in sorted(groups):
            group = groups[cardinality]
            if depth < len(group):
                output.append(group[depth])
                emitted = True
        if not emitted:
            return tuple(output)
        depth += 1


def _structural_order(actions: Sequence[Action]) -> tuple[Action, ...]:
    groups: dict[int, list[Action]] = defaultdict(list)
    for action in actions:
        groups[len(action)].append(action)
    output: list[Action] = []
    for cardinality in sorted(groups):
        group = sorted(groups[cardinality])
        output.append(group[0])
        if len(group) > 1:
            output.append(group[-1])
        output.extend(group[1:-1])
    return tuple(dict.fromkeys(output))


def _gumbel_order(
    actions: Sequence[Action],
    priors: Mapping[Action, float],
    *,
    seed: int,
) -> tuple[Action, ...]:
    rng = random.Random(seed)
    ranked: list[tuple[float, Action]] = []
    for action in actions:
        probability = max(float(priors[action]), 1.0e-30)
        uniform = min(max(rng.random(), 1.0e-12), 1.0 - 1.0e-12)
        gumbel = -math.log(-math.log(uniform))
        ranked.append((math.log(probability) + gumbel, action))
    return tuple(
        action for _, action in sorted(ranked, key=lambda item: (-item[0], item[1]))
    )


def _constructor_random_actions(
    select: Any,
    *,
    ordered: bool,
    seed: int,
    attempts: int,
) -> tuple[Action, ...]:
    """Replay the constructor-owned bounded random source for telemetry."""
    rng = random.Random(seed)
    output: list[Action] = []
    for _ in range(attempts):
        action = list(random_legal_action(select, rng=rng))
        if ordered and len(action) > 1:
            rng.shuffle(action)
        normalized = tuple(int(index) for index in action)
        output.append(normalized if ordered else tuple(sorted(normalized)))
    return tuple(output)


__all__ = [
    "Action",
    "CandidateSourceTelemetry",
    "build_engine_guided_expansion_inputs",
    "build_seed_source_inputs",
    "construct_with_expansion",
    "deterministic_root_seed",
    "source_telemetry",
]
