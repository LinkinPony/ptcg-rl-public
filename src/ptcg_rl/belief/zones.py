"""Hidden-zone split and validation helpers for belief determinizations."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence

from ptcg_rl.belief.card_rules import (
    ACE_SPEC_LIMIT,
    NON_BASIC_ENERGY_COPY_LIMIT,
    STANDARD_DECK_SIZE,
    CardCatalog,
)
from ptcg_rl.belief.observation import ObservationEvidence
from ptcg_rl.engine.session import HiddenInformation


def sample_your_hidden_zones(
    evidence: ObservationEvidence,
    *,
    your_deck: Sequence[int],
    rng: random.Random,
    strict_counts: bool,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split known own decklist into Search API ``your_deck`` and prize inputs."""
    available = Counter(int(card_id) for card_id in your_deck)
    _subtract_in_place(
        available,
        evidence.your_non_deck_visible_counts,
        strict=strict_counts,
    )
    if evidence.search_ignores_your_deck:
        _subtract_in_place(
            available,
            evidence.your_visible_deck_counts,
            strict=strict_counts,
        )
        return (), tuple(
            _draw_from_counts(available, evidence.your_prize_count, rng=rng)
        )

    needed = evidence.your_deck_count + evidence.your_prize_count
    drawn = _draw_from_counts(available, needed, rng=rng)
    your_prize = tuple(drawn[: evidence.your_prize_count])
    your_hidden_deck = tuple(drawn[evidence.your_prize_count :])
    return your_hidden_deck, your_prize


def complete_rule_consistent_deck_counts(
    known_counts: Counter[int],
    *,
    catalog: CardCatalog,
    rng: random.Random,
    deck_size: int = STANDARD_DECK_SIZE,
) -> Counter[int]:
    """Randomly complete known opponent cards to a legal 60-card deck."""
    counts = Counter(
        {card_id: count for card_id, count in known_counts.items() if count > 0}
    )
    total = sum(counts.values())
    if total > deck_size:
        raise ValueError("known opponent card counts exceed a full deck")
    if not _counts_have_basic_pokemon(counts, catalog):
        _add_random_legal(
            counts,
            catalog.basic_pokemon_ids or (catalog.default_basic_pokemon_id,),
            catalog=catalog,
            rng=rng,
        )
        total += 1
    name_counts = _deck_name_counts(counts, catalog)
    ace_count = _deck_ace_spec_count(counts, catalog)
    while total < deck_size:
        card_id = _sample_completion_candidate(
            catalog,
            name_counts=name_counts,
            ace_count=ace_count,
            rng=rng,
        )
        counts[card_id] += 1
        total += 1
        rule = catalog.rule(card_id)
        name_counts[rule.name] += 1
        if rule.is_ace_spec:
            ace_count += 1
    catalog.validate_deck_counts(counts, deck_size=deck_size)
    return counts


_COMPLETION_REJECTION_ATTEMPTS = 64


def _sample_completion_candidate(
    catalog: CardCatalog,
    *,
    name_counts: Mapping[str, int],
    ace_count: int,
    rng: random.Random,
) -> int:
    """Sample one legal completion card uniformly over legal candidates.

    Rejection sampling over the full catalog draws from exactly the same
    uniform distribution as enumerating ``_legal_completion_candidates`` and
    choosing one, because every attempt is uniform over the catalog and
    acceptance keeps only legal candidates. Copy-limited names exclude only a
    tiny fraction of the catalog, so acceptance is nearly certain and the
    per-card cost stays O(1) instead of one full catalog scan per added card.
    """
    card_ids = catalog.card_ids
    if card_ids:
        for _attempt in range(_COMPLETION_REJECTION_ATTEMPTS):
            card_id = card_ids[rng.randrange(len(card_ids))]
            rule = catalog.rule(card_id)
            if rule.is_basic_energy:
                return card_id
            if rule.is_ace_spec and ace_count >= ACE_SPEC_LIMIT:
                continue
            if name_counts.get(rule.name, 0) < NON_BASIC_ENERGY_COPY_LIMIT:
                return card_id
    candidates = _legal_completion_candidates(
        catalog,
        name_counts=name_counts,
        ace_count=ace_count,
    )
    if not candidates:
        return catalog.default_basic_energy_id
    return rng.choice(candidates)


def sample_opponent_hidden_zones(
    evidence: ObservationEvidence,
    *,
    opponent_deck_counts: Counter[int],
    catalog: CardCatalog,
    rng: random.Random,
    opponent_hand_weights: Mapping[int, float] | Sequence[float] | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split an opponent deck composition into hidden zones."""
    available = Counter(opponent_deck_counts)
    _subtract_in_place(
        available,
        evidence.opponent_current_visible_counts,
        strict=False,
    )
    opponent_active = _draw_facedown_active(evidence, available, catalog, rng)
    if opponent_hand_weights is not None:
        return _sample_weighted_opponent_hidden_zones(
            evidence,
            available=available,
            catalog=catalog,
            rng=rng,
            opponent_active=opponent_active,
            opponent_hand_weights=opponent_hand_weights,
        )
    opponent_deck = _draw_opponent_deck_zone(evidence, available, catalog, rng)
    opponent_prize = tuple(
        _draw_from_counts(available, evidence.opponent_prize_count, rng=rng)
    )
    opponent_hand = tuple(
        _draw_from_counts(available, evidence.opponent_hand_count, rng=rng)
    )
    return opponent_deck, opponent_prize, opponent_hand, opponent_active


def _sample_weighted_opponent_hidden_zones(
    evidence: ObservationEvidence,
    *,
    available: Counter[int],
    catalog: CardCatalog,
    rng: random.Random,
    opponent_active: tuple[int, ...],
    opponent_hand_weights: Mapping[int, float] | Sequence[float],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    required_deck: list[int] = []
    remaining_deck_count = evidence.opponent_deck_count
    if evidence.setup_requires_basic_in_opponent_deck and remaining_deck_count > 0:
        basic_candidates = _cards_matching(available, catalog.is_basic_pokemon)
        if basic_candidates:
            card_id = rng.choice(basic_candidates)
            _remove_one(available, card_id)
            required_deck.append(card_id)
        else:
            required_deck.append(catalog.default_basic_pokemon_id)
        remaining_deck_count -= 1

    opponent_hand = tuple(
        _draw_weighted_from_counts(
            available,
            evidence.opponent_hand_count,
            weights=opponent_hand_weights,
            rng=rng,
        )
    )
    opponent_prize = tuple(
        _draw_from_counts(available, evidence.opponent_prize_count, rng=rng)
    )
    opponent_deck_cards = required_deck + _draw_from_counts(
        available,
        remaining_deck_count,
        rng=rng,
    )
    rng.shuffle(opponent_deck_cards)
    return tuple(opponent_deck_cards), opponent_prize, opponent_hand, opponent_active


def validate_hidden_information(
    evidence: ObservationEvidence,
    hidden: HiddenInformation,
    catalog: CardCatalog | None = None,
) -> None:
    """Validate Search API hidden-zone lengths and setup-specific constraints."""
    expected_your_deck_count = (
        0 if evidence.search_ignores_your_deck else evidence.your_deck_count
    )
    _require_length("your_deck", hidden.your_deck, expected_your_deck_count)
    _require_length("your_prize", hidden.your_prize, evidence.your_prize_count)
    _require_length("opponent_deck", hidden.opponent_deck, evidence.opponent_deck_count)
    _require_length(
        "opponent_prize",
        hidden.opponent_prize,
        evidence.opponent_prize_count,
    )
    _require_length("opponent_hand", hidden.opponent_hand, evidence.opponent_hand_count)
    expected_active = 1 if evidence.opponent_active_facedown else 0
    _require_length("opponent_active", hidden.opponent_active, expected_active)

    for zone_name, zone in _hidden_zones(hidden).items():
        if any(card_id <= 0 for card_id in zone):
            raise ValueError(f"{zone_name} contains a non-positive card ID")

    if catalog is None:
        return
    if hidden.opponent_active and not catalog.is_pokemon(hidden.opponent_active[0]):
        raise ValueError("opponent_active must be a Pokemon card")
    if (
        evidence.setup_requires_basic_in_opponent_deck
        and hidden.opponent_deck
        and not catalog.has_basic_pokemon(hidden.opponent_deck)
    ):
        raise ValueError("opponent_deck must contain a Basic Pokemon during setup")


def _draw_facedown_active(
    evidence: ObservationEvidence,
    available: Counter[int],
    catalog: CardCatalog,
    rng: random.Random,
) -> tuple[int, ...]:
    if not evidence.opponent_active_facedown:
        return ()
    predicate = (
        catalog.is_basic_pokemon
        if evidence.setup_requires_basic_in_opponent_deck
        else catalog.is_pokemon
    )
    candidates = _cards_matching(available, predicate)
    if candidates:
        card_id = rng.choice(candidates)
        _remove_one(available, card_id)
        return (card_id,)
    return (catalog.default_basic_pokemon_id,)


def _draw_opponent_deck_zone(
    evidence: ObservationEvidence,
    available: Counter[int],
    catalog: CardCatalog,
    rng: random.Random,
) -> tuple[int, ...]:
    if evidence.opponent_deck_count == 0:
        return ()
    required: list[int] = []
    if evidence.setup_requires_basic_in_opponent_deck:
        basic_candidates = _cards_matching(available, catalog.is_basic_pokemon)
        if basic_candidates:
            card_id = rng.choice(basic_candidates)
            _remove_one(available, card_id)
            required.append(card_id)
        else:
            required.append(catalog.default_basic_pokemon_id)
    remaining = evidence.opponent_deck_count - len(required)
    cards = required + _draw_from_counts(available, remaining, rng=rng)
    rng.shuffle(cards)
    return tuple(cards)


def _draw_from_counts(
    counts: Counter[int],
    count: int,
    *,
    rng: random.Random,
) -> list[int]:
    if count < 0:
        raise ValueError("cannot draw a negative number of cards")
    expanded = _expand_counts(counts)
    if len(expanded) < count:
        raise ValueError("not enough cards to draw requested hidden zone")
    drawn: list[int] = []
    while len(drawn) < count:
        draw_index = rng.randrange(len(expanded))
        card_id = expanded[draw_index]
        expanded[draw_index] = expanded[-1]
        expanded.pop()
        drawn.append(card_id)
    for card_id in drawn:
        _remove_one(counts, card_id)
    return drawn


def _draw_weighted_from_counts(
    counts: Counter[int],
    count: int,
    *,
    weights: Mapping[int, float] | Sequence[float],
    rng: random.Random,
) -> list[int]:
    if count < 0:
        raise ValueError("cannot draw a negative number of cards")
    drawn: list[int] = []
    while len(drawn) < count:
        expanded = _expand_counts(counts)
        if not expanded:
            raise ValueError("not enough cards to draw requested hidden zone")
        weighted = [
            max(0.0, _weight_for_card(weights, card_id)) for card_id in expanded
        ]
        total = sum(weighted)
        if total <= 0.0:
            card_id = rng.choice(expanded)
        else:
            threshold = rng.random() * total
            cumulative = 0.0
            card_id = expanded[-1]
            for candidate, weight in zip(expanded, weighted, strict=True):
                cumulative += weight
                if threshold <= cumulative:
                    card_id = candidate
                    break
        _remove_one(counts, card_id)
        drawn.append(card_id)
    return drawn


def _weight_for_card(
    weights: Mapping[int, float] | Sequence[float], card_id: int
) -> float:
    if isinstance(weights, Mapping):
        return float(weights.get(int(card_id), 0.0))
    index = int(card_id) - 1
    if index < 0 or index >= len(weights):
        return 0.0
    return float(weights[index])


def _add_random_legal(
    counts: Counter[int],
    candidates: Sequence[int],
    *,
    catalog: CardCatalog,
    rng: random.Random,
) -> None:
    legal = [
        card_id for card_id in candidates if catalog.can_add_to_deck(counts, card_id)
    ]
    if not legal:
        raise ValueError("no legal candidate can satisfy Basic Pokemon rule")
    counts[rng.choice(legal)] += 1


def _counts_have_basic_pokemon(
    counts: Mapping[int, int],
    catalog: CardCatalog,
) -> bool:
    return any(
        count > 0 and catalog.is_basic_pokemon(card_id)
        for card_id, count in counts.items()
    )


def _deck_name_counts(
    counts: Mapping[int, int],
    catalog: CardCatalog,
) -> Counter[str]:
    result: Counter[str] = Counter()
    for card_id, count in counts.items():
        if count > 0:
            result[catalog.rule(card_id).name] += count
    return result


def _deck_ace_spec_count(
    counts: Mapping[int, int],
    catalog: CardCatalog,
) -> int:
    return sum(
        count
        for card_id, count in counts.items()
        if count > 0 and catalog.rule(card_id).is_ace_spec
    )


def _legal_completion_candidates(
    catalog: CardCatalog,
    *,
    name_counts: Mapping[str, int],
    ace_count: int,
) -> list[int]:
    candidates: list[int] = []
    for card_id in catalog.card_ids:
        rule = catalog.rule(card_id)
        if rule.is_basic_energy:
            candidates.append(card_id)
            continue
        if rule.is_ace_spec and ace_count >= ACE_SPEC_LIMIT:
            continue
        if name_counts.get(rule.name, 0) < NON_BASIC_ENERGY_COPY_LIMIT:
            candidates.append(card_id)
    return candidates


def _cards_matching(
    counts: Mapping[int, int],
    predicate: Callable[[int], bool],
) -> list[int]:
    return [
        card_id
        for card_id, count in counts.items()
        for _ in range(max(0, count))
        if predicate(card_id)
    ]


def _expand_counts(counts: Mapping[int, int]) -> list[int]:
    return [card_id for card_id, count in counts.items() for _ in range(max(0, count))]


def _subtract_in_place(
    counts: Counter[int],
    remove: Mapping[int, int],
    *,
    strict: bool,
) -> None:
    for card_id, amount in remove.items():
        if amount <= 0:
            continue
        if strict and counts[card_id] < amount:
            raise ValueError(f"visible card count exceeds deck count for {card_id}")
        counts[card_id] = max(0, counts[card_id] - amount)
        if counts[card_id] == 0:
            del counts[card_id]


def _remove_one(counts: Counter[int], card_id: int) -> None:
    if counts[card_id] <= 0:
        raise ValueError(f"card_id={card_id} is not available")
    counts[card_id] -= 1
    if counts[card_id] == 0:
        del counts[card_id]


def _require_length(zone_name: str, zone: Sequence[int], expected: int) -> None:
    if len(zone) != expected:
        raise ValueError(f"{zone_name} must contain {expected} cards, got {len(zone)}")


def _hidden_zones(hidden: HiddenInformation) -> dict[str, tuple[int, ...]]:
    return {
        "your_deck": hidden.your_deck,
        "your_prize": hidden.your_prize,
        "opponent_deck": hidden.opponent_deck,
        "opponent_prize": hidden.opponent_prize,
        "opponent_hand": hidden.opponent_hand,
        "opponent_active": hidden.opponent_active,
    }
