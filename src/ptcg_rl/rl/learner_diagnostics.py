"""Bounded diagnostic summaries for retained learner samples."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class LearnerDiagnosticRow:
    """Small, model-independent view of one retained decision."""

    game_id: str
    seat: int
    policy_version: int
    action_count: int
    max_count: int
    prompt_context: int
    token_count: int
    stop_sampled: bool
    deck_signature: str
    training_advantage_abs: float
    training_metadata: Mapping[str, str]


@dataclass(frozen=True)
class LearnerDiagnostics:
    """Prompt, policy-age, and natural curriculum-mass diagnostics."""

    retained_seat_trajectories: int
    active_tokens: int
    voluntary_stop_tokens: int
    forced_max_decisions: int
    min_policy_version: int | None
    max_policy_version: int | None
    mixed_policy_games: int
    max_game_policy_version_span: int
    policy_age_histogram: Mapping[str, int]
    selection_count_histogram: Mapping[str, int]
    token_count_histogram: Mapping[str, int]
    prompt_context_histogram: Mapping[str, int]
    curriculum_mass: Mapping[str, Mapping[str, Mapping[str, int]]]
    deck_mass: Mapping[str, Mapping[str, int | float]]


def summarize_learner_diagnostics(
    rows: Sequence[LearnerDiagnosticRow],
    *,
    current_policy_version: int,
) -> LearnerDiagnostics:
    """Summarize retained rows without changing sampling or curriculum state."""
    selection_counts: Counter[str] = Counter()
    token_counts: Counter[str] = Counter()
    contexts: Counter[str] = Counter()
    policy_ages: Counter[str] = Counter()
    policy_versions: list[int] = []
    game_policy_versions: defaultdict[str, set[int]] = defaultdict(set)
    seat_trajectories: set[tuple[str, int]] = set()
    active_tokens = 0
    voluntary_stops = 0
    forced_max_decisions = 0
    curriculum_mass: defaultdict[str, dict[str, Counter[str]]] = defaultdict(dict)
    curriculum_seat_chains: defaultdict[
        tuple[str, str], set[tuple[str, int]]
    ] = defaultdict(set)
    deck_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    deck_advantage_abs: defaultdict[str, float] = defaultdict(float)
    deck_games: defaultdict[str, set[str]] = defaultdict(set)
    deck_seat_chains: defaultdict[str, set[tuple[str, int]]] = defaultdict(set)

    for row in rows:
        selection_counts[str(row.action_count)] += 1
        contexts[str(row.prompt_context)] += 1
        policy_versions.append(row.policy_version)
        policy_ages[str(current_policy_version - row.policy_version)] += 1
        game_policy_versions[row.game_id].add(row.policy_version)
        seat_trajectories.add((row.game_id, row.seat))
        deck_counts[row.deck_signature]["decisions"] += 1
        deck_counts[row.deck_signature]["active_tokens"] += row.token_count
        deck_advantage_abs[row.deck_signature] += row.training_advantage_abs
        deck_games[row.deck_signature].add(row.game_id)
        deck_seat_chains[row.deck_signature].add((row.game_id, row.seat))
        if row.action_count >= row.max_count:
            forced_max_decisions += 1
        if row.token_count > 0:
            token_counts[str(row.token_count)] += 1
            active_tokens += row.token_count
            voluntary_stops += int(row.stop_sampled)
        for dimension, label in row.training_metadata.items():
            label_mass = curriculum_mass[dimension].setdefault(label, Counter())
            label_mass["decisions"] += 1
            label_mass["active_tokens"] += row.token_count
            curriculum_seat_chains[(dimension, label)].add((row.game_id, row.seat))

    game_version_spans = tuple(
        max(versions) - min(versions)
        for versions in game_policy_versions.values()
        if versions
    )
    return LearnerDiagnostics(
        retained_seat_trajectories=len(seat_trajectories),
        active_tokens=active_tokens,
        voluntary_stop_tokens=voluntary_stops,
        forced_max_decisions=forced_max_decisions,
        min_policy_version=min(policy_versions) if policy_versions else None,
        max_policy_version=max(policy_versions) if policy_versions else None,
        mixed_policy_games=sum(span > 0 for span in game_version_spans),
        max_game_policy_version_span=max(game_version_spans, default=0),
        policy_age_histogram=dict(sorted(policy_ages.items())),
        selection_count_histogram=dict(sorted(selection_counts.items())),
        token_count_histogram=dict(sorted(token_counts.items())),
        prompt_context_histogram=dict(sorted(contexts.items())),
        curriculum_mass={
            dimension: {
                label: {
                    "retained_seat_trajectories": len(
                        curriculum_seat_chains[(dimension, label)]
                    ),
                    "decisions": int(counts["decisions"]),
                    "active_tokens": int(counts["active_tokens"]),
                }
                for label, counts in sorted(labels.items())
            }
            for dimension, labels in sorted(curriculum_mass.items())
        },
        deck_mass={
            signature: {
                "games": len(deck_games[signature]),
                "retained_seat_trajectories": len(deck_seat_chains[signature]),
                "decisions": int(counts["decisions"]),
                "active_tokens": int(counts["active_tokens"]),
                "sum_abs_training_advantage": deck_advantage_abs[signature],
            }
            for signature, counts in sorted(deck_counts.items())
        },
    )
