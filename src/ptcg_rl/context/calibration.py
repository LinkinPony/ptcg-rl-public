"""Calibration reports for opponent belief features."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.context import (
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
    context_features_from_observation,
)
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.decks import parse_canonical_signature
from ptcg_rl.training.bc_dataset import (
    KaggleStepDataConfig,
    observation_from_step_row,
    resolve_step_shards,
)


class BeliefCalibrationConfig(BaseModel):
    """Config for streaming belief calibration over compact step shards."""

    model_config = ConfigDict(extra="forbid")

    data: KaggleStepDataConfig = KaggleStepDataConfig()
    belief: OpponentBeliefFeatureConfig = OpponentBeliefFeatureConfig()
    checkpoint_path: Path | None = None
    max_rows: int | None = 100_000
    output_path: Path | None = Path("outputs/context/belief_calibration/report.json")

    @field_validator("max_rows")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional limits."""
        if value is not None and value <= 0:
            raise ValueError("max_rows must be positive when set")
        return value


def run_belief_calibration(config: BeliefCalibrationConfig) -> dict[str, Any]:
    """Stream rows and compare belief top-k against true replay decklists."""
    producer = OpponentBeliefFeatureProducer.from_config(config.belief)
    checkpoint_policy = _checkpoint_policy(config.checkpoint_path)
    rows = 0
    supervised_rows = 0
    non_empty_rows = 0
    topk_true_cards = 0
    topk_cards = 0
    true_card_mass_sum = 0.0
    absolute_error_sum = 0.0
    absolute_error_terms = 0
    hand_rows = 0
    uniform_hand_nll_sum = 0.0
    archetype_hand_nll_sum = 0.0
    archetype_hand_nll_rows = 0
    model_hand_nll_sum = 0.0
    model_hand_nll_rows = 0
    archetype_deck_top1_hits = 0
    archetype_deck_top5_hits = 0
    archetype_deck_rows = 0
    model_deck_top1_hits = 0
    model_deck_top5_hits = 0
    model_deck_rows = 0
    for shard_path in resolve_step_shards(config.data):
        if config.max_rows is not None and rows >= config.max_rows:
            break
        parquet_file = pq.ParquetFile(shard_path)
        for record_batch in parquet_file.iter_batches(batch_size=config.data.read_batch_size):
            for row in record_batch.to_pylist():
                if config.max_rows is not None and rows >= config.max_rows:
                    break
                if not isinstance(row, Mapping):
                    continue
                rows += 1
                deck_counts = Counter(
                    int(card_id)
                    for card_id in _sequence(row.get("opponent_deck_ids"))
                    if card_id is not None and int(card_id) > 0
                )
                if not deck_counts:
                    continue
                supervised_rows += 1
                observation = observation_from_step_row(row)
                context_features = producer.augment(
                    observation,
                    context_features_from_observation(observation),
                )
                if not context_features.opponent_belief_empty:
                    non_empty_rows += 1
                expected = {
                    item.card_id: item.expected_count
                    for item in context_features.opponent_belief
                }
                for card_id, expected_count in expected.items():
                    topk_cards += 1
                    if deck_counts[card_id] > 0:
                        topk_true_cards += 1
                    true_count = float(deck_counts[card_id])
                    true_card_mass_sum += true_count
                    absolute_error_sum += abs(float(expected_count) - true_count)
                    absolute_error_terms += 1

                known_counts = _opponent_known_counts(row)
                true_signature = str(row.get("opponent_deck_signature", ""))
                if producer.prior is not None and true_signature:
                    posterior = producer.prior.posterior(known_counts)
                    if not posterior.is_empty:
                        archetype_deck_rows += 1
                        top_signatures = [
                            entry.deck.signature for entry in posterior.entries[:5]
                        ]
                        if top_signatures[:1] == [true_signature]:
                            archetype_deck_top1_hits += 1
                        if true_signature in top_signatures:
                            archetype_deck_top5_hits += 1

                hand_ids = _godview_hand_ids(row)
                unseen_counts = _opponent_unseen_counts(row)
                if hand_ids:
                    hand_rows += 1
                    uniform_hand_nll_sum += _hand_nll(
                        hand_ids,
                        _counts_distribution(unseen_counts),
                    )
                    if producer.prior is not None:
                        archetype_distribution = _archetype_distribution(
                            producer.prior.posterior(known_counts),
                            known_counts,
                        )
                        if archetype_distribution:
                            archetype_hand_nll_rows += 1
                            archetype_hand_nll_sum += _hand_nll(
                                hand_ids,
                                archetype_distribution,
                            )

                if checkpoint_policy is None:
                    continue
                own_signature = str(row.get("deck_signature", ""))
                if own_signature:
                    checkpoint_policy.bind_own_deck(
                        parse_canonical_signature(own_signature).card_ids
                    )
                elif checkpoint_policy.deck_conditioning_enabled:
                    raise ValueError(
                        "conditioned calibration row is missing deck_signature"
                    )
                distributions = checkpoint_policy.belief_distributions(observation)
                if distributions is None:
                    continue
                card_probs, hand_probs = distributions
                if producer.prior is not None and true_signature:
                    posterior = producer.prior.model_posterior(known_counts, card_probs)
                    if not posterior.is_empty:
                        model_deck_rows += 1
                        top_signatures = [
                            entry.deck.signature for entry in posterior.entries[:5]
                        ]
                        if top_signatures[:1] == [true_signature]:
                            model_deck_top1_hits += 1
                        if true_signature in top_signatures:
                            model_deck_top5_hits += 1
                if hand_ids:
                    model_distribution = _masked_distribution(hand_probs, unseen_counts)
                    if model_distribution:
                        model_hand_nll_rows += 1
                        model_hand_nll_sum += _hand_nll(hand_ids, model_distribution)

    report = {
        "rows": rows,
        "supervised_rows": supervised_rows,
        "non_empty_rows": non_empty_rows,
        "empty_rate": 1.0 - _safe_rate(non_empty_rows, supervised_rows),
        "topk_true_card_rate": _safe_rate(topk_true_cards, topk_cards),
        "topk_true_card_mass_mean": _safe_rate(true_card_mass_sum, topk_cards),
        "expected_count_mae": _safe_rate(absolute_error_sum, absolute_error_terms),
        "hand_rows": hand_rows,
        "uniform_hand_nll": _safe_rate(uniform_hand_nll_sum, hand_rows),
        "archetype_hand_nll": _safe_rate(
            archetype_hand_nll_sum,
            archetype_hand_nll_rows,
        ),
        "archetype_hand_nll_rows": archetype_hand_nll_rows,
        "model_hand_nll": _safe_rate(model_hand_nll_sum, model_hand_nll_rows),
        "model_hand_nll_rows": model_hand_nll_rows,
        "archetype_deck_top1": _safe_rate(
            archetype_deck_top1_hits,
            archetype_deck_rows,
        ),
        "archetype_deck_top5": _safe_rate(
            archetype_deck_top5_hits,
            archetype_deck_rows,
        ),
        "archetype_deck_rows": archetype_deck_rows,
        "model_deck_top1": _safe_rate(model_deck_top1_hits, model_deck_rows),
        "model_deck_top5": _safe_rate(model_deck_top5_hits, model_deck_rows),
        "model_deck_rows": model_deck_rows,
    }
    if config.output_path is not None:
        output_path = deck_records.repo_path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def _safe_rate(numerator: float, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _checkpoint_policy(path: Path | None) -> Any | None:
    if path is None:
        return None
    from ptcg_rl.agent.runtime import CheckpointPolicy

    return CheckpointPolicy(deck_records.repo_path(path))


def _godview_hand_ids(row: Mapping[str, Any]) -> tuple[int, ...]:
    if not bool(row.get("godview_opp_hand_available", False)):
        return ()
    if not bool(row.get("godview_opp_hand_count_match", True)):
        return ()
    return tuple(
        int(card_id)
        for card_id in _sequence(row.get("godview_opp_hand_ids"))
        if card_id is not None and int(card_id) > 0
    )


def _opponent_known_counts(row: Mapping[str, Any]) -> Counter[int]:
    your_index = int(row.get("your_index", 0) or 0)
    opponent_index = 1 - your_index
    counts = _visible_player_counts(row, opponent_index)
    counts.update(
        {
            int(card_id): int(count)
            for card_id, count in zip(
                _sequence(row.get("opp_revealed_ids")),
                _sequence(row.get("opp_revealed_counts")),
                strict=True,
            )
            if card_id is not None and count is not None and int(card_id) > 0
        }
    )
    return Counter({card_id: count for card_id, count in counts.items() if count > 0})


def _opponent_unseen_counts(row: Mapping[str, Any]) -> Counter[int]:
    counts = Counter(
        int(card_id)
        for card_id in _sequence(row.get("opponent_deck_ids"))
        if card_id is not None and int(card_id) > 0
    )
    counts.subtract(_opponent_known_counts(row))
    return Counter({card_id: count for card_id, count in counts.items() if count > 0})


def _visible_player_counts(row: Mapping[str, Any], player_index: int) -> Counter[int]:
    prefix = f"player{player_index}"
    counts: Counter[int] = Counter()
    for field_name in (
        f"{prefix}_active_ids",
        f"{prefix}_bench_ids",
        f"{prefix}_discard_ids",
        f"{prefix}_prize_ids",
    ):
        counts.update(
            int(card_id)
            for card_id in _sequence(row.get(field_name))
            if card_id is not None and int(card_id) > 0
        )
    return counts


def _counts_distribution(counts: Mapping[int, float | int]) -> dict[int, float]:
    total = sum(count for count in counts.values() if count > 0)
    if total <= 0:
        return {}
    return {
        int(card_id): float(count) / float(total)
        for card_id, count in counts.items()
        if card_id > 0 and count > 0
    }


def _archetype_distribution(posterior: Any, known_counts: Counter[int]) -> dict[int, float]:
    expected: dict[int, float] = {}
    for entry in posterior.entries:
        for card_id, count in entry.deck.counts.items():
            remaining = int(count) - int(known_counts.get(card_id, 0))
            if remaining > 0:
                expected[card_id] = expected.get(card_id, 0.0) + (
                    float(entry.probability) * float(remaining)
                )
    return _counts_distribution(expected)


def _masked_distribution(
    probabilities: Sequence[float],
    candidates: Counter[int],
) -> dict[int, float]:
    distribution = {
        card_id: _probability_for_card(probabilities, card_id)
        for card_id, count in candidates.items()
        if card_id > 0 and count > 0
    }
    total = sum(distribution.values())
    if total <= 0.0:
        return {}
    return {card_id: probability / total for card_id, probability in distribution.items()}


def _hand_nll(hand_ids: Sequence[int], distribution: Mapping[int, float]) -> float:
    if not hand_ids:
        return 0.0
    return -sum(
        math.log(max(float(distribution.get(int(card_id), 0.0)), 1.0e-12))
        for card_id in hand_ids
    ) / float(len(hand_ids))


def _probability_for_card(probabilities: Sequence[float], card_id: int) -> float:
    index = int(card_id) - 1
    if index < 0 or index >= len(probabilities):
        return 0.0
    return max(0.0, float(probabilities[index]))


def _sequence(value: Any) -> tuple[Any, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return ()
