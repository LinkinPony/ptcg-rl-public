"""Parity check for native rollout probe against the shipped Search API.

Run with:
    PYTHONPATH=data/sample_submission:src python src/tools/native_probe_parity.py
"""

from __future__ import annotations

import json
import math
import random
import statistics
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, SupportsFloat, cast

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.agent.probe import (
    FEATURE_INDEX,
    RuntimeProbeResult,
    core_option_candidates,
)
from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context import GameContext, context_features_from_observation
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.engine.native_probe import NativeProbeBackend
from ptcg_rl.engine.search_api_probe import SearchApiProbeBackend
from ptcg_rl.engine.vector_battle import DeckPair, VectorBattlePool


class NativeProbeParityConfig(BaseModel):
    """Hydra-backed config for native-vs-SearchAPI probe parity."""

    model_config = ConfigDict(extra="forbid")

    deck_paths: tuple[Path, ...]
    worlds: int = 3
    max_battle_steps: int = 120
    max_observations_per_pair: int = 2
    max_deck_pairs: int | None = 8
    seed: int = 0
    tolerance: float = 1.0e-6
    stochastic_sigma: float = 6.0
    min_stochastic_worlds: int = 2048
    sampler: BeliefSamplerConfig = BeliefSamplerConfig(mode="placeholder")
    output_path: Path | None = Path("outputs/engine/native_probe_parity/summary.json")
    hydra: Mapping[str, Any] | None = None

    @field_validator("deck_paths")
    @classmethod
    def valid_deck_paths(cls, value: tuple[Path, ...]) -> tuple[Path, ...]:
        """Require at least one deck path."""
        if not value:
            raise ValueError("deck_paths must not be empty")
        return value

    @field_validator(
        "worlds",
        "max_battle_steps",
        "max_observations_per_pair",
        "min_stochastic_worlds",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive sampling limits."""
        if value <= 0:
            raise ValueError("sampling limits must be positive")
        return value

    @field_validator("max_deck_pairs")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional sampling limits."""
        if value is not None and value <= 0:
            raise ValueError("max_deck_pairs must be positive when set")
        return value

    @field_validator("tolerance", "stochastic_sigma")
    @classmethod
    def valid_tolerance(cls, value: float) -> float:
        """Reject negative parity thresholds."""
        if value < 0.0:
            raise ValueError("tolerance must be non-negative")
        return value


def run_native_probe_parity(config: NativeProbeParityConfig) -> dict[str, Any]:
    """Run native-vs-SearchAPI feature parity checks."""
    decks = tuple(_read_deck(path) for path in config.deck_paths)
    deck_pairs = _deck_pairs(decks, config.max_deck_pairs)
    sampler = BeliefSampler(config=config.sampler)
    search_backend = SearchApiProbeBackend()
    native_backend = NativeProbeBackend()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    max_abs_diff = 0.0
    for pair_index, deck_pair in enumerate(deck_pairs):
        pair_records = _compare_deck_pair(
            deck_pair,
            pair_index=pair_index,
            sampler=sampler,
            search_backend=search_backend,
            native_backend=native_backend,
            config=config,
        )
        records.extend(pair_records)
        for record in pair_records:
            max_abs_diff = max(max_abs_diff, float(record["max_abs_diff"]))
            if not record["ok"]:
                failures.append(record)
    summary = {
        "deck_pair_count": len(deck_pairs),
        "comparison_count": len(records),
        "failure_count": len(failures),
        "max_abs_diff": max_abs_diff,
        "mean_native_ms": _mean(record["native_ms"] for record in records),
        "median_native_ms": _quantile(
            (record["native_ms"] for record in records),
            quantile=0.5,
        ),
        "p95_native_ms": _quantile(
            (record["native_ms"] for record in records),
            quantile=0.95,
        ),
        "max_native_ms": max(
            (float(record["native_ms"]) for record in records),
            default=0.0,
        ),
        "mean_search_api_ms": _mean(record["search_api_ms"] for record in records),
        "max_search_api_ms": max(
            (float(record["search_api_ms"]) for record in records),
            default=0.0,
        ),
        "tolerance": config.tolerance,
        "stochastic_sigma": config.stochastic_sigma,
        "min_stochastic_worlds": config.min_stochastic_worlds,
        "records": records,
    }
    if failures:
        summary["failures"] = failures
    if config.output_path is not None:
        config.output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if failures:
        raise AssertionError(
            f"native probe parity failed for {len(failures)} comparisons"
        )
    return summary


def _compare_deck_pair(
    deck_pair: DeckPair,
    *,
    pair_index: int,
    sampler: BeliefSampler,
    search_backend: SearchApiProbeBackend,
    native_backend: NativeProbeBackend,
    config: NativeProbeParityConfig,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    contexts: dict[tuple[str, int], GameContext] = {}
    with VectorBattlePool(
        1,
        lambda: deck_pair,
        include_search_input=True,
    ) as pool:
        for battle_step in range(config.max_battle_steps):
            pending = pool.pending()
            if not pending:
                return records
            game = pending[0]
            seat = int(game.observation["current"]["yourIndex"])
            key = (game.game_id, seat)
            context = contexts.get(key)
            if context is None:
                context = GameContext(player_index=seat)
                context.reset(player_index=seat, own_deck=game.deck_pair[seat])
                contexts[key] = context
            features = context.update(game.observation)
            observation = dict(game.observation)
            observation["gameContext"] = features.as_observation_dict()
            candidates = core_option_candidates(observation.get("select"))
            if candidates:
                record = _compare_observation(
                    observation,
                    deck=game.deck_pair[seat],
                    pair_index=pair_index,
                    battle_step=battle_step,
                    seat=seat,
                    sampler=sampler,
                    search_backend=search_backend,
                    native_backend=native_backend,
                    config=config,
                )
                records.append(record)
                if len(records) >= config.max_observations_per_pair:
                    return records
            pool.submit(game.game_id, _deterministic_action(game.observation))
    return records


def _compare_observation(
    observation: Mapping[str, Any],
    *,
    deck: Sequence[int],
    pair_index: int,
    battle_step: int,
    seat: int,
    sampler: BeliefSampler,
    search_backend: SearchApiProbeBackend,
    native_backend: NativeProbeBackend,
    config: NativeProbeParityConfig,
) -> dict[str, Any]:
    context_features = context_features_from_observation(observation)
    search_started = time.perf_counter()
    search_result = search_backend.run(
        observation,
        context_features,
        your_deck=deck,
        sampler=sampler,
        worlds=config.worlds,
        rng=random.Random(config.seed),
    )
    search_seconds = time.perf_counter() - search_started
    native_started = time.perf_counter()
    native_result = native_backend.run_exact_facts(
        observation,
        context_features,
        your_deck=deck,
        sampler=sampler,
        worlds=config.worlds,
        rng=random.Random(config.seed),
        require_byte_identical_worlds=True,
    )
    native_seconds = time.perf_counter() - native_started
    if search_result.probe is None or native_result is None:
        raise RuntimeError("core candidate observation produced no probe result")
    search_features, search_masks = _strict_exact_fact_rows(search_result.probe)
    native_features = np.asarray(native_result.features, dtype=np.float32)
    expected_features = np.asarray(search_features, dtype=np.float32)
    masks_equal = tuple(search_masks) == tuple(native_result.masks)
    max_abs_diff = float(
        np.max(np.abs(native_features - expected_features), initial=0.0)
    )
    unresolved_equal = (
        search_result.probe.unresolved_worlds
        == native_result.stats.unresolved_worlds
    )
    parity = {
        "masks_equal": masks_equal,
        "worlds_equal": search_result.probe.worlds_requested == config.worlds,
        "unresolved_equal": unresolved_equal,
        "max_abs_diff": max_abs_diff,
        "max_allowed_abs_diff": config.tolerance,
        "max_standardized_mean_diff": 0.0,
        "stochastic_action_count": 0,
        "insufficient_stochastic_worlds": 0,
        "ok": masks_equal
        and unresolved_equal
        and max_abs_diff <= config.tolerance,
    }
    return {
        "pair_index": pair_index,
        "battle_step": battle_step,
        "seat": seat,
        "candidate_count": len(core_option_candidates(observation.get("select"))),
        **parity,
        "search_api_ms": search_seconds * 1000.0,
        "native_ms": native_seconds * 1000.0,
        "search_api": search_result.stats.__dict__,
        "native": native_result.stats.__dict__,
    }


def _strict_exact_fact_rows(
    probe: RuntimeProbeResult,
) -> tuple[tuple[tuple[float, ...], ...], tuple[bool, ...]]:
    """Apply the production byte-identical-world exact-fact contract."""
    features = [
        (0.0,) * len(probe.features[0])
        for _ in probe.features
    ]
    masks = [False] * len(features)
    for action, vectors in probe.world_vectors.items():
        if len(action) != 1 or len(vectors) != probe.worlds_requested:
            continue
        option_index = int(action[0])
        if not 0 <= option_index < len(features):
            continue
        encoded = tuple(
            np.asarray(vector, dtype=np.float32).tobytes() for vector in vectors
        )
        if any(value != encoded[0] for value in encoded[1:]):
            continue
        features[option_index] = tuple(float(value) for value in vectors[0])
        masks[option_index] = True
    return tuple(features), tuple(masks)


def _probe_parity_metrics(
    search_probe: RuntimeProbeResult,
    native_probe: RuntimeProbeResult,
    *,
    tolerance: float,
    stochastic_sigma: float,
    min_stochastic_worlds: int,
) -> dict[str, Any]:
    """Compare paired deterministic rows and coin-dependent sample means."""
    masks_equal = tuple(search_probe.masks) == tuple(native_probe.masks)
    worlds_equal = search_probe.worlds_requested == native_probe.worlds_requested
    unresolved_equal = (
        search_probe.unresolved_options == native_probe.unresolved_options
        and search_probe.unresolved_worlds == native_probe.unresolved_worlds
    )
    actions = sorted(set(search_probe.world_vectors) | set(native_probe.world_vectors))
    max_abs_diff = 0.0
    max_allowed_abs_diff = tolerance
    max_standardized_mean_diff = 0.0
    stochastic_action_count = 0
    insufficient_stochastic_worlds = 0
    ok = masks_equal and worlds_equal
    coin_index = FEATURE_INDEX["coin_count_norm"]
    for action in actions:
        search_vectors = search_probe.world_vectors.get(action, ())
        native_vectors = native_probe.world_vectors.get(action, ())
        stochastic = any(
            abs(float(vector[coin_index])) > tolerance
            for vector in (*search_vectors, *native_vectors)
        )
        if stochastic:
            stochastic_action_count += 1
            if search_probe.worlds_requested < min_stochastic_worlds:
                insufficient_stochastic_worlds += 1
                ok = False
                continue
            resolution_ok, resolution_metrics = _stochastic_resolution_parity(
                len(search_vectors),
                len(native_vectors),
                worlds=search_probe.worlds_requested,
                tolerance=tolerance,
                stochastic_sigma=stochastic_sigma,
            )
            ok = ok and resolution_ok
            max_abs_diff = max(max_abs_diff, resolution_metrics[0])
            max_allowed_abs_diff = max(
                max_allowed_abs_diff,
                resolution_metrics[1],
            )
            max_standardized_mean_diff = max(
                max_standardized_mean_diff,
                resolution_metrics[2],
            )
            if not search_vectors or not native_vectors:
                continue
            action_ok, action_metrics = _stochastic_mean_parity(
                search_vectors,
                native_vectors,
                tolerance=tolerance,
                stochastic_sigma=stochastic_sigma,
            )
            ok = ok and action_ok
            max_abs_diff = max(max_abs_diff, action_metrics[0])
            max_allowed_abs_diff = max(
                max_allowed_abs_diff,
                action_metrics[1],
            )
            max_standardized_mean_diff = max(
                max_standardized_mean_diff,
                action_metrics[2],
            )
            continue
        if len(search_vectors) != len(native_vectors):
            ok = False
            continue
        if not search_vectors:
            continue
        for search_vector, native_vector in zip(
            search_vectors,
            native_vectors,
            strict=True,
        ):
            paired_diff = max(
                abs(float(search_value) - float(native_value))
                for search_value, native_value in zip(
                    search_vector,
                    native_vector,
                    strict=True,
                )
            )
            max_abs_diff = max(max_abs_diff, paired_diff)
            if paired_diff > tolerance:
                ok = False
    return {
        "masks_equal": masks_equal,
        "worlds_equal": worlds_equal,
        "unresolved_equal": unresolved_equal,
        "max_abs_diff": max_abs_diff,
        "max_allowed_abs_diff": max_allowed_abs_diff,
        "max_standardized_mean_diff": max_standardized_mean_diff,
        "stochastic_action_count": stochastic_action_count,
        "insufficient_stochastic_worlds": insufficient_stochastic_worlds,
        "ok": ok,
    }


def _stochastic_resolution_parity(
    search_resolved: int,
    native_resolved: int,
    *,
    worlds: int,
    tolerance: float,
    stochastic_sigma: float,
) -> tuple[bool, tuple[float, float, float]]:
    """Compare stochastic resolved-world rates with a binomial error bound."""
    search_rate = float(search_resolved) / float(worlds)
    native_rate = float(native_resolved) / float(worlds)
    mean_diff = abs(search_rate - native_rate)
    standard_error = math.sqrt(
        search_rate * (1.0 - search_rate) / float(worlds)
        + native_rate * (1.0 - native_rate) / float(worlds)
    )
    allowed_diff = tolerance + stochastic_sigma * standard_error
    standardized_diff = (
        mean_diff / standard_error
        if standard_error > 0.0
        else (math.inf if mean_diff > tolerance else 0.0)
    )
    return mean_diff <= allowed_diff, (
        mean_diff,
        allowed_diff,
        standardized_diff,
    )


def _stochastic_mean_parity(
    search_vectors: Sequence[Sequence[float]],
    native_vectors: Sequence[Sequence[float]],
    *,
    tolerance: float,
    stochastic_sigma: float,
) -> tuple[bool, tuple[float, float, float]]:
    """Return a conservative standard-error comparison for stochastic rows."""
    feature_count = len(search_vectors[0])
    ok = True
    max_abs_diff = 0.0
    max_allowed_abs_diff = tolerance
    max_standardized_mean_diff = 0.0
    for feature_index in range(feature_count):
        search_values = [float(row[feature_index]) for row in search_vectors]
        native_values = [float(row[feature_index]) for row in native_vectors]
        mean_diff = abs(
            statistics.fmean(search_values) - statistics.fmean(native_values)
        )
        standard_error = math.sqrt(
            _mean_variance(search_values) + _mean_variance(native_values)
        )
        allowed_diff = tolerance + stochastic_sigma * standard_error
        standardized_diff = (
            mean_diff / standard_error
            if standard_error > 0.0
            else (math.inf if mean_diff > tolerance else 0.0)
        )
        max_abs_diff = max(max_abs_diff, mean_diff)
        max_allowed_abs_diff = max(max_allowed_abs_diff, allowed_diff)
        max_standardized_mean_diff = max(
            max_standardized_mean_diff,
            standardized_diff,
        )
        if mean_diff > allowed_diff:
            ok = False
    return ok, (
        max_abs_diff,
        max_allowed_abs_diff,
        max_standardized_mean_diff,
    )


def _mean_variance(values: Sequence[float]) -> float:
    """Return the unbiased variance of a sample mean."""
    if len(values) < 2:
        return 0.0
    return float(statistics.variance(values)) / float(len(values))


def _deterministic_action(observation: Mapping[str, Any]) -> tuple[int, ...]:
    candidates = core_option_candidates(observation.get("select"))
    if candidates:
        return candidates[0]
    select = cast(Mapping[str, Any], observation.get("select", {}))
    min_count = max(0, int(select.get("minCount", 1)))
    return tuple(range(min_count))


def _deck_pairs(
    decks: Sequence[Sequence[int]],
    max_deck_pairs: int | None,
) -> tuple[DeckPair, ...]:
    pairs = tuple((tuple(deck), tuple(deck)) for deck in decks)
    if max_deck_pairs is None:
        return pairs
    return pairs[:max_deck_pairs]


def _read_deck(path: Path) -> tuple[int, ...]:
    deck = tuple(deck_records.read_deck(deck_records.repo_path(path)))
    if len(deck) != 60:
        raise ValueError(f"deck must contain 60 cards: {path}")
    return deck


def _mean(values: Iterable[SupportsFloat]) -> float:
    floats = [float(value) for value in values]
    return sum(floats) / len(floats) if floats else 0.0


def _quantile(values: Iterable[SupportsFloat], *, quantile: float) -> float:
    floats = sorted(float(value) for value in values)
    if not floats:
        return 0.0
    if quantile == 0.5:
        return float(statistics.median(floats))
    index = min(len(floats) - 1, max(0, round(quantile * (len(floats) - 1))))
    return floats[index]


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="engine/native_probe_parity",
)
def main(cfg: DictConfig) -> None:
    """Hydra CLI entrypoint."""
    config = NativeProbeParityConfig.model_validate(
        OmegaConf.to_container(cfg, resolve=True)
    )
    summary = run_native_probe_parity(config)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
