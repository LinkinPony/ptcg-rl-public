"""Build dynamic-effect probe sidecars for compact Kaggle step shards."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler, BeliefSamplerConfig
from ptcg_rl.context import (
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
    context_features_from_observation,
    opponent_belief_state_from_evidence,
)
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.engine.constants import OptionType
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.forward_model import extract_dynamic_effect_features
from ptcg_rl.training.bc_dataset import (
    KaggleStepDataConfig,
    observation_from_step_row,
    resolve_step_shards,
)

CORE_PROBE_OPTION_TYPES = (int(OptionType.ATTACK), int(OptionType.ABILITY))
PROBE_SIDECAR_VERSION = 1


class ProbeSidecarConfig(BaseModel):
    """Hydra-backed config for dynamic-effect probe sidecar extraction."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: Path = Path("outputs/kaggle_steps/full/manifest.json")
    shard_paths: tuple[Path, ...] = ()
    output_dir: Path = Path("outputs/kaggle_steps/probe_sidecar")
    read_batch_size: int = 256
    max_rows: int | None = None
    worlds: int = 3
    seed: int = 0
    manual_coin: bool = False
    belief: OpponentBeliefFeatureConfig = OpponentBeliefFeatureConfig()
    sampler: BeliefSamplerConfig = BeliefSamplerConfig(mode="archetype")

    @field_validator("read_batch_size", "worlds")
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive limits."""
        if value <= 0:
            raise ValueError("limits must be positive")
        return value

    @field_validator("max_rows")
    @classmethod
    def valid_optional_positive_int(cls, value: int | None) -> int | None:
        """Reject non-positive optional row limits."""
        if value is not None and value <= 0:
            raise ValueError("max_rows must be positive when set")
        return value


def run_probe_sidecar(config: ProbeSidecarConfig) -> dict[str, Any]:
    """Generate sidecar Parquet shards and return a compact summary."""
    output_dir = deck_records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_config = _data_config_from_probe_config(config)
    shard_paths = resolve_step_shards(data_config)
    belief_producer = OpponentBeliefFeatureProducer.from_config(config.belief)
    sampler = BeliefSampler(config=config.sampler)
    rng = random.Random(config.seed)

    summaries: list[dict[str, Any]] = []
    total_rows = 0
    total_probed_options = 0
    total_errors = 0
    for shard_path in shard_paths:
        if config.max_rows is not None and total_rows >= config.max_rows:
            break
        remaining = None if config.max_rows is None else config.max_rows - total_rows
        shard_summary = _write_probe_sidecar_shard(
            shard_path,
            output_dir=output_dir,
            read_batch_size=config.read_batch_size,
            max_rows=remaining,
            worlds=config.worlds,
            manual_coin=config.manual_coin,
            belief_producer=belief_producer,
            sampler=sampler,
            rng=rng,
        )
        summaries.append(shard_summary)
        total_rows += int(shard_summary["rows"])
        total_probed_options += int(shard_summary["probed_options"])
        total_errors += int(shard_summary["errors"])

    manifest = {
        "version": PROBE_SIDECAR_VERSION,
        "output_dir": deck_records.display_path(output_dir),
        "worlds": config.worlds,
        "manual_coin": config.manual_coin,
        "shards": summaries,
        "rows": total_rows,
        "probed_options": total_probed_options,
        "errors": total_errors,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def probe_sidecar_schema() -> pa.Schema:
    """Return the sidecar Parquet schema."""
    return pa.schema(
        [
            pa.field("episode_id", pa.int64()),
            pa.field("step_index", pa.int32()),
            pa.field("player_index", pa.int8()),
            pa.field("probe_version", pa.int16()),
            pa.field("probe_worlds", pa.int8()),
            pa.field("probe_effect_features", pa.list_(pa.list_(pa.float32()))),
            pa.field("probe_effect_masks", pa.list_(pa.bool_())),
            pa.field("probe_error", pa.string()),
        ]
    )


def _write_probe_sidecar_shard(
    shard_path: Path,
    *,
    output_dir: Path,
    read_batch_size: int,
    max_rows: int | None,
    worlds: int,
    manual_coin: bool,
    belief_producer: OpponentBeliefFeatureProducer,
    sampler: BeliefSampler,
    rng: random.Random,
) -> dict[str, Any]:
    output_path = output_dir / shard_path.name
    schema = probe_sidecar_schema()
    rows_written = 0
    probed_options = 0
    errors = 0
    with pq.ParquetWriter(output_path, schema=schema) as writer:
        parquet_file = pq.ParquetFile(shard_path)
        for record_batch in parquet_file.iter_batches(batch_size=read_batch_size):
            output_rows: list[dict[str, Any]] = []
            for row in record_batch.to_pylist():
                if max_rows is not None and rows_written >= max_rows:
                    break
                if not isinstance(row, Mapping):
                    continue
                sidecar_row = _probe_row(
                    row,
                    worlds=worlds,
                    manual_coin=manual_coin,
                    belief_producer=belief_producer,
                    sampler=sampler,
                    rng=rng,
                )
                output_rows.append(sidecar_row)
                rows_written += 1
                probed_options += sum(
                    bool(mask) for mask in sidecar_row["probe_effect_masks"]
                )
                if sidecar_row["probe_error"]:
                    errors += 1
            if output_rows:
                writer.write_table(pa.Table.from_pylist(output_rows, schema=schema))
            if max_rows is not None and rows_written >= max_rows:
                break
    return {
        "path": deck_records.display_path(output_path),
        "source_path": deck_records.display_path(shard_path),
        "rows": rows_written,
        "probed_options": probed_options,
        "errors": errors,
    }


def _probe_row(
    row: Mapping[str, Any],
    *,
    worlds: int,
    manual_coin: bool,
    belief_producer: OpponentBeliefFeatureProducer,
    sampler: BeliefSampler,
    rng: random.Random,
) -> dict[str, Any]:
    option_count = int(row.get("select_option_count") or 0)
    vectors = [[0.0] * DYNAMIC_EFFECT_FEATURE_SIZE for _ in range(option_count)]
    masks = [False] * option_count
    error_messages: list[str] = []
    core_candidates = _core_option_candidates(row)
    if core_candidates:
        try:
            observation = observation_from_step_row(
                row,
                belief_producer=belief_producer,
            )
            context_features = context_features_from_observation(observation)
            evidence = extract_observation_evidence(observation)
            opponent_state = opponent_belief_state_from_evidence(
                evidence,
                context_features,
            )
            your_deck = your_deck_from_step_row(row)
            sums: dict[int, list[float]] = {
                index: [0.0] * DYNAMIC_EFFECT_FEATURE_SIZE
                for (index,) in core_candidates
            }
            counts: Counter[int] = Counter()
            for _ in range(worlds):
                determinization = sampler.sample_from_evidence(
                    evidence,
                    your_deck=your_deck,
                    opponent_state=opponent_state,
                    rng=rng,
                )
                rows = extract_dynamic_effect_features(
                    observation,
                    determinization.hidden,
                    candidates=core_candidates,
                    option_types=CORE_PROBE_OPTION_TYPES,
                    manual_coin=manual_coin,
                )
                for feature_row in rows:
                    if len(feature_row.select) != 1:
                        continue
                    option_index = int(feature_row.select[0])
                    sums.setdefault(
                        option_index,
                        [0.0] * DYNAMIC_EFFECT_FEATURE_SIZE,
                    )
                    for feature_index, value in enumerate(feature_row.vector):
                        sums[option_index][feature_index] += float(value)
                    counts[option_index] += 1
            for option_index, total_vector in sums.items():
                count = counts[option_index]
                if 0 <= option_index < option_count and count > 0:
                    vectors[option_index] = [
                        value / float(count) for value in total_vector
                    ]
                    masks[option_index] = True
        except Exception as exc:  # Keep extraction streaming through bad rows.
            error_messages.append(f"{type(exc).__name__}: {exc}")
    return {
        "episode_id": int(row.get("episode_id") or 0),
        "step_index": int(row.get("step_index") or 0),
        "player_index": int(row.get("player_index") or 0),
        "probe_version": PROBE_SIDECAR_VERSION,
        "probe_worlds": worlds,
        "probe_effect_features": vectors,
        "probe_effect_masks": masks,
        "probe_error": "; ".join(error_messages),
    }


def _core_option_candidates(row: Mapping[str, Any]) -> tuple[tuple[int], ...]:
    option_types = _sequence(row.get("option_type"))
    candidates: list[tuple[int]] = []
    for index, option_type in enumerate(option_types):
        if option_type is not None and int(option_type) in CORE_PROBE_OPTION_TYPES:
            candidates.append((index,))
    return tuple(candidates)


def your_deck_from_step_row(row: Mapping[str, Any]) -> tuple[int, ...]:
    """Recover the full own deck from a compact step row."""
    signature = str(row.get("deck_signature") or "")
    if signature:
        signature_counts = deck_records.signature_counts(signature)
        return _expand_counts(signature_counts)
    deck_counts: Counter[int] = Counter()
    deck_counts.update(
        {
            int(card_id): int(count)
            for card_id, count in zip(
                _int_sequence(row.get("own_unseen_ids")),
                _int_sequence(row.get("own_unseen_counts")),
                strict=True,
            )
            if int(card_id) > 0 and int(count) > 0
        }
    )
    your_index = int(row.get("your_index") or row.get("player_index") or 0)
    prefix = f"player{your_index}"
    for field_name in (
        f"{prefix}_hand_ids",
        f"{prefix}_discard_ids",
        f"{prefix}_prize_ids",
        f"{prefix}_active_ids",
        f"{prefix}_bench_ids",
    ):
        deck_counts.update(
            card_id for card_id in _int_sequence(row.get(field_name)) if card_id > 0
        )
    while sum(deck_counts.values()) < 60:
        deck_counts[1] += 1
    return tuple(_expand_counts(deck_counts)[:60])


def _expand_counts(counts: Mapping[int, int]) -> tuple[int, ...]:
    return tuple(
        card_id
        for card_id, count in sorted(counts.items())
        for _ in range(max(0, int(count)))
        if card_id > 0
    )


def _data_config_from_probe_config(config: ProbeSidecarConfig) -> KaggleStepDataConfig:
    return KaggleStepDataConfig(
        manifest_path=config.manifest_path,
        shard_paths=config.shard_paths,
        read_batch_size=config.read_batch_size,
        shuffle_shards=False,
        shuffle_buffer_rows=1,
        probe_sidecar_dir=None,
        belief=config.belief,
    )


def _int_sequence(value: Any) -> tuple[int, ...]:
    return tuple(int(item) for item in _sequence(value) if item is not None)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
