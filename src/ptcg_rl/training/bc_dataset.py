"""Streaming behavior-cloning batches from compact Kaggle step shards."""

from __future__ import annotations

import hashlib
import json
import queue
import random
import threading
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias, cast

import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from torch import Tensor

from ptcg_rl.actions.encoding import EncodedOption, StateTokenLayout, encode_options
from ptcg_rl.cards.static_features import DEFAULT_NUM_CARD_IDS
from ptcg_rl.context import (
    OpponentBeliefFeatureConfig,
    OpponentBeliefFeatureProducer,
    context_features_from_row,
)
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.model.policy import OptionBatch, collate_encoded_options
from ptcg_rl.model.state_encoder import (
    StateBatch,
    StateTokenFeatures,
    collate_state_tokens,
    encode_observation_tokens,
)

SplitName = Literal["train", "validation", "all"]


class SampleWeightConfig(BaseModel):
    """Config for optional row-level BC loss weights."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    sample_weight_column: str | None = None
    win_weight: float = 1.0
    loss_weight: float = 1.0
    draw_weight: float = 1.0
    other_weight: float = 1.0
    min_weight: float = 0.1
    max_weight: float = 10.0

    @field_validator(
        "win_weight",
        "loss_weight",
        "draw_weight",
        "other_weight",
        "min_weight",
        "max_weight",
    )
    @classmethod
    def valid_positive_weight(cls, value: float) -> float:
        """Reject non-positive sample weights."""
        if value <= 0.0:
            raise ValueError("sample weights must be positive")
        return value

    @model_validator(mode="after")
    def valid_weight_bounds(self) -> SampleWeightConfig:
        """Reject inconsistent clipping bounds."""
        if self.max_weight < self.min_weight:
            raise ValueError("max_weight must be >= min_weight")
        return self


class KaggleStepDataConfig(BaseModel):
    """Config for streaming BC batches from extracted Kaggle step shards."""

    model_config = ConfigDict(extra="forbid")

    manifest_path: Path = Path("outputs/kaggle_steps/latest/manifest.json")
    shard_paths: tuple[Path, ...] = ()
    batch_size: int = 64
    read_batch_size: int = 2048
    shuffle_shards: bool = True
    shuffle_buffer_rows: int = 8192
    validation_fraction: float = 0.05
    split_column: str | None = None
    drop_last: bool = False
    num_workers: int = 0
    prefetch_batches: int = 2
    pin_memory: bool = False
    sample_weights: SampleWeightConfig = SampleWeightConfig()
    belief: OpponentBeliefFeatureConfig = OpponentBeliefFeatureConfig()
    probe_sidecar_dir: Path | None = None
    probe_feature_dropout: float = 0.0
    tensor_cache_dir: Path | None = None

    @field_validator(
        "batch_size",
        "read_batch_size",
        "shuffle_buffer_rows",
        "prefetch_batches",
    )
    @classmethod
    def valid_positive_int(cls, value: int) -> int:
        """Reject non-positive data loader limits."""
        if value <= 0:
            raise ValueError("data loader limits must be positive")
        return value

    @field_validator("num_workers")
    @classmethod
    def valid_non_negative_int(cls, value: int) -> int:
        """Reject negative worker counts."""
        if value < 0:
            raise ValueError("num_workers must be non-negative")
        return value

    @field_validator("validation_fraction")
    @classmethod
    def valid_validation_fraction(cls, value: float) -> float:
        """Reject invalid validation split fractions."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("validation_fraction must be in [0, 1)")
        return value

    @field_validator("probe_feature_dropout")
    @classmethod
    def valid_probe_feature_dropout(cls, value: float) -> float:
        """Reject invalid feature dropout probabilities."""
        if value < 0.0 or value >= 1.0:
            raise ValueError("probe_feature_dropout must be in [0, 1)")
        return value

    @field_validator("split_column")
    @classmethod
    def valid_optional_split_column(cls, value: str | None) -> str | None:
        """Reject an empty explicit train/validation split column."""
        if value is not None and not value.strip():
            raise ValueError("split_column must be non-empty when set")
        return value

    @model_validator(mode="after")
    def valid_data_source(self) -> KaggleStepDataConfig:
        """Require either a manifest or explicit shard paths."""
        if (
            self.tensor_cache_dir is None
            and self.manifest_path == Path("")
            and not self.shard_paths
        ):
            raise ValueError("manifest_path or shard_paths must be set")
        return self


@dataclass(frozen=True)
class BCSample:
    """One tensor-ready behavior-cloning training sample."""

    state: StateTokenFeatures
    options: tuple[Any, ...]
    min_count: int
    max_count: int
    select_type: int
    select_context: int
    action: tuple[int, ...]
    value_target: float
    prize_diff_target: float
    prize_diff_mask: bool
    opponent_card_target: tuple[float, ...]
    opponent_card_target_mask: bool
    opponent_hand_target: tuple[float, ...]
    opponent_hand_candidate_mask: tuple[bool, ...]
    opponent_hand_target_mask: bool
    chosen_effect_target: tuple[float, ...]
    chosen_effect_mask: bool
    sample_weight: float


@dataclass(frozen=True)
class BCBatch:
    """A collated BC batch consumed by the policy/value trainer."""

    states: StateBatch
    options: OptionBatch
    select_types: tuple[int, ...]
    select_contexts: tuple[int, ...]
    actions: tuple[tuple[int, ...], ...]
    value_targets: Tensor
    prize_diff_targets: Tensor
    prize_diff_mask: Tensor
    opponent_card_targets: Tensor
    opponent_card_target_mask: Tensor
    opponent_hand_targets: Tensor
    opponent_hand_candidate_mask: Tensor
    opponent_hand_target_mask: Tensor
    chosen_effect_targets: Tensor
    chosen_effect_mask: Tensor
    sample_weights: Tensor


@dataclass(frozen=True)
class _WorkerDone:
    """Signal emitted when a prefetch worker is done."""

    worker_index: int


@dataclass(frozen=True)
class _WorkerError:
    """Signal emitted when a prefetch worker raises."""

    worker_index: int
    error: BaseException


_WorkerItem: TypeAlias = BCBatch | _WorkerDone | _WorkerError


def resolve_step_shards(config: KaggleStepDataConfig) -> tuple[Path, ...]:
    """Return Parquet shard paths from explicit paths or a manifest file."""
    if config.shard_paths:
        return tuple(deck_records.repo_path(path) for path in config.shard_paths)

    manifest_path = deck_records.repo_path(config.manifest_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    if not isinstance(shards, Sequence):
        raise ValueError(f"manifest shards must be a list: {manifest_path}")

    paths: list[Path] = []
    for shard in shards:
        if not isinstance(shard, Mapping):
            continue
        raw_path = shard.get("path")
        if isinstance(raw_path, str) and raw_path:
            paths.append(deck_records.repo_path(Path(raw_path)))
    if not paths:
        raise ValueError(f"no parquet shards found in {manifest_path}")
    return tuple(paths)


def iter_bc_batches(
    config: KaggleStepDataConfig,
    *,
    split: SplitName,
    epoch: int,
    seed: int,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> Iterator[BCBatch]:
    """Yield bounded-memory BC batches for one epoch and split."""
    if config.tensor_cache_dir is not None:
        from ptcg_rl.training.bc_tensor_cache import iter_bc_tensor_cache_batches

        yield from iter_bc_tensor_cache_batches(
            config,
            split=split,
            epoch=epoch,
            seed=seed,
            device=device,
            max_batches=max_batches,
        )
        return

    rng_seed = seed + epoch * 104_729 + _split_seed(split)
    rng = random.Random(rng_seed)
    paths = list(resolve_step_shards(config))
    if config.shuffle_shards and split == "train":
        rng.shuffle(paths)
    if config.num_workers > 0 and split == "train":
        yield from _iter_prefetched_bc_batches(
            config,
            paths=tuple(paths),
            split=split,
            seed=rng_seed,
            device=device,
            max_batches=max_batches,
        )
        return

    samples: list[BCSample] = []
    yielded = 0
    belief_producer = _belief_producer_from_config(config.belief)
    for row in _iter_split_rows_for_paths(config, paths=paths, split=split, rng=rng):
        sample = sample_from_step_row(
            row,
            weight_config=config.sample_weights,
            belief_producer=belief_producer,
            probe_feature_dropout=(
                config.probe_feature_dropout if split == "train" else 0.0
            ),
            rng=rng,
        )
        if sample is None:
            continue
        samples.append(sample)
        if len(samples) >= config.batch_size:
            yield collate_bc_samples(samples, device=device)
            yielded += 1
            samples = []
            if max_batches is not None and yielded >= max_batches:
                return
    if samples and not config.drop_last:
        yield collate_bc_samples(samples, device=device)


def sample_from_step_row(
    row: Mapping[str, Any],
    *,
    weight_config: SampleWeightConfig | None = None,
    belief_producer: OpponentBeliefFeatureProducer | None = None,
    probe_feature_dropout: float = 0.0,
    rng: random.Random | None = None,
) -> BCSample | None:
    """Convert one compact Parquet row to a tensor-ready BC sample."""
    observation = observation_from_step_row(row, belief_producer=belief_producer)
    select = cast(Mapping[str, Any], observation["select"])
    layout = StateTokenLayout.from_observation(observation)
    encoded_options = _attach_probe_features(
        encode_options(select, layout),
        row,
        dropout=probe_feature_dropout,
        rng=rng,
    )
    if not encoded_options:
        return None

    action = tuple(_int_list(row, "action"))
    if not _action_in_bounds(action, len(encoded_options)):
        return None

    state = encode_observation_tokens(observation, layout=layout)
    prize_diff = _optional_float(row.get("terminal_prize_diff"))
    opponent_target, opponent_target_mask = _opponent_card_target(row)
    (
        opponent_hand_target,
        opponent_hand_candidate_mask,
        opponent_hand_target_mask,
    ) = _opponent_hand_target(row)
    chosen_effect_target, chosen_effect_mask = _chosen_effect_target(row)
    return BCSample(
        state=state,
        options=encoded_options,
        min_count=_int_value(row.get("select_min_count")),
        max_count=_int_value(row.get("select_max_count"), len(encoded_options)),
        select_type=_int_value(row.get("select_type")),
        select_context=_int_value(row.get("select_context")),
        action=action,
        value_target=_value_target(row),
        prize_diff_target=0.0 if prize_diff is None else float(prize_diff),
        prize_diff_mask=prize_diff is not None,
        opponent_card_target=opponent_target,
        opponent_card_target_mask=opponent_target_mask,
        opponent_hand_target=opponent_hand_target,
        opponent_hand_candidate_mask=opponent_hand_candidate_mask,
        opponent_hand_target_mask=opponent_hand_target_mask,
        chosen_effect_target=chosen_effect_target,
        chosen_effect_mask=chosen_effect_mask,
        sample_weight=_sample_weight(row, weight_config or SampleWeightConfig()),
    )


def collate_bc_samples(
    samples: Sequence[BCSample],
    *,
    device: torch.device | str | None = None,
) -> BCBatch:
    """Collate tensor-ready samples into one padded model batch."""
    if not samples:
        raise ValueError("samples must be non-empty")
    return BCBatch(
        states=collate_state_tokens([sample.state for sample in samples], device=device),
        options=collate_encoded_options(
            [sample.options for sample in samples],
            min_counts=[sample.min_count for sample in samples],
            max_counts=[sample.max_count for sample in samples],
            device=device,
        ),
        select_types=tuple(sample.select_type for sample in samples),
        select_contexts=tuple(sample.select_context for sample in samples),
        actions=tuple(sample.action for sample in samples),
        value_targets=torch.tensor(
            [sample.value_target for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        prize_diff_targets=torch.tensor(
            [sample.prize_diff_target for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        prize_diff_mask=torch.tensor(
            [sample.prize_diff_mask for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        opponent_card_targets=torch.tensor(
            [sample.opponent_card_target for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        opponent_card_target_mask=torch.tensor(
            [sample.opponent_card_target_mask for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        opponent_hand_targets=torch.tensor(
            [sample.opponent_hand_target for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        opponent_hand_candidate_mask=torch.tensor(
            [sample.opponent_hand_candidate_mask for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        opponent_hand_target_mask=torch.tensor(
            [sample.opponent_hand_target_mask for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        chosen_effect_targets=torch.tensor(
            [sample.chosen_effect_target for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
        chosen_effect_mask=torch.tensor(
            [sample.chosen_effect_mask for sample in samples],
            dtype=torch.bool,
            device=device,
        ),
        sample_weights=torch.tensor(
            [sample.sample_weight for sample in samples],
            dtype=torch.float32,
            device=device,
        ),
    )


def move_bc_batch_to_device(
    batch: BCBatch,
    *,
    device: torch.device,
    non_blocking: bool = False,
) -> BCBatch:
    """Move a collated BC batch to a target torch device."""
    return _move_bc_batch(batch, device=device, non_blocking=non_blocking)


def pin_bc_batch_memory(batch: BCBatch) -> BCBatch:
    """Pin a collated BC batch in host memory for CUDA transfer."""
    return _pin_bc_batch(batch)


def observation_from_step_row(
    row: Mapping[str, Any],
    *,
    belief_producer: OpponentBeliefFeatureProducer | None = None,
) -> dict[str, Any]:
    """Reconstruct the minimal observation shape used by existing encoders."""
    your_index = _int_value(row.get("your_index"))
    players = [_player_from_row(row, 0), _player_from_row(row, 1)]
    context_features = context_features_from_row(row)
    observation: dict[str, Any] = {
        "remainingOverageTime": _optional_float(row.get("remaining_overage_time")),
        "current": {
            "turn": _int_value(row.get("turn")),
            "turnActionCount": _int_value(row.get("turn_action_count")),
            "yourIndex": your_index,
            "firstPlayer": _int_value(row.get("first_player"), -1),
            "result": _int_value(row.get("state_result")),
            "supporterPlayed": bool(row.get("supporter_played", False)),
            "stadiumPlayed": bool(row.get("stadium_played", False)),
            "energyAttached": bool(row.get("energy_attached", False)),
            "retreated": bool(row.get("retreated", False)),
            "looking": _cards_from_ids(
                _int_list(row, "looking_ids"),
                player_index=your_index,
            ),
            "stadium": _cards_from_ids(
                _int_list(row, "stadium_ids"),
                player_index=-1,
            ),
            "players": players,
        },
        "gameContext": context_features.as_observation_dict(),
        "select": _select_from_row(row, your_index=your_index),
        "logs": [],
        "search_begin_input": row.get("search_begin_input"),
    }
    if belief_producer is not None:
        context_features = belief_producer.augment(observation, context_features)
        observation["gameContext"] = context_features.as_observation_dict()
    return observation


def _iter_split_rows(
    config: KaggleStepDataConfig,
    *,
    split: SplitName,
    rng: random.Random,
) -> Iterator[Mapping[str, Any]]:
    paths = list(resolve_step_shards(config))
    if config.shuffle_shards and split == "train":
        rng.shuffle(paths)
    yield from _iter_split_rows_for_paths(config, paths=paths, split=split, rng=rng)


def _iter_split_rows_for_paths(
    config: KaggleStepDataConfig,
    *,
    paths: Sequence[Path],
    split: SplitName,
    rng: random.Random,
) -> Iterator[Mapping[str, Any]]:
    rows = _iter_parquet_rows(
        paths,
        batch_size=config.read_batch_size,
        probe_sidecar_dir=config.probe_sidecar_dir,
    )
    filtered = (
        row
        for row in rows
        if split == "all" or _row_split(row, config) == split
    )
    if split == "train" and config.shuffle_buffer_rows > 1:
        yield from _buffered_shuffle(
            filtered,
            buffer_size=config.shuffle_buffer_rows,
            rng=rng,
        )
    else:
        yield from filtered


def _iter_prefetched_bc_batches(
    config: KaggleStepDataConfig,
    *,
    paths: Sequence[Path],
    split: SplitName,
    seed: int,
    device: torch.device | str | None,
    max_batches: int | None,
) -> Iterator[BCBatch]:
    worker_count = min(config.num_workers, max(1, len(paths)))
    item_queue: queue.Queue[_WorkerItem] = queue.Queue(
        maxsize=max(1, worker_count * config.prefetch_batches)
    )
    stop_event = threading.Event()
    threads = [
        threading.Thread(
            target=_prefetch_worker,
            args=(
                worker_index,
                tuple(worker_paths),
                config,
                split,
                seed,
                item_queue,
                stop_event,
            ),
            name=f"bc-prefetch-{worker_index}",
        )
        for worker_index, worker_paths in enumerate(_partition_paths(paths, worker_count))
    ]
    for thread in threads:
        thread.start()

    done_workers = 0
    yielded = 0
    try:
        while done_workers < worker_count:
            item = item_queue.get()
            if isinstance(item, _WorkerDone):
                done_workers += 1
                continue
            if isinstance(item, _WorkerError):
                stop_event.set()
                raise item.error

            yield _prepare_batch_for_device(
                item,
                device=device,
                pin_memory=config.pin_memory,
            )
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                stop_event.set()
                break
    finally:
        stop_event.set()
        for thread in threads:
            thread.join()


def _prefetch_worker(
    worker_index: int,
    paths: Sequence[Path],
    config: KaggleStepDataConfig,
    split: SplitName,
    seed: int,
    item_queue: queue.Queue[_WorkerItem],
    stop_event: threading.Event,
) -> None:
    try:
        rng = random.Random(seed + worker_index * 1_000_003)
        belief_producer = _belief_producer_from_config(config.belief)
        samples: list[BCSample] = []
        for row in _iter_split_rows_for_paths(
            config,
            paths=paths,
            split=split,
            rng=rng,
        ):
            if stop_event.is_set():
                return
            sample = sample_from_step_row(
                row,
                weight_config=config.sample_weights,
                belief_producer=belief_producer,
                probe_feature_dropout=(
                    config.probe_feature_dropout if split == "train" else 0.0
                ),
                rng=rng,
            )
            if sample is None:
                continue
            samples.append(sample)
            if len(samples) >= config.batch_size:
                batch = collate_bc_samples(samples)
                samples = []
                if not _put_worker_item(item_queue, batch, stop_event):
                    return
        if samples and not config.drop_last:
            batch = collate_bc_samples(samples)
            if not _put_worker_item(item_queue, batch, stop_event):
                return
    except BaseException as exc:
        _put_worker_item(item_queue, _WorkerError(worker_index, exc), stop_event)
    finally:
        _put_worker_item(item_queue, _WorkerDone(worker_index), stop_event)


def _put_worker_item(
    item_queue: queue.Queue[_WorkerItem],
    item: _WorkerItem,
    stop_event: threading.Event,
) -> bool:
    while not stop_event.is_set():
        try:
            item_queue.put(item, timeout=0.1)
            return True
        except queue.Full:
            continue
    return False


def _partition_paths(paths: Sequence[Path], worker_count: int) -> list[list[Path]]:
    partitions: list[list[Path]] = [[] for _ in range(worker_count)]
    for index, path in enumerate(paths):
        partitions[index % worker_count].append(path)
    return partitions


def _prepare_batch_for_device(
    batch: BCBatch,
    *,
    device: torch.device | str | None,
    pin_memory: bool,
) -> BCBatch:
    target_device = torch.device(device) if device is not None else None
    active_batch = (
        _pin_bc_batch(batch)
        if pin_memory and _should_pin_memory(target_device)
        else batch
    )
    if target_device is None:
        return active_batch
    return _move_bc_batch(
        active_batch,
        device=target_device,
        non_blocking=pin_memory and _should_pin_memory(target_device),
    )


def _should_pin_memory(device: torch.device | None) -> bool:
    return device is not None and device.type == "cuda" and torch.cuda.is_available()


def _belief_producer_from_config(
    config: OpponentBeliefFeatureConfig,
) -> OpponentBeliefFeatureProducer | None:
    if not config.enabled:
        return None
    return OpponentBeliefFeatureProducer.from_config(config)


def _pin_bc_batch(batch: BCBatch) -> BCBatch:
    return BCBatch(
        states=_pin_state_batch(batch.states),
        options=_pin_option_batch(batch.options),
        select_types=batch.select_types,
        select_contexts=batch.select_contexts,
        actions=batch.actions,
        value_targets=batch.value_targets.pin_memory(),
        prize_diff_targets=batch.prize_diff_targets.pin_memory(),
        prize_diff_mask=batch.prize_diff_mask.pin_memory(),
        opponent_card_targets=batch.opponent_card_targets.pin_memory(),
        opponent_card_target_mask=batch.opponent_card_target_mask.pin_memory(),
        opponent_hand_targets=batch.opponent_hand_targets.pin_memory(),
        opponent_hand_candidate_mask=batch.opponent_hand_candidate_mask.pin_memory(),
        opponent_hand_target_mask=batch.opponent_hand_target_mask.pin_memory(),
        chosen_effect_targets=batch.chosen_effect_targets.pin_memory(),
        chosen_effect_mask=batch.chosen_effect_mask.pin_memory(),
        sample_weights=batch.sample_weights.pin_memory(),
    )


def _pin_state_batch(batch: StateBatch) -> StateBatch:
    return StateBatch(
        card_ids=batch.card_ids.pin_memory(),
        areas=batch.areas.pin_memory(),
        owner_roles=batch.owner_roles.pin_memory(),
        token_kinds=batch.token_kinds.pin_memory(),
        scalars=batch.scalars.pin_memory(),
        last_attack_ids=batch.last_attack_ids.pin_memory(),
        padding_mask=batch.padding_mask.pin_memory(),
        attachment_card_ids=_optional_pin_tensor(batch.attachment_card_ids),
        attachment_parent_indices=_optional_pin_tensor(
            batch.attachment_parent_indices
        ),
        attachment_kinds=_optional_pin_tensor(batch.attachment_kinds),
        entity_slots=_optional_pin_tensor(batch.entity_slots),
    )


def _optional_pin_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.pin_memory()


def _pin_option_batch(batch: OptionBatch) -> OptionBatch:
    return OptionBatch(
        option_types=batch.option_types.pin_memory(),
        contexts=batch.contexts.pin_memory(),
        entity_slots=batch.entity_slots.pin_memory(),
        entity_slot_mask=batch.entity_slot_mask.pin_memory(),
        attack_ids=batch.attack_ids.pin_memory(),
        card_ids=batch.card_ids.pin_memory(),
        scalars=batch.scalars.pin_memory(),
        dynamic_effect_features=batch.dynamic_effect_features.pin_memory(),
        dynamic_effect_masks=batch.dynamic_effect_masks.pin_memory(),
        valid_options=batch.valid_options.pin_memory(),
        min_counts=batch.min_counts.pin_memory(),
        max_counts=batch.max_counts.pin_memory(),
    )


def _move_bc_batch(
    batch: BCBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> BCBatch:
    return BCBatch(
        states=_move_state_batch(batch.states, device=device, non_blocking=non_blocking),
        options=_move_option_batch(
            batch.options,
            device=device,
            non_blocking=non_blocking,
        ),
        select_types=batch.select_types,
        select_contexts=batch.select_contexts,
        actions=batch.actions,
        value_targets=batch.value_targets.to(device=device, non_blocking=non_blocking),
        prize_diff_targets=batch.prize_diff_targets.to(
            device=device,
            non_blocking=non_blocking,
        ),
        prize_diff_mask=batch.prize_diff_mask.to(device=device, non_blocking=non_blocking),
        opponent_card_targets=batch.opponent_card_targets.to(
            device=device,
            non_blocking=non_blocking,
        ),
        opponent_card_target_mask=batch.opponent_card_target_mask.to(
            device=device,
            non_blocking=non_blocking,
        ),
        opponent_hand_targets=batch.opponent_hand_targets.to(
            device=device,
            non_blocking=non_blocking,
        ),
        opponent_hand_candidate_mask=batch.opponent_hand_candidate_mask.to(
            device=device,
            non_blocking=non_blocking,
        ),
        opponent_hand_target_mask=batch.opponent_hand_target_mask.to(
            device=device,
            non_blocking=non_blocking,
        ),
        chosen_effect_targets=batch.chosen_effect_targets.to(
            device=device,
            non_blocking=non_blocking,
        ),
        chosen_effect_mask=batch.chosen_effect_mask.to(
            device=device,
            non_blocking=non_blocking,
        ),
        sample_weights=batch.sample_weights.to(device=device, non_blocking=non_blocking),
    )


def _move_state_batch(
    batch: StateBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> StateBatch:
    return StateBatch(
        card_ids=batch.card_ids.to(device=device, non_blocking=non_blocking),
        areas=batch.areas.to(device=device, non_blocking=non_blocking),
        owner_roles=batch.owner_roles.to(device=device, non_blocking=non_blocking),
        token_kinds=batch.token_kinds.to(device=device, non_blocking=non_blocking),
        scalars=batch.scalars.to(device=device, non_blocking=non_blocking),
        last_attack_ids=batch.last_attack_ids.to(
            device=device,
            non_blocking=non_blocking,
        ),
        padding_mask=batch.padding_mask.to(device=device, non_blocking=non_blocking),
        attachment_card_ids=_optional_move_tensor(
            batch.attachment_card_ids,
            device=device,
            non_blocking=non_blocking,
        ),
        attachment_parent_indices=_optional_move_tensor(
            batch.attachment_parent_indices,
            device=device,
            non_blocking=non_blocking,
        ),
        attachment_kinds=_optional_move_tensor(
            batch.attachment_kinds,
            device=device,
            non_blocking=non_blocking,
        ),
        entity_slots=_optional_move_tensor(
            batch.entity_slots,
            device=device,
            non_blocking=non_blocking,
        ),
    )


def _optional_move_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    non_blocking: bool,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.to(device=device, non_blocking=non_blocking)


def _move_option_batch(
    batch: OptionBatch,
    *,
    device: torch.device,
    non_blocking: bool,
) -> OptionBatch:
    return OptionBatch(
        option_types=batch.option_types.to(device=device, non_blocking=non_blocking),
        contexts=batch.contexts.to(device=device, non_blocking=non_blocking),
        entity_slots=batch.entity_slots.to(device=device, non_blocking=non_blocking),
        entity_slot_mask=batch.entity_slot_mask.to(
            device=device,
            non_blocking=non_blocking,
        ),
        attack_ids=batch.attack_ids.to(device=device, non_blocking=non_blocking),
        card_ids=batch.card_ids.to(device=device, non_blocking=non_blocking),
        scalars=batch.scalars.to(device=device, non_blocking=non_blocking),
        dynamic_effect_features=batch.dynamic_effect_features.to(
            device=device,
            non_blocking=non_blocking,
        ),
        dynamic_effect_masks=batch.dynamic_effect_masks.to(
            device=device,
            non_blocking=non_blocking,
        ),
        valid_options=batch.valid_options.to(device=device, non_blocking=non_blocking),
        min_counts=batch.min_counts.to(device=device, non_blocking=non_blocking),
        max_counts=batch.max_counts.to(device=device, non_blocking=non_blocking),
    )


def _iter_parquet_rows(
    paths: Sequence[Path],
    *,
    batch_size: int,
    probe_sidecar_dir: Path | None,
) -> Iterator[dict[str, Any]]:
    for path in paths:
        if probe_sidecar_dir is None:
            parquet_file = pq.ParquetFile(path)
            for record_batch in parquet_file.iter_batches(batch_size=batch_size):
                for row in record_batch.to_pylist():
                    if isinstance(row, dict):
                        yield cast(dict[str, Any], row)
            continue
        yield from _iter_parquet_rows_with_probe_sidecar(
            path,
            probe_sidecar_dir=probe_sidecar_dir,
            batch_size=batch_size,
        )


def _iter_parquet_rows_with_probe_sidecar(
    path: Path,
    *,
    probe_sidecar_dir: Path,
    batch_size: int,
) -> Iterator[dict[str, Any]]:
    sidecar_path = _probe_sidecar_path(path, probe_sidecar_dir)
    parquet_file = pq.ParquetFile(path)
    sidecar_file = pq.ParquetFile(sidecar_path)
    main_batches = parquet_file.iter_batches(batch_size=batch_size)
    sidecar_batches = sidecar_file.iter_batches(batch_size=batch_size)
    for record_batch, sidecar_batch in zip(main_batches, sidecar_batches, strict=True):
        rows = record_batch.to_pylist()
        sidecar_rows = sidecar_batch.to_pylist()
        if len(rows) != len(sidecar_rows):
            raise ValueError(f"probe sidecar batch length mismatch: {sidecar_path}")
        for row, sidecar_row in zip(rows, sidecar_rows, strict=True):
            if not isinstance(row, dict) or not isinstance(sidecar_row, dict):
                continue
            typed_row = cast(dict[str, Any], row)
            _validate_probe_sidecar_key(typed_row, sidecar_row, sidecar_path)
            typed_row.update(
                {
                    "probe_effect_features": sidecar_row.get("probe_effect_features"),
                    "probe_effect_masks": sidecar_row.get("probe_effect_masks"),
                }
            )
            yield typed_row


def _probe_sidecar_path(path: Path, probe_sidecar_dir: Path) -> Path:
    sidecar_path = deck_records.repo_path(probe_sidecar_dir) / path.name
    if not sidecar_path.exists():
        raise FileNotFoundError(f"probe sidecar shard not found: {sidecar_path}")
    return sidecar_path


def _validate_probe_sidecar_key(
    row: Mapping[str, Any],
    sidecar_row: Mapping[str, Any],
    sidecar_path: Path,
) -> None:
    key_fields = ("episode_id", "step_index", "player_index")
    for field_name in key_fields:
        if int(row.get(field_name, -1)) != int(sidecar_row.get(field_name, -2)):
            raise ValueError(f"probe sidecar row-key mismatch in {sidecar_path}")


def _buffered_shuffle(
    rows: Iterable[Mapping[str, Any]],
    *,
    buffer_size: int,
    rng: random.Random,
) -> Iterator[Mapping[str, Any]]:
    buffer: list[Mapping[str, Any]] = []
    for row in rows:
        buffer.append(row)
        if len(buffer) < buffer_size:
            continue
        index = rng.randrange(len(buffer))
        yield buffer.pop(index)
    rng.shuffle(buffer)
    yield from buffer


def _row_split(
    row: Mapping[str, Any],
    config: KaggleStepDataConfig,
) -> SplitName:
    if config.split_column is not None:
        raw_split = str(row.get(config.split_column, "")).strip().lower()
        if raw_split not in {"train", "validation"}:
            raise ValueError(
                f"row has invalid {config.split_column!r} split value: {raw_split!r}"
            )
        return cast(SplitName, raw_split)
    if config.validation_fraction <= 0.0:
        return "train"
    key = f"{row.get('date', '')}:{row.get('episode_id', '')}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    bucket = int.from_bytes(digest, byteorder="big") / float(1 << 64)
    return "validation" if bucket < config.validation_fraction else "train"


def _player_from_row(row: Mapping[str, Any], player_index: int) -> dict[str, Any]:
    prefix = f"player{player_index}"
    return {
        "active": _pokemon_zone_from_row(row, f"{prefix}_active", player_index),
        "bench": _pokemon_zone_from_row(row, f"{prefix}_bench", player_index),
        "benchMax": _int_value(row.get(f"{prefix}_bench_max")),
        "deckCount": _int_value(row.get(f"{prefix}_deck_count")),
        "discard": _cards_from_ids(
            _int_list(row, f"{prefix}_discard_ids"),
            player_index=player_index,
        ),
        "prize": _prize_cards_from_ids(
            _int_list(row, f"{prefix}_prize_ids"),
            player_index=player_index,
        ),
        "handCount": _int_value(row.get(f"{prefix}_hand_count")),
        "hand": _cards_from_ids(
            _int_list(row, f"{prefix}_hand_ids"),
            player_index=player_index,
        ),
        "poisoned": bool(row.get(f"{prefix}_poisoned", False)),
        "burned": bool(row.get(f"{prefix}_burned", False)),
        "asleep": bool(row.get(f"{prefix}_asleep", False)),
        "paralyzed": bool(row.get(f"{prefix}_paralyzed", False)),
        "confused": bool(row.get(f"{prefix}_confused", False)),
    }


def _pokemon_zone_from_row(
    row: Mapping[str, Any],
    prefix: str,
    player_index: int,
) -> list[dict[str, Any] | None]:
    ids = _int_list(row, f"{prefix}_ids")
    serials = _int_list(row, f"{prefix}_serials")
    hp_values = _int_list(row, f"{prefix}_hp")
    max_hp_values = _int_list(row, f"{prefix}_max_hp")
    appear_flags = _bool_list(row, f"{prefix}_appear_this_turn")
    energy_counts = _nested_int_lists(row, f"{prefix}_energy_counts")
    tool_counts = _int_list(row, f"{prefix}_tool_counts")
    evolution_depths = _int_list(row, f"{prefix}_evolution_depths")

    pokemon: list[dict[str, Any] | None] = []
    for index, card_id in enumerate(ids):
        if int(card_id) <= 0:
            pokemon.append(None)
            continue
        pokemon.append(
            {
                "id": int(card_id),
                "serial": _list_get(serials, index),
                "playerIndex": player_index,
                "hp": _list_get(hp_values, index),
                "maxHp": _list_get(max_hp_values, index),
                "appearThisTurn": _bool_list_get(appear_flags, index),
                "energies": _energy_list(_nested_list_get(energy_counts, index)),
                "energyCards": [],
                "tools": [None] * max(0, _list_get(tool_counts, index)),
                "preEvolution": [None] * max(0, _list_get(evolution_depths, index)),
            }
        )
    return pokemon


def _select_from_row(row: Mapping[str, Any], *, your_index: int) -> dict[str, Any]:
    return {
        "type": _int_value(row.get("select_type")),
        "context": _int_value(row.get("select_context")),
        "minCount": _int_value(row.get("select_min_count")),
        "maxCount": _int_value(row.get("select_max_count")),
        "remainDamageCounter": _int_value(row.get("remain_damage_counter")),
        "remainEnergyCost": _int_value(row.get("remain_energy_cost")),
        "contextCard": _optional_card(
            _int_value(row.get("context_card_id")),
            _int_value(row.get("context_card_serial")),
            player_index=your_index,
        ),
        "effect": _optional_card(
            _int_value(row.get("effect_card_id")),
            _int_value(row.get("effect_card_serial")),
            player_index=your_index,
        ),
        "deck": (
            _cards_from_ids(
                _int_list(row, "select_deck_ids"),
                player_index=your_index,
            )
            if bool(row.get("select_deck_present", False))
            else None
        ),
        "option": _options_from_row(row),
    }


def _options_from_row(row: Mapping[str, Any]) -> list[dict[str, int | None]]:
    option_count = _int_value(row.get("select_option_count"))
    columns = {
        source_name: _optional_int_list(row, f"option_{column_name}")
        for source_name, column_name in step_records.OPTION_FIELDS
    }
    if option_count <= 0:
        option_count = max((len(values) for values in columns.values()), default=0)

    options: list[dict[str, int | None]] = []
    for index in range(option_count):
        option: dict[str, int | None] = {}
        for source_name, values in columns.items():
            value = values[index] if index < len(values) else None
            if value is not None:
                option[source_name] = int(value)
        options.append(option)
    return options


def _attach_probe_features(
    options: Sequence[EncodedOption],
    row: Mapping[str, Any],
    *,
    dropout: float,
    rng: random.Random | None,
) -> tuple[EncodedOption, ...]:
    features = _sequence(row.get("probe_effect_features"))
    masks = _sequence(row.get("probe_effect_masks"))
    if not features or not masks:
        return tuple(options)
    active_rng = rng or random.Random()
    updated: list[EncodedOption] = []
    for index, option in enumerate(options):
        has_feature = index < len(masks) and bool(masks[index])
        if (
            has_feature
            and dropout > 0.0
            and active_rng.random() < dropout
        ):
            has_feature = False
        if not has_feature or index >= len(features):
            updated.append(option)
            continue
        vector = tuple(float(value) for value in _sequence(features[index]))
        if len(vector) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError("probe_effect_features has invalid width")
        updated.append(
            replace(
                option,
                dynamic_effect_features=vector,
                dynamic_effect_mask=True,
            )
        )
    return tuple(updated)


def _cards_from_ids(card_ids: Sequence[int], *, player_index: int) -> list[Any]:
    return [
        _optional_card(card_id, serial=0, player_index=player_index)
        for card_id in card_ids
    ]


def _prize_cards_from_ids(card_ids: Sequence[int], *, player_index: int) -> list[Any]:
    return [
        None if card_id <= 0 else _optional_card(card_id, serial=0, player_index=player_index)
        for card_id in card_ids
    ]


def _optional_card(card_id: int, serial: int, *, player_index: int) -> dict[str, int] | None:
    if card_id <= 0:
        return None
    return {"id": int(card_id), "serial": int(serial), "playerIndex": int(player_index)}


def _energy_list(counts: Sequence[int]) -> list[int]:
    energies: list[int] = []
    for energy_index, count in enumerate(counts):
        energies.extend([energy_index] * max(0, int(count)))
    return energies


def _opponent_card_target(row: Mapping[str, Any]) -> tuple[tuple[float, ...], bool]:
    positive = _opponent_unseen_counts(row)
    total = sum(positive.values())
    target = [0.0] * DEFAULT_NUM_CARD_IDS
    if total <= 0:
        return tuple(target), False
    for card_id, count in positive.items():
        if 1 <= card_id <= DEFAULT_NUM_CARD_IDS:
            target[card_id - 1] += float(count) / float(total)
    return tuple(target), True


def _opponent_hand_target(
    row: Mapping[str, Any],
) -> tuple[tuple[float, ...], tuple[bool, ...], bool]:
    target = [0.0] * DEFAULT_NUM_CARD_IDS
    candidate_mask = [False] * DEFAULT_NUM_CARD_IDS
    hand_ids = [
        card_id
        for card_id in _int_list(row, "godview_opp_hand_ids")
        if 1 <= card_id <= DEFAULT_NUM_CARD_IDS
    ]
    unseen_counts = _opponent_unseen_counts(row)
    for card_id, count in unseen_counts.items():
        if count > 0 and 1 <= card_id <= DEFAULT_NUM_CARD_IDS:
            candidate_mask[card_id - 1] = True

    if not _has_usable_godview_hand(row, hand_ids):
        return tuple(target), tuple(candidate_mask), False

    hand_counts = Counter(hand_ids)
    total = sum(hand_counts.values())
    if total <= 0:
        return tuple(target), tuple(candidate_mask), False
    for card_id, count in hand_counts.items():
        target[card_id - 1] += float(count) / float(total)
        candidate_mask[card_id - 1] = True
    return tuple(target), tuple(candidate_mask), True


def _has_usable_godview_hand(row: Mapping[str, Any], hand_ids: Sequence[int]) -> bool:
    if not bool(row.get("godview_opp_hand_available", False)):
        return False
    if not bool(row.get("godview_opp_hand_count_match", True)):
        return False
    your_index = _int_value(row.get("your_index"))
    opponent_index = 1 - your_index
    expected_count = _int_value(row.get(f"player{opponent_index}_hand_count"))
    return expected_count > 0 and expected_count == len(hand_ids)


def _opponent_unseen_counts(row: Mapping[str, Any]) -> Counter[int]:
    your_index = _int_value(row["your_index"])
    opponent_index = 1 - your_index
    unseen = Counter(_int_list(row, "opponent_deck_ids"))
    unseen.subtract(_visible_player_counts(row, opponent_index))
    unseen.subtract(
        dict(
            zip(
                _int_list(row, "opp_revealed_ids"),
                _int_list(row, "opp_revealed_counts"),
                strict=True,
            )
        )
    )
    return Counter({card_id: count for card_id, count in unseen.items() if count > 0})


def _visible_player_counts(row: Mapping[str, Any], player_index: int) -> Counter[int]:
    prefix = f"player{player_index}"
    counts: Counter[int] = Counter()
    for field_name in (
        f"{prefix}_active_ids",
        f"{prefix}_bench_ids",
        f"{prefix}_discard_ids",
        f"{prefix}_prize_ids",
    ):
        counts.update(card_id for card_id in _int_list(row, field_name) if card_id > 0)
    return counts


def _chosen_effect_target(row: Mapping[str, Any]) -> tuple[tuple[float, ...], bool]:
    values = tuple(float(value) for value in _sequence(row["chosen_effect_features"]))
    if len(values) != DYNAMIC_EFFECT_FEATURE_SIZE:
        raise ValueError("chosen_effect_features has invalid width")
    return values, bool(row["chosen_effect_mask"])


def _value_target(row: Mapping[str, Any]) -> float:
    reward = _optional_float(row.get("reward"))
    if reward is not None:
        if reward > 0.0:
            return 1.0
        if reward < 0.0:
            return -1.0
        return 0.0

    result = str(row.get("result", "")).lower()
    if result == "win":
        return 1.0
    if result == "loss":
        return -1.0
    return 0.0


def _sample_weight(row: Mapping[str, Any], config: SampleWeightConfig) -> float:
    if not config.enabled:
        return 1.0
    if config.sample_weight_column:
        raw_weight = _optional_float(row.get(config.sample_weight_column))
        if raw_weight is not None:
            return _clamp(raw_weight, config.min_weight, config.max_weight)

    reward = _optional_float(row.get("reward"))
    if reward is not None:
        if reward > 0.0:
            return _clamp(config.win_weight, config.min_weight, config.max_weight)
        if reward < 0.0:
            return _clamp(config.loss_weight, config.min_weight, config.max_weight)
        return _clamp(config.draw_weight, config.min_weight, config.max_weight)

    result = str(row.get("result", "")).lower()
    if result == "win":
        weight = config.win_weight
    elif result == "loss":
        weight = config.loss_weight
    elif result == "draw":
        weight = config.draw_weight
    else:
        weight = config.other_weight
    return _clamp(weight, config.min_weight, config.max_weight)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, float(value)))


def _action_in_bounds(action: Sequence[int], option_count: int) -> bool:
    return len(set(action)) == len(action) and all(0 <= index < option_count for index in action)


def _split_seed(split: SplitName) -> int:
    return {"train": 17, "validation": 31, "all": 43}[split]


def _int_list(row: Mapping[str, Any], name: str) -> list[int]:
    return [int(value) for value in _sequence(row.get(name)) if value is not None]


def _optional_int_list(row: Mapping[str, Any], name: str) -> list[int | None]:
    return [None if value is None else int(value) for value in _sequence(row.get(name))]


def _bool_list(row: Mapping[str, Any], name: str) -> list[bool]:
    return [bool(value) for value in _sequence(row.get(name))]


def _nested_int_lists(row: Mapping[str, Any], name: str) -> list[list[int]]:
    return [
        [int(item) for item in _sequence(value) if item is not None]
        for value in _sequence(row.get(name))
    ]


def _list_get(values: Sequence[int], index: int, default: int = 0) -> int:
    return int(values[index]) if index < len(values) else default


def _bool_list_get(values: Sequence[bool], index: int) -> bool:
    return bool(values[index]) if index < len(values) else False


def _nested_list_get(values: Sequence[Sequence[int]], index: int) -> Sequence[int]:
    return values[index] if index < len(values) else ()


def _sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return ()


def _int_value(value: Any, default: int = 0) -> int:
    return int(value) if value is not None else default


def _optional_float(value: Any) -> float | None:
    return float(value) if value is not None else None
