"""Tensor-ready behavior-cloning batch cache shards."""

from __future__ import annotations

import json
import random
import shutil
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.model.policy import OptionBatch
from ptcg_rl.model.state_encoder import StateBatch
from ptcg_rl.training.bc_dataset import (
    BCBatch,
    KaggleStepDataConfig,
    SplitName,
    iter_bc_batches,
    pin_bc_batch_memory,
)

CACHE_FORMAT_VERSION = 1
CACHE_MANIFEST_NAME = "manifest.json"


def write_bc_tensor_cache(
    config: KaggleStepDataConfig,
    *,
    output_dir: Path,
    seed: int,
    epoch: int = 0,
    max_train_batches: int | None = None,
    max_validation_batches: int | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write tensor-ready batch shards for the configured BC data source."""
    resolved_output_dir = deck_records.repo_path(output_dir)
    manifest_path = resolved_output_dir / CACHE_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"tensor cache already exists: {manifest_path}")
    if resolved_output_dir.exists() and overwrite:
        shutil.rmtree(resolved_output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    source_config = config.model_copy(update={"tensor_cache_dir": None})
    split_limits = {
        "train": max_train_batches,
        "validation": max_validation_batches,
    }
    manifest: dict[str, Any] = {
        "format_version": CACHE_FORMAT_VERSION,
        "source": source_config.model_dump(mode="json"),
        "seed": seed,
        "epoch": epoch,
        "splits": {},
    }
    for split, max_batches in split_limits.items():
        split_manifest = _write_split(
            source_config,
            output_dir=resolved_output_dir,
            split=cast(SplitName, split),
            seed=seed,
            epoch=epoch,
            max_batches=max_batches,
        )
        manifest["splits"][split] = split_manifest

    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def iter_bc_tensor_cache_batches(
    config: KaggleStepDataConfig,
    *,
    split: SplitName,
    epoch: int,
    seed: int,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
) -> Iterator[BCBatch]:
    """Yield BC batches from a tensor-ready cache directory."""
    cache_dir = _cache_dir(config)
    manifest = _read_manifest(cache_dir)
    batch_entries = _split_batch_entries(manifest, split)
    if split == "train" and config.shuffle_shards:
        rng = random.Random(seed + epoch * 104_729 + _split_seed(split))
        batch_entries = list(batch_entries)
        rng.shuffle(batch_entries)
    for yielded, entry in enumerate(batch_entries, start=1):
        batch_path = cache_dir / _entry_path(entry)
        batch = load_bc_tensor_batch(batch_path, device=device)
        if config.pin_memory and device is None and torch.cuda.is_available():
            batch = pin_bc_batch_memory(batch)
        yield batch
        if max_batches is not None and yielded >= max_batches:
            return


def load_bc_tensor_batch(
    path: Path,
    *,
    device: torch.device | str | None = None,
) -> BCBatch:
    """Load one tensor-ready BC batch shard."""
    with np.load(path, allow_pickle=False) as data:
        actions = _actions_from_arrays(data["actions"], data["action_lengths"])
        return BCBatch(
            states=StateBatch(
                card_ids=torch.as_tensor(data["state_card_ids"], device=device),
                areas=torch.as_tensor(data["state_areas"], device=device),
                owner_roles=torch.as_tensor(data["state_owner_roles"], device=device),
                token_kinds=torch.as_tensor(data["state_token_kinds"], device=device),
                scalars=torch.as_tensor(data["state_scalars"], device=device),
                last_attack_ids=torch.as_tensor(
                    data["state_last_attack_ids"],
                    device=device,
                ),
                padding_mask=torch.as_tensor(data["state_padding_mask"], device=device),
                attachment_card_ids=_optional_cached_tensor(
                    data,
                    "state_attachment_card_ids",
                    device=device,
                ),
                attachment_parent_indices=_optional_cached_tensor(
                    data,
                    "state_attachment_parent_indices",
                    device=device,
                ),
                attachment_kinds=_optional_cached_tensor(
                    data,
                    "state_attachment_kinds",
                    device=device,
                ),
                entity_slots=_optional_cached_tensor(
                    data,
                    "state_entity_slots",
                    device=device,
                ),
            ),
            options=OptionBatch(
                option_types=torch.as_tensor(data["option_types"], device=device),
                contexts=torch.as_tensor(data["option_contexts"], device=device),
                entity_slots=torch.as_tensor(data["option_entity_slots"], device=device),
                entity_slot_mask=torch.as_tensor(
                    data["option_entity_slot_mask"],
                    device=device,
                ),
                attack_ids=torch.as_tensor(data["option_attack_ids"], device=device),
                card_ids=torch.as_tensor(data["option_card_ids"], device=device),
                scalars=torch.as_tensor(data["option_scalars"], device=device),
                dynamic_effect_features=torch.as_tensor(
                    data["option_dynamic_effect_features"],
                    device=device,
                ),
                dynamic_effect_masks=torch.as_tensor(
                    data["option_dynamic_effect_masks"],
                    device=device,
                ),
                valid_options=torch.as_tensor(data["option_valid_options"], device=device),
                min_counts=torch.as_tensor(data["option_min_counts"], device=device),
                max_counts=torch.as_tensor(data["option_max_counts"], device=device),
            ),
            select_types=tuple(int(value) for value in data["select_types"]),
            select_contexts=tuple(int(value) for value in data["select_contexts"]),
            actions=actions,
            value_targets=torch.as_tensor(data["value_targets"], device=device),
            prize_diff_targets=torch.as_tensor(
                data["prize_diff_targets"],
                device=device,
            ),
            prize_diff_mask=torch.as_tensor(data["prize_diff_mask"], device=device),
            opponent_card_targets=torch.as_tensor(
                data["opponent_card_targets"],
                device=device,
            ),
            opponent_card_target_mask=torch.as_tensor(
                data["opponent_card_target_mask"],
                device=device,
            ),
            opponent_hand_targets=torch.as_tensor(
                data["opponent_hand_targets"],
                device=device,
            ),
            opponent_hand_candidate_mask=torch.as_tensor(
                data["opponent_hand_candidate_mask"],
                device=device,
            ),
            opponent_hand_target_mask=torch.as_tensor(
                data["opponent_hand_target_mask"],
                device=device,
            ),
            chosen_effect_targets=torch.as_tensor(
                data["chosen_effect_targets"],
                device=device,
            ),
            chosen_effect_mask=torch.as_tensor(data["chosen_effect_mask"], device=device),
            sample_weights=torch.as_tensor(data["sample_weights"], device=device),
        )


def save_bc_tensor_batch(path: Path, batch: BCBatch) -> None:
    """Persist one tensor-ready BC batch shard."""
    path.parent.mkdir(parents=True, exist_ok=True)
    actions, action_lengths = _actions_to_arrays(batch.actions)
    np.savez(
        path,
        state_card_ids=_cpu_array(batch.states.card_ids),
        state_areas=_cpu_array(batch.states.areas),
        state_owner_roles=_cpu_array(batch.states.owner_roles),
        state_token_kinds=_cpu_array(batch.states.token_kinds),
        state_scalars=_cpu_array(batch.states.scalars),
        state_last_attack_ids=_cpu_array(batch.states.last_attack_ids),
        state_padding_mask=_cpu_array(batch.states.padding_mask),
        state_attachment_card_ids=_state_cache_array(
            batch.states.attachment_card_ids,
            batch_size=int(batch.states.card_ids.shape[0]),
            width=1,
            dtype=np.uint16,
        ),
        state_attachment_parent_indices=_state_cache_array(
            batch.states.attachment_parent_indices,
            batch_size=int(batch.states.card_ids.shape[0]),
            width=1,
            dtype=np.uint16,
        ),
        state_attachment_kinds=_state_cache_array(
            batch.states.attachment_kinds,
            batch_size=int(batch.states.card_ids.shape[0]),
            width=1,
            dtype=np.uint8,
        ),
        state_entity_slots=_state_cache_array(
            batch.states.entity_slots,
            batch_size=int(batch.states.card_ids.shape[0]),
            width=int(batch.states.card_ids.shape[1]),
            dtype=np.uint8,
        ),
        option_types=_cpu_array(batch.options.option_types),
        option_contexts=_cpu_array(batch.options.contexts),
        option_entity_slots=_cpu_array(batch.options.entity_slots),
        option_entity_slot_mask=_cpu_array(batch.options.entity_slot_mask),
        option_attack_ids=_cpu_array(batch.options.attack_ids),
        option_card_ids=_cpu_array(batch.options.card_ids),
        option_scalars=_cpu_array(batch.options.scalars),
        option_dynamic_effect_features=_cpu_array(
            batch.options.dynamic_effect_features
        ),
        option_dynamic_effect_masks=_cpu_array(batch.options.dynamic_effect_masks),
        option_valid_options=_cpu_array(batch.options.valid_options),
        option_min_counts=_cpu_array(batch.options.min_counts),
        option_max_counts=_cpu_array(batch.options.max_counts),
        select_types=np.asarray(batch.select_types, dtype=np.int16),
        select_contexts=np.asarray(batch.select_contexts, dtype=np.int16),
        actions=actions,
        action_lengths=action_lengths,
        value_targets=_cpu_array(batch.value_targets),
        prize_diff_targets=_cpu_array(batch.prize_diff_targets),
        prize_diff_mask=_cpu_array(batch.prize_diff_mask),
        opponent_card_targets=_cpu_array(batch.opponent_card_targets),
        opponent_card_target_mask=_cpu_array(batch.opponent_card_target_mask),
        opponent_hand_targets=_cpu_array(batch.opponent_hand_targets),
        opponent_hand_candidate_mask=_cpu_array(batch.opponent_hand_candidate_mask),
        opponent_hand_target_mask=_cpu_array(batch.opponent_hand_target_mask),
        chosen_effect_targets=_cpu_array(batch.chosen_effect_targets),
        chosen_effect_mask=_cpu_array(batch.chosen_effect_mask),
        sample_weights=_cpu_array(batch.sample_weights),
    )


def _write_split(
    config: KaggleStepDataConfig,
    *,
    output_dir: Path,
    split: SplitName,
    seed: int,
    epoch: int,
    max_batches: int | None,
) -> dict[str, Any]:
    split_dir = output_dir / split
    batches: list[dict[str, Any]] = []
    samples = 0
    for batch_index, batch in enumerate(
        iter_bc_batches(
            config,
            split=split,
            epoch=epoch,
            seed=seed,
            device=None,
            max_batches=max_batches,
        )
    ):
        relative_path = Path(split) / f"batch-{batch_index:06d}.npz"
        save_bc_tensor_batch(output_dir / relative_path, batch)
        batch_samples = len(batch.actions)
        samples += batch_samples
        batches.append(
            {
                "path": relative_path.as_posix(),
                "samples": batch_samples,
            }
        )
    return {
        "batches": batches,
        "batch_count": len(batches),
        "samples": samples,
        "directory": split_dir.relative_to(output_dir).as_posix(),
    }


def _cache_dir(config: KaggleStepDataConfig) -> Path:
    if config.tensor_cache_dir is None:
        raise ValueError("tensor_cache_dir must be set")
    return deck_records.repo_path(config.tensor_cache_dir)


def _read_manifest(cache_dir: Path) -> Mapping[str, Any]:
    manifest_path = cache_dir / CACHE_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise ValueError(f"tensor cache manifest must be a mapping: {manifest_path}")
    version = manifest.get("format_version")
    if version != CACHE_FORMAT_VERSION:
        raise ValueError(
            f"unsupported tensor cache format {version}: {manifest_path}"
        )
    return manifest


def _split_batch_entries(
    manifest: Mapping[str, Any],
    split: SplitName,
) -> Sequence[Mapping[str, Any]]:
    splits = manifest.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("tensor cache manifest missing splits")
    if split == "all":
        entries: list[Mapping[str, Any]] = []
        for split_name in ("train", "validation"):
            split_manifest = splits.get(split_name)
            if isinstance(split_manifest, Mapping):
                entries.extend(_batch_entries(split_manifest, split_name))
        return entries
    split_manifest = splits.get(split)
    if not isinstance(split_manifest, Mapping):
        raise ValueError(f"tensor cache missing split: {split}")
    return _batch_entries(split_manifest, split)


def _batch_entries(
    split_manifest: Mapping[str, Any],
    split: str,
) -> Sequence[Mapping[str, Any]]:
    entries = split_manifest.get("batches")
    if not isinstance(entries, Sequence):
        raise ValueError(f"tensor cache split batches must be a list: {split}")
    return [entry for entry in entries if isinstance(entry, Mapping)]


def _entry_path(entry: Mapping[str, Any]) -> Path:
    path = entry.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("tensor cache batch entry missing path")
    return Path(path)


def _cpu_array(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _optional_cached_tensor(
    data: Any,
    name: str,
    *,
    device: torch.device | str | None,
) -> torch.Tensor | None:
    if name not in data:
        return None
    return torch.as_tensor(data[name], device=device)


def _state_cache_array(
    tensor: torch.Tensor | None,
    *,
    batch_size: int,
    width: int,
    dtype: np.dtype[Any] | type[np.generic],
) -> np.ndarray:
    if tensor is not None:
        return _cpu_array(tensor)
    return np.zeros((batch_size, width), dtype=dtype)


def _actions_to_arrays(actions: Sequence[Sequence[int]]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.asarray([len(action) for action in actions], dtype=np.int16)
    max_length = int(lengths.max(initial=0))
    values = np.full((len(actions), max_length), -1, dtype=np.int16)
    for row_index, action in enumerate(actions):
        if action:
            values[row_index, : len(action)] = action
    return values, lengths


def _actions_from_arrays(
    actions: np.ndarray,
    lengths: np.ndarray,
) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in actions[row_index, : int(length)])
        for row_index, length in enumerate(lengths)
    )


def _split_seed(split: SplitName) -> int:
    return {"train": 11, "validation": 17, "all": 23}[split]
