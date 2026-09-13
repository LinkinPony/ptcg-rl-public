"""Streaming source selection and root reconstruction for sibling audits."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import pyarrow.parquet as pq

from ptcg_rl.actions.selection import is_legal_action, normalize_action_order
from ptcg_rl.context import GameContext, context_features_from_row
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.evaluation.search_identity import file_sha256
from ptcg_rl.evaluation.teacher_sibling_config import TeacherSiblingDataConfig


class CandidatePolicy(Protocol):
    """Minimal current-policy surface used to freeze root candidates."""

    def select_action(self, observation: Any) -> tuple[int, ...]:
        """Return the current greedy complete action."""

    def rank_actions(
        self,
        observation: Any,
        *,
        top_k: int,
    ) -> tuple[tuple[int, ...], ...]:
        """Return current complete actions in descending policy order."""

    def action_priors(
        self,
        observation: Any,
        actions: Sequence[tuple[int, ...]],
    ) -> Mapping[tuple[int, ...], float]:
        """Return normalized complete-action priors."""


@dataclass(frozen=True)
class SampledTeacherRoot:
    """One bounded-memory validation row with replayable source identity."""

    source_shard: Path
    source_row_index: int
    row: Mapping[str, Any]

    @property
    def root_id(self) -> str:
        """Return the stable episode/step/seat identity."""
        return ":".join(
            (
                str(int(self.row.get("episode_id") or 0)),
                str(int(self.row.get("step_index") or 0)),
                str(int(self.row.get("player_index") or 0)),
            )
        )


@dataclass(frozen=True)
class FrozenCandidates:
    """Teacher plus current-policy complete actions and provenance."""

    actions: tuple[tuple[int, ...], ...]
    sources: tuple[tuple[str, ...], ...]
    current_greedy: tuple[int, ...]
    priors: tuple[float, ...]


def sample_teacher_validation_roots(
    config: TeacherSiblingDataConfig,
    *,
    reservoir_size: int,
    seed: int,
) -> tuple[tuple[SampledTeacherRoot, ...], dict[str, Any]]:
    """Reservoir-sample eligible rows while streaming every Parquet shard."""
    manifest_path = records.repo_path(config.manifest_path)
    actual_manifest_sha256 = file_sha256(manifest_path)
    if (
        config.expected_manifest_sha256 is not None
        and actual_manifest_sha256 != config.expected_manifest_sha256
    ):
        raise ValueError("teacher manifest SHA-256 does not match frozen config")
    shard_paths = _manifest_shards(manifest_path)
    rng = random.Random(seed)
    reservoir: list[SampledTeacherRoot] = []
    scanned_rows = 0
    split_rows = 0
    behavior_rows = 0
    deck_rows = 0
    eligible_rows = 0
    for shard_path in shard_paths:
        parquet_file = pq.ParquetFile(shard_path)
        shard_row_index = 0
        for batch in parquet_file.iter_batches(batch_size=config.read_batch_size):
            for row in batch.to_pylist():
                current_index = shard_row_index
                shard_row_index += 1
                scanned_rows += 1
                if not isinstance(row, Mapping):
                    continue
                if str(row.get(config.split_column, "")) != config.split_value:
                    continue
                split_rows += 1
                if str(row.get("behavior_kind", "")) != config.expected_behavior_kind:
                    continue
                behavior_rows += 1
                if str(row.get("deck_signature", "")) != config.expected_deck_signature:
                    continue
                deck_rows += 1
                if not _engine_eligible(row):
                    continue
                eligible_rows += 1
                sampled = SampledTeacherRoot(
                    source_shard=shard_path,
                    source_row_index=current_index,
                    row=dict(row),
                )
                if len(reservoir) < reservoir_size:
                    reservoir.append(sampled)
                    continue
                replacement = rng.randrange(eligible_rows)
                if replacement < reservoir_size:
                    reservoir[replacement] = sampled
    reservoir.sort(
        key=lambda item: (
            int(item.row.get("episode_id") or 0),
            int(item.row.get("step_index") or 0),
            int(item.row.get("player_index") or 0),
        )
    )
    diagnostics = {
        "manifest_path": records.display_path(manifest_path),
        "manifest_sha256": actual_manifest_sha256,
        "shards": [records.display_path(path) for path in shard_paths],
        "scanned_rows": scanned_rows,
        "split_rows": split_rows,
        "behavior_rows": behavior_rows,
        "deck_rows": deck_rows,
        "eligible_rows": eligible_rows,
        "reservoir_rows": len(reservoir),
        "reservoir_size": reservoir_size,
        "seed": seed,
    }
    return tuple(reservoir), diagnostics


def build_frozen_candidates(
    observation: Any,
    *,
    teacher_action: Sequence[int],
    current_policy: CandidatePolicy,
    top_k: int,
) -> FrozenCandidates:
    """Freeze legal teacher/current-greedy/current-top-K action siblings."""
    select = _field(observation, "select")
    if select is None:
        return FrozenCandidates((), (), (), ())
    by_action: dict[tuple[int, ...], list[str]] = {}

    def add(action: Sequence[int], source: str) -> None:
        normalized = normalize_action_order(select, action)
        if not is_legal_action(select, normalized):
            return
        sources = by_action.setdefault(normalized, [])
        if source not in sources:
            sources.append(source)

    add(teacher_action, "public_teacher")
    current_greedy = normalize_action_order(
        select,
        current_policy.select_action(observation),
    )
    add(current_greedy, "current_greedy")
    for action in current_policy.rank_actions(observation, top_k=top_k):
        add(action, "current_policy_top_k")
    actions = tuple(by_action)
    priors_by_action = current_policy.action_priors(observation, actions)
    return FrozenCandidates(
        actions=actions,
        sources=tuple(tuple(by_action[action]) for action in actions),
        current_greedy=current_greedy,
        priors=tuple(float(priors_by_action.get(action, 0.0)) for action in actions),
    )


def game_context_from_step_row(
    row: Mapping[str, Any],
    observation: Any,
    *,
    your_deck: Sequence[int],
) -> GameContext:
    """Seed a branchable public context from compact historical row features."""
    player_index = int(row.get("your_index") or row.get("player_index") or 0)
    features = context_features_from_row(row)
    own_history = list(features.history_counts[:4])
    opponent_history = list(features.history_counts[4:])
    history_by_player = (
        [own_history, opponent_history]
        if player_index == 0
        else [opponent_history, own_history]
    )
    context = GameContext(
        player_index=player_index,
        own_deck_counts=Counter(int(card_id) for card_id in your_deck),
        opponent_revealed_no_serial_counts=Counter(
            {
                item.card_id: item.count
                for item in features.opponent_revealed
            }
        ),
        history_by_player=history_by_player,
        last_attack_by_serial=features.last_attack_by_serial(),
    )
    context.update(observation)
    return context


def deterministic_root_seed(seed: int, root_id: str) -> int:
    """Derive a stable world-sampling seed independent of skipped rows."""
    digest = hashlib.blake2b(
        f"{seed}:{root_id}".encode(),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def campaign_fingerprint(identity: Mapping[str, Any]) -> str:
    """Hash the complete resolved input identity into a compact campaign key."""
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _manifest_shards(manifest_path: Path) -> tuple[Path, ...]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_shards = manifest.get("shards")
    if not isinstance(raw_shards, Sequence):
        raise ValueError(f"teacher manifest has no shard list: {manifest_path}")
    shard_paths: list[Path] = []
    for shard in raw_shards:
        if not isinstance(shard, Mapping):
            continue
        raw_path = shard.get("path")
        if isinstance(raw_path, str) and raw_path:
            shard_paths.append(records.repo_path(Path(raw_path)))
    if not shard_paths:
        raise ValueError(f"teacher manifest contains no Parquet shards: {manifest_path}")
    missing = [path for path in shard_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"teacher Parquet shard is missing: {missing[0]}")
    return tuple(shard_paths)


def _engine_eligible(row: Mapping[str, Any]) -> bool:
    action = _sequence(row.get("action"))
    return bool(
        action
        and not bool(row.get("is_forced", False))
        and str(row.get("search_begin_input") or "")
        and int(row.get("select_option_count") or 0) > 1
    )


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


__all__ = [
    "CandidatePolicy",
    "FrozenCandidates",
    "SampledTeacherRoot",
    "build_frozen_candidates",
    "campaign_fingerprint",
    "deterministic_root_seed",
    "game_context_from_step_row",
    "sample_teacher_validation_roots",
]
