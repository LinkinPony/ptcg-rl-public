"""Exact replay reconstruction for selected planner profile roots."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import fields, replace
from pathlib import Path
from typing import cast

from ptcg_rl.actions.selection import is_legal_action, normalize_action_order
from ptcg_rl.agent.search.root_information_producer import (
    context_snapshot_fingerprint,
)
from ptcg_rl.context import GameContext, GameContextFeatures
from ptcg_rl.data.kaggle_steps.records import iter_replay_steps
from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_context import (
    encode_profile_context_snapshot,
    encode_profile_observation,
    profile_observation_fingerprint,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import exact_prompt_identity
from ptcg_rl.evaluation.planner_profile_corpus_types import (
    ExactProfileRoot,
    SelectedSourceRoot,
)


def reconstruct_exact_roots(
    selected: Mapping[str, SelectedSourceRoot],
    *,
    replay_root: Path,
    replay_archive_root: Path,
) -> tuple[Mapping[str, ExactProfileRoot], Mapping[str, str], Mapping[str, str]]:
    """Replay original public callbacks and capture exact online snapshots."""
    episodes: dict[tuple[str, int], list[SelectedSourceRoot]] = defaultdict(list)
    for root in selected.values():
        shape = root.locator.shape
        episodes[(shape.date, shape.episode_id)].append(root)

    exact: dict[str, ExactProfileRoot] = {}
    replay_assets: dict[str, str] = {}
    replay_archives: dict[str, str] = {}
    archived_by_date: dict[str, list[tuple[int, list[SelectedSourceRoot]]]] = (
        defaultdict(list)
    )
    for (date, episode_id), roots in sorted(episodes.items()):
        raw_path = replay_root / date / f"{episode_id}.json"
        if raw_path.is_file():
            replay_sha256 = file_sha256(raw_path)
            _capture_episode_roots(raw_path, roots=roots, output=exact)
            replay_assets[f"{date}/{episode_id}"] = replay_sha256
        else:
            archived_by_date[date].append((episode_id, roots))

    tar_binary = shutil.which("tar")
    if archived_by_date and tar_binary is None:
        raise RuntimeError("exact profile reconstruction requires the tar binary")
    for date, date_episodes in sorted(archived_by_date.items()):
        archive = replay_archive_root / f"{date}.tar.zst"
        archive_sha256 = _validated_replay_archive(archive, date=date)
        replay_archives[date] = archive_sha256
        with tempfile.TemporaryDirectory(prefix=f"planner-profile-{date}-") as raw_dir:
            raw_root = Path(raw_dir)
            members = [f"{episode_id}.json" for episode_id, _ in date_episodes]
            command = [
                cast(str, tar_binary),
                "--zstd",
                "-xf",
                str(archive),
                "-C",
                str(raw_root),
                "--",
                *members,
            ]
            completed = subprocess.run(
                command,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    "failed to extract exact profile replays: "
                    f"{completed.stderr.strip()}"
                )
            for episode_id, roots in date_episodes:
                raw_path = raw_root / f"{episode_id}.json"
                if not raw_path.is_file() or raw_path.is_symlink():
                    raise RuntimeError(
                        "profile replay archive omitted a regular member"
                    )
                replay_sha256 = file_sha256(raw_path)
                _capture_episode_roots(raw_path, roots=roots, output=exact)
                replay_assets[f"{date}/{episode_id}"] = replay_sha256
    if set(exact) != set(selected):
        raise RuntimeError("exact replay reconstruction omitted selected roots")
    return exact, replay_assets, replay_archives


def _validated_replay_archive(path: Path, *, date: str) -> str:
    """Validate one immutable daily archive against its adjacent manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"profile replay archive not found: {path}")
    manifest_path = path.with_name(f"{date}.manifest.json")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"profile replay archive manifest not found: {path}")
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping) or str(manifest.get("date")) != date:
        raise ValueError("profile replay archive manifest has another date")
    expected = manifest.get("archive_sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("profile replay archive manifest lacks SHA-256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError("profile replay archive fingerprint differs from manifest")
    return actual


def _capture_episode_roots(
    replay_path: Path,
    *,
    roots: Sequence[SelectedSourceRoot],
    output: dict[str, ExactProfileRoot],
) -> None:
    """Reproduce per-seat online context updates through every retained root."""
    roots_by_seat: dict[int, dict[int, SelectedSourceRoot]] = defaultdict(dict)
    contexts: dict[int, GameContext] = {}
    registered_decks: dict[int, tuple[int, ...]] = {}
    for root in roots:
        shape = root.locator.shape
        existing = roots_by_seat[shape.player_index].setdefault(
            shape.step_index,
            root,
        )
        if existing is not root:
            raise ValueError("profile selected two roots at one seat/step")
        context = contexts.setdefault(
            shape.player_index,
            GameContext(player_index=shape.player_index),
        )
        if context.own_deck_counts and tuple(
            sorted(context.own_deck_counts.elements())
        ) != tuple(sorted(root.own_deck)):
            raise ValueError("profile episode changes a seat's exact deck")
        context.set_own_deck(root.own_deck)

    remaining = {
        (seat, step_index)
        for seat, seat_roots in roots_by_seat.items()
        for step_index in seat_roots
    }
    pending_actions: dict[int, tuple[str, Mapping[str, object]]] = {}
    maximum_step = max(step_index for _, step_index in remaining)
    for step_index, sides in iter_replay_steps(replay_path):
        for seat, (case_id, pending_select) in tuple(pending_actions.items()):
            if seat >= len(sides):
                raise ValueError("profile replay action step omitted a player side")
            raw_action = sides[seat].get("action")
            if not isinstance(raw_action, Sequence) or isinstance(raw_action, str):
                raise ValueError("profile replay omitted an executed action")
            if any(type(index) is not int for index in raw_action):
                raise ValueError("profile replay action contains a non-integer index")
            executed_action = normalize_action_order(pending_select, raw_action)
            if not is_legal_action(pending_select, executed_action):
                raise ValueError("profile replay executed an illegal action")
            output[case_id] = replace(
                output[case_id],
                executed_action=executed_action,
            )
            del pending_actions[seat]
        for seat in roots_by_seat:
            if seat >= len(sides):
                raise ValueError("profile replay step omitted a player side")
            side = sides[seat]
            action = side.get("action")
            if (
                isinstance(action, list)
                and len(action) == 60
                and all(isinstance(card_id, int) and card_id > 0 for card_id in action)
            ):
                deck = tuple(sorted(int(card_id) for card_id in action))
                prior = registered_decks.setdefault(seat, deck)
                if prior != deck:
                    raise ValueError("profile replay changes a registered deck")
            if not _has_active_select(side):
                continue
            observation = side.get("observation")
            if not isinstance(observation, Mapping):
                raise ValueError("active profile replay side has no observation")
            features = contexts[seat].update(observation)
            target_root = roots_by_seat[seat].get(step_index)
            if target_root is None:
                continue
            source_context_drift_fields = _validate_exact_root_observation(
                observation,
                root=target_root,
                features=features,
            )
            augmented = dict(observation)
            augmented["gameContext"] = features.as_observation_dict()
            snapshot = contexts[seat].snapshot()
            observation_payload = encode_profile_observation(augmented)
            snapshot_payload = encode_profile_context_snapshot(snapshot)
            case_id = target_root.locator.case_id
            output[case_id] = ExactProfileRoot(
                executed_action=(),
                observation_payload=observation_payload,
                observation_fingerprint=profile_observation_fingerprint(
                    observation_payload
                ),
                context_snapshot_payload=snapshot_payload,
                context_snapshot_fingerprint=context_snapshot_fingerprint(snapshot),
                replay_sha256=file_sha256(replay_path),
                source_context_match=not source_context_drift_fields,
                source_context_drift_fields=source_context_drift_fields,
            )
            prompt_select = observation.get("select")
            if not isinstance(prompt_select, Mapping):
                raise AssertionError("active replay root omitted its select prompt")
            pending_actions[seat] = (case_id, prompt_select)
            remaining.remove((seat, step_index))
        if not remaining and not pending_actions and step_index >= maximum_step:
            break
    if remaining:
        raise ValueError(f"profile replay omitted selected active roots: {remaining}")
    if pending_actions:
        raise ValueError("profile replay ended before a selected root action")
    for seat, seat_roots in roots_by_seat.items():
        expected = tuple(sorted(next(iter(seat_roots.values())).own_deck))
        if registered_decks.get(seat) != expected:
            raise ValueError("profile replay registration differs from source deck")


def _has_active_select(side: Mapping[str, object]) -> bool:
    if str(side.get("status", "")) != "ACTIVE":
        return False
    observation = side.get("observation")
    return isinstance(observation, Mapping) and isinstance(
        observation.get("select"),
        Mapping,
    )


def _validate_exact_root_observation(
    observation: Mapping[str, object],
    *,
    root: SelectedSourceRoot,
    features: GameContextFeatures,
) -> tuple[str, ...]:
    source_context_drift_fields = tuple(
        field.name
        for field in fields(GameContextFeatures)
        if getattr(features, field.name) != getattr(root.context_features, field.name)
    )
    state_token = observation.get("search_begin_input")
    if state_token != root.search_begin_input:
        raise ValueError("exact replay engine state differs from compact source")
    shape = root.locator.shape
    actual = exact_prompt_identity(observation)
    expected = (
        shape.player_index,
        shape.select_context,
        shape.select_min_count,
        shape.select_max_count,
        shape.select_option_count,
    )
    if actual != expected:
        raise ValueError(
            "exact replay prompt differs from compact source evidence: "
            f"case={root.locator.case_id} actual={actual} expected={expected}"
        )
    return source_context_drift_fields


__all__ = ["reconstruct_exact_roots"]
