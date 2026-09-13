"""Exact replay successors used by the integrated learner workload."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

from ptcg_rl.actions.selection import normalize_action_order
from ptcg_rl.agent.search.context import public_search_observation
from ptcg_rl.agent.search.executed_endpoint import build_executed_endpoint_leaf
from ptcg_rl.agent.search.root_information import RootInformationLeaf
from ptcg_rl.agent.search.root_information_producer import (
    root_information_belief_summary,
)
from ptcg_rl.context import GameContext, OpponentBeliefFeatureProducer
from ptcg_rl.data.kaggle_steps.records import iter_replay_steps
from ptcg_rl.engine.probe_resolution import ProbeTransition
from ptcg_rl.evaluation.consequence_parity_artifact import file_sha256
from ptcg_rl.evaluation.planner_profile_corpus_types import PlannerProfileCorpusRecord
from ptcg_rl.rl.factual import (
    FactualSuccessor,
    FactualTransitionTarget,
    build_factual_transition_target,
    factual_successor,
)


@dataclass(frozen=True, slots=True)
class ProfileLearnerTransition:
    """Actual executed transition and optional non-terminal endpoint leaf."""

    factual_target: FactualTransitionTarget
    endpoint_leaf: RootInformationLeaf | None


@dataclass(slots=True)
class _PendingTransition:
    record: PlannerProfileCorpusRecord
    root_player: int
    context: GameContext
    previous_observation: Mapping[str, Any]
    transitions: list[ProbeTransition] = field(default_factory=list)
    action_validated: bool = False


def build_profile_learner_transitions(
    records: Sequence[PlannerProfileCorpusRecord],
    *,
    replay_root: Path,
    replay_archive_root: Path,
    belief_producer: OpponentBeliefFeatureProducer,
    belief_summary_width: int,
) -> Mapping[str, ProfileLearnerTransition]:
    """Restore exact public successors without retaining whole replay datasets."""
    if not records:
        raise ValueError("profile learner transition workload is empty")
    grouped: dict[tuple[str, int], list[PlannerProfileCorpusRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.source_date, record.source_episode_id)].append(record)
    result: dict[str, ProfileLearnerTransition] = {}
    with tempfile.TemporaryDirectory(prefix="planner-profile-learner-replays-") as raw:
        extracted_root = Path(raw)
        by_date: dict[str, list[int]] = defaultdict(list)
        for date, episode_id in grouped:
            if not (replay_root / date / f"{episode_id}.json").is_file():
                by_date[date].append(episode_id)
        for date, episode_ids in sorted(by_date.items()):
            _extract_replay_members(
                replay_archive_root / f"{date}.tar.zst",
                date=date,
                episode_ids=tuple(sorted(set(episode_ids))),
                destination=extracted_root / date,
            )
        for (date, episode_id), episode_records in sorted(grouped.items()):
            replay_path = replay_root / date / f"{episode_id}.json"
            if not replay_path.is_file():
                replay_path = extracted_root / date / f"{episode_id}.json"
            _build_episode_transitions(
                replay_path,
                records=episode_records,
                belief_producer=belief_producer,
                belief_summary_width=belief_summary_width,
                output=result,
            )
    expected = {record.row_id for record in records}
    if set(result) != expected:
        missing = sorted(expected.difference(result))
        raise ValueError(f"profile learner replay omitted roots: {missing}")
    return result


def _extract_replay_members(
    archive: Path,
    *,
    date: str,
    episode_ids: Sequence[int],
    destination: Path,
) -> None:
    if not archive.is_file():
        raise FileNotFoundError(f"profile learner replay archive not found: {archive}")
    manifest_path = archive.with_name(f"{date}.manifest.json")
    with manifest_path.open("r", encoding="utf-8") as source:
        manifest = json.load(source)
    if not isinstance(manifest, Mapping) or str(manifest.get("date")) != date:
        raise ValueError("profile learner replay archive manifest has another date")
    if file_sha256(archive) != str(manifest.get("archive_sha256")):
        raise ValueError("profile learner replay archive fingerprint differs")
    tar_binary = shutil.which("tar")
    if tar_binary is None:
        raise RuntimeError("profile learner replay restoration requires tar")
    destination.mkdir(parents=True, exist_ok=False)
    members = tuple(f"{episode_id}.json" for episode_id in episode_ids)
    completed = subprocess.run(
        [
            tar_binary,
            "--zstd",
            "-xf",
            str(archive),
            "-C",
            str(destination),
            "--",
            *members,
        ],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"profile learner replay extraction failed: {completed.stderr.strip()}"
        )
    if any(
        not (destination / member).is_file() or (destination / member).is_symlink()
        for member in members
    ):
        raise RuntimeError("profile learner replay archive omitted a regular member")


def _build_episode_transitions(
    replay_path: Path,
    *,
    records: Sequence[PlannerProfileCorpusRecord],
    belief_producer: OpponentBeliefFeatureProducer,
    belief_summary_width: int,
    output: dict[str, ProfileLearnerTransition],
) -> None:
    expected_sha = {record.replay_sha256 for record in records}
    if len(expected_sha) != 1 or file_sha256(replay_path) not in expected_sha:
        raise ValueError("profile learner replay differs from corpus provenance")
    roots_by_step: dict[int, list[PlannerProfileCorpusRecord]] = defaultdict(list)
    for record in records:
        roots_by_step[record.source_step_index].append(record)
    pending: list[_PendingTransition] = []
    for step_index, sides in iter_replay_steps(replay_path):
        after_observation = _active_or_terminal_observation(sides)
        if pending and after_observation is None:
            raise ValueError("profile learner transition has no successor observation")
        if after_observation is not None:
            retained: list[_PendingTransition] = []
            for item in pending:
                if not item.action_validated:
                    _validate_executed_action(item.record, sides)
                    item.action_validated = True
                root_observation = public_search_observation(
                    after_observation,
                    perspective_player_index=item.root_player,
                )
                item.context.update(root_observation)
                item.transitions.append(
                    ProbeTransition(
                        before_observation=item.previous_observation,
                        after_observation=after_observation,
                        logs=tuple(_sequence(after_observation.get("logs", ()))),
                    )
                )
                successor = factual_successor(
                    after_observation,
                    root_player_index=item.root_player,
                )
                if successor is None:
                    item.previous_observation = after_observation
                    retained.append(item)
                    continue
                output[item.record.row_id] = _completed_transition(
                    item,
                    after_observation=after_observation,
                    successor=successor,
                    belief_producer=belief_producer,
                    belief_summary_width=belief_summary_width,
                )
            pending = retained
        for record in roots_by_step.get(step_index, ()):
            root_player = _record_root_player(record)
            pending.append(
                _PendingTransition(
                    record=record,
                    root_player=root_player,
                    context=GameContext.from_snapshot(record.context_snapshot),
                    previous_observation=record.observation,
                )
            )
    if pending:
        raise ValueError("profile learner replay ended before factual successor")


def _completed_transition(
    pending: _PendingTransition,
    *,
    after_observation: Mapping[str, Any],
    successor: FactualSuccessor,
    belief_producer: OpponentBeliefFeatureProducer,
    belief_summary_width: int,
) -> ProfileLearnerTransition:
    record = pending.record
    target = build_factual_transition_target(
        root_action=record.executed_action,
        before_observation=record.observation,
        after_observation=after_observation,
        transitions=tuple(pending.transitions),
        perspective_player=pending.root_player,
        successor=successor,
    )
    projected = public_search_observation(
        after_observation,
        perspective_player_index=pending.root_player,
    )
    context_features = pending.context.features(projected)
    context_features = belief_producer.augment(projected, context_features)
    leaf = build_executed_endpoint_leaf(
        after_observation,
        root_player=pending.root_player,
        context_features=context_features,
        actor_relation=successor.actor_relation,
        next_context=target.next_context,
        exact_effect=target.effect_features,
        belief_summary=root_information_belief_summary(
            context_features,
            width=belief_summary_width,
        ),
    )
    return ProfileLearnerTransition(factual_target=target, endpoint_leaf=leaf)


def _validate_executed_action(
    record: PlannerProfileCorpusRecord,
    sides: Sequence[Mapping[str, Any]],
) -> None:
    seat = _record_root_player(record)
    if seat >= len(sides):
        raise ValueError("profile learner replay omitted the root seat")
    raw = sides[seat].get("action")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("profile learner replay omitted the executed action")
    replay_action = normalize_action_order(
        record.observation.get("select"),
        tuple(int(value) for value in raw),
    )
    if replay_action != record.executed_action:
        raise ValueError(
            f"profile learner executed action differs from corpus: {record.row_id}"
        )


def _record_root_player(record: PlannerProfileCorpusRecord) -> int:
    """Return the corpus root seat after narrowing its optional snapshot type."""
    root_player = record.context_snapshot.player_index
    if root_player not in (0, 1):
        raise ValueError("profile learner corpus root has no valid player seat")
    return root_player


def _active_or_terminal_observation(
    sides: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    active = [
        side.get("observation")
        for side in sides
        if str(side.get("status", "")) == "ACTIVE"
        and isinstance(side.get("observation"), Mapping)
        and isinstance(side["observation"].get("select"), Mapping)
    ]
    if len(active) > 1:
        raise ValueError("profile learner replay has multiple active observations")
    if active:
        return cast(Mapping[str, Any], active[0])
    for side in sides:
        observation = side.get("observation")
        if not isinstance(observation, Mapping):
            continue
        current = observation.get("current")
        if isinstance(current, Mapping) and int(current.get("result", -1)) >= 0:
            return cast(Mapping[str, Any], observation)
    return None


def _sequence(value: Any) -> Sequence[Any]:
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()


__all__ = ["ProfileLearnerTransition", "build_profile_learner_transitions"]
