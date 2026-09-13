"""Compare a checkpoint with a public Kaggle replay pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator
from torch import Tensor

from ptcg_rl.actions.selection import (
    ENGINE_PROVEN_UNORDERED_SET_CONTEXTS,
    is_forced,
    is_legal_action,
    is_unordered_set_selection,
    normalize_action_order,
)
from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.context import (
    PUBLIC_EVENT_SCHEMA_FINGERPRINT,
    PublicEventDecisionToken,
)
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps.records import iter_replay_steps
from ptcg_rl.engine.constants import OptionType, SelectContext
from ptcg_rl.evaluation.replay_policy_temporal import (
    TemporalReplayDecision,
    audit_temporal_sequences,
)
from ptcg_rl.model.sequence.action import (
    AcceptedActionRecord,
    build_accepted_action_record,
)
from ptcg_rl.model.simple_stateless import (
    SimpleStatelessPolicyValueNet,
    resolve_simple_exact_routes,
    simple_count_first_rows,
    uses_generalist_sequence,
)
from ptcg_rl.rl.policy_inputs import (
    SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
    PolicyInputContract,
    SimpleStatelessActorRow,
    SimpleStatelessPublicInputAdapter,
    collate_simple_stateless_actor_rows,
    simple_stateless_input_contract,
)
from ptcg_rl.rl.stateless_checkpoint import load_stateless_policy_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[3]


class KaggleReplayPolicyAuditConfig(BaseModel):
    """Validated inputs for one immutable replay-to-checkpoint comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    replay_manifest_path: Path
    checkpoint_path: Path
    public_catalog_manifest_path: Path
    card_data_csv: Path
    output_dir: Path
    target_team_name: str
    target_deck_hash: str = Field(pattern=r"^[0-9a-f]{12}$")
    replay_split: Literal["all", "train", "validation", "test"] = "all"
    device: str = "cuda"
    batch_size: int = 128
    replay_chunk_bytes: int = 1 << 20
    replay_prefix_bytes: int = 65_536
    verify_replay_sha256: bool = True
    drop_forced_actions: bool = True

    @field_validator("target_team_name", "device")
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Reject ambiguous empty strings."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("text fields must be non-empty")
        return normalized

    @field_validator("batch_size", "replay_chunk_bytes", "replay_prefix_bytes")
    @classmethod
    def positive_integer(cls, value: int) -> int:
        """Reject unusable batch and streaming sizes."""
        if value <= 0:
            raise ValueError("integer limits must be positive")
        return value


@dataclass(frozen=True)
class _ReplaySource:
    episode_id: int
    replay_path: Path
    expected_sha256: str
    player_index: int
    reward: float
    opponent_team_name: str


@dataclass(frozen=True)
class _Decision:
    episode_id: int
    step_index: int
    player_index: int
    reward: float
    opponent_team_name: str
    select_context: int
    actor_row: SimpleStatelessActorRow
    teacher_action: tuple[int, ...]
    decision_index: int | None = None
    accepted_action: AcceptedActionRecord | None = None


@dataclass(frozen=True)
class _PendingDecision:
    step_index: int
    select: Mapping[str, Any]
    actor_row: SimpleStatelessActorRow | None
    event_token: PublicEventDecisionToken | None
    decision_index: int | None


@dataclass(frozen=True)
class _ModelRuntime:
    model: SimpleStatelessPolicyValueNet
    contract: PolicyInputContract
    checkpoint_version: int
    checkpoint_sha256: str
    checkpoint_model_fingerprint: str
    device: torch.device
    architecture: str
    temporal: bool
    max_context_blocks: int | None


def run(config: KaggleReplayPolicyAuditConfig) -> dict[str, Any]:
    """Audit all selected replay decisions and publish reproducible evidence."""
    output_dir = _path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _path(config.replay_manifest_path)
    manifest = _load_json(manifest_path)
    sources = _load_sources(config, manifest, manifest_path=manifest_path)
    runtime, catalog = _load_runtime(config)
    card_names = _load_card_names(_path(config.card_data_csv))

    decision_rows: list[dict[str, Any]] = []
    pending: list[_Decision] = []
    temporal_sequences: list[tuple[_Decision, ...]] = []
    counters: Counter[str] = Counter()
    for source in sources:
        replay_decisions, replay_counts = _extract_replay_decisions(
            config,
            source,
            catalog=catalog,
            contract=runtime.contract,
            temporal=runtime.temporal,
        )
        counters.update(replay_counts)
        if runtime.temporal:
            temporal_sequences.append(replay_decisions)
        else:
            for decision in replay_decisions:
                pending.append(decision)
                if len(pending) >= config.batch_size:
                    decision_rows.extend(
                        _audit_batch(
                            tuple(pending),
                            runtime=runtime,
                            card_names=card_names,
                        )
                    )
                    pending.clear()
    if runtime.temporal:

        def build_temporal_rows(
            decisions: tuple[TemporalReplayDecision, ...],
            greedy: tuple[tuple[int, ...], ...],
            evaluation: Any,
            root_values: Tensor,
            count_rows: Tensor,
        ) -> list[dict[str, Any]]:
            return _decision_rows(
                cast(tuple[_Decision, ...], decisions),
                greedy=greedy,
                evaluation=evaluation,
                root_values=root_values,
                count_rows=count_rows,
                card_names=card_names,
            )

        decision_rows.extend(
            audit_temporal_sequences(
                cast(
                    tuple[tuple[TemporalReplayDecision, ...], ...],
                    tuple(temporal_sequences),
                ),
                model=runtime.model,
                device=runtime.device,
                batch_size=config.batch_size,
                build_rows=build_temporal_rows,
            )
        )
    elif pending:
        decision_rows.extend(
            _audit_batch(
                tuple(pending),
                runtime=runtime,
                card_names=card_names,
            )
        )
    if not decision_rows:
        raise ValueError("replay audit produced no non-forced decisions")

    manifest_sha256 = _file_sha256(manifest_path)
    source_snapshot = {
        **manifest,
        "source_manifest_path": deck_records.display_path(manifest_path),
        "source_manifest_sha256": manifest_sha256,
    }
    _atomic_write_json(output_dir / "source_manifest.json", source_snapshot)
    decisions_path = output_dir / "decisions.parquet"
    _atomic_write_parquet(decisions_path, decision_rows)
    summary = _build_summary(
        config=config,
        sources=sources,
        rows=decision_rows,
        counters=counters,
        runtime=runtime,
        source_manifest_sha256=manifest_sha256,
        decisions_sha256=_file_sha256(decisions_path),
        decisions_size_bytes=decisions_path.stat().st_size,
    )
    _atomic_write_json(output_dir / "summary.json", summary)
    _atomic_write_text(output_dir / "report.md", _markdown_report(summary))
    return summary


def _load_runtime(
    config: KaggleReplayPolicyAuditConfig,
) -> tuple[_ModelRuntime, Any]:
    checkpoint_path = _path(config.checkpoint_path)
    loaded = load_stateless_policy_checkpoint(checkpoint_path)
    catalog, catalog_manifest = load_public_deck_catalog(
        _path(config.public_catalog_manifest_path)
    )
    contract = simple_stateless_input_contract(
        public_catalog_fingerprint=catalog.fingerprint,
        card_catalog_fingerprint=catalog_manifest.card_catalog_fingerprint,
        public_context_fingerprint=PUBLIC_EVENT_SCHEMA_FINGERPRINT,
        wrapper_runtime_fingerprint=SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
    )
    if contract.fingerprint != loaded.identity.input_contract_fingerprint:
        raise ValueError("checkpoint and replay input contracts differ")
    if catalog.fingerprint != loaded.identity.public_deck_catalog_fingerprint:
        raise ValueError("checkpoint and public catalog fingerprints differ")
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay audit requested but CUDA is unavailable")
    model = SimpleStatelessPolicyValueNet(
        loaded.model_config_value,
        load_static_features=False,
        initialize=False,
    )
    model.load_state_dict(loaded.model_state, strict=True)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model.to(device=device, dtype=dtype).eval()
    temporal = uses_generalist_sequence(model.config)
    if temporal and not config.drop_forced_actions:
        raise ValueError("sequence replay audit must drop forced actions")
    sequence_config = model.config.sequence
    runtime = _ModelRuntime(
        model=model,
        contract=contract,
        checkpoint_version=loaded.artifact.version,
        checkpoint_sha256=loaded.artifact.policy_sha256,
        checkpoint_model_fingerprint=(loaded.artifact.policy_model_fingerprint),
        device=device,
        architecture=model.config.architecture,
        temporal=temporal,
        max_context_blocks=(
            None if sequence_config is None else sequence_config.max_context_blocks
        ),
    )
    return runtime, catalog


def _load_sources(
    config: KaggleReplayPolicyAuditConfig,
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> tuple[_ReplaySource, ...]:
    rows = manifest.get("episodes")
    path_field = "replay_path"
    if not isinstance(rows, Sequence) or isinstance(rows, str) or not rows:
        rows = manifest.get("sides")
        path_field = "relative_path"
    if not isinstance(rows, Sequence) or isinstance(rows, str) or not rows:
        raise ValueError("replay manifest contains no episodes or acting sides")
    sources: list[_ReplaySource] = []
    seen: set[int] = set()
    for raw in rows:
        row = _mapping(raw)
        row_split = str(row.get("split", "all"))
        if config.replay_split != "all" and row_split != config.replay_split:
            continue
        episode_id = int(row["episode_id"])
        if episode_id in seen:
            raise ValueError("replay manifest contains duplicate episode IDs")
        seen.add(episode_id)
        raw_path = Path(str(row[path_field]))
        replay_path = (
            (manifest_path.parent / raw_path).resolve()
            if path_field == "relative_path" and not raw_path.is_absolute()
            else _path(raw_path)
        )
        if not replay_path.is_file():
            raise FileNotFoundError(replay_path)
        expected_sha256 = str(row["sha256"])
        if config.verify_replay_sha256 and _file_sha256(replay_path) != expected_sha256:
            raise ValueError(f"replay SHA-256 changed: {replay_path}")
        if all(key in row for key in ("player_index", "reward", "opponent_team_name")):
            player_index = int(row["player_index"])
            reward = float(row["reward"])
            opponent_team_name = str(row["opponent_team_name"])
        else:
            player_index, reward, opponent_team_name = _infer_source_metadata(
                config,
                replay_path,
            )
        sources.append(
            _ReplaySource(
                episode_id=episode_id,
                replay_path=replay_path,
                expected_sha256=expected_sha256,
                player_index=player_index,
                reward=reward,
                opponent_team_name=opponent_team_name,
            )
        )
    if not sources:
        raise ValueError(f"replay manifest has no {config.replay_split} sources")
    return tuple(sources)


def _infer_source_metadata(
    config: KaggleReplayPolicyAuditConfig,
    replay_path: Path,
) -> tuple[int, float, str]:
    """Resolve the audited seat from a generic frozen replay inventory."""
    side_rows = deck_records.fast_episode_side_rows(
        replay_path=replay_path,
        card_meta={},
        known_decks={},
        prefix_bytes=config.replay_prefix_bytes,
        include_step_count=False,
    )
    if side_rows is None or len(side_rows) != 2:
        raise ValueError(f"cannot resolve replay sides: {replay_path}")
    deck_matches = tuple(
        row
        for row in side_rows
        if str(row.get("deck_hash", "")) == config.target_deck_hash
    )
    if len(deck_matches) == 1:
        target = deck_matches[0]
    else:
        named = tuple(
            row
            for row in deck_matches
            if str(row.get("team_name", "")).casefold()
            == config.target_team_name.casefold()
        )
        if len(named) != 1:
            raise ValueError(f"cannot uniquely resolve target deck seat: {replay_path}")
        target = named[0]
    player_index = int(target["player_index"])
    opponent = side_rows[1 - player_index]
    return (
        player_index,
        float(target["reward"]),
        str(opponent["team_name"]),
    )


def _extract_replay_decisions(
    config: KaggleReplayPolicyAuditConfig,
    source: _ReplaySource,
    *,
    catalog: Any,
    contract: PolicyInputContract,
    temporal: bool,
) -> tuple[tuple[_Decision, ...], Counter[str]]:
    side_rows = deck_records.fast_episode_side_rows(
        replay_path=source.replay_path,
        card_meta={},
        known_decks={},
        prefix_bytes=config.replay_prefix_bytes,
        include_step_count=False,
    )
    if side_rows is None or len(side_rows) != 2:
        raise ValueError(f"cannot resolve replay decks: {source.replay_path}")
    metadata = {int(row["player_index"]): row for row in side_rows}
    target = metadata[source.player_index]
    if str(target["deck_hash"]) != config.target_deck_hash:
        raise ValueError(f"target deck differs in episode {source.episode_id}")
    if float(target["reward"]) != source.reward:
        raise ValueError(f"target reward differs in episode {source.episode_id}")
    own_deck = tuple(int(value) for value in cast(Sequence[int], target["deck_ids"]))
    adapter = SimpleStatelessPublicInputAdapter(
        catalog,
        contract=contract,
        player_index=source.player_index,
        own_deck=own_deck,
    )
    pending: _PendingDecision | None = None
    decisions: list[_Decision] = []
    counters: Counter[str] = Counter(replays=1)
    if str(target["team_name"]).casefold() != config.target_team_name.casefold():
        counters["replays_with_historical_team_alias"] += 1
    for step_index, sides in iter_replay_steps(
        source.replay_path,
        chunk_size=config.replay_chunk_bytes,
    ):
        counters["steps"] += 1
        if source.player_index >= len(sides):
            raise ValueError(f"replay step misses target seat: {source.episode_id}")
        side = sides[source.player_index]
        if pending is not None:
            action = _integer_action(side.get("action"))
            if action is None or not is_legal_action(pending.select, action):
                raise ValueError(
                    f"illegal replay action: {source.episode_id}:{pending.step_index}"
                )
            action = _canonical_action(pending.select, action)
            if config.drop_forced_actions and is_forced(pending.select):
                counters["forced_actions_dropped"] += 1
            else:
                actor_row = pending.actor_row
                if actor_row is None:
                    raise RuntimeError("audited replay decision has no actor row")
                accepted_action = None
                if temporal:
                    if pending.event_token is None or pending.decision_index is None:
                        raise RuntimeError(
                            "temporal replay decision has no transaction"
                        )
                    accepted_action = build_accepted_action_record(
                        state=actor_row.state,
                        options=actor_row.options,
                        action=action,
                        min_count=actor_row.min_count,
                        max_count=actor_row.max_count,
                        stop_sampled=_stop_sampled(actor_row, action),
                    )
                    committed_delta = adapter.commit_decision(pending.event_token)
                    if committed_delta != actor_row.public_event_delta:
                        raise RuntimeError("temporal event delta changed before commit")
                decisions.append(
                    _Decision(
                        episode_id=source.episode_id,
                        step_index=pending.step_index,
                        player_index=source.player_index,
                        reward=source.reward,
                        opponent_team_name=source.opponent_team_name,
                        select_context=int(pending.select.get("context", -1)),
                        actor_row=actor_row,
                        teacher_action=action,
                        decision_index=pending.decision_index,
                        accepted_action=accepted_action,
                    )
                )
                counters["decisions"] += 1
            pending = None
        if str(side.get("status", "")) != "ACTIVE":
            continue
        observation = _mapping(side.get("observation"))
        select = _mapping(observation.get("select"))
        if not select:
            continue
        context = adapter.observe(observation)
        forced = is_forced(select)
        actor_row = None
        event_token = None
        decision_index = None
        if not temporal or not forced:
            if temporal:
                event_token = adapter.prepare_decision()
                decision_index = len(decisions)
            actor_row = adapter.tensorize_observed(
                observation,
                context=context,
            )
            if (
                temporal
                and event_token is not None
                and actor_row.public_event_delta != event_token.delta
            ):
                raise RuntimeError(
                    "tensorized temporal events differ from prepared events"
                )
        pending = _PendingDecision(
            step_index=step_index,
            select=select,
            actor_row=actor_row,
            event_token=event_token,
            decision_index=decision_index,
        )
        counters["active_select_observations"] += 1
    if pending is not None:
        raise ValueError(f"replay ended with pending action: {source.episode_id}")
    return tuple(decisions), counters


def _audit_batch(
    decisions: tuple[_Decision, ...],
    *,
    runtime: _ModelRuntime,
    card_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    actor_rows = tuple(decision.actor_row for decision in decisions)
    batch = collate_simple_stateless_actor_rows(
        actor_rows,
        device=runtime.device,
        deduplicate_belief=True,
    )
    route_plan = resolve_simple_exact_routes(
        batch.deck_signatures,
        runtime.model.config,
        device=runtime.device,
    )
    autocast_enabled = runtime.device.type == "cuda"
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ),
    ):
        state = runtime.model.encode_observation_state(
            state=batch.states,
            unique_deck_card_ids=batch.unique_deck_card_ids,
            deck_counts=batch.deck_counts,
            deck_valid_mask=batch.deck_valid_mask,
            belief_summary=batch.belief_summary,
            route_plan=route_plan,
        )
        option_embeddings = runtime.model.encode_legal_options(
            state,
            batch.options,
            route_plan=route_plan,
        )
        actions = tuple(decision.teacher_action for decision in decisions)
        evaluation = runtime.model.heads.teacher_forced(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            actions,
            route_plan=route_plan,
        )
        greedy = runtime.model.heads.greedy_decode(
            state.policy,
            state.opponent_belief,
            option_embeddings,
            batch.options,
            route_plan=route_plan,
        )
        root_values = runtime.model.heads.root_value(
            state.value,
            state.opponent_belief,
            route_plan=route_plan,
        )
        count_rows = simple_count_first_rows(batch.options)
    return _decision_rows(
        decisions,
        greedy=greedy,
        evaluation=evaluation,
        root_values=root_values,
        count_rows=count_rows,
        card_names=card_names,
    )


def _decision_rows(
    decisions: tuple[_Decision, ...],
    *,
    greedy: tuple[tuple[int, ...], ...],
    evaluation: Any,
    root_values: Tensor,
    count_rows: Tensor,
    card_names: Mapping[int, str],
) -> list[dict[str, Any]]:
    action_logprobs = evaluation.action_logprobs.float().cpu().numpy()
    token_logprobs = evaluation.token_logprobs.float().cpu().numpy()
    token_entropies = evaluation.token_entropies.float().cpu().numpy()
    token_mask = evaluation.token_mask.cpu().numpy()
    count_mask = count_rows.cpu().numpy()
    count_logits = evaluation.count_logits.float().cpu()
    first_step_logits = evaluation.step_logits[0].float().cpu()
    roots = root_values.float().cpu().numpy()
    max_options = int(first_step_logits.shape[1]) - 1
    rows: list[dict[str, Any]] = []
    for index, decision in enumerate(decisions):
        action = decision.teacher_action
        predicted = greedy[index]
        teacher_legal = _encoded_action_is_legal(decision.actor_row, action)
        greedy_legal = _encoded_action_is_legal(decision.actor_row, predicted)
        if not teacher_legal or not greedy_legal:
            raise RuntimeError("audited action does not resolve to legal options")
        if bool(count_mask[index]):
            first_logits = count_logits[index]
            first_target = len(action)
            first_token_kind = "count"
        else:
            first_logits = first_step_logits[index]
            first_target = action[0] if action else max_options
            first_token_kind = "option"
        first_rank = _target_rank(first_logits, first_target)
        active_logprobs = token_logprobs[index][token_mask[index]]
        active_entropies = token_entropies[index][token_mask[index]]
        teacher_label, teacher_type, teacher_card = _action_label(
            decision.actor_row,
            action,
            card_names=card_names,
        )
        greedy_label, greedy_type, greedy_card = _action_label(
            decision.actor_row,
            predicted,
            card_names=card_names,
        )
        rows.append(
            {
                "episode_id": decision.episode_id,
                "step_index": decision.step_index,
                "player_index": decision.player_index,
                "decision_index": decision.decision_index,
                "reward": decision.reward,
                "result": "win" if decision.reward > 0 else "loss",
                "opponent_team_name": decision.opponent_team_name,
                "exact_deck_digest": decision.actor_row.own_deck.deck_digest,
                "select_context": decision.select_context,
                "select_context_name": _enum_name(
                    SelectContext,
                    decision.select_context,
                    prefix="CONTEXT",
                ),
                "option_count": len(decision.actor_row.options),
                "min_count": decision.actor_row.min_count,
                "max_count": decision.actor_row.max_count,
                "teacher_action": list(action),
                "greedy_action": list(predicted),
                "teacher_action_legal": teacher_legal,
                "greedy_action_legal": greedy_legal,
                "exact_agreement": action == predicted,
                "teacher_action_logprob": float(action_logprobs[index]),
                "teacher_action_probability": _safe_exp(action_logprobs[index]),
                "first_token_kind": first_token_kind,
                "first_token_probability": _safe_exp(active_logprobs[0]),
                "first_token_rank": first_rank,
                "first_token_top3": first_rank <= 3,
                "mean_token_entropy": float(np.mean(active_entropies)),
                "decode_tokens": int(active_logprobs.size),
                "root_value": float(roots[index]),
                "teacher_label": teacher_label,
                "teacher_option_type": teacher_type,
                "teacher_card_id": teacher_card,
                "greedy_label": greedy_label,
                "greedy_option_type": greedy_type,
                "greedy_card_id": greedy_card,
            }
        )
    return rows


def _action_label(
    actor_row: SimpleStatelessActorRow,
    action: tuple[int, ...],
    *,
    card_names: Mapping[int, str],
) -> tuple[str, str, int]:
    if not action:
        return "STOP", "STOP", 0
    labels: list[str] = []
    first_type = "UNKNOWN"
    first_card = 0
    options = actor_row.options
    for action_position, option_index in enumerate(action):
        option_type = int(options.option_types[option_index])
        type_name = _enum_name(OptionType, option_type, prefix="OPTION")
        card_id = int(options.card_ids[option_index])
        attack_id = int(options.attack_ids[option_index])
        entity_names = _entity_card_names(
            actor_row,
            option_index,
            card_names=card_names,
        )
        detail = card_names.get(card_id, "") if card_id > 0 else ""
        if not detail and entity_names:
            detail = " -> ".join(entity_names)
        if attack_id > 0:
            detail = f"{detail} attack#{attack_id}".strip()
        labels.append(f"{type_name}:{detail}" if detail else type_name)
        if action_position == 0:
            first_type = type_name
            first_card = card_id
    return " + ".join(labels), first_type, first_card


def _entity_card_names(
    actor_row: SimpleStatelessActorRow,
    option_index: int,
    *,
    card_names: Mapping[int, str],
) -> tuple[str, ...]:
    slots = actor_row.options.entity_slots[option_index]
    masks = actor_row.options.entity_slot_mask[option_index]
    names: list[str] = []
    for slot, present in zip(slots, masks, strict=True):
        if not bool(present) or int(slot) < 0:
            continue
        card_id = int(actor_row.state.card_ids[int(slot)])
        name = card_names.get(card_id)
        if name and (not names or names[-1] != name):
            names.append(name)
    return tuple(names)


def _build_summary(
    *,
    config: KaggleReplayPolicyAuditConfig,
    sources: tuple[_ReplaySource, ...],
    rows: list[dict[str, Any]],
    counters: Counter[str],
    runtime: _ModelRuntime,
    source_manifest_sha256: str,
    decisions_sha256: str,
    decisions_size_bytes: int,
) -> dict[str, Any]:
    main_rows = [
        row for row in rows if row["select_context"] == int(SelectContext.MAIN)
    ]
    deck_digests = {str(row["exact_deck_digest"]) for row in rows}
    if len(deck_digests) != 1:
        raise ValueError("replay audit crossed exact deck identities")
    deck_digest = next(iter(deck_digests))
    route = next(
        (
            item
            for item in runtime.model.config.exact_routes
            if item.deck_digest == deck_digest
        ),
        None,
    )
    if route is None:
        raise ValueError("audited exact deck has no checkpoint route")
    source_manifest = _load_json(_path(config.replay_manifest_path))
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "submission_id": _manifest_submission_id(source_manifest),
            "target_team_name": config.target_team_name,
            "legacy_replay_deck_hash": config.target_deck_hash,
            "target_deck_digest": deck_digest,
            "exact_strategy_id": route.expert_id,
            "exact_registry_fingerprint": (
                runtime.model.config.resolved_registry_sha256
            ),
            "replay_split": config.replay_split,
            "source_manifest_path": deck_records.display_path(
                _path(config.replay_manifest_path)
            ),
            "source_manifest_sha256": source_manifest_sha256,
            "checkpoint_path": deck_records.display_path(_path(config.checkpoint_path)),
            "checkpoint_version": runtime.checkpoint_version,
            "checkpoint_sha256": runtime.checkpoint_sha256,
            "checkpoint_model_fingerprint": (runtime.checkpoint_model_fingerprint),
            "model_architecture": runtime.architecture,
            "temporal_history_replayed": runtime.temporal,
            "max_context_blocks": runtime.max_context_blocks,
            "input_contract_fingerprint": runtime.contract.fingerprint,
            "public_catalog_manifest_path": deck_records.display_path(
                _path(config.public_catalog_manifest_path)
            ),
            "device": str(runtime.device),
        },
        "sample": {
            "episodes": len(sources),
            "wins": sum(source.reward > 0 for source in sources),
            "losses": sum(source.reward < 0 for source in sources),
            "unique_opponents": len({source.opponent_team_name for source in sources}),
            "replay_bytes": sum(
                source.replay_path.stat().st_size for source in sources
            ),
            "extraction_counters": dict(sorted(counters.items())),
        },
        "overall": _metric_summary(rows),
        "main_phase": _metric_summary(main_rows),
        "by_result": _group_summaries(rows, "result"),
        "main_by_result": _group_summaries(main_rows, "result"),
        "by_context": _group_summaries(rows, "select_context_name", minimum=20),
        "main_by_teacher_type": _group_summaries(
            main_rows,
            "teacher_option_type",
            minimum=10,
        ),
        "main_by_teacher_action": _group_summaries(
            main_rows,
            "teacher_label",
            minimum=10,
        ),
        "main_confusions": _confusion_rows(main_rows),
        "main_type_confusions": _type_confusion_rows(main_rows),
        "lowest_probability_teacher_actions": _lowest_probability_actions(rows),
        "outputs": {
            "decisions": deck_records.display_path(
                _path(config.output_dir) / "decisions.parquet"
            ),
            "decisions_sha256": decisions_sha256,
            "decisions_size_bytes": decisions_size_bytes,
            "summary": deck_records.display_path(
                _path(config.output_dir) / "summary.json"
            ),
            "report": deck_records.display_path(_path(config.output_dir) / "report.md"),
        },
    }


def _metric_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"decisions": 0}
    probabilities = np.asarray(
        [float(row["teacher_action_probability"]) for row in rows],
        dtype=np.float64,
    )
    first_probabilities = np.asarray(
        [float(row["first_token_probability"]) for row in rows],
        dtype=np.float64,
    )
    roots = np.asarray([float(row["root_value"]) for row in rows], dtype=np.float64)
    outcomes = np.asarray([float(row["reward"]) for row in rows], dtype=np.float64)
    return {
        "decisions": len(rows),
        "exact_agreement": float(
            np.mean([bool(row["exact_agreement"]) for row in rows])
        ),
        "teacher_action_legal": float(
            np.mean([bool(row["teacher_action_legal"]) for row in rows])
        ),
        "greedy_action_legal": float(
            np.mean([bool(row["greedy_action_legal"]) for row in rows])
        ),
        "first_token_top1": float(
            np.mean([int(row["first_token_rank"]) == 1 for row in rows])
        ),
        "first_token_top3": float(
            np.mean([bool(row["first_token_top3"]) for row in rows])
        ),
        "teacher_action_probability_median": float(np.median(probabilities)),
        "teacher_action_probability_mean": float(np.mean(probabilities)),
        "first_token_probability_median": float(np.median(first_probabilities)),
        "teacher_action_nll_mean": float(
            -np.mean([float(row["teacher_action_logprob"]) for row in rows])
        ),
        "mean_token_entropy": float(
            np.mean([float(row["mean_token_entropy"]) for row in rows])
        ),
        "root_value_mean": float(np.mean(roots)),
        "root_value_mae": float(np.mean(np.abs(roots - outcomes))),
    }


def _group_summaries(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    *,
    minimum: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    result = [
        {"group": key, **_metric_summary(group)}
        for key, group in grouped.items()
        if len(group) >= minimum
    ]
    result.sort(key=lambda row: (-int(row["decisions"]), str(row["group"])))
    return result


def _confusion_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(
        (str(row["teacher_label"]), str(row["greedy_label"]))
        for row in rows
        if not bool(row["exact_agreement"])
    )
    return [
        {"teacher": teacher, "greedy": greedy, "decisions": count}
        for (teacher, greedy), count in counts.most_common(30)
    ]


def _type_confusion_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    counts = Counter(
        (str(row["teacher_option_type"]), str(row["greedy_option_type"]))
        for row in rows
    )
    return [
        {"teacher": teacher, "greedy": greedy, "decisions": count}
        for (teacher, greedy), count in counts.most_common()
    ]


def _lowest_probability_actions(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["teacher_label"])].append(row)
    summaries = []
    for label, group in grouped.items():
        if len(group) < 10:
            continue
        metric = _metric_summary(group)
        summaries.append({"teacher": label, **metric})
    summaries.sort(
        key=lambda row: (
            float(row["teacher_action_probability_median"]),
            -int(row["decisions"]),
        )
    )
    return summaries[:30]


def _markdown_report(summary: Mapping[str, Any]) -> str:
    inputs = _mapping(summary["inputs"])
    sample = _mapping(summary["sample"])
    overall = _mapping(summary["overall"])
    main = _mapping(summary["main_phase"])
    lines = [
        "# Kaggle replay policy audit",
        "",
        "## Evidence snapshot",
        "",
        f"- Submission: `{inputs['submission_id']}` / `{inputs['target_team_name']}`",
        f"- Exact deck: `{inputs['target_deck_digest']}` "
        f"(replay hash `{inputs['legacy_replay_deck_hash']}`)",
        f"- Exact strategy: `{inputs['exact_strategy_id']}`",
        f"- Checkpoint: v{inputs['checkpoint_version']} / `{inputs['checkpoint_sha256']}`",
        f"- Runtime: `{inputs['model_architecture']}`; temporal history replayed: "
        f"`{inputs['temporal_history_replayed']}`",
        f"- Replays: {sample['episodes']} ({sample['wins']} wins, {sample['losses']} losses)",
        f"- Decisions: {overall['decisions']} non-forced public decisions",
        "",
        "## Headline metrics",
        "",
        "| Scope | Decisions | Exact greedy agreement | First-token top-3 | Median teacher action probability |",
        "|---|---:|---:|---:|---:|",
        _metric_table_row("All", overall),
        _metric_table_row("Main phase", main),
        "",
        "These are matched-state imitation diagnostics, not counterfactual win-rate "
        "estimates. A disagreement does not by itself prove that either action is "
        "wrong.",
        "Every teacher action was checked against the replay's engine legal options; "
        "checkpoint greedy actions are decoded only from those same options.",
        "",
        "## Main-phase teacher actions",
        "",
        "| Teacher action | Decisions | Agreement | First-token top-3 | Median probability |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in cast(Sequence[Mapping[str, Any]], summary["main_by_teacher_action"]):
        lines.append(
            f"| {row['group']} | {row['decisions']} | "
            f"{_percent(row['exact_agreement'])} | "
            f"{_percent(row['first_token_top3'])} | "
            f"{_percent(row['teacher_action_probability_median'])} |"
        )
    lines.extend(
        [
            "",
            "## Largest main-phase disagreements",
            "",
            "| Teacher | Checkpoint greedy | Decisions |",
            "|---|---|---:|",
        ]
    )
    for row in cast(Sequence[Mapping[str, Any]], summary["main_confusions"]):
        lines.append(f"| {row['teacher']} | {row['greedy']} | {row['decisions']} |")
    lines.extend(
        [
            "",
            "## Reproduction identities",
            "",
            f"- Source manifest SHA-256: `{inputs['source_manifest_sha256']}`",
            f"- Checkpoint model fingerprint: `{inputs['checkpoint_model_fingerprint']}`",
            f"- Input contract fingerprint: `{inputs['input_contract_fingerprint']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _metric_table_row(label: str, row: Mapping[str, Any]) -> str:
    return (
        f"| {label} | {row['decisions']} | {_percent(row['exact_agreement'])} | "
        f"{_percent(row['first_token_top3'])} | "
        f"{_percent(row['teacher_action_probability_median'])} |"
    )


def _percent(value: Any) -> str:
    return f"{100.0 * float(value):.1f}%"


def _target_rank(logits: Tensor, target: int) -> int:
    target_value = logits[target]
    return int(torch.sum(logits > target_value).item()) + 1


def _safe_exp(value: float | np.floating[Any]) -> float:
    return float(math.exp(max(float(value), -745.0)))


def _canonical_action(
    select: Mapping[str, Any],
    action: tuple[int, ...],
) -> tuple[int, ...]:
    options = select.get("option")
    option_count = (
        len(options)
        if isinstance(options, Sequence) and not isinstance(options, str)
        else 0
    )
    minimum = min(option_count, max(0, int(select.get("minCount", 0))))
    maximum = min(
        option_count,
        max(minimum, int(select.get("maxCount", option_count))),
    )
    if is_unordered_set_selection(
        context=int(select.get("context", -1)),
        min_count=minimum,
        max_count=maximum,
    ):
        return tuple(sorted(action))
    return normalize_action_order(select, action)


def _encoded_action_is_legal(
    row: SimpleStatelessActorRow,
    action: tuple[int, ...],
) -> bool:
    """Validate an action against the actor-visible encoded option domain."""
    return (
        row.min_count <= len(action) <= row.max_count
        and len(action) == len(set(action))
        and all(0 <= index < len(row.options) for index in action)
    )


def _stop_sampled(
    row: SimpleStatelessActorRow,
    action: tuple[int, ...],
) -> bool:
    """Mirror the production complete-action STOP semantics."""
    unordered = row.min_count < row.max_count and any(
        int(context) in ENGINE_PROVEN_UNORDERED_SET_CONTEXTS
        for context in row.options.contexts
    )
    return not unordered and len(action) < row.max_count


def _integer_action(value: Any) -> tuple[int, ...] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str)
        or not all(type(item) is int for item in value)
    ):
        return None
    return tuple(int(item) for item in value)


def _load_card_names(path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            card_id = int(row["Card ID"])
            name = str(row["Card Name"]).strip()
            previous = names.setdefault(card_id, name)
            if previous != name:
                raise ValueError(f"card ID {card_id} has conflicting names")
    return names


def _enum_name(enum_type: Any, value: int, *, prefix: str) -> str:
    try:
        return str(enum_type(value).name)
    except ValueError:
        return f"{prefix}_{value}"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _path(value: Path) -> Path:
    return value.resolve() if value.is_absolute() else (_REPO_ROOT / value).resolve()


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _manifest_submission_id(manifest: Mapping[str, Any]) -> Any:
    """Read submission identity from audit or generic replay manifests."""
    direct = manifest.get("submission_id")
    if direct is not None:
        return direct
    return _mapping(manifest.get("config")).get("submission_id")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    table = pa.Table.from_pylist(list(rows))
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)


__all__ = ["KaggleReplayPolicyAuditConfig", "run"]
