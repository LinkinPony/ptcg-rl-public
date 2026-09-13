"""Replay-root S1 audit for frozen paired-belief same-turn search."""

from __future__ import annotations

import random
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from ptcg_rl.actions.selection import is_forced
from ptcg_rl.agent.probe import (
    observation_with_probe_features,
    run_runtime_probe_features,
)
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.config import SearchRuntimeConfig
from ptcg_rl.agent.search.context import observation_with_context
from ptcg_rl.agent.search.reranker import (
    MacroSearchPolicy,
    MacroSearchResult,
    PairedMacroSearcher,
)
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import (
    GameContext,
    OpponentBeliefFeatureProducer,
)
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_steps.records import iter_replay_steps, replay_stub
from ptcg_rl.evaluation.search_counterfactual_config import (
    SearchCounterfactualConfig,
)
from ptcg_rl.evaluation.search_counterfactual_io import (
    CounterfactualAuditWriter,
)
from ptcg_rl.evaluation.search_counterfactual_metrics import CounterfactualAuditStats
from ptcg_rl.evaluation.search_counterfactual_rows import (
    build_counterfactual_rows,
    prior_diagnostics,
)
from ptcg_rl.evaluation.search_counterfactual_support import (
    belief_distributions,
    episode_split,
    int_field,
    is_own_decision,
    mapping,
    optional_float,
    phase,
    prepare_output_dir,
    resolve_belief_config,
    resolve_replay_paths,
    resolve_sampler_config,
    resolve_seat,
    root_seed,
    terminal_value,
    write_campaign_artifacts,
)
from ptcg_rl.evaluation.search_identity import (
    SearchCampaignIdentityConfig,
    build_search_campaign_identity,
)


def run_search_counterfactual_audit(
    config: SearchCounterfactualConfig,
) -> dict[str, Any]:
    """Run frozen replay-root search and stream grouped counterfactual evidence."""
    replay_paths = resolve_replay_paths(config)
    output_dir = records.repo_path(config.output_dir)
    prepare_output_dir(output_dir, overwrite=config.overwrite)
    deck_path = records.repo_path(config.deck_path)
    checkpoint_path = records.repo_path(config.checkpoint_path)
    belief_path = (
        config.belief.deck_signature_summary_path
        or config.sampler.prior_deck_signature_summary_path
    )
    resolved_belief_path = records.repo_path(belief_path) if belief_path else None
    identity = build_search_campaign_identity(
        SearchCampaignIdentityConfig(
            experiment_id=config.experiment_id,
            stage="S1",
            deck_path=deck_path,
            checkpoint_path=checkpoint_path,
            belief_path=resolved_belief_path,
            resolved_search=config.macro,
            runtime_definition={
                "device": config.device,
                "precision": config.precision,
                "include_probe_features": config.include_probe_features,
                "root_quota_seconds": config.root_quota_seconds,
                "belief": config.belief.model_dump(mode="json"),
                "sampler": config.sampler.model_dump(mode="json"),
            },
            replay_paths=replay_paths,
            replay_definition={
                "team_name": config.team_name,
                "seat_index": config.seat_index,
                "split_seed": config.split_seed,
                "holdout_fraction": config.holdout_fraction,
                "roots_per_phase_per_replay": config.roots_per_phase_per_replay,
                "early_turn_max": config.early_turn_max,
                "mid_turn_max": config.mid_turn_max,
            },
            runtime_source_paths=config.runtime_source_paths,
            engine_asset_paths=config.engine_asset_paths,
            stage_parameters={
                "max_replays": config.max_replays,
                "min_roots": config.min_roots,
                "max_roots": config.max_roots,
                "margin_sweep": config.margin_sweep,
                "seed": config.seed,
                "references": config.references.model_dump(mode="json"),
            },
        )
    )
    deck = records.read_deck(deck_path)
    policy = CheckpointPolicy(
        checkpoint_path,
        device=config.device,
        own_deck=deck,
    )
    policy.configure_inference_cache(
        enabled=config.macro.root_inference_cache_enabled,
    )
    belief = OpponentBeliefFeatureProducer.from_config(
        resolve_belief_config(config.belief)
    )
    sampler = BeliefSampler(config=resolve_sampler_config(config.sampler))
    stats = CounterfactualAuditStats(
        rerank=config.macro.rerank,
        margin_sweep=config.margin_sweep,
    )
    roots = 0
    replays = 0
    split_counts: Counter[str] = Counter()
    phase_counts: Counter[str] = Counter()
    started_at = time.perf_counter()
    with CounterfactualAuditWriter(
        output_dir,
        compression=config.compression,
    ) as writer:
        for replay_path in replay_paths:
            if roots >= config.max_roots:
                break
            replays += 1
            produced = _audit_replay(
                replay_path,
                config=config,
                deck=deck,
                policy=policy,
                belief=belief,
                sampler=sampler,
                identity=identity,
                writer=writer,
                stats=stats,
                roots_remaining=config.max_roots - roots,
                split_counts=split_counts,
                phase_counts=phase_counts,
            )
            roots += produced

    metric_summary = stats.summary(config.references)
    elapsed_seconds = time.perf_counter() - started_at
    summary = {
        "protocol": "ITS-EVAL-v1-S1",
        "experiment_id": config.experiment_id,
        "campaign_fp": identity["campaign_fp"],
        "stage_fp": identity["stage_fp"],
        "roots": roots,
        "replays": replays,
        "split_counts": dict(sorted(split_counts.items())),
        "phase_counts": dict(sorted(phase_counts.items())),
        "elapsed_seconds": elapsed_seconds,
        "roots_per_second": roots / elapsed_seconds if elapsed_seconds > 0.0 else 0.0,
        "runner_complete": roots >= config.min_roots,
        **metric_summary,
        "selection_claim": None,
        "config": config.model_dump(mode="json"),
    }
    write_campaign_artifacts(output_dir, config, identity, summary)
    return summary


def _audit_replay(
    replay_path: Path,
    *,
    config: SearchCounterfactualConfig,
    deck: Sequence[int],
    policy: CheckpointPolicy,
    belief: OpponentBeliefFeatureProducer,
    sampler: BeliefSampler,
    identity: Mapping[str, Any],
    writer: CounterfactualAuditWriter,
    stats: CounterfactualAuditStats,
    roots_remaining: int,
    split_counts: Counter[str],
    phase_counts: Counter[str],
) -> int:
    metadata = replay_stub(replay_path, chunk_size=config.chunk_size)
    seat = resolve_seat(metadata, config)
    episode_id = int(mapping(metadata.get("info")).get("EpisodeId", replay_path.stem))
    split = episode_split(
        episode_id,
        seed=config.split_seed,
        holdout_fraction=config.holdout_fraction,
    )
    terminal_outcome = terminal_value(metadata, seat)
    context = GameContext(player_index=seat)
    context.set_own_deck(deck)
    selected_per_phase: Counter[str] = Counter()
    produced = 0
    for step_index, sides in iter_replay_steps(
        replay_path,
        chunk_size=config.chunk_size,
    ):
        if produced >= roots_remaining or seat >= len(sides):
            break
        side = sides[seat]
        observation = mapping(side.get("observation"))
        if str(side.get("status", "")) != "ACTIVE" or not is_own_decision(
            observation,
            seat,
        ):
            continue
        context_features = context.update(observation)
        select = mapping(observation.get("select"))
        if not select or is_forced(select):
            continue
        phase_name = phase(observation, config)
        if selected_per_phase[phase_name] >= config.roots_per_phase_per_replay:
            continue
        selected_per_phase[phase_name] += 1
        root_group = _audit_root(
            observation,
            episode_id=episode_id,
            step_index=step_index,
            seat=seat,
            split=split,
            phase=phase_name,
            terminal_value=terminal_outcome,
            context=context,
            context_features=context_features,
            config=config,
            deck=deck,
            policy=policy,
            belief=belief,
            sampler=sampler,
            identity=identity,
        )
        writer.write_root_group(
            root_group[0],
            root_group[1],
            root_group[2],
        )
        root_row = root_group[0]
        result = root_group[3]
        stats.update(
            split=split,
            episode_id=episode_id,
            terminal_value=terminal_outcome,
            root_value=optional_float(root_row["root_value"]),
            greedy_action=tuple(int(index) for index in root_row["greedy_action"]),
            result=result,
            search_seconds=float(root_row["search_seconds"]),
            whole_act_seconds=float(root_row["whole_act_seconds"]),
            deadline_overshoot_seconds=float(
                root_row["deadline_overshoot_seconds"]
            ),
            illegal_candidates=int(root_row["illegal_candidate_count"]),
            telemetry_complete=bool(root_row["telemetry_complete"]),
        )
        produced += 1
        split_counts[split] += 1
        phase_counts[phase_name] += 1
    return produced


def _audit_root(
    observation: Mapping[str, Any],
    *,
    episode_id: int,
    step_index: int,
    seat: int,
    split: str,
    phase: str,
    terminal_value: float | None,
    context: GameContext,
    context_features: Any,
    config: SearchCounterfactualConfig,
    deck: Sequence[int],
    policy: CheckpointPolicy,
    belief: OpponentBeliefFeatureProducer,
    sampler: BeliefSampler,
    identity: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    MacroSearchResult,
]:
    policy.clear_inference_cache()
    root_started = time.perf_counter()
    enriched_features = belief.augment(observation, context_features)
    serving_observation = observation_with_context(observation, enriched_features)
    preparation_seconds = time.perf_counter() - root_started

    probe_started = time.perf_counter()
    if config.include_probe_features:
        opponent_card_probs, opponent_hand_weights = belief_distributions(
            policy,
            serving_observation,
            sampler,
        )
        probe_result = run_runtime_probe_features(
            serving_observation,
            enriched_features,
            your_deck=deck,
            sampler=sampler,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=random.Random(root_seed(config.seed, episode_id, step_index, "probe")),
            config=SearchRuntimeConfig(
                enabled=True,
                worlds=config.macro.worlds,
                top_k=config.macro.top_k,
                manual_coin=config.macro.manual_coin,
                sampler=config.sampler,
                macro=config.macro.model_copy(update={"mode": "disabled"}),
            ),
        )
        if probe_result is not None:
            serving_observation = observation_with_probe_features(
                serving_observation,
                probe_result,
            )
    probe_seconds = time.perf_counter() - probe_started

    base_started = time.perf_counter()
    greedy_action = tuple(policy.select_action(serving_observation))
    base_policy_seconds = time.perf_counter() - base_started
    search_started = time.perf_counter()
    deadline = search_started + config.root_quota_seconds
    opponent_card_probs, opponent_hand_weights = belief_distributions(
        policy,
        serving_observation,
        sampler,
    )
    searcher = PairedMacroSearcher(
        policy=cast(MacroSearchPolicy, policy),
        sampler=sampler,
        your_deck=deck,
        context_snapshot=context.snapshot(),
        root_context_features=enriched_features,
        belief_producer=belief,
        config=config.macro,
        rng=random.Random(root_seed(config.seed, episode_id, step_index, "search")),
        opponent_card_probs=opponent_card_probs,
        opponent_hand_weights=opponent_hand_weights,
    )
    result = searcher.run(
        serving_observation,
        greedy_action=greedy_action,
        deadline=deadline,
    )
    search_finished = time.perf_counter()
    search_seconds = search_finished - search_started
    whole_act_seconds = search_finished - root_started
    deadline_overshoot_seconds = max(0.0, search_finished - deadline)

    audit_started = time.perf_counter()
    root_player = int_field(serving_observation.get("current"), "yourIndex", seat)
    root_value = policy.value(serving_observation, root_player)
    priors = policy.action_priors(serving_observation, result.candidates.actions)
    entropy, top_gap = prior_diagnostics(priors)
    audit_seconds = time.perf_counter() - audit_started
    return build_counterfactual_rows(
        episode_id=episode_id,
        step_index=step_index,
        seat=seat,
        split=split,
        phase=phase,
        terminal_value=terminal_value,
        root_value=root_value,
        serving_observation=serving_observation,
        greedy_action=greedy_action,
        result=result,
        priors=priors,
        identity=identity,
        timing={
            "preparation_seconds": preparation_seconds,
            "probe_seconds": probe_seconds,
            "base_policy_seconds": base_policy_seconds,
            "search_seconds": search_seconds,
            "whole_act_seconds": whole_act_seconds,
            "audit_seconds": audit_seconds,
            "deadline_overshoot_seconds": deadline_overshoot_seconds,
        },
        policy_entropy=entropy,
        policy_top_gap=top_gap,
    )
