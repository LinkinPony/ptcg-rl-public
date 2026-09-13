"""Cross-deck/prompt exhaustive-fit candidate constructor/scorer audit."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.runtime import CheckpointPolicy
from ptcg_rl.agent.search.candidate_budget import (
    CandidateBudgetPolicy,
    CandidateBudgetRequest,
)
from ptcg_rl.agent.search.candidates import (
    CandidateExpansionInputs,
    CandidateSourceInputs,
    MultiSourceCandidateConstructor,
)
from ptcg_rl.agent.search.hierarchical_contract import (
    DeploymentContinuationControllerIdentity,
)
from ptcg_rl.agent.search.prompt_actions import build_prompt_action_candidates
from ptcg_rl.decks.identity import parse_canonical_signature
from ptcg_rl.engine.native_consequence import NativeConsequenceLane
from ptcg_rl.evaluation.candidate_regret_config import CandidateRegretAuditConfig
from ptcg_rl.evaluation.candidate_regret_corpus import (
    CandidateAuditCase,
    load_candidate_audit_cases,
)
from ptcg_rl.evaluation.candidate_regret_engine import (
    UnsupportedExactReference,
    materialize_paired_scenarios,
    score_exhaustive_candidates,
    semantic_config_fingerprint,
)
from ptcg_rl.evaluation.candidate_regret_io import CandidateRegretWriter
from ptcg_rl.evaluation.candidate_regret_metrics import (
    candidate_regret_curve,
    post_state_alias_counts,
)
from ptcg_rl.evaluation.candidate_regret_report import (
    CandidateAuditStats,
    build_candidate_rows,
    build_root_row,
    build_summary,
    mapping_field,
    options,
)
from ptcg_rl.evaluation.candidate_regret_sampling import (
    sample_candidate_audit_roots,
)
from ptcg_rl.evaluation.candidate_regret_sources import (
    build_engine_guided_expansion_inputs,
    build_seed_source_inputs,
    construct_with_expansion,
    deterministic_root_seed,
    source_telemetry,
)
from ptcg_rl.evaluation.consequence_parity_artifact import (
    file_sha256,
    resolve_parquet_paths,
    write_json_atomic,
)


def run_candidate_regret_audit(
    config: CandidateRegretAuditConfig,
) -> dict[str, Any]:
    """Run one diagnostic campaign without screening or promotion semantics."""
    campaign_started = time.perf_counter()
    paths = resolve_parquet_paths(config.steps_globs)
    checkpoint_path = _require_file(config.checkpoint_path, "checkpoint")
    library_path = _require_file(config.engine.library_path, "native library")
    checkpoint_sha256 = file_sha256(checkpoint_path)
    library_sha256 = file_sha256(library_path)
    if checkpoint_sha256 != config.expected_checkpoint_sha256:
        raise ValueError("candidate audit checkpoint SHA-256 does not match config")
    summary_path = config.output_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError("candidate audit summary exists; use a fresh output")

    sampling_started = time.perf_counter()
    locators, sampling = sample_candidate_audit_roots(
        paths,
        max_strata=config.sampling.max_strata,
        roots_per_stratum=config.sampling.roots_per_stratum,
        max_roots=config.sampling.max_roots,
        min_legal_actions=config.sampling.min_legal_actions,
        exhaustive_reference_cap=config.sampling.exhaustive_reference_cap,
        batch_size=config.sampling.batch_size,
        seed=config.sampling.seed,
    )
    sampling_seconds = time.perf_counter() - sampling_started
    if not locators:
        raise RuntimeError("replay corpus yielded no exhaustive-fit audit roots")
    loading_started = time.perf_counter()
    cases = load_candidate_audit_cases(
        locators,
        batch_size=config.sampling.batch_size,
        fallback_card_id=config.engine.fallback_card_id,
        fallback_basic_pokemon_id=config.engine.fallback_basic_pokemon_id,
    )
    loading_seconds = time.perf_counter() - loading_started
    constructor_fingerprint = semantic_config_fingerprint(
        config.constructor.model_dump(mode="json")
    )
    resolved_semantics = {
        "constructor": config.constructor.model_dump(mode="json"),
        "scorer": config.scorer.model_dump(mode="json"),
        "k_values": list(config.k_values),
        "epsilon": config.epsilon,
        "proposal_source_mode": config.proposal_source_mode,
        "scenario_support_mode": "sampled_belief_chance_unsupported",
    }
    resolved_fingerprint = semantic_config_fingerprint(resolved_semantics)
    controller = DeploymentContinuationControllerIdentity.create(
        controller_version=config.controller_version,
        model_fingerprint=checkpoint_sha256,
        constructor_fingerprint=constructor_fingerprint,
        scorer_fingerprint=config.scorer.scorer_fingerprint,
        resolved_config_fingerprint=resolved_fingerprint,
    )
    constructor = MultiSourceCandidateConstructor(config.constructor)
    budget_policy = CandidateBudgetPolicy(config.constructor)
    model_started = time.perf_counter()
    policy = CheckpointPolicy(checkpoint_path, device=config.device)
    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("candidate audit checkpoint changed while loading")
    policy.configure_inference_cache(enabled=True)
    model_load_seconds = time.perf_counter() - model_started
    stats = CandidateAuditStats(config.k_values)
    evaluation_started = time.perf_counter()

    with (
        NativeConsequenceLane(library_path=library_path) as lane,
        CandidateRegretWriter(
            config.output_dir,
            shard_roots=config.output_shard_roots,
            compression=config.compression,
        ) as writer,
    ):
        if lane.engine_library_fingerprint != library_sha256:
            raise RuntimeError("native library changed before audit execution")
        native_abi_fingerprint = lane.native_abi_fingerprint
        for case in cases:
            root_row, candidate_rows, regret_rows = _audit_case(
                case,
                config=config,
                policy=policy,
                lane=lane,
                constructor=constructor,
                budget_policy=budget_policy,
                controller=controller,
                stats=stats,
            )
            writer.append_group(root_row, candidate_rows, regret_rows)
        writer.close()
        part_counts = dict(writer.part_counts)

    if file_sha256(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("candidate audit checkpoint changed during execution")
    if file_sha256(library_path) != library_sha256:
        raise RuntimeError("native library changed during audit execution")
    if stats.exact_roots <= 0:
        raise RuntimeError(
            "candidate audit produced no comparable exhaustive reference roots"
        )
    evaluation_seconds = time.perf_counter() - evaluation_started
    elapsed = time.perf_counter() - campaign_started
    summary = build_summary(
        config,
        sampling=asdict(sampling),
        stats=stats,
        elapsed_seconds=elapsed,
        checkpoint_sha256=checkpoint_sha256,
        library_sha256=library_sha256,
        native_abi_fingerprint=native_abi_fingerprint,
        constructor_fingerprint=constructor_fingerprint,
        resolved_fingerprint=resolved_fingerprint,
        controller=controller,
        part_counts=part_counts,
        stage_seconds={
            "sampling": sampling_seconds,
            "selected_row_loading": loading_seconds,
            "model_loading": model_load_seconds,
            "candidate_evaluation": evaluation_seconds,
        },
    )
    write_json_atomic(
        config.output_dir / "resolved_config.json",
        config.model_dump(mode="json"),
    )
    # The summary is the commit marker and is published only after every
    # referenced part and the resolved configuration are durable.
    write_json_atomic(summary_path, summary)
    return summary


def _audit_case(
    case: CandidateAuditCase,
    *,
    config: CandidateRegretAuditConfig,
    policy: CheckpointPolicy,
    lane: NativeConsequenceLane,
    constructor: MultiSourceCandidateConstructor,
    budget_policy: CandidateBudgetPolicy,
    controller: DeploymentContinuationControllerIdentity,
    stats: CandidateAuditStats,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    deck = parse_canonical_signature(case.deck_signature)
    policy.bind_own_deck(deck.card_ids)
    policy.clear_inference_cache()
    select = case.observation.get("select")
    if select is None:
        raise ValueError("candidate audit root has no select prompt")
    greedy = policy.select_action(case.observation)
    if not is_legal_action(select, greedy):
        greedy = case.consequence.observed_action
    if not is_legal_action(select, greedy):
        raise ValueError("neither policy nor replay action is legal at audit root")
    exhaustive = build_prompt_action_candidates(
        select,
        greedy_action=greedy,
        exhaustive_action_cap=config.sampling.exhaustive_reference_cap,
        beam_width=config.sampling.exhaustive_reference_cap,
    )
    if not exhaustive.exhaustive or len(exhaustive.actions) != (
        case.legal_action_count
    ):
        raise ValueError("sampled root does not fit the declared exhaustive cap")
    priors = policy.action_priors(case.observation, exhaustive.actions)
    if set(priors) != set(exhaustive.actions):
        raise ValueError("checkpoint did not score the exhaustive legal support")
    root_seed = deterministic_root_seed(case.case_id, config.sampling.seed)
    scenarios = materialize_paired_scenarios(
        case.consequence.hidden,
        requested_count=config.engine.scenario_count,
        seed=root_seed,
    )
    budget = budget_policy.resolve(
        CandidateBudgetRequest(
            legal_action_count=case.legal_action_count,
            option_count=len(options(select)),
            min_count=int(mapping_field(select, "minCount", 0)),
            max_count=int(mapping_field(select, "maxCount", len(options(select)))),
            ordered=exhaustive.ordered,
            scenario_count=len(scenarios),
            estimated_engine_steps_per_cell=(
                config.cost_model.estimated_engine_steps_per_cell
            ),
            estimated_prefix_nodes_per_candidate=(
                config.cost_model.estimated_prefix_nodes_per_candidate
            ),
            estimated_cell_time_us=config.cost_model.estimated_cell_time_us,
            semantic_boundary="terminal_handoff_or_same_main",
        )
    )
    if not budget.feasible or budget.k_seed <= 0:
        raise ValueError("audit constructor budget cannot retain its base anchor")
    seed_inputs = (
        CandidateSourceInputs()
        if budget.exhaustive
        else build_seed_source_inputs(
            exhaustive.actions,
            priors,
            seed=root_seed,
            stream_limit=budget.k_seed,
        )
    )
    seed_result = construct_with_expansion(
        constructor,
        select,
        greedy_action=greedy,
        budget=budget,
        seed_inputs=seed_inputs,
        expansion_inputs=CandidateExpansionInputs(),
        ordered=exhaustive.ordered,
        stochastic_seed=root_seed,
    )
    exact = score_exhaustive_candidates(
        lane,
        policy,
        case,
        exhaustive.actions,
        scenarios=scenarios,
        scoring_config=config.scorer,
        controller=controller,
        max_cells=config.engine.max_cells,
        max_engine_steps=config.engine.max_engine_steps,
        max_forced_steps=config.engine.max_forced_steps,
        max_observation_bytes=config.engine.max_observation_bytes,
    )
    stats.scenario_grid_complete_roots += int(exact.scenario_grid_complete)
    stats.paired_support_integrity_roots += int(exact.paired_support_integrity)
    stats.nonanticipativity_integrity_roots += int(
        exact.nonanticipativity_integrity
    )
    if isinstance(exact, UnsupportedExactReference):
        final_result = seed_result
        telemetry = source_telemetry(
            source_inputs=seed_inputs,
            expansion_inputs=CandidateExpansionInputs(),
            budget=budget,
            result=final_result,
            select=select,
            ordered=exhaustive.ordered,
            stochastic_seed=root_seed,
        )
        stats.status_counts[exact.status] += 1
        stats.native_call_ms += exact.native_call_ms
        return (
            build_root_row(
                case,
                config=config,
                budget=budget,
                result=final_result,
                telemetry=telemetry,
                exact=exact,
                alias_groups=0,
                alias_disagreements=0,
            ),
            (),
            (),
        )

    retained_seed_scores = {
        action: exact.scores[action]
        for action in seed_result.candidates.actions
        if action in exact.scores
    }
    if len(retained_seed_scores) != len(seed_result.candidates.actions):
        raise ValueError("an admitted seed candidate has no exhaustive score")
    expansion_inputs = build_engine_guided_expansion_inputs(
        select,
        seed_result,
        robust_scores=retained_seed_scores,
        ordered=exhaustive.ordered,
    )
    final_result = construct_with_expansion(
        constructor,
        select,
        greedy_action=greedy,
        budget=budget,
        seed_inputs=seed_inputs,
        expansion_inputs=expansion_inputs,
        ordered=exhaustive.ordered,
        stochastic_seed=root_seed,
    )
    if not final_result.valid or not final_result.candidates.actions:
        raise ValueError("constructor returned no valid retained support")
    telemetry = source_telemetry(
        source_inputs=seed_inputs,
        expansion_inputs=expansion_inputs,
        budget=budget,
        result=final_result,
        select=select,
        ordered=exhaustive.ordered,
        stochastic_seed=root_seed,
    )
    curve = candidate_regret_curve(
        exact.scores,
        final_result.candidates.actions,
        k_values=config.k_values,
        epsilon=config.epsilon,
    )
    successor_values = tuple(
        exact.successor_fingerprints[action] for action in exhaustive.actions
    )
    score_values = tuple(exact.scores[action] for action in exhaustive.actions)
    alias_groups, alias_disagreements = post_state_alias_counts(
        successor_values,
        score_values,
    )
    candidate_rows = build_candidate_rows(
        case,
        exact=exact,
        actions=exhaustive.actions,
        result=final_result,
        epsilon=config.epsilon,
    )
    regret_rows = tuple(
        {
            "case_id": case.case_id,
            "k": point.k,
            "retained_count": point.retained_count,
            "exhaustive_best_score": point.exhaustive_best_score,
            "retained_best_score": point.retained_best_score,
            "best_regret": point.best_regret,
            "epsilon_recall": point.epsilon_recall,
        }
        for point in curve
    )
    stats.status_counts["comparable_exhaustive_reference"] += 1
    stats.exact_roots += 1
    stats.exact_decks.add(case.deck_fingerprint)
    stats.exact_contexts.add(int(mapping_field(select, "context", 0)))
    stats.candidate_rows += len(candidate_rows)
    stats.regret_rows += len(regret_rows)
    stats.native_call_ms += exact.native_call_ms
    retained_sources = dict(
        zip(
            final_result.candidates.actions,
            final_result.candidates.sources,
            strict=True,
        )
    )
    best = max(exact.scores.values())
    for action, sources in retained_sources.items():
        if exact.scores[action] >= best - config.epsilon:
            stats.near_best_provenance.update(sources)
    for point in curve:
        stats.regret_sum[point.k] += point.best_regret
        stats.recall_count[point.k] += int(point.epsilon_recall)
        stats.regret_count[point.k] += 1
    return (
        build_root_row(
            case,
            config=config,
            budget=budget,
            result=final_result,
            telemetry=telemetry,
            exact=exact,
            alias_groups=alias_groups,
            alias_disagreements=alias_disagreements,
        ),
        candidate_rows,
        regret_rows,
    )


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"candidate audit {label} does not exist: {resolved}")
    return resolved


__all__ = ["run_candidate_regret_audit"]
