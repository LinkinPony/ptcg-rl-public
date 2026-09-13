"""Production two-stage pre-action planner behavior orchestration."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import torch

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.search.candidate_budget import (
    CandidateBudgetPlan,
    CandidateBudgetPolicy,
)
from ptcg_rl.agent.search.candidates import (
    CandidateConstructionResult,
    CandidateExpansionInputs,
    MultiSourceCandidateConstructor,
)
from ptcg_rl.agent.search.eligibility import (
    FixedSingleSelectEquivalenceProbe,
    FixedSingleSelectProbeExecution,
    PlannerEligibilityDecision,
    decide_planner_eligibility,
)
from ptcg_rl.agent.search.hierarchical_contract import (
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.mutations import legal_local_mutations
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import SharedRootInformationLeafScorer
from ptcg_rl.agent.search.planning_seed_reuse import (
    outcome_batch_from_reused_probe,
)
from ptcg_rl.agent.search.planning_session_contract import (
    HierarchicalOutcomeBatch,
    HierarchicalSearchRequest,
)
from ptcg_rl.agent.search.planning_session_scoring import (
    ScoredHierarchicalEvidence,
    merge_scored_hierarchical_evidence,
    score_hierarchical_outcomes,
)
from ptcg_rl.agent.search.planning_session_tree import (
    HierarchicalPlanningSessionExecutor,
    root_planning_information_fingerprint,
)
from ptcg_rl.agent.search.prompt_actions import describe_prompt_action_space
from ptcg_rl.engine.compact_consequence import (
    ScenarioSupportMode as EngineSupportMode,
)
from ptcg_rl.engine.compact_consequence import (
    SemanticEndpoint,
)
from ptcg_rl.engine.consequence_identity import normalize_scenario_support
from ptcg_rl.rl.macro_teacher import build_native_macro_teacher_target
from ptcg_rl.rl.planner_behavior import (
    PlannerCollectionMetadata,
    base_fallback_evidence,
    sample_planner_conditioned_action,
)
from ptcg_rl.rl.planner_behavior_policy_contract import (
    PlannerBehaviorPolicyConfig,
    PlannerCandidateEvaluator,
    PlannerDecisionRequest,
    PlannerPolicyDecision,
)
from ptcg_rl.rl.planner_evidence import (
    PLANNER_SOURCE_NAMES,
    ScenarioSupportMode,
    planner_source_bits,
)
from ptcg_rl.rl.planner_seed_reuse import splice_reusable_probe_evidence
from ptcg_rl.runtime.planner_telemetry import (
    PlannerRequestTelemetry,
    PlannerStage,
    PlannerStageEvent,
)
from ptcg_rl.runtime.work_ledger import (
    PlannerRequestLedger,
    PlannerWorkReservation,
    PlannerWorkStopReason,
)

TensorInputs = TypeVar("TensorInputs")


class PlannerBehaviorPolicy(Generic[TensorInputs]):
    """Apply eligibility, two-stage exact search, and one behavior sample."""

    def __init__(
        self,
        *,
        config: PlannerBehaviorPolicyConfig,
        executor: HierarchicalPlanningSessionExecutor,
        scorer: SharedRootInformationLeafScorer[TensorInputs],
        controller_identity: StableContinuationControllerIdentity,
        generator: torch.Generator | None = None,
        telemetry: PlannerRequestTelemetry | None = None,
    ) -> None:
        self.config = config
        self._executor = executor
        self._scorer = scorer
        self._controller_identity = controller_identity
        self._constructor = MultiSourceCandidateConstructor(config.constructor)
        self._budget_policy = CandidateBudgetPolicy(config.constructor)
        self._generator = generator
        self._telemetry = telemetry
        self._engine_reuse_hits = 0

    @property
    def runtime_stats(self) -> tuple[int, int, int, int, int, int]:
        """Return exact engine/prefix/leaf request-local reuse counters."""
        prefix_reuse, native_rows = self._executor.runtime_stats
        leaf_lookups, unique_leaves, consequence_cells = self._scorer.reuse_stats
        return (
            self._engine_reuse_hits,
            native_rows,
            prefix_reuse,
            leaf_lookups,
            unique_leaves,
            consequence_cells,
        )

    def decide(
        self,
        request: PlannerDecisionRequest,
        *,
        evaluator: PlannerCandidateEvaluator,
        ledger: PlannerRequestLedger,
    ) -> PlannerPolicyDecision:
        """Return planner-conditioned action or preserve the full base trace."""
        started = ledger.started_monotonic
        select = request.root_observation.get("select")
        if not is_legal_action(select, request.base_action):
            raise ValueError("sampled base fallback action is illegal")
        try:
            self._validate_identity(request)
            self._validate_ledger(ledger)
            eligibility = decide_planner_eligibility(
                select,
                scenario_count=len(request.scenarios),
                baseline_transition_budget=(
                    self.config.constructor.work_budget.engine_transition_limit
                ),
                config=self.config.eligibility,
                probe=_ledgered_probe(request, ledger),
                ordered=request.budget_request.ordered,
            )
            if not eligibility.eligible:
                reason = _fallback_for_eligibility(eligibility.branch, ledger)
                return self._base_fallback(
                    request,
                    ledger=ledger,
                    started=started,
                    reason=reason,
                )
            plan = self._budget_policy.resolve(request.budget_request)
            if not plan.feasible:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.BUDGET_TRUNCATED,
                    plan.fallback_reason or "candidate budget is infeasible",
                )
            construction_started = time.perf_counter()
            seeds = self._constructor.construct(
                select,
                greedy_action=request.greedy_action,
                budget=plan,
                seed_inputs=request.seed_inputs,
                ordered=request.budget_request.ordered,
                stochastic_seed=request.stochastic_seed,
            )
            self._record_construction(
                construction_started,
                rows=len(seeds.candidates.actions),
            )
            _require_construction(seeds)
            _reserve_and_complete(
                ledger,
                candidates=len(seeds.candidates.actions),
            )
            seed_request = _hierarchical_request(
                request,
                actions=seeds.candidates.actions,
            )
            seed_scored = self._score_seeds(
                seed_request,
                eligibility=eligibility,
                request=request,
                ledger=ledger,
            )
            final_construction, final_scored = self._expand_and_score(
                request,
                plan=plan,
                seeds=seeds,
                seed_scored=seed_scored,
                ledger=ledger,
            )
            features = torch.tensor(
                final_scored.search_evidence.feature_rows,
                dtype=torch.float32,
            )
            gpu_reservation = _reserve_gpu_rows(
                ledger, len(final_construction.candidates.actions)
            )
            evaluation_success = False
            try:
                model_evaluation = evaluator.evaluate_planner_candidates(
                    actions=final_construction.candidates.actions,
                    aggregate_features=features,
                )
                evaluation_success = True
            finally:
                ledger.complete(
                    gpu_reservation,
                    elapsed_seconds=0.0,
                    success=evaluation_success,
                )
            if model_evaluation.candidate_counts != (
                len(final_construction.candidates.actions),
            ):
                raise PlannerEvidenceError(
                    PlannerFallbackReason.MODEL_VERSION_MISMATCH,
                    "dedicated planner evaluator returned another support",
                )
            metadata = self._metadata(
                request,
                ledger=ledger,
                started=started,
                construction=final_construction,
                scored=final_scored,
            )
            sample_started = time.perf_counter()
            selection = sample_planner_conditioned_action(
                actions=final_construction.candidates.actions,
                aggregate_features=final_scored.search_evidence.feature_rows,
                source_bits=tuple(
                    _source_bits(sources)
                    for sources in final_construction.candidates.sources
                ),
                rules_exact=tuple(final_scored.search_evidence.rules_exact),
                base_logprobs_at_collection=(model_evaluation.base_action_logprobs),
                proposal_logprobs_at_collection=(
                    model_evaluation.proposal_action_logprobs
                ),
                reranker_residual_at_collection=(model_evaluation.reranker_residuals),
                robust_scores=final_scored.robust_scores.to(
                    device=model_evaluation.base_action_logprobs.device
                ),
                scoring_config=self._scorer.config,
                metadata=metadata,
                generator=self._generator,
            )
            if self._telemetry is not None:
                self._telemetry.record(
                    PlannerStageEvent(
                        stage=PlannerStage.BEHAVIOR_SAMPLE,
                        seconds=time.perf_counter() - sample_started,
                        rows=len(final_construction.candidates.actions),
                    )
                )
            macro_teacher_target = None
            if self.config.emit_macro_teacher:
                try:
                    macro_teacher_target = build_native_macro_teacher_target(
                        actions=final_construction.candidates.actions,
                        scored=final_scored,
                        metadata=metadata,
                    )
                except (RuntimeError, TypeError, ValueError, OverflowError):
                    macro_teacher_target = None
            return PlannerPolicyDecision(
                action=selection.action,
                old_logprob=selection.old_logprob,
                planner_behavior=selection.evidence,
                used_base_trace=False,
                macro_teacher_target=macro_teacher_target,
            )
        except PlannerEvidenceError as exc:
            return self._base_fallback(
                request,
                ledger=ledger,
                started=started,
                reason=exc.reason,
            )
        except (RuntimeError, TypeError, ValueError, OverflowError):
            return self._base_fallback(
                request,
                ledger=ledger,
                started=started,
                reason=PlannerFallbackReason.ENGINE_ERROR,
            )

    def base_fallback(
        self,
        request: PlannerDecisionRequest,
        *,
        ledger: PlannerRequestLedger,
        reason: PlannerFallbackReason,
    ) -> PlannerPolicyDecision:
        """Materialize an explicit schema-9 fallback without launching work."""
        if reason is PlannerFallbackReason.NONE:
            raise ValueError("base fallback requires a concrete reason")
        ledger.stop(PlannerWorkStopReason.EXPLICIT_FALLBACK)
        return self._base_fallback(
            request,
            ledger=ledger,
            started=ledger.started_monotonic,
            reason=reason,
        )

    def _score_seeds(
        self,
        seed_request: HierarchicalSearchRequest,
        *,
        eligibility: PlannerEligibilityDecision,
        request: PlannerDecisionRequest,
        ledger: PlannerRequestLedger,
    ) -> ScoredHierarchicalEvidence[TensorInputs]:
        reusable_v5 = eligibility.reusable_v5_evidence
        if reusable_v5 is not None:
            if reusable_v5.actions != seed_request.candidate_actions:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "reusable v5 support differs from constructed seeds",
                )
            expected_fingerprint = self._executor.producer_contract_fingerprint(
                seed_request
            )
            if reusable_v5.request_fingerprint != expected_fingerprint:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.FINGERPRINT_MISMATCH,
                    "reusable v5 producer differs from seed request",
                )
            scored = reusable_v5.scored
            if scored.leaf_scores.scorer_fingerprint != self._scorer.scorer_fingerprint:
                raise PlannerEvidenceError(
                    PlannerFallbackReason.MODEL_VERSION_MISMATCH,
                    "reusable v5 seed uses another scorer lease",
                )
            self._engine_reuse_hits += sum(
                len(outcome.cells) for outcome in scored.outcome_batch.outcomes
            )
            return scored
        envelope = eligibility.reusable_evidence
        if envelope is None:
            outcomes = self._executor.evaluate(seed_request, ledger=ledger)
            return self._score_outcomes(
                outcomes,
                request=seed_request,
                ledger=ledger,
            )
        expected = request.expected_probe_identity
        if expected is None:
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "reusable probe has no prepared request identity",
            )
        try:
            reuse = splice_reusable_probe_evidence(
                envelope=envelope,
                expected_request_identity=expected,
                retained_actions=seed_request.candidate_actions,
            )
        except ValueError as exc:
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "reusable probe identity differs from the planner request",
            ) from exc
        reused_rows = reuse.retained_cell_to_cached_cell.reshape(-1)
        self._engine_reuse_hits += sum(int(row) >= 0 for row in reused_rows)
        reused_nonterminal_rows = sum(
            int(
                envelope.batch.row_at(
                    int(row) // envelope.batch.scenario_count,
                    int(row) % envelope.batch.scenario_count,
                ).endpoint
                is not SemanticEndpoint.TERMINAL
            )
            for row in reused_rows
            if int(row) >= 0
        )
        gpu_reservation = (
            None
            if reused_nonterminal_rows == 0
            else _reserve_gpu_rows(ledger, reused_nonterminal_rows)
        )
        success = False
        try:
            reused: Any = outcome_batch_from_reused_probe(
                request=seed_request,
                reuse=reuse,
                cached=envelope.batch,
                root_information_history_fingerprint=(
                    root_planning_information_fingerprint(seed_request)
                ),
                controller=self._controller_identity,
                scorer=self._scorer,
            )
            success = True
        finally:
            if gpu_reservation is not None:
                ledger.complete(
                    gpu_reservation,
                    elapsed_seconds=0.0,
                    success=success,
                )
        return score_hierarchical_outcomes(
            reused.outcomes,
            scorer=self._scorer,
            legal_action_count=seed_request.legal_action_count,
            support_exhaustive=seed_request.support_exhaustive,
            precomputed_leaf_scores=reused.leaf_scores,
        )

    def _expand_and_score(
        self,
        request: PlannerDecisionRequest,
        *,
        plan: CandidateBudgetPlan,
        seeds: CandidateConstructionResult,
        seed_scored: ScoredHierarchicalEvidence[TensorInputs],
        ledger: PlannerRequestLedger,
    ) -> tuple[CandidateConstructionResult, ScoredHierarchicalEvidence[TensorInputs]]:
        if plan.k_total == plan.k_seed:
            return seeds, seed_scored
        select = request.root_observation.get("select")
        mutations = _ranked_mutations(
            select,
            actions=seeds.candidates.actions,
            robust_scores=tuple(
                item.robust_score for item in seed_scored.aggregation.candidates
            ),
            ordered=request.budget_request.ordered,
            max_parents=self.config.max_mutation_parents,
        )
        construction_started = time.perf_counter()
        final = self._constructor.construct(
            select,
            greedy_action=request.greedy_action,
            budget=plan,
            seed_inputs=request.seed_inputs,
            expansion_inputs=CandidateExpansionInputs(
                mutation=mutations,
                novelty=request.novelty_actions,
            ),
            ordered=request.budget_request.ordered,
            stochastic_seed=request.stochastic_seed,
        )
        self._record_construction(
            construction_started,
            rows=len(final.candidates.actions),
        )
        _require_construction(final)
        if final.candidates.actions[: len(seeds.candidates.actions)] != (
            seeds.candidates.actions
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.CONSTRUCTOR_INVALID,
                "post-seed expansion changed the immutable seed prefix",
            )
        novel_count = len(final.candidates.actions) - len(seeds.candidates.actions)
        if novel_count:
            _reserve_and_complete(ledger, candidates=novel_count)
        novel_actions = final.candidates.actions[len(seeds.candidates.actions) :]
        if not novel_actions:
            return final, seed_scored
        novel_request = _hierarchical_request(
            request,
            actions=novel_actions,
            force_nonexhaustive=True,
        )
        novel_outcomes = self._executor.evaluate(novel_request, ledger=ledger)
        novel_scored = self._score_outcomes(
            novel_outcomes,
            request=novel_request,
            ledger=ledger,
        )
        final_request = _hierarchical_request(
            request,
            actions=final.candidates.actions,
        )
        return final, merge_scored_hierarchical_evidence(
            (seed_scored, novel_scored),
            scorer=self._scorer,
            legal_action_count=final_request.legal_action_count,
            support_exhaustive=final_request.support_exhaustive,
            producer_contract_fingerprint=(
                self._executor.producer_contract_fingerprint(final_request)
            ),
        )

    def _record_construction(self, started: float, *, rows: int) -> None:
        if self._telemetry is None:
            return
        self._telemetry.record(
            PlannerStageEvent(
                stage=PlannerStage.CANDIDATE_CONSTRUCTION,
                seconds=time.perf_counter() - started,
                rows=rows,
            )
        )

    def _score_outcomes(
        self,
        outcomes: HierarchicalOutcomeBatch,
        *,
        request: HierarchicalSearchRequest,
        ledger: PlannerRequestLedger,
    ) -> ScoredHierarchicalEvidence[TensorInputs]:
        leaf_count = len(outcomes.leaves.leaves)
        reservation = None if leaf_count == 0 else _reserve_gpu_rows(ledger, leaf_count)
        success = False
        try:
            result: ScoredHierarchicalEvidence[TensorInputs] = (
                score_hierarchical_outcomes(
                    outcomes,
                    scorer=self._scorer,
                    legal_action_count=request.legal_action_count,
                    support_exhaustive=request.support_exhaustive,
                )
            )
            success = True
            return result
        finally:
            if reservation is not None:
                ledger.complete(reservation, elapsed_seconds=0.0, success=success)

    def _validate_identity(self, request: PlannerDecisionRequest) -> None:
        identity = request.identity
        if identity.constructor_version != self.config.constructor.architecture_version:
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "constructor version differs from the resolved policy",
            )
        if identity.scorer_fingerprint != self._scorer.scorer_fingerprint:
            raise PlannerEvidenceError(
                PlannerFallbackReason.MODEL_VERSION_MISMATCH,
                "scorer fingerprint differs from the leased policy",
            )
        controller = self._controller_identity
        if (
            identity.controller_fingerprint != controller.controller_fingerprint
            or identity.constructor_fingerprint != controller.constructor_fingerprint
            or identity.scorer_fingerprint != controller.scorer_fingerprint
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.MODEL_VERSION_MISMATCH,
                "continuation controller differs from planner runtime identity",
            )

    def _validate_ledger(self, ledger: PlannerRequestLedger) -> None:
        budget = self.config.constructor.work_budget
        if (
            ledger.limits.max_transitions != budget.engine_transition_limit
            or ledger.limits.max_nodes != budget.prefix_node_limit
        ):
            raise PlannerEvidenceError(
                PlannerFallbackReason.FINGERPRINT_MISMATCH,
                "request ledger differs from the constructor work budget",
            )

    def _metadata(
        self,
        request: PlannerDecisionRequest,
        *,
        ledger: PlannerRequestLedger,
        started: float,
        construction: CandidateConstructionResult | None,
        scored: ScoredHierarchicalEvidence[TensorInputs] | None,
    ) -> PlannerCollectionMetadata:
        snapshot = ledger.snapshot()
        budget = self.config.constructor.work_budget
        if scored is None:
            support, _ = normalize_scenario_support(
                request.scenarios,
                mode=EngineSupportMode.SAMPLED_BELIEF_NO_CHANCE,
            )
            root_request = _hierarchical_request(
                request,
                actions=(request.base_action,),
                force_nonexhaustive=True,
            )
            root_fingerprint = root_planning_information_fingerprint(root_request)
            support_fingerprint = support.support_fingerprint
            scenario_count = len(support.scenarios)
            support_exhaustive = False
            scenario_grid_complete = False
            leaf_bootstrapped = False
        else:
            outcome = scored.outcome_batch.outcomes[0]
            root_fingerprint = outcome.root_information_history_fingerprint
            support_fingerprint = outcome.scenario_support_fingerprint
            scenario_count = len(outcome.cells)
            support_exhaustive = scored.search_evidence.exhaustive
            scenario_grid_complete = True
            leaf_bootstrapped = scored.leaf_scores.leaf_bootstrapped
        configured = _configured_source_quotas(self.config)
        used = _used_source_quotas(construction)
        wall_used = max(0, math.floor((time.monotonic() - started) * 1_000.0))
        if scored is not None and wall_used > budget.wall_clock_limit_ms:
            raise PlannerEvidenceError(
                PlannerFallbackReason.DEADLINE,
                "planner completed after its wall-clock contract",
            )
        return PlannerCollectionMetadata(
            root_information_fingerprint=root_fingerprint,
            scenario_support_fingerprint=support_fingerprint,
            model_fingerprint=request.identity.model_fingerprint,
            constructor_fingerprint=request.identity.constructor_fingerprint,
            scorer_fingerprint=request.identity.scorer_fingerprint,
            controller_fingerprint=request.identity.controller_fingerprint,
            planner_fingerprint=request.identity.planner_fingerprint,
            policy_version=request.identity.policy_version,
            proposal_version=request.identity.proposal_version,
            constructor_version=request.identity.constructor_version,
            planner_version=request.identity.planner_version,
            scenario_support_mode=(
                ScenarioSupportMode.BELIEF_SAMPLED_CHANCE_ENUMERATED
            ),
            legal_action_count=request.budget_request.legal_action_count,
            scenario_count=scenario_count,
            support_exhaustive=support_exhaustive,
            scenario_grid_complete=scenario_grid_complete,
            leaf_bootstrapped=leaf_bootstrapped,
            configured_source_quotas=configured,
            used_source_quotas=used,
            engine_transition_limit=budget.engine_transition_limit,
            engine_transitions_used=snapshot.transitions,
            prefix_node_limit=budget.prefix_node_limit,
            prefix_nodes_used=snapshot.nodes,
            wall_clock_limit_ms=budget.wall_clock_limit_ms,
            wall_clock_used_ms=wall_used,
            planner_temperature=self._scorer.config.planner_temperature,
        )

    def _base_fallback(
        self,
        request: PlannerDecisionRequest,
        *,
        ledger: PlannerRequestLedger,
        started: float,
        reason: PlannerFallbackReason,
    ) -> PlannerPolicyDecision:
        metadata = self._metadata(
            request,
            ledger=ledger,
            started=started,
            construction=None,
            scored=None,
        )
        return PlannerPolicyDecision(
            action=request.base_action,
            old_logprob=request.base_old_logprob,
            planner_behavior=base_fallback_evidence(
                metadata=metadata,
                reason=reason,
            ),
            used_base_trace=True,
        )


@dataclass(slots=True)
class _LedgeredProbe:
    inner: FixedSingleSelectEquivalenceProbe
    ledger: PlannerRequestLedger
    required_transitions: int

    def probe(
        self,
        select: Any,
        *,
        scenario_count: int,
        transition_budget: int,
    ) -> FixedSingleSelectProbeExecution:
        if self.required_transitions > transition_budget:
            raise ValueError("equivalence probe exceeds its declared budget")
        if bool(getattr(self.inner, "work_already_accounted", False)):
            return self.inner.probe(
                select,
                scenario_count=scenario_count,
                transition_budget=transition_budget,
            )
        reservation = self.ledger.reserve(
            transitions=self.required_transitions,
            native_calls=1,
        )
        if reservation is None:
            raise ValueError("equivalence probe exceeds the global request ledger")
        success = False
        try:
            result = self.inner.probe(
                select,
                scenario_count=scenario_count,
                transition_budget=transition_budget,
            )
            success = True
            return result
        finally:
            self.ledger.complete(
                reservation,
                elapsed_seconds=0.0,
                success=success,
            )


def _ledgered_probe(
    request: PlannerDecisionRequest,
    ledger: PlannerRequestLedger,
) -> FixedSingleSelectEquivalenceProbe | None:
    probe = request.equivalence_probe
    if probe is None:
        return None
    select = request.root_observation.get("select")
    legal_count = describe_prompt_action_space(select).legal_action_count
    return _LedgeredProbe(
        inner=probe,
        ledger=ledger,
        required_transitions=legal_count * len(request.scenarios),
    )


def _hierarchical_request(
    request: PlannerDecisionRequest,
    *,
    actions: tuple[tuple[int, ...], ...],
    force_nonexhaustive: bool = False,
) -> HierarchicalSearchRequest:
    exhaustive = (
        not force_nonexhaustive
        and len(actions) == request.budget_request.legal_action_count
    )
    return HierarchicalSearchRequest.from_sequences(
        request.state_token,
        root_observation=request.root_observation,
        scenarios=request.scenarios,
        candidate_actions=actions,
        legal_action_count=request.budget_request.legal_action_count,
        root_player=request.root_player,
        context_snapshot=request.context_snapshot,
        belief_feature_producer=request.belief_feature_producer,
        belief_summary_width=request.belief_summary_width,
        producer_context=request.producer_context,
        belief_summary=request.belief_summary,
        producer_contract_fingerprint=request.producer_contract_fingerprint,
        support_exhaustive=exhaustive,
    )


def _ranked_mutations(
    select: Any,
    *,
    actions: tuple[tuple[int, ...], ...],
    robust_scores: tuple[float, ...],
    ordered: bool,
    max_parents: int,
) -> tuple[tuple[int, ...], ...]:
    if len(actions) != len(robust_scores):
        raise ValueError("seed actions and scores must align")
    parent_indices = sorted(
        range(len(actions)),
        key=lambda index: (-robust_scores[index], index),
    )[:max_parents]
    retained = set(actions)
    mutations: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for parent_index in parent_indices:
        for mutation in legal_local_mutations(
            select,
            actions[parent_index],
            ordered=ordered,
        ):
            if mutation.action in retained or mutation.action in seen:
                continue
            seen.add(mutation.action)
            mutations.append(mutation.action)
    return tuple(mutations)


def _require_construction(result: CandidateConstructionResult) -> None:
    if not result.valid or not result.candidates.actions:
        raise PlannerEvidenceError(
            PlannerFallbackReason.CONSTRUCTOR_INVALID,
            result.fallback_reason or "candidate construction returned no support",
        )


def _reserve_and_complete(
    ledger: PlannerRequestLedger,
    *,
    candidates: int,
) -> None:
    reservation = ledger.reserve(candidates=candidates)
    if reservation is None:
        raise PlannerEvidenceError(
            PlannerFallbackReason.BUDGET_TRUNCATED,
            "retained candidates exceed the global request ledger",
        )
    ledger.complete(reservation, elapsed_seconds=0.0, success=True)


def _reserve_gpu_rows(
    ledger: PlannerRequestLedger,
    rows: int,
) -> PlannerWorkReservation:
    reservation = ledger.reserve(gpu_rows=rows)
    if reservation is None:
        reason = (
            PlannerFallbackReason.DEADLINE
            if ledger.snapshot().stop_reason is PlannerWorkStopReason.DEADLINE_GUARD
            else PlannerFallbackReason.BUDGET_TRUNCATED
        )
        raise PlannerEvidenceError(reason, "planner GPU rows exceed the request ledger")
    return reservation


def _source_bits(sources: tuple[str, ...]) -> int:
    normalized = tuple(
        "base" if source == "base_greedy" else source for source in sources
    )
    return planner_source_bits(normalized) if normalized else 0


def _configured_source_quotas(
    config: PlannerBehaviorPolicyConfig,
) -> tuple[int, ...]:
    seed = config.constructor.seed_quotas.as_dict()
    expansion = config.constructor.expansion_quotas.as_dict()
    values: dict[str, int] = {}
    for name, value in seed.items():
        values[name] = value
    for expansion_name, value in expansion.items():
        values[expansion_name] = value
    return tuple(int(values.get(name, 0)) for name in PLANNER_SOURCE_NAMES)


def _used_source_quotas(
    construction: CandidateConstructionResult | None,
) -> tuple[int, ...]:
    if construction is None:
        return tuple(0 for _ in PLANNER_SOURCE_NAMES)
    values: dict[str, int] = {}
    values.update(construction.seed_source_usage)
    values.update(construction.expansion_source_usage)
    return tuple(int(values.get(name, 0)) for name in PLANNER_SOURCE_NAMES)


def _fallback_for_eligibility(
    branch: str,
    ledger: PlannerRequestLedger,
) -> PlannerFallbackReason:
    stop_reason = ledger.snapshot().stop_reason
    if stop_reason is PlannerWorkStopReason.DEADLINE_GUARD:
        return PlannerFallbackReason.DEADLINE
    if stop_reason is not PlannerWorkStopReason.ACTIVE:
        return PlannerFallbackReason.BUDGET_TRUNCATED
    return (
        PlannerFallbackReason.INELIGIBLE
        if branch == "base_fast_path"
        else PlannerFallbackReason.EVIDENCE_ABSENT
    )


__all__ = ["PlannerBehaviorPolicy"]
