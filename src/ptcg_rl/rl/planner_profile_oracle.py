"""Separately timed exhaustive v5 references for planner profiling."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, fields

from ptcg_rl.agent.search.hierarchical_contract import (
    StableContinuationControllerIdentity,
)
from ptcg_rl.agent.search.planner_fallback import (
    PlannerEvidenceError,
    PlannerFallbackReason,
)
from ptcg_rl.agent.search.planner_scoring import SharedRootInformationLeafScorer
from ptcg_rl.agent.search.planning_session_contract import (
    ContinuationPrompt,
    HierarchicalSearchRequest,
)
from ptcg_rl.agent.search.planning_session_scoring import (
    ScoredHierarchicalEvidence,
    merge_scored_hierarchical_evidence,
    score_hierarchical_outcomes,
)
from ptcg_rl.agent.search.planning_session_tree import (
    HierarchicalPlanningSessionExecutor,
)
from ptcg_rl.agent.search.prompt_actions import build_prompt_action_candidates
from ptcg_rl.agent.search.root_information_tensorizer import (
    ProductionRootInformationTensorizer,
    RootInformationModelInputBatch,
)
from ptcg_rl.engine.compact_consequence import ScenarioSupportMode
from ptcg_rl.engine.consequence_identity import normalize_scenario_support
from ptcg_rl.engine.native_planning_session_pool import NativePlanningSessionLanePool
from ptcg_rl.evaluation.planner_profile_config import PlannerProfileOracleConfig
from ptcg_rl.rl.planner_behavior_policy_contract import PlannerDecisionRequest
from ptcg_rl.rl.planner_inference_session import PlannerInferenceSession
from ptcg_rl.rl.planner_runtime_identity import (
    ResolvedPlannerRuntimeConfig,
    ResolvedPlannerRuntimeIdentity,
)
from ptcg_rl.rl.planner_service_inputs import (
    PlannerDecisionRequestFactory,
    PlannerRootRow,
)
from ptcg_rl.runtime.work_ledger import (
    PlannerRequestLedger,
    PlannerWorkStopReason,
)

Action = tuple[int, ...]
_ORACLE_UNRANKED_REASONS = frozenset(
    {
        PlannerFallbackReason.EVIDENCE_ABSENT,
        PlannerFallbackReason.UNSUPPORTED_CHANCE,
        PlannerFallbackReason.SCENARIO_GRID_INCOMPLETE,
        PlannerFallbackReason.RULES_INEXACT,
        PlannerFallbackReason.DEADLINE,
        PlannerFallbackReason.LEAF_VALUE_UNAVAILABLE,
        PlannerFallbackReason.BUDGET_TRUNCATED,
    }
)


@dataclass(frozen=True, slots=True)
class MaterializedProfileRequest:
    """One deterministic paired belief support reused by profile diagnostics."""

    request: PlannerDecisionRequest
    support_fingerprint: str
    belief_world_count: int


@dataclass(frozen=True, slots=True)
class ProfileOracleResult:
    """Full-support scores or an explicit exact-reference exclusion."""

    compatible: bool
    exclusion_reason: str | None
    scores: Mapping[Action, float]
    candidate_count: int
    scenario_count: int
    engine_transitions: int
    scenario_grid_complete: bool
    rules_exact: bool
    latency_ms: float | None


@dataclass(frozen=True, slots=True)
class ProfileOracleCacheKey:
    """Exact campaign/root/reference identity for one reusable computation."""

    campaign_id: str
    corpus_row_id: str
    support_fingerprint: str
    model_fingerprint: str
    scorer_fingerprint: str
    controller_fingerprint: str
    continuation_fingerprint: str
    tensorizer_fingerprint: str
    native_library_fingerprint: str
    native_abi_fingerprint: str
    native_schema_fingerprint: str
    oracle_config_fingerprint: str

    @property
    def computation_fingerprint(self) -> str:
        """Return the repetition-independent exhaustive-work identity."""
        payload = {field.name: getattr(self, field.name) for field in fields(self)}
        return hashlib.sha256(
            b"ptcg-rl/planner-profile-oracle-computation/v2\x00"
            + json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class ProfileOracleCacheValue:
    """One exact result plus whether this lookup performed the work."""

    result: ProfileOracleResult
    provenance_fingerprint: str
    executed: bool


@dataclass(frozen=True, slots=True)
class ProfileOracleCacheSummary:
    """Bounded diagnostics for dedicated premeasurement oracle work."""

    computations: int
    compatible_computations: int
    excluded_computations: int
    engine_transitions: int
    elapsed_seconds: float

    def as_dict(self) -> dict[str, int | float]:
        """Return a JSON-native campaign summary."""
        return {
            "computations": self.computations,
            "compatible_computations": self.compatible_computations,
            "excluded_computations": self.excluded_computations,
            "engine_transitions": self.engine_transitions,
            "elapsed_seconds": self.elapsed_seconds,
        }


class ProfileOracleCache:
    """Campaign-scoped immutable oracle cache shared across all profile points."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, Future[ProfileOracleResult]] = {}

    def get_or_compute(
        self,
        key: ProfileOracleCacheKey,
        compute: Callable[[], ProfileOracleResult],
        *,
        corpus_repetition: int,
    ) -> ProfileOracleCacheValue:
        """Execute one support once while retaining consumer provenance."""
        computation = key.computation_fingerprint
        provenance = profile_oracle_reference_fingerprint(
            key,
            corpus_repetition=corpus_repetition,
        )
        with self._lock:
            future = self._entries.get(computation)
            executed = future is None
            if future is None:
                future = Future()
                self._entries[computation] = future
        if executed:
            try:
                future.set_result(compute())
            except BaseException as exc:
                future.set_exception(exc)
                with self._lock:
                    if self._entries.get(computation) is future:
                        del self._entries[computation]
                raise
        return ProfileOracleCacheValue(
            result=future.result(),
            provenance_fingerprint=provenance,
            executed=executed,
        )

    def summary(self) -> ProfileOracleCacheSummary:
        """Return completed unique computation diagnostics."""
        with self._lock:
            futures = tuple(self._entries.values())
        if any(not future.done() for future in futures):
            raise RuntimeError("profile oracle cache still contains pending work")
        results = tuple(future.result() for future in futures)
        compatible = tuple(result for result in results if result.compatible)
        return ProfileOracleCacheSummary(
            computations=len(results),
            compatible_computations=len(compatible),
            excluded_computations=len(results) - len(compatible),
            engine_transitions=sum(result.engine_transitions for result in compatible),
            elapsed_seconds=sum(result.latency_ms or 0.0 for result in compatible)
            / 1_000.0,
        )


def profile_oracle_reference_fingerprint(
    key: ProfileOracleCacheKey,
    *,
    corpus_repetition: int,
) -> str:
    """Bind one repeated evidence row to its shared exact computation."""
    if corpus_repetition < 0:
        raise ValueError("profile oracle repetition must be non-negative")
    payload = {
        "computation_fingerprint": key.computation_fingerprint,
        "corpus_repetition": int(corpus_repetition),
    }
    return hashlib.sha256(
        b"ptcg-rl/planner-profile-oracle-reference/v2\x00"
        + json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def oracle_config_fingerprint(config: PlannerProfileOracleConfig) -> str:
    """Bind all exhaustive-only work, pool, scoring, and timeout geometry."""
    return hashlib.sha256(
        b"ptcg-rl/planner-profile-oracle-config/v1\x00"
        + json.dumps(
            config.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def profile_oracle_controller_identity(
    *,
    runtime: ResolvedPlannerRuntimeConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    oracle_config: PlannerProfileOracleConfig,
) -> StableContinuationControllerIdentity:
    """Resolve continuation identity without inheriting candidate-K semantics."""
    oracle_constructor = hashlib.sha256(
        b"ptcg-rl/planner-profile-oracle-constructor/v1\x00"
        + bytes.fromhex(oracle_config_fingerprint(oracle_config))
    ).hexdigest()
    return StableContinuationControllerIdentity.create(
        controller_version=runtime.controller_version,
        constructor_fingerprint=oracle_constructor,
        scorer_fingerprint=resolved.static.scorer_fingerprint,
        continuation_semantics_fingerprint=(
            resolved.continuation_semantics_fingerprint
        ),
    )


def materialize_profile_request(
    *,
    row: PlannerRootRow,
    request_factory: PlannerDecisionRequestFactory,
    identity: ResolvedPlannerRuntimeIdentity,
) -> MaterializedProfileRequest:
    """Build the same deterministic public posterior support as production."""
    request = request_factory.build(
        row,
        greedy_action=row.base_action,
        proposal_actions=(),
        identity=identity.runtime_identity,
    )
    support, scenarios = normalize_scenario_support(
        request.scenarios,
        mode=ScenarioSupportMode.SAMPLED_BELIEF_NO_CHANCE,
    )
    return MaterializedProfileRequest(
        request=request,
        support_fingerprint=support.support_fingerprint,
        belief_world_count=len(scenarios),
    )


def run_profile_oracle(
    *,
    materialized: MaterializedProfileRequest,
    runtime: ResolvedPlannerRuntimeConfig,
    resolved: ResolvedPlannerRuntimeIdentity,
    session_pool: NativePlanningSessionLanePool,
    inference_session: PlannerInferenceSession,
    own_deck: Sequence[int],
    oracle_config: PlannerProfileOracleConfig,
    excluded_reason: str | None = None,
) -> ProfileOracleResult:
    """Execute current v5 continuation/chance semantics on every legal action."""
    request = materialized.request
    if excluded_reason is not None:
        return _excluded(excluded_reason)
    legal_count = request.budget_request.legal_action_count
    if legal_count > oracle_config.max_legal_actions:
        return _excluded("legal_action_count_above_preregistered_cap")
    candidates = build_prompt_action_candidates(
        request.root_observation.get("select"),
        greedy_action=request.base_action,
        exhaustive_action_cap=oracle_config.max_legal_actions,
        beam_width=oracle_config.max_legal_actions,
    )
    if not candidates.exhaustive or len(candidates.actions) != legal_count:
        raise RuntimeError("profile oracle did not enumerate the full legal support")

    started = time.perf_counter()
    deadline = time.monotonic() + oracle_config.timeout_seconds
    controller = _SessionContinuationProvider(
        session=inference_session,
        own_deck=tuple(int(value) for value in own_deck),
    )
    executor = HierarchicalPlanningSessionExecutor(
        config=runtime.hierarchical_search,
        pool=session_pool,
        controller=controller,
        controller_identity=profile_oracle_controller_identity(
            runtime=runtime,
            resolved=resolved,
            oracle_config=oracle_config,
        ),
    )
    scorer = SharedRootInformationLeafScorer[RootInformationModelInputBatch](
        config=runtime.scoring,
        tensorizer=ProductionRootInformationTensorizer(runtime.tensorizer),
        value_provider=inference_session.root_value_provider(
            root_deck=own_deck,
            max_rows=oracle_config.root_value_microbatch_rows,
        ),
    )
    ledger = PlannerRequestLedger(
        oracle_config.work_limits,
        deadline_monotonic=deadline,
        started_monotonic=time.monotonic(),
    )
    full_request = _hierarchical_request(
        request,
        actions=candidates.actions,
        support_exhaustive=True,
    )
    try:
        scored = _score_in_vram_safe_shards(
            full_request,
            executor=executor,
            scorer=scorer,
            ledger=ledger,
            maximum_native_rows=(oracle_config.session_pool.max_transitions_per_call),
        )
    except PlannerEvidenceError as exc:
        exclusion = _planner_error_exclusion(exc)
        if exclusion is not None:
            return exclusion
        raise
    elapsed_ms = max((time.perf_counter() - started) * 1_000.0, 1.0e-9)
    evidence = scored.search_evidence
    if not evidence.exhaustive or len(evidence.candidates) != legal_count:
        raise RuntimeError("profile oracle scored a non-exhaustive support")
    ledger_snapshot = ledger.snapshot()
    if ledger_snapshot.failed_reservations:
        raise RuntimeError("profile oracle contains failed work reservations")
    if ledger_snapshot.stop_reason is not PlannerWorkStopReason.ACTIVE:
        raise RuntimeError(
            "profile oracle exhausted its preregistered work contract: "
            f"{ledger_snapshot.stop_reason}"
        )
    rules_exact = all(evidence.rules_exact)
    if not rules_exact:
        return _excluded("v5_exact_reference_rules_inexact")
    # The aggregate object is the source of truth for robust score location.
    scores = {
        action: float(aggregate.robust_score)
        for action, aggregate in zip(
            evidence.actions,
            scored.aggregation.candidates,
            strict=True,
        )
    }
    _prefix_reuse, native_rows = executor.runtime_stats
    if native_rows <= 0:
        raise RuntimeError("profile oracle emitted no native engine work")
    return ProfileOracleResult(
        compatible=True,
        exclusion_reason=None,
        scores=scores,
        candidate_count=legal_count,
        scenario_count=evidence.scenario_count,
        engine_transitions=native_rows,
        scenario_grid_complete=True,
        rules_exact=True,
        latency_ms=elapsed_ms,
    )


def exclude_profile_oracle(reason: str) -> ProfileOracleResult:
    """Return an explicit no-work result for a preregistered exclusion."""
    return _excluded(reason)


def _planner_error_exclusion(
    error: PlannerEvidenceError,
) -> ProfileOracleResult | None:
    """Convert safe serving fallbacks into explicitly unranked oracle roots."""
    if error.reason in _ORACLE_UNRANKED_REASONS:
        reason = error.reason.name.lower()
        return _excluded(f"v5_exact_reference_{reason}")
    return None


def _score_in_vram_safe_shards(
    request: HierarchicalSearchRequest,
    *,
    executor: HierarchicalPlanningSessionExecutor,
    scorer: SharedRootInformationLeafScorer[RootInformationModelInputBatch],
    ledger: PlannerRequestLedger,
    maximum_native_rows: int,
) -> ScoredHierarchicalEvidence[RootInformationModelInputBatch]:
    scenario_count = len(request.scenarios)
    candidates_per_shard = maximum_native_rows // scenario_count
    if candidates_per_shard <= 0:
        raise ValueError("one oracle candidate cannot fit the native row cap")
    if len(request.candidate_actions) <= candidates_per_shard:
        outcomes = executor.evaluate(request, ledger=ledger)
        return score_hierarchical_outcomes(
            outcomes,
            scorer=scorer,
            legal_action_count=request.legal_action_count,
            support_exhaustive=True,
        )
    parts: list[ScoredHierarchicalEvidence[RootInformationModelInputBatch]] = []
    for start in range(0, len(request.candidate_actions), candidates_per_shard):
        actions = request.candidate_actions[start : start + candidates_per_shard]
        shard = _hierarchical_request(
            request,
            actions=actions,
            support_exhaustive=False,
        )
        outcomes = executor.evaluate(shard, ledger=ledger)
        parts.append(
            score_hierarchical_outcomes(
                outcomes,
                scorer=scorer,
                legal_action_count=request.legal_action_count,
                support_exhaustive=False,
            )
        )
    return merge_scored_hierarchical_evidence(
        tuple(parts),
        scorer=scorer,
        legal_action_count=request.legal_action_count,
        support_exhaustive=True,
        producer_contract_fingerprint=(executor.producer_contract_fingerprint(request)),
    )


def _hierarchical_request(
    source: PlannerDecisionRequest | HierarchicalSearchRequest,
    *,
    actions: Sequence[Sequence[int]],
    support_exhaustive: bool,
) -> HierarchicalSearchRequest:
    legal_action_count = (
        source.budget_request.legal_action_count
        if isinstance(source, PlannerDecisionRequest)
        else source.legal_action_count
    )
    return HierarchicalSearchRequest.from_sequences(
        source.state_token,
        root_observation=source.root_observation,
        scenarios=source.scenarios,
        candidate_actions=actions,
        legal_action_count=legal_action_count,
        root_player=source.root_player,
        context_snapshot=source.context_snapshot,
        belief_feature_producer=source.belief_feature_producer,
        belief_summary_width=source.belief_summary_width,
        producer_context=source.producer_context,
        belief_summary=source.belief_summary,
        producer_contract_fingerprint=source.producer_contract_fingerprint,
        support_exhaustive=support_exhaustive,
    )


@dataclass(slots=True)
class _SessionContinuationProvider:
    session: PlannerInferenceSession
    own_deck: tuple[int, ...]

    def select_actions(
        self,
        prompts: tuple[ContinuationPrompt, ...],
    ) -> Sequence[Sequence[int]]:
        return tuple(
            tuple(int(value) for value in action)
            for action in self.session.continuation_actions(
                prompts,
                root_deck=self.own_deck,
            )
        )


def _excluded(reason: str) -> ProfileOracleResult:
    return ProfileOracleResult(
        compatible=False,
        exclusion_reason=reason,
        scores={},
        candidate_count=0,
        scenario_count=0,
        engine_transitions=0,
        scenario_grid_complete=False,
        rules_exact=False,
        latency_ms=None,
    )


__all__ = [
    "MaterializedProfileRequest",
    "ProfileOracleCache",
    "ProfileOracleCacheKey",
    "ProfileOracleCacheSummary",
    "ProfileOracleCacheValue",
    "ProfileOracleResult",
    "exclude_profile_oracle",
    "materialize_profile_request",
    "oracle_config_fingerprint",
    "profile_oracle_controller_identity",
    "profile_oracle_reference_fingerprint",
    "run_profile_oracle",
]
