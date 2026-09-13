"""Campaign-scoped exact oracle preparation for planner profile backends."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, cast

from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.engine.native_planning_session_pool import (
    NativePlanningSessionLanePool,
)
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfilePointConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusReader,
    PlannerProfileCorpusRecord,
)
from ptcg_rl.rl.planner_inference_session import (
    PlannerInferenceLease,
    PlannerInferenceSession,
)
from ptcg_rl.rl.planner_profile_inputs import collate_profile_roots
from ptcg_rl.rl.planner_profile_oracle import (
    MaterializedProfileRequest,
    ProfileOracleCache,
    ProfileOracleCacheKey,
    ProfileOracleCacheValue,
    ProfileOracleResult,
    exclude_profile_oracle,
    materialize_profile_request,
    oracle_config_fingerprint,
    profile_oracle_controller_identity,
    run_profile_oracle,
)
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity
from ptcg_rl.rl.planner_service_inputs import (
    PlannerDecisionRequestFactory,
    PlannerRootRow,
)


@dataclass(frozen=True, slots=True)
class ProfileOracleReference:
    """Paired scenario support and one cached exact quality reference."""

    materialized: MaterializedProfileRequest
    cache_value: ProfileOracleCacheValue


class ProfileOracleReferenceStore:
    """Prepare each exact serving-model reference once before measurement."""

    def __init__(
        self,
        *,
        config: IntegratedPlannerProfileConfig,
        point: PlannerProfilePointConfig,
        runtime: PlannerProfileRuntimeConfig,
        resolved: ResolvedPlannerRuntimeIdentity,
        cache: ProfileOracleCache,
        policy: Any,
        sampler: BeliefSampler,
        producer: OpponentBeliefFeatureProducer,
        model_deck: tuple[int, ...] | None,
    ) -> None:
        self._config = config
        self._point = point
        self._runtime = runtime
        self._resolved = resolved
        self._cache = cache
        self._policy = policy
        self._sampler = sampler
        self._producer = producer
        self._model_deck = model_deck
        self._pool = NativePlanningSessionLanePool(
            config.oracle.session_pool,
            library_path=config.native_library_path,
        )
        self._references: dict[tuple[int, str], ProfileOracleReference] = {}
        self._closed = False

    def prepare(self) -> None:
        """Populate point lookups; cache hits perform no model or engine work."""
        reader = PlannerProfileCorpusReader(
            self._config.decision_corpus_path,
            expected_sha256=self._config.expected_decision_corpus_sha256,
        )
        for repetition in range(self._point.decision_repetitions):
            for record in reader:
                materialized = self.materialize(
                    record,
                    base_action=record.executed_action,
                    base_old_logprob=0.0,
                )
                key = self._cache_key(
                    record=record,
                    materialized=materialized,
                )

                def execute_reference(
                    source: PlannerProfileCorpusRecord = record,
                    request: MaterializedProfileRequest = materialized,
                ) -> ProfileOracleResult:
                    return self._execute(source, materialized=request)

                value = self._cache.get_or_compute(
                    key,
                    execute_reference,
                    corpus_repetition=repetition,
                )
                self._references[(repetition, record.row_id)] = ProfileOracleReference(
                    materialized=materialized,
                    cache_value=value,
                )

    def reference_for(
        self,
        repetition: int,
        row_id: str,
    ) -> ProfileOracleReference:
        """Return one prepared point-local lookup."""
        try:
            return self._references[(repetition, row_id)]
        except KeyError as exc:
            raise KeyError("profile oracle reference was not prepared") from exc

    def materialize(
        self,
        record: PlannerProfileCorpusRecord,
        *,
        base_action: tuple[int, ...],
        base_old_logprob: float,
    ) -> MaterializedProfileRequest:
        """Build the deterministic public posterior support for one root."""
        factory = PlannerDecisionRequestFactory(
            belief_sampler=self._sampler,
            scenario_count=self._runtime.planner.scenario.belief_world_count,
            belief_summary_dim=self._runtime.planner.tensorizer.belief_summary_dim,
            costs=self._runtime.planner.request_costs,
            producer_contract_fingerprint=bytes.fromhex(
                self._resolved.continuation_semantics_fingerprint
            ),
            belief_feature_producer=self._producer,
            stochastic_seed=self._runtime.belief.stochastic_seed,
        )
        prepared = collate_profile_roots(
            (record,),
            producer=self._producer,
            device="cpu",
            model_deck=self._model_deck,
        ).prepared[0]
        return materialize_profile_request(
            row=PlannerRootRow(
                row_id=record.row_id,
                seat=_record_seat(record),
                observation=prepared.observation,
                context_features=prepared.context_features,
                context_snapshot=record.context_snapshot,
                own_deck=record.own_deck,
                base_action=base_action,
                base_old_logprob=base_old_logprob,
            ),
            request_factory=factory,
            identity=self._resolved,
        )

    @property
    def executions(self) -> tuple[ProfileOracleCacheValue, ...]:
        """Return only compatible work executed by this point."""
        return tuple(
            reference.cache_value
            for reference in self._references.values()
            if reference.cache_value.executed
            and reference.cache_value.result.compatible
        )

    @property
    def cache_misses(self) -> tuple[ProfileOracleCacheValue, ...]:
        """Return every computation performed while preparing this store."""
        return tuple(
            reference.cache_value
            for reference in self._references.values()
            if reference.cache_value.executed
        )

    def close(self) -> None:
        """Release the reference-only native lane exactly once."""
        if self._closed:
            return
        self._closed = True
        self._pool.close()

    def _execute(
        self,
        record: PlannerProfileCorpusRecord,
        *,
        materialized: MaterializedProfileRequest,
    ) -> ProfileOracleResult:
        exclusion = _oracle_exclusion(
            record,
            legal_action_count=(materialized.request.budget_request.legal_action_count),
            maximum_legal_actions=self._config.oracle.max_legal_actions,
        )
        if exclusion is not None:
            return exclude_profile_oracle(exclusion)
        collated = collate_profile_roots(
            (record,),
            producer=self._producer,
            device="cpu",
            model_deck=self._model_deck,
        )
        trace = _decode_with_retained_context(self._policy, collated)
        if len(trace.planner_context_handles) != 1:
            raise RuntimeError("profile oracle decode omitted its root context")
        session = PlannerInferenceSession(
            policy=self._policy,
            states=collated.states,
            options=collated.options,
            decks=collated.decks,
            context_handles=trace.planner_context_handles,
            lease=PlannerInferenceLease(
                model_fingerprint=self._resolved.runtime_identity.model_fingerprint,
                policy_version=self._resolved.runtime_identity.policy_version,
                tensor_schema_fingerprint=self._resolved.tensor_schema_fingerprint,
                deadline_monotonic=(
                    time.monotonic() + self._config.oracle.timeout_seconds + 30.0
                ),
                inference_device_type=cast(
                    Literal["cpu", "cuda"],
                    self._policy.planner_inference_device_type,
                ),
                inference_timeout_seconds=(
                    self._runtime.planner.deadlines.inference_timeout_seconds
                ),
            ),
        )
        try:
            return run_profile_oracle(
                materialized=materialized,
                runtime=self._runtime.planner,
                resolved=self._resolved,
                session_pool=self._pool,
                inference_session=session,
                own_deck=(self._model_deck or record.own_deck),
                oracle_config=self._config.oracle,
            )
        finally:
            session.release_rows(
                (0,),
                deadline_monotonic=(
                    time.monotonic()
                    + self._runtime.planner.deadlines.cleanup_timeout_seconds
                ),
            )

    def _cache_key(
        self,
        *,
        record: PlannerProfileCorpusRecord,
        materialized: MaterializedProfileRequest,
    ) -> ProfileOracleCacheKey:
        controller = profile_oracle_controller_identity(
            runtime=self._runtime.planner,
            resolved=self._resolved,
            oracle_config=self._config.oracle,
        )
        static = self._resolved.static
        engine = self._runtime.planner.engine
        return ProfileOracleCacheKey(
            campaign_id=self._config.campaign_id,
            corpus_row_id=record.row_id,
            support_fingerprint=materialized.support_fingerprint,
            model_fingerprint=self._resolved.runtime_identity.model_fingerprint,
            scorer_fingerprint=static.scorer_fingerprint,
            controller_fingerprint=controller.controller_fingerprint,
            continuation_fingerprint=static.continuation_semantics_fingerprint,
            tensorizer_fingerprint=static.tensorizer_fingerprint,
            native_library_fingerprint=engine.library_fingerprint,
            native_abi_fingerprint=engine.native_abi_fingerprint,
            native_schema_fingerprint=engine.native_schema_fingerprint,
            oracle_config_fingerprint=oracle_config_fingerprint(self._config.oracle),
        )


def _decode_with_retained_context(policy: Any, collated: Any) -> Any:
    submit = getattr(policy, "submit_decode_for_request", None)
    if callable(submit):
        handle = submit(
            collated.states,
            collated.options,
            collated.decks,
            temperature=0.0,
            retain_planner_context=True,
        )
        return policy.receive_decode_with_trace(handle)
    return policy.sample_decode_with_trace_for_request(
        collated.states,
        collated.options,
        collated.decks,
        temperature=0.0,
        model_version_lease=None,
        retain_planner_context=True,
    )


def _oracle_exclusion(
    record: PlannerProfileCorpusRecord,
    *,
    legal_action_count: int,
    maximum_legal_actions: int,
) -> str | None:
    if "engine_chance" in record.shapes:
        return "engine_rng_consumption_not_exactly_replayable"
    if legal_action_count > maximum_legal_actions:
        return "legal_action_count_above_preregistered_cap"
    return None


def _record_seat(record: PlannerProfileCorpusRecord) -> int:
    seat = record.context_snapshot.player_index
    if seat not in (0, 1):
        raise ValueError("planner profile root has no exact player index")
    return int(seat)


__all__ = ["ProfileOracleReference", "ProfileOracleReferenceStore"]
