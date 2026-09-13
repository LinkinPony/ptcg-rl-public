"""Production-service adapter for integrated planner profile points."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from ptcg_rl.agent.runtime import PolicyRuntimeAgent
from ptcg_rl.evaluation.planner_profile_config import (
    IntegratedPlannerProfileConfig,
    PlannerProfilePointConfig,
    PlannerProfileRuntimeConfig,
)
from ptcg_rl.evaluation.planner_profile_corpus_reader import (
    PlannerProfileCorpusReader,
    PlannerProfileCorpusRecord,
)
from ptcg_rl.evaluation.planner_profile_metrics import PlannerProfileSink
from ptcg_rl.evaluation.planner_profile_package_models import (
    PlannerProfilePackageAssetsValidation,
)
from ptcg_rl.evaluation.planner_profile_records import (
    PlannerProfileDecisionRecord,
    PlannerProfileRunRecord,
)
from ptcg_rl.rl.planner_profile_oracle import ProfileOracleCache
from ptcg_rl.rl.planner_runtime_identity import ResolvedPlannerRuntimeIdentity
from ptcg_rl.rl.rollout import RolloutPlannerBehaviorService
from ptcg_rl.runtime.planner_telemetry import PlannerStageEvent


@dataclass(frozen=True)
class PlannerProfileMeasuredDecision:
    """One production behavior result and request-local stage telemetry."""

    record: PlannerProfileDecisionRecord
    events: tuple[PlannerStageEvent, ...]


class PlannerProfileProductionBackend(Protocol):
    """Narrow adapter implemented beside the shared production service."""

    @property
    def planner_service(self) -> RolloutPlannerBehaviorService:
        """Return the exact shared service used by rollout and packaged paths."""

    @property
    def packaged_agent(self) -> PolicyRuntimeAgent | None:
        """Return the actual packaged agent for CPU points, otherwise ``None``."""

    def execute_batch(
        self,
        records: Sequence[PlannerProfileCorpusRecord],
        *,
        decision_index: int,
        planner_enabled: bool,
    ) -> Sequence[PlannerProfileMeasuredDecision]:
        """Execute base decode plus shared planner/agent work for aligned roots."""

    def finish_run(self, *, decisions: int) -> PlannerProfileRunRecord:
        """Stop concurrent work and return measured rates/resources."""

    def close(self) -> None:
        """Release native lanes, model leases, agents, and learner workers."""


class ProductionPlannerProfilePointRunner:
    """Stream one immutable corpus through a real production backend."""

    def __init__(
        self,
        config: IntegratedPlannerProfileConfig,
        *,
        package_validation: PlannerProfilePackageAssetsValidation,
    ) -> None:
        self._config = config
        self._package_validation = package_validation
        self._oracle_cache = ProfileOracleCache()
        self._oracles_prepared = False

    def prepare_oracles(self) -> dict[str, int | float]:
        """Populate exact references in discarded, premeasurement backends."""
        if self._oracles_prepared:
            raise RuntimeError("profile oracles were already prepared")
        environments: set[str] = set()
        for point in self._config.points:
            if point.environment in environments:
                continue
            environments.add(point.environment)
            runtime = self._config.runtime_for(point)
            backend = _load_production_backend(
                self._config,
                point=point,
                runtime=runtime,
                oracle_cache=self._oracle_cache,
                package_validation=self._package_validation,
                oracle_precompute_only=True,
            )
            backend.close()
        summary = self._oracle_cache.summary()
        if summary.compatible_computations <= 0:
            raise RuntimeError("profile oracle precompute produced no exact references")
        self._oracles_prepared = True
        return summary.as_dict()

    def execute(
        self,
        point: PlannerProfilePointConfig,
        sink: PlannerProfileSink,
    ) -> PlannerProfileRunRecord:
        """Execute one fixed point without buffering the decision corpus."""
        if not self._oracles_prepared:
            raise RuntimeError("profile oracles must be prepared before measurement")
        runtime = self._config.runtime_for(point)
        reader = PlannerProfileCorpusReader(
            self._config.decision_corpus_path,
            expected_sha256=self._config.expected_decision_corpus_sha256,
        )
        backend = _load_production_backend(
            self._config,
            point=point,
            runtime=runtime,
            oracle_cache=self._oracle_cache,
            package_validation=self._package_validation,
            oracle_precompute_only=False,
        )
        model_identity = self._config.model_identity_for(point)
        expected_identity = runtime.planner.resolve_for_lease(
            model_fingerprint=model_identity.model_fingerprint,
            policy_version=model_identity.policy_version,
            proposal_version=model_identity.proposal_version,
        )
        decision_index = 0
        try:
            self._validate_backend_surface(backend, point=point)
            for batch in _corpus_batches(
                reader,
                repetitions=point.decision_repetitions,
                batch_size=point.request_batch_size,
            ):
                measured = tuple(
                    backend.execute_batch(
                        batch,
                        decision_index=decision_index,
                        planner_enabled=point.planner_enabled,
                    )
                )
                if len(measured) != len(batch):
                    raise RuntimeError(
                        "production profile backend returned a misaligned batch"
                    )
                for offset, (record, item) in enumerate(
                    zip(batch, measured, strict=True)
                ):
                    expected_index = decision_index + offset
                    _validate_measured_decision(
                        item,
                        source=record,
                        expected_index=expected_index,
                        expected_repetition=expected_index // reader.rows,
                        expected_identity=expected_identity,
                        expected_batch_row_position=(
                            offset % runtime.planner.batching.max_root_rows_per_request
                        ),
                        oracle_max_legal_actions=(
                            self._config.oracle.max_legal_actions
                        ),
                        oracle_epsilon=self._config.oracle.epsilon_regret,
                        planner_enabled=point.planner_enabled,
                    )
                    sink.record(item.record, item.events)
                decision_index += len(batch)
            run = backend.finish_run(decisions=decision_index)
        finally:
            backend.close()
        if decision_index <= 0:
            raise RuntimeError("production profile point emitted no decisions")
        _validate_run_workload(
            run,
            config=self._config,
            point=point,
            runtime=runtime,
            decisions=decision_index,
        )
        return run

    @staticmethod
    def _validate_backend_surface(
        backend: PlannerProfileProductionBackend,
        *,
        point: PlannerProfilePointConfig,
    ) -> None:
        service = backend.planner_service
        if not callable(getattr(service, "plan_batch", None)):
            raise TypeError("profile backend does not expose PlannerBehaviorService")
        agent = backend.packaged_agent
        if point.environment == "packaged_cpu_acttime":
            if not isinstance(agent, PolicyRuntimeAgent):
                raise TypeError("packaged profile must use PolicyRuntimeAgent")
        elif agent is not None:
            raise TypeError("H200 rollout profile cannot substitute a packaged agent")


class _BackendContext:
    """Normalize a backend factory that optionally returns a context manager."""

    def __init__(self, value: Any) -> None:
        self._raw = value
        self._entered = False
        self._backend: PlannerProfileProductionBackend | None = None

    def enter(self) -> PlannerProfileProductionBackend:
        enter = getattr(self._raw, "__enter__", None)
        value = enter() if callable(enter) else self._raw
        self._entered = callable(enter)
        required = (
            "planner_service",
            "packaged_agent",
            "execute_batch",
            "finish_run",
            "close",
        )
        if any(not hasattr(value, name) for name in required):
            if self._entered:
                exit_method = getattr(self._raw, "__exit__", None)
                if callable(exit_method):
                    exit_method(None, None, None)
            raise TypeError("production profile factory returned another interface")
        self._backend = cast(PlannerProfileProductionBackend, value)
        return self._backend

    def close(self) -> None:
        if self._backend is None:
            return
        if self._entered:
            exit_method = getattr(self._raw, "__exit__", None)
            if callable(exit_method):
                exit_method(None, None, None)
                self._backend = None
                return
        self._backend.close()
        self._backend = None


class _ManagedProductionBackend:
    """Delegate calls while preserving one factory context lifecycle."""

    def __init__(self, context: _BackendContext) -> None:
        self._context = context
        self._backend = context.enter()

    @property
    def planner_service(self) -> RolloutPlannerBehaviorService:
        return self._backend.planner_service

    @property
    def packaged_agent(self) -> PolicyRuntimeAgent | None:
        return self._backend.packaged_agent

    def execute_batch(
        self,
        records: Sequence[PlannerProfileCorpusRecord],
        *,
        decision_index: int,
        planner_enabled: bool,
    ) -> Sequence[PlannerProfileMeasuredDecision]:
        return self._backend.execute_batch(
            records,
            decision_index=decision_index,
            planner_enabled=planner_enabled,
        )

    def finish_run(self, *, decisions: int) -> PlannerProfileRunRecord:
        return self._backend.finish_run(decisions=decisions)

    def close(self) -> None:
        self._context.close()


def _load_production_backend(
    config: IntegratedPlannerProfileConfig,
    *,
    point: PlannerProfilePointConfig,
    runtime: PlannerProfileRuntimeConfig,
    oracle_cache: ProfileOracleCache,
    package_validation: PlannerProfilePackageAssetsValidation,
    oracle_precompute_only: bool,
) -> PlannerProfileProductionBackend:
    module_name, factory_name = config.production_backend_factory.split(":", 1)
    factory = getattr(importlib.import_module(module_name), factory_name, None)
    if not callable(factory):
        raise TypeError("production profile backend factory is not callable")
    typed = cast(Callable[..., Any], factory)
    raw = typed(
        config=config,
        point=point,
        runtime=runtime,
        oracle_cache=oracle_cache,
        package_validation=package_validation,
        oracle_precompute_only=oracle_precompute_only,
    )
    return _ManagedProductionBackend(_BackendContext(raw))


def _corpus_batches(
    reader: PlannerProfileCorpusReader,
    *,
    repetitions: int,
    batch_size: int,
) -> Iterator[tuple[PlannerProfileCorpusRecord, ...]]:
    pending: list[PlannerProfileCorpusRecord] = []
    for _repetition in range(repetitions):
        for record in reader:
            pending.append(record)
            if len(pending) == batch_size:
                yield tuple(pending)
                pending.clear()
    if pending:
        yield tuple(pending)


def _validate_measured_decision(
    measured: PlannerProfileMeasuredDecision,
    *,
    source: PlannerProfileCorpusRecord,
    expected_index: int,
    expected_repetition: int,
    expected_identity: ResolvedPlannerRuntimeIdentity,
    expected_batch_row_position: int,
    oracle_max_legal_actions: int,
    oracle_epsilon: float,
    planner_enabled: bool,
) -> None:
    record = measured.record
    if record.decision_index != expected_index:
        raise ValueError("profile backend returned a noncontiguous decision index")
    if record.corpus_repetition != expected_repetition:
        raise ValueError("profile backend returned another corpus repetition")
    if record.batch_row_position != expected_batch_row_position:
        raise ValueError("profile backend changed actor-local row position")
    if record.corpus_row_id != source.row_id:
        raise ValueError("profile backend returned another corpus root")
    if tuple(sorted(record.decision_shapes)) != tuple(sorted(source.shapes)):
        raise ValueError("profile backend changed the corpus decision shapes")
    identity = expected_identity.runtime_identity
    if (
        record.model_fingerprint != identity.model_fingerprint
        or record.runtime_fingerprint != expected_identity.runtime_fingerprint
        or record.planner_fingerprint != identity.planner_fingerprint
        or record.policy_version != identity.policy_version
        or record.proposal_version != identity.proposal_version
    ):
        raise ValueError("profile decision differs from the resolved point lease")
    if record.scenario_support_fingerprint is None:
        raise ValueError("profile decision omitted paired scenario support identity")
    if "engine_chance" in source.shapes and record.oracle_compatible:
        raise ValueError("engine RNG-consumption roots are not exact-oracle compatible")
    if record.served_action_regret is not None:
        if record.legal_action_count > oracle_max_legal_actions:
            raise ValueError("profile backend exceeded the preregistered oracle cap")
        expected_served = record.served_action_regret <= oracle_epsilon
        if record.served_epsilon_optimal != expected_served:
            raise ValueError("profile backend returned inconsistent served epsilon")
        if record.candidate_best_regret is not None:
            expected_candidate = record.candidate_best_regret <= oracle_epsilon
            if record.candidate_epsilon_recall != expected_candidate:
                raise ValueError(
                    "profile backend returned inconsistent candidate recall"
                )
    elif (
        record.legal_action_count <= oracle_max_legal_actions
        and record.oracle_compatible
    ):
        raise ValueError("profile backend omitted a compatible oracle reference")
    if not planner_enabled and record.planner_used:
        raise ValueError("planner-off control emitted planner behavior")


def _validate_run_workload(
    run: PlannerProfileRunRecord,
    *,
    config: IntegratedPlannerProfileConfig,
    point: PlannerProfilePointConfig,
    runtime: PlannerProfileRuntimeConfig,
    decisions: int,
) -> None:
    if run.decisions != decisions:
        raise ValueError("profile run counters differ from streamed decisions")
    if point.environment == "h200_mps":
        learner = runtime.learner
        if learner is None:
            raise ValueError("H200 point has no configured learner workload")
        if (
            run.learner_warmup_updates != learner.warmup_updates
            or run.learner_updates != learner.updates
            or run.learner_optimizer_steps != learner.updates * learner.ppo_epochs
            or run.learner_kernel_rows
            != learner.updates * learner.kernel_rows_per_update
        ):
            raise ValueError("H200 profile differs from exact learner workload")
        if run.policy_runtime_agent_calls != 0:
            raise ValueError("H200 profile mixed in packaged agent calls")
        if (
            run.policy_runtime_agent_warmup_calls
            or run.packaged_isolated_processes
            or run.act_time_replay_episodes
            or run.act_time_replay_decisions
        ):
            raise ValueError("H200 profile claimed packaged workload evidence")
    else:
        packaged = runtime.packaged
        if packaged is None:
            raise ValueError("packaged point has no configured agent workload")
        if (
            run.learner_warmup_updates
            or run.learner_updates
            or run.learner_optimizer_steps
            or run.learner_kernel_rows
        ):
            raise ValueError("packaged profile cannot claim H200 learner work")
        expected_seat_episodes = len(config.act_time_replay.assets) * len(
            config.act_time_replay.seats
        )
        if run.policy_runtime_agent_warmup_calls != packaged.warmup_decisions:
            raise ValueError("packaged profile differs from configured warmup work")
        if run.act_time_replay_episodes != expected_seat_episodes:
            raise ValueError("packaged profile omitted ordered ActTime seat episodes")
        if run.packaged_isolated_processes != expected_seat_episodes:
            raise ValueError("packaged ActTime episodes were not process-isolated")
        expected_replay_calls = sum(
            asset.active_callbacks_by_seat[seat]
            for asset in config.act_time_replay.assets
            for seat in config.act_time_replay.seats
        )
        if run.act_time_replay_decisions != expected_replay_calls:
            raise ValueError("packaged profile changed the replay callback workload")
        if run.policy_runtime_agent_calls != run.act_time_replay_decisions:
            raise ValueError("packaged profile bypassed PolicyRuntimeAgent")


__all__ = [
    "PlannerProfileMeasuredDecision",
    "PlannerProfileProductionBackend",
    "ProductionPlannerProfilePointRunner",
]
