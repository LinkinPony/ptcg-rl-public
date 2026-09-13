"""Online engine-grounded complete-action targets for RL actors."""

from __future__ import annotations

import hashlib
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ptcg_rl.agent.search.complete_action_teacher import CompleteActionTeacherTarget
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.belief.state import Determinization
from ptcg_rl.context import OpponentBeliefFeatureProducer
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.constants import SelectContext
from ptcg_rl.rl.bounded_worker import (
    BoundedProcessWorker,
    BoundedWorkerStartupError,
    BoundedWorkerTimeoutError,
    BoundedWorkerUnavailableError,
    BoundedWorkerWarmingError,
    RemoteWorkerError,
)
from ptcg_rl.rl.engine_teacher import (
    EngineTeacherProducer,
    EngineTeacherRequest,
    EngineTeacherTarget,
)
from ptcg_rl.rl.engine_teacher_policy import DecodePolicy
from ptcg_rl.rl.online_engine_teacher_config import OnlineEngineTeacherConfig
from ptcg_rl.rl.online_engine_teacher_process import (
    CompactTeacherResult,
    DecodeCall,
    DecodeReply,
    TeacherDeadlineExpiredError,
    ValueCall,
    ValueReply,
    WorkerInit,
    WorkerTask,
    initialize_worker,
    policy_response_version,
    produce_target_impl,
    sample_worlds_impl,
    search_evidence_from_complete_action_target,
    worker_produce_target,
)
from ptcg_rl.rl.online_engine_teacher_process import (
    world_seed as _process_world_seed,
)

TeacherSearchResult = CompleteActionTeacherTarget | CompactTeacherResult


@dataclass(frozen=True)
class _Candidate:
    request_index: int
    request: EngineTeacherRequest
    decision_seed: int
    priority: int


class OnlineEngineTeacherProducer(EngineTeacherProducer):
    """Sample public belief worlds and distill complete engine action search."""

    def __init__(
        self,
        *,
        policy: DecodePolicy,
        device: torch.device | str | None,
        config: OnlineEngineTeacherConfig,
        seed: int,
        belief_producer: OpponentBeliefFeatureProducer | None = None,
        sampler: BeliefSampler | None = None,
        isolate_process: bool | None = None,
    ) -> None:
        self._policy = policy
        self._device = device
        self._seed = int(seed)
        self._belief_producer = belief_producer
        remote_teacher_policy = getattr(policy, "request_purpose", None) == "teacher"
        self._isolate_process = (
            remote_teacher_policy if isolate_process is None else isolate_process
        )
        active_config = _config_with_absolute_prior(config)
        self._config = active_config
        if self._isolate_process and not remote_teacher_policy:
            raise ValueError(
                "isolated engine teaching requires a bounded remote teacher policy"
            )
        if self._isolate_process and sampler is not None:
            raise ValueError("isolated engine teaching cannot use an injected sampler")
        self._sampler = (
            None
            if self._isolate_process
            else sampler or BeliefSampler(config=active_config.sampler)
        )
        self._worker = (
            BoundedProcessWorker(
                initializer=initialize_worker,
                handler=worker_produce_target,
                initializer_payload=WorkerInit(
                    config=active_config,
                    belief_producer=belief_producer,
                ),
                startup_timeout_seconds=(active_config.worker_startup_timeout_seconds),
                kill_reap_timeout_seconds=(
                    active_config.worker_kill_reap_timeout_seconds
                ),
                max_startup_attempts=active_config.worker_max_startup_attempts,
            )
            if self._isolate_process
            else None
        )
        self._eligible = 0
        self._attempted = 0
        self._emitted = 0
        self._behavior_matches = 0
        self._worlds = 0
        self._nodes = 0
        self._confidence_sum = 0.0
        self._coverage_sum = 0.0
        self._elapsed_seconds = 0.0
        self._step_calls = 0
        self._max_step_seconds = 0.0
        self._probability_skips = 0
        self._attempt_cap_skips = 0
        self._step_budget_skips = 0
        self._deadline_expiries = 0
        self._reason_counts: Counter[str] = Counter()
        self._value_error_counts: Counter[str] = Counter()
        self._error_counts: Counter[str] = Counter()
        self._last_error_detail = ""
        self._context_counts: Counter[str] = Counter()
        if self._worker is not None:
            try:
                self._worker.prepare()
            except BoundedWorkerTimeoutError:
                self._error_counts["worker_initialization"] += 1
                self._worker.restart_in_background()
            except BoundedWorkerStartupError:
                self._worker.close()
                raise

    def produce(self, request: EngineTeacherRequest) -> EngineTeacherTarget | None:
        """Produce one target through the same bounded batch scheduling path."""
        return self.produce_batch((request,))[0]

    def produce_batch(
        self,
        requests: Sequence[EngineTeacherRequest],
    ) -> tuple[EngineTeacherTarget | None, ...]:
        """Produce at most the configured actor-step work in stable priority order."""
        return self._produce_batch(requests, deadline=None)

    def produce_batch_until(
        self,
        requests: Sequence[EngineTeacherRequest],
        *,
        deadline: float,
    ) -> tuple[EngineTeacherTarget | None, ...]:
        """Produce a batch without extending its actor-submission deadline."""
        return self._produce_batch(requests, deadline=float(deadline))

    def screen_batch(
        self,
        requests: Sequence[EngineTeacherRequest],
    ) -> tuple[int, ...]:
        """Return deterministic probability-gate matches without side effects.

        The background lane uses this cheap actor-side screen before enqueueing
        native search. The selected requests are screened again by
        :meth:`produce_batch_until`, which keeps the synchronous producer as the
        single source of scheduling and metrics semantics.
        """
        if not self._config.enabled:
            return ()
        return tuple(
            index
            for index, request in enumerate(requests)
            if self._candidate_for(index, request) is not None
        )

    def _produce_batch(
        self,
        requests: Sequence[EngineTeacherRequest],
        *,
        deadline: float | None,
    ) -> tuple[EngineTeacherTarget | None, ...]:
        results: list[EngineTeacherTarget | None] = [None] * len(requests)
        if not self._config.enabled or not requests:
            return tuple(results)

        step_started_at = time.perf_counter()
        step_deadline = min(
            step_started_at + self._config.max_teacher_seconds_per_actor_step,
            float("inf") if deadline is None else deadline,
        )
        self._step_calls += 1
        if step_started_at >= step_deadline:
            self._step_budget_skips += len(requests)
            self._reason_counts["step_budget_skip"] += len(requests)
            self._record_step_elapsed(step_started_at)
            return tuple(results)
        if self._worker is not None and not self._worker.ready:
            self._worker.restart_in_background()
            self._error_counts["worker_warming"] += 1
            self._reason_counts["teacher_unavailable"] += 1
            self._record_step_elapsed(step_started_at)
            return tuple(results)

        candidates: list[_Candidate] = []
        for index, request in enumerate(requests):
            context = _request_context(request)
            context_label = _context_label(context)
            self._eligible += 1
            self._context_counts[context_label] += 1
            candidate = self._candidate_for(index, request)
            if candidate is None:
                self._probability_skips += 1
                self._reason_counts["probability_skip"] += 1
                continue
            candidates.append(candidate)

        ordered = sorted(
            candidates,
            key=lambda item: (
                item.priority,
                item.request.game_id,
                item.request.seat,
                item.decision_seed,
            ),
        )
        selected = ordered[: self._config.max_attempts_per_actor_step]
        cap_skips = len(ordered) - len(selected)
        self._attempt_cap_skips += cap_skips
        if cap_skips:
            self._reason_counts["attempt_cap_skip"] += cap_skips

        for offset, candidate in enumerate(selected):
            now = time.perf_counter()
            if now >= step_deadline:
                remaining = len(selected) - offset
                self._step_budget_skips += remaining
                self._reason_counts["step_budget_skip"] += remaining
                break
            self._attempted += 1
            request_deadline = min(
                step_deadline,
                now + self._config.deadline_seconds,
            )
            try:
                result = self._produce_target(
                    candidate.request,
                    decision_seed=candidate.decision_seed,
                    deadline=request_deadline,
                )
            except TeacherDeadlineExpiredError as exc:
                reason = _unavailable_reason_label(exc.reason)
                self._reason_counts[reason] += 1
                if reason == "deadline_expired":
                    self._deadline_expiries += 1
                continue
            except BoundedWorkerTimeoutError as exc:
                self._last_error_detail = _error_detail(exc)
                self._error_counts["hard_timeout"] += 1
                self._reason_counts["deadline_expired"] += 1
                self._deadline_expiries += 1
                continue
            except BoundedWorkerWarmingError as exc:
                self._last_error_detail = _error_detail(exc)
                self._error_counts["worker_warming"] += 1
                self._reason_counts["teacher_unavailable"] += 1
                continue
            except BoundedWorkerUnavailableError as exc:
                self._last_error_detail = _error_detail(exc)
                self._error_counts["worker_unavailable"] += 1
                self._reason_counts["teacher_unavailable"] += 1
                continue
            except RemoteWorkerError as exc:
                self._last_error_detail = _error_detail(exc)
                self._error_counts[_remote_worker_error_label(exc)] += 1
                self._reason_counts["producer_error"] += 1
                continue
            except Exception as exc:
                self._last_error_detail = _error_detail(exc)
                self._error_counts[_error_label(exc)] += 1
                self._reason_counts["producer_error"] += 1
                continue

            self._worlds += result.worlds
            self._nodes += result.nodes_expanded
            self._coverage_sum += result.coverage
            if time.perf_counter() >= request_deadline or _result_hit_deadline(result):
                self._reason_counts["deadline_expired"] += 1
                self._deadline_expiries += 1
                continue
            reason_label = _result_reason_label(result.reason)
            self._reason_counts[reason_label] += 1
            value_error_label = _value_error_label(result.reason)
            if value_error_label is not None:
                self._value_error_counts[value_error_label] += 1
            target = self._target_from_result(result)
            if target is not None:
                results[candidate.request_index] = target

        self._record_step_elapsed(step_started_at)
        return tuple(results)

    def _record_step_elapsed(self, step_started_at: float) -> None:
        step_seconds = time.perf_counter() - step_started_at
        self._elapsed_seconds += step_seconds
        self._max_step_seconds = max(self._max_step_seconds, step_seconds)

    def _candidate_for(
        self,
        index: int,
        request: EngineTeacherRequest,
    ) -> _Candidate | None:
        context = _request_context(request)
        context_label = _context_label(context)
        probability = (
            self._config.main_probability
            if context_label == "main"
            else self._config.strategic_probability
        )
        decision_seed = _decision_seed(self._seed, request, context=context)
        if not _sample_request(_derive_seed(decision_seed, "gate"), probability):
            return None
        return _Candidate(
            request_index=index,
            request=request,
            decision_seed=decision_seed,
            priority=_derive_seed(decision_seed, "priority"),
        )

    def summary(self) -> Mapping[str, Any]:
        """Return bounded diagnostics; no value is a promotion gate."""
        attempted = self._attempted
        emitted = self._emitted
        worker_stats = self._worker.stats if self._worker is not None else None
        return {
            "engine_teacher_eligible": self._eligible,
            "engine_teacher_attempted": attempted,
            "engine_teacher_emitted": emitted,
            "engine_teacher_attempt_rate": (
                attempted / self._eligible if self._eligible else 0.0
            ),
            "engine_teacher_emit_rate": emitted / attempted if attempted else 0.0,
            "engine_teacher_behavior_matches_online": self._behavior_matches,
            "engine_teacher_worlds": self._worlds,
            "engine_teacher_nodes": self._nodes,
            "engine_teacher_coverage_sum": self._coverage_sum,
            "engine_teacher_confidence_sum_online": self._confidence_sum,
            "engine_teacher_mean_attempt_coverage": (
                self._coverage_sum / attempted if attempted else 0.0
            ),
            "engine_teacher_mean_emitted_confidence": (
                self._confidence_sum / emitted if emitted else 0.0
            ),
            "engine_teacher_seconds": self._elapsed_seconds,
            "engine_teacher_step_calls": self._step_calls,
            "engine_teacher_max_step_seconds": self._max_step_seconds,
            "engine_teacher_probability_skips": self._probability_skips,
            "engine_teacher_attempt_cap_skips": self._attempt_cap_skips,
            "engine_teacher_step_budget_skips": self._step_budget_skips,
            "engine_teacher_deadline_expiries": self._deadline_expiries,
            "engine_teacher_isolated_process": int(self._worker is not None),
            "engine_teacher_worker_starts": (
                0 if worker_stats is None else worker_stats.starts
            ),
            "engine_teacher_worker_restarts": (
                0 if worker_stats is None else worker_stats.restarts
            ),
            "engine_teacher_worker_startup_failures": (
                0 if worker_stats is None else worker_stats.startup_failures
            ),
            "engine_teacher_worker_hard_timeouts": (
                0 if worker_stats is None else worker_stats.hard_timeouts
            ),
            "engine_teacher_worker_crashes": (
                0 if worker_stats is None else worker_stats.crashes
            ),
            "engine_teacher_worker_forced_terminations": (
                0 if worker_stats is None else worker_stats.forced_terminations
            ),
            "engine_teacher_reason_counts": dict(sorted(self._reason_counts.items())),
            "engine_teacher_error_counts": dict(sorted(self._error_counts.items())),
            "engine_teacher_value_error_counts": dict(
                sorted(self._value_error_counts.items())
            ),
            "engine_teacher_last_error_detail": self._last_error_detail,
            "engine_teacher_context_counts": dict(sorted(self._context_counts.items())),
        }

    def close(self) -> None:
        """Release the isolated native worker, if configured."""
        if self._worker is not None:
            self._worker.close()

    def _target_from_result(
        self,
        result: TeacherSearchResult,
    ) -> EngineTeacherTarget | None:
        if not result.valid or result.confidence <= 0.0:
            return None
        self._emitted += 1
        self._behavior_matches += int(result.target_action == result.behavior_action)
        self._confidence_sum += result.confidence
        search_evidence = (
            result.search_evidence
            if isinstance(result, CompactTeacherResult)
            else search_evidence_from_complete_action_target(result)
        )
        return EngineTeacherTarget(
            action=result.target_action,
            confidence=result.confidence,
            weight=self._config.target_weight,
            search_evidence=search_evidence,
        )

    def _produce_target(
        self,
        request: EngineTeacherRequest,
        *,
        decision_seed: int,
        deadline: float,
    ) -> TeacherSearchResult:
        if self._worker is not None:
            planner_deadline = deadline - self._config.minimum_inference_budget_seconds
            if time.perf_counter() >= planner_deadline:
                raise TeacherDeadlineExpiredError("deadline_expired")
            try:
                result = self._worker.execute(
                    WorkerTask(
                        request=request,
                        decision_seed=decision_seed,
                        planner_deadline=planner_deadline,
                    ),
                    deadline=deadline,
                    broker_handler=self._handle_policy_call,
                    minimum_broker_seconds=(
                        self._config.minimum_inference_budget_seconds
                    ),
                )
            except RemoteWorkerError as exc:
                if exc.error_type == "TeacherDeadlineExpiredError":
                    raise TeacherDeadlineExpiredError("deadline_expired") from exc
                raise
            if not isinstance(
                result,
                (CompleteActionTeacherTarget, CompactTeacherResult),
            ):
                raise RuntimeError("isolated teacher returned an invalid result")
            return result
        return produce_target_impl(
            request,
            decision_seed=decision_seed,
            deadline=deadline,
            config=self._config,
            sampler=self._require_local_sampler(),
            belief_producer=self._belief_producer,
            policy=self._policy,
            device=self._device,
        )

    def _handle_policy_call(self, call: Any) -> DecodeReply | ValueReply:
        """Run inference only in the actor process that owns response queues."""
        deadline_monotonic = _broker_deadline_monotonic(
            call.deadline,
            minimum_seconds=self._config.minimum_inference_budget_seconds,
        )
        if isinstance(call, DecodeCall):
            sample_until = getattr(self._policy, "sample_decode_until", None)
            if callable(sample_until):
                actions, logprobs, values = sample_until(
                    call.states,
                    call.options,
                    call.decks,
                    temperature=call.temperature,
                    deadline_monotonic=deadline_monotonic,
                )
            else:
                actions, logprobs, values = self._policy.sample_decode(
                    call.states,
                    call.options,
                    call.decks,
                    temperature=call.temperature,
                )
            return DecodeReply(
                actions=tuple(
                    tuple(int(index) for index in action) for action in actions
                ),
                logprobs=logprobs.detach().cpu(),
                values=values.detach().cpu(),
                policy_version=policy_response_version(self._policy),
            )
        if isinstance(call, ValueCall):
            predict_until = getattr(self._policy, "predict_values_until", None)
            if callable(predict_until):
                values = predict_until(
                    call.states,
                    call.decks,
                    deadline_monotonic=deadline_monotonic,
                )
            else:
                values = self._policy.predict_values(call.states, call.decks)
            return ValueReply(
                values=values.detach().cpu(),
                policy_version=policy_response_version(self._policy),
            )
        raise TypeError("isolated teacher sent an unknown policy call")

    def _sample_worlds(
        self,
        *,
        evidence: Any,
        opponent_state: Any,
        own_deck: Sequence[int],
        seed: int,
        deadline: float,
    ) -> tuple[Determinization, ...]:
        return sample_worlds_impl(
            evidence=evidence,
            opponent_state=opponent_state,
            own_deck=own_deck,
            seed=seed,
            deadline=deadline,
            worlds=self._config.worlds,
            sampler=self._require_local_sampler(),
        )

    def _require_local_sampler(self) -> BeliefSampler:
        if self._sampler is None:
            raise RuntimeError("local engine teacher sampler is unavailable")
        return self._sampler


def _request_context(request: EngineTeacherRequest) -> int:
    return _int_field(_field(request.observation, "select"), "context", -1)


def _config_with_absolute_prior(
    config: OnlineEngineTeacherConfig,
) -> OnlineEngineTeacherConfig:
    """Freeze the prior path before concurrent opponents can change process cwd."""
    path = config.sampler.prior_deck_signature_summary_path
    if path is None or path.is_absolute():
        return config
    sampler = config.sampler.model_copy(
        update={"prior_deck_signature_summary_path": path.resolve(strict=True)}
    )
    return config.model_copy(update={"sampler": sampler})


def _context_label(context: int) -> str:
    return "main" if context == int(SelectContext.MAIN) else "strategic"


def _decision_seed(
    seed: int,
    request: EngineTeacherRequest,
    *,
    context: int,
) -> int:
    """Hash only public decision identity; behavior cannot affect eligibility."""
    current = _field(request.observation, "current")
    payload = "|".join(
        (
            "engine-teacher-decision-v1",
            str(seed),
            request.game_id,
            str(request.seat),
            str(_int_field(current, "turn", -1)),
            str(_int_field(current, "turnActionCount", -1)),
            str(context),
        )
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _world_seed(decision_seed: int, behavior_action: Sequence[int]) -> int:
    return _process_world_seed(decision_seed, behavior_action)


def _derive_seed(seed: int, domain: str) -> int:
    payload = f"{domain}|{seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _sample_request(seed: int, probability: float) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return seed < int(probability * float(1 << 64))


def _sampled_opponent_deck(sample: Determinization) -> tuple[int, ...]:
    return canonicalize_deck(sample.opponent_deck_counts.elements()).card_ids


_FIXED_RESULT_REASONS = frozenset(
    {
        "complete_exact",
        "complete_approximate",
        "paired_grid_incomplete",
        "continuation_incomplete",
        "leaf_scores_incomplete",
        "paired_scores_incomplete",
        "nonanticipative_strategy_unavailable",
        "uncontrolled_coin",
        "joint_strategy_cap",
        "value_deadline",
        "backup_deadline",
        "deadline_expired",
    }
)


def _result_reason_label(reason: str) -> str:
    if reason in _FIXED_RESULT_REASONS:
        return reason
    if reason.startswith("world_session_error:"):
        return "world_session_error"
    if reason.startswith("value_error:"):
        return "value_error"
    return "other_invalid_evidence"


def _value_error_label(reason: str) -> str | None:
    """Retain the bounded exception subtype hidden by validity aggregation."""
    prefix = "value_error:"
    if not reason.startswith(prefix):
        return None
    detail = reason[len(prefix) :]
    if "TimeoutError" in detail:
        return "timeout"
    error_type = detail.split(":", maxsplit=1)[0].strip()
    return error_type or "unknown"


def _broker_deadline_monotonic(
    deadline: float,
    *,
    minimum_seconds: float,
) -> float:
    remaining = float(deadline) - time.perf_counter()
    if remaining <= 2.0 * minimum_seconds:
        raise TimeoutError("insufficient teacher inference deadline")
    # Reserve one minimum-budget slice for broker IPC, scoring backup, and the
    # child's WorkResult after an inference response or explicit rejection.
    return time.monotonic() + remaining - minimum_seconds


def _unavailable_reason_label(reason: str) -> str:
    if reason == "deadline_expired":
        return reason
    return "teacher_unavailable"


def _error_label(exc: Exception) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout_error"
    if isinstance(exc, ValueError):
        return "value_error"
    if isinstance(exc, RuntimeError):
        return "runtime_error"
    return f"unexpected_error:{type(exc).__name__}"


def _error_detail(exc: Exception) -> str:
    message = str(exc).replace("\n", " ")[:512]
    return f"{type(exc).__name__}: {message}"


def _remote_worker_error_label(exc: RemoteWorkerError) -> str:
    if exc.error_type in {"ValueError", "TypeError"}:
        return "worker_input_error"
    if exc.error_type in {"RuntimeError", "_TeacherUnavailableError"}:
        return "worker_runtime_error"
    return "worker_unexpected_error"


def _result_hit_deadline(result: TeacherSearchResult) -> bool:
    if isinstance(result, CompactTeacherResult):
        return result.hit_deadline
    return any("deadline" in world_plan.plan.stop_reason for world_plan in result.plans)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    item = _field(value, name, default)
    return int(item) if item is not None else default


__all__ = ["OnlineEngineTeacherConfig", "OnlineEngineTeacherProducer"]
