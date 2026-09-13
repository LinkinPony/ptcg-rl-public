"""Single-game native lane adapter for the persistent evaluation league."""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any, Literal

import numpy as np

from ptcg_rl.actions.selection import is_legal_action
from ptcg_rl.agent.search.budget import ActTimeLedger, ActTimeTimeoutError
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.native_training import (
    NativeTrainingLane,
    NativeTrainingOutputBuffer,
)
from ptcg_rl.evaluation.continuous_league.models import (
    ContinuousLeagueConfig,
    GameOutcome,
    MatchLease,
    NativeMatchConfig,
    TerminalReason,
)
from ptcg_rl.evaluation.continuous_league.native_protocol import (
    NativeMatchProtocolError,
    read_frame,
    write_frame,
)
from ptcg_rl.evaluation.continuous_league.native_runtime import (
    ControllerFactory,
    MatchAgent,
)
from ptcg_rl.rl.native_collection_control import (
    require_native_ready,
    require_native_step_output,
)

_SLOT = np.asarray((0,), dtype=np.uint32)
ActionParityAuditor = Callable[[int, Mapping[str, Any], tuple[int, ...]], None]
ParticipantFaultKind = Literal["timeout", "act_error", "illegal_action"]


@dataclass(frozen=True)
class NativeMatchResponse:
    """Executor response completed by the outer worker with match ownership."""

    outcome: GameOutcome
    terminal_reason: TerminalReason
    started_at: str
    finished_at: str
    steps: int
    duration_seconds: float
    telemetry: Mapping[str, Any]

    def payload(self) -> dict[str, Any]:
        """Return the msgpack-safe wire representation."""
        return {
            "outcome": self.outcome,
            "terminal_reason": self.terminal_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": self.steps,
            "duration_seconds": self.duration_seconds,
            "telemetry": dict(self.telemetry),
        }


class NativeMatchExecutor:
    """Persist one native lane and a bounded model cache across match leases."""

    def __init__(
        self,
        config: NativeMatchConfig,
        *,
        repo_root: Path,
        controllers: ControllerFactory | None = None,
    ) -> None:
        self.config = config
        self.repo_root = repo_root.resolve()
        self.library_path = _resolve_file(config.library_path, self.repo_root)
        self.library_sha256 = _file_sha256(self.library_path)
        expected_sha = config.expected_library_sha256
        if expected_sha is None:
            raise ValueError("native match library SHA-256 is not frozen")
        if self.library_sha256 != expected_sha:
            raise ValueError("native match library differs from frozen SHA-256")
        self.lane = NativeTrainingLane(
            1,
            library_path=self.library_path,
            worker_count=config.lane_worker_count,
        )
        self.outputs = (
            NativeTrainingOutputBuffer(
                slot_capacity=1,
                option_capacity=config.option_capacity,
            ),
            NativeTrainingOutputBuffer(
                slot_capacity=1,
                option_capacity=config.option_capacity,
            ),
        )
        self._owns_controllers = controllers is None
        self.controllers = controllers or ControllerFactory(
            config,
            repo_root=self.repo_root,
        )
        self._closed = False

    def execute(
        self,
        lease: MatchLease,
        *,
        action_auditor: ActionParityAuditor | None = None,
    ) -> NativeMatchResponse:
        """Run one exact game and classify only participant-attributed faults."""
        started = datetime.now(UTC)
        started_clock = time.perf_counter()
        agents: list[MatchAgent] = []
        try:
            self._validate_lease(lease)
            decks = self._load_decks(lease)
            engine_seed, agent_seeds = match_seeds(lease.match_id)
            for seat in (0, 1):
                side = lease.side_a if seat == 0 else lease.side_b
                kind = (
                    lease.side_a_controller_kind
                    if seat == 0
                    else lease.side_b_controller_kind
                )
                controller_path = (
                    lease.side_a_controller_path
                    if seat == 0
                    else lease.side_b_controller_path
                )
                deck_path = (
                    lease.side_a_deck_path if seat == 0 else lease.side_b_deck_path
                )
                agents.append(
                    self.controllers.build(
                        bundle=side,
                        kind=kind,
                        controller_path=controller_path,
                        deck_path=deck_path,
                        deck=decks[seat],
                        seat=seat,
                        seed=agent_seeds[seat],
                    )
                )
            return self._run_game(
                lease,
                decks=decks,
                agents=(agents[0], agents[1]),
                engine_seed=engine_seed,
                agent_seeds=agent_seeds,
                started=started,
                started_clock=started_clock,
                action_auditor=action_auditor,
            )
        except _ParticipantFaultError as fault:
            return self._fault_response(
                fault,
                started=started,
                started_clock=started_clock,
            )
        except Exception as error:
            return self._infrastructure_response(
                error,
                started=started,
                started_clock=started_clock,
            )
        finally:
            for agent in agents:
                with suppress(Exception):
                    agent.close()

    def close(self) -> None:
        """Release the lane and every cache-owned model exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._owns_controllers:
            self.controllers.close()
        self.lane.close()

    def _run_game(
        self,
        lease: MatchLease,
        *,
        decks: tuple[tuple[int, ...], tuple[int, ...]],
        agents: tuple[MatchAgent, MatchAgent],
        engine_seed: int,
        agent_seeds: tuple[int, int],
        started: datetime,
        started_clock: float,
        action_auditor: ActionParityAuditor | None,
    ) -> NativeMatchResponse:
        ledger = ActTimeLedger(self.config.act_time_ledger)
        deck_rows = np.asarray([[decks[0], decks[1]]], dtype=np.int32)
        view = self.lane.reset(
            deck_rows,
            np.asarray((engine_seed,), dtype=np.uint32),
            slots=_SLOT,
            output=self.outputs[0],
        )
        require_native_ready(view)
        engine_steps = int(view.selection_advance_count[0])
        first_player: int | None = None
        decision_counts = [0, 0]
        action_seconds = [0.0, 0.0]
        action_trace: list[dict[str, Any]] = []
        output_index = 1
        while True:
            seat = int(view.select_player[0])
            if seat not in (0, 1):
                raise RuntimeError("native lane exposed an invalid select player")
            encoded_observation = self.lane.export_public_observations(_SLOT)[0]
            state_token = self.lane.export_public_state_tokens(_SLOT)[0]
            observation = json.loads(encoded_observation)
            if not isinstance(observation, dict):
                raise TypeError("native public observation must be a mapping")
            observation["search_begin_input"] = state_token.decode("ascii")
            try:
                action, elapsed, charged_observation = _act(
                    agents[seat],
                    observation,
                    ledger=ledger,
                    seat=seat,
                )
            except _ParticipantFaultError as fault:
                if fault.callback_attempted:
                    decision_counts[seat] += 1
                    action_seconds[seat] += fault.elapsed_seconds
                fault.steps = engine_steps
                fault.telemetry = self._telemetry(
                    lease,
                    engine_seed=engine_seed,
                    agent_seeds=agent_seeds,
                    first_player=first_player,
                    decision_counts=decision_counts,
                    action_seconds=action_seconds,
                    ledger=ledger,
                    action_trace=action_trace,
                )
                raise
            if action_auditor is not None:
                action_auditor(seat, charged_observation, action)
            decision_counts[seat] += 1
            action_seconds[seat] += elapsed
            action_trace.append(
                {
                    "decision": len(action_trace),
                    "seat": seat,
                    "turn": int(view.turn[0]),
                    "public_state_sha256": _public_state_sha256(
                        encoded_observation,
                        state_token,
                    ),
                    "action": list(action),
                    "act_seconds": elapsed,
                    "runtime": dict(agents[seat].telemetry()),
                }
            )
            next_view = self.lane.step(
                _SLOT,
                np.asarray((0, len(action)), dtype=np.uint32),
                np.asarray(action, dtype=np.int32),
                output=self.outputs[output_index],
            )
            output_index = 1 - output_index
            require_native_step_output(next_view)
            observed_first_player = int(next_view.first_player[0])
            if observed_first_player in (0, 1):
                first_player = observed_first_player
            advances = int(next_view.selection_advance_count[0])
            if advances <= 0:
                raise RuntimeError("native lane did not count a submitted selection")
            engine_steps += advances
            if int(next_view.status[0]) == 2:
                outcome = _native_outcome(int(next_view.result[0]))
                return _finished_response(
                    outcome=outcome,
                    terminal_reason="normal",
                    started=started,
                    started_clock=started_clock,
                    steps=engine_steps,
                    telemetry=self._telemetry(
                        lease,
                        engine_seed=engine_seed,
                        agent_seeds=agent_seeds,
                        first_player=first_player,
                        decision_counts=decision_counts,
                        action_seconds=action_seconds,
                        ledger=ledger,
                        action_trace=action_trace,
                    ),
                )
            if engine_steps >= self.config.maximum_engine_steps:
                return _finished_response(
                    outcome="unresolved",
                    terminal_reason="max_steps",
                    started=started,
                    started_clock=started_clock,
                    steps=engine_steps,
                    telemetry=self._telemetry(
                        lease,
                        engine_seed=engine_seed,
                        agent_seeds=agent_seeds,
                        first_player=first_player,
                        decision_counts=decision_counts,
                        action_seconds=action_seconds,
                        ledger=ledger,
                        action_trace=action_trace,
                    ),
                )
            view = next_view

    def _load_decks(
        self,
        lease: MatchLease,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        loaded: list[tuple[int, ...]] = []
        for side, raw_path in (
            (lease.side_a, lease.side_a_deck_path),
            (lease.side_b, lease.side_b_deck_path),
        ):
            path = _resolve_file(raw_path, self.repo_root)
            deck = canonicalize_deck(records.read_deck(path))
            if deck.deck_digest != side.deck_digest:
                raise ValueError("deck bytes differ from bundle identity")
            loaded.append(deck.card_ids)
        return loaded[0], loaded[1]

    def _validate_lease(self, lease: MatchLease) -> None:
        if self._closed:
            raise RuntimeError("native match executor is closed")
        if lease.side_a.bundle_id == lease.side_b.bundle_id:
            raise ValueError("native match sides must be distinct bundles")
        for kind, side in (
            (lease.side_a_controller_kind, lease.side_a),
            (lease.side_b_controller_kind, lease.side_b),
        ):
            if not side.controller_id.startswith(f"{kind}:"):
                raise ValueError("controller kind and identity prefix differ")
            if kind == "checkpoint" and not lease.requires_cuda:
                raise ValueError("checkpoint match must require CUDA")

    def _telemetry(
        self,
        lease: MatchLease,
        *,
        engine_seed: int,
        agent_seeds: tuple[int, int],
        first_player: int | None,
        decision_counts: Sequence[int],
        action_seconds: Sequence[float],
        ledger: ActTimeLedger,
        action_trace: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "protocol": self.config.protocol,
            "native_library_sha256": self.library_sha256,
            "runtime_fingerprint": lease.runtime_fingerprint,
            "belief_fingerprint": lease.belief_fingerprint,
            "engine_seed": engine_seed,
            "agent_seeds": list(agent_seeds),
            "first_player": first_player,
            "decision_counts": list(decision_counts),
            "action_seconds": list(action_seconds),
            "act_time_used_seconds": [ledger.used(0), ledger.used(1)],
            "actions": list(action_trace),
        }

    def _fault_response(
        self,
        fault: _ParticipantFaultError,
        *,
        started: datetime,
        started_clock: float,
    ) -> NativeMatchResponse:
        reasons: dict[tuple[int, ParticipantFaultKind], TerminalReason] = {
            (0, "timeout"): "side_a_timeout",
            (0, "act_error"): "side_a_act_error",
            (0, "illegal_action"): "side_a_illegal_action",
            (1, "timeout"): "side_b_timeout",
            (1, "act_error"): "side_b_act_error",
            (1, "illegal_action"): "side_b_illegal_action",
        }
        reason = reasons[(fault.seat, fault.kind)]
        return _finished_response(
            outcome="side_b_win" if fault.seat == 0 else "side_a_win",
            terminal_reason=reason,
            started=started,
            started_clock=started_clock,
            steps=fault.steps,
            telemetry={
                **fault.telemetry,
                "protocol": self.config.protocol,
                "native_library_sha256": self.library_sha256,
                "fault_seat": fault.seat,
                "fault_kind": fault.kind,
                "error": fault.detail,
            },
        )

    def _infrastructure_response(
        self,
        error: Exception,
        *,
        started: datetime,
        started_clock: float,
    ) -> NativeMatchResponse:
        return _finished_response(
            outcome="unresolved",
            terminal_reason="infrastructure_error",
            started=started,
            started_clock=started_clock,
            steps=0,
            telemetry={
                "protocol": self.config.protocol,
                "native_library_sha256": self.library_sha256,
                "error": f"{type(error).__name__}: {error}",
            },
        )


class _ParticipantFaultError(RuntimeError):
    def __init__(
        self,
        *,
        seat: int,
        kind: ParticipantFaultKind,
        detail: str,
        steps: int = 0,
        callback_attempted: bool = False,
        elapsed_seconds: float = 0.0,
    ) -> None:
        super().__init__(detail)
        self.seat = seat
        self.kind = kind
        self.detail = detail
        self.steps = steps
        self.callback_attempted = callback_attempted
        self.elapsed_seconds = elapsed_seconds
        self.telemetry: Mapping[str, Any] = {}


def _act(
    agent: MatchAgent,
    observation: dict[str, Any],
    *,
    ledger: ActTimeLedger,
    seat: int,
) -> tuple[tuple[int, ...], float, dict[str, Any]]:
    try:
        charged_observation = ledger.observation_for(observation, seat)
    except ActTimeTimeoutError as error:
        raise _ParticipantFaultError(
            seat=seat,
            kind="timeout",
            detail=str(error),
        ) from error
    started = time.perf_counter()
    action_error: Exception | None = None
    raw_action: Sequence[int] = ()
    try:
        raw_action = agent.act(charged_observation)
    except Exception as error:
        action_error = error
    elapsed = max(0.0, time.perf_counter() - started)
    try:
        ledger.charge(seat, elapsed)
    except ActTimeTimeoutError as error:
        raise _ParticipantFaultError(
            seat=seat,
            kind="timeout",
            detail=str(error),
            callback_attempted=True,
            elapsed_seconds=elapsed,
        ) from error
    if action_error is not None:
        raise _ParticipantFaultError(
            seat=seat,
            kind="act_error",
            detail=f"{type(action_error).__name__}: {action_error}",
            callback_attempted=True,
            elapsed_seconds=elapsed,
        ) from action_error
    try:
        action = tuple(int(index) for index in raw_action)
    except (TypeError, ValueError) as error:
        raise _ParticipantFaultError(
            seat=seat,
            kind="act_error",
            detail=f"invalid action payload: {error}",
            callback_attempted=True,
            elapsed_seconds=elapsed,
        ) from error
    select = charged_observation.get("select")
    if not is_legal_action(select, action):
        raise _ParticipantFaultError(
            seat=seat,
            kind="illegal_action",
            detail=f"illegal action indices: {action}",
            callback_attempted=True,
            elapsed_seconds=elapsed,
        )
    return action, elapsed, charged_observation


def run_native_match_server(
    config: ContinuousLeagueConfig,
    *,
    repo_root: Path,
    concurrency: int = 1,
) -> None:
    """Serve framed leases while sharing one model cache across native lanes."""
    if concurrency <= 0:
        raise ValueError("native match concurrency must be positive")
    runtime_fingerprint, belief_fingerprint = native_match_contract_fingerprints(
        config.native_match
    )
    if runtime_fingerprint != config.runtime_fingerprint:
        raise ValueError("native match runtime fingerprint differs from its profile")
    if belief_fingerprint != config.belief_fingerprint:
        raise ValueError("native match belief fingerprint differs from its profile")
    if concurrency == 1:
        _run_serial_native_match_server(config, repo_root=repo_root)
        return
    _run_concurrent_native_match_server(
        config,
        repo_root=repo_root,
        concurrency=concurrency,
    )


def _run_serial_native_match_server(
    config: ContinuousLeagueConfig,
    *,
    repo_root: Path,
) -> None:
    executor = NativeMatchExecutor(config.native_match, repo_root=repo_root)
    try:
        while True:
            payload = read_frame(sys.stdin.buffer)
            if payload is None:
                return
            lease = MatchLease.model_validate(payload)
            if (
                lease.runtime_fingerprint != config.runtime_fingerprint
                or lease.belief_fingerprint != config.belief_fingerprint
            ):
                response = executor._infrastructure_response(
                    ValueError("lease protocol fingerprints differ from executor"),
                    started=datetime.now(UTC),
                    started_clock=time.perf_counter(),
                )
            else:
                response = executor.execute(lease)
            write_frame(sys.stdout.buffer, response.payload())
    finally:
        executor.close()


def _run_concurrent_native_match_server(
    config: ContinuousLeagueConfig,
    *,
    repo_root: Path,
    concurrency: int,
) -> None:
    """Multiplex independent leases over lanes with one checkpoint cache."""
    controllers = ControllerFactory(config.native_match, repo_root=repo_root)
    executors = tuple(
        NativeMatchExecutor(
            config.native_match,
            repo_root=repo_root,
            controllers=controllers,
        )
        for _ in range(concurrency)
    )
    available: Queue[NativeMatchExecutor] = Queue()
    for executor in executors:
        available.put(executor)
    output_lock = threading.Lock()

    def serve(payload: Mapping[str, Any]) -> None:
        request_id = str(payload.get("request_id", ""))
        if not request_id:
            raise NativeMatchProtocolError(
                "concurrent native match request has no request_id"
            )
        raw_lease = payload.get("lease")
        if not isinstance(raw_lease, Mapping):
            raise NativeMatchProtocolError(
                "concurrent native match request has no lease mapping"
            )
        lease = MatchLease.model_validate(raw_lease)
        executor = available.get()
        try:
            if (
                lease.runtime_fingerprint != config.runtime_fingerprint
                or lease.belief_fingerprint != config.belief_fingerprint
            ):
                response = executor._infrastructure_response(
                    ValueError("lease protocol fingerprints differ from executor"),
                    started=datetime.now(UTC),
                    started_clock=time.perf_counter(),
                )
            else:
                response = executor.execute(lease)
        finally:
            available.put(executor)
        with output_lock:
            write_frame(
                sys.stdout.buffer,
                {"request_id": request_id, "response": response.payload()},
            )

    try:
        with ThreadPoolExecutor(
            max_workers=concurrency,
            thread_name_prefix="league-native-lane",
        ) as pool:
            while True:
                payload = read_frame(sys.stdin.buffer)
                if payload is None:
                    break
                pool.submit(serve, payload)
    finally:
        for executor in executors:
            executor.close()
        controllers.close()


def _finished_response(
    *,
    outcome: GameOutcome,
    terminal_reason: TerminalReason,
    started: datetime,
    started_clock: float,
    steps: int,
    telemetry: Mapping[str, Any],
) -> NativeMatchResponse:
    finished = datetime.now(UTC)
    return NativeMatchResponse(
        outcome=outcome,
        terminal_reason=terminal_reason,
        started_at=started.isoformat(),
        finished_at=finished.isoformat(),
        steps=max(0, steps),
        duration_seconds=max(0.0, time.perf_counter() - started_clock),
        telemetry=telemetry,
    )


def _native_outcome(result: int) -> GameOutcome:
    outcomes: dict[int, GameOutcome] = {
        0: "side_a_win",
        1: "side_b_win",
        2: "draw",
    }
    try:
        return outcomes[result]
    except KeyError as error:
        raise RuntimeError(
            f"native lane returned invalid terminal result: {result}"
        ) from error


def match_seeds(match_id: str) -> tuple[int, tuple[int, int]]:
    digest = hashlib.sha256(
        b"ptcg-rl/continuous-league/native-match-seeds/v1\0" + match_id.encode("utf-8")
    ).digest()
    values = tuple(
        int.from_bytes(digest[offset : offset + 4], "big") for offset in (0, 4, 8)
    )
    return values[0], (values[1], values[2])


def native_match_contract_fingerprints(
    config: NativeMatchConfig,
) -> tuple[str, str]:
    """Derive separate runtime and belief identities from resolved semantics."""
    payload = config.model_dump(mode="json")
    act_time = dict(payload.pop("act_time"))
    for operational_field in (
        "library_path",
        "public_catalog_manifest_path",
        "lane_worker_count",
        "option_capacity",
        "checkpoint_cache_entries",
        "policy_batch_max_rows",
        "policy_batch_wait_ms",
    ):
        payload.pop(operational_field, None)
    for per_bundle_field in ("deck_path", "checkpoint_path", "seed"):
        act_time.pop(per_bundle_field, None)
    belief = act_time.pop("belief")
    search = dict(act_time["search"])
    sampler = search.pop("sampler")
    act_time["search"] = search
    common = {
        "protocol": config.protocol,
        "public_catalog_manifest_sha256": (
            config.expected_public_catalog_manifest_sha256
        ),
    }
    runtime_payload = {
        **common,
        "native_match": payload,
        "act_time": act_time,
    }
    belief_payload = {
        **common,
        "belief_producer": belief,
        "belief_sampler": sampler,
    }
    return (
        _contract_fingerprint(
            b"ptcg-rl/continuous-league/runtime-contract/v1\0",
            runtime_payload,
        ),
        _contract_fingerprint(
            b"ptcg-rl/continuous-league/belief-contract/v1\0",
            belief_payload,
        ),
    )


def _public_state_sha256(observation: bytes, token: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(observation).to_bytes(8, "big"))
    digest.update(observation)
    digest.update(token)
    return digest.hexdigest()


def _contract_fingerprint(domain: bytes, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


def _resolve_file(path: Path, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ActionParityAuditor",
    "NativeMatchExecutor",
    "NativeMatchResponse",
    "match_seeds",
    "native_match_contract_fingerprints",
    "run_native_match_server",
]
