"""Low-overhead dynamic effect probes using the bundled Search API."""

from __future__ import annotations

import ctypes
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import orjson

from ptcg_rl.agent.probe import (
    RuntimeProbeResult,
    build_runtime_probe_result,
    core_option_candidates,
)
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import GameContextFeatures, opponent_belief_state_from_evidence
from ptcg_rl.engine.forward_model import (
    dynamic_effect_feature_from_dict_resolution,
)
from ptcg_rl.engine.probe_resolution import resolve_probe_chain
from ptcg_rl.engine.runtime import _import_cg_module
from ptcg_rl.engine.session import HiddenInformation

ProbeBackendName = Literal["search_api"]


@dataclass(frozen=True)
class SearchApiProbeStats:
    """Counters from one N2 probe call."""

    backend: ProbeBackendName
    eligible_options: int
    probed_options: int
    worlds: int
    search_begin_calls: int
    search_step_calls: int
    unresolved_options: int
    unresolved_worlds: int


@dataclass(frozen=True)
class SearchApiProbeResult:
    """Probe feature rows plus backend diagnostics."""

    probe: RuntimeProbeResult | None
    stats: SearchApiProbeStats


class SearchApiProbeBackend:
    """N2 backend that skips cg.api dataclass reconstruction."""

    def __init__(self, *, manual_coin: bool = False) -> None:
        """Load the shipped libcg Search functions."""
        sim = _import_cg_module("cg.sim")
        self._lib = sim.lib
        self._engine_abi_fingerprint = _engine_abi_fingerprint(
            sim_path=Path(str(sim.__file__)),
            library_path=Path(str(self._lib._name)),
        )
        self._agent_ptr: int | None = None
        self._manual_coin = bool(manual_coin)

    @property
    def backend_name(self) -> ProbeBackendName:
        """Return the concrete backend name."""
        return "search_api"

    @property
    def engine_abi_fingerprint(self) -> str:
        """Return the content identity of the exact simulator and binding."""
        return self._engine_abi_fingerprint

    def run(
        self,
        observation: Mapping[str, Any],
        context_features: GameContextFeatures,
        *,
        your_deck: Sequence[int],
        sampler: BeliefSampler,
        worlds: int,
        rng: Any,
        opponent_card_probs: Sequence[float] | None = None,
        opponent_hand_weights: Sequence[float] | None = None,
    ) -> SearchApiProbeResult:
        """Probe all root ATTACK/ABILITY single-option candidates."""
        if worlds <= 0:
            raise ValueError("worlds must be positive")
        select = observation.get("select")
        core_candidates = core_option_candidates(select)
        option_count = len(_options(select))
        if not core_candidates:
            return SearchApiProbeResult(
                probe=None,
                stats=SearchApiProbeStats(
                    backend="search_api",
                    eligible_options=0,
                    probed_options=0,
                    worlds=worlds,
                    search_begin_calls=0,
                    search_step_calls=0,
                    unresolved_options=0,
                    unresolved_worlds=0,
                ),
            )
        evidence = extract_observation_evidence(observation)
        opponent_state = opponent_belief_state_from_evidence(evidence, context_features)
        vectors_by_action: dict[tuple[int, ...], list[tuple[float, ...]]] = {
            candidate: [] for candidate in core_candidates
        }
        unresolved_by_action = dict.fromkeys(core_candidates, 0)
        begin_calls = 0
        step_calls = 0
        for _ in range(worlds):
            determinization = sampler.sample_from_evidence(
                evidence,
                your_deck=your_deck,
                opponent_state=opponent_state,
                opponent_card_probs=opponent_card_probs,
                opponent_hand_weights=opponent_hand_weights,
                rng=rng,
            )
            root_state = self._begin(
                observation,
                determinization.hidden,
            )
            begin_calls += 1
            try:
                root_observation = _dict_field(root_state, "observation")
                for candidate in core_candidates:

                    def step(
                        state: Mapping[str, Any],
                        action: tuple[int, ...],
                    ) -> Mapping[str, Any]:
                        nonlocal step_calls
                        step_calls += 1
                        return self._step(_int_field(state, "searchId", 0), action)

                    chain = resolve_probe_chain(
                        root_state,
                        candidate,
                        step=step,
                        release=lambda state: self._release(
                            _int_field(state, "searchId", 0)
                        ),
                        observation=lambda state: _dict_field(state, "observation"),
                    )
                    try:
                        if chain.resolved:
                            feature_row = dynamic_effect_feature_from_dict_resolution(
                                select=tuple(int(index) for index in candidate),
                                before_observation=root_observation,
                                after_observation=_dict_field(
                                    chain.state,
                                    "observation",
                                ),
                                logs=chain.logs,
                                probe_transitions=chain.transitions,
                            )
                            vectors_by_action[candidate].append(feature_row.vector)
                        else:
                            unresolved_by_action[candidate] += 1
                    finally:
                        self._release(_int_field(chain.state, "searchId", 0))
            finally:
                self._end()

        probe = build_runtime_probe_result(
            option_count=option_count,
            vectors_by_action=vectors_by_action,
            worlds_requested=worlds,
            unresolved_by_action=unresolved_by_action,
        )
        return SearchApiProbeResult(
            probe=probe,
            stats=SearchApiProbeStats(
                backend="search_api",
                eligible_options=len(core_candidates),
                probed_options=sum(1 for mask in probe.masks if mask),
                worlds=worlds,
                search_begin_calls=begin_calls,
                search_step_calls=step_calls,
                unresolved_options=probe.unresolved_options,
                unresolved_worlds=probe.unresolved_worlds,
            ),
        )
    def _begin(
        self,
        observation: Mapping[str, Any],
        hidden: HiddenInformation,
    ) -> Mapping[str, Any]:
        search_input = observation.get("search_begin_input")
        if isinstance(search_input, bytes):
            search_bytes = search_input
        elif isinstance(search_input, str):
            search_bytes = search_input.encode("ascii")
        else:
            raise ValueError("rollout probe requires search_begin_input")
        raw = self._lib.SearchBegin(
            self._agent(),
            search_bytes,
            len(search_bytes),
            _int_array(hidden.your_deck),
            _int_array(hidden.your_prize),
            _int_array(hidden.opponent_deck),
            _int_array(hidden.opponent_prize),
            _int_array(hidden.opponent_hand),
            _int_array(hidden.opponent_active),
            int(self._manual_coin),
        )
        result = _loads_result(raw)
        error = _int_field(result, "error", 0)
        if error != 0:
            raise RuntimeError(f"SearchBegin failed with error={error}")
        state = _dict_field(result, "state")
        if not state:
            raise RuntimeError("SearchBegin returned no state")
        return state

    def _step(
        self,
        search_id: int,
        select: Sequence[int],
    ) -> Mapping[str, Any]:
        raw = self._lib.SearchStep(
            self._agent(),
            int(search_id),
            _int_array(select),
            len(select),
        )
        result = _loads_result(raw)
        error = _int_field(result, "error", 0)
        if error != 0:
            raise RuntimeError(f"SearchStep failed with error={error}")
        state = _dict_field(result, "state")
        if not state:
            raise RuntimeError("SearchStep returned no state")
        return state

    def _release(self, search_id: int) -> None:
        if search_id > 0:
            self._lib.SearchRelease(self._agent(), int(search_id))

    def _end(self) -> None:
        self._lib.SearchEnd(self._agent())

    def _agent(self) -> int:
        if self._agent_ptr is None:
            self._agent_ptr = int(self._lib.AgentStart())
        return self._agent_ptr


def _engine_abi_fingerprint(*, sim_path: Path, library_path: Path) -> str:
    """Hash engine bytes without binding identity to host-local paths."""
    digest = hashlib.sha256(b"ptcg-rl/search-api-engine-abi/v1\x00")
    for label, path in (("sim", sim_path), ("libcg", library_path)):
        if not path.is_file():
            raise FileNotFoundError(f"Search API engine asset is missing: {label}")
        digest.update(label.encode("ascii"))
        digest.update(b"\x00")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _loads_result(raw: bytes | str) -> Mapping[str, Any]:
    payload = raw.encode("utf-8") if isinstance(raw, str) else raw
    result = orjson.loads(payload)
    if not isinstance(result, Mapping):
        raise RuntimeError("Search API returned a non-object payload")
    return result


def _int_array(values: Sequence[int]) -> Any:
    return (ctypes.c_int * len(values))(*[int(value) for value in values])


def _dict_field(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    field = value.get(name)
    if not isinstance(field, Mapping):
        raise RuntimeError(f"Search API payload missing object field: {name}")
    return field


def _int_field(value: Mapping[str, Any], name: str, default: int) -> int:
    raw = value.get(name, default)
    return int(raw) if raw is not None else default


def _options(select: Any) -> Sequence[Any]:
    if isinstance(select, Mapping):
        value = select.get("option", ())
    else:
        value = getattr(select, "option", ())
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
