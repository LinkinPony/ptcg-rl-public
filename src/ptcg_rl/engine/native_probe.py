"""Native dynamic effect probe backend for rollout training."""

from __future__ import annotations

import ctypes
import hashlib
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ptcg_rl.agent.probe import core_option_candidates
from ptcg_rl.belief.observation import extract_observation_evidence
from ptcg_rl.belief.sampling import BeliefSampler
from ptcg_rl.context import GameContextFeatures, opponent_belief_state_from_evidence
from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE

NativeProbeBackendName = Literal["native"]
_HIDDEN_LIST_COUNT = 6
_NATIVE_FACT_VERSION = 1
_NATIVE_FACT_WIDTH = DYNAMIC_EFFECT_FEATURE_SIZE


class NativeProbeLoadError(RuntimeError):
    """Raised when the native probe shared library cannot be loaded."""


@dataclass(frozen=True)
class NativeProbeStats:
    """Counters from one N1 probe call."""

    backend: NativeProbeBackendName
    eligible_options: int
    probed_options: int
    worlds: int
    native_batch_calls: int
    native_transitions: int
    native_errors: int
    unresolved_options: int
    unresolved_worlds: int


@dataclass(frozen=True)
class NativeExactFactResult:
    """Exact option-aligned facts plus native backend diagnostics."""

    features: tuple[tuple[float, ...], ...]
    masks: tuple[bool, ...]
    stats: NativeProbeStats


@dataclass(frozen=True)
class NativeFactBatchResult:
    """Dense candidate facts returned by one ragged multi-root native call."""

    features: npt.NDArray[np.float32]
    masks: npt.NDArray[np.bool_]
    root_errors: npt.NDArray[np.int32]
    root_unresolved_worlds: npt.NDArray[np.int32]


class NativeProbeBackend:
    """Ragged multi-root fact backend for ``libcg_probe.so``."""

    def __init__(
        self,
        *,
        manual_coin: bool = False,
        library_path: Path | str | None = None,
    ) -> None:
        """Load the native probe shared library."""
        self._manual_coin = bool(manual_coin)
        self._lib = _load_library(library_path)
        self._lib.CgProbeInitialize.argtypes = []
        self._lib.CgProbeInitialize.restype = None
        self._lib.CgProbeLastError.argtypes = []
        self._lib.CgProbeLastError.restype = ctypes.c_char_p
        self._fact_abi_bound = False
        self._lib.CgProbeInitialize()
        self._bind_fact_abi()
        library_file = Path(str(self._lib._name)).resolve()
        self._engine_abi_fingerprint = hashlib.sha256(
            library_file.read_bytes()
        ).hexdigest()

    @property
    def backend_name(self) -> NativeProbeBackendName:
        """Return the concrete backend name."""
        return "native"

    @property
    def engine_abi_fingerprint(self) -> str:
        """Return the content identity of the native engine bridge."""
        return self._engine_abi_fingerprint

    def run_exact_facts(
        self,
        observation: Mapping[str, Any],
        context_features: GameContextFeatures,
        *,
        your_deck: Sequence[int],
        sampler: BeliefSampler,
        worlds: int,
        rng: Any,
        require_byte_identical_worlds: bool,
        opponent_card_probs: Sequence[float] | None = None,
        opponent_hand_weights: Sequence[float] | None = None,
    ) -> NativeExactFactResult | None:
        """Return option-aligned exact facts without a Python engine path."""
        if worlds <= 0:
            raise ValueError("worlds must be positive")
        select = observation.get("select")
        core_candidates = core_option_candidates(select)
        option_count = len(_options(select))
        if not core_candidates:
            return None

        hidden_counts, hidden_values = _sample_hidden_inputs(
            observation,
            context_features,
            your_deck=your_deck,
            sampler=sampler,
            worlds=worlds,
            rng=rng,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
        )
        candidate_values = tuple(int(candidate[0]) for candidate in core_candidates)
        result = self.run_fact_batch(
            (_state_token(observation),),
            hidden_counts=hidden_counts,
            hidden_values=hidden_values,
            hidden_value_offsets=(0, len(hidden_values)),
            candidate_offsets=(0, len(candidate_values)),
            candidate_values=candidate_values,
            worlds=worlds,
            require_byte_identical_worlds=require_byte_identical_worlds,
        )
        root_error = int(result.root_errors[0])
        if root_error:
            raise RuntimeError(
                f"native engine fact transition failed with error={root_error}"
            )
        features = np.zeros(
            (option_count, DYNAMIC_EFFECT_FEATURE_SIZE),
            dtype=np.float32,
        )
        masks = np.zeros(option_count, dtype=np.bool_)
        indices = np.asarray(candidate_values, dtype=np.int64)
        features[indices] = result.features
        masks[indices] = result.masks
        return NativeExactFactResult(
            features=tuple(tuple(float(value) for value in row) for row in features),
            masks=tuple(bool(value) for value in masks),
            stats=NativeProbeStats(
                backend="native",
                eligible_options=len(core_candidates),
                probed_options=int(result.masks.sum()),
                worlds=worlds,
                native_batch_calls=1,
                native_transitions=len(core_candidates) * worlds,
                native_errors=0,
                unresolved_options=int((~result.masks).sum()),
                unresolved_worlds=int(result.root_unresolved_worlds[0]),
            ),
        )

    def run_fact_batch(
        self,
        state_tokens: Sequence[bytes],
        *,
        hidden_counts: Sequence[int],
        hidden_values: Sequence[int],
        hidden_value_offsets: Sequence[int],
        candidate_offsets: Sequence[int],
        candidate_values: Sequence[int],
        worlds: int,
        require_byte_identical_worlds: bool,
    ) -> NativeFactBatchResult:
        """Run a ragged collection of singleton candidates in native code."""
        root_count = len(state_tokens)
        if root_count <= 0 or worlds <= 0:
            raise ValueError("native fact dimensions must be positive")
        if any(not token for token in state_tokens):
            raise ValueError("native fact state tokens must be non-empty")
        if len(hidden_counts) != root_count * worlds * _HIDDEN_LIST_COUNT:
            raise ValueError("native fact hidden-list counts are misaligned")
        _validate_offsets(
            hidden_value_offsets,
            expected_parts=root_count,
            expected_values=len(hidden_values),
            name="hidden-value",
            allow_empty_parts=True,
        )
        _validate_offsets(
            candidate_offsets,
            expected_parts=root_count,
            expected_values=len(candidate_values),
            name="candidate",
        )
        for root_index in range(root_count):
            counts_start = root_index * worlds * _HIDDEN_LIST_COUNT
            counts_stop = counts_start + worlds * _HIDDEN_LIST_COUNT
            expected_values = sum(
                int(value) for value in hidden_counts[counts_start:counts_stop]
            )
            actual_values = (
                int(hidden_value_offsets[root_index + 1])
                - int(hidden_value_offsets[root_index])
            )
            if expected_values != actual_values:
                raise ValueError("native fact hidden values are misaligned")

        self._bind_fact_abi()
        state_payload = b"".join(state_tokens)
        state_offsets = [0]
        for token in state_tokens:
            state_offsets.append(state_offsets[-1] + len(token))
        hidden_count_array = _int_array(hidden_counts)
        hidden_value_array = _int_array(hidden_values)
        hidden_offset_array = _int_array(hidden_value_offsets)
        candidate_offset_array = _int_array(candidate_offsets)
        candidate_value_array = _int_array(candidate_values)
        state_offset_array = _int_array(state_offsets)
        candidate_count = len(candidate_values)
        feature_array = (
            ctypes.c_float * (candidate_count * _NATIVE_FACT_WIDTH)
        )()
        mask_array = (ctypes.c_ubyte * candidate_count)()
        error_array = (ctypes.c_int * root_count)()
        unresolved_array = (ctypes.c_int * root_count)()
        error = int(
            self._lib.CgProbeFactBatch(
                state_payload,
                state_offset_array,
                root_count,
                hidden_count_array,
                hidden_value_array,
                hidden_offset_array,
                worlds,
                candidate_offset_array,
                candidate_value_array,
                int(self._manual_coin),
                int(require_byte_identical_worlds),
                feature_array,
                mask_array,
                error_array,
                unresolved_array,
            )
        )
        if error != 0:
            raise RuntimeError(
                f"native fact batch failed with error={error}: {self._last_error()}"
            )
        return NativeFactBatchResult(
            features=np.ctypeslib.as_array(feature_array)
            .reshape(candidate_count, _NATIVE_FACT_WIDTH)
            .copy(),
            masks=np.ctypeslib.as_array(mask_array).astype(np.bool_, copy=True),
            root_errors=np.ctypeslib.as_array(error_array).astype(
                np.int32,
                copy=True,
            ),
            root_unresolved_worlds=np.ctypeslib.as_array(
                unresolved_array
            ).astype(np.int32, copy=True),
        )

    def _bind_fact_abi(self) -> None:
        if self._fact_abi_bound:
            return
        try:
            fact_version = self._lib.CgProbeFactVersion
            fact_width = self._lib.CgProbeFactWidth
            fact_batch = self._lib.CgProbeFactBatch
        except AttributeError as exc:
            raise NativeProbeLoadError(
                "native probe library has no fact-batch ABI; rebuild it"
            ) from exc
        fact_version.argtypes = []
        fact_version.restype = ctypes.c_int
        fact_width.argtypes = []
        fact_width.restype = ctypes.c_int
        if int(fact_version()) != _NATIVE_FACT_VERSION:
            raise NativeProbeLoadError("native fact ABI version does not match Python")
        if int(fact_width()) != _NATIVE_FACT_WIDTH:
            raise NativeProbeLoadError("native fact feature width does not match Python")
        fact_batch.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        fact_batch.restype = ctypes.c_int
        self._fact_abi_bound = True

    def _last_error(self) -> str:
        raw = self._lib.CgProbeLastError()
        if not isinstance(raw, bytes):
            return ""
        return raw.decode("utf-8", errors="replace")


def _load_library(library_path: Path | str | None) -> Any:
    candidates = _library_candidates(library_path)
    attempted: list[str] = []
    for candidate in candidates:
        attempted.append(str(candidate))
        if not candidate.exists():
            continue
        try:
            return ctypes.CDLL(str(candidate))
        except OSError as error:
            attempted[-1] = f"{candidate}: {error}"
    raise NativeProbeLoadError(
        "libcg_probe.so not found or not loadable; tried " + ", ".join(attempted)
    )


def _library_candidates(library_path: Path | str | None) -> tuple[Path, ...]:
    if library_path is not None:
        return (Path(library_path),)
    paths: list[Path] = []
    env_path = os.environ.get("PTCG_RL_CG_PROBE_LIB")
    if env_path:
        paths.append(Path(env_path))
    repo_root = Path(__file__).resolve().parents[3]
    package_root = Path(__file__).resolve().parents[2]
    paths.extend(
        [
            repo_root / "src" / "native" / "cg_probe" / "libcg_probe.so",
            package_root / "src" / "native" / "cg_probe" / "libcg_probe.so",
            Path.cwd() / "src" / "native" / "cg_probe" / "libcg_probe.so",
        ]
    )
    deduped: list[Path] = []
    for path in paths:
        resolved = path.expanduser()
        if resolved not in deduped:
            deduped.append(resolved)
    return tuple(deduped)


def _sample_hidden_inputs(
    observation: Mapping[str, Any],
    context_features: GameContextFeatures,
    *,
    your_deck: Sequence[int],
    sampler: BeliefSampler,
    worlds: int,
    rng: Any,
    opponent_card_probs: Sequence[float] | None,
    opponent_hand_weights: Sequence[float] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    evidence = extract_observation_evidence(observation)
    opponent_state = opponent_belief_state_from_evidence(evidence, context_features)
    counts: list[int] = []
    values: list[int] = []
    for _ in range(worlds):
        determinization = sampler.sample_from_evidence(
            evidence,
            your_deck=your_deck,
            opponent_state=opponent_state,
            opponent_card_probs=opponent_card_probs,
            opponent_hand_weights=opponent_hand_weights,
            rng=rng,
        )
        hidden = determinization.hidden
        for cards in (
            hidden.your_deck,
            hidden.your_prize,
            hidden.opponent_deck,
            hidden.opponent_prize,
            hidden.opponent_hand,
            hidden.opponent_active,
        ):
            counts.append(len(cards))
            values.extend(int(card) for card in cards)
    if len(counts) != worlds * _HIDDEN_LIST_COUNT:
        raise RuntimeError("internal hidden-list count mismatch")
    return tuple(counts), tuple(values)


def _state_token(observation: Mapping[str, Any]) -> bytes:
    search_input = observation.get("search_begin_input")
    if isinstance(search_input, bytes):
        return search_input
    if isinstance(search_input, str):
        return search_input.encode("ascii")
    raise ValueError("native rollout probe requires search_begin_input")


def _int_array(values: Sequence[int]) -> Any:
    return (ctypes.c_int * len(values))(*[int(value) for value in values])


def _validate_offsets(
    values: Sequence[int],
    *,
    expected_parts: int,
    expected_values: int,
    name: str,
    allow_empty_parts: bool = False,
) -> None:
    if len(values) != expected_parts + 1:
        raise ValueError(f"native fact {name} offsets are misaligned")
    offsets = tuple(int(value) for value in values)
    if (
        offsets[0] != 0
        or offsets[-1] != expected_values
        or any(
            stop < start if allow_empty_parts else stop <= start
            for start, stop in pairwise(offsets)
        )
    ):
        raise ValueError(f"native fact {name} offsets are invalid")


def _options(select: Any) -> Sequence[Any]:
    if isinstance(select, Mapping):
        value = select.get("option", ())
    else:
        value = getattr(select, "option", ())
    return value if isinstance(value, Sequence) and not isinstance(value, str) else ()
