"""Adapters for bundled third-party benchmark opponents."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import random
import shutil
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from itertools import count
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.engine.protocols import ObservationInput

_THIRD_PARTY_SRC_ENV = "PTCG_RL_THIRD_PARTY_SRC"


def third_party_source_root(
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Return the bundled or explicitly mounted read-only opponent source."""
    environment = os.environ if environ is None else environ
    override = environment.get(_THIRD_PARTY_SRC_ENV, "").strip()
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            raise ValueError(f"{_THIRD_PARTY_SRC_ENV} must be an absolute path")
        return path
    return records.REPO_ROOT / "third_party" / "pokemon-tcg-ai-battle" / "src"


_TP_SRC = third_party_source_root()
_PUBLIC_ROOT = _TP_SRC / "public_opponents" / "kaggle_public"
_CAPTURED_PUBLIC_ROOT = (
    records.REPO_ROOT / "data" / "public_opponents" / "kaggle_20260723"
)
_FROZEN_PUBLIC_ROOT = (
    records.REPO_ROOT
    / "src"
    / "ptcg_rl"
    / "opponents"
    / "public_agents"
    / "kaggle_20260807"
)
_SAMPLE_SUBMISSION = records.REPO_ROOT / "data" / "sample_submission"
_ARCHALUDON_LEAN_CONFIG_ENV = "POKEMON_TCG_ARCHALUDON_LEAN_CONFIG"
_PUBLIC_MODULE_SEQUENCE = count()

PUBLIC_OPPONENT_FIXTURES: dict[str, str] = {
    "kaggle_score1_pixiux_lucario_v62": "kaggle_score1_pixiux_lucario_v62",
    "kaggle_score2_kojimar_simple_baseline": "kaggle_score2_kojimar_simple_baseline",
    "kaggle_score3_aristophanivan_multiply_940": (
        "kaggle_score3_aristophanivan_multiply_940"
    ),
    "kaggle_score4_makthanithin_lucario_v62": "kaggle_score4_makthanithin_lucario_v62",
    "kaggle_score5_makthanithin_1084_baseline": "kaggle_score5_makthanithin_1084_baseline",
    "public_alakazam": "alakazam_public",
    "public_dragapult": "dragapult_public",
    "public_lucario_lab": "lucario_lab_public",
    "public_multiply_lucario": "multiply_lucario_public",
    "kaggle_20260720_daniil_conservative": "daniil_conservative",
    "kaggle_20260720_roman_strong_start_v10": "roman_strong_start_v10",
    "kaggle_20260722_nursrijan_advanced_planning": ("nursrijan_advanced_planning"),
    "kaggle_20260722_prvsiyan_alakazam_v12": "prvsiyan_alakazam_v12",
    "kaggle_20260722_prvsiyan_control_v11": "prvsiyan_control_v11",
    "kaggle_20260722_prvsiyan_tusk_crustle_v1": ("prvsiyan_tusk_crustle_v1"),
    "kaggle_20260807_jazivxt_crustle_v29": "jazivxt_crustle_v29",
    "kaggle_20260807_jazivxt_garchomp_v28": "jazivxt_garchomp_v28",
    "kaggle_20260807_kiyotah_abomasnow_rule": "kiyotah_abomasnow_rule",
    "kaggle_20260807_kiyotah_default_abomasnow_rule": (
        "kiyotah_default_abomasnow_rule"
    ),
    "kaggle_20260807_makimakiai_water38": "makimakiai_water38",
}

_CAPTURED_PUBLIC_OPPONENTS = frozenset(
    {
        "kaggle_20260720_daniil_conservative",
        "kaggle_20260720_roman_strong_start_v10",
        "kaggle_20260722_nursrijan_advanced_planning",
        "kaggle_20260722_prvsiyan_alakazam_v12",
        "kaggle_20260722_prvsiyan_control_v11",
        "kaggle_20260722_prvsiyan_tusk_crustle_v1",
    }
)

_FROZEN_PUBLIC_OPPONENTS = frozenset(
    {
        "kaggle_20260807_jazivxt_crustle_v29",
        "kaggle_20260807_jazivxt_garchomp_v28",
        "kaggle_20260807_kiyotah_abomasnow_rule",
        "kaggle_20260807_kiyotah_default_abomasnow_rule",
        "kaggle_20260807_makimakiai_water38",
    }
)

_PUBLIC_RUNTIME_FILES: dict[str, tuple[str, ...]] = {
    "kaggle_20260807_jazivxt_crustle_v29": (
        "deck.csv",
        "gated_submission_inference_v29.py",
        "gpu_submission_inference_v28.py",
        "main.py",
        "opponent_card_counter_v28.py",
        "selector_templates_v29.json",
        "selector_weights_v28.json",
    ),
    "kaggle_20260807_jazivxt_garchomp_v28": (
        "deck.csv",
        "gpu_submission_inference_v28.py",
        "main.py",
        "selector_weights_v28.json",
    ),
}

_SEARCH_PUBLIC_OPPONENTS = frozenset(
    {
        "kaggle_score3_aristophanivan_multiply_940",
        "public_lucario_lab",
        "public_multiply_lucario",
        "kaggle_20260720_daniil_conservative",
        "kaggle_20260720_roman_strong_start_v10",
        "kaggle_20260722_nursrijan_advanced_planning",
        "kaggle_20260722_prvsiyan_alakazam_v12",
    }
)


class ObservationCallable(Protocol):
    """Callable shape used by third-party Kaggle agents."""

    def __call__(self, obs_dict: dict[str, Any]) -> Sequence[int]:
        """Return selected option indices."""


@dataclass
class ThirdPartyAgent:
    """BattleAgent wrapper around a third-party callable."""

    name: str
    _agent: ObservationCallable
    _reset: Callable[[], None] | None = None

    def act(self, observation: ObservationInput) -> Sequence[int]:
        """Call the wrapped third-party agent with a mapping observation."""
        action = self._agent(_observation_dict(observation))
        return tuple(int(index) for index in action)

    def reset(self) -> None:
        """Reset wrapped per-game state, when the wrapped agent exposes it."""
        if self._reset is not None:
            self._reset()


def build_heuristic_agent(name: str = "heuristic") -> ThirdPartyAgent:
    """Build the third-party heuristic opponent."""
    module = _heuristic_module()
    return ThirdPartyAgent(name=name, _agent=_module_agent(module))


def build_mcts_agent(name: str = "mcts") -> ThirdPartyAgent:
    """Build the third-party MCTS opponent."""
    module = _mcts_module()
    return ThirdPartyAgent(name=name, _agent=_module_agent(module))


def build_archaludon_lean_agent(name: str) -> ThirdPartyAgent:
    """Build an engine-search-free, per-instance Archaludon specialist."""
    module = _archaludon_lean_module()
    namespace = vars(module)
    memory_module = namespace["memory_mod"]
    memory = memory_module.IntentMemory()
    module_agent = cast(ObservationCallable, namespace["agent"])

    def agent(observation: dict[str, Any]) -> Sequence[int]:
        previous = namespace["_MEMORY"]
        namespace["_MEMORY"] = memory
        try:
            return module_agent(observation)
        finally:
            namespace["_MEMORY"] = previous

    return ThirdPartyAgent(
        name=name,
        _agent=cast(ObservationCallable, agent),
        _reset=memory.reset,
    )


def build_public_agent(name: str, *, seed: int = 0) -> ThirdPartyAgent:
    """Build one public Kaggle notebook opponent by registry name."""
    if name not in PUBLIC_OPPONENT_FIXTURES:
        raise KeyError(f"unknown public opponent: {name}")
    agent = _PublicNotebookAgent(name, seed=seed)
    return ThirdPartyAgent(
        name=name,
        _agent=cast(ObservationCallable, agent),
        _reset=agent.reset,
    )


def public_deck_path(name: str) -> Path:
    """Return the bundled deck path for one public opponent."""
    return _public_fixture_root(name) / "deck.csv"


def public_implementation_files(name: str) -> tuple[str, ...]:
    """Return the complete repository-relative runtime source inventory."""
    root = _public_fixture_root(name)
    runtime_files = _PUBLIC_RUNTIME_FILES.get(name, ("deck.csv", "main.py"))
    files = (
        *(root / relative for relative in runtime_files),
        records.REPO_ROOT / "src" / "ptcg_rl" / "opponents" / "spec.py",
        records.REPO_ROOT / "src" / "ptcg_rl" / "opponents" / "third_party.py",
    )
    return tuple(
        sorted(
            str(path.relative_to(records.REPO_ROOT)).replace("\\", "/")
            for path in files
        )
    )


def public_requires_search(name: str) -> bool:
    """Return whether a public opponent may use the process-global Search API."""
    return name in _SEARCH_PUBLIC_OPPONENTS


class _PublicNotebookAgent:
    """Per-game module instance for one immutable public notebook bundle."""

    def __init__(self, name: str, *, seed: int) -> None:
        self.name = name
        self.root = _public_fixture_root(name)
        self.main_path = self.root / "main.py"
        self.runtime_root = (
            records.REPO_ROOT
            / "tmp"
            / "public_opponent_runtime"
            / str(os.getpid())
            / name
        )
        self._random_state = random.Random(seed).getstate()
        self._module: ModuleType | None = None

    def reset(self) -> None:
        """Discard notebook globals so the next battle gets fresh game state."""
        self._module = None

    def __call__(self, observation: dict[str, Any]) -> Sequence[int]:
        """Run one decision while isolating cwd and the stdlib random stream."""
        self._prepare_runtime()
        with self._random_context(), _cwd(self.runtime_root):
            module = self._module
            if module is None:
                module = self._load_module()
                self._module = module
            return tuple(int(index) for index in module.agent(observation))

    def _prepare_runtime(self) -> None:
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        source_deck = self.root / "deck.csv"
        runtime_deck = self.runtime_root / "deck.csv"
        if not runtime_deck.exists():
            shutil.copyfile(source_deck, runtime_deck)

    def _load_module(self) -> ModuleType:
        if not self.main_path.is_file():
            raise FileNotFoundError(f"missing public opponent agent: {self.main_path}")
        module_name = f"_ptcg_public_{self.name}_{next(_PUBLIC_MODULE_SEQUENCE)}"
        spec = importlib.util.spec_from_file_location(module_name, self.main_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import public opponent: {self.main_path}")
        module = importlib.util.module_from_spec(spec)
        with _public_import_context(self.root):
            spec.loader.exec_module(module)
        if not callable(getattr(module, "agent", None)):
            raise AttributeError(
                f"public opponent has no callable agent: {self.main_path}"
            )
        return module

    @contextmanager
    def _random_context(self) -> Iterator[None]:
        global_state = random.getstate()
        random.setstate(self._random_state)
        try:
            yield
        finally:
            self._random_state = random.getstate()
            random.setstate(global_state)


def _public_fixture_root(name: str) -> Path:
    try:
        fixture = PUBLIC_OPPONENT_FIXTURES[name]
    except KeyError as error:
        raise KeyError(f"unknown public opponent: {name}") from error
    if name in _CAPTURED_PUBLIC_OPPONENTS:
        root = _CAPTURED_PUBLIC_ROOT
    elif name in _FROZEN_PUBLIC_OPPONENTS:
        root = _FROZEN_PUBLIC_ROOT
    else:
        root = _PUBLIC_ROOT
    return root / fixture


@contextmanager
def _public_import_context(root: Path) -> Iterator[None]:
    original_path = list(sys.path)
    try:
        for path in reversed((str(_SAMPLE_SUBMISSION), str(_TP_SRC), str(root))):
            if path not in sys.path:
                sys.path.insert(0, path)
        yield
    finally:
        sys.path[:] = original_path


@contextmanager
def _cwd(path: Path) -> Iterator[None]:
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


@contextmanager
def _third_party_import_context() -> Iterator[None]:
    """Append the third-party src path for unique package imports only."""
    path = str(_TP_SRC)
    added = path not in sys.path
    if added:
        sys.path.append(path)
    try:
        yield
    finally:
        if added:
            with suppress(ValueError):
                sys.path.remove(path)


@lru_cache(maxsize=1)
def _heuristic_module() -> ModuleType:
    with _third_party_import_context():
        return importlib.import_module("heuristic.agent")


@lru_cache(maxsize=1)
def _mcts_module() -> ModuleType:
    with _third_party_import_context():
        return importlib.import_module("mcts.agent")


@lru_cache(maxsize=1)
def _archaludon_lean_module() -> ModuleType:
    overrides: dict[str, Any] = {}
    existing = os.getenv(_ARCHALUDON_LEAN_CONFIG_ENV)
    if existing:
        parsed = json.loads(existing)
        if isinstance(parsed, dict):
            overrides.update(parsed)
    overrides["effect_oracle_enabled"] = False
    os.environ[_ARCHALUDON_LEAN_CONFIG_ENV] = json.dumps(overrides, sort_keys=True)
    with _third_party_import_context():
        module = importlib.import_module("agents.archaludon_lean.policy")
    vars(module)["tactics"].load_config.cache_clear()
    return module


def _module_agent(module: ModuleType) -> ObservationCallable:
    agent = getattr(module, "agent", None)
    if not callable(agent):
        raise AttributeError(f"third-party module has no callable agent: {module}")
    return cast(ObservationCallable, agent)


def _observation_dict(observation: ObservationInput) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError("third-party opponents require mapping observations")
    return dict(observation)
