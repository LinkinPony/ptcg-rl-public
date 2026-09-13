"""Runtime loading helpers for the bundled Kaggle ``cg`` package."""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from ptcg_rl.engine.protocols import ObservationInput, ObservationLike


def _import_error_message(module: str) -> str:
    return (
        f"Could not import {module}. Run with "
        "PYTHONPATH=data/sample_submission:src, or from a packaged Kaggle agent."
    )


def _candidate_cg_roots() -> tuple[Path, ...]:
    # Kaggle removes the extracted agent dir from sys.path right after main.py
    # is imported, so lazy `cg` imports must be re-resolved module-relative:
    # packaged layout ships `cg/` next to the `ptcg_rl` package, while the repo
    # layout keeps it under data/sample_submission.
    module_path = Path(__file__).resolve()
    package_root = module_path.parents[2]
    repo_root = module_path.parents[3]
    return (package_root, repo_root / "data" / "sample_submission")


def _import_cg_module(module: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    for root in _candidate_cg_roots():
        if (root / "cg" / "__init__.py").exists() and str(root) not in sys.path:
            sys.path.append(str(root))
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(_import_error_message(module)) from exc


@lru_cache(maxsize=1)
def load_cg_api() -> Any:
    """Return the bundled ``cg.api`` module, loading ``libcg`` on demand."""
    return _import_cg_module("cg.api")


@lru_cache(maxsize=1)
def load_cg_game() -> Any:
    """Return the bundled ``cg.game`` module, loading ``libcg`` on demand."""
    return _import_cg_module("cg.game")


def to_engine_observation(observation: ObservationInput) -> ObservationLike:
    """Convert a raw observation dict to the engine dataclass when needed."""
    if isinstance(observation, Mapping):
        cg_api = load_cg_api()
        return cast(ObservationLike, cg_api.to_observation_class(dict(observation)))
    return observation
