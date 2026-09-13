"""Kaggle submission entrypoint."""

from __future__ import annotations

import atexit
from typing import Any

from ptcg_rl.agent.runtime import PolicyRuntimeAgent

_AGENT = PolicyRuntimeAgent.from_env()
atexit.register(_AGENT.close)


def agent(observation: Any, configuration: Any | None = None) -> list[int]:
    """Return a legal action for Kaggle validation games."""
    return _AGENT.act(observation, configuration)
