"""Dashboard server bootstrap used by the repository launcher."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from ptcg_rl.dashboard.api import create_app
from ptcg_rl.dashboard.network_policy import require_loopback_bind_host
from ptcg_rl.dashboard.repository import DashboardRepository


def run_dashboard(
    *,
    config_path: Path,
    repo_root: Path,
    host: str,
    port: int,
) -> None:
    """Run one loopback-only FastAPI process with the built Vue SPA."""
    bind_host = require_loopback_bind_host(host)
    repository = DashboardRepository.from_config(config_path, repo_root=repo_root)
    uvicorn.run(
        create_app(
            repository,
            actions_enabled=True,
        ),
        host=bind_host,
        port=port,
        log_level="info",
    )
