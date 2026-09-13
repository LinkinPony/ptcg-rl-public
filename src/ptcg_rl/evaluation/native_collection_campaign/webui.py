"""Read-only FastAPI WebUI for a native collection campaign."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator
from starlette.responses import Response

from ptcg_rl.dashboard.network_policy import (
    is_loopback_host,
    require_loopback_bind_host,
)
from ptcg_rl.evaluation.native_collection_campaign.live import CampaignLiveReader

_STATIC_DIR = Path(__file__).resolve().parent / "webui_static"


class CampaignWebUiConfig(BaseModel):
    """Validated launch settings for the localhost-only monitoring service."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_path: Path
    host: str = "127.0.0.1"
    port: int = Field(default=8792, ge=1, le=65_535)
    refresh_seconds: float = Field(default=2.0, ge=0.5, le=60.0)
    gpu_poll_seconds: float = Field(default=5.0, ge=1.0, le=60.0)
    standings_limit: int = Field(default=250, ge=1, le=5_000)
    cells_limit: int = Field(default=6_000, ge=1, le=100_000)

    @field_validator("host")
    @classmethod
    def loopback_host(cls, value: str) -> str:
        """Keep the operational dashboard off unauthenticated networks."""
        return require_loopback_bind_host(value)


def create_campaign_webui_app(
    reader: CampaignLiveReader,
    *,
    refresh_seconds: float = 2.0,
) -> FastAPI:
    """Create a read-only API and dependency-free monitoring application."""
    if refresh_seconds <= 0.0:
        raise ValueError("refresh interval must be positive")
    if not (_STATIC_DIR / "index.html").is_file():
        raise FileNotFoundError(_STATIC_DIR / "index.html")

    app = FastAPI(
        title="PTCG Native Campaign Monitor",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
    )

    @app.middleware("http")
    async def require_loopback_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Reject browser and API requests that bypass loopback."""
        client_host = request.client.host if request.client is not None else None
        if not is_loopback_host(
            request.url.hostname,
            allow_test_hosts=True,
        ) or not is_loopback_host(client_host, allow_test_hosts=True):
            return JSONResponse(
                status_code=403,
                content={"detail": "campaign WebUI is localhost-only"},
            )
        return await call_next(request)

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "refresh_seconds": refresh_seconds,
            **reader.plan_summary(),
        }

    @app.get("/api/plan")
    def plan() -> dict[str, object]:
        return reader.plan_summary()

    @app.get("/api/snapshot")
    def snapshot() -> dict[str, object]:
        try:
            return reader.snapshot()
        except (OSError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_STATIC_DIR / "index.html")

    return app


def run_campaign_webui(
    config: CampaignWebUiConfig,
    *,
    root: Path,
) -> None:
    """Load the immutable plan and serve its localhost-only live monitor."""
    reader = CampaignLiveReader.from_plan_path(
        config.plan_path,
        root=root,
        gpu_poll_seconds=config.gpu_poll_seconds,
        standings_limit=config.standings_limit,
        cells_limit=config.cells_limit,
    )
    uvicorn.run(
        create_campaign_webui_app(
            reader,
            refresh_seconds=config.refresh_seconds,
        ),
        host=config.host,
        port=config.port,
        log_level="info",
    )


__all__ = [
    "CampaignWebUiConfig",
    "create_campaign_webui_app",
    "run_campaign_webui",
]
