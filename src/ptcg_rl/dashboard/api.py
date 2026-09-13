"""FastAPI application for the local RL performance dashboard."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, TypeVar

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

from ptcg_rl.dashboard.jobs import DashboardJobManager
from ptcg_rl.dashboard.league_api import create_league_router
from ptcg_rl.dashboard.models import (
    DeckSeries,
    HealthPayload,
    MatchupRow,
    PerformanceTable,
    RunInfo,
    ScopeName,
    SeriesPoint,
    WindowName,
)
from ptcg_rl.dashboard.network_policy import is_loopback_host
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.dashboard.tasks import DashboardTaskManager
from ptcg_rl.dashboard.workbench import WorkbenchService
from ptcg_rl.dashboard.workbench_api import create_workbench_router

_T = TypeVar("_T")


def create_app(
    repository: DashboardRepository,
    *,
    actions_enabled: bool = False,
) -> FastAPI:
    """Create the dashboard API and serve the built Vue application."""
    jobs = DashboardJobManager(
        repo_root=repository.repo_root,
        enabled=actions_enabled,
        refresh_seconds=repository.config.refresh_seconds,
    )
    tasks = DashboardTaskManager(
        repo_root=repository.repo_root,
        enabled=actions_enabled,
        refresh_seconds=repository.config.refresh_seconds,
        request_token=jobs.request_token,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        del _app
        try:
            yield
        finally:
            tasks.close()

    app = FastAPI(
        title="PTCG RL Training Dashboard",
        version="2.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def require_loopback_request(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Reject reads and writes that do not originate on loopback."""
        client_host = request.client.host if request.client is not None else None
        if not is_loopback_host(
            request.url.hostname,
            allow_test_hosts=True,
        ) or not is_loopback_host(client_host, allow_test_hosts=True):
            return JSONResponse(
                status_code=403,
                content={"detail": "dashboard is localhost-only"},
            )
        return await call_next(request)

    app.include_router(
        create_workbench_router(WorkbenchService(repository), jobs, tasks)
    )
    app.include_router(
        create_league_router(
            repo_root=repository.repo_root,
            database_path=(
                repository.repo_root / repository.config.league_database_path
            ),
            coordinator_url=repository.config.league_coordinator_url,
            deck_aliases=repository.config.deck_aliases,
        )
    )

    @app.get("/api/v1/runs", response_model=list[RunInfo])
    def runs() -> tuple[RunInfo, ...]:
        return repository.list_runs()

    @app.get("/api/v1/runs/{run}/overview", response_model=PerformanceTable)
    def overview(
        run: str,
        window: WindowName = "cumulative",
        scope: ScopeName = "segment",
    ) -> PerformanceTable:
        return _translate_errors(
            lambda: repository.performance_table(run, window=window, scope=scope)
        )

    @app.get("/api/v1/runs/{run}/decks", response_model=PerformanceTable)
    def decks(
        run: str,
        window: WindowName = "cumulative",
        scope: ScopeName = "segment",
    ) -> PerformanceTable:
        return overview(run, window, scope)

    @app.get("/api/v1/runs/{run}/series", response_model=list[SeriesPoint])
    def series(
        run: str,
        deck_label: str | None = None,
        opponent_slice: str = "all",
        rolling_minutes: Annotated[int, Query(ge=1, le=1440)] = 15,
    ) -> tuple[SeriesPoint, ...]:
        return _translate_errors(
            lambda: repository.series(
                run,
                deck_label=deck_label,
                opponent_slice=opponent_slice,
                rolling_minutes=rolling_minutes,
            )
        )

    @app.get("/api/v1/runs/{run}/deck-series", response_model=list[DeckSeries])
    def deck_series(
        run: str,
        opponent_slice: str = "all",
        rolling_minutes: Annotated[int, Query(ge=1, le=1440)] = 15,
        scope: ScopeName = "segment",
    ) -> tuple[DeckSeries, ...]:
        return _translate_errors(
            lambda: repository.deck_series(
                run,
                opponent_slice=opponent_slice,
                rolling_minutes=rolling_minutes,
                scope=scope,
            )
        )

    @app.get("/api/v1/runs/{run}/matchups", response_model=list[MatchupRow])
    def matchups(
        run: str,
        window: WindowName = "cumulative",
        opponent_kind: str | None = None,
        candidate_seat: Literal[0, 1] | None = None,
    ) -> tuple[MatchupRow, ...]:
        return _translate_errors(
            lambda: repository.matchups(
                run,
                window=window,
                opponent_kind=opponent_kind,
                candidate_seat=candidate_seat,
            )
        )

    @app.get("/api/v1/runs/{run}/health", response_model=HealthPayload)
    def health(run: str) -> HealthPayload:
        return _translate_errors(lambda: repository.health(run))

    frontend_dist = Path(__file__).resolve().parent / "frontend" / "dist"
    assets = frontend_dist / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{path:path}", include_in_schema=False, response_model=None)
    def frontend(path: str) -> FileResponse | JSONResponse | dict[str, str]:
        if path == "api" or path.startswith("api/"):
            return JSONResponse(
                status_code=404,
                content={"detail": f"API route does not exist: /{path}"},
            )
        index = frontend_dist / "index.html"
        if index.is_file():
            return FileResponse(index)
        return {
            "detail": "Vue assets are not built; run npm install && npm run build "
            "under src/ptcg_rl/dashboard/frontend"
        }

    return app


def _translate_errors(callback: Callable[[], _T]) -> _T:
    try:
        return callback()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
