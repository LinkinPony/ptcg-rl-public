"""FastAPI routes for the decision-oriented dashboard workbench."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Callable
from typing import Annotated, TypeVar, cast
from urllib.parse import urlparse

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse

from ptcg_rl.dashboard.deck_selection_models import DeckSelectionPayload
from ptcg_rl.dashboard.job_models import (
    DashboardSession,
    JobLogPayload,
    JobProgressPayload,
    JobReceipt,
    JobTemplate,
    StartJobRequest,
)
from ptcg_rl.dashboard.jobs import DashboardJobManager
from ptcg_rl.dashboard.models import RunInfo, WindowName
from ptcg_rl.dashboard.opponent_allocation_models import (
    AllocationSort,
    OpponentAllocationMatchupPage,
    OpponentAllocationSummaryPayload,
)
from ptcg_rl.dashboard.public_environment_models import (
    EnvironmentWindow,
    PublicEnvironmentMatchupsPayload,
    PublicEnvironmentMatrixPayload,
    PublicEnvironmentPayload,
)
from ptcg_rl.dashboard.task_models import (
    TaskArtifactContent,
    TaskArtifactList,
    TaskCatalog,
    TaskCreateRequest,
    TaskListPayload,
    TaskLogPayload,
    TaskReceipt,
    TaskResultSummary,
    TaskTablePayload,
)
from ptcg_rl.dashboard.tasks import DashboardTaskManager
from ptcg_rl.dashboard.training_deck_strength_models import (
    OpponentSet,
    TrainingController,
    TrainingDeckMatchupsPayload,
    TrainingDeckMatrixPayload,
    TrainingDeckSeriesPayload,
    TrainingDeckStrengthPayload,
    TrainingEvidenceRange,
)
from ptcg_rl.dashboard.two_deck_selection_models import TwoDeckSelectionPayload
from ptcg_rl.dashboard.workbench import WorkbenchService
from ptcg_rl.dashboard.workbench_models import (
    CheckpointInfo,
    ComparisonPayload,
    ComparisonRequest,
    DeckEvidencePayload,
    DeckMatchupPayload,
    LearnerSeriesPayload,
    WorkbenchSummary,
)
from ptcg_rl.rl.opponent_pool.adaptive import PortfolioName
from ptcg_rl.rl.opponent_pool.role_budget import RoleBudgetName
from ptcg_rl.rl.training_game_statistics import TrainingGameStatistics

_T = TypeVar("_T")


def create_workbench_router(
    service: WorkbenchService,
    jobs: DashboardJobManager,
    tasks: DashboardTaskManager,
) -> APIRouter:
    """Create the v2 router without changing existing v1 contracts."""
    router = APIRouter(prefix="/api/v2")

    @router.get("/session", response_model=DashboardSession)
    def session() -> DashboardSession:
        return jobs.session

    @router.get("/runs", response_model=list[RunInfo])
    def runs() -> tuple[RunInfo, ...]:
        return service.repository.list_runs()

    @router.get("/runs/{run}/summary", response_model=WorkbenchSummary)
    def summary(
        run: str,
        window: WindowName = "15m",
    ) -> WorkbenchSummary:
        return _translate(lambda: service.summary(run, window=window))

    @router.get(
        "/runs/{run}/training-games",
        response_model=TrainingGameStatistics,
    )
    def training_games(run: str) -> TrainingGameStatistics:
        return _translate(lambda: service.training_game_statistics(run))

    @router.get("/runs/{run}/checkpoints", response_model=list[CheckpointInfo])
    def checkpoints(run: str) -> tuple[CheckpointInfo, ...]:
        return _translate(lambda: service.checkpoints(run))

    @router.get(
        "/runs/{run}/learner-series",
        response_model=LearnerSeriesPayload,
    )
    def learner_series(run: str) -> LearnerSeriesPayload:
        return _translate(lambda: service.learner_series(run))

    @router.get(
        "/runs/{run}/opponent-allocation",
        response_model=OpponentAllocationSummaryPayload,
    )
    def opponent_allocation(run: str) -> OpponentAllocationSummaryPayload:
        return _translate(lambda: service.opponent_allocation_summary(run))

    @router.get(
        "/runs/{run}/opponent-allocation/matchups",
        response_model=OpponentAllocationMatchupPage,
    )
    def opponent_allocation_matchups(
        run: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=500)] = 100,
        sort: AllocationSort = "planned_games",
        candidate_deck_digest: str | None = None,
        artifact_id: str | None = None,
        portfolio: PortfolioName | None = None,
        role: RoleBudgetName | None = None,
        candidate_seat: Annotated[int | None, Query(ge=0, le=1)] = None,
    ) -> OpponentAllocationMatchupPage:
        return _translate(
            lambda: service.opponent_allocation_matchups(
                run,
                offset=offset,
                limit=limit,
                sort=sort,
                candidate_deck_digest=candidate_deck_digest,
                artifact_id=artifact_id,
                portfolio=portfolio,
                role=role,
                candidate_seat=candidate_seat,
            )
        )

    @router.get(
        "/runs/{run}/deck-evidence",
        response_model=DeckEvidencePayload,
    )
    def deck_evidence(
        run: str,
        window: WindowName = "cumulative",
    ) -> DeckEvidencePayload:
        return _translate(lambda: service.deck_evidence(run, window=window))

    @router.get(
        "/runs/{run}/decks/{deck_label}/matchups",
        response_model=DeckMatchupPayload,
    )
    def deck_matchups(
        run: str,
        deck_label: str,
        window: WindowName = "cumulative",
    ) -> DeckMatchupPayload:
        return _translate(
            lambda: service.deck_matchups(
                run,
                deck_label=deck_label,
                window=window,
            )
        )

    @router.get(
        "/runs/{run}/training-deck-strength",
        response_model=TrainingDeckStrengthPayload,
    )
    def training_deck_strength(
        run: str,
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
    ) -> TrainingDeckStrengthPayload:
        return _translate(
            lambda: service.training_deck_strength(
                run,
                checkpoint_version=checkpoint_version,
            )
        )

    @router.get(
        "/runs/{run}/two-deck-recommendation",
        response_model=TwoDeckSelectionPayload,
    )
    def two_deck_recommendation(
        run: str,
        training_range: TrainingEvidenceRange = "checkpoint",
        window_days: Annotated[int, Query()] = 7,
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
    ) -> TwoDeckSelectionPayload:
        resolved_window = _translate(lambda: _environment_window(window_days))
        return _translate(
            lambda: service.two_deck_recommendation(
                run,
                checkpoint_version=checkpoint_version,
                training_range=training_range,
                window_days=resolved_window,
            )
        )

    @router.get(
        "/runs/{run}/public-environment",
        response_model=PublicEnvironmentPayload,
    )
    def public_environment(
        run: str,
        window_days: Annotated[int, Query()] = 7,
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
    ) -> PublicEnvironmentPayload:
        resolved_window = _translate(lambda: _environment_window(window_days))
        return _translate(
            lambda: service.public_environment_summary(
                run,
                window_days=resolved_window,
                checkpoint_version=checkpoint_version,
            )
        )

    @router.get(
        "/runs/{run}/public-environment/matrix",
        response_model=PublicEnvironmentMatrixPayload,
    )
    def public_environment_matrix(
        run: str,
        window_days: Annotated[int, Query()] = 7,
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
    ) -> PublicEnvironmentMatrixPayload:
        resolved_window = _translate(lambda: _environment_window(window_days))
        return _translate(
            lambda: service.public_environment_matrix(
                run,
                window_days=resolved_window,
                checkpoint_version=checkpoint_version,
            )
        )

    @router.get(
        "/runs/{run}/public-environment/decks/{deck_hash}/matchups",
        response_model=PublicEnvironmentMatchupsPayload,
    )
    def public_environment_matchups(
        run: str,
        deck_hash: str,
        window_days: Annotated[int, Query()] = 7,
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
    ) -> PublicEnvironmentMatchupsPayload:
        resolved_window = _translate(lambda: _environment_window(window_days))
        return _translate(
            lambda: service.public_environment_matchups(
                run,
                deck_hash=deck_hash,
                window_days=resolved_window,
                checkpoint_version=checkpoint_version,
            )
        )

    @router.post(
        "/runs/{run}/public-environment/refresh",
        response_model=JobReceipt,
    )
    def refresh_public_environment(
        run: str,
        request: Request,
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
    ) -> JobReceipt:
        _require_local_origin(request)
        resolved_run = _translate(lambda: service.repository.resolve_run_id(run))
        parameters: dict[str, object] = {
            "run": resolved_run,
            "checkpoint_version": checkpoint_version,
        }
        active = next(
            (
                receipt
                for receipt in jobs.list_jobs()
                if receipt.template_id == "public_environment_refresh"
                and receipt.state in {"starting", "running", "cancelling"}
                and receipt.parameters == parameters
            ),
            None,
        )
        if active is not None:
            return active
        return _translate(
            lambda: jobs.start(
                StartJobRequest(
                    template_id="public_environment_refresh",
                    parameters=parameters,
                )
            )
        )

    @router.get(
        "/runs/{run}/training-deck-strength/series",
        response_model=TrainingDeckSeriesPayload,
    )
    def training_deck_strength_series(
        run: str,
        range_name: Annotated[
            TrainingEvidenceRange,
            Query(alias="range"),
        ] = "checkpoint",
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
        controller: TrainingController = "all",
    ) -> TrainingDeckSeriesPayload:
        return _translate(
            lambda: service.training_deck_strength_series(
                run,
                range_name=range_name,
                checkpoint_version=checkpoint_version,
                controller=controller,
            )
        )

    @router.get(
        "/runs/{run}/training-deck-strength/matrix",
        response_model=TrainingDeckMatrixPayload,
    )
    def training_deck_strength_matrix(
        run: str,
        range_name: Annotated[
            TrainingEvidenceRange,
            Query(alias="range"),
        ] = "checkpoint",
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
        controller: TrainingController = "all",
        candidate_seat: Annotated[int | None, Query(ge=0, le=1)] = None,
        opponent_set: OpponentSet = "active",
    ) -> TrainingDeckMatrixPayload:
        return _translate(
            lambda: service.training_deck_strength_matrix(
                run,
                range_name=range_name,
                checkpoint_version=checkpoint_version,
                controller=controller,
                candidate_seat=candidate_seat,
                opponent_set=opponent_set,
            )
        )

    @router.get(
        "/runs/{run}/training-deck-strength/decks/{deck_label}/matchups",
        response_model=TrainingDeckMatchupsPayload,
    )
    def training_deck_strength_matchups(
        run: str,
        deck_label: str,
        range_name: Annotated[
            TrainingEvidenceRange,
            Query(alias="range"),
        ] = "checkpoint",
        checkpoint_version: Annotated[int | None, Query(ge=0)] = None,
        controller: TrainingController = "all",
        candidate_seat: Annotated[int | None, Query(ge=0, le=1)] = None,
        opponent_set: OpponentSet = "all",
    ) -> TrainingDeckMatchupsPayload:
        return _translate(
            lambda: service.training_deck_strength_matchups(
                run,
                deck_label=deck_label,
                range_name=range_name,
                checkpoint_version=checkpoint_version,
                controller=controller,
                candidate_seat=candidate_seat,
                opponent_set=opponent_set,
            )
        )

    @router.post("/comparison", response_model=ComparisonPayload)
    def comparison(request: ComparisonRequest) -> ComparisonPayload:
        return _translate(lambda: service.compare(request))

    @router.get("/jobs/templates", response_model=list[JobTemplate])
    def templates() -> tuple[JobTemplate, ...]:
        return jobs.templates()

    @router.get("/jobs", response_model=list[JobReceipt])
    def list_jobs() -> tuple[JobReceipt, ...]:
        return jobs.list_jobs()

    @router.get("/jobs/{job_id}", response_model=JobReceipt)
    def get_job(job_id: str) -> JobReceipt:
        return _translate(lambda: jobs.get_job(job_id))

    @router.get("/jobs/{job_id}/progress", response_model=JobProgressPayload)
    def job_progress(job_id: str) -> JobProgressPayload:
        return _translate(lambda: jobs.job_progress(job_id))

    @router.get("/jobs/{job_id}/log", response_model=JobLogPayload)
    def job_log(
        job_id: str,
        tail_bytes: Annotated[int, Query(ge=1, le=1_000_000)] = 100_000,
    ) -> JobLogPayload:
        return _translate(lambda: jobs.job_log(job_id, tail_bytes=tail_bytes))

    @router.get("/tasks/catalog", response_model=TaskCatalog)
    def task_catalog() -> TaskCatalog:
        return _translate(tasks.catalog)

    @router.get("/tasks", response_model=TaskListPayload)
    def list_tasks(
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        kind: str | None = None,
        state: str | None = None,
    ) -> TaskListPayload:
        return _translate(
            lambda: tasks.list_tasks(
                offset=offset,
                limit=limit,
                kind=kind,
                state=state,
            )
        )

    @router.get("/tasks/{task_id}", response_model=TaskReceipt)
    def get_task(task_id: str) -> TaskReceipt:
        return _translate(lambda: tasks.get(task_id))

    @router.get("/tasks/{task_id}/result", response_model=TaskResultSummary)
    def task_result(task_id: str) -> TaskResultSummary:
        return _translate(lambda: tasks.results.summary(tasks.get(task_id)))

    @router.get(
        "/tasks/{task_id}/runtime-ladder",
        response_model=DeckSelectionPayload,
    )
    def task_runtime_ladder(task_id: str) -> DeckSelectionPayload:
        return _translate(lambda: tasks.results.deck_selection(tasks.get(task_id)))

    @router.get(
        "/tasks/{task_id}/tables/{table}",
        response_model=TaskTablePayload,
    )
    def task_table(
        task_id: str,
        table: str,
        offset: Annotated[int, Query(ge=0)] = 0,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        filters: str | None = None,
    ) -> TaskTablePayload:
        return _translate(
            lambda: tasks.results.table(
                tasks.get(task_id),
                table,
                offset=offset,
                limit=limit,
                filters=_parse_filters(filters),
            )
        )

    @router.get("/tasks/{task_id}/artifacts", response_model=TaskArtifactList)
    def task_artifacts(task_id: str) -> TaskArtifactList:
        return _translate(lambda: tasks.results.artifacts(tasks.get(task_id)))

    @router.get(
        "/tasks/{task_id}/artifacts/{artifact_id}/content",
        response_model=TaskArtifactContent,
    )
    def task_artifact_content(
        task_id: str,
        artifact_id: str,
        max_bytes: Annotated[int, Query(ge=1, le=1_000_000)] = 200_000,
    ) -> TaskArtifactContent:
        return _translate(
            lambda: tasks.results.artifact_content(
                tasks.get(task_id),
                artifact_id,
                max_bytes=max_bytes,
            )
        )

    @router.get(
        "/tasks/{task_id}/artifacts/{artifact_id}/download",
        response_model=None,
    )
    def task_artifact_download(task_id: str, artifact_id: str) -> FileResponse:
        path = _translate(
            lambda: tasks.results.artifact_path(tasks.get(task_id), artifact_id)
        )
        return FileResponse(path, filename=path.name)

    @router.get("/tasks/{task_id}/log", response_model=TaskLogPayload)
    def task_log(
        task_id: str,
        tail_bytes: Annotated[int, Query(ge=1, le=1_000_000)] = 100_000,
    ) -> TaskLogPayload:
        return _translate(
            lambda: tasks.results.log(
                tasks.get(task_id),
                tail_bytes=tail_bytes,
            )
        )

    @router.post("/tasks", response_model=TaskReceipt)
    def start_task(
        request: Request,
        payload: TaskCreateRequest,
        token: Annotated[str | None, Header(alias="X-Dashboard-Token")] = None,
    ) -> TaskReceipt:
        _require_local_action(request, token=token, jobs=jobs)
        return _translate(lambda: tasks.start(payload))

    @router.post("/tasks/{task_id}/cancel", response_model=TaskReceipt)
    def cancel_task(
        task_id: str,
        request: Request,
        token: Annotated[str | None, Header(alias="X-Dashboard-Token")] = None,
    ) -> TaskReceipt:
        _require_local_action(request, token=token, jobs=jobs)
        return _translate(lambda: tasks.cancel(task_id))

    @router.post("/tasks/{task_id}/retry", response_model=TaskReceipt)
    def retry_task(
        task_id: str,
        request: Request,
        token: Annotated[str | None, Header(alias="X-Dashboard-Token")] = None,
    ) -> TaskReceipt:
        _require_local_action(request, token=token, jobs=jobs)
        return _translate(lambda: tasks.retry(task_id))

    return router


def _require_local_action(
    request: Request,
    *,
    token: str | None,
    jobs: DashboardJobManager,
) -> None:
    if not jobs.enabled or jobs.request_token is None:
        raise HTTPException(status_code=403, detail="dashboard actions are disabled")
    if token != jobs.request_token:
        raise HTTPException(status_code=403, detail="invalid dashboard request token")
    host = request.url.hostname
    client = request.client.host if request.client is not None else None
    if host not in {"localhost", "testserver"} and not _is_loopback(host):
        raise HTTPException(
            status_code=403, detail="dashboard action host is not trusted"
        )
    if client not in {"testclient"} and not _is_loopback(client):
        raise HTTPException(
            status_code=403, detail="dashboard action client is not trusted"
        )
    origin = request.headers.get("origin")
    if origin:
        origin_host = urlparse(origin).hostname
        if origin_host not in {"localhost", "testserver"} and not _is_loopback(
            origin_host
        ):
            raise HTTPException(
                status_code=403,
                detail="dashboard action origin is not trusted",
            )


def _is_loopback(value: str | None) -> bool:
    if value is None:
        return False
    try:
        return ipaddress.ip_address(value).is_loopback
    except ValueError:
        return False


def _translate(callback: Callable[[], _T]) -> _T:
    try:
        return callback()
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except PermissionError as error:
        raise HTTPException(status_code=403, detail=str(error)) from error
    except (OSError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def _parse_filters(value: str | None) -> dict[str, str]:
    if value is None or not value.strip():
        return {}
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as error:
        raise ValueError("table filters must be one JSON object") from error
    if not isinstance(payload, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in payload.items()
    ):
        raise ValueError("table filters must map string columns to string values")
    return payload


def _environment_window(value: int) -> EnvironmentWindow:
    if value not in {1, 2, 7, 14}:
        raise ValueError("window_days must be one of 1, 2, 7, or 14")
    return cast(EnvironmentWindow, value)


def _require_local_origin(request: Request) -> None:
    """Reject cross-origin browser writes; app middleware owns network trust."""
    origin = request.headers.get("origin")
    if not origin:
        return
    origin_host = urlparse(origin).hostname
    if origin_host not in {"localhost", "testserver"} and not _is_loopback(origin_host):
        raise HTTPException(
            status_code=403,
            detail="dashboard action origin is not trusted",
        )
