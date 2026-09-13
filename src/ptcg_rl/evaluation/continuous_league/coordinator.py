"""FastAPI single-writer coordinator and perpetual maintenance loops."""

from __future__ import annotations

import asyncio
import fcntl
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated

import uvicorn
from fastapi import FastAPI, HTTPException, Query

from ptcg_rl.evaluation.continuous_league.discovery import AssetDiscovery
from ptcg_rl.evaluation.continuous_league.ledger import LeagueLedger
from ptcg_rl.evaluation.continuous_league.models import (
    ActivationRequest,
    ContinuousLeagueConfig,
    ForceChallengeRequest,
    LeaseRequest,
    ManualCheckpointRequest,
    ManualDeckRequest,
    ManualReleaseRequest,
    MatchLease,
    MatchResult,
    WorkerHeartbeat,
)
from ptcg_rl.evaluation.continuous_league.reporting import (
    LeagueRepository,
    LeagueSummary,
)
from ptcg_rl.evaluation.continuous_league.scheduler import ChallengeScheduler
from ptcg_rl.evaluation.continuous_league.telemetry import TelemetryExporter
from ptcg_rl.training.source_identity import resolve_training_source_identity

_LOGGER = logging.getLogger(__name__)


def create_coordinator_app(
    config: ContinuousLeagueConfig,
    *,
    repo_root: Path,
    run_background: bool = True,
    source_commit: str | None = None,
    allow_source_revision: bool = False,
) -> FastAPI:
    """Create the sole state-mutating league service."""
    ledger = LeagueLedger(config, repo_root=repo_root)
    discovery = AssetDiscovery(config, ledger, repo_root=repo_root)
    scheduler = ChallengeScheduler(ledger)
    exporter = TelemetryExporter(ledger)
    repository = LeagueRepository(ledger.database.path)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        del _app
        tasks: list[asyncio.Task[None]] = []
        try:
            if source_commit is not None:
                ledger.bind_source_revision(
                    source_commit,
                    allow_revision=allow_source_revision,
                )
            discovery.initialize()
            discovery.add_anchors()
            if run_background:
                tasks = [
                    asyncio.create_task(
                        _discovery_loop(discovery, config.discovery_seconds),
                        name="continuous-league-discovery",
                    ),
                    asyncio.create_task(
                        _scheduler_loop(scheduler, config.scheduling_seconds),
                        name="continuous-league-scheduler",
                    ),
                    asyncio.create_task(
                        _telemetry_loop(exporter, config.scheduling_seconds),
                        name="continuous-league-telemetry",
                    ),
                ]
            yield
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                with suppress(asyncio.CancelledError):
                    await task
            ledger.close()

    app = FastAPI(
        title="PTCG Continuous League Coordinator",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    @app.post("/worker/v1/heartbeat", status_code=204)
    def heartbeat(request: WorkerHeartbeat) -> None:
        ledger.heartbeat(request)

    @app.post("/worker/v1/lease", response_model=MatchLease | None)
    def lease(request: LeaseRequest) -> MatchLease | None:
        if request.heartbeat.runtime_fingerprint != config.runtime_fingerprint:
            raise HTTPException(status_code=409, detail="worker runtime differs")
        if request.heartbeat.belief_fingerprint != config.belief_fingerprint:
            raise HTTPException(status_code=409, detail="worker belief differs")
        if (
            source_commit is not None
            and request.heartbeat.source_commit != source_commit
        ):
            raise HTTPException(status_code=409, detail="worker source commit differs")
        return ledger.lease_match(request)

    @app.post("/worker/v1/results")
    def result(request: MatchResult) -> dict[str, bool]:
        try:
            return {"recorded": ledger.submit_result(request)}
        except KeyError as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        except ValueError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.get("/control/v1/status", response_model=LeagueSummary)
    def status() -> LeagueSummary:
        return repository.summary()

    @app.post("/control/v1/add-checkpoint")
    def add_checkpoint(request: ManualCheckpointRequest) -> dict[str, str]:
        try:
            asset = discovery.add_checkpoint(
                request.pair_manifest_path,
                automatic=False,
                label=request.label,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "controller_id": f"checkpoint:{asset.policy_sha256}",
            "pair_manifest_sha256": asset.manifest_sha256,
        }

    @app.post("/control/v1/add-deck")
    def add_deck(request: ManualDeckRequest) -> dict[str, bool]:
        try:
            return {
                "created": discovery.add_deck(request.deck_path, label=request.label)
            }
        except (OSError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/control/v1/add-release")
    def add_release(request: ManualReleaseRequest) -> dict[str, str]:
        try:
            controller_id = discovery.add_release(
                request.release_manifest_path, alias=request.alias
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"controller_id": controller_id}

    @app.post("/control/v1/controllers/{controller_id}/activation")
    def activation(controller_id: str, request: ActivationRequest) -> dict[str, bool]:
        try:
            ledger.set_controller_active(
                controller_id, active=request.active, reason=request.reason
            )
        except (KeyError, ValueError) as error:
            raise HTTPException(status_code=404, detail=str(error)) from error
        return {"active": request.active}

    @app.post("/control/v1/challenges")
    def challenge(request: ForceChallengeRequest) -> dict[str, str]:
        try:
            match_id = ledger.enqueue_match(
                request.side_a_bundle_id,
                request.side_b_bundle_id,
                priority=10_000.0,
                reason="manual_forced_challenge",
            )
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"match_id": match_id}

    @app.post("/control/v1/rebuild-ratings")
    def rebuild_ratings(
        confirm: Annotated[bool, Query()] = False,
    ) -> dict[str, int]:
        if not confirm:
            raise HTTPException(status_code=422, detail="confirm=true is required")
        return {"rating_events_replayed": ledger.rebuild_ratings()}

    return app


def run_coordinator(
    config: ContinuousLeagueConfig,
    *,
    repo_root: Path,
    allow_source_revision: bool = False,
) -> None:
    """Hold an exclusive process lock and serve the coordinator."""
    database_path = (
        config.database_path
        if config.database_path.is_absolute()
        else repo_root / config.database_path
    ).resolve()
    lock = _CoordinatorLock(database_path.with_suffix(".coordinator.lock"))
    source_identity = resolve_training_source_identity(repo_root)
    with lock:
        uvicorn.run(
            create_coordinator_app(
                config,
                repo_root=repo_root,
                source_commit=source_identity.source_git_commit,
                allow_source_revision=allow_source_revision,
            ),
            host=config.coordinator_host,
            port=config.coordinator_port,
            log_level="info",
        )


async def _discovery_loop(discovery: AssetDiscovery, interval: float) -> None:
    while True:
        try:
            await asyncio.to_thread(discovery.scan)
        except Exception:
            _LOGGER.exception("continuous league discovery pass failed")
        await asyncio.sleep(interval)


async def _scheduler_loop(scheduler: ChallengeScheduler, interval: float) -> None:
    while True:
        try:
            await asyncio.to_thread(scheduler.fill_queue)
        except Exception:
            _LOGGER.exception("continuous league scheduler pass failed")
        await asyncio.sleep(interval)


async def _telemetry_loop(exporter: TelemetryExporter, interval: float) -> None:
    while True:
        try:
            exported = await asyncio.to_thread(exporter.export_available)
        except Exception:
            _LOGGER.exception("continuous league telemetry export failed")
            exported = 0
        await asyncio.sleep(0.0 if exported else interval)


class _CoordinatorLock:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a+", encoding="utf-8")

    def __enter__(self) -> None:
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "continuous league coordinator is already running"
            ) from error

    def __exit__(self, *_error: object) -> None:
        fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
        self._file.close()


__all__ = ["create_coordinator_app", "run_coordinator"]
