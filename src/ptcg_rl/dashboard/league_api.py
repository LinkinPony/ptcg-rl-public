"""Dashboard v3 read projections and coordinator-proxied league controls."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query

from ptcg_rl.evaluation.continuous_league.models import (
    ActivationRequest,
    ForceChallengeRequest,
    ManualCheckpointRequest,
    ManualDeckRequest,
    ManualReleaseRequest,
)
from ptcg_rl.evaluation.continuous_league.reporting import (
    LeagueRepository,
    LeagueSummary,
    MatchupRow,
    StandingRow,
    TrendPoint,
    WorkerRow,
)


def create_league_router(
    *,
    repo_root: Path,
    database_path: Path | None = None,
    coordinator_url: str = "http://127.0.0.1:8788",
    deck_aliases: Mapping[str, str] | None = None,
) -> APIRouter:
    """Create trusted-network league routes without another permission model."""
    repository = LeagueRepository(
        database_path
        or repo_root / "outputs" / "evaluation" / "continuous_league" / "league.sqlite3",
        deck_aliases=deck_aliases,
    )
    client = _CoordinatorClient(coordinator_url)
    router = APIRouter(prefix="/api/v3/league")

    @router.get("/summary", response_model=LeagueSummary)
    def summary() -> LeagueSummary:
        return repository.summary()

    @router.get("/standings", response_model=list[StandingRow])
    def global_standings() -> tuple[StandingRow, ...]:
        return repository.standings("bundles")

    @router.get("/checkpoints", response_model=list[StandingRow])
    def checkpoints() -> tuple[StandingRow, ...]:
        return tuple(
            item
            for item in repository.standings("controllers")
            if item.kind == "checkpoint"
        )

    @router.get("/decks", response_model=list[StandingRow])
    def decks() -> tuple[StandingRow, ...]:
        return repository.standings("decks")

    @router.get("/bundles", response_model=list[StandingRow])
    def bundles() -> tuple[StandingRow, ...]:
        return repository.standings("bundles")

    @router.get("/candidates", response_model=list[StandingRow])
    def candidates() -> tuple[StandingRow, ...]:
        return tuple(
            item
            for item in repository.standings("controllers")
            if item.candidate_kind == "automatic"
        )

    @router.get("/matchups", response_model=list[MatchupRow])
    def matchups(
        limit: Annotated[int, Query(ge=1, le=5000)] = 500,
    ) -> tuple[MatchupRow, ...]:
        return repository.matchups(limit=limit)

    @router.get("/workers", response_model=list[WorkerRow])
    def workers() -> tuple[WorkerRow, ...]:
        return repository.workers()

    @router.get("/trend/{component_id}", response_model=list[TrendPoint])
    def trend(
        component_id: str,
        limit: Annotated[int, Query(ge=1, le=5000)] = 512,
    ) -> tuple[TrendPoint, ...]:
        return repository.trend(component_id, limit=limit)

    @router.post("/assets/checkpoints")
    def add_checkpoint(request: ManualCheckpointRequest) -> Any:
        return client.post("/control/v1/add-checkpoint", request)

    @router.post("/assets/decks")
    def add_deck(request: ManualDeckRequest) -> Any:
        return client.post("/control/v1/add-deck", request)

    @router.post("/assets/releases")
    def add_release(request: ManualReleaseRequest) -> Any:
        return client.post("/control/v1/add-release", request)

    @router.post("/controllers/{controller_id}/activation")
    def activation(controller_id: str, request: ActivationRequest) -> Any:
        return client.post(
            f"/control/v1/controllers/{controller_id}/activation", request
        )

    @router.post("/challenges")
    def challenge(request: ForceChallengeRequest) -> Any:
        return client.post("/control/v1/challenges", request)

    return router


class _CoordinatorClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def post(self, path: str, payload: Any) -> Any:
        body = payload.model_dump_json().encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600.0) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise HTTPException(status_code=error.code, detail=detail) from error
        except (TimeoutError, urllib.error.URLError) as error:
            raise HTTPException(
                status_code=503,
                detail="continuous league coordinator is unavailable",
            ) from error


__all__ = ["create_league_router"]
