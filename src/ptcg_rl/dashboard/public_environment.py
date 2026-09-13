"""Read-only Dashboard service for compact public Daily snapshots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ptcg_rl.dashboard.public_environment_models import (
    EnvironmentScoreSemantics,
    EnvironmentWindow,
    PublicEnvironmentMatchupsPayload,
    PublicEnvironmentMatrixPayload,
    PublicEnvironmentPayload,
)
from ptcg_rl.dashboard.repository import DashboardRepository
from ptcg_rl.data.kaggle.public_environment_analysis import (
    load_roster,
    roster_fingerprint,
)
from ptcg_rl.data.kaggle.public_environment_models import PublicEnvironmentConfig


class PublicEnvironmentService:
    """Load bounded snapshot JSON without launching work or touching the network."""

    def __init__(self, repository: DashboardRepository) -> None:
        self.repository = repository
        self.output_root = (
            repository.repo_root / "outputs" / "kaggle_public_environment"
        ).resolve()

    def summary(
        self,
        run: str,
        *,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentPayload:
        """Return one local immutable summary, or a typed unavailable payload."""
        run_id = self.repository.resolve_run_id(run)
        latest = self._latest(run_id, checkpoint_version)
        if not latest:
            return _missing_payload(
                run_id=run_id,
                checkpoint_version=checkpoint_version,
                window_days=window_days,
                reason="public environment snapshot has not been generated",
            )
        windows = latest.get("windows")
        detail = windows.get(str(window_days)) if isinstance(windows, dict) else None
        relative = detail.get("path") if isinstance(detail, dict) else None
        if not isinstance(relative, str):
            return _missing_payload(
                run_id=run_id,
                checkpoint_version=checkpoint_version,
                window_days=window_days,
                reason="public environment window is absent from latest manifest",
            )
        path = (self.output_root / relative).resolve()
        if not path.is_relative_to(self.output_root) or not path.is_file():
            raise ValueError("public environment manifest contains an invalid path")
        raw = _read_json(path)
        _bind_request_identity(
            raw,
            run_id=run_id,
            checkpoint_version=checkpoint_version,
        )
        self._add_display_names(raw)
        payload = PublicEnvironmentPayload.model_validate(raw)
        return payload.model_copy(update={"matrix": ()})

    def matrix(
        self,
        run: str,
        *,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentMatrixPayload:
        """Return the complete bounded roster-by-major-meta matrix."""
        payload = self._full(run, window_days, checkpoint_version)
        return PublicEnvironmentMatrixPayload(
            run_id=payload.run_id,
            checkpoint_version=payload.checkpoint_version,
            window_days=payload.window_days,
            available=payload.available,
            snapshot_fingerprint=payload.snapshot_fingerprint,
            cells=payload.matrix,
        )

    def matchups(
        self,
        run: str,
        *,
        deck_hash: str,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentMatchupsPayload:
        """Return one active compact deck identity's public matchup rows."""
        payload = self._full(run, window_days, checkpoint_version)
        known = {row.deck_hash for row in payload.roster_standings}
        if payload.available and deck_hash not in known:
            raise KeyError(f"unknown active deck_hash: {deck_hash}")
        return PublicEnvironmentMatchupsPayload(
            run_id=payload.run_id,
            checkpoint_version=payload.checkpoint_version,
            window_days=payload.window_days,
            available=payload.available,
            snapshot_fingerprint=payload.snapshot_fingerprint,
            deck_hash=deck_hash,
            cells=tuple(
                cell for cell in payload.matrix if cell.candidate_deck_hash == deck_hash
            ),
        )

    def _full(
        self,
        run: str,
        window_days: EnvironmentWindow,
        checkpoint_version: int | None,
    ) -> PublicEnvironmentPayload:
        run_id = self.repository.resolve_run_id(run)
        latest = self._latest(run_id, checkpoint_version)
        if not latest:
            return _missing_payload(
                run_id=run_id,
                checkpoint_version=checkpoint_version,
                window_days=window_days,
                reason="public environment snapshot has not been generated",
            )
        windows = latest.get("windows")
        detail = windows.get(str(window_days)) if isinstance(windows, dict) else None
        relative = detail.get("path") if isinstance(detail, dict) else None
        if not isinstance(relative, str):
            raise ValueError("public environment window path is unavailable")
        path = (self.output_root / relative).resolve()
        if not path.is_relative_to(self.output_root) or not path.is_file():
            raise ValueError("public environment manifest contains an invalid path")
        raw = _read_json(path)
        _bind_request_identity(
            raw,
            run_id=run_id,
            checkpoint_version=checkpoint_version,
        )
        self._add_display_names(raw)
        return PublicEnvironmentPayload.model_validate(raw)

    def _latest(self, run_id: str, checkpoint_version: int | None) -> dict[str, Any]:
        roster = load_roster(
            PublicEnvironmentConfig(
                run_root=self.repository.run_root,
                checkpoint_version=checkpoint_version,
            ),
            run_id=run_id,
        )
        identity = roster_fingerprint(roster)
        current = _read_json(
            self.output_root / "rosters" / identity / "latest.json"
        )
        if current:
            return current
        return self._legacy_latest(identity, checkpoint_version)

    def _legacy_latest(
        self,
        roster_identity: str,
        checkpoint_version: int | None,
    ) -> dict[str, Any]:
        """Read compatible pre-migration artifacts without copying or rewriting."""
        checkpoint = (
            "current"
            if checkpoint_version is None
            else f"checkpoint_{checkpoint_version}"
        )
        compatible = []
        for path in self.output_root.glob(f"runs/*/{checkpoint}/latest.json"):
            payload = _read_json(path)
            if payload.get("roster_fingerprint") == roster_identity:
                compatible.append(payload)
        return max(
            compatible,
            key=lambda payload: (
                str(payload.get("latest_date") or ""),
                str(payload.get("generated_at_utc") or ""),
            ),
            default={},
        )

    def _add_display_names(self, payload: dict[str, Any]) -> None:
        for raw in payload.get("roster_standings") or []:
            if isinstance(raw, dict) and isinstance(raw.get("deck_label"), str):
                raw["display_name"] = self.repository.deck_display_name(
                    raw["deck_label"]
                )
        for raw in payload.get("meta_decks") or []:
            if not isinstance(raw, dict):
                continue
            label = raw.get("deck_label")
            raw["display_name"] = (
                self.repository.deck_display_name(label)
                if bool(raw.get("active_roster")) and isinstance(label, str)
                else label
            )


def _missing_payload(
    *,
    run_id: str,
    checkpoint_version: int | None,
    window_days: EnvironmentWindow,
    reason: str,
) -> PublicEnvironmentPayload:
    return PublicEnvironmentPayload(
        available=False,
        run_id=run_id,
        checkpoint_version=checkpoint_version,
        window_days=window_days,
        as_of_date=None,
        generated_at_utc=None,
        snapshot_fingerprint=None,
        source_scope="kaggle_daily_all_episodes",
        unavailable_reason=reason,
        score_semantics=EnvironmentScoreSemantics(submission_data_used=False),
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _bind_request_identity(
    payload: dict[str, Any],
    *,
    run_id: str,
    checkpoint_version: int | None,
) -> None:
    """Bind roster-scoped evidence to the selected compatible training run."""
    payload["run_id"] = run_id
    payload["checkpoint_version"] = checkpoint_version


__all__ = ["PublicEnvironmentService"]
