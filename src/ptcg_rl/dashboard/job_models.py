"""Validated contracts for dashboard-owned local jobs."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

JobState = Literal[
    "starting",
    "running",
    "succeeded",
    "failed",
    "cancelling",
    "cancelled",
    "unknown",
]
ActionScope = Literal["disabled", "loopback"]
TemplateId = Literal[
    "bundle_evaluation",
    "package_validation",
    "config_dry_run",
    "public_environment_refresh",
]


class JobTemplate(BaseModel):
    """One fixed command adapter exposed to the browser."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: TemplateId
    label: str
    description: str
    fields: tuple[dict[str, object], ...]


class StartJobRequest(BaseModel):
    """Validated template selection plus adapter-owned parameters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    template_id: TemplateId
    parameters: dict[str, object] = Field(default_factory=dict)


class JobReceipt(BaseModel):
    """Durable identity and lifecycle state for one dashboard-owned process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: str
    template_id: TemplateId
    state: JobState
    created_at_utc: str
    updated_at_utc: str
    parameters: dict[str, object]
    argv: tuple[str, ...]
    cwd: str
    log_path: str
    worker_pid: int | None = Field(default=None, ge=1)
    exit_code: int | None = None
    detail: str | None = None


class DashboardSession(BaseModel):
    """Capabilities and anti-CSRF token for one local server process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    actions_enabled: bool
    action_scope: ActionScope = "disabled"
    request_token: str | None
    refresh_seconds: int = Field(gt=0)


class JobProgressPayload(BaseModel):
    """Durable structured progress for one detached dashboard job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    job_id: str
    template_id: TemplateId
    state: JobState
    phase: str
    message: str
    completed: int = Field(default=0, ge=0)
    total: int = Field(default=0, ge=0)
    percent: float = Field(default=0.0, ge=0.0, le=100.0)
    current_date: str | None = None
    updated_at_utc: str
    detail: str | None = None


class JobLogPayload(BaseModel):
    """Bounded tail of one dashboard job log."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str
    text: str
    truncated: bool
