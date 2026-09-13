"""Typed contracts for the dashboard evaluation task workbench."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskKind = Literal[
    "bundle_strength",
    "release_h2h",
    "runtime_elo",
    "package_validation",
    "config_dry_run",
]
TaskMode = Literal["formal", "diagnostic", "utility"]
TaskExecution = Literal["local", "distributed", "remote"]
TaskState = Literal[
    "queued",
    "starting",
    "running",
    "succeeded",
    "failed",
    "cancelling",
    "cancelled",
    "unknown",
]
TaskResourceClass = Literal[
    "light",
    "local_heavy",
    "local_cuda",
    "remote_heavy",
]


class TaskWorkflow(BaseModel):
    """One business workflow exposed by the task wizard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: TaskKind
    label: str
    description: str
    modes: tuple[TaskMode, ...]
    result_views: tuple[str, ...]


class TaskCatalogItem(BaseModel):
    """One opaque artifact or profile selectable by the browser."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    kind: Literal[
        "release_bundle",
        "checkpoint",
        "deck",
        "registered_opponent",
        "public_catalog",
        "side_observations",
        "evaluation_profile",
        "submission_profile",
        "training_profile",
        "runtime_template",
    ]
    label: str
    detail: str | None = None
    fingerprint: str | None = None
    available: bool = True
    unavailable_reason: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskResourceSnapshot(BaseModel):
    """Current resource gate state shown before and after task creation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    training_active: bool
    local_gpu_busy: bool
    running_heavy_tasks: int = Field(ge=0)
    running_light_tasks: int = Field(ge=0)
    detail: str


class TaskCatalog(BaseModel):
    """Complete bounded selection catalog for the business wizard."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflows: tuple[TaskWorkflow, ...]
    releases: tuple[TaskCatalogItem, ...]
    checkpoints: tuple[TaskCatalogItem, ...]
    decks: tuple[TaskCatalogItem, ...]
    registered_opponents: tuple[TaskCatalogItem, ...]
    public_catalogs: tuple[TaskCatalogItem, ...]
    side_observations: tuple[TaskCatalogItem, ...]
    evaluation_profiles: tuple[TaskCatalogItem, ...]
    submission_profiles: tuple[TaskCatalogItem, ...]
    training_profiles: tuple[TaskCatalogItem, ...]
    runtime_templates: tuple[TaskCatalogItem, ...]
    resources: TaskResourceSnapshot


class TaskParticipantRef(BaseModel):
    """Formal bundle, public opponent, or diagnostic checkpoint participant."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["release_bundle", "registered_opponent", "checkpoint"]
    artifact_id: str
    label: str | None = None
    deck_id: str | None = None
    public_catalog_id: str | None = None
    runtime_template_id: str | None = None

    @model_validator(mode="after")
    def valid_source_fields(self) -> TaskParticipantRef:
        """Keep diagnostic-only fields off immutable formal participants."""
        diagnostic_fields = (
            self.deck_id,
            self.public_catalog_id,
            self.runtime_template_id,
        )
        if self.source == "checkpoint":
            if any(value is None for value in diagnostic_fields):
                raise ValueError(
                    "checkpoint participant requires deck, public catalog, "
                    "and runtime template"
                )
        elif any(value is not None for value in diagnostic_fields):
            raise ValueError(
                "release and registered participants do not accept diagnostic fields"
            )
        if self.source == "registered_opponent" and self.label is not None:
            raise ValueError("registered opponent label is catalog-owned")
        return self


class BundleStrengthTaskRequest(BaseModel):
    """Exact bundle gauntlet followed by deck-strength posterior scoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["bundle_strength"] = "bundle_strength"
    mode: Literal["formal", "diagnostic"]
    label: str
    candidates: tuple[TaskParticipantRef, ...]
    opponents: tuple[TaskParticipantRef, ...]
    games_per_matchup: int = Field(ge=2)
    execution: TaskExecution = "distributed"
    side_observations_id: str
    output_label: str

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        return clean_task_label(value)

    @model_validator(mode="after")
    def coherent_participants(self) -> BundleStrengthTaskRequest:
        """Formal tasks use deployment bundles and diagnostic tasks use checkpoints."""
        if not self.candidates or not self.opponents:
            raise ValueError("bundle strength requires candidates and opponents")
        if self.games_per_matchup % 2:
            raise ValueError("mirrored bundle strength games must be even")
        candidate_sources = {item.source for item in self.candidates}
        if self.mode == "formal" and candidate_sources != {"release_bundle"}:
            raise ValueError("formal candidates must be immutable release bundles")
        if self.mode == "diagnostic" and "checkpoint" not in candidate_sources:
            raise ValueError("diagnostic candidates must include a raw checkpoint")
        if any(item.source == "checkpoint" for item in self.opponents):
            raise ValueError("raw checkpoint opponents are not supported")
        return self


class ReleaseH2HTaskRequest(BaseModel):
    """Deployment-native direct H2H between two immutable release bundles."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["release_h2h"] = "release_h2h"
    mode: Literal["formal"] = "formal"
    label: str
    candidate_id: str
    opponent_id: str
    games: int = Field(ge=2)
    execution: TaskExecution = "distributed"
    output_label: str

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        return clean_task_label(value)

    @model_validator(mode="after")
    def balanced_distinct(self) -> ReleaseH2HTaskRequest:
        """Require a balanced campaign between distinct deployments."""
        if self.games % 2:
            raise ValueError("release H2H games must be even")
        if self.candidate_id == self.opponent_id:
            raise ValueError("release H2H participants must be distinct")
        return self


class RuntimeEloTaskRequest(BaseModel):
    """Diagnostic-only runtime deck ladder for one exact checkpoint."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["runtime_elo"] = "runtime_elo"
    mode: Literal["diagnostic"] = "diagnostic"
    label: str
    checkpoint_id: str
    deck_ids: tuple[str, ...]
    public_catalog_id: str
    runtime_template_id: str
    games_per_pair: int = Field(ge=2)
    execution: Literal["local"] = "local"
    output_label: str

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        return clean_task_label(value)

    @model_validator(mode="after")
    def useful_ladder(self) -> RuntimeEloTaskRequest:
        """Require a mirrored ladder with at least two exact decks."""
        if len(set(self.deck_ids)) < 2:
            raise ValueError("runtime Elo requires at least two distinct decks")
        if self.games_per_pair % 2:
            raise ValueError("mirrored runtime Elo games must be even")
        return self


class PackageValidationTaskRequest(BaseModel):
    """Build and validate one submission package without upload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["package_validation"] = "package_validation"
    mode: Literal["utility"] = "utility"
    label: str
    submission_profile_id: str
    output_label: str

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        return clean_task_label(value)


class ConfigDryRunTaskRequest(BaseModel):
    """Resolve one training profile without starting training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["config_dry_run"] = "config_dry_run"
    mode: Literal["utility"] = "utility"
    label: str
    training_profile_id: str

    @field_validator("label")
    @classmethod
    def valid_label(cls, value: str) -> str:
        return clean_task_label(value)


TaskCreateRequest = Annotated[
    BundleStrengthTaskRequest
    | ReleaseH2HTaskRequest
    | RuntimeEloTaskRequest
    | PackageValidationTaskRequest
    | ConfigDryRunTaskRequest,
    Field(discriminator="kind"),
]


class TaskProgress(BaseModel):
    """Normalized progress projected from each tool's durable status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    phase: str
    games_total: int | None = Field(default=None, ge=0)
    games_committed: int | None = Field(default=None, ge=0)
    games_finished: int | None = Field(default=None, ge=0)
    percent: float | None = Field(default=None, ge=0.0, le=100.0)
    rate_per_second: float | None = Field(default=None, ge=0.0)
    eta_seconds: float | None = Field(default=None, ge=0.0)
    quality_warnings: tuple[str, ...] = ()
    detail: str | None = None


class TaskReceipt(BaseModel):
    """Durable schema-v2 identity and lifecycle for one task attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    task_id: str
    kind: TaskKind
    mode: TaskMode
    label: str
    state: TaskState
    resource_class: TaskResourceClass
    queue_reason: str | None = None
    created_at_utc: str
    updated_at_utc: str
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    spec_fingerprint: str
    spec_path: str
    output_dir: str | None = None
    status_path: str | None = None
    log_path: str
    argv: tuple[str, ...]
    cwd: str
    attempt: int = Field(default=1, ge=1)
    retry_of: str | None = None
    worker_pid: int | None = Field(default=None, ge=1)
    worker_start_ticks: int | None = Field(default=None, ge=0)
    exit_code: int | None = None
    detail: str | None = None
    progress: TaskProgress | None = None


class TaskListPayload(BaseModel):
    """Paginated task history."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tasks: tuple[TaskReceipt, ...]
    total: int = Field(ge=0)
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)


class TaskArtifact(BaseModel):
    """One allowlisted result artifact owned by a task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    label: str
    media_type: str
    size_bytes: int = Field(ge=0)
    sha256: str
    previewable: bool
    downloadable: bool


class TaskArtifactList(BaseModel):
    """Bounded artifact index for one task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    artifacts: tuple[TaskArtifact, ...]


class TaskResultSummary(BaseModel):
    """Workflow-specific small result summary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    kind: TaskKind
    state: TaskState
    semantics: str
    headline: str
    metrics: dict[str, Any] = Field(default_factory=dict)
    quality_warnings: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    report_artifact_id: str | None = None


class TaskTablePayload(BaseModel):
    """One bounded result table page."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    table: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, Any], ...]
    offset: int = Field(ge=0)
    limit: int = Field(gt=0)
    returned: int = Field(ge=0)
    has_more: bool


class TaskLogPayload(BaseModel):
    """Bounded UTF-8-safe task log tail."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    text: str
    truncated: bool


class TaskArtifactContent(BaseModel):
    """Bounded textual artifact preview."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    artifact_id: str
    media_type: str
    text: str
    truncated: bool


def clean_task_label(value: str) -> str:
    """Normalize one human-facing task label."""
    cleaned = value.strip()
    if not cleaned:
        raise ValueError("task label must be non-empty")
    if len(cleaned) > 120:
        raise ValueError("task label must contain at most 120 characters")
    return cleaned
