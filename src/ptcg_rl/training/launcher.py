"""Reusable command-line launcher for RL training and monitoring."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, Self, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.rl.native_distributed.lifecycle import (
    NATIVE_WORKER_RECYCLE_EXIT_CODE,
)
from ptcg_rl.training.cuda_allocator import configure_cuda_allocator
from ptcg_rl.training.learner_supervisor import (
    SupervisedCommand,
    supervise_learner,
)
from ptcg_rl.training.run_config import TrainingRunConfig
from ptcg_rl.training.source_identity import (
    load_training_source_identity,
    run_source_binding_path,
    source_adoption_environment_name,
    source_manifest_environment_name,
)
from ptcg_rl.training.source_snapshot import (
    materialize_training_source,
    require_clean_main_checkout,
)

DEFAULT_PROFILE = "rl/train/default"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PYTHONPATH = ("data/sample_submission", "src")
_PAIR_CHECKPOINT_PROFILE_MARKERS = ("simple_stateless", "generalist_sequence")
_COLLECTION_WORKER_THREAD_ENV = {
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}


class TrainingLaunchConfig(BaseModel):
    """Validated options for launching an RL training process."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["learner", "collection-worker"] = "learner"
    profile: str = DEFAULT_PROFILE
    run_version: str | None = None
    output_dir: Path | None = None
    checkpoint: Path | None = None
    resume_checkpoint: Path | None = None
    transition_checkpoint: Path | None = None
    anchor_checkpoint: Path | None = None
    overrides: tuple[str, ...] = ()
    mps: bool = False
    mps_run_id: str | None = None
    worker_id: str | None = None
    coordinator_host: str | None = None
    worker_profile: str | None = None
    immutable_source: bool = False
    source_commit: str | None = None
    adopt_source_identity: bool = False
    dry_run: bool = False

    @field_validator("profile")
    @classmethod
    def valid_profile(cls, value: str) -> str:
        """Reject empty Hydra profile names."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("profile must be non-empty")
        return cleaned

    @field_validator("run_version")
    @classmethod
    def valid_run_version(cls, value: str | None) -> str | None:
        """Reuse the training run version validation."""
        if value is None:
            return None
        return TrainingRunConfig(version=value).version

    @field_validator(
        "mps_run_id",
        "worker_id",
        "coordinator_host",
        "worker_profile",
    )
    @classmethod
    def valid_mps_run_id(cls, value: str | None) -> str | None:
        """Reject empty MPS run IDs."""
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("launcher runtime identifiers must be non-empty")
        return cleaned

    @field_validator("source_commit")
    @classmethod
    def valid_source_commit(cls, value: str | None) -> str | None:
        """Require an unambiguous formal source commit when one is supplied."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) not in (40, 64) or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("source commit must be a full Git object identity")
        return normalized

    @field_validator("overrides")
    @classmethod
    def valid_overrides(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject empty Hydra overrides."""
        cleaned = tuple(item.strip() for item in value)
        if any(not item for item in cleaned):
            raise ValueError("Hydra overrides must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def valid_checkpoint_mode(self) -> Self:
        """Keep model-only warm starts distinct from exact resumes."""
        selected = sum(
            path is not None
            for path in (
                self.checkpoint,
                self.resume_checkpoint,
                self.transition_checkpoint,
            )
        )
        if selected > 1:
            raise ValueError(
                "checkpoint, resume_checkpoint, and transition_checkpoint "
                "are mutually exclusive"
            )
        if self.transition_checkpoint is not None and not _uses_pair_checkpoint(
            self.profile
        ):
            raise ValueError(
                "controller-state transitions require a simple_stateless profile"
            )
        worker_values = (
            self.worker_id,
            self.coordinator_host,
            self.worker_profile,
        )
        if self.role == "learner":
            if any(value is not None for value in worker_values):
                raise ValueError(
                    "worker ID, coordinator host, and worker profile require the "
                    "collection-worker role"
                )
        elif (
            self.run_version is None
            or any(value is None for value in worker_values)
            or selected
            or self.anchor_checkpoint is not None
            or self.mps
            or self.mps_run_id is not None
        ):
            raise ValueError(
                "collection workers require their run version, ID, host, and "
                "profile and forbid learner checkpoint, transition, anchor, "
                "and MPS flags"
            )
        if self.source_commit is not None and not self.immutable_source:
            raise ValueError("source commit requires immutable source execution")
        if self.immutable_source and (
            self.run_version is None or self.output_dir is None
        ):
            raise ValueError(
                "immutable source execution requires run version and output dir"
            )
        if (
            self.role == "collection-worker"
            and self.immutable_source
            and self.source_commit is None
        ):
            raise ValueError(
                "immutable collection workers require the learner source commit"
            )
        if self.adopt_source_identity and (
            self.role != "learner" or self.resume_checkpoint is None
        ):
            raise ValueError(
                "source identity adoption requires an exact learner resume"
            )
        return self


class StatusConfig(BaseModel):
    """Validated options for printing a training run status."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_dir: Path
    watch: float | None = None
    json_output: bool = False

    @field_validator("watch")
    @classmethod
    def valid_watch(cls, value: float | None) -> float | None:
        """Reject invalid watch intervals."""
        if value is not None and value <= 0.0:
            raise ValueError("watch interval must be positive")
        return value


class CommandSpec(BaseModel):
    """A subprocess command and its execution environment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: tuple[str, ...]
    cwd: Path
    env: Mapping[str, str] = Field(default_factory=dict)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments and run the requested launcher action."""
    parser = _argument_parser()
    args = parser.parse_args(argv)
    actions = sum(
        (
            args.status is not None,
            args.performance is not None,
            bool(args.dashboard),
        )
    )
    if actions > 1:
        parser.error("--status, --performance, and --dashboard are mutually exclusive")
    if args.status is not None:
        return print_status(
            StatusConfig(
                run_dir=Path(args.status),
                watch=args.watch,
                json_output=bool(args.json),
            )
        )
    if args.performance is not None:
        return print_performance(
            run=str(args.performance),
            config_path=Path(args.dashboard_config),
            window=str(args.window),
            scope=str(args.scope),
            output_format=str(args.format),
        )
    if args.dashboard:
        return serve_dashboard(
            config_path=Path(args.dashboard_config),
            host=str(args.host),
            port=int(args.port),
        )
    if args.watch is not None or args.json:
        parser.error("--watch and --json require --status")

    config = TrainingLaunchConfig(
        role=cast(
            Literal["learner", "collection-worker"],
            str(args.role),
        ),
        profile=str(args.profile),
        run_version=args.run_version,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        resume_checkpoint=args.resume,
        transition_checkpoint=args.transition,
        anchor_checkpoint=args.anchor_checkpoint,
        overrides=(*args.named_overrides, *args.overrides),
        mps=bool(args.mps),
        mps_run_id=args.mps_run_id,
        worker_id=args.worker_id,
        coordinator_host=args.coordinator_host,
        worker_profile=args.worker_profile,
        immutable_source=bool(args.immutable_source),
        source_commit=args.source_commit,
        adopt_source_identity=bool(args.adopt_source_identity),
        dry_run=bool(args.dry_run),
    )
    return run_training(config)


def print_performance(
    *,
    run: str,
    config_path: Path,
    window: str,
    scope: str,
    output_format: str,
) -> int:
    """Print one shared dashboard performance table."""
    from typing import cast

    from ptcg_rl.dashboard.cli import OutputFormat, render_performance
    from ptcg_rl.dashboard.models import ScopeName, WindowName
    from ptcg_rl.dashboard.repository import DashboardRepository

    resolved_config = (
        config_path if config_path.is_absolute() else REPO_ROOT / config_path
    )
    repository = DashboardRepository.from_config(
        resolved_config,
        repo_root=REPO_ROOT,
    )
    print(
        render_performance(
            repository,
            run,
            window=cast(WindowName, window),
            scope=cast(ScopeName, scope),
            output_format=cast(OutputFormat, output_format),
        )
    )
    return 0


def serve_dashboard(
    *,
    config_path: Path,
    host: str,
    port: int,
) -> int:
    """Serve the dashboard on an explicitly loopback-only bind host."""
    if port <= 0 or port > 65535:
        raise ValueError("dashboard port must be between 1 and 65535")
    from ptcg_rl.dashboard.server import run_dashboard

    resolved_config = (
        config_path if config_path.is_absolute() else REPO_ROOT / config_path
    )
    run_dashboard(
        config_path=resolved_config,
        repo_root=REPO_ROOT,
        host=host,
        port=port,
    )
    return 0


def build_training_command(
    config: TrainingLaunchConfig,
    *,
    source_root: Path = REPO_ROOT,
    source_manifest: Path | None = None,
) -> CommandSpec:
    """Return the subprocess command for one training launch."""
    command = [
        sys.executable,
        str(source_root / "src" / "tools" / "rl_train.py"),
        "--config-name",
        config.profile,
        *_hydra_overrides(config),
    ]
    env = _training_env(
        role=config.role,
        mps_run_id=config.mps_run_id,
        worker_id=config.worker_id,
        coordinator_host=config.coordinator_host,
        worker_profile=config.worker_profile,
        source_root=(None if source_root == REPO_ROOT else source_root),
        source_manifest=source_manifest,
        adopt_source_identity=config.adopt_source_identity,
    )
    if config.mps:
        command = [str(source_root / "src" / "tools" / "rl_train_mps.sh"), *command]
    return CommandSpec(argv=tuple(command), cwd=source_root, env=env)


def run_training(config: TrainingLaunchConfig) -> int:
    """Launch training or print the command in dry-run mode."""
    if config.dry_run:
        spec = build_training_command(config)
        _print_dry_run(spec)
        if config.immutable_source:
            print(
                "immutable source revision: "
                f"{config.source_commit or '<run-binding-or-HEAD>'}"
            )
        return 0
    launch_config = config
    if config.immutable_source:
        require_clean_main_checkout(REPO_ROOT)
        revision = _immutable_source_revision(config)
        source = materialize_training_source(REPO_ROOT, revision=revision)
        launch_config = _absolute_artifact_paths(config)
        spec = build_training_command(
            launch_config,
            source_root=source.root,
            source_manifest=source.manifest_path,
        )
    else:
        spec = build_training_command(config)
    if (
        launch_config.role == "learner"
        and launch_config.immutable_source
        and launch_config.output_dir is not None
    ):
        return supervise_learner(
            _supervised_command(spec),
            output_dir=launch_config.output_dir,
        )
    while True:
        completed = subprocess.run(
            spec.argv,
            cwd=spec.cwd,
            env=dict(spec.env),
            check=False,
        )
        return_code = int(completed.returncode)
        if (
            launch_config.role != "collection-worker"
            or return_code != NATIVE_WORKER_RECYCLE_EXIT_CODE
        ):
            return return_code
        # The child already closed its sockets, native arenas, CUDA objects,
        # and artifact frames at a committed window boundary. Starting a new
        # process image is the allocator recovery boundary; the immutable
        # source, stable worker ID, and tmux owner remain unchanged.
        time.sleep(1.0)


def _supervised_command(spec: CommandSpec) -> SupervisedCommand:
    return SupervisedCommand(argv=spec.argv, cwd=spec.cwd, env=spec.env)


def print_status(config: StatusConfig) -> int:
    """Print one or more monitor snapshots for a run directory."""
    from tools.rl_monitor import build_status, format_status

    while True:
        status = build_status(config.run_dir)
        if config.json_output:
            print(json.dumps(status, indent=2, sort_keys=True))
        else:
            print(format_status(status))
        if config.watch is None:
            return 0
        time.sleep(config.watch)


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role",
        choices=("learner", "collection-worker"),
        default="learner",
        help="Run the H200 learner/coordinator or a remote native worker.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help="Hydra training config name.",
    )
    parser.add_argument(
        "--run-version",
        default=None,
        help="Override run.version for a new artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output_dir explicitly.",
    )
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Warm-start model weights without restoring learner state.",
    )
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Exactly resume model, optimizer, scheduler, and progress state.",
    )
    checkpoint_group.add_argument(
        "--transition",
        type=Path,
        default=None,
        help=(
            "Start a new simple-stateless run from an exact pair while preserving "
            "settled curriculum and deck-balance state."
        ),
    )
    parser.add_argument(
        "--anchor-checkpoint",
        type=Path,
        default=None,
        help="Override anchor_checkpoint_path.",
    )
    parser.add_argument(
        "-o",
        "--override",
        action="append",
        dest="named_overrides",
        default=[],
        help="Additional Hydra override. May be repeated.",
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Wrap training with src/tools/rl_train_mps.sh.",
    )
    parser.add_argument(
        "--mps-run-id",
        default=None,
        help="Set PTCG_RL_MPS_RUN_ID for the MPS wrapper.",
    )
    parser.add_argument(
        "--worker-id",
        default=None,
        help="Stable formal collection-worker identity.",
    )
    parser.add_argument(
        "--coordinator-host",
        default=None,
        help="Private-LAN learner/coordinator host for a collection worker.",
    )
    parser.add_argument(
        "--worker-profile",
        default=None,
        help="Worker runtime profile key from native_distributed.worker_profiles.",
    )
    parser.add_argument(
        "--immutable-source",
        action="store_true",
        help="Execute from a verified no-.git snapshot of the run source commit.",
    )
    parser.add_argument(
        "--source-commit",
        default=None,
        help="Full source commit for a new snapshot or remote formal worker.",
    )
    parser.add_argument(
        "--adopt-source-identity",
        action="store_true",
        help="Bind one audited legacy exact-resume run to its current source.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the launch command without starting training.",
    )
    parser.add_argument(
        "--status",
        type=Path,
        default=None,
        help="Print monitor status for an existing run directory.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        help="Refresh interval in seconds for --status.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON for --status.",
    )
    parser.add_argument(
        "--performance",
        default=None,
        metavar="CURRENT_OR_RUN",
        help="Print deck performance for `current` or one local run ID.",
    )
    parser.add_argument(
        "--window",
        choices=("cumulative", "15m", "60m"),
        default="cumulative",
        help="Performance time window.",
    )
    parser.add_argument(
        "--scope",
        choices=("segment", "lineage"),
        default="segment",
        help="Aggregate the selected segment or its configured lineage.",
    )
    parser.add_argument(
        "--format",
        choices=("table", "json", "markdown"),
        default="table",
        help="Performance output format.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Serve the local training dashboard.",
    )
    parser.add_argument(
        "--dashboard-config",
        type=Path,
        default=Path("configs/rl/dashboard.yaml"),
        help="Dashboard Hydra-compatible YAML profile.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Dashboard loopback bind host; non-loopback values are rejected.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="Dashboard bind port.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Trailing Hydra overrides passed directly to the training profile.",
    )
    return parser


def _hydra_overrides(config: TrainingLaunchConfig) -> tuple[str, ...]:
    overrides: list[str] = []
    if config.run_version is not None:
        overrides.append(f"run.version={config.run_version}")
    if config.output_dir is not None:
        overrides.append(f"output_dir={config.output_dir}")
    if config.checkpoint is not None:
        if _uses_pair_checkpoint(config.profile):
            overrides.append("resume.mode=weights_only")
            overrides.append(
                f"resume.pair_manifest_path={_pair_manifest_path(config.checkpoint)}"
            )
        else:
            overrides.append(f"checkpoint_path={config.checkpoint}")
            overrides.append("resume.mode=warm_start")
    if config.resume_checkpoint is not None:
        if _uses_pair_checkpoint(config.profile):
            overrides.append("resume.mode=resume")
            overrides.append("resume.transition_action=continue_training")
            overrides.append("resume.startup_action=continue_training")
            overrides.append(
                "resume.pair_manifest_path="
                f"{_pair_manifest_path(config.resume_checkpoint)}"
            )
            # A transition profile may pin source/target controller identities.
            # An explicit exact resume restores the pair's own settled state and
            # must not inherit those transition-only assertions.
            overrides.append("resume.preserve_controller_state=false")
            overrides.append("resume.expected_source_pair_version=null")
            overrides.append("resume.expected_source_pair_manifest_sha256=null")
            overrides.append("resume.expected_source_curriculum_fingerprint=null")
            overrides.append("resume.expected_target_curriculum_fingerprint=null")
            overrides.append("resume.gae_lambda_transition_from=null")
            overrides.append("resume.supervised_artifact_manifest_path=null")
            overrides.append("resume.optimizer_scope_transition=null")
            overrides.append("resume.deck_balance_transition=null")
            overrides.append("resume.curriculum_lane_coverage_rebase=null")
            overrides.append("resume.registry_transition=null")
            overrides.append("resume.topology_transition=null")
            overrides.append("resume.public_catalog_transition=null")
            overrides.append("resume.anchor_transition=null")
            overrides.append("resume.bc_overlay=null")
            overrides.append("resume.supervised=null")
        else:
            overrides.append(f"checkpoint_path={config.resume_checkpoint}")
            overrides.append("resume.mode=resume")
            # A profile may pin a historical sidecar. Selecting a checkpoint on
            # the command line must infer the matching versioned sidecar instead.
            overrides.append("resume.state_path=null")
            overrides.append("resume.state_size_bytes=null")
            overrides.append("resume.state_sha256=null")
    if config.transition_checkpoint is not None:
        overrides.append("resume.mode=transition")
        overrides.append(
            "resume.pair_manifest_path="
            f"{_pair_manifest_path(config.transition_checkpoint)}"
        )
        overrides.append("resume.preserve_controller_state=true")
    if config.anchor_checkpoint is not None:
        overrides.append(f"anchor_checkpoint_path={config.anchor_checkpoint}")
    overrides.extend(config.overrides)
    return tuple(overrides)


def _uses_pair_checkpoint(profile: str) -> bool:
    """Return whether a profile uses exact policy/learner pair manifests."""
    return any(marker in profile for marker in _PAIR_CHECKPOINT_PROFILE_MARKERS)


def _pair_manifest_path(checkpoint: Path) -> Path:
    """Map a versioned policy path to its colocated exact-pair manifest."""
    name = checkpoint.name
    if name.startswith("policy_v") and name.endswith(".pt"):
        version = name.removeprefix("policy_v").removesuffix(".pt")
        if version.isdigit():
            return checkpoint.with_name(f"checkpoint_pair_v{version}.json")
    return checkpoint


def _training_env(
    *,
    role: Literal["learner", "collection-worker"],
    mps_run_id: str | None,
    worker_id: str | None,
    coordinator_host: str | None,
    worker_profile: str | None,
    source_root: Path | None,
    source_manifest: Path | None,
    adopt_source_identity: bool,
) -> dict[str, str]:
    env = dict(os.environ)
    configure_cuda_allocator(env)
    env["PYTHONPATH"] = _pythonpath(
        env.get("PYTHONPATH"),
        source_root=source_root,
    )
    env["PTCG_RL_TRAINING_ROLE"] = role
    if role == "collection-worker":
        env.update(_COLLECTION_WORKER_THREAD_ENV)
    if mps_run_id is not None:
        env["PTCG_RL_MPS_RUN_ID"] = mps_run_id
    for name, value in (
        ("PTCG_RL_WORKER_ID", worker_id),
        ("PTCG_RL_COORDINATOR_HOST", coordinator_host),
        ("PTCG_RL_WORKER_PROFILE", worker_profile),
    ):
        if value is not None:
            env[name] = value
    if source_manifest is not None:
        env[source_manifest_environment_name()] = str(source_manifest)
    if adopt_source_identity:
        env[source_adoption_environment_name()] = "1"
    return env


def _pythonpath(existing: str | None, *, source_root: Path | None) -> str:
    parts = (
        list(DEFAULT_PYTHONPATH)
        if source_root is None
        else [str(source_root / path) for path in DEFAULT_PYTHONPATH]
    )
    if existing:
        parts.extend(
            part for part in existing.split(os.pathsep) if part and part not in parts
        )
    return os.pathsep.join(parts)


def _immutable_source_revision(config: TrainingLaunchConfig) -> str:
    """Select the immutable source commit, preferring an existing run binding."""
    if config.output_dir is None:
        raise ValueError("immutable source execution requires output dir")
    output_dir = _absolute_path(config.output_dir)
    binding_path = run_source_binding_path(output_dir)
    if binding_path.is_file():
        bound = load_training_source_identity(binding_path)
        requested = config.source_commit
        if requested is not None and requested != bound.source_git_commit:
            raise RuntimeError(
                "requested source commit differs from the immutable run binding"
            )
        return bound.source_git_commit
    return config.source_commit or "HEAD"


def _absolute_artifact_paths(config: TrainingLaunchConfig) -> TrainingLaunchConfig:
    """Keep live artifacts outside the extracted source tree."""
    updates: dict[str, object] = {}
    for name in (
        "output_dir",
        "checkpoint",
        "resume_checkpoint",
        "transition_checkpoint",
        "anchor_checkpoint",
    ):
        value = getattr(config, name)
        if value is not None:
            updates[name] = _absolute_path(value)
    return config.model_copy(update=updates)


def _absolute_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _print_dry_run(spec: CommandSpec) -> None:
    print(f"cwd: {spec.cwd}")
    print(f"PYTHONPATH: {spec.env.get('PYTHONPATH', '')}")
    mps_run_id = spec.env.get("PTCG_RL_MPS_RUN_ID")
    if mps_run_id is not None:
        print(f"PTCG_RL_MPS_RUN_ID: {mps_run_id}")
    print(f"PTCG_RL_TRAINING_ROLE: {spec.env.get('PTCG_RL_TRAINING_ROLE', '')}")
    source_manifest = spec.env.get(source_manifest_environment_name())
    if source_manifest is not None:
        print(f"{source_manifest_environment_name()}: {source_manifest}")
    for name in (
        "PTCG_RL_WORKER_ID",
        "PTCG_RL_COORDINATOR_HOST",
        "PTCG_RL_WORKER_PROFILE",
    ):
        value = spec.env.get(name)
        if value is not None:
            print(f"{name}: {value}")
    print(f"command: {shlex.join(spec.argv)}")


if __name__ == "__main__":
    raise SystemExit(main())
