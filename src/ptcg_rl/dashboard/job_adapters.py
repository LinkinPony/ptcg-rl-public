"""Fixed, repository-confined command adapters for dashboard jobs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

from ptcg_rl.dashboard.job_models import JobTemplate, StartJobRequest

JOB_TEMPLATES = (
    JobTemplate(
        template_id="bundle_evaluation",
        label="Bundle gauntlet + posterior scoring",
        description="启动或恢复现有 bundle/score JSON 定义的标准评测。",
        fields=(
            {
                "name": "bundle_config",
                "label": "Bundle config JSON",
                "kind": "repo_file",
                "required": True,
            },
            {
                "name": "score_config",
                "label": "Score config JSON",
                "kind": "repo_file",
                "required": True,
            },
            {
                "name": "status_path",
                "label": "Status output",
                "kind": "output_file",
                "required": True,
            },
            {
                "name": "mode",
                "label": "Execution mode",
                "kind": "select",
                "options": ["distributed", "local"],
                "required": True,
            },
        ),
    ),
    JobTemplate(
        template_id="public_environment_refresh",
        label="Refresh Kaggle Daily public environment",
        description="刷新紧凑Daily语料并重建当前roster快照；不读取submission。",
        fields=(
            {
                "name": "run",
                "label": "Training run",
                "kind": "text",
                "required": True,
            },
            {
                "name": "checkpoint_version",
                "label": "Checkpoint version",
                "kind": "integer_optional",
                "required": False,
            },
        ),
    ),
    JobTemplate(
        template_id="package_validation",
        label="Submission package validation",
        description="构建并验证 release package；该适配器永远不传 --submit。",
        fields=(
            {
                "name": "profile",
                "label": "Submission profile",
                "kind": "submission_profile",
                "required": True,
            },
            {
                "name": "output",
                "label": "Archive output",
                "kind": "output_file",
                "required": True,
            },
        ),
    ),
    JobTemplate(
        template_id="config_dry_run",
        label="Training config dry-run",
        description="解析 Hydra training profile 并打印命令，不启动训练。",
        fields=(
            {
                "name": "profile",
                "label": "Training profile",
                "kind": "training_profile",
                "required": True,
            },
        ),
    ),
)


class JobCommandAdapters:
    """Resolve validated requests into argv without shell interpretation."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def build(
        self,
        request: StartJobRequest,
        *,
        job_id: str,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        """Build one allowlisted command."""
        if request.template_id == "bundle_evaluation":
            return self._bundle(request.parameters)
        if request.template_id == "public_environment_refresh":
            return self._public_environment(request.parameters, job_id=job_id)
        if request.template_id == "package_validation":
            return self._package(request.parameters)
        return self._dry_run(request.parameters, job_id=job_id)

    def _public_environment(
        self,
        raw: dict[str, object],
        *,
        job_id: str,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        _exact_fields(raw, {"run", "checkpoint_version"})
        run = _safe_run_id(str(raw["run"]))
        run_dir = self.repo_root / "outputs" / "training" / "rl" / run
        if not run_dir.is_dir():
            raise FileNotFoundError(f"unknown local training run: {run}")
        checkpoint_raw = raw["checkpoint_version"]
        checkpoint = None if checkpoint_raw is None else int(str(checkpoint_raw))
        if checkpoint is not None and checkpoint < 0:
            raise ValueError("checkpoint_version must be non-negative")
        argv = (
            sys.executable,
            str(
                self.repo_root
                / "src"
                / "tools"
                / "refresh_kaggle_public_environment.py"
            ),
            f"run_id={run}",
            f"checkpoint_version={checkpoint if checkpoint is not None else 'null'}",
            f"progress_path=outputs/dashboard/jobs/{job_id}/progress.json",
        )
        return argv, {"run": run, "checkpoint_version": checkpoint}

    def _bundle(
        self,
        raw: dict[str, object],
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        required = {"bundle_config", "score_config", "status_path", "mode"}
        _exact_fields(raw, required)
        bundle = self._input_path(raw["bundle_config"], suffix=".json")
        score = self._input_path(raw["score_config"], suffix=".json")
        status = self._output_path(raw["status_path"], roots=("outputs",))
        mode = str(raw["mode"])
        if mode not in {"distributed", "local"}:
            raise ValueError("bundle evaluation mode must be distributed or local")
        argv = [
            sys.executable,
            str(self.repo_root / "src" / "tools" / "run_bundle_evaluation.py"),
            "--bundle-config",
            str(bundle),
            "--score-config",
            str(score),
            "--status-path",
            str(status),
        ]
        if mode == "local":
            argv.append("--local-only")
        return tuple(argv), {
            "bundle_config": str(bundle.relative_to(self.repo_root)),
            "score_config": str(score.relative_to(self.repo_root)),
            "status_path": str(status.relative_to(self.repo_root)),
            "mode": mode,
        }

    def _package(
        self,
        raw: dict[str, object],
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        _exact_fields(raw, {"profile", "output"})
        profile = _safe_profile(str(raw["profile"]))
        profile_path = self.repo_root / "configs" / "submission" / f"{profile}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"unknown submission profile: {profile}")
        output = self._output_path(raw["output"], roots=("dist", "outputs"))
        argv = (
            sys.executable,
            str(self.repo_root / "src" / "tools" / "submission" / "protocol.py"),
            "--profile",
            profile,
            "--output",
            str(output),
        )
        return argv, {
            "profile": profile,
            "output": str(output.relative_to(self.repo_root)),
        }

    def _dry_run(
        self,
        raw: dict[str, object],
        *,
        job_id: str,
    ) -> tuple[tuple[str, ...], dict[str, object]]:
        _exact_fields(raw, {"profile"})
        profile = _safe_config_profile(str(raw["profile"]))
        profile_path = self.repo_root / "configs" / f"{profile}.yaml"
        if not profile_path.is_file():
            raise FileNotFoundError(f"unknown training profile: {profile}")
        argv = (
            sys.executable,
            str(self.repo_root / "train.py"),
            "--dry-run",
            "--profile",
            profile,
            "--run-version",
            f"dashboard_dry_run_{job_id[:12]}",
        )
        return argv, {"profile": profile}

    def _input_path(self, value: object, *, suffix: str) -> Path:
        path = self._repo_path(value)
        if path.suffix != suffix or not path.is_file():
            raise FileNotFoundError(f"input must be an existing {suffix} file: {path}")
        return path

    def _output_path(self, value: object, *, roots: tuple[str, ...]) -> Path:
        path = self._repo_path(value)
        if not any(path.is_relative_to(self.repo_root / root) for root in roots):
            raise ValueError(f"output must be under one of: {', '.join(roots)}")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _repo_path(self, value: object) -> Path:
        raw = Path(str(value))
        path = (
            (self.repo_root / raw).resolve() if not raw.is_absolute() else raw.resolve()
        )
        if not path.is_relative_to(self.repo_root):
            raise ValueError("dashboard job paths must stay inside the repository")
        return path


def _exact_fields(raw: dict[str, object], expected: set[str]) -> None:
    actual = set(raw)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"invalid job fields; missing={missing}, extra={extra}")


def _safe_profile(value: str) -> str:
    cleaned = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", cleaned) is None:
        raise ValueError("submission profile must be one safe filename stem")
    return cleaned


def _safe_config_profile(value: str) -> str:
    cleaned = value.strip().strip("/")
    path = Path(cleaned)
    if (
        not cleaned
        or path.is_absolute()
        or ".." in path.parts
        or any(re.fullmatch(r"[A-Za-z0-9_.-]+", part) is None for part in path.parts)
    ):
        raise ValueError("training profile must be a safe config-relative path")
    return cleaned


def _safe_run_id(value: str) -> str:
    cleaned = value.strip()
    if re.fullmatch(r"[A-Za-z0-9_.-]+", cleaned) is None:
        raise ValueError("run must be one safe directory name")
    return cleaned
