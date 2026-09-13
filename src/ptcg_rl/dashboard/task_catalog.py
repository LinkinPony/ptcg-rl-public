"""Repository-backed, opaque selection catalog for dashboard tasks."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.dashboard.task_models import (
    TaskCatalog,
    TaskCatalogItem,
    TaskReceipt,
    TaskWorkflow,
)
from ptcg_rl.dashboard.task_resources import TaskResourceGate
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.evaluation.search_identity import file_sha256, fingerprint_payload
from ptcg_rl.opponents.spec import opponent_registry

_CACHE_SECONDS = 30.0


@dataclass(frozen=True)
class CatalogSelection:
    """Server-side resolution hidden behind one opaque browser identifier."""

    item: TaskCatalogItem
    path: Path | None
    value: str | None = None


class TaskCatalogService:
    """Discover safe immutable inputs without accepting browser paths."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()
        self.resource_gate = TaskResourceGate(self.repo_root)
        self._updated_at = 0.0
        self._selections: dict[str, CatalogSelection] = {}
        self._groups: dict[str, tuple[TaskCatalogItem, ...]] = {}

    def catalog(self, receipts: tuple[TaskReceipt, ...]) -> TaskCatalog:
        """Return current choices and resource state."""
        self._refresh()
        return TaskCatalog(
            workflows=_WORKFLOWS,
            releases=self._groups["releases"],
            checkpoints=self._groups["checkpoints"],
            decks=self._groups["decks"],
            registered_opponents=self._groups["registered_opponents"],
            public_catalogs=self._groups["public_catalogs"],
            side_observations=self._groups["side_observations"],
            evaluation_profiles=self._groups["evaluation_profiles"],
            submission_profiles=self._groups["submission_profiles"],
            training_profiles=self._groups["training_profiles"],
            runtime_templates=self._groups["runtime_templates"],
            resources=self.resource_gate.snapshot(receipts),
        )

    def resolve(self, artifact_id: str, *, kind: str) -> CatalogSelection:
        """Resolve one exact item and reject stale or cross-kind identifiers."""
        self._refresh()
        selection = self._selections.get(artifact_id)
        if selection is None or selection.item.kind != kind:
            raise KeyError(f"unknown {kind} artifact")
        if not selection.item.available:
            raise ValueError(
                selection.item.unavailable_reason or f"{kind} artifact is unavailable"
            )
        return selection

    def _refresh(self) -> None:
        now = time.monotonic()
        if self._groups and now - self._updated_at < _CACHE_SECONDS:
            return
        selections: dict[str, CatalogSelection] = {}
        groups = {
            "releases": self._releases(selections),
            "checkpoints": self._checkpoints(selections),
            "decks": self._decks(selections),
            "registered_opponents": self._registered_opponents(selections),
            "public_catalogs": self._manifest_files(
                selections,
                root=self.repo_root / "outputs" / "belief" / "public_catalog",
                pattern="**/manifest.json",
                kind="public_catalog",
            ),
            "side_observations": self._side_observations(selections),
            "evaluation_profiles": self._profiles(
                selections,
                root=self.repo_root / "configs" / "evaluation",
                kind="evaluation_profile",
            ),
            "submission_profiles": self._profiles(
                selections,
                root=self.repo_root / "configs" / "submission",
                kind="submission_profile",
            ),
            "training_profiles": self._profiles(
                selections,
                root=self.repo_root / "configs" / "rl" / "train",
                kind="training_profile",
            ),
            "runtime_templates": self._runtime_templates(selections),
        }
        self._selections = selections
        self._groups = groups
        self._updated_at = now

    def _releases(
        self, selections: dict[str, CatalogSelection]
    ) -> tuple[TaskCatalogItem, ...]:
        output: dict[str, TaskCatalogItem] = {}
        root = self.repo_root / "outputs" / "submission"
        if not root.is_dir():
            return ()
        for path in sorted(root.glob("*/release_asset/bundle_manifest.json")):
            if _has_latest(path):
                continue
            try:
                payload = _read_object(path)
                fingerprint = str(payload["bundle_fingerprint"])
                bundle_id = str(payload["bundle_id"])
                deck_path = _repo_file(self.repo_root, payload["deck_path"])
                archive_path = _repo_file(
                    self.repo_root, payload["runtime_archive_path"]
                )
                available = deck_path.is_file() and archive_path.is_file()
                artifact_id = f"release:{fingerprint}"
                item = TaskCatalogItem(
                    artifact_id=artifact_id,
                    kind="release_bundle",
                    label=str(payload.get("checkpoint_tag") or path.parents[2].name),
                    detail=bundle_id,
                    fingerprint=fingerprint,
                    available=available,
                    unavailable_reason=(
                        None
                        if available
                        else "release manifest references a missing deck or archive"
                    ),
                    metadata={
                        "protocol": str(payload.get("protocol", "")),
                        "deck_sha256": str(payload.get("deck_sha256", "")),
                        "checkpoint_sha256": str(payload.get("checkpoint_sha256", "")),
                        "bundle_id": bundle_id,
                    },
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
            output.setdefault(artifact_id, item)
            selections.setdefault(
                artifact_id,
                CatalogSelection(item=item, path=path.resolve()),
            )
        return tuple(sorted(output.values(), key=lambda item: item.label, reverse=True))

    def _checkpoints(
        self, selections: dict[str, CatalogSelection]
    ) -> tuple[TaskCatalogItem, ...]:
        output: list[TaskCatalogItem] = []
        root = self.repo_root / "outputs" / "training"
        if not root.is_dir():
            return ()
        for path in sorted(root.glob("**/weights/checkpoint_pair_v*.json")):
            if _has_latest(path):
                continue
            try:
                payload = _read_object(path)
                version = int(payload["version"])
                pair_sha = file_sha256(path)
                policy_raw = payload.get("policy")
                if not isinstance(policy_raw, dict):
                    continue
                policy_path = path.parent / f"policy_v{version}.pt"
                if not policy_path.is_file():
                    configured = Path(str(policy_raw.get("path", "")))
                    policy_path = (
                        configured
                        if configured.is_absolute()
                        else self.repo_root / configured
                    )
                policy_path = policy_path.resolve()
                if not policy_path.is_relative_to(self.repo_root):
                    continue
                run_id = path.parent.parent.name
                identity = _simple_stateless_identity(payload)
                deck_digests = identity.get("active_exact_deck_digests", ())
                if not isinstance(deck_digests, (list, tuple)):
                    deck_digests = ()
                artifact_id = f"checkpoint:{pair_sha}"
                available = policy_path.is_file()
                item = TaskCatalogItem(
                    artifact_id=artifact_id,
                    kind="checkpoint",
                    label=f"{run_id} · v{version}",
                    detail=policy_path.name,
                    fingerprint=str(policy_raw.get("sha256") or pair_sha),
                    available=available,
                    unavailable_reason=None if available else "policy file is missing",
                    metadata={
                        "run_id": run_id,
                        "version": version,
                        "pair_manifest_sha256": pair_sha,
                        "active_exact_deck_digests": list(deck_digests),
                        "model_config_fingerprint": identity.get(
                            "model_config_fingerprint"
                        ),
                    },
                )
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                continue
            selections[artifact_id] = CatalogSelection(item=item, path=policy_path)
            output.append(item)
        output.sort(
            key=lambda item: (
                str(item.metadata.get("run_id", "")),
                int(item.metadata.get("version", -1)),
            ),
            reverse=True,
        )
        return tuple(output)

    def _decks(
        self, selections: dict[str, CatalogSelection]
    ) -> tuple[TaskCatalogItem, ...]:
        paths = list((self.repo_root / "docs" / "experiments").glob("**/decks/*.csv"))
        for spec in opponent_registry().values():
            if spec.deck_path is not None:
                try:
                    paths.append(_repo_file(self.repo_root, spec.deck_path))
                except ValueError:
                    continue
        output: dict[str, TaskCatalogItem] = {}
        for path in sorted(set(paths)):
            if not path.is_file() or _has_latest(path):
                continue
            try:
                deck = canonicalize_deck(records.read_deck(path))
            except (OSError, ValueError):
                continue
            artifact_id = f"deck:{deck.deck_digest}"
            item = TaskCatalogItem(
                artifact_id=artifact_id,
                kind="deck",
                label=path.stem,
                detail=_display(self.repo_root, path),
                fingerprint=deck.deck_digest,
                metadata={
                    "deck_digest": deck.deck_digest,
                    "signature": deck.signature,
                    "deck_hash": records.signature_hash(deck.signature),
                },
            )
            prior = output.get(artifact_id)
            if prior is None or len(item.detail or "") < len(prior.detail or ""):
                output[artifact_id] = item
                selections[artifact_id] = CatalogSelection(
                    item=item, path=path.resolve()
                )
        return tuple(sorted(output.values(), key=lambda item: item.label))

    def _registered_opponents(
        self, selections: dict[str, CatalogSelection]
    ) -> tuple[TaskCatalogItem, ...]:
        output: list[TaskCatalogItem] = []
        for name, spec in sorted(opponent_registry().items()):
            if spec.deck_path is None:
                continue
            try:
                path = _repo_file(self.repo_root, spec.deck_path)
            except ValueError:
                continue
            available = path.is_file()
            artifact_id = f"registered:{name}"
            item = TaskCatalogItem(
                artifact_id=artifact_id,
                kind="registered_opponent",
                label=name,
                detail=f"Tier {spec.tier} · {spec.source}",
                available=available,
                unavailable_reason=None if available else "fixed deck file is missing",
                metadata={
                    "name": name,
                    "tier": spec.tier,
                    "source": spec.source,
                },
            )
            selections[artifact_id] = CatalogSelection(
                item=item,
                path=path.resolve(),
                value=name,
            )
            output.append(item)
        return tuple(output)

    def _manifest_files(
        self,
        selections: dict[str, CatalogSelection],
        *,
        root: Path,
        pattern: str,
        kind: str,
    ) -> tuple[TaskCatalogItem, ...]:
        output: list[TaskCatalogItem] = []
        if not root.is_dir():
            return ()
        for path in sorted(root.glob(pattern)):
            if _has_latest(path):
                continue
            try:
                fingerprint = file_sha256(path)
                _read_object(path)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            artifact_id = f"{kind}:{fingerprint}"
            item = TaskCatalogItem(
                artifact_id=artifact_id,
                kind=kind,  # type: ignore[arg-type]
                label=path.parent.name,
                detail=_display(self.repo_root, path),
                fingerprint=fingerprint,
            )
            selections[artifact_id] = CatalogSelection(item=item, path=path.resolve())
            output.append(item)
        return tuple(output)

    def _side_observations(
        self, selections: dict[str, CatalogSelection]
    ) -> tuple[TaskCatalogItem, ...]:
        output: list[TaskCatalogItem] = []
        for path in sorted(
            (self.repo_root / "outputs").glob("**/*side_observations*.parquet")
        ):
            if _has_latest(path):
                continue
            relative = _display(self.repo_root, path)
            artifact_id = f"side-observations:{_opaque(relative)}"
            item = TaskCatalogItem(
                artifact_id=artifact_id,
                kind="side_observations",
                label=path.parent.name,
                detail=relative,
                metadata={"size_bytes": path.stat().st_size},
            )
            selections[artifact_id] = CatalogSelection(item=item, path=path.resolve())
            output.append(item)
        output.sort(key=lambda item: item.detail or "", reverse=True)
        return tuple(output)

    def _profiles(
        self,
        selections: dict[str, CatalogSelection],
        *,
        root: Path,
        kind: str,
    ) -> tuple[TaskCatalogItem, ...]:
        output: list[TaskCatalogItem] = []
        if not root.is_dir():
            return ()
        for path in sorted(root.glob("**/*.yaml")):
            if _has_latest(path):
                continue
            relative = path.relative_to(root).with_suffix("").as_posix()
            selected_value = (
                f"rl/train/{relative}" if kind == "training_profile" else relative
            )
            artifact_id = f"{kind}:{_opaque(relative)}"
            item = TaskCatalogItem(
                artifact_id=artifact_id,
                kind=kind,  # type: ignore[arg-type]
                label=relative,
                detail=_display(self.repo_root, path),
                fingerprint=file_sha256(path),
            )
            selections[artifact_id] = CatalogSelection(
                item=item,
                path=path.resolve(),
                value=selected_value,
            )
            output.append(item)
        return tuple(output)

    def _runtime_templates(
        self, selections: dict[str, CatalogSelection]
    ) -> tuple[TaskCatalogItem, ...]:
        template_id = "acttime-default-v1"
        payload = ActTimeConfig().model_dump(mode="json")
        item = TaskCatalogItem(
            artifact_id=f"runtime-template:{template_id}",
            kind="runtime_template",
            label="部署默认 ActTime",
            detail="固定 runtime 模板；任务 receipt 会记录完整指纹",
            fingerprint=fingerprint_payload(payload),
            metadata={"template_id": template_id},
        )
        selections[item.artifact_id] = CatalogSelection(
            item=item,
            path=None,
            value=json.dumps(payload, sort_keys=True),
        )
        return (item,)


_WORKFLOWS = (
    TaskWorkflow(
        kind="bundle_strength",
        label="Bundle 强度评测",
        description="对固定 candidate × opponent 支撑运行平衡对局并进行后验排名。",
        modes=("formal", "diagnostic"),
        result_views=(
            "summary",
            "raw_bundle_standings",
            "raw_pilot_standings",
            "raw_deck_standings",
            "bundle_standings",
            "cells",
            "report",
        ),
    ),
    TaskWorkflow(
        kind="release_h2h",
        label="Release 直接对比",
        description="两个不可变部署 bundle 的原生 runtime H2H。",
        modes=("formal",),
        result_views=("summary", "games", "score", "log"),
    ),
    TaskWorkflow(
        kind="runtime_elo",
        label="Checkpoint 牌组梯度诊断",
        description="同一裸 checkpoint 在多个精确牌组上的 runtime Elo 诊断。",
        modes=("diagnostic",),
        result_views=("summary", "standings", "matchups", "report"),
    ),
    TaskWorkflow(
        kind="package_validation",
        label="Submission 包校验",
        description="构建并验证本地包，永不上传 Kaggle。",
        modes=("utility",),
        result_views=("summary", "archive", "log"),
    ),
    TaskWorkflow(
        kind="config_dry_run",
        label="训练配置 Dry-run",
        description="解析训练 profile 并打印命令，不启动训练。",
        modes=("utility",),
        result_views=("summary", "log"),
    ),
)


def _simple_stateless_identity(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        return {}
    identity = metadata.get("simple_stateless_identity")
    return identity if isinstance(identity, dict) else {}


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _repo_file(repo_root: Path, value: object) -> Path:
    path = Path(str(value))
    resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
    if not resolved.is_relative_to(repo_root):
        raise ValueError("catalog artifact escapes repository")
    return resolved


def _display(repo_root: Path, path: Path) -> str:
    return path.resolve().relative_to(repo_root).as_posix()


def _has_latest(path: Path) -> bool:
    return any("latest" in part.lower() for part in path.parts)


def _opaque(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
