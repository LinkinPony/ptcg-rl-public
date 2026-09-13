"""Build immutable fixed-deck deployment bundles for the stateless runtime."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

import torch
from pydantic import BaseModel, ConfigDict, field_validator

from ptcg_rl.agent.runtime import ActTimeConfig
from ptcg_rl.belief.public_catalog import load_public_deck_catalog
from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.data.kaggle_deck.records import read_deck
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.engine.constants import OptionType
from ptcg_rl.evaluation.bundle_models import BundleAgentConfig
from ptcg_rl.evaluation.search_identity import (
    file_sha256,
    fingerprint_payload,
    write_identity_atomic,
)
from ptcg_rl.model.simple_stateless import SimpleStatelessModelConfig
from ptcg_rl.model.simple_stateless.network import SimpleStatelessPolicyValueNet
from ptcg_rl.rl.policy_inputs import (
    SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT,
    SimpleStatelessActorRow,
    SimpleStatelessPublicInputAdapter,
    simple_stateless_input_contract,
)
from ptcg_rl.rl.stateless_checkpoint import (
    StatelessPolicyIdentity,
    load_stateless_policy_checkpoint,
)
from ptcg_rl.rl.stateless_export import (
    FixedDeckParityReport,
    export_fixed_deck_checkpoint,
    load_fixed_deck_checkpoint,
    validate_fixed_deck_parity,
    write_fixed_deck_export_report,
)
from ptcg_rl.submission.release_assets.manifest import runtime_config_sha256
from ptcg_rl.submission.release_assets.models import ReleaseBundleManifest
from ptcg_rl.submission.release_assets.native_library import (
    VerifiedNativeLibrary,
    load_verified_native_library,
)

_PAIR_FORMAT = "exact_policy_learner_pair_v1"
_HEX_DIGITS = frozenset("0123456789abcdef")
_DEPLOYMENT_NATIVE_LIBRARY_PATH = Path("src/native/cg_probe/libcg_probe.so")
_DEPLOYMENT_NATIVE_MANIFEST_PATH = Path("src/native/cg_probe/libcg_probe.kaggle.json")
_RUNTIME_NATIVE_TARGET = PurePosixPath("src/native/cg_probe/libcg_probe.so")
_RUNTIME_PUBLIC_CATALOG_MANIFEST_TARGET = PurePosixPath(
    "public_catalog/manifest.json"
)
_RUNTIME_DEPLOYMENT_CONFIG_TARGET = PurePosixPath("deployment_runtime.json")


class StatelessReleaseBundleConfig(BaseModel):
    """Frozen inputs for one fixed-deck stateless deployment bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_id: str
    checkpoint_tag: str
    checkpoint_file_stem: str
    pair_manifest_path: Path
    deck_path: Path
    public_catalog_manifest_path: Path
    runtime_template_archive_path: Path
    runtime_template_sha256: str
    output_dir: Path
    policy_temperature: float = 0.0

    @field_validator("bundle_id", "checkpoint_tag", "checkpoint_file_stem")
    @classmethod
    def immutable_identity(cls, value: str) -> str:
        """Reject blank or moving deployment labels."""
        normalized = value.strip()
        if not normalized or "latest" in normalized.lower():
            raise ValueError("deployment identity must be non-empty and immutable")
        return normalized

    @field_validator(
        "pair_manifest_path",
        "deck_path",
        "public_catalog_manifest_path",
        "runtime_template_archive_path",
        "output_dir",
    )
    @classmethod
    def immutable_path(cls, value: Path) -> Path:
        """Reject moving aliases in deployment inputs and outputs."""
        if any("latest" in part.lower() for part in value.parts):
            raise ValueError("deployment bundle paths cannot contain 'latest'")
        return value

    @field_validator("runtime_template_sha256")
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require an explicit digest for the approved runtime template."""
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in _HEX_DIGITS for character in normalized
        ):
            raise ValueError("runtime template digest must be 64 hex characters")
        return normalized

    @field_validator("policy_temperature")
    @classmethod
    def valid_policy_temperature(cls, value: float) -> float:
        """Require a finite deployment temperature; zero preserves greedy."""
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("policy_temperature must be finite and non-negative")
        return value


def prepare_stateless_release_bundle(
    config: StatelessReleaseBundleConfig,
) -> dict[str, Any]:
    """Export, package, and atomically freeze one fixed-deck deployment."""
    output_dir = records.repo_path(config.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable release asset already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = output_dir.parent / f".{output_dir.name}.{time.time_ns()}.tmp"
    temporary_dir.mkdir()
    try:
        summary = _prepare_in_directory(
            config,
            temporary_dir=temporary_dir,
            final_dir=output_dir,
        )
        temporary_dir.replace(output_dir)
        return summary
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _prepare_in_directory(
    config: StatelessReleaseBundleConfig,
    *,
    temporary_dir: Path,
    final_dir: Path,
) -> dict[str, Any]:
    pair_path = _required_file(config.pair_manifest_path, "checkpoint pair manifest")
    deck_path = _required_file(config.deck_path, "deployment deck")
    catalog_path = _required_file(
        config.public_catalog_manifest_path,
        "public catalog manifest",
    )
    template_path = _required_file(
        config.runtime_template_archive_path,
        "approved runtime archive",
    )
    version, identity, policy_path = _pair_identity(pair_path)
    loaded = load_stateless_policy_checkpoint(policy_path, expected_identity=identity)
    sequence_config = loaded.model_config_value.sequence
    native_library = (
        load_verified_native_library(
            _required_file(
                _DEPLOYMENT_NATIVE_LIBRARY_PATH,
                "tracked Kaggle native library",
            ),
            _required_file(
                _DEPLOYMENT_NATIVE_MANIFEST_PATH,
                "tracked Kaggle native build manifest",
            ),
        )
        if sequence_config is not None and sequence_config.engine_facts.enabled
        else None
    )
    deck = canonicalize_deck(read_deck(deck_path))
    if deck.deck_digest not in identity.active_exact_deck_digests:
        raise ValueError("deployment deck is not an active exact route of the policy")
    deck_sha256 = file_sha256(deck_path)

    checkpoint_relative = Path("checkpoint") / f"{config.checkpoint_file_stem}.pt"
    report_relative = Path("checkpoint") / "export_report.json"
    parity_relative = Path("checkpoint") / "parity_report.json"
    archive_relative = Path("runtime") / f"{config.checkpoint_tag}_runtime.tar.gz"
    report = export_fixed_deck_checkpoint(
        temporary_dir / checkpoint_relative,
        source_state=loaded.model_state,
        source_config=loaded.model_config_value,
        source_policy_sha256=loaded.artifact.policy_sha256,
        source_model_fingerprint=loaded.artifact.policy_model_fingerprint,
        target_deck_digest=deck.deck_digest,
        public_deck_catalog_fingerprint=identity.public_deck_catalog_fingerprint,
        asset_manifest_fingerprints={
            "action_schema": identity.action_schema_fingerprint,
            "belief_target": identity.belief_target_semantics_fingerprint,
            "card_catalog_static": identity.card_catalog_fingerprint,
            "checkpoint_pair": file_sha256(pair_path),
            "input_contract": identity.input_contract_fingerprint,
            "public_context_schema": identity.public_context_fingerprint,
            "scripted_manifest": identity.scripted_manifest_fingerprint,
            "target_deck_file": deck_sha256,
        },
    )
    published = report.model_copy(
        update={"checkpoint_path": final_dir / checkpoint_relative}
    )
    write_fixed_deck_export_report(temporary_dir / report_relative, published)
    parity = _validate_default_export_parity(
        source_model_config=loaded.model_config_value,
        source_model_state=loaded.model_state,
        fixed_checkpoint_path=temporary_dir / checkpoint_relative,
        deck=deck.card_ids,
        catalog_manifest_path=catalog_path,
        identity=identity,
    )
    write_identity_atomic(
        temporary_dir / parity_relative,
        parity.model_dump(mode="json"),
    )

    deployment_config_path = temporary_dir / "deployment_runtime.json"
    write_identity_atomic(
        deployment_config_path,
        {
            "format": "ptcg_rl_deployment_runtime_v1",
            "policy_temperature": config.policy_temperature,
        },
    )
    runtime_sha256 = _copy_runtime_template(
        template_path,
        temporary_dir / archive_relative,
        expected_sha256=config.runtime_template_sha256,
        public_catalog_manifest_path=catalog_path,
        native_library_path=(None if native_library is None else native_library.path),
        deployment_config_path=deployment_config_path,
    )
    runtime_semantics = runtime_config_sha256(
        BundleAgentConfig(
            kind="simple_stateless_greedy",
            checkpoint_path=final_dir / checkpoint_relative,
            public_catalog_manifest_path=catalog_path,
            act_time=ActTimeConfig(policy_temperature=config.policy_temperature),
        ).model_dump(mode="json")
    )
    bundle_fingerprint = fingerprint_payload(
        {
            "bundle_id": config.bundle_id,
            "checkpoint_sha256": report.checkpoint_sha256,
            "deck_sha256": deck_sha256,
            "belief_sha256": None,
            "runtime_config_sha256": runtime_semantics,
            "runtime_archive_sha256": runtime_sha256,
        }
    )
    manifest = ReleaseBundleManifest(
        bundle_id=config.bundle_id,
        checkpoint_tag=config.checkpoint_tag,
        checkpoint_path=Path(records.display_path(final_dir / checkpoint_relative)),
        checkpoint_sha256=report.checkpoint_sha256,
        deck_path=Path(records.display_path(deck_path)),
        deck_sha256=deck_sha256,
        runtime_archive_path=Path(records.display_path(final_dir / archive_relative)),
        runtime_sha256=runtime_sha256,
        runtime_config_sha256=runtime_semantics,
        bundle_fingerprint=bundle_fingerprint,
    )
    write_identity_atomic(
        temporary_dir / "bundle_manifest.json",
        manifest.model_dump(mode="json"),
    )
    build_manifest: dict[str, Any] = {
        "protocol": "STATELESS-RELEASE-BUNDLE-BUILD-v1",
        "bundle_manifest_path": records.display_path(
            final_dir / "bundle_manifest.json"
        ),
        "bundle_fingerprint": bundle_fingerprint,
        "policy_version": version,
        "pair_manifest_path": records.display_path(pair_path),
        "pair_manifest_sha256": file_sha256(pair_path),
        "source_policy_path": records.display_path(policy_path),
        "source_policy_sha256": loaded.artifact.policy_sha256,
        "source_model_fingerprint": loaded.artifact.policy_model_fingerprint,
        "public_catalog_manifest_path": records.display_path(catalog_path),
        "runtime_template_archive_path": records.display_path(template_path),
        "runtime_archive_sha256": runtime_sha256,
        "policy_temperature": config.policy_temperature,
        "deployment_native_library": _native_build_record(native_library),
        "deck_signature": deck.signature,
        "deck_digest": deck.deck_digest,
        "export": published.model_dump(mode="json"),
        "parity": parity.model_dump(mode="json"),
    }
    write_identity_atomic(temporary_dir / "build_manifest.json", build_manifest)
    return build_manifest


def _validate_default_export_parity(
    *,
    source_model_config: SimpleStatelessModelConfig,
    source_model_state: dict[str, torch.Tensor],
    fixed_checkpoint_path: Path,
    deck: tuple[int, ...],
    catalog_manifest_path: Path,
    identity: StatelessPolicyIdentity,
) -> FixedDeckParityReport:
    """Fail release if the default fixed route changes policy outputs."""
    rows = _default_parity_rows(
        deck=deck,
        catalog_manifest_path=catalog_manifest_path,
        identity=identity,
    )
    source_model = SimpleStatelessPolicyValueNet(
        source_model_config,
        load_static_features=False,
        initialize=False,
    )
    source_model.half()
    source_model.load_state_dict(source_model_state, strict=True)
    fixed_model, _payload = load_fixed_deck_checkpoint(fixed_checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        source_model.float()
        fixed_model.float()
    try:
        return validate_fixed_deck_parity(
            source_model,
            fixed_model,
            rows,
            device=device,
            absolute_tolerance=1.0e-4,
        )
    finally:
        del source_model
        del fixed_model
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _default_parity_rows(
    *,
    deck: tuple[int, ...],
    catalog_manifest_path: Path,
    identity: StatelessPolicyIdentity,
) -> tuple[SimpleStatelessActorRow, ...]:
    """Build deterministic public-only rows from immutable release inputs."""
    catalog, _manifest = load_public_deck_catalog(catalog_manifest_path)
    contract = simple_stateless_input_contract(
        public_catalog_fingerprint=catalog.fingerprint,
        card_catalog_fingerprint=identity.card_catalog_fingerprint,
        public_context_fingerprint=identity.public_context_fingerprint,
        wrapper_runtime_fingerprint=(
            SIMPLE_STATELESS_WRAPPER_RUNTIME_FINGERPRINT
        ),
    )
    if contract.fingerprint != identity.input_contract_fingerprint:
        raise ValueError("release parity input contract differs from checkpoint")
    if contract.action_schema_fingerprint != identity.action_schema_fingerprint:
        raise ValueError("release parity action schema differs from checkpoint")
    rows = []
    for seat in (0, 1):
        adapter = SimpleStatelessPublicInputAdapter(
            catalog,
            contract=contract,
            player_index=seat,
            own_deck=deck,
        )
        rows.append(adapter.prepare(_parity_observation(deck[0], seat=seat)))
    return tuple(rows)


def _parity_observation(card_id: int, *, seat: int) -> dict[str, Any]:
    """Return a small public decision used only for export equivalence."""
    return {
        "current": {
            "turn": 3,
            "turnActionCount": 2,
            "yourIndex": seat,
            "firstPlayer": 0,
            "supporterPlayed": False,
            "stadiumPlayed": False,
            "energyAttached": False,
            "retreated": False,
            "result": -1,
            "stadium": [],
            "looking": [],
            "players": [
                _parity_player(card_id, player_index=0),
                _parity_player(card_id, player_index=1),
            ],
        },
        "logs": [],
        "search_begin_input": "",
        "select": {
            "type": 0,
            "context": 0,
            "minCount": 1,
            "maxCount": 1,
            "remainDamageCounter": 0,
            "remainEnergyCost": 0,
            "deck": [],
            "contextCard": None,
            "effect": None,
            "option": [
                {"type": int(OptionType.END)},
                {"type": int(OptionType.YES)},
            ],
        },
    }


def _parity_player(card_id: int, *, player_index: int) -> dict[str, Any]:
    """Return one deterministic public player state for parity tensorization."""
    return {
        "active": [
            {
                "id": card_id,
                "serial": 100 + player_index,
                "playerIndex": player_index,
                "hp": 80,
                "maxHp": 100,
                "appearThisTurn": False,
                "energies": [],
                "energyCards": [],
                "tools": [],
                "preEvolution": [],
            }
        ],
        "bench": [],
        "benchMax": 5,
        "deckCount": 40,
        "discard": [],
        "prize": [None] * 6,
        "handCount": 0,
        "hand": [],
        "poisoned": False,
        "burned": False,
        "asleep": False,
        "paralyzed": False,
        "confused": False,
    }


def _pair_identity(
    pair_path: Path,
) -> tuple[int, StatelessPolicyIdentity, Path]:
    """Read one immutable pair manifest and verify its policy bytes."""
    payload = json.loads(pair_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("format") != _PAIR_FORMAT:
        raise ValueError(f"unsupported checkpoint pair manifest: {pair_path}")
    identity = StatelessPolicyIdentity.model_validate(
        payload["metadata"]["simple_stateless_identity"]
    )
    policy_record = payload["policy"]
    policy_path = _required_file(Path(str(policy_record["path"])), "policy checkpoint")
    if policy_path.stat().st_size != int(policy_record["size_bytes"]):
        raise ValueError("policy checkpoint size differs from its pair manifest")
    if file_sha256(policy_path) != str(policy_record["sha256"]):
        raise ValueError("policy checkpoint bytes differ from its pair manifest")
    return int(payload["version"]), identity, policy_path


def _copy_runtime_template(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    public_catalog_manifest_path: Path,
    native_library_path: Path | None = None,
    deployment_config_path: Path | None = None,
) -> str:
    """Copy a runtime while binding its catalog and optional native library."""
    actual = file_sha256(source)
    if actual != expected_sha256:
        raise ValueError("approved runtime archive digest changed")
    catalog_manifest_path = _required_file(
        public_catalog_manifest_path,
        "public catalog manifest",
    )
    catalog_artifact_path, catalog_artifact_target = _catalog_artifact(
        catalog_manifest_path
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    replacements = {
        _RUNTIME_PUBLIC_CATALOG_MANIFEST_TARGET: catalog_manifest_path,
        catalog_artifact_target: catalog_artifact_path,
    }
    if native_library_path is not None:
        replacements[_RUNTIME_NATIVE_TARGET] = native_library_path
    if deployment_config_path is not None:
        replacements[_RUNTIME_DEPLOYMENT_CONFIG_TARGET] = _required_file(
            deployment_config_path,
            "deployment runtime config",
        )
    _write_runtime_with_replacements(
        source,
        destination,
        replacements=replacements,
    )
    _verify_runtime_file(
        destination,
        target=_RUNTIME_PUBLIC_CATALOG_MANIFEST_TARGET,
        expected_sha256=file_sha256(catalog_manifest_path),
    )
    _verify_runtime_file(
        destination,
        target=catalog_artifact_target,
        expected_sha256=file_sha256(catalog_artifact_path),
    )
    if native_library_path is not None:
        _verify_runtime_file(
            destination,
            target=_RUNTIME_NATIVE_TARGET,
            expected_sha256=file_sha256(native_library_path),
        )
    if deployment_config_path is not None:
        _verify_runtime_file(
            destination,
            target=_RUNTIME_DEPLOYMENT_CONFIG_TARGET,
            expected_sha256=file_sha256(deployment_config_path),
        )
    return file_sha256(destination)


def _catalog_artifact(
    manifest_path: Path,
) -> tuple[Path, PurePosixPath]:
    """Resolve and verify the catalog artifact named by one manifest."""
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("public catalog manifest must contain an object")
    artifact_relative = PurePosixPath(str(payload.get("artifact_filename", "")))
    if (
        artifact_relative.is_absolute()
        or ".." in artifact_relative.parts
        or not artifact_relative.parts
        or str(artifact_relative) in {"", "."}
    ):
        raise ValueError("public catalog artifact filename is unsafe")
    artifact_path = _required_file(
        manifest_path.parent / Path(*artifact_relative.parts),
        "public catalog artifact",
    )
    expected_sha256 = str(payload.get("artifact_sha256", ""))
    if file_sha256(artifact_path) != expected_sha256:
        raise ValueError("public catalog artifact digest differs from its manifest")
    return (
        artifact_path,
        PurePosixPath("public_catalog") / artifact_relative,
    )


def _write_runtime_with_replacements(
    source: Path,
    destination: Path,
    *,
    replacements: dict[PurePosixPath, Path],
) -> None:
    """Write a deterministic runtime archive with canonical replacement files."""
    seen: set[PurePosixPath] = set()
    with (
        tarfile.open(source, "r:*") as source_archive,
        destination.open("wb") as raw_output,
        gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_output,
            mtime=0,
        ) as compressed_output,
        tarfile.open(
            fileobj=compressed_output,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as destination_archive,
    ):
        for member in source_archive.getmembers():
            member_path = _safe_runtime_member(member)
            if member_path in seen:
                raise ValueError(
                    f"runtime template contains a duplicate path: {member.name}"
                )
            seen.add(member_path)
            if member_path in replacements:
                if not member.isfile():
                    raise ValueError(
                        "runtime replacement target is not a regular file: "
                        f"{member.name}"
                    )
                continue
            if member.isdir():
                destination_archive.addfile(member)
                continue
            file_obj = source_archive.extractfile(member)
            if file_obj is None:
                raise ValueError(f"cannot read runtime member: {member.name}")
            with file_obj:
                destination_archive.addfile(member, fileobj=file_obj)

        for target, replacement_path in sorted(
            replacements.items(),
            key=lambda item: item[0].as_posix(),
        ):
            replacement_info = tarfile.TarInfo(target.as_posix())
            replacement_info.size = replacement_path.stat().st_size
            replacement_info.mode = 0o755 if target == _RUNTIME_NATIVE_TARGET else 0o644
            replacement_info.mtime = 0
            replacement_info.uid = 0
            replacement_info.gid = 0
            replacement_info.uname = ""
            replacement_info.gname = ""
            with replacement_path.open("rb") as replacement_stream:
                destination_archive.addfile(
                    replacement_info,
                    fileobj=replacement_stream,
                )


def _safe_runtime_member(member: tarfile.TarInfo) -> PurePosixPath:
    """Validate one runtime archive member before copying it."""
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"runtime template contains an unsafe path: {member.name}")
    if not member.isfile() and not member.isdir():
        raise ValueError(f"runtime template contains a non-file: {member.name}")
    return PurePosixPath(*tuple(part for part in path.parts if part not in ("", ".")))


def _verify_runtime_file(
    path: Path,
    *,
    target: PurePosixPath,
    expected_sha256: str,
) -> None:
    """Require exactly one runtime member with the approved bytes."""
    matches = 0
    with tarfile.open(path, "r:*") as archive:
        for member in archive.getmembers():
            if _safe_runtime_member(member) != target:
                continue
            matches += 1
            file_obj = archive.extractfile(member)
            if file_obj is None:
                raise ValueError(f"cannot read injected runtime file: {target}")
            digest = hashlib.sha256()
            with file_obj:
                while chunk := file_obj.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError(f"runtime file differs from tracked bytes: {target}")
    if matches != 1:
        raise ValueError(f"runtime archive must contain exactly one {target}")


def _native_build_record(
    native_library: VerifiedNativeLibrary | None,
) -> dict[str, Any] | None:
    """Return the immutable native build identities recorded by this release."""
    if native_library is None:
        return None
    return {
        "library_path": records.display_path(native_library.path),
        "library_sha256": native_library.sha256,
        "manifest_path": records.display_path(native_library.manifest_path),
        "manifest_sha256": native_library.manifest_sha256,
        "kernel_ref": native_library.manifest.kernel_ref,
        "source_git_commit": native_library.manifest.source_git_commit,
        "test_agent_archive_sha256": (
            native_library.manifest.test_agent_archive_sha256
        ),
    }


def _required_file(path: Path, label: str) -> Path:
    resolved = records.repo_path(path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    return resolved


__all__ = [
    "StatelessReleaseBundleConfig",
    "prepare_stateless_release_bundle",
]
