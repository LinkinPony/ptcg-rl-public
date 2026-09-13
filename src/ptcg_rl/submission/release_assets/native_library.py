"""Verified Kaggle-built native library assets for deployment releases."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ptcg_rl.engine.feature_vectors import DYNAMIC_EFFECT_FEATURE_SIZE
from ptcg_rl.engine.native_probe import NativeProbeBackend
from ptcg_rl.evaluation.search_identity import file_sha256

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class KaggleNativeAbi(BaseModel):
    """Native fact ABI tested by the Kaggle build."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_version: Literal[1]
    fact_width: int = Field(gt=0)


class KaggleNativeElfVersions(BaseModel):
    """Dynamic symbol versions required by the built ELF."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    glibc: tuple[str, ...]
    glibcxx: tuple[str, ...]
    cxxabi: tuple[str, ...]


class KaggleInferenceRuntimeStatus(BaseModel):
    """Fail-closed status from one real packaged-agent inference."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    belief_error: None
    engine_prewarm_error: None
    policy_error: None
    policy_loaded: Literal[True]
    prewarm_error: None
    search_error: None
    simple_stateless_runtime: Literal[True]
    used_random_fallback: Literal[False]


class KaggleNativeInferenceTest(BaseModel):
    """End-to-end deployment inference evidence from the Kaggle runtime."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    passed: Literal[True]
    startup_action_count: Literal[60]
    startup_seconds: float = Field(ge=0.0)
    inference_action: tuple[int, ...]
    inference_seconds: float = Field(ge=0.0)
    option_count: int = Field(gt=1)
    search_begin_input_bytes: int = Field(gt=0)
    dynamic_effect_feature_size: int = Field(gt=0)
    runtime: KaggleInferenceRuntimeStatus

    @field_validator("inference_action")
    @classmethod
    def nonempty_action(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        """Require the test to have served an actual selection."""
        if not value:
            raise ValueError("Kaggle inference action cannot be empty")
        return value


class KaggleNativeBuildManifest(BaseModel):
    """Immutable provenance and inference gate for one deployment binary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol: Literal["PTCG-RL-KAGGLE-NATIVE-BUILD-v1"]
    kernel_ref: str
    built_at_utc: datetime
    source_git_commit: str
    source_archive_sha256: str
    official_competition_source: Literal["pokemon-tcg-ai-battle"]
    compiler_version: str
    glibc_version: str
    binary_name: Literal["libcg_probe.so"]
    binary_sha256: str
    binary_size_bytes: int = Field(gt=0)
    elf_required_versions: KaggleNativeElfVersions
    ldd: tuple[str, ...]
    abi: KaggleNativeAbi
    inference_test: KaggleNativeInferenceTest
    test_agent_archive_sha256: str
    test_observation_sha256: str

    @field_validator("kernel_ref", "compiler_version", "glibc_version")
    @classmethod
    def nonempty_provenance(cls, value: str) -> str:
        """Reject provenance records that cannot identify their environment."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Kaggle native build provenance cannot be blank")
        return normalized

    @field_validator("source_git_commit")
    @classmethod
    def valid_git_commit(cls, value: str) -> str:
        """Require the exact source commit used by the Notebook."""
        normalized = value.strip().lower()
        if _GIT_COMMIT_PATTERN.fullmatch(normalized) is None:
            raise ValueError("native build source commit must contain 40 hex digits")
        return normalized

    @field_validator(
        "source_archive_sha256",
        "binary_sha256",
        "test_agent_archive_sha256",
        "test_observation_sha256",
    )
    @classmethod
    def valid_sha256(cls, value: str) -> str:
        """Require full content identities for every build and test input."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("native build identities must be lowercase SHA-256")
        return normalized

    @model_validator(mode="after")
    def compatible_fact_schema(self) -> Self:
        """Bind Notebook evidence to the Python deployment feature schema."""
        if self.built_at_utc.tzinfo is None:
            raise ValueError("native build timestamp must include a timezone")
        if (
            self.abi.fact_width != DYNAMIC_EFFECT_FEATURE_SIZE
            or self.inference_test.dynamic_effect_feature_size
            != DYNAMIC_EFFECT_FEATURE_SIZE
        ):
            raise ValueError("Kaggle native fact width differs from Python schema")
        return self


@dataclass(frozen=True)
class VerifiedNativeLibrary:
    """Locally verified deployment binary and its Kaggle evidence."""

    path: Path
    sha256: str
    manifest_path: Path
    manifest_sha256: str
    manifest: KaggleNativeBuildManifest


def load_verified_native_library(
    library_path: Path,
    manifest_path: Path,
) -> VerifiedNativeLibrary:
    """Verify downloaded bytes, Kaggle inference evidence, and the local ABI."""
    if not library_path.is_file():
        raise FileNotFoundError(f"deployment native library is missing: {library_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"deployment native build manifest is missing: {manifest_path}"
        )
    manifest = KaggleNativeBuildManifest.model_validate(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    if library_path.name != manifest.binary_name:
        raise ValueError("deployment native library name differs from its manifest")
    if library_path.stat().st_size != manifest.binary_size_bytes:
        raise ValueError("deployment native library size differs from its manifest")
    library_sha256 = file_sha256(library_path)
    if library_sha256 != manifest.binary_sha256:
        raise ValueError("deployment native library SHA256 differs from its manifest")
    backend = NativeProbeBackend(library_path=library_path)
    if backend.engine_abi_fingerprint != library_sha256:
        raise ValueError("loaded native ABI bytes differ from the verified library")
    return VerifiedNativeLibrary(
        path=library_path,
        sha256=library_sha256,
        manifest_path=manifest_path,
        manifest_sha256=file_sha256(manifest_path),
        manifest=manifest,
    )


__all__ = [
    "KaggleNativeBuildManifest",
    "VerifiedNativeLibrary",
    "load_verified_native_library",
]
