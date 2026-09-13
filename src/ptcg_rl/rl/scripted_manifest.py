"""Versioned immutable scripted-opponent manifests for vector rollout."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

from ptcg_rl.decks.identity import parse_canonical_signature
from ptcg_rl.opponents.spec import BattleAgent
from ptcg_rl.rl.checkpoint_pair_io import atomic_write_bytes, json_payload

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SCRIPTED_ENTRY_DOMAIN = b"ptcg-rl/scripted-opponent-entry/v1\x00"
_SCRIPTED_MANIFEST_DOMAIN = b"ptcg-rl/scripted-opponent-manifest/v1\x00"
_SCRIPTED_CODE_DOMAIN = b"ptcg-rl/scripted-opponent-code/v1\x00"

ScriptedFactory = Callable[[int, Mapping[str, JsonValue]], BattleAgent]


class ScriptedOpponentArtifact(BaseModel):
    """One immutable fixed-code pilot and exact deck."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    opponent_id: str
    script_name: str
    implementation_code_fingerprint: str
    implementation_files: tuple[str, ...]
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    exact_deck_signature: str
    exact_deck_digest: str
    engine_runtime_fingerprint: str
    random_stream_semantics: str
    vector_safe: bool
    requires_search: bool

    @field_validator(
        "opponent_id",
        "script_name",
        "random_stream_semantics",
    )
    @classmethod
    def non_empty_text(cls, value: str) -> str:
        """Normalize identifiers and random-stream declarations."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("scripted opponent text fields must be non-empty")
        return normalized

    @field_validator(
        "implementation_code_fingerprint",
        "exact_deck_digest",
        "engine_runtime_fingerprint",
    )
    @classmethod
    def valid_fingerprint(cls, value: str) -> str:
        """Require content fingerprints rather than moving labels."""
        normalized = value.strip().lower()
        if _SHA256_PATTERN.fullmatch(normalized) is None:
            raise ValueError("scripted artifact fingerprints must be SHA-256")
        return normalized

    @field_validator("implementation_files")
    @classmethod
    def valid_implementation_files(
        cls,
        values: tuple[str, ...],
    ) -> tuple[str, ...]:
        """Require canonical, relative, duplicate-free source paths."""
        normalized = tuple(value.strip().replace("\\", "/") for value in values)
        if not normalized or any(
            not value or value.startswith("/") or ".." in Path(value).parts
            for value in normalized
        ):
            raise ValueError("implementation files must be safe relative paths")
        if len(set(normalized)) != len(normalized):
            raise ValueError("implementation files must be unique")
        if tuple(sorted(normalized)) != normalized:
            raise ValueError("implementation files must be sorted")
        return normalized

    @model_validator(mode="after")
    def coherent_deck_identity(self) -> ScriptedOpponentArtifact:
        """Validate the exact 60-card signature and digest pair."""
        deck = parse_canonical_signature(self.exact_deck_signature)
        if deck.deck_digest != self.exact_deck_digest:
            raise ValueError("scripted exact deck fingerprint mismatch")
        return self

    @property
    def fingerprint(self) -> str:
        """Return one immutable scripted bundle fingerprint."""
        return _fingerprint(
            _SCRIPTED_ENTRY_DOMAIN,
            self.model_dump(mode="json"),
        )


class ScriptedOpponentManifest(BaseModel):
    """Small immutable manifest for the independent scripted lane."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    manifest_id: str
    entries: tuple[ScriptedOpponentArtifact, ...]

    @field_validator("manifest_id")
    @classmethod
    def non_empty_manifest_id(cls, value: str) -> str:
        """Reject anonymous scripted manifests."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("scripted manifest ID must be non-empty")
        return normalized

    @field_validator("entries")
    @classmethod
    def unique_entries(
        cls,
        values: tuple[ScriptedOpponentArtifact, ...],
    ) -> tuple[ScriptedOpponentArtifact, ...]:
        """Require a stable opponent ID and bundle identity per entry."""
        if not values:
            raise ValueError("scripted manifest needs at least one entry")
        opponent_ids = tuple(value.opponent_id for value in values)
        fingerprints = tuple(value.fingerprint for value in values)
        if len(set(opponent_ids)) != len(opponent_ids):
            raise ValueError("scripted opponent IDs must be unique")
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("scripted bundle fingerprints must be unique")
        return values

    @property
    def fingerprint(self) -> str:
        """Return the manifest content identity persisted in checkpoints."""
        return _fingerprint(
            _SCRIPTED_MANIFEST_DOMAIN,
            self.model_dump(mode="json"),
        )


@dataclass(frozen=True)
class ScriptedImplementation:
    """Allowlisted factory plus source files used to verify one script."""

    script_name: str
    factory: ScriptedFactory
    implementation_files: tuple[str, ...]
    vector_safe: bool
    requires_search: bool


@dataclass(frozen=True)
class ResolvedScriptedOpponent:
    """Verified immutable scripted bundle ready for per-game construction."""

    artifact: ScriptedOpponentArtifact
    factory: ScriptedFactory

    def build(self, *, seed: int) -> BattleAgent:
        """Construct one game-local scripted agent with a derived seed."""
        return self.factory(seed, self.artifact.parameters)

    @property
    def deck(self) -> tuple[int, ...]:
        """Return the exact canonical engine deck for this bundle."""
        return parse_canonical_signature(self.artifact.exact_deck_signature).card_ids


def resolve_scripted_manifest(
    manifest: ScriptedOpponentManifest,
    *,
    implementations: Sequence[ScriptedImplementation],
    source_root: Path,
) -> tuple[ResolvedScriptedOpponent, ...]:
    """Verify code fingerprints and resolve only explicit allowlisted scripts."""
    by_name = {
        implementation.script_name: implementation for implementation in implementations
    }
    if len(by_name) != len(implementations):
        raise ValueError("scripted implementation names must be unique")
    resolved: list[ResolvedScriptedOpponent] = []
    for artifact in manifest.entries:
        try:
            implementation = by_name[artifact.script_name]
        except KeyError as error:
            raise KeyError(
                f"scripted implementation is not allowlisted: {artifact.script_name}"
            ) from error
        if implementation.implementation_files != artifact.implementation_files:
            raise ValueError("scripted implementation file inventory changed")
        if (
            implementation.vector_safe != artifact.vector_safe
            or implementation.requires_search != artifact.requires_search
        ):
            raise ValueError("scripted runtime capability declaration changed")
        actual = fingerprint_scripted_sources(
            source_root,
            implementation.implementation_files,
        )
        if actual != artifact.implementation_code_fingerprint:
            raise ValueError("scripted implementation code fingerprint changed")
        resolved.append(
            ResolvedScriptedOpponent(
                artifact=artifact,
                factory=implementation.factory,
            )
        )
    return tuple(resolved)


def builtin_scripted_implementations(
    script_names: Sequence[str],
) -> tuple[ScriptedImplementation, ...]:
    """Return allowlisted fixed-code factories and complete source inventories."""
    allowed = {
        "random": (
            "src/ptcg_rl/opponents/builtin.py",
            "src/ptcg_rl/opponents/spec.py",
        ),
        "end": (
            "src/ptcg_rl/opponents/builtin.py",
            "src/ptcg_rl/opponents/spec.py",
        ),
        "mixed25": (
            "src/ptcg_rl/opponents/builtin.py",
            "src/ptcg_rl/opponents/spec.py",
            "src/ptcg_rl/opponents/third_party.py",
        ),
        "mixed50": (
            "src/ptcg_rl/opponents/builtin.py",
            "src/ptcg_rl/opponents/spec.py",
            "src/ptcg_rl/opponents/third_party.py",
        ),
        "mixed75": (
            "src/ptcg_rl/opponents/builtin.py",
            "src/ptcg_rl/opponents/spec.py",
            "src/ptcg_rl/opponents/third_party.py",
        ),
    }
    implementations: list[ScriptedImplementation] = []
    for name in sorted(set(script_names)):
        files: tuple[str, ...]
        if name == "slowking_copy_v1":
            from ptcg_rl.opponents import slowking_copy

            files = slowking_copy.IMPLEMENTATION_FILES
            vector_safe = True
            requires_search = False
        elif name == "slowking_copy_v2":
            from ptcg_rl.opponents import slowking_copy_v2

            files = slowking_copy_v2.IMPLEMENTATION_FILES
            vector_safe = True
            requires_search = False
        elif name == "lopunny_dudunsparce_v1":
            from ptcg_rl.opponents import lopunny_dudunsparce

            files = lopunny_dudunsparce.IMPLEMENTATION_FILES
            vector_safe = False
            requires_search = True
        elif name in allowed:
            files = allowed[name]
            vector_safe = True
            requires_search = False
        else:
            from ptcg_rl.opponents import third_party

            if name not in third_party.PUBLIC_OPPONENT_FIXTURES:
                raise KeyError(f"fixed-code opponent is not allowlisted: {name}")
            files = third_party.public_implementation_files(name)
            vector_safe = False
            requires_search = third_party.public_requires_search(name)

        def factory(
            seed: int,
            parameters: Mapping[str, JsonValue],
            *,
            script_name: str = name,
        ) -> BattleAgent:
            if parameters:
                raise ValueError("fixed-code opponent takes no parameters")
            if script_name == "slowking_copy_v1":
                from ptcg_rl.opponents import slowking_copy

                return slowking_copy.build_slowking_copy_agent(script_name)
            if script_name == "slowking_copy_v2":
                from ptcg_rl.opponents import slowking_copy_v2

                return slowking_copy_v2.build_slowking_copy_agent(script_name)
            if script_name == "lopunny_dudunsparce_v1":
                from ptcg_rl.opponents import lopunny_dudunsparce

                return lopunny_dudunsparce.build_lopunny_dudunsparce_agent(
                    script_name
                )
            from ptcg_rl.opponents import build_opponent, opponent_registry

            return build_opponent(opponent_registry()[script_name], seed=seed)

        implementations.append(
            ScriptedImplementation(
                script_name=name,
                factory=factory,
                implementation_files=files,
                vector_safe=vector_safe,
                requires_search=requires_search,
            )
        )
    return tuple(implementations)


def fingerprint_scripted_sources(
    source_root: Path,
    relative_paths: Sequence[str],
) -> str:
    """Stream a canonical source inventory into one implementation digest."""
    root = source_root.resolve()
    digest = hashlib.sha256(_SCRIPTED_CODE_DOMAIN)
    normalized = tuple(path.strip().replace("\\", "/") for path in relative_paths)
    if not normalized or tuple(sorted(set(normalized))) != normalized:
        raise ValueError("scripted source paths must be sorted and unique")
    for relative in normalized:
        path = (root / relative).resolve()
        if root not in path.parents:
            raise ValueError("scripted source path escapes source root")
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded_name = relative.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(4, "big"))
        digest.update(encoded_name)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def save_scripted_manifest(
    path: Path,
    manifest: ScriptedOpponentManifest,
) -> None:
    """Atomically save a small manifest envelope with its semantic digest."""
    atomic_write_bytes(
        path,
        json_payload(
            {
                "manifest": manifest.model_dump(mode="json"),
                "fingerprint": manifest.fingerprint,
            }
        ),
        overwrite=True,
    )


def load_scripted_manifest(path: Path) -> ScriptedOpponentManifest:
    """Load and verify one immutable scripted manifest envelope."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scripted manifest envelope must be a mapping")
    manifest = ScriptedOpponentManifest.model_validate(payload.get("manifest"))
    if payload.get("fingerprint") != manifest.fingerprint:
        raise ValueError("scripted manifest fingerprint mismatch")
    return manifest


def _fingerprint(domain: bytes, payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(domain + encoded).hexdigest()


__all__ = [
    "ResolvedScriptedOpponent",
    "ScriptedFactory",
    "ScriptedImplementation",
    "ScriptedOpponentArtifact",
    "ScriptedOpponentManifest",
    "builtin_scripted_implementations",
    "fingerprint_scripted_sources",
    "load_scripted_manifest",
    "resolve_scripted_manifest",
    "save_scripted_manifest",
]
