"""Stream and verify the sanitized Git LFS submission archives."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import IO

import hydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict


class SubmissionRecord(BaseModel):
    """Required independent identities in the public release manifest."""

    model_config = ConfigDict(extra="ignore")

    path: Path
    sha256: str
    bytes: int
    submission_ref: str
    deck_hash: str
    checkpoint_sha256: str
    deck_sha256: str


class VerifyConfig(BaseModel):
    """Location of the versioned public artifact manifest."""

    model_config = ConfigDict(extra="forbid")
    manifest_path: Path


def _digest(stream: IO[bytes]) -> str:
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


def verify_submission(root: Path, record: SubmissionRecord) -> dict[str, str]:
    """Validate bytes, archive safety, excluded components, and asset identity."""
    candidate = root / record.path
    path = candidate.resolve()
    if not path.is_relative_to(root.resolve()) or candidate.is_symlink():
        raise ValueError("Submission path must stay inside the repository")
    with path.open("rb") as stream:
        if stream.read(100).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("Found an LFS pointer: run git lfs pull first")
        stream.seek(0)
        if _digest(stream) != record.sha256 or path.stat().st_size != record.bytes:
            raise ValueError("Sanitized submission checksum or size mismatch")
    expected = {
        "agent_checkpoint.pt": record.checkpoint_sha256,
        "deck.csv": record.deck_sha256,
    }
    found: set[str] = set()
    manifest_seen = False
    with tarfile.open(path, "r|gz") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                name.is_absolute()
                or ".." in name.parts
                or member.issym()
                or member.islnk()
            ):
                raise ValueError("Unsafe submission archive member")
            if member.name in found:
                raise ValueError("Duplicate submission archive member")
            found.add(member.name)
            if (
                "writeup" in name.parts
                or name.suffix in {".so", ".dll", ".dylib"}
                or name.parts[0] == "cg"
                or member.name.startswith("ptcg_rl/opponents/public_agents/")
            ):
                raise ValueError("Excluded component found in sanitized submission")
            if member.name in expected:
                member_stream = archive.extractfile(member)
                if member_stream is None or _digest(member_stream) != expected[member.name]:
                    raise ValueError("Original checkpoint or deck bytes changed")
            elif member.name == "public_release_manifest.json":
                manifest_stream = archive.extractfile(member)
                if manifest_stream is None:
                    raise ValueError("Missing internal public release manifest")
                manifest = json.load(manifest_stream)
                if (
                    manifest["deck_hash"] != record.deck_hash
                    or manifest["submission_ref"] != record.submission_ref
                ):
                    raise ValueError("Submission traceability does not match manifest")
                manifest_seen = True
    if not (expected.keys() | {"main.py", "deployment_runtime.json"}) <= found or not manifest_seen:
        raise ValueError("Submission is missing required assets or provenance")
    return {
        "submission_ref": record.submission_ref,
        "deck_hash": record.deck_hash,
        "status": "verified",
    }


@hydra.main(
    version_base=None,
    config_path="../../../configs",
    config_name="public_assets/verify",
)
def main(config: DictConfig) -> None:
    """Verify every LFS archive named by the public manifest."""
    parsed = VerifyConfig.model_validate(OmegaConf.to_container(config, resolve=True))
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / parsed.manifest_path).read_text())
    if manifest["format"] != "ptcg_public_submissions_v1":
        raise ValueError("Unsupported public submission manifest")
    results = [
        verify_submission(root, SubmissionRecord.model_validate(row))
        for row in manifest["submissions"]
    ]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
