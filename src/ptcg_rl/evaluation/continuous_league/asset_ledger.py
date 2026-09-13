"""Asset and bundle registration operations for the league ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from ptcg_rl.evaluation.continuous_league.database import LeagueDatabase
from ptcg_rl.evaluation.continuous_league.models import (
    CandidateKind,
    ContinuousLeagueConfig,
    validate_safe_id,
    validate_sha256,
)
from ptcg_rl.evaluation.continuous_league.rating import ComponentTrueSkill


class LeagueAssetLedgerMixin:
    """Controller, deck, alias, and activation writes shared by the ledger."""

    config: ContinuousLeagueConfig
    database: LeagueDatabase
    skill: ComponentTrueSkill

    def controller_asset_binding(
        self,
        controller_id: str,
    ) -> tuple[Path | None, str] | None:
        """Return the immutable path/fingerprint currently bound to a controller."""
        identity = validate_safe_id(controller_id)
        with self.database.read() as connection:
            row = connection.execute(
                "SELECT asset_path, asset_sha256 FROM controllers WHERE controller_id = ?",
                (identity,),
            ).fetchone()
        if row is None:
            return None
        path = row["asset_path"]
        return (
            None if path is None else Path(str(path)),
            str(row["asset_sha256"]),
        )

    def migrate_legacy_script_fingerprint(
        self,
        *,
        controller_id: str,
        expected_legacy_sha256: str,
        stable_sha256: str,
        asset_path: Path,
    ) -> None:
        """Replace a verified path-bound legacy hash with its stable content hash."""
        identity = validate_safe_id(controller_id)
        legacy = validate_sha256(expected_legacy_sha256)
        stable = validate_sha256(stable_sha256)
        with self.database.transaction() as connection:
            changed = connection.execute(
                """UPDATE controllers SET asset_sha256 = ?, asset_path = ?
                   WHERE controller_id = ? AND kind = 'script'
                     AND asset_sha256 = ?""",
                (stable, str(asset_path.resolve()), identity, legacy),
            ).rowcount
            if changed != 1:
                raise ValueError("legacy script fingerprint binding changed")
            connection.execute(
                """INSERT INTO league_meta(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (
                    f"script_fingerprint_migration:{identity}",
                    json.dumps(
                        {
                            "legacy_sha256": legacy,
                            "stable_sha256": stable,
                            "migrated_at": _utc_now(),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            )

    def register_deck(
        self,
        *,
        deck_digest: str,
        signature: str,
        label: str,
        path: Path,
        file_sha256: str,
    ) -> bool:
        """Register one exact deck and backfill every compatible bundle."""
        digest = validate_sha256(deck_digest)
        sha256 = validate_sha256(file_sha256)
        resolved = path.resolve()
        now = _utc_now()
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT signature, path FROM decks WHERE deck_digest = ?",
                (digest,),
            ).fetchone()
            created = existing is None
            if existing is not None and str(existing["signature"]) != signature:
                raise ValueError("deck digest is already bound to another signature")
            if existing is None:
                connection.execute(
                    """INSERT INTO decks(
                           deck_digest, signature, label, path, file_sha256,
                           active, introduced_at
                       ) VALUES (?, ?, ?, ?, ?, 1, ?)""",
                    (digest, signature, label, str(resolved), sha256, now),
                )
                self._insert_initial_rating(connection, digest, "deck")
            elif not Path(str(existing["path"])).is_file():
                connection.execute(
                    """UPDATE decks SET label = ?, path = ?, file_sha256 = ?,
                           active = 1 WHERE deck_digest = ?""",
                    (label, str(resolved), sha256, digest),
                )
            compatible = connection.execute(
                """SELECT controller_id FROM controller_deck_compatibility
                   WHERE deck_digest = ?""",
                (digest,),
            ).fetchall()
            for row in compatible:
                self._insert_bundle(connection, str(row["controller_id"]), digest, now)
        return created

    def register_controller(
        self,
        *,
        controller_id: str,
        kind: str,
        label: str,
        asset_path: Path | None,
        asset_sha256: str,
        pair_manifest_path: Path | None,
        candidate_kind: CandidateKind,
        compatible_deck_digests: Sequence[str],
        requires_cuda: bool,
    ) -> bool:
        """Register one immutable controller and all currently available bundles."""
        identity = validate_safe_id(controller_id)
        fingerprint = validate_sha256(asset_sha256)
        if kind not in {"checkpoint", "script"}:
            raise ValueError("unknown controller kind")
        digests = tuple(
            sorted({validate_sha256(item) for item in compatible_deck_digests})
        )
        if not digests:
            raise ValueError("controller must declare at least one exact deck")
        state = {
            "automatic": "candidate",
            "manual": "incumbent",
            "anchor": "anchor",
        }[candidate_kind]
        now = _utc_now()
        with self.database.transaction() as connection:
            existing = connection.execute(
                "SELECT asset_sha256, kind FROM controllers WHERE controller_id = ?",
                (identity,),
            ).fetchone()
            created = existing is None
            if existing is not None and (
                str(existing["asset_sha256"]) != fingerprint
                or str(existing["kind"]) != kind
            ):
                raise ValueError(
                    "controller identity is already bound to another asset"
                )
            if existing is None:
                connection.execute(
                    """INSERT INTO controllers(
                           controller_id, kind, label, asset_path, asset_sha256,
                           pair_manifest_path, candidate_kind, candidate_state,
                           active, requires_cuda, runtime_fingerprint,
                           belief_fingerprint, introduced_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?, ?)""",
                    (
                        identity,
                        kind,
                        label,
                        None if asset_path is None else str(asset_path.resolve()),
                        fingerprint,
                        None
                        if pair_manifest_path is None
                        else str(pair_manifest_path.resolve()),
                        candidate_kind,
                        state,
                        int(requires_cuda),
                        self.config.runtime_fingerprint,
                        self.config.belief_fingerprint,
                        now,
                    ),
                )
                self._insert_initial_rating(connection, identity, "controller")
            for digest in digests:
                connection.execute(
                    """INSERT OR IGNORE INTO controller_deck_compatibility(
                           controller_id, deck_digest) VALUES (?, ?)""",
                    (identity, digest),
                )
                if (
                    connection.execute(
                        "SELECT 1 FROM decks WHERE deck_digest = ? AND active = 1",
                        (digest,),
                    ).fetchone()
                    is not None
                ):
                    self._insert_bundle(connection, identity, digest, now)
        return created

    def add_submission_alias(
        self,
        *,
        alias: str,
        controller_id: str,
        release_manifest_path: Path,
        release_manifest_sha256: str,
    ) -> None:
        """Attach one immutable release/submission name to a controller."""
        identity = validate_safe_id(controller_id)
        manifest_sha = validate_sha256(release_manifest_sha256)
        with self.database.transaction() as connection:
            existing = connection.execute(
                """SELECT controller_id, release_manifest_sha256
                   FROM submission_aliases WHERE alias = ?""",
                (alias,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["controller_id"]) != identity
                    or str(existing["release_manifest_sha256"]) != manifest_sha
                ):
                    raise ValueError(
                        "submission alias is already bound to another release"
                    )
                return
            connection.execute(
                """INSERT INTO submission_aliases(
                       alias, controller_id, release_manifest_path,
                       release_manifest_sha256, introduced_at
                   ) VALUES (?, ?, ?, ?, ?)""",
                (
                    alias,
                    identity,
                    str(release_manifest_path.resolve()),
                    manifest_sha,
                    _utc_now(),
                ),
            )

    def set_controller_active(
        self, controller_id: str, *, active: bool, reason: str
    ) -> None:
        """Manually disable or restore one controller and all combinations."""
        identity = validate_safe_id(controller_id)
        with self.database.transaction() as connection:
            changed = connection.execute(
                "UPDATE controllers SET active = ? WHERE controller_id = ?",
                (int(active), identity),
            ).rowcount
            if changed != 1:
                raise KeyError(f"unknown league controller: {identity}")
            connection.execute(
                "UPDATE bundles SET active = ? WHERE controller_id = ?",
                (int(active), identity),
            )
            if not active:
                connection.execute(
                    """UPDATE matches SET state = 'cancelled', leased_by = NULL,
                           lease_expires_at = NULL
                       WHERE state = 'queued' AND (
                           side_a_bundle_id IN (
                               SELECT bundle_id FROM bundles
                               WHERE controller_id = ?
                           ) OR side_b_bundle_id IN (
                               SELECT bundle_id FROM bundles
                               WHERE controller_id = ?
                           )
                       )""",
                    (identity, identity),
                )
            connection.execute(
                """INSERT INTO league_meta(key, value) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET value = excluded.value""",
                (
                    f"activation:{identity}",
                    json.dumps(
                        {"active": active, "reason": reason, "at": _utc_now()},
                        sort_keys=True,
                    ),
                ),
            )

    def _insert_initial_rating(
        self, connection: sqlite3.Connection, component_id: str, kind: str
    ) -> None:
        rating = self.skill.initial(
            component_id,
            cast(Literal["controller", "deck"], kind),
        )
        connection.execute(
            """INSERT INTO component_ratings(
                   component_id, component_kind, mu, sigma, games
               ) VALUES (?, ?, ?, ?, ?)""",
            (component_id, kind, rating.mu, rating.sigma, rating.games),
        )

    @staticmethod
    def _insert_bundle(
        connection: sqlite3.Connection,
        controller_id: str,
        deck_digest: str,
        introduced_at: str,
    ) -> None:
        bundle_id = _bundle_id(controller_id, deck_digest)
        active = connection.execute(
            "SELECT active FROM controllers WHERE controller_id = ?",
            (controller_id,),
        ).fetchone()["active"]
        connection.execute(
            """INSERT OR IGNORE INTO bundles(
                   bundle_id, controller_id, deck_digest, active, introduced_at
               ) VALUES (?, ?, ?, ?, ?)""",
            (bundle_id, controller_id, deck_digest, int(active), introduced_at),
        )


def _bundle_id(controller_id: str, deck_digest: str) -> str:
    digest = hashlib.sha256(f"{controller_id}\0{deck_digest}".encode()).hexdigest()
    return f"bundle:{digest}"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["LeagueAssetLedgerMixin"]
