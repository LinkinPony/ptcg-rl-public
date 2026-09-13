"""Create fingerprinted bootstrap states for metric-driven league promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ptcg_rl.rl.curriculum import (
    FrozenPoolState,
    read_frozen_pool_state,
    write_frozen_pool_state,
)
from ptcg_rl.rl.league_promotion import (
    LeaguePromotionState,
    write_league_promotion_state,
)


def main() -> None:
    """Validate one immutable pool and publish a bounded-role migration."""
    args = _parser().parse_args()
    source_path = args.source_state.resolve()
    actual_source_sha256 = _file_sha256(source_path)
    if actual_source_sha256 != args.source_sha256:
        raise ValueError("source frozen-pool state SHA256 mismatch")
    source = read_frozen_pool_state(source_path)
    migrated, promotion = _migrate(
        source,
        anchor_ids=tuple(args.anchor_id),
        champion_id=args.champion_id,
        evicted_ids=tuple(args.evict_id),
        baseline_policy_version=args.baseline_policy_version,
        max_opponents=args.max_opponents,
    )
    output_dir = args.output_dir.resolve()
    pool_path = output_dir / "frozen_pool_state.json"
    promotion_path = output_dir / "league_promotion_state.json"
    manifest_path = output_dir / "migration_manifest.json"
    existing = [
        path for path in (pool_path, promotion_path, manifest_path) if path.exists()
    ]
    if existing:
        raise FileExistsError(f"league migration outputs already exist: {existing}")
    output_dir.mkdir(parents=True, exist_ok=True)
    write_frozen_pool_state(pool_path, migrated)
    write_league_promotion_state(promotion_path, promotion)
    manifest = {
        "schema_version": 1,
        "source_state_path": str(source_path),
        "source_state_sha256": actual_source_sha256,
        "baseline_policy_version": args.baseline_policy_version,
        "anchor_ids": list(args.anchor_id),
        "champion_id": args.champion_id,
        "evicted_ids": list(args.evict_id),
        "max_opponents": args.max_opponents,
        "frozen_pool_state_path": str(pool_path),
        "frozen_pool_state_sha256": _file_sha256(pool_path),
        "league_promotion_state_path": str(promotion_path),
        "league_promotion_state_sha256": _file_sha256(promotion_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def _migrate(
    source: FrozenPoolState,
    *,
    anchor_ids: tuple[str, ...],
    champion_id: str,
    evicted_ids: tuple[str, ...],
    baseline_policy_version: int,
    max_opponents: int,
) -> tuple[FrozenPoolState, LeaguePromotionState]:
    if not anchor_ids or len(anchor_ids) != len(set(anchor_ids)):
        raise ValueError("anchor ids must be non-empty and unique")
    if len(evicted_ids) != len(set(evicted_ids)):
        raise ValueError("evicted ids must be unique")
    if baseline_policy_version < 0:
        raise ValueError("baseline policy version must be non-negative")
    if max_opponents <= 1:
        raise ValueError("max opponents must be greater than one")
    by_id = {member.opponent_id: member for member in source.members}
    required = set(anchor_ids) | set(evicted_ids) | {champion_id}
    unknown = required - by_id.keys()
    if unknown:
        raise ValueError(f"league migration references unknown members: {unknown}")
    if champion_id in anchor_ids:
        raise ValueError("league champion must not also be a fixed anchor")
    if set(evicted_ids) & (set(anchor_ids) | {champion_id}):
        raise ValueError("league migration cannot evict an anchor or champion")
    kept = tuple(
        member.model_copy(update={"pinned": member.opponent_id in anchor_ids})
        for member in source.members
        if member.opponent_id not in evicted_ids
    )
    if len(kept) > max_opponents - 1:
        raise ValueError("migrated pool must leave one resident challenger slot")
    migrated = source.model_copy(update={"members": kept})
    exploiters = tuple(
        member.opponent_id
        for member in sorted(kept, key=lambda item: item.added_order)
        if member.opponent_id not in anchor_ids
        and member.opponent_id != champion_id
    )
    promotion = LeaguePromotionState(
        anchor_ids=anchor_ids,
        champion_id=champion_id,
        exploiter_ids=exploiters,
        last_candidate_version=baseline_policy_version,
        last_promotion_version=baseline_policy_version,
    )
    return migrated, promotion


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--anchor-id", action="append", required=True)
    parser.add_argument("--champion-id", required=True)
    parser.add_argument("--evict-id", action="append", default=[])
    parser.add_argument("--baseline-policy-version", type=int, required=True)
    parser.add_argument("--max-opponents", type=int, default=6)
    return parser


if __name__ == "__main__":
    main()
