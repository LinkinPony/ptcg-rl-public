"""Select traceable high-quality BC sides from compact Kaggle Daily data."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import hydra
from omegaconf import DictConfig, OmegaConf

from ptcg_rl.data.kaggle.recent_bc_artifacts import (
    atomic_write_bytes,
    display_path,
    fingerprint,
    load_daily_rows,
    read_json,
    repo_path,
    sha256_file,
    utc_now,
    validate_selected_episode_members,
    validation_split,
    write_binding_csvs,
    write_parquet,
    write_pretraining_source_csvs,
)
from ptcg_rl.data.kaggle.recent_bc_candidates import (
    candidate_deck_rows,
    candidate_sort_key,
    deck_hashes_by_digest,
    load_active_routes,
    load_rank10_evidence,
    mode,
    score_rate,
    single_deck_hash,
    wilson_lcb,
)
from ptcg_rl.data.kaggle.recent_bc_leaderboard import (
    LeaderboardCohort,
    load_leaderboard_cohort,
)
from ptcg_rl.data.kaggle.recent_bc_models import RecentBCSelectionConfig

SELECTION_SCHEMA_VERSION = 1
_RESULTS = frozenset(("win", "draw", "loss"))


def run(config: RecentBCSelectionConfig) -> dict[str, Any]:
    """Build and publish one compact, side-bound BC source selection."""
    output_dir = repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows, source_records = load_daily_rows(config)
    resolved_path = repo_path(config.run_resolved_config_path)
    resolved = read_json(resolved_path)
    active_routes, active_signatures = load_active_routes(resolved)
    active_digests = frozenset(active_signatures)

    valid_rows = [
        dict(row)
        for row in source_rows
        if bool(row["terminal_valid"]) and str(row["result"]) in _RESULTS
    ]
    leaderboard = _leaderboard_cohort(config)
    cohort_rows = _leaderboard_rows(valid_rows, leaderboard=leaderboard)
    group_rows = _group_quality_rows(
        cohort_rows,
        active_digests=active_digests,
        config=config,
    )
    admitted_keys = {
        (str(row["pilot_key"]), str(row["deck_digest"]))
        for row in group_rows
        if bool(row["qualified"]) or not config.require_group_quality
    }
    selected_sides = [
        dict(row)
        for row in cohort_rows
        if (str(row["pilot_key"]), str(row["deck_digest"])) in admitted_keys
    ]
    for row in selected_sides:
        row["roster_covered"] = str(row["deck_digest"]) in active_digests
    if config.require_roster_coverage:
        selected_sides = [
            row for row in selected_sides if bool(row["roster_covered"])
        ]
    if not selected_sides:
        raise ValueError("quality and roster gates selected no BC sides")
    selected_by_episode: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_sides:
        selected_by_episode[int(row["episode_id"])].append(row)
    if any(len(rows) not in (1, 2) for rows in selected_by_episode.values()):
        raise ValueError("selected episode has an invalid number of sides")

    replay_rows, decision_counts = validate_selected_episode_members(
        config,
        selected_by_episode=selected_by_episode,
        source_records=source_records,
    )
    for row in selected_sides:
        key = (int(row["episode_id"]), int(row["player_index"]))
        row["decision_count"] = decision_counts.get(key)
        row["split"] = validation_split(
            int(row["episode_id"]),
            seed=config.split_seed,
            fraction=config.validation_fraction,
        )
    if config.validate_selected_replays and any(
        not isinstance(row["decision_count"], int)
        or int(row["decision_count"]) <= 0
        for row in selected_sides
    ):
        raise ValueError("selected BC side has no validated trainable decisions")

    deck_hashes = deck_hashes_by_digest(valid_rows)
    candidate_rows = candidate_deck_rows(
        valid_rows,
        group_rows=group_rows,
        active_routes=active_routes,
        active_signatures=active_signatures,
        active_deck_hashes=deck_hashes,
        rank10=load_rank10_evidence(config),
        card_data_csv=repo_path(config.card_data_csv),
        dates=config.dates,
        wilson_z=config.wilson_z,
    )
    selected_keys = {
        (str(row["pilot_key"]), str(row["deck_digest"]))
        for row in selected_sides
    }
    qualified_rows = [
        {**row, "bc_admitted": True}
        for row in group_rows
        if (str(row["pilot_key"]), str(row["deck_digest"])) in selected_keys
    ]
    _sort_outputs(selected_sides, replay_rows, qualified_rows, candidate_rows)

    outputs = {
        "selected_sides": write_parquet(
            output_dir / "selected_sides.parquet", selected_sides
        ),
        "selected_episodes": write_parquet(
            output_dir / "selected_episodes.parquet", replay_rows
        ),
        "qualified_groups": write_parquet(
            output_dir / "qualified_groups.parquet", qualified_rows
        ),
        "candidate_decks": write_parquet(
            output_dir / "candidate_decks.parquet", candidate_rows
        ),
    }
    outputs.update(
        write_binding_csvs(
            output_dir,
            selected_sides=selected_sides,
            replay_rows=replay_rows,
        )
    )
    per_date: list[dict[str, Any]] | None = None
    if config.write_pretraining_source:
        source_outputs, per_date = write_pretraining_source_csvs(
            output_dir,
            selected_sides=selected_sides,
            replay_rows=replay_rows,
        )
        outputs.update(
            {
                "pretraining_replays": source_outputs["replays_manifest"],
                "pretraining_teams": source_outputs["top_teams"],
                "pretraining_episode_teams": source_outputs["episode_teams"],
                "replays_manifest_path": source_outputs["replays_manifest"]["path"],
                "replays_manifest_sha256": source_outputs["replays_manifest"]["sha256"],
                "top30_path": source_outputs["top_teams"]["path"],
                "top30_sha256": source_outputs["top_teams"]["sha256"],
                "episode_teams_path": source_outputs["episode_teams"]["path"],
                "episode_teams_sha256": source_outputs["episode_teams"]["sha256"],
            }
        )
    identity = _selection_identity(
        config,
        source_records=source_records,
        resolved=resolved,
        resolved_path=resolved_path,
        active_exact_decks=len(active_digests),
        leaderboard=leaderboard,
    )
    artifact_fingerprint = fingerprint(identity)
    covered = [row for row in selected_sides if bool(row["roster_covered"])]
    quarantined = [
        row for row in selected_sides if not bool(row["roster_covered"])
    ]
    summary = _summary(
        source_rows=source_rows,
        valid_rows=valid_rows,
        qualified_rows=qualified_rows,
        selected_sides=selected_sides,
        replay_rows=replay_rows,
        covered=covered,
        quarantined=quarantined,
        candidate_rows=candidate_rows,
    )
    manifest = {
        **identity,
        "artifact_fingerprint": artifact_fingerprint,
        "created_at_utc": utc_now(),
        "selection": {
            "mode": (
                "episode_team_bindings"
                if config.write_pretraining_source
                else "episode_side_bindings_v1"
            ),
            "training_boundary": (
                "only leaderboard-admitted, terminal-valid, positive-decision "
                "sides covered by the bound exact registry are BC eligible"
                if config.require_roster_coverage
                else "roster_covered sides are current-registry BC eligible; "
                "uncovered sides remain quarantined until an explicit exact-route "
                "topology transition"
            ),
        },
        "summary": summary,
        "outputs": outputs,
    }
    if per_date is not None:
        manifest["per_date"] = per_date
    manifest_path = output_dir / "manifest.json"
    atomic_write_bytes(
        manifest_path,
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode(),
    )
    result = {
        "manifest_path": display_path(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifact_fingerprint": artifact_fingerprint,
        **summary,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _group_quality_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    active_digests: frozenset[str],
    config: RecentBCSelectionConfig,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["pilot_key"]), str(row["deck_digest"]))].append(row)
    output: list[dict[str, Any]] = []
    for (pilot_key, digest), group in sorted(grouped.items()):
        sides = len(group)
        wins = sum(str(row["result"]) == "win" for row in group)
        draws = sum(str(row["result"]) == "draw" for row in group)
        losses = sum(str(row["result"]) == "loss" for row in group)
        first_rows = [row for row in group if bool(row["went_first"])]
        second_rows = [row for row in group if not bool(row["went_first"])]
        lcb = wilson_lcb(wins + 0.5 * draws, sides, z=config.wilson_z)
        distinct_dates = len({str(row["date"]) for row in group})
        qualified = (
            sides >= config.min_group_sides
            and len(first_rows) >= config.min_sides_per_seat
            and len(second_rows) >= config.min_sides_per_seat
            and distinct_dates >= config.min_dates
            and lcb >= config.min_wilson_lcb
        )
        output.append(
            {
                "pilot_key": pilot_key,
                "deck_digest": digest,
                "deck_hash": single_deck_hash(group),
                "deck_label": mode(str(row["deck_label"]) for row in group),
                "sides": sides,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score_rate": (wins + 0.5 * draws) / sides,
                "wilson_lcb": lcb,
                "distinct_dates": distinct_dates,
                "first_sides": len(first_rows),
                "first_score_rate": score_rate(first_rows),
                "second_sides": len(second_rows),
                "second_score_rate": score_rate(second_rows),
                "qualified": qualified,
                "roster_covered": digest in active_digests,
            }
        )
    return output


def _leaderboard_cohort(
    config: RecentBCSelectionConfig,
) -> LeaderboardCohort | None:
    if config.leaderboard_path is None:
        return None
    return load_leaderboard_cohort(
        config.leaderboard_path,
        top_fraction=config.leaderboard_top_fraction,
        max_rank=config.leaderboard_max_rank,
        min_score=config.leaderboard_min_score,
    )


def _leaderboard_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    leaderboard: LeaderboardCohort | None,
) -> list[dict[str, Any]]:
    if leaderboard is None:
        return [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    for row in rows:
        evidence = leaderboard.teams_by_pilot.get(str(row["pilot_key"]))
        if evidence is None:
            continue
        selected.append(
            {
                **row,
                "team_id": str(evidence["team_id"]),
                "team_name": str(evidence["team_name"]),
                "leaderboard_rank": int(evidence["rank"]),
                "leaderboard_score": float(evidence["score"]),
                "leaderboard_criteria": "|".join(evidence["criteria"]),
            }
        )
    if not selected:
        raise ValueError("leaderboard cohort matched no Daily replay sides")
    return selected


def _sort_outputs(
    selected_sides: list[dict[str, Any]],
    replay_rows: list[dict[str, Any]],
    qualified_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> None:
    selected_sides.sort(
        key=lambda row: (int(row["episode_id"]), int(row["player_index"]))
    )
    replay_rows.sort(key=lambda row: int(row["episode_id"]))
    qualified_rows.sort(
        key=lambda row: (
            -float(row["wilson_lcb"]),
            -int(row["sides"]),
            str(row["deck_hash"]),
            str(row["pilot_key"]),
        )
    )
    candidate_rows.sort(key=candidate_sort_key)


def _selection_identity(
    config: RecentBCSelectionConfig,
    *,
    source_records: Sequence[Mapping[str, Any]],
    resolved: Mapping[str, Any],
    resolved_path: Path,
    active_exact_decks: int,
    leaderboard: LeaderboardCohort | None,
) -> dict[str, Any]:
    rank10_manifest = repo_path(config.rank10_collection_manifest_path)
    rank10_summary = repo_path(config.rank10_team_deck_summary_path)
    return {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "dates": list(config.dates),
        "sources": list(source_records),
        "quality_gates": {
            "unit": "pilot_key_x_exact_deck_digest",
            "min_group_sides": config.min_group_sides,
            "min_sides_per_seat": config.min_sides_per_seat,
            "min_dates": config.min_dates,
            "min_wilson_lcb": config.min_wilson_lcb,
            "wilson_z": config.wilson_z,
            "draw_value": 0.5,
            "individual_outcome_filter": False,
            "require_terminal_valid": True,
            "require_positive_validated_decisions": config.validate_selected_replays,
            "group_quality_required": config.require_group_quality,
            "exact_roster_coverage_required": config.require_roster_coverage,
        },
        "split": {
            "domain": "ptcg-rl/recent-bc-selection-split/v1",
            "seed": config.split_seed,
            "validation_fraction": config.validation_fraction,
            "unit": "complete_episode",
        },
        "roster": {
            "resolved_config_path": display_path(resolved_path),
            "resolved_config_sha256": sha256_file(resolved_path),
            "training_roster_fingerprint": resolved.get(
                "training_roster_fingerprint"
            ),
            "exact_registry_fingerprint": resolved.get(
                "exact_registry_fingerprint"
            ),
            "active_exact_decks": active_exact_decks,
        },
        "rank10": {
            "collection_manifest_path": display_path(rank10_manifest),
            "collection_manifest_sha256": sha256_file(rank10_manifest),
            "team_deck_summary_path": display_path(rank10_summary),
            "team_deck_summary_sha256": sha256_file(rank10_summary),
        },
        "leaderboard_gate": (
            None if leaderboard is None else dict(leaderboard.identity)
        ),
    }


def _summary(
    *,
    source_rows: Sequence[Mapping[str, Any]],
    valid_rows: Sequence[Mapping[str, Any]],
    qualified_rows: Sequence[Mapping[str, Any]],
    selected_sides: Sequence[Mapping[str, Any]],
    replay_rows: Sequence[Mapping[str, Any]],
    covered: Sequence[Mapping[str, Any]],
    quarantined: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "source_episodes": len({int(row["episode_id"]) for row in valid_rows}),
        "source_sides": len(source_rows),
        "terminal_valid_sides": len(valid_rows),
        "terminal_invalid_sides": len(source_rows) - len(valid_rows),
        "qualified_groups": len(qualified_rows),
        "qualified_pilots": len({str(row["pilot_key"]) for row in qualified_rows}),
        "selected_sides": len(selected_sides),
        "selected_episodes": len(replay_rows),
        "selected_decisions": sum(
            int(row["decision_count"] or 0) for row in selected_sides
        ),
        "train_sides": sum(row["split"] == "train" for row in selected_sides),
        "validation_sides": sum(
            row["split"] == "validation" for row in selected_sides
        ),
        "roster_covered_sides": len(covered),
        "roster_covered_episodes": len({int(row["episode_id"]) for row in covered}),
        "quarantined_uncovered_sides": len(quarantined),
        "quarantined_uncovered_episodes": len(
            {int(row["episode_id"]) for row in quarantined}
        ),
        "selected_exact_decks": len(
            {str(row["deck_digest"]) for row in selected_sides}
        ),
        "roster_covered_exact_decks": len(
            {str(row["deck_digest"]) for row in covered}
        ),
        "quarantined_uncovered_exact_decks": len(
            {str(row["deck_digest"]) for row in quarantined}
        ),
        "missing_roster_candidate_decks": len(candidate_rows),
    }


@hydra.main(
    version_base=None,
    config_path="../../../../configs",
    config_name="data/kaggle_recent_bc_selection_4d_20260815",
)
def main(hydra_config: DictConfig) -> None:
    """Hydra entry point."""
    raw = OmegaConf.to_container(hydra_config, resolve=True)
    if not isinstance(raw, dict):
        raise ValueError("Hydra config must resolve to an object")
    run(RecentBCSelectionConfig.model_validate(cast(dict[str, Any], raw)))


if __name__ == "__main__":
    main()


__all__ = ["RecentBCSelectionConfig", "main", "run"]
