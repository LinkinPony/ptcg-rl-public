"""Exact-deck extraction and reporting for a fixed top-team replay cohort."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent import futures
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from ptcg_rl.data.kaggle.top_team_live_models import (
    EpisodeAssociation,
    InventoryEpisode,
    ManifestEpisode,
    ReplayManifest,
    Result,
    SideObservation,
    SideSource,
    TargetTeam,
    TopTeamLiveConfig,
)
from ptcg_rl.data.kaggle.top_team_live_sync import DailyReplay
from ptcg_rl.data.kaggle_deck import records as deck_records
from ptcg_rl.data.kaggle_steps import records as step_records

_SIDE_SOURCES = (
    "inventory_daily",
    "inventory_live",
    "inventory_inferred",
    "daily_discovered",
    "live_discovered",
)


def analyze_replays(
    config: TopTeamLiveConfig,
    inventory: list[InventoryEpisode],
    manifest: ReplayManifest,
    daily_replays: Mapping[int, DailyReplay],
    inventory_query_counts: Mapping[int, int],
) -> dict[str, Any]:
    """Extract target sides, aggregate exact decks, and atomically write outputs."""
    card_meta = deck_records.load_card_meta(
        deck_records.repo_path(config.card_data_csv)
    )
    observations, inference = _extract_observations(
        config,
        inventory,
        manifest,
        daily_replays,
        card_meta,
    )
    if not observations:
        raise ValueError("no target-team side with a readable exact deck was observed")
    deck_rows = _team_deck_rows(observations, config.teams)
    team_rows = _team_rows(observations, config.teams)
    output_dir = deck_records.repo_path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_inventory_parquet(output_dir / "inventory_sides.parquet", inventory)
    _atomic_write_parquet(output_dir / "side_observations.parquet", observations)
    _atomic_write_csv(
        output_dir / "team_deck_summary.csv",
        deck_rows,
        _team_deck_fields(),
    )
    _atomic_write_csv(
        output_dir / "team_summary.csv",
        team_rows,
        _team_fields(),
    )
    source_counts = Counter(row.source for row in observations)
    inventory_sources = Counter(row.source for row in manifest.episodes)
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "window": {
            "start_time_utc": config.start_time_utc.isoformat(),
            "end_time_utc": config.end_time_utc.isoformat(),
            "semantics": "start-inclusive, end-exclusive",
        },
        "cohort": [team.model_dump(mode="json") for team in config.teams],
        "inventory": {
            "submission_query_episode_counts": {
                str(key): value for key, value in sorted(inventory_query_counts.items())
            },
            "unique_episodes": len(inventory),
            "inventory_sides": sum(len(row.associations) for row in inventory),
            "replay_evidence_episodes": manifest.episode_count,
            "replay_evidence_sides": manifest.association_count,
            "daily_reused_episodes": inventory_sources["daily_reuse"],
            "live_only_episodes": (
                inventory_sources["live_cache"] + inventory_sources["live_download"]
            ),
            "live_downloaded_this_run": inventory_sources["live_download"],
            "live_cache_reused": inventory_sources["live_cache"],
            "manifest": deck_records.display_path(
                deck_records.repo_path(config.replay_output_dir) / "manifest.json"
            ),
        },
        "analysis": {
            "target_sides": len(observations),
            "inventory_sides": (
                source_counts["inventory_daily"]
                + source_counts["inventory_live"]
                + source_counts["inventory_inferred"]
            ),
            "daily_discovered_extra_sides": source_counts["daily_discovered"],
            "live_discovered_extra_sides": source_counts["live_discovered"],
            "source_counts": dict(sorted(source_counts.items())),
            "inferred_inventory_sides": inference["inferred_sides"],
            "unattributed_inventory_sides": inference["unattributed_sides"],
            "deck_anchor_submissions": inference["deck_anchor_submissions"],
            "ambiguous_deck_submissions": inference[
                "ambiguous_deck_submissions"
            ],
            "teams_observed": len({row.team_id for row in observations}),
            "exact_team_decks": len(deck_rows),
            "unique_exact_decks": len({row.deck_signature for row in observations}),
            "daily_replay_files_scanned": len(daily_replays),
        },
        "outputs": {
            "inventory_sides": deck_records.display_path(
                output_dir / "inventory_sides.parquet"
            ),
            "side_observations": deck_records.display_path(
                output_dir / "side_observations.parquet"
            ),
            "team_deck_summary": deck_records.display_path(
                output_dir / "team_deck_summary.csv"
            ),
            "team_summary": deck_records.display_path(output_dir / "team_summary.csv"),
            "summary": deck_records.display_path(output_dir / "summary.json"),
        },
        "teams": team_rows,
    }
    _atomic_write_json(output_dir / "summary.json", report)
    return report


def _extract_observations(
    config: TopTeamLiveConfig,
    inventory: list[InventoryEpisode],
    manifest: ReplayManifest,
    daily_replays: Mapping[int, DailyReplay],
    card_meta: Mapping[int, deck_records.CardMeta],
) -> tuple[list[SideObservation], dict[str, int]]:
    manifest_by_id = {row.episode_id: row for row in manifest.episodes}
    associations = {
        (episode.episode_id, association.team_id): association
        for episode in inventory
        for association in episode.associations
    }
    paths: dict[int, Path] = {
        episode_id: row.path for episode_id, row in daily_replays.items()
    }
    for episode in manifest.episodes:
        path = deck_records.repo_path(episode.replay_path)
        previous = paths.setdefault(episode.episode_id, path)
        if previous != path:
            raise ValueError(
                f"episode {episode.episode_id} resolves to two replay files: "
                f"{previous} and {path}"
            )

    teams_by_name = {team.team_name.casefold(): team for team in config.teams}
    observations: list[SideObservation] = []
    worker_count = min(config.download_workers, max(1, len(paths)))
    with futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = {
            executor.submit(
                _extract_episode,
                episode_id,
                path,
                teams_by_name,
                associations,
                manifest_by_id.get(episode_id),
                daily_replays.get(episode_id),
                card_meta,
            ): episode_id
            for episode_id, path in paths.items()
        }
        for future in futures.as_completed(pending):
            observations.extend(future.result())

    by_key: dict[tuple[int, str], SideObservation] = {}
    for row in observations:
        key = (row.episode_id, row.team_id)
        existing_observation = by_key.setdefault(key, row)
        if existing_observation != row:
            raise ValueError(f"conflicting target side observations for {key}")
    inferred, inference = _infer_inventory_observations(
        inventory,
        associations,
        by_key,
    )
    result = [*by_key.values(), *inferred]
    result.sort(
        key=lambda row: (
            row.create_time_utc or datetime.min.replace(tzinfo=UTC),
            row.episode_id,
            row.team_id,
        )
    )
    return result, inference


def _extract_episode(
    episode_id: int,
    path: Path,
    teams_by_name: Mapping[str, TargetTeam],
    associations: Mapping[tuple[int, str], EpisodeAssociation],
    manifest_episode: ManifestEpisode | None,
    daily_replay: DailyReplay | None,
    card_meta: Mapping[int, deck_records.CardMeta],
) -> list[SideObservation]:
    replay_stub = step_records.replay_stub(path, chunk_size=65_536)
    actual_id = deck_records.episode_id_from_replay(replay_stub, path)
    if actual_id != episode_id:
        raise ValueError(
            f"replay id mismatch while scanning: expected={episode_id} actual={actual_id}"
        )
    team_names = deck_records.team_names_from_replay(replay_stub)
    target_names = {
        name.casefold() for name in team_names if name.casefold() in teams_by_name
    }
    if not target_names:
        return []
    side_rows = _exact_side_rows(path, card_meta)
    side_by_name: dict[str, dict[str, Any]] = {}
    for row in side_rows:
        folded = str(row["team_name"]).casefold()
        if folded not in target_names:
            continue
        if folded in side_by_name:
            raise ValueError(
                f"target team appears twice in episode {episode_id}: {team_names}"
            )
        side_by_name[folded] = row
    missing_names = target_names.difference(side_by_name)
    if missing_names:
        raise ValueError(
            f"target sides have no readable 60-card registration in episode "
            f"{episode_id}: {sorted(missing_names)}"
        )

    result: list[SideObservation] = []
    for folded, row in side_by_name.items():
        team = teams_by_name[folded]
        association = associations.get((episode_id, team.team_id))
        source = _side_source(association, manifest_episode, daily_replay)
        player_index = int(row["player_index"])
        create_time = (
            manifest_episode.create_time_utc
            if manifest_episode is not None
            else daily_replay.create_time_utc
            if daily_replay is not None
            else None
        )
        end_time = (
            manifest_episode.end_time_utc if manifest_episode is not None else None
        )
        replay_source = (
            manifest_episode.source if manifest_episode is not None else "daily_reuse"
        )
        result.append(
            SideObservation(
                episode_id=episode_id,
                create_time_utc=create_time,
                end_time_utc=end_time,
                date=(
                    create_time.date().isoformat()
                    if create_time is not None
                    else path.parent.name
                ),
                team_id=team.team_id,
                team_name=team.team_name,
                player_index=player_index,
                opponent_team_name=(
                    team_names[1 - player_index]
                    if len(team_names) > 1 - player_index
                    else ""
                ),
                reward=row["reward"],
                status=str(row["status"]),
                result=cast(Result, str(row["result"])),
                deck_signature=str(row["deck_signature"]),
                deck_hash=str(row["deck_hash"]),
                deck_label=str(row["deck_label"]),
                deck_ids=tuple(int(value) for value in row["deck_ids"]),
                unique_card_ids=int(row["unique_card_ids"]),
                total_cards=int(row["total_cards"]),
                pokemon_summary=str(row["pokemon_summary"]),
                top_cards=str(row["top_cards"]),
                opponent_deck_signature=str(row["opponent_deck_signature"]),
                opponent_deck_hash=str(row["opponent_deck_hash"]),
                opponent_deck_label=str(row["opponent_deck_label"]),
                submission_ids=(
                    association.submission_ids if association is not None else ()
                ),
                source=source,
                replay_source=replay_source,
                replay_path=Path(deck_records.display_path(path)),
                size_bytes=path.stat().st_size,
                deck_evidence_episode_id=episode_id,
                exact_replay=True,
            )
        )
    return result


def _infer_inventory_observations(
    inventory: list[InventoryEpisode],
    associations: Mapping[tuple[int, str], EpisodeAssociation],
    exact_by_key: Mapping[tuple[int, str], SideObservation],
) -> tuple[list[SideObservation], dict[str, int]]:
    """Attribute inventory results through each submission's exact deck anchor."""
    evidence: dict[int, dict[str, SideObservation]] = {}
    for key, observation in exact_by_key.items():
        association = associations.get(key)
        if association is None:
            continue
        for submission_id in association.submission_ids:
            evidence.setdefault(submission_id, {}).setdefault(
                observation.deck_signature,
                observation,
            )

    inferred: list[SideObservation] = []
    unattributed = 0
    for episode in inventory:
        association_by_index = {
            association.player_index: association
            for association in episode.associations
        }
        for association in episode.associations:
            key = (episode.episode_id, association.team_id)
            if key in exact_by_key:
                _validate_exact_reward(exact_by_key[key], association)
                continue
            representatives = {
                representative.deck_signature: representative
                for submission_id in association.submission_ids
                for representative in evidence.get(submission_id, {}).values()
            }
            if len(representatives) != 1:
                unattributed += 1
                continue
            representative = next(iter(representatives.values()))
            opponent_association = association_by_index.get(
                1 - association.player_index
            )
            opponent_representative = _association_representative(
                opponent_association,
                evidence,
            )
            inferred.append(
                SideObservation(
                    episode_id=episode.episode_id,
                    create_time_utc=episode.create_time_utc,
                    end_time_utc=episode.end_time_utc,
                    date=episode.create_time_utc.date().isoformat(),
                    team_id=association.team_id,
                    team_name=association.team_name,
                    player_index=association.player_index,
                    opponent_team_name=association.opponent_team_name,
                    reward=association.reward,
                    status="DONE",
                    result=_result_from_reward(association.reward),
                    deck_signature=representative.deck_signature,
                    deck_hash=representative.deck_hash,
                    deck_label=representative.deck_label,
                    deck_ids=representative.deck_ids,
                    unique_card_ids=representative.unique_card_ids,
                    total_cards=representative.total_cards,
                    pokemon_summary=representative.pokemon_summary,
                    top_cards=representative.top_cards,
                    opponent_deck_signature=(
                        opponent_representative.deck_signature
                        if opponent_representative is not None
                        else ""
                    ),
                    opponent_deck_hash=(
                        opponent_representative.deck_hash
                        if opponent_representative is not None
                        else ""
                    ),
                    opponent_deck_label=(
                        opponent_representative.deck_label
                        if opponent_representative is not None
                        else ""
                    ),
                    submission_ids=association.submission_ids,
                    source="inventory_inferred",
                    replay_source=representative.replay_source,
                    replay_path=representative.replay_path,
                    size_bytes=representative.size_bytes,
                    deck_evidence_episode_id=representative.episode_id,
                    exact_replay=False,
                )
            )

    ambiguous = sum(len(rows) > 1 for rows in evidence.values())
    return inferred, {
        "inferred_sides": len(inferred),
        "unattributed_sides": unattributed,
        "deck_anchor_submissions": len(evidence),
        "ambiguous_deck_submissions": ambiguous,
    }


def _association_representative(
    association: EpisodeAssociation | None,
    evidence: Mapping[int, Mapping[str, SideObservation]],
) -> SideObservation | None:
    if association is None:
        return None
    representatives = {
        representative.deck_signature: representative
        for submission_id in association.submission_ids
        for representative in evidence.get(submission_id, {}).values()
    }
    if len(representatives) != 1:
        return None
    return next(iter(representatives.values()))


def _validate_exact_reward(
    observation: SideObservation,
    association: EpisodeAssociation,
) -> None:
    if (
        observation.reward is not None
        and association.reward is not None
        and observation.reward != association.reward
    ):
        raise ValueError(
            f"reward mismatch for episode {observation.episode_id}, "
            f"team {observation.team_id}: replay={observation.reward} "
            f"inventory={association.reward}"
        )


def _result_from_reward(reward: float | None) -> Result:
    if reward is None:
        return "other"
    if reward > 0.0:
        return "win"
    if reward < 0.0:
        return "loss"
    return "draw"


def _exact_side_rows(
    path: Path,
    card_meta: Mapping[int, deck_records.CardMeta],
) -> list[dict[str, Any]]:
    try:
        rows = deck_records.fast_episode_side_rows(
            replay_path=path,
            card_meta=dict(card_meta),
            known_decks={},
            prefix_bytes=65_536,
            include_step_count=False,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        rows = None
    if rows is not None:
        return rows
    replay = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(replay, dict):
        raise ValueError(f"replay root is not an object: {path}")
    return deck_records.episode_side_rows(
        replay_path=path,
        replay=replay,
        card_meta=dict(card_meta),
        known_decks={},
    )


def _side_source(
    association: EpisodeAssociation | None,
    manifest_episode: ManifestEpisode | None,
    daily_replay: DailyReplay | None,
) -> SideSource:
    if association is not None:
        if manifest_episode is None:
            raise ValueError("inventory association has no manifest episode")
        return (
            "inventory_daily"
            if manifest_episode.source == "daily_reuse"
            else "inventory_live"
        )
    return "daily_discovered" if daily_replay is not None else "live_discovered"


def _team_deck_rows(
    observations: list[SideObservation],
    teams: tuple[TargetTeam, ...],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[SideObservation]] = {}
    for row in observations:
        groups.setdefault((row.team_id, row.deck_signature), []).append(row)
    order = {team.team_id: index for index, team in enumerate(teams)}
    rows = [_summary_group(group, include_deck=True) for group in groups.values()]
    rows.sort(
        key=lambda row: (
            order[str(row["team_id"])],
            -int(row["games"]),
            str(row["deck_signature"]),
        )
    )
    return rows


def _team_rows(
    observations: list[SideObservation],
    teams: tuple[TargetTeam, ...],
) -> list[dict[str, Any]]:
    by_team: dict[str, list[SideObservation]] = {}
    for observation in observations:
        by_team.setdefault(observation.team_id, []).append(observation)
    rows: list[dict[str, Any]] = []
    for team in teams:
        group = by_team.get(team.team_id, [])
        if group:
            summary = _summary_group(group, include_deck=False)
        else:
            summary = _empty_team_row(team)
        summary["configured_submission_ids"] = _ids_text(team.submission_ids)
        summary["unique_exact_decks"] = len(
            {observation.deck_signature for observation in group}
        )
        rows.append(summary)
    return rows


def _summary_group(
    group: list[SideObservation],
    *,
    include_deck: bool,
) -> dict[str, Any]:
    first = group[0]
    counts = Counter(row.result for row in group)
    result_games = counts["win"] + counts["loss"] + counts["draw"]
    times = [row.create_time_utc for row in group if row.create_time_utc is not None]
    sources = Counter(row.source for row in group)
    row: dict[str, Any] = {
        "team_id": first.team_id,
        "team_name": first.team_name,
        "games": len(group),
        "result_games": result_games,
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "other": counts["other"],
        "win_rate": _rate(counts["win"], result_games),
        "score_rate": _rate(counts["win"] + 0.5 * counts["draw"], result_games),
        "first_seen_utc": min(times).isoformat() if times else "",
        "last_seen_utc": max(times).isoformat() if times else "",
        "submission_ids": _ids_text(
            sorted({value for item in group for value in item.submission_ids})
        ),
        "inventory_daily_games": sources["inventory_daily"],
        "inventory_live_games": sources["inventory_live"],
        "inventory_inferred_games": sources["inventory_inferred"],
        "daily_discovered_games": sources["daily_discovered"],
        "live_discovered_games": sources["live_discovered"],
        "source_counts": json.dumps(dict(sorted(sources.items())), sort_keys=True),
    }
    if include_deck:
        row.update(
            {
                "deck_hash": first.deck_hash,
                "deck_label": first.deck_label,
                "unique_card_ids": first.unique_card_ids,
                "total_cards": first.total_cards,
                "pokemon_summary": first.pokemon_summary,
                "top_cards": first.top_cards,
                "deck_signature": first.deck_signature,
            }
        )
    return row


def _empty_team_row(team: TargetTeam) -> dict[str, Any]:
    return {
        "team_id": team.team_id,
        "team_name": team.team_name,
        "games": 0,
        "result_games": 0,
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "other": 0,
        "win_rate": None,
        "score_rate": None,
        "first_seen_utc": "",
        "last_seen_utc": "",
        "submission_ids": "",
        "inventory_daily_games": 0,
        "inventory_live_games": 0,
        "inventory_inferred_games": 0,
        "daily_discovered_games": 0,
        "live_discovered_games": 0,
        "source_counts": "{}",
    }


def _rate(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _ids_text(values: Iterable[int]) -> str:
    return "|".join(str(value) for value in values)


def _base_fields() -> list[str]:
    return [
        "team_id",
        "team_name",
        "games",
        "result_games",
        "wins",
        "losses",
        "draws",
        "other",
        "win_rate",
        "score_rate",
        "first_seen_utc",
        "last_seen_utc",
        "submission_ids",
        "inventory_daily_games",
        "inventory_live_games",
        "inventory_inferred_games",
        "daily_discovered_games",
        "live_discovered_games",
        "source_counts",
    ]


def _team_deck_fields() -> list[str]:
    return [
        *_base_fields(),
        "deck_hash",
        "deck_label",
        "unique_card_ids",
        "total_cards",
        "pokemon_summary",
        "top_cards",
        "deck_signature",
    ]


def _team_fields() -> list[str]:
    return [*_base_fields(), "configured_submission_ids", "unique_exact_decks"]


def _atomic_write_parquet(path: Path, rows: list[SideObservation]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    payloads: list[dict[str, Any]] = []
    for row in rows:
        payload = row.model_dump()
        payload["replay_path"] = str(row.replay_path)
        payload["deck_ids"] = list(row.deck_ids)
        payload["submission_ids"] = list(row.submission_ids)
        payloads.append(payload)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(payloads), temporary)
    os.replace(temporary, path)


def _atomic_write_inventory_parquet(
    path: Path,
    inventory: list[InventoryEpisode],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    payloads = [
        {
            "episode_id": episode.episode_id,
            "create_time_utc": episode.create_time_utc,
            "end_time_utc": episode.end_time_utc,
            "state": episode.state,
            "episode_type": episode.episode_type,
            **association.model_dump(),
            "submission_ids": list(association.submission_ids),
        }
        for episode in inventory
        for association in episode.associations
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(payloads), temporary)
    os.replace(temporary, path)


def _atomic_write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
