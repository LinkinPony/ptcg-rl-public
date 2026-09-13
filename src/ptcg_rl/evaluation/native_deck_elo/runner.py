"""Bounded native deck Elo orchestration with shared CUDA batching."""

from __future__ import annotations

import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from queue import Queue
from typing import Any

from ptcg_rl.data.kaggle_deck import records
from ptcg_rl.decks.identity import canonicalize_deck
from ptcg_rl.evaluation.continuous_league.models import (
    BundleIdentity,
    MatchLease,
    NativeMatchConfig,
)
from ptcg_rl.evaluation.continuous_league.native_match import (
    NativeMatchExecutor,
    native_match_contract_fingerprints,
)
from ptcg_rl.evaluation.continuous_league.native_runtime import ControllerFactory
from ptcg_rl.evaluation.native_deck_elo.models import (
    DeckAsset,
    NativeDeckEloConfig,
    ScheduledGame,
)
from ptcg_rl.evaluation.native_deck_elo.results import result_row
from ptcg_rl.evaluation.native_deck_elo.schedule import schedule_games
from ptcg_rl.evaluation.native_deck_elo.scoring import write_final_artifacts
from ptcg_rl.evaluation.native_deck_elo.storage import (
    campaign_payload,
    fingerprint,
    load_parts,
    next_part_index,
    publish_or_validate_manifest,
    resolve_file,
    resolve_path,
    verified_file,
    write_part,
    write_progress,
)


def run_native_deck_elo(config: NativeDeckEloConfig) -> dict[str, Any]:
    """Run or resume exactly one bounded deck Elo campaign."""
    root = records.repo_path(Path(".")).resolve()
    checkpoint_path = verified_file(
        config.checkpoint_path,
        root=root,
        expected_sha256=config.expected_checkpoint_sha256,
    )
    catalog_path = verified_file(
        config.public_catalog_manifest_path,
        root=root,
        expected_sha256=config.expected_public_catalog_manifest_sha256,
    )
    library_path = verified_file(
        config.native_library_path,
        root=root,
        expected_sha256=config.expected_native_library_sha256,
    )
    native_config = NativeMatchConfig(
        library_path=library_path,
        expected_library_sha256=config.expected_native_library_sha256,
        public_catalog_manifest_path=catalog_path,
        expected_public_catalog_manifest_sha256=(
            config.expected_public_catalog_manifest_sha256
        ),
        checkpoint_device=config.device,
        lane_worker_count=config.native_lane_worker_count,
        option_capacity=config.native_option_capacity,
        maximum_engine_steps=config.maximum_engine_steps,
        checkpoint_cache_entries=config.checkpoint_cache_entries,
        policy_batch_max_rows=config.policy_batch_max_rows,
        policy_batch_wait_ms=config.policy_batch_wait_ms,
        act_time=config.act_time,
    )
    runtime_fingerprint, belief_fingerprint = native_match_contract_fingerprints(
        native_config
    )
    decks = _load_decks(
        config,
        root=root,
        controller_id=f"checkpoint:{config.expected_checkpoint_sha256}",
    )
    campaign = campaign_payload(
        config,
        decks=decks,
        runtime_fingerprint=runtime_fingerprint,
        belief_fingerprint=belief_fingerprint,
    )
    campaign_fingerprint = fingerprint(campaign)
    output_dir = resolve_path(config.output_dir, root=root)
    output_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = output_dir / "games_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    publish_or_validate_manifest(
        output_dir / "manifest.json",
        {**campaign, "campaign_fingerprint": campaign_fingerprint},
    )

    games = schedule_games(
        decks,
        total_games=config.total_games,
        seed=config.seed,
        campaign_fingerprint=campaign_fingerprint,
    )
    rows = load_parts(
        parts_dir,
        games=games,
        campaign_fingerprint=campaign_fingerprint,
    )
    completed = {int(row["game_index"]) for row in rows}
    remaining = tuple(game for game in games if game.game_index not in completed)
    started_clock = time.perf_counter()
    resumed_games = len(rows)
    progress_path = output_dir / "progress.json"
    write_progress(
        progress_path,
        total_games=len(games),
        completed_games=len(rows),
        resumed_games=resumed_games,
        started_clock=started_clock,
        complete=not remaining,
    )

    if remaining:
        _execute_games(
            remaining,
            rows=rows,
            parts_dir=parts_dir,
            progress_path=progress_path,
            config=config,
            native_config=native_config,
            root=root,
            campaign_fingerprint=campaign_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            belief_fingerprint=belief_fingerprint,
            checkpoint_path=checkpoint_path,
            started_clock=started_clock,
            resumed_games=resumed_games,
        )

    rows = load_parts(
        parts_dir,
        games=games,
        campaign_fingerprint=campaign_fingerprint,
    )
    if len(rows) != len(games):
        raise RuntimeError(
            f"native deck Elo retained {len(rows)} games, expected {len(games)}"
        )
    summary = write_final_artifacts(
        config,
        rows=rows,
        output_dir=output_dir,
        campaign_fingerprint=campaign_fingerprint,
        runtime_fingerprint=runtime_fingerprint,
        belief_fingerprint=belief_fingerprint,
        elapsed_seconds=time.perf_counter() - started_clock,
        resumed_games=resumed_games,
    )
    write_progress(
        progress_path,
        total_games=len(games),
        completed_games=len(rows),
        resumed_games=resumed_games,
        started_clock=started_clock,
        complete=True,
    )
    return summary


def _execute_games(
    games: Sequence[ScheduledGame],
    *,
    rows: list[dict[str, Any]],
    parts_dir: Path,
    progress_path: Path,
    config: NativeDeckEloConfig,
    native_config: NativeMatchConfig,
    root: Path,
    campaign_fingerprint: str,
    runtime_fingerprint: str,
    belief_fingerprint: str,
    checkpoint_path: Path,
    started_clock: float,
    resumed_games: int,
) -> None:
    """Own the shared controller factory and per-lane native executors."""
    controllers = ControllerFactory(native_config, repo_root=root)
    executors: list[NativeMatchExecutor] = []
    try:
        for _ in range(min(config.concurrency, len(games))):
            executors.append(
                NativeMatchExecutor(
                    native_config,
                    repo_root=root,
                    controllers=controllers,
                )
            )
        _run_remaining(
            games,
            rows=rows,
            parts_dir=parts_dir,
            progress_path=progress_path,
            config=config,
            campaign_fingerprint=campaign_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
            belief_fingerprint=belief_fingerprint,
            checkpoint_path=checkpoint_path,
            executors=executors,
            started_clock=started_clock,
            resumed_games=resumed_games,
        )
    finally:
        for executor in executors:
            executor.close()
        controllers.close()


def _run_remaining(
    games: Sequence[ScheduledGame],
    *,
    rows: list[dict[str, Any]],
    parts_dir: Path,
    progress_path: Path,
    config: NativeDeckEloConfig,
    campaign_fingerprint: str,
    runtime_fingerprint: str,
    belief_fingerprint: str,
    checkpoint_path: Path,
    executors: Sequence[NativeMatchExecutor],
    started_clock: float,
    resumed_games: int,
) -> None:
    """Execute remaining games while sharing one checkpoint cache and batcher."""
    available: Queue[NativeMatchExecutor] = Queue()
    for executor in executors:
        available.put(executor)

    def execute(game: ScheduledGame) -> dict[str, Any]:
        executor = available.get()
        try:
            lease = _lease(
                game,
                checkpoint_path=checkpoint_path,
                runtime_fingerprint=runtime_fingerprint,
                belief_fingerprint=belief_fingerprint,
            )
            return result_row(
                game,
                executor.execute(lease),
                campaign_fingerprint=campaign_fingerprint,
            )
        finally:
            available.put(executor)

    pending_rows: list[dict[str, Any]] = []
    next_part = next_part_index(parts_dir)
    remaining_games = tuple(games)
    if resumed_games == 0:
        preflight_games = remaining_games[: min(2, len(remaining_games))]
        with ThreadPoolExecutor(
            max_workers=len(preflight_games),
            thread_name_prefix="native-deck-elo-preflight",
        ) as preflight_pool:
            preflight_rows = list(preflight_pool.map(execute, preflight_games))
        failed = [row for row in preflight_rows if row["terminal_reason"] != "normal"]
        if failed:
            reasons = ", ".join(str(row["terminal_reason"]) for row in failed)
            raise RuntimeError(
                f"native deck Elo preflight did not finish normally: {reasons}"
            )
        pending_rows.extend(preflight_rows)
        remaining_games = remaining_games[len(preflight_games) :]
    with ThreadPoolExecutor(
        max_workers=len(executors),
        thread_name_prefix="native-deck-elo",
    ) as pool:
        futures: tuple[Future[dict[str, Any]], ...] = tuple(
            pool.submit(execute, game) for game in remaining_games
        )
        for future in as_completed(futures):
            pending_rows.append(future.result())
            if len(pending_rows) >= config.flush_shard_games:
                write_part(
                    parts_dir / f"part-{next_part:06d}.parquet",
                    pending_rows,
                    compression=config.compression,
                )
                rows.extend(pending_rows)
                pending_rows.clear()
                next_part += 1
                write_progress(
                    progress_path,
                    total_games=config.total_games,
                    completed_games=len(rows),
                    resumed_games=resumed_games,
                    started_clock=started_clock,
                    complete=False,
                )
    if pending_rows:
        write_part(
            parts_dir / f"part-{next_part:06d}.parquet",
            pending_rows,
            compression=config.compression,
        )
        rows.extend(pending_rows)
        write_progress(
            progress_path,
            total_games=config.total_games,
            completed_games=len(rows),
            resumed_games=resumed_games,
            started_clock=started_clock,
            complete=False,
        )


def _load_decks(
    config: NativeDeckEloConfig,
    *,
    root: Path,
    controller_id: str,
) -> tuple[DeckAsset, ...]:
    output: list[DeckAsset] = []
    seen_digests: set[str] = set()
    for item in config.decks:
        path = resolve_file(item.path, root=root)
        canonical = canonicalize_deck(records.read_deck(path))
        if canonical.deck_digest in seen_digests:
            raise ValueError(f"duplicate exact deck: {canonical.deck_digest}")
        seen_digests.add(canonical.deck_digest)
        deck_hash = records.signature_hash(canonical.signature)
        output.append(
            DeckAsset(
                label=item.label,
                path=path,
                deck_digest=canonical.deck_digest,
                deck_hash=deck_hash,
                deck_signature=canonical.signature,
                bundle=BundleIdentity(
                    bundle_id=(
                        f"checkpoint:{config.expected_checkpoint_sha256[:16]}:"
                        f"deck:{deck_hash}"
                    ),
                    controller_id=controller_id,
                    deck_digest=canonical.deck_digest,
                ),
            )
        )
    return tuple(output)


def _lease(
    game: ScheduledGame,
    *,
    checkpoint_path: Path,
    runtime_fingerprint: str,
    belief_fingerprint: str,
) -> MatchLease:
    side_a = game.deck_a if game.deck_a_seat == 0 else game.deck_b
    side_b = game.deck_b if game.deck_a_seat == 0 else game.deck_a
    return MatchLease(
        match_id=game.match_id,
        side_a=side_a.bundle,
        side_b=side_b.bundle,
        side_a_controller_kind="checkpoint",
        side_b_controller_kind="checkpoint",
        side_a_controller_path=checkpoint_path,
        side_b_controller_path=checkpoint_path,
        side_a_deck_path=side_a.path,
        side_b_deck_path=side_b.path,
        requires_cuda=True,
        runtime_fingerprint=runtime_fingerprint,
        belief_fingerprint=belief_fingerprint,
        lease_expires_at=datetime.now(UTC).isoformat(),
    )


__all__ = ["run_native_deck_elo"]
