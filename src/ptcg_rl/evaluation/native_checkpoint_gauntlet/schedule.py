"""Deterministic balanced scheduling for cross-checkpoint roster evaluation."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Sequence

from ptcg_rl.evaluation.native_checkpoint_gauntlet.models import (
    ScheduledCrossCheckpointGame,
)
from ptcg_rl.evaluation.native_deck_elo.models import DeckAsset


def schedule_cross_checkpoint_games(
    candidate_decks: Sequence[DeckAsset],
    baseline_decks: Sequence[DeckAsset],
    *,
    total_games: int,
    seed: int,
    campaign_fingerprint: str,
    match_seed_namespace: str | None = None,
) -> tuple[ScheduledCrossCheckpointGame, ...]:
    """Cover every ordered deck cell with evenly repeated mirrored seat blocks.

    A shared ``match_seed_namespace`` makes aligned campaigns emit identical
    match IDs, and therefore identical engine and controller seeds, while their
    campaign fingerprints and result ownership remain distinct. Its caller is
    responsible for ensuring every other seed-relevant input is identical.
    """
    cells = tuple(
        (candidate, baseline)
        for candidate in candidate_decks
        for baseline in baseline_decks
    )
    if not cells or total_games < len(cells) * 2 or total_games % 2:
        raise ValueError("cross-checkpoint schedule cannot cover all mirrored cells")
    block_count = total_games // 2
    rng = random.Random(seed)
    blocks: list[tuple[DeckAsset, DeckAsset]] = []
    while len(blocks) < block_count:
        cycle = list(cells)
        rng.shuffle(cycle)
        blocks.extend(cycle[: block_count - len(blocks)])

    games: list[ScheduledCrossCheckpointGame] = []
    seed_identity = (
        campaign_fingerprint
        if match_seed_namespace is None
        else f"match-seed-namespace\0{match_seed_namespace}\0{seed}"
    )
    for block_index, (candidate, baseline) in enumerate(blocks):
        for candidate_seat in (0, 1):
            game_index = len(games)
            identity = (
                f"{seed_identity}\0{block_index}\0{candidate.deck_digest}\0"
                f"{baseline.deck_digest}\0{candidate_seat}"
            ).encode()
            match_id = f"ncg-{hashlib.sha256(identity).hexdigest()[:40]}"
            games.append(
                ScheduledCrossCheckpointGame(
                    game_index=game_index,
                    match_id=match_id,
                    candidate_deck=candidate,
                    baseline_deck=baseline,
                    candidate_seat=candidate_seat,
                )
            )
    return tuple(games)


__all__ = ["schedule_cross_checkpoint_games"]
