"""Exact-game scheduling for bounded native deck Elo campaigns."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Literal, cast

from ptcg_rl.evaluation.native_deck_elo.models import DeckAsset, ScheduledGame


def schedule_games(
    decks: Sequence[DeckAsset],
    *,
    total_games: int,
    seed: int,
    campaign_fingerprint: str,
) -> tuple[ScheduledGame, ...]:
    """Create exact, mirrored, full-coverage games with balanced extras."""
    pairs = tuple(
        (left, right)
        for left in range(len(decks))
        for right in range(left + 1, len(decks))
    )
    total_blocks = total_games // 2
    base_blocks = total_blocks // len(pairs)
    extra_blocks = total_blocks - base_blocks * len(pairs)
    extra_pairs = round_robin_pairs(len(decks), seed=seed)[:extra_blocks]
    blocks = [pair for pair in pairs for _ in range(base_blocks)]
    blocks.extend(extra_pairs)
    rng = random.Random(seed)
    rng.shuffle(blocks)
    games: list[ScheduledGame] = []
    campaign_id = campaign_fingerprint[:20]
    for block_index, (left, right) in enumerate(blocks):
        first_seat = cast(Literal[0, 1], block_index % 2)
        for offset in range(2):
            deck_a_seat = cast(Literal[0, 1], first_seat ^ offset)
            game_index = len(games)
            games.append(
                ScheduledGame(
                    game_index=game_index,
                    match_id=f"deck-elo:{campaign_id}:g{game_index:06d}",
                    deck_a=decks[left],
                    deck_b=decks[right],
                    deck_a_seat=deck_a_seat,
                )
            )
    if len(games) != total_games:
        raise RuntimeError("native deck Elo scheduler produced the wrong game count")
    return tuple(games)


def round_robin_pairs(deck_count: int, *, seed: int) -> list[tuple[int, int]]:
    """Return every unordered pair once in degree-balanced round order."""
    players = list(range(deck_count))
    random.Random(seed + 1_947).shuffle(players)
    if len(players) % 2:
        players.append(-1)
    pairs: list[tuple[int, int]] = []
    for _ in range(len(players) - 1):
        for index in range(len(players) // 2):
            left = players[index]
            right = players[-1 - index]
            if left >= 0 and right >= 0:
                pairs.append((min(left, right), max(left, right)))
        players = [players[0], players[-1], *players[1:-1]]
    expected = deck_count * (deck_count - 1) // 2
    if len(pairs) != expected or len(set(pairs)) != expected:
        raise RuntimeError("round-robin pair generator lost pair coverage")
    return pairs


__all__ = ["round_robin_pairs", "schedule_games"]
