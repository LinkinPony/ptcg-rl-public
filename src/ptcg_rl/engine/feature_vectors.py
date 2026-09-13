"""Network-facing dynamic effect feature vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from ptcg_rl.engine.constants import AreaType, SpecialCondition
from ptcg_rl.engine.effect_types import EffectSummary
from ptcg_rl.engine.effects import (
    active_pokemon_ref,
    bench_pokemon_refs,
)

MAX_DAMAGE_FEATURE = 400.0
MAX_BENCH_DAMAGE_FEATURE = 1_000.0
MAX_BENCH_COUNT_FEATURE = 5.0
MAX_DRAW_FEATURE = 10.0
MAX_ENERGY_DELTA_FEATURE = 10.0
MAX_PRIZE_FEATURE = 6.0
MAX_COIN_FEATURE = 16.0
MAX_DISCARD_INFLOW_FEATURE = 20.0
MAX_KO_COUNT_FEATURE = 6.0

DYNAMIC_EFFECT_FEATURE_NAMES: tuple[str, ...] = (
    "opponent_active_damage_norm",
    "opponent_active_ko",
    "opponent_bench_total_damage_norm",
    "opponent_bench_max_damage_norm",
    "opponent_bench_damaged_count_norm",
    "self_active_damage_norm",
    "self_active_ko",
    "self_bench_total_damage_norm",
    "status_poisoned",
    "status_burned",
    "status_asleep",
    "status_paralyzed",
    "status_confused",
    "cards_drawn_norm",
    "energy_delta_norm",
    "prizes_taken_norm",
    "coin_count_norm",
    "coin_heads_norm",
    "terminal_win",
    "terminal_loss",
    "terminal_draw",
    "self_status_poisoned",
    "self_status_burned",
    "self_status_asleep",
    "self_status_paralyzed",
    "self_status_confused",
    "self_active_healing_norm",
    "opponent_energy_delta_norm",
    "opponent_cards_drawn_norm",
    "self_discard_inflow_norm",
    "opponent_discard_inflow_norm",
    "self_visible_ko_count_norm",
    "opponent_visible_ko_count_norm",
)
DYNAMIC_EFFECT_FEATURE_SIZE = len(DYNAMIC_EFFECT_FEATURE_NAMES)
_DYNAMIC_EFFECT_BINARY_NAMES = frozenset(
    {
        "opponent_active_ko",
        "self_active_ko",
        "status_poisoned",
        "status_burned",
        "status_asleep",
        "status_paralyzed",
        "status_confused",
        "terminal_win",
        "terminal_loss",
        "terminal_draw",
        "self_status_poisoned",
        "self_status_burned",
        "self_status_asleep",
        "self_status_paralyzed",
        "self_status_confused",
    }
)
DYNAMIC_EFFECT_BINARY_INDICES = tuple(
    index
    for index, name in enumerate(DYNAMIC_EFFECT_FEATURE_NAMES)
    if name in _DYNAMIC_EFFECT_BINARY_NAMES
)
DYNAMIC_EFFECT_MAGNITUDE_INDICES = tuple(
    index
    for index, name in enumerate(DYNAMIC_EFFECT_FEATURE_NAMES)
    if name not in _DYNAMIC_EFFECT_BINARY_NAMES
)
StatusFlagVector = tuple[float, float, float, float, float]


@dataclass(frozen=True)
class DynamicEffectFeatureRow:
    """Network-facing dynamic effect feature row for one candidate action."""

    select: tuple[int, ...]
    vector: tuple[float, ...]
    option_types: tuple[int, ...] = ()
    attack_ids: tuple[int, ...] = ()
    card_ids: tuple[int, ...] = ()
    summary: EffectSummary | None = None

    def to_numpy(self) -> npt.NDArray[np.float32]:
        """Return ``float32[DYNAMIC_EFFECT_FEATURE_SIZE]``."""
        if len(self.vector) != DYNAMIC_EFFECT_FEATURE_SIZE:
            raise ValueError("dynamic effect feature vector has invalid width")
        return np.asarray(self.vector, dtype=np.float32)


def make_dynamic_effect_feature_row(
    *,
    select: tuple[int, ...],
    option_types: tuple[int, ...],
    attack_ids: tuple[int, ...],
    card_ids: tuple[int, ...],
    summary: EffectSummary,
    before_state: Any | None,
    after_state: Any | None,
    perspective_player: int | None = None,
) -> DynamicEffectFeatureRow:
    """Build one fixed-width feature row from a parsed transition."""
    perspective = _perspective_player(before_state, perspective_player)
    opponent = 1 - perspective
    opponent_active = active_pokemon_ref(before_state, opponent)
    self_active = active_pokemon_ref(before_state, perspective)
    opponent_bench = bench_pokemon_refs(before_state, opponent)
    self_bench = bench_pokemon_refs(before_state, perspective)

    opponent_bench_damage = [summary.damage_to(ref) for ref in opponent_bench]
    self_bench_damage = [summary.damage_to(ref) for ref in self_bench]
    terminal_win, terminal_loss, terminal_draw = _terminal_features(
        after_state,
        perspective,
    )
    self_discard_inflow, opponent_discard_inflow = _discard_inflows(
        summary,
        perspective,
    )
    self_ko_count, opponent_ko_count = _visible_ko_counts(summary, perspective)
    vector = (
        _norm(summary.damage_to(opponent_active), MAX_DAMAGE_FEATURE),
        float(summary.knocked_out(opponent_active)),
        _norm(sum(opponent_bench_damage), MAX_BENCH_DAMAGE_FEATURE),
        _norm(max(opponent_bench_damage, default=0), MAX_DAMAGE_FEATURE),
        _norm(
            sum(1 for amount in opponent_bench_damage if amount > 0),
            MAX_BENCH_COUNT_FEATURE,
        ),
        _norm(summary.damage_to(self_active), MAX_DAMAGE_FEATURE),
        float(summary.knocked_out(self_active)),
        _norm(sum(self_bench_damage), MAX_BENCH_DAMAGE_FEATURE),
        *_status_flags(summary, opponent_active),
        _norm(summary.draws_by_player[perspective], MAX_DRAW_FEATURE),
        _signed_norm(
            summary.energy_delta_by_player[perspective], MAX_ENERGY_DELTA_FEATURE
        ),
        _norm(summary.prizes_taken_by_player[perspective], MAX_PRIZE_FEATURE),
        _norm(len(summary.coins), MAX_COIN_FEATURE),
        _norm(sum(1 for coin in summary.coins if coin.head), MAX_COIN_FEATURE),
        terminal_win,
        terminal_loss,
        terminal_draw,
        *_status_flags(summary, self_active),
        _norm(summary.healing_to(self_active), MAX_DAMAGE_FEATURE),
        _signed_norm(
            summary.energy_delta_by_player[opponent],
            MAX_ENERGY_DELTA_FEATURE,
        ),
        _norm(summary.draws_by_player[opponent], MAX_DRAW_FEATURE),
        _norm(self_discard_inflow, MAX_DISCARD_INFLOW_FEATURE),
        _norm(opponent_discard_inflow, MAX_DISCARD_INFLOW_FEATURE),
        _norm(self_ko_count, MAX_KO_COUNT_FEATURE),
        _norm(opponent_ko_count, MAX_KO_COUNT_FEATURE),
    )
    if len(vector) != DYNAMIC_EFFECT_FEATURE_SIZE:
        raise ValueError("internal dynamic feature width mismatch")
    return DynamicEffectFeatureRow(
        select=select,
        vector=tuple(float(value) for value in vector),
        option_types=option_types,
        attack_ids=attack_ids,
        card_ids=card_ids,
        summary=summary,
    )


def build_dynamic_effect_feature_table(
    rows: Sequence[DynamicEffectFeatureRow],
) -> npt.NDArray[np.float32]:
    """Stack dynamic feature rows into ``float32[N, D]``."""
    if not rows:
        return np.zeros((0, DYNAMIC_EFFECT_FEATURE_SIZE), dtype=np.float32)
    return np.stack([row.to_numpy() for row in rows]).astype(np.float32, copy=False)


def _perspective_player(
    state: Any | None,
    perspective_player: int | None,
) -> int:
    if perspective_player is not None:
        if perspective_player not in {0, 1}:
            raise ValueError("perspective_player must be 0 or 1")
        return perspective_player
    if state is None:
        return 0
    return _int_field(state, "yourIndex", 0)


def _status_flags(
    summary: EffectSummary,
    opponent_active: object | None,
) -> StatusFlagVector:
    flags = [0.0] * len(SpecialCondition)
    if opponent_active is None:
        return cast(StatusFlagVector, tuple(flags))
    for change in summary.status_changes:
        if not change.recovered and change.target == opponent_active:
            flags[int(change.condition)] = 1.0
    return cast(StatusFlagVector, tuple(flags))


def _terminal_features(
    state: Any | None,
    perspective_player: int,
) -> tuple[float, float, float]:
    result = _int_field(state, "result", -1) if state is not None else -1
    if result < 0:
        return 0.0, 0.0, 0.0
    if result == 2:
        return 0.0, 0.0, 1.0
    if result == perspective_player:
        return 1.0, 0.0, 0.0
    return 0.0, 1.0, 0.0


def _discard_inflows(summary: EffectSummary, perspective: int) -> tuple[int, int]:
    counts = [0, 0]
    for move in summary.card_moves:
        if move.player_index in (0, 1) and move.to_area == int(AreaType.DISCARD):
            counts[int(move.player_index)] += 1
    return counts[perspective], counts[1 - perspective]


def _visible_ko_counts(summary: EffectSummary, perspective: int) -> tuple[int, int]:
    self_count = 0
    opponent_count = 0
    for knockout in summary.knockouts:
        if knockout.target.player_index == perspective:
            self_count += 1
        elif knockout.target.player_index == 1 - perspective:
            opponent_count += 1
    return self_count, opponent_count


def _norm(value: int | float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("normalization denominator must be positive")
    return min(max(float(value), 0.0), denominator) / denominator


def _signed_norm(value: int | float, denominator: float) -> float:
    if denominator <= 0.0:
        raise ValueError("normalization denominator must be positive")
    clipped = min(max(float(value), -denominator), denominator)
    return clipped / denominator


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _int_field(value: Any, name: str, default: int) -> int:
    field_value = _field(value, name, default)
    return int(field_value) if field_value is not None else default
