"""Structured effect consequence data types."""

from __future__ import annotations

from dataclasses import dataclass

from ptcg_rl.engine.constants import SpecialCondition


@dataclass(frozen=True, order=True)
class CardRef:
    """Stable visible card instance key within one game."""

    player_index: int
    card_id: int
    serial: int


@dataclass(frozen=True)
class BoardPosition:
    """Visible board location for a card or Pokemon."""

    area: int
    index: int | None = None


@dataclass(frozen=True)
class PokemonSnapshot:
    """Small immutable snapshot of one visible Pokemon in play."""

    ref: CardRef
    hp: int
    max_hp: int
    position: BoardPosition


@dataclass(frozen=True)
class TargetAmount:
    """Aggregated amount for one visible target."""

    target: CardRef
    amount: int


@dataclass(frozen=True)
class HpChange:
    """One engine ``HP_CHANGE`` event grounded against state when available."""

    target: CardRef
    raw_value: int
    damage: int
    healing: int
    put_damage_counter: bool
    before: PokemonSnapshot | None = None
    after: PokemonSnapshot | None = None


@dataclass(frozen=True)
class StatusChange:
    """Special-condition application or recovery."""

    target: CardRef
    condition: SpecialCondition
    recovered: bool


@dataclass(frozen=True)
class CoinFlip:
    """One engine coin result."""

    player_index: int | None
    head: bool


@dataclass(frozen=True)
class CardMove:
    """Visible or hidden card-zone movement."""

    player_index: int | None
    card: CardRef | None
    from_area: int | None
    to_area: int | None
    hidden: bool


@dataclass(frozen=True)
class DrawEvent:
    """Visible or hidden draw event."""

    player_index: int | None
    card: CardRef | None
    hidden: bool


@dataclass(frozen=True)
class AttackEvent:
    """Attack declaration event."""

    attacker: CardRef
    attack_id: int


@dataclass(frozen=True)
class AttachmentEvent:
    """Attach event emitted by the engine."""

    player_index: int
    attached: CardRef
    target: CardRef


@dataclass(frozen=True)
class EvolutionEvent:
    """Evolution or devolution event emitted by the engine."""

    player_index: int
    card: CardRef
    target: CardRef
    devolve: bool = False


@dataclass(frozen=True)
class SwitchEvent:
    """Active/bench switch event."""

    player_index: int
    active_to_bench: CardRef
    bench_to_active: CardRef


@dataclass(frozen=True)
class MatchResult:
    """Terminal result log."""

    result: int
    reason: int | None


@dataclass(frozen=True)
class Knockout:
    """Visible Pokemon moved from play to discard."""

    target: CardRef
    from_area: int


@dataclass(frozen=True)
class EffectSummary:
    """Structured consequence summary for one resolved engine transition."""

    hp_changes: tuple[HpChange, ...] = ()
    damage_by_target: tuple[TargetAmount, ...] = ()
    healing_by_target: tuple[TargetAmount, ...] = ()
    status_changes: tuple[StatusChange, ...] = ()
    coins: tuple[CoinFlip, ...] = ()
    card_moves: tuple[CardMove, ...] = ()
    draws: tuple[DrawEvent, ...] = ()
    attacks: tuple[AttackEvent, ...] = ()
    attachments: tuple[AttachmentEvent, ...] = ()
    evolutions: tuple[EvolutionEvent, ...] = ()
    switches: tuple[SwitchEvent, ...] = ()
    knockouts: tuple[Knockout, ...] = ()
    result: MatchResult | None = None
    draws_by_player: tuple[int, int] = (0, 0)
    hidden_draws_by_player: tuple[int, int] = (0, 0)
    prizes_taken_by_player: tuple[int, int] = (0, 0)
    energy_delta_by_player: tuple[int, int] = (0, 0)
    unknown_log_types: tuple[int, ...] = ()

    def damage_to(self, target: CardRef | None) -> int:
        """Return total grounded damage to ``target``."""
        if target is None:
            return 0
        return _amount_for(self.damage_by_target, target)

    def healing_to(self, target: CardRef | None) -> int:
        """Return total grounded healing to ``target``."""
        if target is None:
            return 0
        return _amount_for(self.healing_by_target, target)

    def knocked_out(self, target: CardRef | None) -> bool:
        """Whether ``target`` was visibly knocked out in this transition."""
        if target is None:
            return False
        return any(knockout.target == target for knockout in self.knockouts)


def _amount_for(amounts: tuple[TargetAmount, ...], target: CardRef) -> int:
    for amount in amounts:
        if amount.target == target:
            return amount.amount
    return 0
