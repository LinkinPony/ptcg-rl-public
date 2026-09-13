"""Board and card-value primitives for the native scripted heuristic."""

from __future__ import annotations

from collections import Counter

from ptcg_rl.rl.native_scripted_catalog import (
    MIXED75_71EB_CARD_COUNTS,
    NativeAttackSemantics,
    NativeCardSemantics,
    NativeScriptedCatalog,
)
from ptcg_rl.rl.native_scripted_state import NativeScriptedRow

_CARD_TYPE_POKEMON = 0
_CARD_TYPE_ITEM = 1
_CARD_TYPE_TOOL = 2
_CARD_TYPE_SUPPORTER = 3
_CARD_TYPE_STADIUM = 4
_CARD_TYPE_BASIC_ENERGY = 5
_CARD_TYPE_SPECIAL_ENERGY = 6
_ENERGY_COLORLESS = 0
_ENERGY_RAINBOW = 10
_DECK_COUNTS = Counter(dict(MIXED75_71EB_CARD_COUNTS))


class NativeHeuristicBoard:
    """Shared low-level calculations over one validated native row."""

    def __init__(
        self,
        row: NativeScriptedRow,
        catalog: NativeScriptedCatalog,
    ) -> None:
        self.row = row
        self.catalog = catalog

    def _minimum_attack_shortfall(self, pokemon: int) -> int:
        card = self.catalog.card(self.row.pokemon_id(pokemon))
        if not card.attacks:
            return 99
        energies = self.row.pokemon_energies(pokemon)
        return min(
            self._energy_shortfall(attack, energies)
            for attack in card.attacks
        )

    def _best_attack_value(
        self,
        pokemon: int,
        energies: tuple[int, ...] | None = None,
    ) -> float:
        card = self.catalog.card(self.row.pokemon_id(pokemon))
        available = (
            self.row.pokemon_energies(pokemon)
            if energies is None
            else energies
        )
        best = 0.0
        for attack in card.attacks:
            shortfall = self._energy_shortfall(attack, available)
            damage = self._estimated_attack_damage(attack, pokemon)
            value = damage - shortfall * 70.0
            text = attack.text.lower()
            if "search your deck" in text:
                value += 35.0
            if "takes 30 less damage" in text:
                value += 25.0
            best = max(best, value)
        return best

    @staticmethod
    def _energy_shortfall(
        attack: NativeAttackSemantics,
        energies: tuple[int, ...],
    ) -> int:
        available = Counter(energies)
        missing = 0
        colorless_need = 0
        for need in attack.energies:
            if need == _ENERGY_COLORLESS:
                colorless_need += 1
            elif available[need] > 0:
                available[need] -= 1
            elif available[_ENERGY_RAINBOW] > 0:
                available[_ENERGY_RAINBOW] -= 1
            else:
                missing += 1
        missing += max(0, colorless_need - sum(available.values()))
        return missing

    def _estimated_attack_damage(
        self,
        attack: NativeAttackSemantics,
        attacker: int | None,
    ) -> float:
        text = attack.text.lower()
        damage = float(attack.damage)
        if "for each basic {w} energy card in your discard pile" in text:
            damage = max(damage, float(self._discard_count(3) * 20))
        if (
            "discard the top 6 cards of your deck" in text
            and "100 damage" in text
        ):
            energy_ratio = self._estimated_remaining_deck_energy_ratio(3)
            expected = (
                min(6, self.row.player_deck_count(self.row.perspective))
                * energy_ratio
                * 100.0
            )
            damage = max(damage, expected)
        if "this attack does 50 more damage" in text:
            damage += 50.0
        if attacker is not None:
            for tool_id in self.row.pokemon_attachment_ids(
                attacker,
                kind=2,
            ):
                if (
                    self.catalog.is_maximum_belt(tool_id)
                    and self._opponent_active_is_ex()
                ):
                    damage += 50.0
        return damage

    def _apply_weakness_resistance(
        self,
        damage: float,
        attacker: int | None,
        target: int,
    ) -> float:
        if attacker is None or damage <= 0:
            return damage
        attacker_card = self.catalog.card(self.row.pokemon_id(attacker))
        target_card = self.catalog.card(self.row.pokemon_id(target))
        if target_card.weakness == attacker_card.energy_type:
            damage *= 2
        if target_card.resistance == attacker_card.energy_type:
            damage = max(0.0, damage - 30.0)
        return damage

    def _pokemon_board_value(self, pokemon: int | None) -> float:
        if pokemon is None:
            return 0.0
        card = self.catalog.card(self.row.pokemon_id(pokemon))
        value = (
            self.row.pokemon_hp(pokemon) * 0.35
            + len(self.row.pokemon_energies(pokemon)) * 18.0
            + self._best_attack_value(pokemon) * 0.18
            + self.row.pokemon_max_hp(pokemon) * 0.08
        )
        if card.ex:
            value += 8.0
        if card.mega_ex:
            value += 20.0
        if self.row.is_your_active(pokemon):
            value += 8.0
        return value

    def _card_keep_value(self, card_id: int | None) -> float:
        if card_id is None:
            return 0.0
        card = self.catalog.card(card_id)
        if card.card_type == _CARD_TYPE_POKEMON:
            value = self._pokemon_card_value(card) * 0.10
            if card.mega_ex and self._has_evolution_source(card):
                value += 35.0
            if card.basic and self._has_evolution_target(card.name):
                value += 16.0
            return value
        if card.card_type in (
            _CARD_TYPE_BASIC_ENERGY,
            _CARD_TYPE_SPECIAL_ENERGY,
        ):
            return 18.0 if self._needs_energy_on_board() else 8.0
        if card.card_type == _CARD_TYPE_TOOL:
            return 20.0
        if card.card_type in (
            _CARD_TYPE_ITEM,
            _CARD_TYPE_SUPPORTER,
            _CARD_TYPE_STADIUM,
        ):
            return self._score_trainer_play(card) * 0.9
        return 1.0

    def _card_discard_cost(self, card_id: int | None) -> float:
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        value = self._card_keep_value(card_id)
        if card.card_type in (
            _CARD_TYPE_BASIC_ENERGY,
            _CARD_TYPE_SPECIAL_ENERGY,
        ):
            active = self._your_active()
            active_card = (
                self.catalog.card(self.row.pokemon_id(active))
                if active is not None
                else None
            )
            if active_card is not None and active_card.name.lower() == "kyogre":
                value -= 10.0
        return value

    def _has_pokemon_named(self, name: str) -> bool:
        return any(
            self.catalog.own_card(self.row.pokemon_id(pokemon)).name == name
            for pokemon in self._your_pokemon()
        )

    def _has_evolution_source(self, evolved: NativeCardSemantics) -> bool:
        return bool(
            evolved.evolves_from
            and self._has_pokemon_named(evolved.evolves_from)
        )

    @staticmethod
    def _has_evolution_target(name: str) -> bool:
        return name == "Dwebble"

    def _need_mega_piece(self) -> bool:
        has_source = any(
            self._has_evolution_target(
                self.catalog.own_card(self.row.pokemon_id(pokemon)).name
            )
            for pokemon in self._your_pokemon()
        )
        has_mega = any(
            self.catalog.own_card(card_id).mega_ex
            for card_id in self.row.hand_card_ids(self.row.perspective)
        )
        return has_source and not has_mega

    def _need_ex_piece(self) -> bool:
        strong = [
            pokemon
            for pokemon in self._your_pokemon()
            if (
                self.catalog.own_card(self.row.pokemon_id(pokemon)).ex
                or self.catalog.own_card(
                    self.row.pokemon_id(pokemon)
                ).mega_ex
            )
        ]
        hand_ex = [
            card_id
            for card_id in self.row.hand_card_ids(self.row.perspective)
            if (
                self.catalog.own_card(card_id).ex
                or self.catalog.own_card(card_id).mega_ex
            )
        ]
        return len(strong) + len(hand_ex) < 2

    def _needs_energy_on_board(self) -> bool:
        return any(
            self._minimum_attack_shortfall(pokemon) > 0
            for pokemon in self._your_pokemon()
        )

    def _your_pokemon(self) -> tuple[int, ...]:
        return self.row.pokemon_rows(self.row.perspective)

    def _your_active(self) -> int | None:
        return self.row.active_row(self.row.perspective)

    def _opponent_active(self) -> int | None:
        return self.row.active_row(self.row.opponent)

    def _energy_in_hand_count(self) -> int:
        return sum(
            1
            for card_id in self.row.hand_card_ids(self.row.perspective)
            if self.catalog.own_card(card_id).card_type
            in (_CARD_TYPE_BASIC_ENERGY, _CARD_TYPE_SPECIAL_ENERGY)
        )

    def _discard_count(self, card_id: int) -> int:
        return sum(
            int(discarded == card_id)
            for discarded in self.row.discard_card_ids(self.row.perspective)
        )

    def _estimated_remaining_deck_energy_ratio(
        self,
        energy_card_id: int,
    ) -> float:
        known: Counter[int] = Counter()
        known.update(self.row.discard_card_ids(self.row.perspective))
        known.update(self.row.hand_card_ids(self.row.perspective))
        for pokemon in self._your_pokemon():
            known[self.row.pokemon_id(pokemon)] += 1
            known.update(self.row.pokemon_attachment_ids(pokemon, kind=1))
            known.update(self.row.pokemon_attachment_ids(pokemon, kind=2))
            known.update(self.row.pokemon_attachment_ids(pokemon, kind=3))
        remaining = max(
            0,
            _DECK_COUNTS[energy_card_id] - known[energy_card_id],
        )
        deck_count = self.row.player_deck_count(self.row.perspective)
        return max(0.05, min(0.95, remaining / max(1, deck_count)))

    def _opponent_active_is_ex(self) -> bool:
        active = self._opponent_active()
        if active is None:
            return False
        card = self.catalog.card(self.row.pokemon_id(active))
        return card.ex or card.mega_ex

    def _first_option_index(self, option_type: int) -> int | None:
        for index in range(self.row.option_count):
            if self.row.option(index).option_type == option_type:
                return index
        return None

    @staticmethod
    def _pokemon_card_value(card: NativeCardSemantics) -> float:
        best_damage = 0.0
        best_attack_bonus = 0.0
        for attack in card.attacks:
            damage = float(attack.damage)
            text = attack.text.lower()
            if "100 damage for each basic {w} energy" in text:
                damage = max(damage, 260.0)
            if "for each basic {w} energy card in your discard pile" in text:
                damage = max(damage, 120.0)
            if "search your deck" in text:
                best_attack_bonus = max(best_attack_bonus, 35.0)
            if "takes 30 less damage" in text:
                best_attack_bonus = max(best_attack_bonus, 25.0)
            best_damage = max(best_damage, damage)
        value = (
            card.hp * 0.45
            + best_damage
            + best_attack_bonus
            - card.retreat_cost * 5.0
        )
        if card.basic:
            value += 18.0
        if card.stage1:
            value += 8.0
        if card.stage2:
            value -= 8.0
        if card.ex:
            value += 20.0
        if card.mega_ex:
            value += 40.0
        return value

    def _score_trainer_play(self, card: NativeCardSemantics) -> float:
        raise NotImplementedError


__all__ = ["NativeHeuristicBoard"]
