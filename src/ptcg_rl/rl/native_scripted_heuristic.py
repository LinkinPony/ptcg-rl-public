"""Exact native-column port of the frozen generic heuristic chooser."""

from __future__ import annotations

from ptcg_rl.engine.constants import OptionType
from ptcg_rl.rl.native_scripted_catalog import NativeCardSemantics
from ptcg_rl.rl.native_scripted_heuristic_board import NativeHeuristicBoard
from ptcg_rl.rl.native_scripted_state import NativeScriptedOption

_CARD_TYPE_POKEMON = 0
_CARD_TYPE_ITEM = 1
_CARD_TYPE_TOOL = 2
_CARD_TYPE_SUPPORTER = 3
_CARD_TYPE_STADIUM = 4
_CARD_TYPE_BASIC_ENERGY = 5
_CARD_TYPE_SPECIAL_ENERGY = 6
_ENERGY_WATER = 3

_SELECT_COUNT = 8
_SELECT_YES_NO = 9

_CONTEXT_MAIN = 0
_CONTEXT_SETUP_ACTIVE = 1
_CONTEXT_SETUP_BENCH = 2
_CONTEXT_SWITCH = 3
_CONTEXT_TO_ACTIVE = 4
_CONTEXT_TO_BENCH = 5
_CONTEXT_TO_FIELD = 6
_CONTEXT_TO_HAND = 7
_CONTEXT_DISCARD = 8
_CONTEXT_TO_DECK = 9
_CONTEXT_TO_DECK_BOTTOM = 10
_CONTEXT_TO_PRIZE = 11
_CONTEXT_NOT_MOVE = 12
_CONTEXT_DAMAGE_COUNTER = 13
_CONTEXT_DAMAGE_COUNTER_ANY = 14
_CONTEXT_DAMAGE = 15
_CONTEXT_REMOVE_DAMAGE_COUNTER = 16
_CONTEXT_HEAL = 17
_CONTEXT_EVOLVES_FROM = 18
_CONTEXT_EVOLVES_TO = 19
_CONTEXT_ATTACH_FROM = 21
_CONTEXT_ATTACH_TO = 22
_CONTEXT_LOOK = 24
_CONTEXT_EFFECT_TARGET = 25
_CONTEXT_DISCARD_ENERGY_CARD = 26
_CONTEXT_DISCARD_ENERGY = 30
_CONTEXT_TO_DECK_ENERGY = 32
_CONTEXT_ATTACK = 35
_CONTEXT_EVOLVE = 37
_CONTEXT_REMOVE_DAMAGE_COUNTER_COUNT = 40
_CONTEXT_IS_FIRST = 41
_CONTEXT_MULLIGAN = 42
_CONTEXT_ACTIVATE = 43
_CONTEXT_FIRST_EFFECT = 44
_CONTEXT_COIN_HEAD = 46

class NativeHeuristicChooser(NativeHeuristicBoard):
    """Score one native arena row with the immutable third-party algorithm."""

    def choose(self) -> tuple[int, ...]:
        """Choose indices satisfying the current min/max constraint."""
        if self.row.option_count <= 0:
            return ()
        if self.row.select_type == _SELECT_YES_NO:
            return self._choose_yes_no()
        if self.row.select_type == _SELECT_COUNT:
            return self._choose_count()

        scored = [
            (self.score_option(self.row.option(index)), index)
            for index in range(self.row.option_count)
        ]
        scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
        if self.row.context == _CONTEXT_MAIN:
            return (self._choose_main(scored),)

        count = self.row.maximum
        if self.row.minimum == 0:
            useful = [
                (score, index) for score, index in scored if score > 0.05
            ]
            if useful:
                count = min(
                    self.row.maximum,
                    max(self.row.minimum, len(useful)),
                )
            else:
                count = 0
        else:
            count = max(
                self.row.minimum,
                min(self.row.maximum, len(scored)),
            )
        return tuple(sorted(index for _score, index in scored[:count]))

    def score_option(self, option: NativeScriptedOption) -> float:
        """Return the exact frozen heuristic score for one legal option."""
        context_score = self._score_context_option(option)
        if context_score is not None:
            return context_score

        option_type = option.option_type
        if option_type == int(OptionType.PLAY):
            return self._score_play(option)
        if option_type == int(OptionType.ATTACH):
            return self._score_attach(option)
        if option_type == int(OptionType.EVOLVE):
            return self._score_evolve(option)
        if option_type == int(OptionType.ABILITY):
            return self._score_ability(option)
        if option_type == int(OptionType.ATTACK):
            return self._score_attack_option(option)
        if option_type == int(OptionType.RETREAT):
            return self._score_retreat()
        if option_type == int(OptionType.DISCARD):
            return -self._card_discard_cost(
                self.row.card_id_from_option(option)
            )
        if option_type == int(OptionType.END):
            return 0.0
        if option_type in {
            int(OptionType.CARD),
            int(OptionType.TOOL_CARD),
            int(OptionType.ENERGY_CARD),
        }:
            return self._score_card_choice(option)
        if option_type == int(OptionType.ENERGY):
            return self._score_energy_choice(option)
        if option_type == int(OptionType.SKILL):
            return self._score_skill_choice(option)
        if option_type == int(OptionType.NUMBER):
            return float(option.number or 0)
        return 0.0

    def _choose_main(self, scored: list[tuple[float, int]]) -> int:
        end_index = self._first_option_index(int(OptionType.END))
        best_score, best_index = scored[0]
        best_type = self.row.option(best_index).option_type

        non_terminal = [
            (score, index)
            for score, index in scored
            if self.row.option(index).option_type
            not in (int(OptionType.ATTACK), int(OptionType.END))
        ]
        if non_terminal and non_terminal[0][0] >= 15.0:
            return non_terminal[0][1]

        attacks = [
            (score, index)
            for score, index in scored
            if self.row.option(index).option_type == int(OptionType.ATTACK)
        ]
        if attacks and attacks[0][0] >= max(18.0, best_score - 6.0):
            return attacks[0][1]
        if best_type == int(OptionType.END) and end_index is not None:
            return end_index
        if best_score > 1.0:
            return best_index
        if attacks:
            return attacks[0][1]
        if end_index is not None:
            return end_index
        return best_index

    def _choose_yes_no(self) -> tuple[int, ...]:
        yes = self._first_option_index(int(OptionType.YES))
        no = self._first_option_index(int(OptionType.NO))
        choose_yes = True
        if self.row.context == _CONTEXT_IS_FIRST:
            choose_yes = True
        elif self.row.context == _CONTEXT_MULLIGAN:
            choose_yes = self._should_mulligan()
        elif self.row.context in (
            _CONTEXT_ACTIVATE,
            _CONTEXT_FIRST_EFFECT,
        ):
            choose_yes = self._should_activate()
        elif self.row.context == _CONTEXT_COIN_HEAD:
            choose_yes = True
        if choose_yes and yes is not None:
            return (yes,)
        if no is not None:
            return (no,)
        return (0,)

    def _choose_count(self) -> tuple[int, ...]:
        scores = [
            (self.score_option(self.row.option(index)), index)
            for index in range(self.row.option_count)
        ]
        scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return (scores[0][1],)

    def _score_context_option(
        self,
        option: NativeScriptedOption,
    ) -> float | None:
        context = self.row.context
        if context == _CONTEXT_SETUP_ACTIVE:
            return self._score_setup_active(option)
        if context == _CONTEXT_SETUP_BENCH:
            return self._score_setup_bench(option)
        if context in (
            _CONTEXT_TO_BENCH,
            _CONTEXT_TO_FIELD,
            _CONTEXT_NOT_MOVE,
        ):
            return self._score_to_bench_or_field(option)
        if context in (_CONTEXT_TO_HAND, _CONTEXT_LOOK):
            return self._score_to_hand(option)
        if context in (_CONTEXT_DISCARD, _CONTEXT_TO_DECK_BOTTOM):
            return -self._card_discard_cost(
                self.row.card_id_from_option(option)
            )
        if context in (_CONTEXT_TO_DECK, _CONTEXT_TO_PRIZE):
            return -self._card_keep_value(
                self.row.card_id_from_option(option)
            )
        if context in (_CONTEXT_ATTACH_TO, _CONTEXT_ATTACH_FROM):
            return self._score_attach_target(option)
        if context in (
            _CONTEXT_EVOLVES_FROM,
            _CONTEXT_EVOLVES_TO,
            _CONTEXT_EVOLVE,
        ):
            return self._score_evolve(option)
        if context in (_CONTEXT_SWITCH, _CONTEXT_TO_ACTIVE):
            return self._score_switch_target(option)
        if context in (
            _CONTEXT_DAMAGE,
            _CONTEXT_DAMAGE_COUNTER,
            _CONTEXT_DAMAGE_COUNTER_ANY,
            _CONTEXT_EFFECT_TARGET,
        ):
            return self._score_damage_target(option)
        if context in (
            _CONTEXT_HEAL,
            _CONTEXT_REMOVE_DAMAGE_COUNTER,
            _CONTEXT_REMOVE_DAMAGE_COUNTER_COUNT,
        ):
            return self._score_heal_target(option)
        if context in (
            _CONTEXT_DISCARD_ENERGY,
            _CONTEXT_DISCARD_ENERGY_CARD,
            _CONTEXT_TO_DECK_ENERGY,
        ):
            return self._score_discard_energy(option)
        if context == _CONTEXT_ATTACK:
            return self._score_attack_option(option)
        return None

    def _score_play(self, option: NativeScriptedOption) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        if card.card_type == _CARD_TYPE_POKEMON:
            if card.basic:
                return self._score_basic_play(card)
            return -2.0
        if card.card_type in (
            _CARD_TYPE_ITEM,
            _CARD_TYPE_SUPPORTER,
            _CARD_TYPE_STADIUM,
        ):
            return self._score_trainer_play(card)
        if card.card_type == _CARD_TYPE_TOOL:
            return self._score_tool_play(card)
        return 0.0

    def _score_basic_play(self, card: NativeCardSemantics) -> float:
        if len(self.row.bench_rows(self.row.perspective)) >= self.row.player_bench_max(
            self.row.perspective
        ):
            return -5.0
        board_size = len(self._your_pokemon())
        if self._has_pokemon_named(card.name):
            return 8.0 + self._pokemon_card_value(card) * 0.02
        score = 20.0 + self._pokemon_card_value(card) * 0.06
        if board_size <= 1:
            score += 32.0
        elif board_size == 2:
            score += 14.0
        if self._has_evolution_target(card.name):
            score += 12.0
        if card.ex or card.mega_ex:
            score += 4.0
        return score

    def _score_trainer_play(self, card: NativeCardSemantics) -> float:
        name = card.name.lower()
        hand_count = self.row.player_hand_count(self.row.perspective)
        text = card.skill_text
        score = 16.0
        if "mega signal" in name:
            score = 36.0 if self._need_mega_piece() else 8.0
        elif "cyrano" in name:
            score = 31.0 if self._need_ex_piece() else 9.0
        elif "waitress" in name:
            score = 32.0 if self._needs_energy_on_board() else 11.0
        elif "lillie" in name or "draw" in text:
            score = 28.0 if hand_count <= 5 else 14.0
            if self.row.player_prize_count(self.row.perspective) == 6:
                score += 8.0
        elif "search" in text:
            score = 24.0
            if "stage 1" in text and self._need_mega_piece():
                score += 16.0
        elif "shuffle your hand" in text:
            score = 18.0 if hand_count <= 4 else 6.0
        return score

    def _score_tool_play(self, card: NativeCardSemantics) -> float:
        score = 14.0
        name = card.name.lower()
        text = card.skill_text
        if "maximum belt" in name:
            score += 12.0
            if self._opponent_active_is_ex():
                score += 16.0
        if "hero" in name and "cape" in name:
            score += 22.0
        if "+100 hp" in text or "gets +100 hp" in text:
            score += 22.0
        if (
            "powerglass" in name
            or "attach a basic energy card from your discard pile" in text
        ):
            score += 14.0
        if "survival brace" in name:
            score += 18.0
        return score

    def _score_attach(self, option: NativeScriptedOption) -> float:
        if option.area is None or option.index is None:
            return 0.0
        card_id = self.row.card_id(
            option.area,
            option.index,
            option.player_index,
        )
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        target = self.row.pokemon_from_option(option)
        if card.card_type in (
            _CARD_TYPE_BASIC_ENERGY,
            _CARD_TYPE_SPECIAL_ENERGY,
        ):
            return self._score_energy_attach_to(target, card.energy_type)
        if card.card_type == _CARD_TYPE_TOOL:
            return self._score_tool_attach_to(target, card)
        return 0.0

    def _score_attach_target(self, option: NativeScriptedOption) -> float:
        target = self.row.pokemon_from_option(option)
        if target is None:
            return self._score_card_choice(option)
        return self._score_energy_attach_to(target, _ENERGY_WATER)

    def _score_energy_attach_to(
        self,
        target: int | None,
        energy: int,
    ) -> float:
        if target is None:
            return 0.0
        card = self.catalog.card(self.row.pokemon_id(target))
        before = self._best_attack_value(target)
        after = self._best_attack_value(
            target,
            (*self.row.pokemon_energies(target), energy),
        )
        improvement = max(0.0, after - before)
        missing = self._minimum_attack_shortfall(target)
        score = 14.0 + improvement * 0.18 + max(0, 4 - missing) * 4.0
        if self.row.is_your_active(target):
            score += 7.0
        if card.name.lower() == "snover" or card.mega_ex:
            score += 16.0
        if card.ex or card.mega_ex:
            score += 5.0
        if missing <= 0:
            score -= 12.0
        return score

    def _score_tool_attach_to(
        self,
        target: int | None,
        tool: NativeCardSemantics,
    ) -> float:
        if target is None:
            return 0.0
        score = 12.0 + self._best_attack_value(target) * 0.08
        text = tool.skill_text
        name = tool.name.lower()
        if self.row.is_your_active(target):
            score += 6.0
        if "maximum belt" in name and self._opponent_active_is_ex():
            score += 20.0
        if "hero" in name and "cape" in name:
            score += 24.0
        if "+100 hp" in text or "gets +100 hp" in text:
            score += 24.0
        if "powerglass" in name:
            score += 12.0 if self.row.is_your_active(target) else -4.0
        if "survival brace" in name:
            score += 16.0
        return score

    def _score_evolve(self, option: NativeScriptedOption) -> float:
        evo_id = self.row.card_id_from_option(option)
        if evo_id is None and option.card_id is not None:
            evo_id = option.card_id
        if evo_id is None:
            return 0.0
        card = self.catalog.own_card(evo_id)
        score = 34.0 + self._pokemon_card_value(card) * 0.10
        if card.mega_ex:
            score += 28.0
        elif card.ex:
            score += 12.0
        return score

    def _score_ability(self, option: NativeScriptedOption) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            return 15.0
        # Stadium abilities may belong to the other player while remaining
        # publicly usable by the acting player.
        card = self.catalog.card(card_id)
        text = card.skill_text
        score = 18.0
        if "search your deck" in text:
            score += 13.0
        if "attach" in text and "energy" in text:
            score += 14.0
        if "draw" in text:
            score += 10.0
        if "damage" in text:
            score += 6.0
        return score

    def _score_attack_option(self, option: NativeScriptedOption) -> float:
        if option.attack_id is None:
            return 0.0
        attack = self.catalog.attack(option.attack_id)
        attacker = self._your_active()
        damage = self._estimated_attack_damage(attack, attacker)
        target = self._opponent_active()
        if target is not None:
            damage = self._apply_weakness_resistance(
                damage,
                attacker,
                target,
            )
        score = 12.0 + damage * 0.22
        if target is not None and damage >= self.row.pokemon_hp(target):
            score += 55.0
            target_card = self.catalog.card(self.row.pokemon_id(target))
            if target_card.ex or target_card.mega_ex:
                score += 20.0
        text = attack.text.lower()
        if "takes 30 less damage" in text:
            score += 10.0
        if "discard 2 energy" in text:
            score -= 12.0
        if "discard the top 6 cards" in text:
            deck_count = self.row.player_deck_count(self.row.perspective)
            if deck_count <= 12:
                score -= 95.0
            elif deck_count <= 18:
                score -= 28.0
            elif damage >= 180:
                score += 12.0
        return score

    def _score_retreat(self) -> float:
        active = self._your_active()
        bench = self.row.bench_rows(self.row.perspective)
        if active is None or not bench:
            return -10.0
        active_score = self._pokemon_board_value(active)
        best_bench = max(self._pokemon_board_value(pokemon) for pokemon in bench)
        damage_ratio = 1.0 - (
            self.row.pokemon_hp(active)
            / max(1, self.row.pokemon_max_hp(active))
        )
        return best_bench - active_score + damage_ratio * 25.0 - 10.0

    def _score_setup_active(self, option: NativeScriptedOption) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        score = self._pokemon_card_value(card) * 0.08 + card.hp * 0.05
        if card.basic and self._has_evolution_target(card.name):
            score -= 8.0
        if card.ex:
            score += 6.0
        if card.name.lower() == "kyogre":
            score += 20.0
        return score

    def _score_setup_bench(self, option: NativeScriptedOption) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        score = self._pokemon_card_value(card) * 0.08
        if card.basic and self._has_evolution_target(card.name):
            score += 24.0
        if card.basic:
            score += 8.0
        return score

    def _score_to_bench_or_field(
        self,
        option: NativeScriptedOption,
    ) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        if card.card_type == _CARD_TYPE_POKEMON:
            if card.basic:
                return self._score_basic_play(card)
            return self._score_evolve(option)
        return self._card_keep_value(card_id)

    def _score_to_hand(self, option: NativeScriptedOption) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            return 0.0
        card = self.catalog.own_card(card_id)
        score = self._card_keep_value(card_id)
        name = card.name.lower()
        if card.card_type == _CARD_TYPE_POKEMON:
            if card.mega_ex and self._has_evolution_source(card):
                score += 55.0
            elif card.basic and self._has_evolution_target(card.name):
                score += 25.0
            if card.basic and len(self._your_pokemon()) <= 1:
                score += 35.0
        elif card.card_type in (
            _CARD_TYPE_BASIC_ENERGY,
            _CARD_TYPE_SPECIAL_ENERGY,
        ):
            if self._energy_in_hand_count() == 0:
                score += 22.0
            if self._needs_energy_on_board():
                score += 14.0
        elif "mega signal" in name and self._need_mega_piece():
            score += 30.0
        elif "waitress" in name and self._needs_energy_on_board():
            score += 24.0
        elif (
            "lillie" in name
            and self.row.player_hand_count(self.row.perspective) <= 4
        ):
            score += 20.0
        return score

    def _score_card_choice(self, option: NativeScriptedOption) -> float:
        card_id = self.row.card_id_from_option(option)
        if card_id is None:
            pokemon = self.row.pokemon_from_option(option)
            return self._pokemon_board_value(pokemon) if pokemon is not None else 0.0
        return self._card_keep_value(card_id)

    @staticmethod
    def _score_energy_choice(option: NativeScriptedOption) -> float:
        return float(option.count) if option.count is not None else 1.0

    def _score_skill_choice(self, option: NativeScriptedOption) -> float:
        if not option.card_id:
            return 0.0
        card = self.catalog.card(option.card_id)
        return (
            self._score_ability(option)
            + self._card_keep_value(card.card_id) * 0.02
        )

    def _score_switch_target(self, option: NativeScriptedOption) -> float:
        pokemon = self.row.pokemon_from_option(option)
        if pokemon is None:
            return self._score_card_choice(option)
        return self._pokemon_board_value(pokemon)

    def _score_damage_target(self, option: NativeScriptedOption) -> float:
        pokemon = self.row.pokemon_from_option(option)
        if pokemon is None:
            return self._score_card_choice(option)
        value = (
            (self.row.pokemon_max_hp(pokemon) - self.row.pokemon_hp(pokemon))
            * 0.10
            + max(1, 220 - self.row.pokemon_hp(pokemon)) * 0.08
        )
        owner = option.player_index
        if owner is not None and owner != self.row.perspective:
            value += 30.0
            card = self.catalog.card(self.row.pokemon_id(pokemon))
            if card.ex or card.mega_ex:
                value += 12.0
        else:
            value -= 30.0
        return value

    def _score_heal_target(self, option: NativeScriptedOption) -> float:
        pokemon = self.row.pokemon_from_option(option)
        if pokemon is None:
            return 0.0
        damaged = (
            self.row.pokemon_max_hp(pokemon) - self.row.pokemon_hp(pokemon)
        )
        score = damaged * 0.20
        owner = option.player_index
        if owner is not None and owner != self.row.perspective:
            score = -score
        return score

    def _score_discard_energy(self, option: NativeScriptedOption) -> float:
        pokemon = self.row.pokemon_from_option(option)
        if pokemon is None:
            return -1.0
        value = -self._pokemon_board_value(pokemon) * 0.02
        if not self.row.is_your_active(pokemon):
            value += 4.0
        return value

    def _should_mulligan(self) -> bool:
        basics = [
            self.catalog.own_card(card_id)
            for card_id in self.row.hand_card_ids(self.row.perspective)
            if self.catalog.own_card(card_id).basic
        ]
        if not basics:
            return True
        best_basic = max(self._pokemon_card_value(card) for card in basics)
        return best_basic < 90 and len(basics) == 1

    def _should_activate(self) -> bool:
        card_id = self.row.context_card_id()
        if card_id is None:
            return True
        text = self.catalog.own_card(card_id).skill_text
        return not (
            "you may" in text
            and "discard" in text
            and "draw" not in text
            and "search" not in text
        )


__all__ = ["NativeHeuristicChooser"]
