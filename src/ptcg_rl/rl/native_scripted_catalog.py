"""Static metadata for the native-column ``mixed75_71eb_v1`` pilot.

The hot path never imports ``cg.api``.  Generic Pokémon fields come from the
same immutable static feature table already bound to the simple-stateless
model.  The small amount of text-sensitive behavior used by the frozen 71eb
script is represented explicitly below.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from ptcg_rl.cards.static_features import (
    FEATURE_OFFSETS,
    HP_NORMALIZER,
    MAX_PRINTED_DAMAGE,
    validate_static_feature_table,
)
from ptcg_rl.rl.native_scripted_predicates import (
    MAXIMUM_BELT_CARD_IDS,
    SEARCH_DECK_ATTACK_SLOT0,
    SKILL_ATTACH_ENERGY_IDS,
    SKILL_DAMAGE_IDS,
    SKILL_DRAW_IDS,
    SKILL_SEARCH_DECK_IDS,
    TAKES_30_LESS_ATTACK_SLOT0,
    TAKES_30_LESS_ATTACK_SLOT1,
)

_CARD_TYPE_POKEMON = 0
_CARD_TYPE_BASIC_ENERGY = 5
_CARD_TYPE_SPECIAL_ENERGY = 6
_UNKNOWN_NAME = ""

MIXED75_71EB_DECK_DIGEST: Final = (
    "7fad986267efc75584f01d04045f331a3b622266ac9d55389249ebb255662607"
)
MIXED75_71EB_CARD_COUNTS: Final = (
    (1, 3),
    (3, 1),
    (6, 1),
    (7, 3),
    (16, 4),
    (18, 2),
    (112, 4),
    (117, 2),
    (344, 4),
    (345, 4),
    (414, 2),
    (1086, 4),
    (1120, 1),
    (1147, 4),
    (1152, 4),
    (1159, 1),
    (1198, 4),
    (1210, 4),
    (1227, 4),
    (1236, 4),
)
MIXED75_71EB_CARD_IDS: Final = frozenset(
    card_id for card_id, _count in MIXED75_71EB_CARD_COUNTS
)


@dataclass(frozen=True)
class NativeAttackSemantics:
    """Attack fields consumed by the frozen heuristic scorer."""

    attack_id: int
    damage: int
    energies: tuple[int, ...]
    text: str = ""


@dataclass(frozen=True)
class NativeCardSemantics:
    """Card fields consumed by the frozen heuristic scorer."""

    card_id: int
    name: str
    card_type: int
    retreat_cost: int
    hp: int
    weakness: int | None
    resistance: int | None
    energy_type: int
    basic: bool
    stage1: bool
    stage2: bool
    ex: bool
    mega_ex: bool
    evolves_from: str | None
    skill_text: str
    attacks: tuple[NativeAttackSemantics, ...]


_ATTACKS: Final = {
    141: NativeAttackSemantics(
        attack_id=141,
        damage=60,
        energies=(5, 0),
        text="your opponent's active pokemon is now confused.",
    ),
    148: NativeAttackSemantics(
        attack_id=148,
        damage=140,
        energies=(6, 0, 0),
        text=(
            "this attack's damage isn't affected by weakness or resistance, "
            "or by any effects on your opponent's active pokemon."
        ),
    ),
    478: NativeAttackSemantics(
        attack_id=478,
        damage=0,
        energies=(0,),
        text=(
            "search your deck for a card that evolves from this pokemon and "
            "put it onto this pokemon to evolve it. then, shuffle your deck."
        ),
    ),
    479: NativeAttackSemantics(
        attack_id=479,
        damage=120,
        energies=(1, 0, 0),
        text=(
            "this attack's damage isn't affected by any effects on your "
            "opponent's active pokemon."
        ),
    ),
    583: NativeAttackSemantics(
        attack_id=583,
        damage=60,
        energies=(3, 0, 0),
        text=(
            "if this pokemon has any team rocket's energy attached, this "
            "attack does 60 more damage."
        ),
    ),
}


def _card(
    card_id: int,
    name: str,
    card_type: int,
    *,
    hp: int = 0,
    energy_type: int = 0,
    retreat_cost: int = 0,
    weakness: int | None = None,
    resistance: int | None = None,
    basic: bool = False,
    stage1: bool = False,
    ex: bool = False,
    evolves_from: str | None = None,
    skill_text: str = "",
    attacks: tuple[int, ...] = (),
) -> NativeCardSemantics:
    return NativeCardSemantics(
        card_id=card_id,
        name=name,
        card_type=card_type,
        retreat_cost=retreat_cost,
        hp=hp,
        weakness=weakness,
        resistance=resistance,
        energy_type=energy_type,
        basic=basic,
        stage1=stage1,
        stage2=False,
        ex=ex,
        mega_ex=False,
        evolves_from=evolves_from,
        skill_text=skill_text.lower(),
        attacks=tuple(_ATTACKS[attack_id] for attack_id in attacks),
    )


_FIXED_CARDS: Final = {
    1: _card(1, "Basic {G} Energy", _CARD_TYPE_BASIC_ENERGY, energy_type=1),
    3: _card(3, "Basic {W} Energy", _CARD_TYPE_BASIC_ENERGY, energy_type=3),
    6: _card(6, "Basic {F} Energy", _CARD_TYPE_BASIC_ENERGY, energy_type=6),
    7: _card(7, "Basic {D} Energy", _CARD_TYPE_BASIC_ENERGY, energy_type=7),
    16: _card(
        16,
        "Prism Energy",
        _CARD_TYPE_SPECIAL_ENERGY,
        skill_text=(
            "as long as this card is attached to a pokemon, it provides "
            "{c} energy. if this card is attached to a basic pokemon, this "
            "card provides every type of energy but provides only 1 energy "
            "at a time."
        ),
    ),
    18: _card(
        18,
        "Grow Grass Energy",
        _CARD_TYPE_SPECIAL_ENERGY,
        energy_type=1,
        skill_text=(
            "as long as this card is attached to a pokemon, it provides {g} "
            "energy. the {g} pokemon this card is attached to gets +20 hp."
        ),
    ),
    112: _card(
        112,
        "Munkidori",
        _CARD_TYPE_POKEMON,
        hp=110,
        energy_type=5,
        retreat_cost=1,
        weakness=7,
        resistance=6,
        basic=True,
        skill_text=(
            "once during your turn, if this pokemon has any {d} energy "
            "attached, you may move up to 3 damage counters from 1 of your "
            "pokemon to 1 of your opponent's pokemon."
        ),
        attacks=(141,),
    ),
    117: _card(
        117,
        "Cornerstone Mask Ogerpon ex",
        _CARD_TYPE_POKEMON,
        hp=210,
        energy_type=6,
        retreat_cost=1,
        weakness=1,
        basic=True,
        ex=True,
        skill_text=(
            "prevent all damage from attacks done to this pokemon by your "
            "opponent's pokemon that have an ability."
        ),
        attacks=(148,),
    ),
    344: _card(
        344,
        "Dwebble",
        _CARD_TYPE_POKEMON,
        hp=70,
        energy_type=1,
        retreat_cost=2,
        weakness=2,
        basic=True,
        attacks=(478,),
    ),
    345: _card(
        345,
        "Crustle",
        _CARD_TYPE_POKEMON,
        hp=150,
        energy_type=1,
        retreat_cost=3,
        weakness=2,
        stage1=True,
        evolves_from="Dwebble",
        skill_text=(
            "prevent all damage done to this pokemon by attacks from your "
            "opponent's pokemon {ex}."
        ),
        attacks=(479,),
    ),
    414: _card(
        414,
        "Team Rocket's Articuno",
        _CARD_TYPE_POKEMON,
        hp=120,
        energy_type=3,
        retreat_cost=1,
        weakness=4,
        resistance=6,
        basic=True,
        skill_text=(
            "prevent all effects of attacks used by your opponent's pokemon "
            "done to your basic team rocket's pokemon."
        ),
        attacks=(583,),
    ),
    1086: _card(
        1086,
        "Buddy-Buddy Poffin",
        1,
        skill_text=(
            "search your deck for up to 2 basic pokemon with 70 hp or less "
            "and put them onto your bench. then, shuffle your deck."
        ),
    ),
    1120: _card(
        1120,
        "Crushing Hammer",
        1,
        skill_text=(
            "flip a coin. if heads, discard an energy from 1 of your "
            "opponent's pokemon."
        ),
    ),
    1147: _card(
        1147,
        "Jumbo Ice Cream",
        1,
        skill_text=(
            "heal 80 damage from your active pokemon that has 3 or more "
            "energy attached."
        ),
    ),
    1152: _card(
        1152,
        "Poke Pad",
        1,
        skill_text=(
            "search your deck for a pokemon that doesn't have a rule box, "
            "reveal it, and put it into your hand. then, shuffle your deck."
        ),
    ),
    1159: _card(
        1159,
        "Hero's Cape",
        2,
        skill_text="the pokemon this card is attached to gets +100 hp.",
    ),
    1198: _card(
        1198,
        "Crispin",
        3,
        skill_text=(
            "search your deck for up to 2 basic energy cards of different "
            "types, reveal them, and put 1 of them into your hand. attach the "
            "other to 1 of your pokemon. then, shuffle your deck."
        ),
    ),
    1210: _card(
        1210,
        "Brock's Scouting",
        3,
        skill_text=(
            "search your deck for up to 2 basic pokemon or 1 evolution "
            "pokemon, reveal them, and put them into your hand. then, shuffle "
            "your deck."
        ),
    ),
    1227: _card(
        1227,
        "Lillie's Determination",
        3,
        skill_text=(
            "shuffle your hand into your deck. then, draw 6 cards. if you "
            "have exactly 6 prize cards remaining, draw 8 cards instead."
        ),
    ),
    1236: _card(
        1236,
        "Urbain",
        3,
        skill_text="draw 3 cards.",
    ),
}

class NativeScriptedCatalog:
    """Read-only exact metadata without importing the Python engine API."""

    def __init__(self, static_features: npt.NDArray[np.float32]) -> None:
        validate_static_feature_table(static_features)
        required_maximum = max(
            *MIXED75_71EB_CARD_IDS,
            *SEARCH_DECK_ATTACK_SLOT0,
            *TAKES_30_LESS_ATTACK_SLOT0,
            *TAKES_30_LESS_ATTACK_SLOT1,
            *MAXIMUM_BELT_CARD_IDS,
            *SKILL_SEARCH_DECK_IDS,
            *SKILL_ATTACH_ENERGY_IDS,
            *SKILL_DRAW_IDS,
            *SKILL_DAMAGE_IDS,
        )
        if static_features.shape[0] <= required_maximum:
            raise ValueError(
                "static card catalog cannot cover mixed75 semantic predicates"
        )
        self._features = static_features
        self._validate_fixed_cards()
        self._validate_predicate_rows()

    def card(self, card_id: int) -> NativeCardSemantics:
        """Return fixed own-card or static generic Pokémon metadata."""
        fixed = _FIXED_CARDS.get(card_id)
        if fixed is not None:
            return fixed
        return self._generic_card(card_id)

    def own_card(self, card_id: int) -> NativeCardSemantics:
        """Return one immutable 71eb card or fail closed."""
        try:
            return _FIXED_CARDS[card_id]
        except KeyError as exc:
            raise ValueError(
                f"mixed75_71eb encountered non-artifact own card: {card_id}"
            ) from exc

    def attack(self, attack_id: int) -> NativeAttackSemantics:
        """Return one immutable 71eb attack or fail closed."""
        try:
            return _ATTACKS[attack_id]
        except KeyError as exc:
            raise ValueError(
                f"mixed75_71eb encountered unknown own attack: {attack_id}"
            ) from exc

    @staticmethod
    def is_maximum_belt(card_id: int) -> bool:
        """Return the catalog-exact Maximum Belt name predicate."""
        return card_id in MAXIMUM_BELT_CARD_IDS

    def _generic_card(self, card_id: int) -> NativeCardSemantics:
        if card_id <= 0 or card_id >= self._features.shape[0]:
            raise ValueError(f"card ID is outside the static catalog: {card_id}")
        row = self._features[card_id]
        card_type = _one_hot_index(row, "card_type")
        attacks: list[NativeAttackSemantics] = []
        if card_type == _CARD_TYPE_POKEMON:
            for slot in range(2):
                prefix = f"attack_{slot + 1}"
                exists = _scalar(row, f"{prefix}_exists")
                if exists == 0.0:
                    continue
                cost_start, cost_stop = FEATURE_OFFSETS[f"{prefix}_cost"]
                costs = tuple(
                    energy
                    for energy, count in enumerate(row[cost_start:cost_stop])
                    for _ in range(int(round(float(count))))
                )
                damage = int(
                    round(
                        _scalar(row, f"{prefix}_printed_damage_norm")
                        * MAX_PRINTED_DAMAGE
                    )
                )
                text_parts: list[str] = []
                if slot == 0 and card_id in SEARCH_DECK_ATTACK_SLOT0:
                    text_parts.append("search your deck")
                if (
                    slot == 0
                    and card_id in TAKES_30_LESS_ATTACK_SLOT0
                    or slot == 1
                    and card_id in TAKES_30_LESS_ATTACK_SLOT1
                ):
                    text_parts.append("takes 30 less damage")
                attacks.append(
                    NativeAttackSemantics(
                        attack_id=-(slot + 1),
                        damage=damage,
                        energies=costs,
                        text=" ".join(text_parts),
                    )
                )
        skill_parts: list[str] = []
        if card_id in SKILL_SEARCH_DECK_IDS:
            skill_parts.append("search your deck")
        if card_id in SKILL_ATTACH_ENERGY_IDS:
            skill_parts.append("attach energy")
        if card_id in SKILL_DRAW_IDS:
            skill_parts.append("draw")
        if card_id in SKILL_DAMAGE_IDS:
            skill_parts.append("damage")
        energy_type = 0
        if card_type == _CARD_TYPE_POKEMON:
            energy_type = _one_hot_index(row, "pokemon_type")
        elif card_type in (
            _CARD_TYPE_BASIC_ENERGY,
            _CARD_TYPE_SPECIAL_ENERGY,
        ):
            energy_type = _one_hot_index(row, "provides")
        return NativeCardSemantics(
            card_id=card_id,
            name=_UNKNOWN_NAME,
            card_type=card_type,
            retreat_cost=(
                _one_hot_index(row, "retreat_cost")
                if card_type == _CARD_TYPE_POKEMON
                else 0
            ),
            hp=int(round(_scalar(row, "hp_norm") * HP_NORMALIZER)),
            weakness=(
                _one_hot_index(row, "weakness")
                if _scalar(row, "has_weakness") == 1.0
                else None
            ),
            resistance=(
                _one_hot_index(row, "resistance")
                if _scalar(row, "has_resistance") == 1.0
                else None
            ),
            energy_type=energy_type,
            basic=(
                card_type == _CARD_TYPE_POKEMON
                and _one_hot_index(row, "stage") == 0
            ),
            stage1=(
                card_type == _CARD_TYPE_POKEMON
                and _one_hot_index(row, "stage") == 1
            ),
            stage2=(
                card_type == _CARD_TYPE_POKEMON
                and _one_hot_index(row, "stage") == 2
            ),
            ex=_scalar(row, "ex_mega_ex_tera", offset=0) == 1.0,
            mega_ex=_scalar(row, "ex_mega_ex_tera", offset=1) == 1.0,
            evolves_from=None,
            skill_text=" ".join(skill_parts),
            attacks=tuple(attacks),
        )

    def _validate_fixed_cards(self) -> None:
        for card_id, expected in _FIXED_CARDS.items():
            row = self._features[card_id]
            if not np.any(row):
                raise ValueError(
                    f"static card catalog is missing artifact card {card_id}"
                )
            checks: list[tuple[int | bool, int | bool, str]] = [
                (_one_hot_index(row, "card_type"), expected.card_type, "type"),
                (
                    int(round(_scalar(row, "hp_norm") * HP_NORMALIZER)),
                    expected.hp,
                    "hp",
                ),
            ]
            if expected.card_type == _CARD_TYPE_POKEMON:
                stage = (expected.basic, expected.stage1, expected.stage2)
                checks.extend(
                    (
                        (
                            _one_hot_index(row, "pokemon_type"),
                            expected.energy_type,
                            "energy type",
                        ),
                        (
                            _one_hot_index(row, "retreat_cost"),
                            expected.retreat_cost,
                            "retreat",
                        ),
                        (
                            _one_hot_index(row, "stage"),
                            stage.index(True),
                            "stage",
                        ),
                        (
                            _scalar(row, "has_weakness") == 1.0,
                            expected.weakness is not None,
                            "weakness presence",
                        ),
                        (
                            _scalar(row, "has_resistance") == 1.0,
                            expected.resistance is not None,
                            "resistance presence",
                        ),
                        (
                            _scalar(
                                row,
                                "ex_mega_ex_tera",
                                offset=0,
                            )
                            == 1.0,
                            expected.ex,
                            "ex",
                        ),
                        (
                            _scalar(
                                row,
                                "ex_mega_ex_tera",
                                offset=1,
                            )
                            == 1.0,
                            expected.mega_ex,
                            "mega ex",
                        ),
                    )
                )
                if expected.weakness is not None:
                    checks.append(
                        (
                            _one_hot_index(row, "weakness"),
                            expected.weakness,
                            "weakness",
                        )
                    )
                if expected.resistance is not None:
                    checks.append(
                        (
                            _one_hot_index(row, "resistance"),
                            expected.resistance,
                            "resistance",
                        )
                    )
                for slot, attack in enumerate(expected.attacks):
                    prefix = f"attack_{slot + 1}"
                    checks.extend(
                        (
                            (
                                _scalar(row, f"{prefix}_exists") == 1.0,
                                True,
                                f"{prefix} presence",
                            ),
                            (
                                int(
                                    round(
                                        _scalar(
                                            row,
                                            f"{prefix}_printed_damage_norm",
                                        )
                                        * MAX_PRINTED_DAMAGE
                                    )
                                ),
                                attack.damage,
                                f"{prefix} damage",
                            ),
                        )
                    )
                    start, stop = FEATURE_OFFSETS[f"{prefix}_cost"]
                    actual_cost = tuple(
                        energy
                        for energy, count in enumerate(row[start:stop])
                        for _ in range(int(round(float(count))))
                    )
                    if Counter(actual_cost) != Counter(attack.energies):
                        raise ValueError(
                            f"static catalog artifact card {card_id} "
                            f"{prefix} cost changed"
                        )
            elif expected.card_type in (
                _CARD_TYPE_BASIC_ENERGY,
                _CARD_TYPE_SPECIAL_ENERGY,
            ):
                checks.append(
                    (
                        _one_hot_index(row, "provides"),
                        expected.energy_type,
                        "energy type",
                    )
                )
            for actual, wanted, field in checks:
                if actual != wanted:
                    raise ValueError(
                        f"static catalog artifact card {card_id} {field} "
                        f"changed: {actual} != {wanted}"
                    )

    def _validate_predicate_rows(self) -> None:
        predicate_ids = (
            SEARCH_DECK_ATTACK_SLOT0
            | TAKES_30_LESS_ATTACK_SLOT0
            | TAKES_30_LESS_ATTACK_SLOT1
            | MAXIMUM_BELT_CARD_IDS
            | SKILL_SEARCH_DECK_IDS
            | SKILL_ATTACH_ENERGY_IDS
            | SKILL_DRAW_IDS
            | SKILL_DAMAGE_IDS
        )
        for card_id in predicate_ids:
            row = self._features[card_id]
            if not np.any(row):
                raise ValueError(
                    "static catalog is missing a scripted text-predicate "
                    f"card: {card_id}"
                )
            _one_hot_index(row, "card_type")


def _scalar(
    row: npt.NDArray[np.float32],
    field: str,
    *,
    offset: int = 0,
) -> float:
    start, stop = FEATURE_OFFSETS[field]
    index = start + offset
    if index >= stop:
        raise ValueError(f"static feature offset is invalid: {field}[{offset}]")
    value = float(row[index])
    if not np.isfinite(value):
        raise ValueError(f"static feature is non-finite: {field}")
    return value


def _one_hot_index(
    row: npt.NDArray[np.float32],
    field: str,
) -> int:
    start, stop = FEATURE_OFFSETS[field]
    values = row[start:stop]
    selected = np.flatnonzero(values == 1.0)
    if selected.size != 1 or np.count_nonzero(values) != 1:
        raise ValueError(f"static feature is not exact one-hot: {field}")
    return int(selected[0])


__all__ = [
    "MIXED75_71EB_CARD_COUNTS",
    "MIXED75_71EB_CARD_IDS",
    "MIXED75_71EB_DECK_DIGEST",
    "NativeAttackSemantics",
    "NativeCardSemantics",
    "NativeScriptedCatalog",
]
