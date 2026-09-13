"""Immutable card identities for the Majkel Lopunny scripted opponent."""

from __future__ import annotations

from typing import Final

# Special Energy.
MIST_ENERGY: Final = 11
ENRICHING_ENERGY: Final = 13
SPIKY_ENERGY: Final = 14

# Pokemon.
DUDUNSPARCE: Final = 66
FAN_ROTOM: Final = 174
DUNSPARCE: Final = 305
BUNEARY: Final = 848
MEGA_LOPUNNY_EX: Final = 849

# Trainers.
BUDDY_BUDDY_POFFIN: Final = 1086
ULTRA_BALL: Final = 1121
POKEGEAR_30: Final = 1122
POKE_PAD: Final = 1152
AIR_BALLOON: Final = 1174
BOSSES_ORDERS: Final = 1182
XEROSICS_MACHINATIONS: Final = 1197
HILDA: Final = 1225
LILLIES_DETERMINATION: Final = 1227
WALLYS_COMPASSION: Final = 1229

# Attacks. These are engine attack identities, not card identities.
FAN_ASSAULT_LANDING: Final = 230
DUDUNSPARCE_LAND_CRUSH: Final = 76
DUNSPARCE_TRADING_PLACES: Final = 423
DUNSPARCE_RAM: Final = 424
BUNEARY_RUN_AROUND: Final = 1223
BUNEARY_KICK: Final = 1224
LOPUNNY_GALE_THRUST: Final = 1225
LOPUNNY_SPIKY_HOPPER: Final = 1226

SCRIPT_NAME: Final = "lopunny_dudunsparce_v1"
OPPONENT_NAME: Final = SCRIPT_NAME

DECK_COUNTS: Final = (
    (MIST_ENERGY, 4),
    (ENRICHING_ENERGY, 1),
    (SPIKY_ENERGY, 3),
    (DUDUNSPARCE, 4),
    (FAN_ROTOM, 1),
    (DUNSPARCE, 4),
    (BUNEARY, 4),
    (MEGA_LOPUNNY_EX, 3),
    (BUDDY_BUDDY_POFFIN, 4),
    (ULTRA_BALL, 4),
    (POKEGEAR_30, 4),
    (POKE_PAD, 4),
    (AIR_BALLOON, 4),
    (BOSSES_ORDERS, 3),
    (XEROSICS_MACHINATIONS, 1),
    (HILDA, 4),
    (LILLIES_DETERMINATION, 4),
    (WALLYS_COMPASSION, 4),
)
DECK: Final = tuple(card_id for card_id, count in DECK_COUNTS for _ in range(count))
DECK_DIGEST: Final = "6fa64c0e3b2eb67c2653efeb8ae78d2079594ed23169442ff780397681f159f7"

ENERGY_CARDS: Final = frozenset({MIST_ENERGY, ENRICHING_ENERGY, SPIKY_ENERGY})
BASIC_POKEMON: Final = frozenset({FAN_ROTOM, DUNSPARCE, BUNEARY})
EVOLUTION_POKEMON: Final = frozenset({DUDUNSPARCE, MEGA_LOPUNNY_EX})
POKEMON: Final = BASIC_POKEMON | EVOLUTION_POKEMON
SUPPORTERS: Final = frozenset(
    {
        BOSSES_ORDERS,
        XEROSICS_MACHINATIONS,
        HILDA,
        LILLIES_DETERMINATION,
        WALLYS_COMPASSION,
    }
)

# Every project-owned source that can change this fixed-code policy's behavior.
IMPLEMENTATION_FILES: Final = (
    "src/ptcg_rl/actions/selection.py",
    "src/ptcg_rl/engine/constants.py",
    "src/ptcg_rl/engine/protocols.py",
    "src/ptcg_rl/engine/runtime.py",
    "src/ptcg_rl/engine/session.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/__init__.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/agent.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/cards.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/oracle.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/selection_tactics.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/tactics.py",
    "src/ptcg_rl/opponents/lopunny_dudunsparce/view.py",
)
