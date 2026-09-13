"""Immutable card catalog for the Slowking Copy Engine opponent."""

from __future__ import annotations

from typing import Final

# Energy.
BASIC_PSYCHIC: Final = 5
BOOMERANG_ENERGY: Final = 9
TELEPATH_PSYCHIC_ENERGY: Final = 19

# Pokémon.
CONKELDURR: Final = 115
FEZANDIPITI_EX: Final = 140
KYUREM: Final = 144
SLOWPOKE: Final = 162
SLOWKING: Final = 163
SMOOCHUM: Final = 183
LATIAS_EX: Final = 184
ANNIHILAPE: Final = 224
MEGA_KANGASKHAN_EX: Final = 756
MEOWTH_EX: Final = 1071

# Trainers.
SECRET_BOX: Final = 1092
NIGHT_STRETCHER: Final = 1097
ULTRA_BALL: Final = 1121
WONDROUS_PATCH: Final = 1146
POKE_PAD: Final = 1152
COUNTER_GAIN: Final = 1168
CIPHERMANIAC: Final = 1188
LILLIES_DETERMINATION: Final = 1227
ACADEMY_AT_NIGHT: Final = 1248

# Attacks.
CONKELDURR_TANTRUM: Final = 145
GUTSY_SWING: Final = 146
TRIFROST: Final = 188
SEEK_INSPIRATION: Final = 213
SUPER_PSY_BOLT: Final = 214
DELIGHTFUL_KISS: Final = 242
ANNIHILAPE_TANTRUM: Final = 304
DESTINED_FIGHT: Final = 305

SCRIPT_NAME: Final = "slowking_copy_v1"
OPPONENT_NAME: Final = "slowking_copy_v1"

DECK_COUNTS: Final = (
    (BASIC_PSYCHIC, 4),
    (BOOMERANG_ENERGY, 2),
    (TELEPATH_PSYCHIC_ENERGY, 4),
    (CONKELDURR, 2),
    (FEZANDIPITI_EX, 1),
    (KYUREM, 2),
    (SLOWPOKE, 4),
    (SLOWKING, 4),
    (SMOOCHUM, 2),
    (LATIAS_EX, 2),
    (ANNIHILAPE, 2),
    (MEGA_KANGASKHAN_EX, 2),
    (MEOWTH_EX, 1),
    (SECRET_BOX, 1),
    (NIGHT_STRETCHER, 3),
    (ULTRA_BALL, 4),
    (WONDROUS_PATCH, 3),
    (POKE_PAD, 4),
    (COUNTER_GAIN, 1),
    (CIPHERMANIAC, 4),
    (LILLIES_DETERMINATION, 4),
    (ACADEMY_AT_NIGHT, 4),
)
DECK: Final = tuple(card_id for card_id, count in DECK_COUNTS for _ in range(count))
DECK_DIGEST: Final = "48047de8a9cc5ca2686b6bc7172b76a4056edbc4182b45a2d528ba82bcdeee42"

PAYLOAD_POKEMON: Final = frozenset({CONKELDURR, KYUREM, ANNIHILAPE})
ENERGY_CARDS: Final = frozenset(
    {BASIC_PSYCHIC, BOOMERANG_ENERGY, TELEPATH_PSYCHIC_ENERGY}
)
NON_RULE_POKEMON: Final = frozenset(
    {CONKELDURR, KYUREM, SLOWPOKE, SLOWKING, SMOOCHUM, ANNIHILAPE}
)
RULE_BOX_POKEMON: Final = frozenset(
    {FEZANDIPITI_EX, LATIAS_EX, MEGA_KANGASKHAN_EX, MEOWTH_EX}
)
POKEMON: Final = NON_RULE_POKEMON | RULE_BOX_POKEMON

IMPLEMENTATION_FILES: Final = (
    "src/ptcg_rl/actions/selection.py",
    "src/ptcg_rl/engine/constants.py",
    "src/ptcg_rl/engine/runtime.py",
    "src/ptcg_rl/opponents/slowking_copy/__init__.py",
    "src/ptcg_rl/opponents/slowking_copy/agent.py",
    "src/ptcg_rl/opponents/slowking_copy/cards.py",
    "src/ptcg_rl/opponents/slowking_copy/tactics.py",
    "src/ptcg_rl/opponents/slowking_copy/view.py",
)
